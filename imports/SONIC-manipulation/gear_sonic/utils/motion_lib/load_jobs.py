"""Fork-free parallel execution of the native motion-load chunks (stdlib only).

The native loader (MotionLibBase.load_motions -> load_motion_with_skeleton) is per-motion work: joblib.load of the
clip, torch forward kinematics on the skeleton, resampling/interpolation. Its historical parallelism forked
multiprocessing workers, which is unsafe after CUDA initialization (the formal legs therefore run it with one job:
"Loading motions with 1 jobs", 482 s for 8192 motions on Generalist 36593474 rank 7). Threads need no fork: the torch
ops and the zlib/file I/O release the GIL, ``torch.set_num_threads(1)`` is already in force during the load, and every
chunk writes only its own result dictionary (ordered motions and reference semantics are those of the serial path; the
chunks are the same slices the process path used).

Thread count: ``MOTION_LIB_LOAD_THREADS=<n>`` explicit; otherwise min(8, CPUs available to this process / ranks on the
node) - the CPU budget the job already holds, never more than the number of chunks. ``MOTION_LIB_LOAD_THREADS=1``
keeps the serial path.
"""
from __future__ import annotations

import os
import queue as _queue
import threading
from collections.abc import Callable, Sequence
from typing import Any

MAX_DEFAULT_THREADS = 8


def _ranks_on_this_node() -> int:
    for name in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE", "WORLD_SIZE"):
        value = os.environ.get(name, "").strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    return 1


def _available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def loader_thread_count(num_items: int) -> int:
    """Threads for ``num_items`` load items: explicit MOTION_LIB_LOAD_THREADS, else the per-rank CPU share capped at
    MAX_DEFAULT_THREADS; never more than the items; 1 for one item or an invalid setting."""
    if num_items <= 1:
        return 1
    explicit = os.environ.get("MOTION_LIB_LOAD_THREADS", "").strip()
    if explicit:
        try:
            count = int(explicit)
        except ValueError:
            count = 1
        return max(1, min(count, num_items))
    share = _available_cpus() // max(1, _ranks_on_this_node())
    return max(1, min(MAX_DEFAULT_THREADS, share, num_items))


def new_queue() -> _queue.Queue:
    return _queue.Queue()


def run_in_threads(target: Callable[..., Any], jobs: Sequence[tuple], *, queue: _queue.Queue) -> dict:
    """Run ``target(*job, queue, pid)`` for jobs[1:] on daemon threads and jobs[0] inline as pid 0 (which returns its
    dictionary), gather one dictionary per started thread from ``queue`` and merge everything.

    Failure contract: every thread that was actually started SETTLES (its queue item gathered, the thread joined)
    before any failure leaves this function - whether the inline chunk raised, a thread could not be started, or a
    worker reported an error - so a caller's serial fallback never overlaps a still-running worker. All failures are
    reported the same way, as one RuntimeError naming the failed chunk(s); the caller's serial re-run then raises
    the native exception of a persistent failure."""
    if not jobs:
        return {}
    started: list[threading.Thread] = []
    errors: list[str] = []

    def _worker(args: tuple, pid: int) -> None:
        try:
            target(*args, queue, pid)
        except BaseException as exc:  # noqa: BLE001 - reported to the gathering thread, re-raised there
            queue.put(("__load_error__", pid, f"{type(exc).__name__}: {exc}"))

    merged: dict = {}
    try:
        for pid in range(1, len(jobs)):
            thread = threading.Thread(target=_worker, args=(jobs[pid], pid), name=f"motion-load-{pid}", daemon=True)
            thread.start()
            started.append(thread)
    except BaseException as exc:  # noqa: BLE001 - a thread could not be started: settle the started ones below
        errors.append(f"start of chunk {len(started) + 1}: {type(exc).__name__}: {exc}")
    if not errors:
        try:
            first = target(*jobs[0], None, 0)
            if isinstance(first, dict):
                merged.update(first)
        except BaseException as exc:  # noqa: BLE001 - settle the workers first, then report like a worker failure
            errors.append(f"chunk 0: {type(exc).__name__}: {exc}")
    for _ in started:  # every started worker puts exactly one item (result or error marker)
        item = queue.get()
        if isinstance(item, tuple) and len(item) == 3 and item[0] == "__load_error__":
            errors.append(f"chunk {item[1]}: {item[2]}")
            continue
        merged.update(item)
    for thread in started:
        thread.join()
    if errors:
        raise RuntimeError("motion load chunk(s) failed: " + "; ".join(errors))
    return merged
