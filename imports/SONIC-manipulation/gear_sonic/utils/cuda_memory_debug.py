"""Opt-in CUDA memory-leak diagnostics for PPO training (LOG_CUDA_MEMORY=2).

Records allocator stack traces and, when live allocation grows by more than
1 GiB between iterations, dumps an allocator snapshot and a census of live
CUDA tensors grouped by shape with their referrer types.
"""

from __future__ import annotations

import collections
import gc
import os
import pickle

import torch

_state = {"started": False, "prev_alloc": None, "dumps": 0, "baseline_dumped": False}


def enabled() -> bool:
    return os.environ.get("LOG_CUDA_MEMORY", "") == "2" and torch.cuda.is_available()


def start() -> None:
    if enabled() and not _state["started"]:
        torch.cuda.memory._record_memory_history(max_entries=100000, stacks="python")
        _state["started"] = True


def _census(path: str) -> None:
    by_shape = collections.Counter()
    referrers = collections.defaultdict(collections.Counter)
    for obj in gc.get_objects():
        try:
            if not (torch.is_tensor(obj) and obj.is_cuda):
                continue
        except Exception:  # noqa: BLE001
            continue
        key = (tuple(obj.shape), str(obj.dtype), obj.requires_grad, obj.grad_fn is not None)
        by_shape[key] += obj.numel() * obj.element_size()
        if len(referrers[key]) < 8:
            for ref in gc.get_referrers(obj):
                if isinstance(ref, dict):
                    owners = [type(o).__name__ for o in gc.get_referrers(ref)][:3]
                    keys = [k for k, v in ref.items() if v is obj][:3]
                    referrers[key][f"dict{keys}<-{owners}"] += 1
                else:
                    referrers[key][type(ref).__name__] += 1
    with open(path, "w") as handle:
        for key, size in by_shape.most_common(40):
            handle.write(f"{size / 2**30:8.3f} GiB {key} refs={dict(referrers[key])}\n")


def on_iteration(step: int, out_dir: str) -> None:
    if not enabled():
        return
    gc.collect()
    alloc = torch.cuda.memory_allocated()
    prev = _state["prev_alloc"]
    _state["prev_alloc"] = alloc
    jumped = prev is not None and alloc - prev > 2**30
    baseline = step >= 20 and not _state["baseline_dumped"]
    if not (jumped and _state["dumps"] < 2) and not baseline:
        return
    tag = "baseline" if baseline and not jumped else "jump"
    if baseline:
        _state["baseline_dumped"] = True
    if jumped:
        _state["dumps"] += 1
    prefix = os.path.join(out_dir, f"cuda_mem_{tag}_it{step:05d}")
    with open(prefix + ".pickle", "wb") as handle:
        pickle.dump(torch.cuda.memory._snapshot(), handle)
    _census(prefix + "_tensors.txt")
    print(f"[cuda-mem] dumped {prefix} alloc={alloc / 2**30:.2f} GiB", flush=True)  # noqa: T201
