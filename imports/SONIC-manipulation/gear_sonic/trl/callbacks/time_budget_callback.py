"""Stop a formal training leg safely before its scheduler allocation ends.

An optional launcher wrapper exports ``TRAIN_DEADLINE_EPOCH`` (allocation EndTime minus a measured
grace) for formal workloads and may create ``<output_dir>/.stop_request`` when the USR1
warning arrives without a computable deadline. At the end of every PPO update rank 0 checks
both; the decision is broadcast to every rank so all of them take the same collective path:
adaptive-sampling sync, barrier, one more ``last.pt`` through the existing atomic checkpoint
writer (a failed save is retried once; then, after every rank has completed the collectives,
all ranks raise so the leg fails through the actual process result and the next leg resumes
from the older durable ``last.pt`` under the failure budget - never a healthy stop) and
``control.should_training_stop = True`` (the stop reason is printed as a log line; no telemetry file). The wrapper then classifies the leg from the saved
checkpoint step (progress past the step at leg start -> continue) and requeues the same job,
which resumes from that checkpoint. Verification workloads never set the deadline.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path

from transformers import TrainerCallback

from gear_sonic.trl.callbacks.model_save_callback import ModelSaveCallback

REASON_NONE, REASON_TIME_BUDGET, REASON_STOP_REQUEST = 0, 1, 2
_REASON_NAMES = {REASON_TIME_BUDGET: "time_budget", REASON_STOP_REQUEST: "stop_request"}
# Printed on the mandatory-save failure path; tools/leg_continuation.py reads this line (plain log text,
# no sidecar) to name the outcome ``boundary_save_failed``. Keep both in sync.
SAVE_FAILED_LINE = "[time_budget] BOUNDARY CHECKPOINT SAVE FAILED"
# Per-attempt warning (never the terminal marker): a save that succeeds on the retry is a healthy boundary.
SAVE_RETRY_LINE = "[time_budget] boundary checkpoint save attempt"
_SAVE_ATTEMPTS = 2
_RETRY_DELAY_S = 5.0


def _env_float(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class TimeBudgetCallback(TrainerCallback):
    def __init__(self, deadline_epoch=None, stop_file=None, save_dir=None, enabled=True):
        self.enabled = bool(enabled)
        self.deadline_epoch = float(deadline_epoch) if deadline_epoch not in (None, "", "none", "null") else _env_float("TRAIN_DEADLINE_EPOCH")
        self._stop_file = Path(stop_file) if stop_file else None
        self._save_dir = Path(save_dir) if save_dir else None
        self.triggered_reason = None

    # -- paths default to the trainer output dir (= experiment_dir) ---------------------
    def _paths(self, args):
        base = self._save_dir or Path(getattr(args, "output_dir", "."))
        return base, (self._stop_file or base / ".stop_request")

    def on_train_begin(self, args, state, control, **kwargs):
        base, stop_file = self._paths(args)
        if self.deadline_epoch is None:
            print("[time_budget] no deadline (TRAIN_DEADLINE_EPOCH unset): leg runs to the target", flush=True)
        else:
            remaining = self.deadline_epoch - time.time()
            print(
                f"[time_budget] deadline_epoch={self.deadline_epoch:.0f} remaining_s={remaining:.0f} "
                f"stop_file={stop_file} target={getattr(state, 'max_steps', None)}",
                flush=True,
            )
        return control

    def _decide_rank0(self, stop_file: Path) -> int:
        if self.deadline_epoch is not None and time.time() >= self.deadline_epoch:
            return REASON_TIME_BUDGET
        if stop_file.exists():
            return REASON_STOP_REQUEST
        return REASON_NONE

    def _broadcast(self, reason: int, accelerator) -> int:
        if accelerator is None or getattr(accelerator, "num_processes", 1) <= 1:
            return reason
        import torch

        flag = torch.tensor([int(reason)], dtype=torch.int64, device=accelerator.device)
        torch.distributed.broadcast(flag, src=0)
        return int(flag.item())

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled or self.triggered_reason is not None:
            return control
        accelerator = kwargs.get("accelerator")
        base, stop_file = self._paths(args)
        reason = self._decide_rank0(stop_file) if state.is_world_process_zero else REASON_NONE
        reason = self._broadcast(reason, accelerator)
        if reason == REASON_NONE:
            return control
        self.triggered_reason = _REASON_NAMES[reason]
        env = kwargs.get("env")
        if accelerator is not None:
            if env is not None and hasattr(env, "sync_and_compute_adaptive_sampling"):
                from gear_sonic.trl.callbacks.adaptive_sync_gate import sync_once_per_step

                sync_once_per_step(env, accelerator, getattr(state, "global_step", None), source="time_budget")
            accelerator.wait_for_everyone()
        save_error = None
        if state.is_world_process_zero:
            from gear_sonic.trl.callbacks import completed_saves

            last_path = str(base / "last.pt")
            prior = completed_saves.completed(state.global_step)
            attempt = 0
            reused = None
            if prior.get("last") and completed_saves.same_file_name(prior["last"], last_path):
                reused = prior["last"]  # the save callback already wrote last.pt at this very step: nothing to serialize again
            elif prior:
                # a checkpoint of this step exists under another name: publish last.pt as a second name of that file
                source = prior.get("last") or prior.get("regular")
                if ModelSaveCallback.publish_same_checkpoint(source, last_path, state.global_step):
                    reused = source
            if reused is not None:
                completed_saves.record(state.global_step, "last", last_path)
                print(f"[time_budget] boundary checkpoint reused from the same-step save {reused} (not serialized again)", flush=True)
            else:
                for attempt in range(1, _SAVE_ATTEMPTS + 1):
                    try:
                        env_state = env.get_env_state_dict() if env is not None and hasattr(env, "get_env_state_dict") else None
                        ModelSaveCallback.save_checkpoint(
                            kwargs.get("model"), kwargs.get("optimizer"), kwargs.get("lr_scheduler"), state, env_state, args, str(base / "last.pt")
                        )
                        save_error = None
                        break
                    except Exception as exc:  # noqa: BLE001 - never strand the peer ranks at the barrier below
                        save_error = exc
                        if attempt < _SAVE_ATTEMPTS:
                            print(
                                f"{SAVE_RETRY_LINE} {attempt}/{_SAVE_ATTEMPTS} failed on rank 0 at step {state.global_step}: {exc!r}; "
                                f"retrying in {_RETRY_DELAY_S:g} s",
                                flush=True,
                            )
                            time.sleep(_RETRY_DELAY_S)
                        else:
                            # terminal marker: printed only when every attempt failed (read by tools/leg_continuation.py)
                            print(
                                f"{SAVE_FAILED_LINE} after {_SAVE_ATTEMPTS} attempts rank=0 step={state.global_step} error={exc!r}; "
                                "the older durable last.pt stays the resume point",
                                flush=True,
                            )
            if save_error is None and attempt >= 1:
                completed_saves.record(state.global_step, "last", last_path)
            if save_error is None and attempt > 1:
                print(f"[time_budget] boundary checkpoint saved on attempt {attempt}/{_SAVE_ATTEMPTS}", flush=True)
            print(
                f"[time_budget] segment_end reason={self.triggered_reason} global_step={state.global_step} "
                f"target={int(getattr(state, 'max_steps', 0) or 0)} saved={'FAILED' if save_error else base / 'last.pt'}",
                flush=True,
            )
        # rank 0 tells every rank whether the mandatory checkpoint exists; all ranks then finish the
        # collectives before anyone raises, so DDP never strands a peer.
        save_ok = self._broadcast(0 if save_error is not None else 1, accelerator)
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if not save_ok:
            # Fail coherently on every rank: the process result and the traceback in the leg log make
            # tools/leg_continuation.py classify this leg as boundary_save_failed (failure budget, resume
            # from the older durable last.pt) - never as a healthy continuation. Only the mandatory
            # checkpoint matters here; a telemetry write failure stays harmless.
            raise RuntimeError(
                f"{SAVE_FAILED_LINE}: boundary checkpoint at step {state.global_step} was not saved after "
                f"{_SAVE_ATTEMPTS} attempts" + (f" ({save_error!r})" if save_error is not None else " (reported by rank 0)")
            )
        control.should_training_stop = True
        return control
