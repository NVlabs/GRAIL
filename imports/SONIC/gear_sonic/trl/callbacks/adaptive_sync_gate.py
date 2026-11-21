"""One cross-GPU adaptive-sampling synchronization per completed training step.

The PPO trainer synchronizes the keyed sampler across GPUs every ``sync_adaptive_sampling_all_gpus_freq`` steps
(default 200), the save callback does it again before every checkpoint (every ``save_last_frequency`` steps, 20 in
the formal runs) and the time-budget callback once more at the boundary. At a step where several of them coincide
the same collective ran twice or three times on the same state (measured 2026-10-14: a sync is the bulk of the
~165 s save callback at 8x8192). This gate records the last step whose cross-GPU sync completed on the environment
object and skips a repeated request for the SAME step; every rank sees the same ``step``, so the collective is
either entered by all ranks or by none. What the checkpoint contains is unchanged: the state after this step's sync.
No gate on training, no file, nothing persisted.
"""
from __future__ import annotations

_MARKER = "_adaptive_cross_gpu_sync_step"


def last_synced_step(env) -> int | None:
    """Step whose cross-GPU sync already completed on ``env`` (None before the first one)."""
    return getattr(env, _MARKER, None)


def sync_once_per_step(env, accelerator, step: int | None, *, source: str = "") -> bool:
    """Run ``env.sync_and_compute_adaptive_sampling(accelerator, sync_across_gpus=True)`` unless it already ran for
    ``step``. Returns True when the collective ran. ``step=None`` always runs (callers without a step)."""
    if env is None or not hasattr(env, "sync_and_compute_adaptive_sampling"):
        return False
    if step is not None and last_synced_step(env) == int(step):
        return False
    env.sync_and_compute_adaptive_sampling(accelerator, sync_across_gpus=True)
    if step is not None:
        try:
            setattr(env, _MARKER, int(step))
        except Exception:  # noqa: BLE001 - a wrapper that refuses attributes simply never dedups
            pass
    return True
