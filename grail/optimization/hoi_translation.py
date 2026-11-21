"""Bounded, sequence-constant world translation for reviewed HOI trials."""

import math
from numbers import Real

import numpy as np


def validate_translation_bounds(bounds):
    """Require three finite metre bounds; zero freezes the corresponding axis."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 3:
        raise ValueError("human_translation_bounds_m must contain three world-axis bounds")
    if any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or not math.isfinite(x) or x < 0
        for x in bounds
    ) or not any(x > 0 for x in bounds):
        raise ValueError("human_translation_bounds_m must be finite, nonnegative and nonzero")
    return [float(x) for x in bounds]


def bounded_global_translation(original, latent, bounds):
    """Return original[T,3] + bounds*tanh(latent[3]); no per-frame variables."""
    import torch

    bounds = validate_translation_bounds(bounds)
    if (
        not torch.is_tensor(original)
        or not original.is_floating_point()
        or original.ndim != 2
        or original.shape[1] != 3
        or not len(original)
    ):
        raise ValueError("original translation must be a nonempty floating [T,3] tensor")
    if (
        not torch.is_tensor(latent)
        or latent.shape != (3,)
        or latent.dtype != original.dtype
        or latent.device != original.device
    ):
        raise ValueError(
            "global translation latent must have shape [3] and match original dtype/device"
        )
    if not torch.isfinite(original).all() or not torch.isfinite(latent).all():
        raise ValueError("global translation inputs must be finite")
    limit = original.new_tensor(bounds)
    if not torch.isfinite(limit).all():
        raise ValueError("translation bounds must be representable in the native dtype")
    delta = limit * torch.tanh(latent)
    # Preserve even signed zero in locked native axes; do not merely add zero.
    return torch.where(limit[None] > 0, original + delta[None], original)


def audit_bounded_translation(initial, saved, bounds):
    """Independently check serialized native translations with roundoff tolerance.

    Subtracting a constant from distinct float32 baselines cannot be bitwise
    constant. Active axes allow four native epsilons at their coordinate scale;
    zero-bound axes must retain the original dtype and exact bytes.
    """
    bounds = np.asarray(validate_translation_bounds(bounds), dtype=np.float64)
    before, after = np.asarray(initial), np.asarray(saved)
    if (
        before.ndim != 2
        or before.shape[1] != 3
        or not len(before)
        or before.shape != after.shape
        or not np.issubdtype(before.dtype, np.floating)
        or not np.issubdtype(after.dtype, np.floating)
    ):
        raise ValueError("Native translations must have matching nonempty floating [T,3] shapes")
    finite = bool(np.isfinite(before).all() and np.isfinite(after).all())
    fixed = bounds == 0
    frozen = before.dtype == after.dtype and before[:, fixed].tobytes() == after[:, fixed].tobytes()
    if finite:
        delta = after.astype(np.float64) - before.astype(np.float64)
        eps = max(np.finfo(before.dtype).eps, np.finfo(after.dtype).eps)
        tolerance = 4 * eps * np.maximum(1, np.maximum(np.abs(before).max(0), np.abs(after).max(0)))
        variation = np.ptp(delta, axis=0)
        bounded = bool((np.abs(delta) <= bounds[None] + tolerance[None]).all())
        constant = bool((variation <= 2 * tolerance).all())
        mean = delta.mean(0).tolist()
    else:
        bounded = constant = False
        mean = variation = tolerance = None
    checks = {
        "translation_finite": finite,
        "translation_bounds_satisfied": bounded,
        "translation_sequence_constant": constant,
        "translation_frozen_axes_exact": bool(frozen),
    }
    return {
        "mode": "bounded_global_constant",
        "axis_order": "world_xyz",
        "bounds_m": bounds.tolist(),
        "mean_delta_m": mean,
        "delta_range_m": variation.tolist() if finite else None,
        "roundoff_tolerance_m": tolerance.tolist() if finite else None,
        "checks": checks,
        "all_satisfied": all(checks.values()),
    }
