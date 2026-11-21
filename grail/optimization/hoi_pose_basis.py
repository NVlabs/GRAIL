"""Optional experimental uniform cubic basis for reviewed HOI pose corrections.

The curve is sampled at t = frame / stride. The four coefficients at indices
floor(t)..floor(t)+3 use the uniform cubic B-spline weights. This is an unclamped
basis: endpoint values are weighted averages, not interpolated endpoint knots.

C2 continuity applies to the continuous 6D correction curve. It does not imply
bounded wrist motion after rotation conversion, body FK, or mixing with a noisy
initial pose, and it does not establish valid contact or collision avoidance.
"""

from numbers import Integral

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def cubic_pose_basis(frame_count, stride):
    """Return a float64 [T,K] uniform cubic basis with at most four entries/row.

    K = floor((T-1)/stride)+4. The basis is nonnegative and sums to one at every
    frame. Linear coefficient values c[k] = a*k+b reproduce a*(f/stride+1)+b;
    the +1 offset follows this explicit knot-index convention.
    """
    frame_count = _positive_integer(frame_count, "frame_count")
    stride = _positive_integer(stride, "stride")
    knot_count = (frame_count - 1) // stride + 4
    frames = np.arange(frame_count, dtype=np.int64)
    segments, remainder = np.divmod(frames, stride)
    u = remainder.astype(np.float64) / stride
    weights = (
        np.stack(
            (
                (1.0 - u) ** 3,
                3.0 * u**3 - 6.0 * u**2 + 4.0,
                -3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0,
                u**3,
            ),
            axis=1,
        )
        / 6.0
    )
    basis = np.zeros((frame_count, knot_count), dtype=np.float64)
    basis[frames[:, None], segments[:, None] + np.arange(4)] = weights
    return basis


def expand_pose_knots(knots, basis):
    """Expand float torch [K,J,6] knots into [T,J,6], preserving gradients.

    The temporal matrix may be the NumPy array from cubic_pose_basis or a torch
    tensor. Multiplication takes place in the knot tensor's dtype and device.
    """
    import torch

    if not torch.is_tensor(knots) or knots.ndim != 3 or knots.shape[-1] != 6:
        raise ValueError("knots must be a torch tensor with shape [K,J,6]")
    if not knots.is_floating_point() or knots.shape[0] < 1 or knots.shape[1] < 1:
        raise ValueError("knots must have nonempty floating point knot and joint axes")
    if not torch.isfinite(knots).all():
        raise ValueError("knots must be finite")
    try:
        matrix = torch.as_tensor(basis, device=knots.device, dtype=knots.dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("basis must be a finite rectangular matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != knots.shape[0]:
        raise ValueError("basis must have shape [T,K] matching the knots")
    if not torch.isfinite(matrix).all():
        raise ValueError("basis must be finite")
    return (matrix @ knots.reshape(knots.shape[0], -1)).reshape(matrix.shape[0], knots.shape[1], 6)
