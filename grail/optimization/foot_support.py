"""World-space drift of independently supported foot regions."""

import math

import torch


def stance_drift_loss(points, probabilities, *, threshold=0.5, min_frames=3,
                      gravity_axis=2, robust_delta=0.005):
    """Penalize planar spread within each uninterrupted support interval.

    The shared support location is the differentiable mean of that interval.
    Thus a constant translation is free: the source/depth objectives determine
    placement, without locking in an erroneous first-frame reconstruction.
    Each region and each recontact gets its own location; no interval crosses
    an unconfident or swinging frame. Probabilities are fixed observations.
    Returns metres (quadratic near zero), averaged over supported samples.
    This is neither a floor-height nor a collision constraint.
    """
    if points.ndim != 3 or points.shape[-1] != 3 or probabilities.shape != points.shape[:2]:
        raise ValueError("Foot support requires points [T,P,3] and probabilities [T,P]")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Foot stance threshold must be in [0,1]")
    if type(min_frames) is not int or min_frames < 2:
        raise ValueError("Foot stance min_frames must be an integer >=2")
    if type(gravity_axis) is not int or gravity_axis not in (0, 1, 2):
        raise ValueError("Foot stance gravity_axis must be 0, 1 or 2")
    if not math.isfinite(robust_delta) or robust_delta <= 0:
        raise ValueError("Foot stance robust_delta must be finite and positive")
    planar = points[..., [axis for axis in range(3) if axis != gravity_axis]]
    active = (probabilities.detach() > threshold).cpu()
    total = points.sum() * 0.0
    count = 0
    for part in range(points.shape[1]):
        padded = torch.cat([torch.zeros(1, dtype=torch.bool), active[:, part],
                            torch.zeros(1, dtype=torch.bool)])
        boundaries = torch.where(padded[1:] != padded[:-1])[0].tolist()
        for start, stop in zip(boundaries[::2], boundaries[1::2]):
            if stop - start < min_frames:
                continue
            segment = planar[start:stop, part]
            distance = (segment - segment.mean(dim=0)).norm(dim=-1)
            penalty = torch.where(distance < robust_delta,
                                  0.5 * distance.square() / robust_delta,
                                  distance - 0.5 * robust_delta)
            total = total + penalty.sum()
            count += stop - start
    return total / max(count, 1)
