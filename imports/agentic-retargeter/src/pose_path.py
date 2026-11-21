"""Select a temporally consistent path through a reviewed key-pose bank.

This finite dynamic program selects only among supplied poses. It does not solve
IK, infer feasibility, interpolate poses or accept a motion. Failed candidates
must be masked by the caller's geometry/contact/source review.
"""
from __future__ import annotations

import numpy as np


def select_pose_path(poses, costs, times, *, valid=None, scales=None, velocity_weight=1.0):
    """Return bank indices minimizing unary cost plus scaled squared velocity.

    ``poses`` is [key, candidate, hinge]. Values remain native limited hinge
    angles: wrapping by 2pi would invent a path through physical joint limits.
    ``times`` is in seconds, and scales are explicit positive radians per hinge.
    An entirely infeasible key raises; infeasible nodes never enter the path.
    """
    poses, costs, times = np.asarray(poses, float), np.asarray(costs, float), np.asarray(times, float)
    if (poses.ndim != 3 or not all(poses.shape) or costs.shape != poses.shape[:2]
            or times.shape != (len(poses),) or not np.isfinite(poses).all()
            or not np.isfinite(costs).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Expected finite pose bank/costs and strictly increasing key times")
    valid = np.ones(costs.shape, bool) if valid is None else np.asarray(valid)
    scales = np.ones(poses.shape[-1]) if scales is None else np.asarray(scales, float)
    if valid.dtype != bool or valid.shape != costs.shape or np.any(~valid.any(axis=1)):
        raise ValueError("Every key requires at least one explicitly valid candidate")
    if scales.shape != (poses.shape[-1],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Hinge scales must be finite and positive")
    if not np.isfinite(velocity_weight) or velocity_weight < 0:
        raise ValueError("velocity_weight must be finite and nonnegative")
    total = np.where(valid[0], costs[0], np.inf)
    parents = []
    for key in range(1, len(poses)):
        dt = times[key] - times[key - 1]
        delta = (poses[key][None] - poses[key - 1][:, None]) / scales
        transition = velocity_weight * np.sum(delta ** 2, axis=-1) / dt
        scores = total[:, None] + transition
        parent = np.argmin(scores, axis=0)
        total = scores[parent, np.arange(len(parent))] + costs[key]
        total[~valid[key]] = np.inf
        parents.append(parent)
    path = [int(np.argmin(total))]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    return np.asarray(path[::-1], dtype=int)
