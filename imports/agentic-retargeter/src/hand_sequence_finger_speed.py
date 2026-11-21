"""Optional physical excess-speed cost on original scalar finger coordinates.

This soft animation policy does not change native limits, contact phases or
the source clock. Motion below the authored gate has exactly zero gradient.
"""
from __future__ import annotations
import numpy as np
from .grasp_io import number


def finger_speed_policy(option):
    if option is None:
        return None
    if not isinstance(option, dict) or set(option) != {'weight', 'sigma_rad_s', 'limit_rad_s'}:
        raise ValueError('finger_speed requires weight, sigma_rad_s and limit_rad_s')
    value = {key: number(option[key], 'finger_speed.' + key,
                         0. if key == 'weight' else 1e-12) for key in option}
    return value if value['weight'] else None


def finger_speed_terms(fingers, intervals, policy):
    """Half mean squared normalized excess over every original interval/joint.

No wrapping of native hinge angles, phase mask, endpoint padding, root or
object columns. ``intervals`` comes from unchanged source timestamps.
"""
    import torch
    velocity = (fingers[1:] - fingers[:-1]) / intervals[:, None]
    excess = torch.relu(velocity.abs() - policy['limit_rad_s'])
    normalized = .5 * (excess / policy['sigma_rad_s']).square().mean()
    return policy['weight'] * normalized, normalized


def finger_speed_measure(fingers, timestamps, policy, names):
    q, time = np.asarray(fingers, dtype=float), np.asarray(timestamps, dtype=float)
    if (q.ndim != 2 or len(q) < 2 or time.shape != (len(q),)
            or not np.isfinite(q).all() or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0) or len(names) != q.shape[1]):
        raise ValueError('Finger speed diagnostics require finite original clock/joints')
    speed = np.abs(np.diff(q, axis=0) / np.diff(time)[:, None])
    excess = np.maximum(speed - policy['limit_rad_s'], 0.)
    return dict(maximum_rad_s=float(speed.max()), max_excess_rad_s=float(excess.max()),
        violating_samples=int(np.count_nonzero(excess)), interval_count=len(q)-1,
        joint_count=q.shape[1], passed=not bool(np.any(excess)),
        weighted_energy=float(.5 * policy['weight'] * np.mean((excess / policy['sigma_rad_s'])**2)),
        per_joint=[dict(joint=name, maximum_rad_s=float(speed[:, j].max()),
            violating_frame_pairs=[[int(i), int(i+1)] for i in np.flatnonzero(excess[:, j] > 0)])
            for j, name in enumerate(names)])
