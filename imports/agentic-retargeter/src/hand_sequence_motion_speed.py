"""Optional wrist excess-speed cost on explicit original adjacent frame pairs.

This is a soft animation objective, not a hardware limit or a continuous-time
guarantee. It changes neither the source clock nor contact/phase assignments.
Callers supply poses in the explicitly selected object or world coordinates.
"""
from __future__ import annotations

import numpy as np

from .grasp_io import number
from .hand_sequence_rotation import so3_log


def motion_speed_policy(option, hands, frame_count):
    if option is None:
        return None
    keys = {'coordinate_space', 'linear_limit_m_s', 'angular_limit_rad_s',
            'linear_sigma_m_s', 'angular_sigma_rad_s', 'weight', 'intervals'}
    if not isinstance(option, dict) or set(option) != keys:
        raise ValueError('motion_speed requires exactly ' + ', '.join(sorted(keys)))
    if option['coordinate_space'] not in ('object', 'world'):
        raise ValueError('motion_speed.coordinate_space must be object or world')
    if type(frame_count) is not int or frame_count < 2:
        raise ValueError('motion_speed requires at least two original frames')
    selected = option['intervals']
    if (not isinstance(selected, dict) or not selected
            or set(selected) - set(hands) or set(selected) - {'left', 'right'}):
        raise ValueError('motion_speed.intervals must explicitly map bound hands to frame-pair indices')
    result = {'coordinate_space': option['coordinate_space'], 'intervals': {}}
    for key in keys - {'coordinate_space', 'intervals'}:
        result[key] = number(option[key], 'motion_speed.' + key,
                             0. if key == 'weight' else 1e-12)
    for side, indices in selected.items():
        if (not isinstance(indices, list)
                or any(type(i) is not int or not 0 <= i < frame_count - 1 for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError('motion_speed interval indices must be sorted unique original pair starts')
        result['intervals'][side] = list(indices)
    return result if result['weight'] else None


def motion_speed_terms(position, rotation, intervals, policy, interval_indices):
    """Weighted half mean of squared normalized excess linear/angular speed.

    ``intervals`` contains unchanged source timestamp differences; index i
    always means the pair (i, i+1), including phase and editable/frozen joins.
    Rotation uses the principal SO(3) increment (finite branch at pi; not
    differentiable across the pi cut locus). Empty explicit selections cost zero.
    """
    import torch
    linear = torch.linalg.vector_norm(position[1:] - position[:-1], dim=-1) / intervals
    relative = rotation[1:] @ rotation[:-1].transpose(-1, -2)
    angular = torch.linalg.vector_norm(so3_log(relative), dim=-1) / intervals
    linear_excess = torch.relu(linear - policy['linear_limit_m_s'])
    angular_excess = torch.relu(angular - policy['angular_limit_rad_s'])
    if len(interval_indices):
        linear_term = .5 * (linear_excess[interval_indices] / policy['linear_sigma_m_s']).square().mean()
        angular_term = .5 * (angular_excess[interval_indices] / policy['angular_sigma_rad_s']).square().mean()
    else:
        linear_term = linear.sum() * 0.
        angular_term = angular.sum() * 0.
    return policy['weight'] * (linear_term + angular_term), {
        'linear_normalized_excess': linear_term, 'angular_normalized_excess': angular_term}


def motion_speed_measure(position, rotation, timestamps, policy, interval_indices):
    """Independent NumPy diagnostics and exportable original-interval arrays."""
    from scipy.spatial.transform import Rotation
    p, r, time = (np.asarray(v, dtype=float) for v in (position, rotation, timestamps))
    if (p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 2
            or r.shape != (len(p), 3, 3) or time.shape != (len(p),)
            or not all(np.isfinite(v).all() for v in (p, r, time))
            or np.any(np.diff(time) <= 0)):
        raise ValueError('Wrist speed diagnostics require finite original poses and increasing clock')
    mask = np.zeros(len(p)-1, dtype=bool)
    indices = list(interval_indices)
    if (any(type(i) is not int or not 0 <= i < len(mask) for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Invalid original wrist-speed interval indices')
    mask[indices] = True
    linear = np.linalg.norm(np.diff(p, axis=0), axis=-1) / np.diff(time)
    angular = Rotation.from_matrix(r[1:] @ r[:-1].transpose(0, 2, 1)).magnitude() / np.diff(time)
    le = np.maximum(linear - policy['linear_limit_m_s'], 0.)
    ae = np.maximum(angular - policy['angular_limit_rad_s'], 0.)

    def summarize(selected):
        ids = np.flatnonzero(selected)
        count = len(ids)
        energy = (.5 * policy['weight'] * np.mean(
            (le[ids] / policy['linear_sigma_m_s'])**2
            + (ae[ids] / policy['angular_sigma_rad_s'])**2)) if count else 0.
        return dict(interval_count=count, weighted_energy=float(energy),
            maximum_linear_m_s=float(linear[ids].max()) if count else None,
            maximum_angular_rad_s=float(angular[ids].max()) if count else None,
            max_linear_excess_m_s=float(le[ids].max()) if count else None,
            max_angular_excess_rad_s=float(ae[ids].max()) if count else None,
            linear_violating_frame_pairs=[[int(i), int(i+1)] for i in ids[le[ids] > 0]],
            angular_violating_frame_pairs=[[int(i), int(i+1)] for i in ids[ae[ids] > 0]],
            within_soft_limits=not bool(np.any(le[ids]) or np.any(ae[ids])))

    return dict(coordinate_space=policy['coordinate_space'], selected=summarize(mask),
                all_sequence=summarize(np.ones_like(mask))), {
        'selected_interval_mask': mask, 'linear_m_s': linear, 'angular_rad_s': angular,
        'linear_excess_m_s': le, 'angular_excess_rad_s': ae}
