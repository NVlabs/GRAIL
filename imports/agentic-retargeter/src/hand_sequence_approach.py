"""Compile a bounded approach-only refinement of an existing hand sequence.

This leaves the source clock, contact assignments, first pose and stable holds
unchanged. Speed costs are soft animation objectives, not hardware guarantees.
The caller must independently inspect geometry before selecting the result.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np


def compile_approach_refinement(base_config, arrays, window_seconds=1.5):
    """Return a per-frame GPU recipe and its explicit original-frame scope.

    ``arrays`` is the selected pre-SOMA trajectory, used unchanged as the warm
    start. The caller binds ``warm_start_npz`` to that exact file. Every frame
    outside the approach windows is frozen, including all current stable holds.
    A sequence without editable approach frames returns a receipt with
    ``needs_solve=False`` and must be passed through rather than sent to solve.
    """
    if not isinstance(base_config, dict):
        raise ValueError('base_config must be a dictionary')
    if (isinstance(window_seconds, bool) or not isinstance(window_seconds, (int, float))
            or not math.isfinite(window_seconds) or window_seconds <= 0):
        raise ValueError('window_seconds must be finite and positive')
    time = np.asarray(arrays['timestamps'], dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Expected an increasing original frame clock')
    count = len(time)
    hands = [s for s in ('left', 'right') if s + '_finger_qpos' in arrays]
    if not hands:
        raise ValueError('No bound hand trajectories')
    if (not isinstance(base_config.get('phase_labels'), dict)
            or set(base_config['phase_labels']) != set(hands)):
        raise ValueError('Explicit phase labels must match bound hands')
    if (not isinstance(base_config.get('contact_mask'), dict)
            or set(base_config['contact_mask']) != set(hands)):
        raise ValueError('Explicit contact masks must match bound hands')
    for side in hands:
        labels = base_config['phase_labels'][side]
        active = base_config['contact_mask'][side]
        if (not isinstance(labels, list) or len(labels) != count
                or any(not isinstance(label, str) for label in labels)):
            raise ValueError('Phase labels must be explicit original-frame string lists')
        if (not isinstance(active, list) or len(active) != count
                or any(type(value) is not bool for value in active)):
            raise ValueError('Contact masks must be explicit original-frame boolean lists')
    config = deepcopy(base_config)
    receipt = dict(schema='hand_approach_refinement.v1', window_seconds=float(window_seconds),
        base_config_sha256=hashlib.sha256(json.dumps(base_config, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        original_frame_count=count, hands={}, needs_solve=False,
        scope='pre-SOMA free-hand approach only; geometry quality unreviewed',
        clock_policy='unchanged', warm_policy='exact selected sequence; no time warp',
        speed_policy='soft excess costs; fixed endpoints may require exceeding limits')
    editable, intervals = {}, {}
    shared = base_config.get('strict_hold_parameterization') or {}
    if not isinstance(shared, dict):
        raise ValueError('strict_hold_parameterization must be a dictionary')
    shared_intervals = shared.get('intervals', {})
    if (not isinstance(shared_intervals, dict) or set(shared_intervals) - set(hands)
            or any(not isinstance(value, list) for value in shared_intervals.values())):
        raise ValueError('Stable hold intervals must map bound hands to explicit lists')
    if not isinstance(shared.get('protected_frames', []), list):
        raise ValueError('Protected frames must be an explicit original-frame list')
    for side in hands:
        labels = np.asarray(arrays[side + '_phase_labels'])
        active = np.asarray(arrays[side + '_contact_active'])
        if labels.shape != (count,) or active.shape != (count,) or active.dtype.kind != 'b':
            raise ValueError('Invalid original phase/contact arrays: ' + side)
        if not np.array_equal(labels, base_config['phase_labels'][side]):
            raise ValueError('Phase labels differ from the selected configuration')
        if not np.array_equal(active, base_config['contact_mask'][side]):
            raise ValueError('Contact mask differs from the selected configuration')
        held = active & np.isin(labels, ['hold', 'grasp', 'stable_hold'])
        explicit_starts = []
        for item in shared_intervals.get(side, []):
            if not isinstance(item, dict) or not {'start', 'end_exclusive'}.issubset(item):
                raise ValueError('Stable hold intervals require start and end_exclusive')
            start, end = item['start'], item['end_exclusive']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= count:
                raise ValueError('Invalid original stable hold interval')
            held[start:end] = True
            explicit_starts.append(start)
        protected = held.copy()
        protected[0] = True
        for frame in shared.get('protected_frames', []):
            if type(frame) is not int or not 0 <= frame < count:
                raise ValueError('Invalid original protected frame')
            protected[frame] = True
        onsets = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        hold_starts = np.flatnonzero(held & ~np.r_[False, held[:-1]])
        events = sorted(set(onsets.tolist() + hold_starts.tolist() + explicit_starts))
        edit = np.zeros(count, dtype=bool)
        windows = []
        for end in events:
            if end == 0:
                continue
            start = int(np.searchsorted(time, time[end] - window_seconds, side='left'))
            # Never reach back through a frozen hold or an explicitly protected
            # frame to change an unrelated preceding movement.
            stops = np.flatnonzero(protected[start:end])
            if len(stops):
                start += int(stops[-1]) + 1
            edit[start:end] = True
            windows.append(dict(event_frame=end, start=start, end_exclusive=end,
                contact_onset=bool(end in onsets), stable_hold_start=bool(end in hold_starts)))
        edit[protected] = False
        editable[side] = np.flatnonzero(edit).tolist()
        intervals[side] = np.flatnonzero(edit[:-1] | edit[1:]).tolist()
        receipt['hands'][side] = dict(events=windows, editable_frames=editable[side],
            frozen_frames=np.flatnonzero(~edit).tolist(),
            stable_hold_frames=np.flatnonzero(held).tolist(),
            speed_interval_starts=intervals[side])
    receipt['needs_solve'] = any(editable.values())
    removed = ['strict_hold_parameterization', 'trajectory_parameterization',
        'feasible_shared_hold_step', 'feasible_spline_step',
        'finger_speed_initialization', 'enforce_finger_speed_feasibility']
    receipt['removed_parameterization_keys'] = [k for k in removed if k in config]
    for key in removed:
        config.pop(key, None)
    config.update(refinement_editable_frames=editable, paired_steps=900,
        execution_backend='cuda_graph', rotation_backend='rodrigues', device='cuda:0',
        motion_speed=dict(coordinate_space='object', linear_limit_m_s=.5,
            angular_limit_rad_s=3., linear_sigma_m_s=.2, angular_sigma_rad_s=1.,
            weight=.1, intervals=intervals),
        finger_speed=dict(weight=1., sigma_rad_s=1., limit_rad_s=3.))
    config['workflow_policy'] = dict(config.get('workflow_policy') or {},
        stage='approach_refinement', approach_recipe_id='frozen_hold_speed_900_v002',
        approach_base_config_sha256=receipt['base_config_sha256'])
    return config, receipt
