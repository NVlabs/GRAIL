"""Decode-only backtracking for source-bounded hand spline Adam updates.

This is a feasibility filter, not an objective line search. It preserves the
existing componentwise translation/rotation-vector boxes and native finger
limits at original frames. It does not evaluate FK, collision or subframes.
Adam's moments and step counter retain the full proposed update history.
"""
from __future__ import annotations

import math
import time


def validate_feasible_spline_step(value):
    if value is None or value is False:
        return None
    if not isinstance(value, dict) or set(value)-{'kind', 'max_backtracks', 'shrink_factor'}:
        raise ValueError('feasible_spline_step requires kind, max_backtracks, shrink_factor only')
    if value.get('kind') not in ('backtracking','affine_projection_backtracking'):
        raise ValueError('feasible_spline_step kind must be backtracking or affine_projection_backtracking')
    count = value.get('max_backtracks', 12)
    shrink = value.get('shrink_factor', .5)
    if type(count) is not int or not 1 <= count <= 24:
        raise ValueError('max_backtracks must be an integer 1..24')
    if isinstance(shrink, bool) or not isinstance(shrink, (int, float)) or not math.isfinite(shrink) or not 0 < shrink < 1:
        raise ValueError('shrink_factor must be finite and strictly between zero and one')
    return dict(kind=value['kind'], max_backtracks=count, shrink_factor=float(shrink))


def decoded_feasibility(value, hands, position_bound, rotation_bound):
    """Check the same original-frame wrist boxes plus finite native hinges."""
    import torch
    with torch.no_grad():
        if value.ndim != 3 or value.shape[1] != len(hands) or value.shape[2] != 13:
            raise ValueError('Expected frame-by-hand source-relative Dex3 parameters')
        if not bool(torch.isfinite(value).all()):
            return dict(feasible=False, finite=False, position_box_excess_m=None,
                        rotation_vector_box_excess_rad=None, native_finger_limit_excess_rad=None,
                        rotation_vector_magnitude_max_rad=None, bounds_are_componentwise_boxes=True)
        fingers = []
        for index, h in enumerate(hands.values()):
            q = h['source_finger']+value[:, index, 6:]
            if not bool(torch.isfinite(q).all()):
                return dict(feasible=False, finite=False, position_box_excess_m=None,
                            rotation_vector_box_excess_rad=None, native_finger_limit_excess_rad=None,
                            rotation_vector_magnitude_max_rad=None, bounds_are_componentwise_boxes=True)
            fingers.append(torch.maximum(h['low']-q, q-h['high']).clamp_min(0).max())
        metrics = torch.stack((torch.relu(value[:, :, :3].abs()-position_bound).max(),
                               torch.relu(value[:, :, 3:6].abs()-rotation_bound).max(),
                               torch.stack(fingers).max(),
                               torch.linalg.vector_norm(value[:, :, 3:6], dim=-1).max())).cpu().tolist()
        position, rotation, finger, magnitude = metrics
        return dict(feasible=max(position, rotation, finger) <= 1e-6, finite=True,
                    position_box_excess_m=position, rotation_vector_box_excess_rad=rotation,
                    native_finger_limit_excess_rad=finger, rotation_vector_magnitude_max_rad=magnitude,
                    bounds_are_componentwise_boxes=True)


def backtrack_proposed_step(parameters, previous, previous_status, status, options):
    """Filter an already proposed optimizer step along one shared control line.

The caller snapshots parameters before Adam and supplies their checked status.
The optimizer is intentionally not passed here: moments/counters are untouched.
Rejected proposals are decoded only; the next full objective uses the accepted
controls. A zero fraction restores the exact previous parameter bytes.
"""
    import torch
    if not previous_status.get('feasible') or not previous_status.get('finite'):
        raise ValueError('Backtracking must start from an independently checked finite feasible state')
    if len(parameters) != len(previous) or not parameters:
        raise ValueError('Need matching nonempty previous controls')
    for parameter, saved in zip(parameters, previous):
        if parameter.shape != saved.shape or not bool(torch.isfinite(saved).all()):
            raise ValueError('Invalid previous control snapshot')
    started = time.monotonic()
    trials = []
    with torch.no_grad():
        proposed = [p.detach().clone() for p in parameters]
        finite_proposal = all(bool(torch.isfinite(p).all()) for p in proposed)
        step_max = max(float((p-old).abs().max()) for p, old in zip(proposed, previous)) if finite_proposal else None
        if finite_proposal:
            proposed_status = status()
        else:
            proposed_status = dict(feasible=False, finite=False, reason='nonfinite_control_proposal')
        trials.append(dict(fraction=1., status=proposed_status))
        accepted_status = proposed_status
        fraction = 1.
        backtracks = 0
        if not proposed_status['feasible'] and finite_proposal:
            for backtracks in range(1, options['max_backtracks']+1):
                fraction = options['shrink_factor']**backtracks
                for parameter, old, proposal in zip(parameters, previous, proposed):
                    parameter.copy_(old+(proposal-old)*fraction)
                accepted_status = status()
                trials.append(dict(fraction=fraction, status=accepted_status))
                if accepted_status['feasible']:
                    break
        if not accepted_status['feasible']:
            fraction = 0.
            for parameter, old in zip(parameters, previous):
                parameter.copy_(old)
            accepted_status = status()
            if not accepted_status['feasible'] or not accepted_status.get('finite'):
                raise RuntimeError('Restored control snapshot no longer decodes to its feasible state')
        accepted_max = max(float((p-old).abs().max()) for p, old in zip(parameters, previous))
    return dict(proposed_fraction=1., accepted_fraction=fraction, backtracks=backtracks,
                rejected_trial_count=sum(not t['status']['feasible'] for t in trials),
                restored_previous=fraction == 0., zero_step=accepted_max == 0.,
                proposed_control_max_abs_step=step_max, accepted_control_max_abs_step=accepted_max,
                proposed_status=proposed_status, accepted_status=accepted_status,
                trials=trials, elapsed_seconds=time.monotonic()-started,
                adam_state_policy='Retain full-step Adam moments and step counter; only parameters are filtered.')
