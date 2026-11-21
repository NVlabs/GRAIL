"""Phase-derived quintic initial guesses for bounded hand approach refinement.

This is not an exported smoothing filter or a collision repair. The returned
poses must enter the unchanged contact-aware optimizer and be audited again.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .hand_sequence_approach import compile_approach_refinement


SUFFIXES = ('wrist_position_object', 'wrist_rotation_object', 'finger_qpos')
RECIPE_ID = 'phase_quintic_22_over_24s_v001'
BOUND_TOLERANCE = 1e-6
CONSTANT_HOLD_TOLERANCE = 1e-6


def _exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _proper(value, shape, label):
    value = np.asarray(value)
    if (value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all()
            or not np.allclose(value @ value.swapaxes(-1, -2), np.eye(3), atol=2e-5, rtol=0)
            or not np.allclose(np.linalg.det(value), 1., atol=2e-5, rtol=0)):
        raise ValueError('Invalid proper rotation array: ' + label)


def _bounds(p, r, q, source_p, source_r, limits, position_bound, rotation_bound, ids):
    ids = np.asarray(ids, dtype=int)
    if not len(ids):
        return dict(frame_count=0, within_original_tolerance=True,
                    position_box_excess_m=0., rotation_vector_box_excess_rad=0.,
                    native_finger_limit_excess_rad=0., violating_frames=[])
    dp = np.maximum(np.abs(p[ids].astype(float) - source_p[ids]) - position_bound, 0.).max(axis=1)
    rv = Rotation.from_matrix(r[ids].astype(float) @ source_r[ids].swapaxes(-1, -2)).as_rotvec()
    dr = np.maximum(np.abs(rv) - rotation_bound, 0.).max(axis=1)
    dq = np.maximum(np.maximum(limits[:, 0] - q[ids], q[ids] - limits[:, 1]), 0.).max(axis=1)
    bad = np.maximum.reduce((dp, dr, dq)) > BOUND_TOLERANCE
    return dict(frame_count=len(ids), within_original_tolerance=not bool(bad.any()),
                position_box_excess_m=float(dp.max()), rotation_vector_box_excess_rad=float(dr.max()),
                native_finger_limit_excess_rad=float(dq.max()), violating_frames=ids[bad].tolist())


def _derivatives(values, times, ids, endpoint):
    """Quadratic least squares at actual source times, never an assumed FPS."""
    coeff = np.polyfit(times[ids] - times[endpoint], values[ids], 2)
    return coeff[1], 2 * coeff[0]


def _bridge(values, times, start, end, incoming, outgoing, constant_end):
    velocity, acceleration = _derivatives(values, times, incoming, start)
    if constant_end:
        end_velocity = np.zeros(values.shape[1])
        end_acceleration = np.zeros(values.shape[1])
    else:
        end_velocity, end_acceleration = _derivatives(values, times, outgoing, end)
    duration = times[end] - times[start]
    c0 = values[start].astype(float)
    c1 = velocity * duration
    c2 = acceleration * duration**2 / 2
    rhs = np.stack((values[end] - c0 - c1 - c2,
                    end_velocity * duration - c1 - 2 * c2,
                    end_acceleration * duration**2 - 2 * c2))
    tail = np.linalg.solve(np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20.]]), rhs)
    coeff = np.concatenate((np.stack((c0, c1, c2)), tail))
    u = ((times[start:end + 1] - times[start]) / duration)[:, None]
    result = sum(u**i * coeff[i] for i in range(6))
    return result, dict(start_velocity=velocity.tolist(), start_acceleration=acceleration.tolist(),
                        end_velocity=end_velocity.tolist(), end_acceleration=end_acceleration.tolist())


def _events(scope, labels, active):
    """Use the hold entry once for an uninterrupted acquisition-to-hold phase."""
    held = np.zeros(len(labels), dtype=bool)
    held[scope['stable_hold_frames']] = True
    hold_entries = set(np.flatnonzero(held & ~np.r_[False, held[:-1]]).tolist())
    hold_entries.update(row['event_frame'] for row in scope['events'] if row['stable_hold_start'])
    onsets = [row['event_frame'] for row in scope['events'] if row['contact_onset']]
    retained = set(hold_entries)
    suppressed = []
    acquisition_roles = {'contact_acquisition', 'approach', 'hold', 'grasp', 'stable_hold'}
    for onset in onsets:
        later = [end for end in hold_entries if end >= onset and active[onset:end + 1].all()
                 and all(role in acquisition_roles for role in labels[onset:end + 1])]
        if later:
            suppressed.append(dict(contact_onset=onset, represented_by_hold_entry=min(later)))
        else:
            retained.add(onset)
    return sorted(retained), held, suppressed


def seed_approach_arrays(base_config, arrays, *, joint_limits, horizon_seconds=22 / 24):
    """Return a bound-valid warm guess and its auditable, non-accepting receipt.

    ``base_config`` is the original selected sequence configuration, including
    its phase labels, contact masks and original protected/strict-hold scope.
    ``arrays`` is the selected correction trajectory, not a raw human source.
    ``joint_limits`` maps each hand to native joint [low,high] rows in exactly
    ``inputs.hands[side].finger_joints`` order. Every non-pose array is copied
    byte-for-byte, including inherited diagnostic fields explicitly marked stale.

    A single deterministic policy is used for all hands: a 22/24-second bridge,
    five-sample incoming quadratic derivative fit, and exact original endpoint
    values. Constant stable holds get zero endpoint derivatives. Other endpoints
    use outgoing source derivatives rather than being made rigid. Insufficient
    data or infeasible proposals retain the original with a declared fallback.
    """
    if (isinstance(horizon_seconds, bool) or not isinstance(horizon_seconds, (float, int))
            or not math.isfinite(horizon_seconds) or horizon_seconds <= 0):
        raise ValueError('horizon_seconds must be finite and positive')
    config, scope = compile_approach_refinement(base_config, arrays)
    times = np.asarray(arrays['timestamps'], dtype=float)
    count = len(times)
    sides = list(scope['hands'])
    if not isinstance(joint_limits, dict) or set(joint_limits) != set(sides):
        raise ValueError('Native joint_limits must bind all and only selected hands')
    if any(not isinstance(v, np.ndarray) or v.dtype.kind == 'O' for v in arrays.values()):
        raise ValueError('Warm fields must be non-object NumPy arrays')
    position_bound = config.get('translation_limit_m', .15)
    rotation_bound = config.get('rotation_limit_rad', 1.5)
    for name, value, maximum in (('translation_limit_m', position_bound, 1.),
                                 ('rotation_limit_rad', rotation_bound, np.pi)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not 1e-6 <= value <= maximum):
            raise ValueError('Invalid original ' + name)
    out = {key: value.copy() for key, value in arrays.items()}
    receipt = dict(schema='hand_approach.quintic_warm_seed.v1', recipe_id=RECIPE_ID,
        horizon_seconds=float(horizon_seconds), incoming_sample_count=5,
        feasibility_tolerance=BOUND_TOLERANCE, maximum_backtracks=20,
        constant_hold_tolerance_m_rad=CONSTANT_HOLD_TOLERANCE,
        scope=scope, hands={}, initialization_only=True, posthoc_filter=False,
        geometry_checked=False, motion_accepted=False,
        stale_diagnostics_policy='All non-pose fields are inherited byte-identically for the warm contract. '
            'Cached contact/collision/speed diagnostics describe the input, NOT this seed. '
            'The subsequent native solve must recompute output diagnostics.',
        input_array_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()})
    changed_fields = set()
    for side in sides:
        keys = [side + '_' + suffix for suffix in SUFFIXES]
        p, r, q = [np.asarray(arrays[key]) for key in keys]
        names = base_config.get('inputs', {}).get('hands', {}).get(side, {}).get('finger_joints')
        if not isinstance(names, list) or not names or len(names) != len(set(names)):
            raise ValueError('Explicit native finger names required: ' + side)
        if list(arrays.get(side + '_finger_joint_names', [])) != names:
            raise ValueError('Native finger order mismatch: ' + side)
        source_p = np.asarray(arrays['source_' + side + '_wrist_position_object'])
        source_r = np.asarray(arrays['source_' + side + '_wrist_rotation_object'])
        for value, shape, label in ((p, (count, 3), keys[0]), (q, (count, len(names)), keys[2]),
                                     (source_p, (count, 3), 'source ' + side)):
            if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise ValueError('Invalid hand array: ' + label)
        _proper(r, (count, 3, 3), keys[1])
        _proper(source_r, (count, 3, 3), 'source ' + side)
        limits = np.asarray(joint_limits[side], dtype=float)
        if (limits.shape != (len(names), 2) or not np.isfinite(limits).all()
                or np.any(limits[:, 0] >= limits[:, 1])):
            raise ValueError('Invalid native finger limits: ' + side)
        side_scope = scope['hands'][side]
        edit = np.zeros(count, dtype=bool)
        edit[side_scope['editable_frames']] = True
        # A precompiled input may carry a narrower original mask. Do not widen
        # it silently when its removed shared-parameterization keys are absent.
        original_mask = base_config.get('refinement_editable_frames', {}).get(side)
        if original_mask is not None:
            if (not isinstance(original_mask, list) or any(type(i) is not int or not 0 <= i < count
                                                          for i in original_mask)):
                raise ValueError('Invalid original refinement edit mask')
            declared = np.zeros(count, dtype=bool); declared[original_mask] = True
            edit &= declared
        before = _bounds(p, r, q, source_p, source_r, limits, position_bound, rotation_bound, np.flatnonzero(edit))
        if not before['within_original_tolerance']:
            raise ValueError('Selected editable input violates original bounds: ' + side)
        labels = np.asarray(arrays[side + '_phase_labels'])
        active = np.asarray(arrays[side + '_contact_active'])
        events, held, suppressed = _events(side_scope, labels, active)
        records = []
        # All proposals and derivative fits use the same selected input, never
        # a preceding proposal. Later event windows cannot overwrite an earlier
        # endpoint or borrow invented outgoing motion from its seed.
        previous_end = 0
        for end in events:
            threshold = times[end] - horizon_seconds
            # Include a timestamp equal up to clock arithmetic roundoff, e.g.
            # exactly frame28 at 24fps for an event at frame50.
            tol = np.finfo(float).eps * max(1., abs(times[end])) * 8
            requested_start = int(np.searchsorted(times, threshold - tol, side='left'))
            start = max(requested_start, 4, previous_end)
            frozen_inside = np.flatnonzero(~edit[start + 1:end])
            if len(frozen_inside):
                start += int(frozen_inside[-1]) + 1
            row = dict(event_frame=int(end), start_frame=int(start), requested_start_frame=requested_start,
                end_frame=int(end), stable_hold_entry=bool(held[end]),
                shortened=bool(start > requested_start), blend_alpha=0., changed_frames=[],
                status='retained_original', fallback_reason=None)
            previous_end = end
            if start + 1 >= end or start < 4:
                row['fallback_reason'] = 'insufficient_editable_window_for_five_sample_incoming_fit'
                records.append(row); continue
            incoming = np.arange(start - 4, start + 1)
            hold_end = end
            while hold_end < count and held[hold_end]:
                hold_end += 1
            hold_variation = None
            if held[end] and hold_end - end >= 2:
                hold_variation = dict(position_m=float(np.max(np.abs(p[end:hold_end].astype(float) - p[end]))),
                    rotation_rad=float(Rotation.from_matrix(r[end:hold_end].astype(float) @ r[end].T).magnitude().max()),
                    native_fingers_rad=float(np.max(np.abs(q[end:hold_end].astype(float) - q[end]))))
            # A shared native hold can acquire sub-microunit serialization/FK
            # noise. Detect numerical constancy using an explicit common
            # tolerance, but preserve every actual held sample byte-for-byte.
            constant_end = bool(hold_variation is not None and
                                max(hold_variation.values()) <= CONSTANT_HOLD_TOLERANCE)
            outgoing = np.arange(end, min(count, end + 5))
            row.update(incoming_fit_frames=incoming.tolist(), outgoing_fit_frames=[] if constant_end else outgoing.tolist(),
                       measured_hold_variation=hold_variation,
                       endpoint_derivative_policy='zero_constant_stable_hold' if constant_end else 'quadratic_fit_original_outgoing_motion')
            if not constant_end and len(outgoing) < 3:
                row['fallback_reason'] = 'insufficient_outgoing_samples_without_constant_hold'
                records.append(row); continue
            local = Rotation.from_matrix(r.astype(float) @ r[start].T).as_rotvec()
            used = np.unique(np.r_[incoming, np.arange(start, end + 1), outgoing])
            # Principal SO(3) tangent coordinates cannot represent a continuous
            # chart through the pi cut. Report and retain rather than smoothing
            # a fictitious log-vector jump.
            if np.any(np.linalg.norm(np.diff(local[used], axis=0), axis=1) >= np.pi):
                row['fallback_reason'] = 'SO3_tangent_chart_discontinuity'
                records.append(row); continue
            pv, pd = _bridge(p.astype(float), times, start, end, incoming, outgoing, constant_end)
            rv, rd = _bridge(local, times, start, end, incoming, outgoing, constant_end)
            qv, qd = _bridge(q.astype(float), times, start, end, incoming, outgoing, constant_end)
            proposal_r = Rotation.from_rotvec(rv).as_matrix() @ r[start]
            ids = np.arange(start + 1, end)
            row['endpoint_derivatives'] = dict(position=pd, SO3_tangent=rd, native_fingers=qd)
            trials = []
            for backtrack in range(21):
                alpha = 2.**-backtrack
                trial_p, trial_r, trial_q = [value.copy() for value in (p, r, q)]
                trial_p[ids] = (p[ids] + alpha * (pv[1:-1] - p[ids])).astype(p.dtype)
                trial_q[ids] = (q[ids] + alpha * (qv[1:-1] - q[ids])).astype(q.dtype)
                if backtrack == 0:
                    trial_r[ids] = proposal_r[1:-1].astype(r.dtype)
                else:
                    correction = Rotation.from_matrix(proposal_r[1:-1] @ r[ids].swapaxes(-1, -2)).as_rotvec()
                    trial_r[ids] = (Rotation.from_rotvec(alpha * correction).as_matrix() @ r[ids]).astype(r.dtype)
                status = _bounds(trial_p, trial_r, trial_q, source_p, source_r, limits,
                                 position_bound, rotation_bound, ids)
                trials.append(dict(alpha=alpha, **status))
                if status['within_original_tolerance']:
                    for key, value in zip(keys, (trial_p, trial_r, trial_q)):
                        out[key][ids] = value[ids]
                    row.update(status='seeded' if backtrack == 0 else 'seeded_bound_valid_blend',
                               blend_alpha=alpha, changed_frames=ids.tolist(),
                               fallback_reason=None if backtrack == 0 else 'full_quintic_exceeded_original_bounds')
                    break
            else:
                row['fallback_reason'] = 'all_twenty_one_proposals_exceeded_original_bounds'
            row['bound_trials'] = trials
            records.append(row)
        actual_changed = np.flatnonzero(np.logical_or.reduce([
            np.any(out[key] != arrays[key], axis=tuple(range(1, arrays[key].ndim))) for key in keys]))
        frozen = np.flatnonzero(~edit)
        if not all(_exact(out[key][frozen], arrays[key][frozen]) for key in keys):
            raise AssertionError('Seed changed an original frozen pose')
        after = _bounds(*(out[key] for key in keys), source_p, source_r, limits,
                        position_bound, rotation_bound, np.flatnonzero(edit))
        if not after['within_original_tolerance']:
            raise AssertionError('Seed has an editable bound violation')
        receipt['hands'][side] = dict(events=records, suppressed_onsets=suppressed,
            original_editable_frames=np.flatnonzero(edit).tolist(), changed_frames=actual_changed.tolist(),
            first_frame_byte_identical=all(_exact(out[key][0], arrays[key][0]) for key in keys),
            frozen_frames_byte_identical=True, editable_bounds_before=before, editable_bounds_after=after,
            frozen_bounds=_bounds(p, r, q, source_p, source_r, limits, position_bound, rotation_bound, frozen),
            native_joint_names=names, native_joint_limits=limits.tolist())
        changed_fields.update(keys)
    if not all(_exact(value, out[key]) for key, value in arrays.items() if key not in changed_fields):
        raise AssertionError('Seed changed original metadata/source/object/clock')
    receipt['all_non_pose_fields_byte_identical'] = True
    receipt['changed_pose_fields'] = [key for key in sorted(changed_fields) if not _exact(out[key], arrays[key])]
    receipt['stale_inherited_diagnostic_fields'] = [key for key in arrays if
        key.startswith(('contact_selection_', 'motion_speed_', 'collocation_', 'paired_sampled_'))
        or '_sampled_object_' in key or '_sampled_scene_' in key or '_per_frame_contact_loss_' in key]
    return out, receipt
