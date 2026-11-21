"""Independent NumPy diagnostics for phase-aware free-hand trajectories.

No optimization, FK, mesh query, contact-role inference or motion acceptance is
performed. Archives use the hand_sequence schema; phases are [start, end).
Optional material evidence requires identically named persistent samples and
fixed targets in both archives. Its provenance must be audited by the caller.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def _rotations(value):
    """Remove float roundoff only; reject matrices that are not close to SO(3)."""
    value = np.asarray(value, dtype=float)
    if value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation matrices must be finite (...,3,3)")
    if np.max(np.abs(value @ value.swapaxes(-1, -2) - np.eye(3)), initial=0) > 1e-3:
        raise ValueError("rotation matrices are not near orthogonal")
    if np.any(np.linalg.det(value) < .999):
        raise ValueError("rotation matrices must be proper SO(3)")
    u, _, vh = np.linalg.svd(value)
    u = u.copy()
    u[..., :, -1] *= np.linalg.det(u @ vh)[..., None]
    return u @ vh


def _log_so3(value):
    rotations = _rotations(value)
    vectors = []
    for rotation in rotations.reshape(-1, 3, 3):
        v = np.array([rotation[2, 1] - rotation[1, 2],
                      rotation[0, 2] - rotation[2, 0],
                      rotation[1, 0] - rotation[0, 1]]) * .5
        sine = np.linalg.norm(v)
        angle = np.arctan2(sine, np.clip((np.trace(rotation) - 1) * .5, -1, 1))
        if angle < 1e-8:
            vectors.append(v)
        elif np.pi - angle < 1e-6:
            values, axes = np.linalg.eigh((rotation + rotation.T) * .5)
            axis = axes[:, np.argmax(values)]
            if np.dot(axis, v) < 0:
                axis = -axis
            vectors.append(axis * angle)
        else:
            vectors.append(v * angle / sine)
    return np.array(vectors).reshape(rotations.shape[:-2] + (3,))


def _stats(value):
    value = np.asarray(value, dtype=float)
    if not len(value):
        return dict(samples=0, vector_rms=None, vector_peak=None, vector_p95=None,
                    component_rms=None, component_peak=None)
    norms = np.linalg.norm(value, axis=-1)
    return dict(samples=len(value), vector_rms=float(np.sqrt(np.mean(norms ** 2))),
                vector_peak=float(norms.max()), vector_p95=float(np.quantile(norms, .95)),
                component_rms=float(np.sqrt(np.mean(value ** 2))),
                component_peak=float(np.abs(value).max()))


def _linear(value, dt):
    value = np.asarray(value, dtype=float)
    result = dict(frame_count=len(value),
                  endpoint_displacement=float(np.linalg.norm(value[-1] - value[0])),
                  excursion_from_first=_stats(value - value[0]),
                  spread_about_mean=_stats(value - value.mean(0)),
                  component_ranges=np.ptp(value, axis=0).tolist(),
                  path_length=float(np.linalg.norm(np.diff(value, axis=0), axis=-1).sum()))
    for order, name in ((1, 'velocity'), (2, 'acceleration'), (3, 'jerk')):
        result[name] = _stats(np.diff(value, n=order, axis=0) / dt ** order)
    return result


def _angular(value, dt):
    value = _rotations(value)
    offsets = _log_so3(value @ value[0].T)
    velocity = _log_so3(value[1:] @ value[:-1].swapaxes(-1, -2)) / dt
    return dict(endpoint_angle_rad=float(np.linalg.norm(offsets[-1])),
                excursion_from_first_rad=_stats(offsets),
                path_length_rad=float(np.linalg.norm(velocity, axis=-1).sum() * dt),
                velocity=_stats(velocity), acceleration=_stats(np.diff(velocity, axis=0) / dt),
                jerk=_stats(np.diff(velocity, n=2, axis=0) / dt ** 2))


def _window_mean(value, width=5):
    """Diagnostic five-frame box mean, edge repeated only within this phase."""
    width = min(width, len(value))
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return value.copy()
    padded = np.pad(value, ((width // 2, width // 2), (0, 0)), mode='edge')
    return np.stack([padded[i:i + width].mean(0) for i in range(len(value))])


def _ratio(candidate, baseline):
    return float(candidate / baseline) if baseline > 1e-12 else None


def _rotation_window_mean(value):
    matrices = _window_mean(value.reshape(len(value), 9)).reshape(-1, 3, 3)
    u, _, vh = np.linalg.svd(matrices)
    u = u.copy()
    u[..., :, -1] *= np.linalg.det(u @ vh)[..., None]
    return u @ vh


def _difference(baseline, candidate):
    bp, br, bq = baseline
    cp, cr, cq = candidate
    bp_path = _linear(_window_mean(bp), 1)['path_length']
    cp_path = _linear(_window_mean(cp), 1)['path_length']
    br_path = _angular(_rotation_window_mean(br), 1)['path_length_rad']
    cr_path = _angular(_rotation_window_mean(cr), 1)['path_length_rad']
    return dict(wrist_position_delta_m=_stats(cp - bp),
                wrist_orientation_delta_rad=_stats(_log_so3(cr @ br.swapaxes(-1, -2))),
                finger_delta_rad=_stats(cq - bq),
                wrist_start_delta_m=float(np.linalg.norm(cp[0] - bp[0])),
                wrist_end_delta_m=float(np.linalg.norm(cp[-1] - bp[-1])),
                wrist_endpoint_displacement_retention_ratio=_ratio(np.linalg.norm(cp[-1] - cp[0]), np.linalg.norm(bp[-1] - bp[0])),
                five_frame_mean_position_path_baseline_m=bp_path,
                five_frame_mean_position_path_candidate_m=cp_path,
                five_frame_mean_position_path_retention_ratio=_ratio(cp_path, bp_path),
                five_frame_mean_rotation_path_baseline_rad=br_path,
                five_frame_mean_rotation_path_candidate_rad=cr_path,
                five_frame_mean_rotation_path_retention_ratio=_ratio(cr_path, br_path),
                finger_component_range_retention_ratio=[_ratio(c, b) for c, b in zip(np.ptp(cq, axis=0), np.ptp(bq, axis=0))])


def _trajectory(arrays, side, prefix=''):
    return tuple(np.asarray(arrays[f'{prefix}{side}_{key}'], dtype=float)
                 for key in ('wrist_position_object', 'wrist_rotation_object', 'finger_qpos'))


def _metrics(trajectory, dt):
    p, r, q = trajectory
    return dict(wrist_translation_m=_linear(p, dt), wrist_rotation=_angular(r, dt),
                finger_rad=_linear(q, dt))


def _phases(config, side, count, labels, active):
    declared = config.get('contact_phases', {}).get(side)
    if declared is None:
        bounds = [0] + [i for i in range(1, count) if labels[i] != labels[i-1] or active[i] != active[i-1]] + [count]
        declared = [dict(frames=[a, b], role=str(labels[a])) for a, b in zip(bounds[:-1], bounds[1:])]
    phases = []
    covered = np.zeros(count, dtype=int)
    for index, phase in enumerate(declared):
        frames = phase.get('frames')
        if not isinstance(frames, (list, tuple)) or len(frames) != 2 or any(type(f) is not int for f in frames):
            raise ValueError('phase frames require two integer half-open bounds')
        start, end = frames
        if not 0 <= start < end <= count:
            raise ValueError('phase outside source frames')
        role = phase.get('role')
        if not isinstance(role, str) or set(labels[start:end]) != {role}:
            raise ValueError('declared roles disagree with archive')
        covered[start:end] += 1
        phases.append(dict(id=phase.get('id', f'{side}_phase_{index}'), frames=[start, end],
                           role=role, contact_active_frames=int(active[start:end].sum())))
    if not np.all(covered == 1) or [p['frames'][0] for p in phases] != sorted(p['frames'][0] for p in phases):
        raise ValueError('phases must cover the complete clock exactly once in order')
    return phases


def _material(baseline, candidate, side, start, end, role):
    """Optional fixed-reference evidence; no nearest-target recomputation."""
    names = {name: f'{side}_material_{name}' for name in ('points_object', 'point_ids', 'targets_object')}
    available = all(key in arrays for arrays in (baseline, candidate) for key in names.values())
    if not available:
        return dict(available=False, reason='Persistent material sample IDs, points and fixed targets not supplied in both archives.')
    ids = baseline[names['point_ids']]
    if not np.array_equal(ids, candidate[names['point_ids']]) or len(set(map(str, ids))) != len(ids):
        raise ValueError('material point identity changed or IDs are duplicated')
    targets = np.asarray(baseline[names['targets_object']], dtype=float)
    if not np.array_equal(targets, candidate[names['targets_object']]):
        raise ValueError('material reference targets changed between baseline and candidate')
    if targets.shape != (len(ids), 3) or not np.isfinite(targets).all() or not len(ids):
        raise ValueError('material targets must be nonempty finite fixed (K,3) object points')
    rows = {}
    for name, arrays in (('baseline', baseline), ('candidate', candidate)):
        points = np.asarray(arrays[names['points_object']], dtype=float)
        if points.shape != (len(arrays['timestamps']), len(ids), 3) or not np.isfinite(points).all():
            raise ValueError('material points must be finite (T,K,3)')
        values = points[start:end]
        rows[name] = dict(fixed_target_error_m=_stats((values - targets).reshape(-1, 3)),
                          material_excursion_from_phase_start_m=_stats((values - values[0]).reshape(-1, 3)),
                          material_endpoint_displacement_m=_stats(values[-1] - values[0]))
    return dict(available=True, point_count=len(ids), metrics=rows,
                interpreted_as_hold_slip_evidence=role in ('hold', 'grasp', 'stable_hold'),
                source_surface_identity_provenance_verified=False,
                limitation='Caller must verify IDs bind original robot material points and original object triangle/barycentric targets. This module does not infer physical contact or certify surface identity.')


def audit(config, baseline_arrays, candidate_arrays):
    """Audit two aligned archive mappings without modifying either mapping.

    Optional material keys per side: ``material_points_object`` (T,K,3),
    ``material_point_ids`` (K unique persistent labels), and
    ``material_targets_object`` (K,3 fixed object-local targets). Missing material
    evidence remains unavailable. Contacts/collisions require a separate audit.
    """
    if not all(isinstance(a, Mapping) or hasattr(a, 'files') for a in (baseline_arrays, candidate_arrays)):
        raise TypeError('archives must be mappings or np.load archives')
    b, c = baseline_arrays, candidate_arrays
    required = ('timestamps', 'fps', 'object_R', 'object_trans', 'object_scale', 'source_qpos')
    for key in required:
        if key not in b or key not in c:
            raise ValueError(f'missing required invariant {key}')
        if np.shape(b[key]) != np.shape(c[key]) or not np.isfinite(b[key]).all() or not np.isfinite(c[key]).all():
            raise ValueError(f'invalid shape or nonfinite invariant {key}')
    timestamps = np.asarray(b['timestamps'], dtype=float)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError('at least two source timestamps required')
    dt = float(np.median(np.diff(timestamps)))
    if dt <= 0 or not np.allclose(np.diff(timestamps), dt, rtol=1e-7, atol=1e-9):
        raise ValueError('audit requires a strictly increasing uniform source clock')
    if not np.isclose(float(b['fps']), 1 / dt, rtol=1e-6):
        raise ValueError('fps disagrees with source timestamps')
    count = len(timestamps)
    if np.shape(b['object_R']) != (count, 3, 3) or np.shape(b['object_trans']) != (count, 3):
        raise ValueError('object arrays must align with source frames')
    if np.ndim(b['source_qpos']) != 2 or np.shape(b['source_qpos'])[0] != count:
        raise ValueError('source_qpos must align with source frames')
    invariants = {key: bool(np.array_equal(b[key], c[key])) for key in required}
    sides = config.get('sides', list(config.get('inputs', {}).get('hands', {})))
    if not sides or len(set(sides)) != len(sides) or not set(sides).issubset({'left', 'right'}):
        raise ValueError('explicit distinct active hand sides required')
    result = dict(schema='hand_contact_stability_audit.v1', frame_count=count, fps=1 / dt,
                  invariants=invariants, hands={}, motion_accepted=False,
                  full_mesh_collision_certified=False,
                  definitions=dict(coordinates='Object-relative meters/radians; no object-scale division.',
                    derivatives='SI finite differences, only stencils within one authored phase. Short intervals yield null.',
                    angular='Spatial log(R[t+1]R[t]^T)/dt in the fixed object frame; not Euler differences.',
                    fidelity='Comparisons use the same immutable baseline. Five-frame box mean is a diagnostic, not source ground truth or a pass gate.',
                    contact='No distance-field contact score is recomputed here; optional fixed material evidence is separate from nearest-surface proxies.'))
    for side in sides:
        for arrays in (b, c):
            for key in (f'{side}_phase_labels', f'{side}_contact_active'):
                if key not in arrays or np.shape(arrays[key]) != (count,):
                    raise ValueError(f'missing or invalid {key}')
            p, r, q = _trajectory(arrays, side)
            if p.shape != (count, 3) or r.shape != (count, 3, 3) or q.shape != (count, 7):
                raise ValueError('hand trajectory shape must be (T,3), (T,3,3), (T,7)')
            if not all(np.isfinite(v).all() for v in (p, r, q)):
                raise ValueError('hand trajectories must be finite')
            _rotations(r)
        labels = np.asarray(b[f'{side}_phase_labels']).astype(str)
        active = np.asarray(b[f'{side}_contact_active'])
        if active.dtype.kind != 'b':
            raise ValueError('contact activity must be Boolean')
        phase_equal = bool(np.array_equal(labels, c[f'{side}_phase_labels']))
        active_equal = bool(np.array_equal(active, c[f'{side}_contact_active']))
        if side in config.get('phase_labels', {}) and not np.array_equal(labels, config['phase_labels'][side]):
            raise ValueError('config labels disagree with frozen baseline')
        if side in config.get('contact_mask', {}) and not np.array_equal(active, config['contact_mask'][side]):
            raise ValueError('config activity disagrees with frozen baseline')
        baseline = _trajectory(b, side)
        candidate = _trajectory(c, side)
        source_keys = [f'source_{side}_{name}' for name in ('wrist_position_object', 'wrist_rotation_object', 'finger_qpos')]
        source_available = all(key in b and key in c for key in source_keys)
        any_source = any(key in b or key in c for key in source_keys)
        source_equal = (source_available and all(np.array_equal(b[key], c[key]) for key in source_keys)) if any_source else None
        if source_available:
            for key, expected in zip(source_keys, ((count, 3), (count, 3, 3), (count, 7))):
                if any(np.shape(arrays[key]) != expected or not np.isfinite(arrays[key]).all() for arrays in (b, c)):
                    raise ValueError('source hand prior arrays must be finite and aligned')
        first_equal = all(np.array_equal(x[0], y[0]) for x, y in zip(baseline, candidate))
        rows = []
        phases = _phases(config, side, count, labels, active)
        for phase in phases:
            start, end = phase['frames']
            before = tuple(v[start:end] for v in baseline)
            after = tuple(v[start:end] for v in candidate)
            row = {**phase, 'baseline': _metrics(before, dt), 'candidate': _metrics(after, dt),
                   'candidate_minus_baseline': _difference(before, after),
                   'material_evidence': _material(b, c, side, start, end, phase['role']),
                   'zero_relative_motion_authorized_by_this_audit': False}
            if source_available:
                source = tuple(v[start:end] for v in _trajectory(b, side, 'source_'))
                row['baseline_minus_source'] = _difference(source, before)
                row['candidate_minus_source'] = _difference(source, after)
            rows.append(row)
        boundaries = []
        for prev, phase in zip(phases[:-1], phases[1:]):
            frame = phase['frames'][0]
            row = dict(frame=frame, from_role=prev['role'], to_role=phase['role'])
            for name, (p, r, q) in (('baseline', baseline), ('candidate', candidate)):
                values = dict(translation_jump_m=float(np.linalg.norm(p[frame] - p[frame-1])),
                              angular_jump_rad=float(np.linalg.norm(_log_so3(r[frame] @ r[frame-1].T))),
                              finger_peak_joint_jump_rad=float(np.abs(q[frame] - q[frame-1]).max()))
                if 1 < frame < count - 1:
                    values['translation_velocity_change_m_s'] = float(np.linalg.norm((p[frame+1] - p[frame]) - (p[frame-1] - p[frame-2])) / dt)
                row[name] = values
            boundaries.append(row)
        result['hands'][side] = dict(phases=rows, boundaries=boundaries,
            phase_labels_unchanged=phase_equal, contact_activity_unchanged=active_equal,
            source_prior_arrays_available=source_available, source_prior_arrays_unchanged=source_equal,
            first_frame_unchanged=first_equal,
            first_frame_difference=_difference(tuple(v[:1] for v in baseline), tuple(v[:1] for v in candidate)),
            boundary_peak_candidate={key: max((row['candidate'].get(key, 0.) for row in boundaries), default=None)
                                     for key in ('translation_jump_m', 'angular_jump_rad', 'finger_peak_joint_jump_rad')})
    result['required_invariants_preserved'] = all(invariants.values()) and all(
        h['phase_labels_unchanged'] and h['contact_activity_unchanged'] and h['first_frame_unchanged']
        and h['source_prior_arrays_unchanged'] is not False for h in result['hands'].values())
    return result
