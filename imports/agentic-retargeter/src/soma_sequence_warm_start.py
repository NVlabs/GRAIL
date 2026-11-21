"""Optional global hinge-sequence seed before SOMA's normal coupled sweeps.

This solves a small fidelity-plus-temporal quadratic using the *same* derivative
units/energy multipliers as the conditional temporal objective. It is only a
starting state: human, contact, collision and object objectives must still be
solved normally afterward. It neither edits source targets nor filters exports.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

import numpy as np


def validate_sequence_warm_start_config(value):
    """Require explicit bound hinge names and a strictly positive fidelity cost."""
    if not isinstance(value, dict) or set(value) != {'joint_names', 'fidelity_weight'}:
        raise ValueError('sequence_warm_start requires joint_names and fidelity_weight')
    names = value['joint_names']
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name or name != name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('sequence_warm_start joint_names must be unique explicit names')
    weight = value['fidelity_weight']
    if (isinstance(weight, bool) or not isinstance(weight, Real)
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError('sequence_warm_start fidelity_weight must be finite and positive')
    return {'joint_names': list(names), 'fidelity_weight': float(weight)}


def _resolve_selected(requested, bound_names):
    names = list(bound_names)
    if (not names or any(not isinstance(name, str) or not name or name != name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Bound sequence hinge names must be unique nonempty strings')
    selected = []
    for requested_name in requested:
        matches = [i for i, name in enumerate(names) if name == requested_name]
        if not matches:
            matches = [i for i, name in enumerate(names) if name.rsplit('/', 1)[-1] == requested_name]
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError(f'Sequence warm-start name is absent, ambiguous or aliased: {requested_name!r}')
        selected.append(matches[0])
    return names, np.asarray(selected, dtype=np.int64)


def derivative_gram_matrix(frame_count, fps, temporal_config):
    """Return sum lambda_k * fps**(2*k) * D_k.T@D_k, without endpoint padding."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, (int, np.integer)) or frame_count < 1:
        raise ValueError('Sequence frame_count must be a positive integer')
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Sequence fps must be finite and positive')
    result = np.zeros((frame_count, frame_count), dtype=np.float64)
    identity = np.eye(frame_count, dtype=np.float64)
    for order, derivative in enumerate(('velocity', 'acceleration', 'jerk'), 1):
        weight = temporal_config[derivative+'_weight']
        if (isinstance(weight, bool) or not isinstance(weight, Real)
                or not math.isfinite(weight) or weight < 0):
            raise ValueError('Sequence derivative weights must be finite and nonnegative')
        if frame_count > order and weight > 0:
            difference = np.diff(identity, n=order, axis=0)
            result += weight * fps**(2*order) * (difference.T @ difference)
    if not np.isfinite(result).all():
        raise ValueError('Sequence temporal quadratic overflowed')
    return result


def _hash_array(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _energy(value, reference, fps, temporal_config, weights, fidelity):
    value = np.asarray(value, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    result = {'fidelity': float(fidelity*np.sum((value-reference)**2)), 'derivatives': {}}
    for order, derivative in enumerate(('velocity', 'acceleration', 'jerk'), 1):
        delta = np.diff(value, n=order, axis=0) * fps**order
        result['derivatives'][derivative] = float(temporal_config[derivative+'_weight'] * np.sum(delta**2 * weights))
    result['temporal'] = sum(result['derivatives'].values())
    result['total'] = result['fidelity'] + result['temporal']
    return result


def _representable_bounds(lower, upper, dtype):
    """Native float32 seeds must not round outward past bound float64 limits."""
    lo, hi = lower.astype(dtype), upper.astype(dtype)
    lo = np.where(lo.astype(np.float64) < lower, np.nextafter(lo, np.full_like(lo, np.inf)), lo)
    hi = np.where(hi.astype(np.float64) > upper, np.nextafter(hi, np.full_like(hi, -np.inf)), hi)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError('No finite native-dtype value exists inside one selected hinge limit')
    return lo, hi


def sequence_warm_start(q, fps, temporal_config, joint_names, lower_limits, upper_limits):
    """Return ``(seeded_hinges, JSON_report, NPZ_evidence_arrays)``.

    ``q`` is [source frames, bound scalar hinges], excluding synthetic warmup
    frames and free-root coordinates. Only explicitly named columns are edited;
    all others are byte-identical. Caller binds these hinge columns to the real
    model and preserves root/object/body/source arrays outside selected hinges.

    For each selected hinge j, solve exactly (up to floating arithmetic):
      min_x mu ||x - q_forward[:,j]||^2 + w_j sum_k lambda_k ||D_k x||^2 fps^(2k).
    The unconstrained dense SPD solve is followed by explicit elementwise native
    limit projection. Clipping is NOT an exact solution of the constrained
    temporal problem. Saved evidence includes all stages and replay inputs.
    """
    if __package__:
        from .soma_temporal import joint_weight_vector, validate_temporal_config
    else:
        from soma_temporal import joint_weight_vector, validate_temporal_config
    config = validate_temporal_config(temporal_config)
    if 'sequence_warm_start' not in config:
        raise ValueError('sequence_warm_start is an explicit opt-in; no seed requested')
    controls = config['sequence_warm_start']
    original = np.asarray(q)
    if (original.ndim != 2 or min(original.shape) < 1
            or original.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(original).all()):
        raise ValueError('Sequence hinges must be finite float32/float64 [frames, joints]')
    names, selected = _resolve_selected(controls['joint_names'], joint_names)
    if original.shape[1] != len(names):
        raise ValueError('Sequence hinge columns do not match bound joint names')
    lower, upper = np.asarray(lower_limits, dtype=np.float64), np.asarray(upper_limits, dtype=np.float64)
    if (lower.shape != (len(names),) or upper.shape != lower.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper)):
        raise ValueError('Finite lower/upper native limits must bind every hinge column')
    gram = derivative_gram_matrix(len(original), fps, config)
    full_weights = joint_weight_vector(config, names)
    weights = full_weights[selected]
    fidelity = controls['fidelity_weight']
    reference = original[:, selected].astype(np.float64)
    unconstrained = reference.copy()
    stationarity = np.zeros(len(selected), dtype=np.float64)
    matrices = {}
    # Reuse one factorization/solve for equal joint energy multipliers.
    for weight in np.unique(weights):
        columns = np.flatnonzero(weights == weight)
        matrix = fidelity*np.eye(len(original)) + weight*gram
        if not np.isfinite(matrix).all():
            raise ValueError('Sequence fidelity-plus-temporal matrix overflowed')
        if np.any(gram):
            unconstrained[:, columns] = np.linalg.solve(matrix, fidelity*reference[:, columns])
        # A constant reference has zero temporal energy: preserve it exactly.
        constant = np.all(reference[:, columns] == reference[0, columns], axis=0)
        unconstrained[:, columns[constant]] = reference[:, columns[constant]]
        stationarity[columns] = np.max(np.abs(matrix@unconstrained[:, columns] - fidelity*reference[:, columns]), axis=0)
        matrices[str(float(weight))] = _hash_array(matrix)
    if not np.isfinite(unconstrained).all():
        raise ValueError('Global sequence solve produced nonfinite hinge values')
    low, high = lower[selected], upper[selected]
    projected = np.clip(unconstrained, low, high)
    lo_native, hi_native = _representable_bounds(low, high, original.dtype)
    seeded = original.copy()
    seeded[:, selected] = np.clip(projected.astype(original.dtype), lo_native, hi_native)
    if (not np.isfinite(seeded).all() or np.any(seeded[:, selected] < low)
            or np.any(seeded[:, selected] > high)):
        raise ValueError('Global sequence seed is nonfinite or outside native limits')
    protected = np.asarray([i for i in range(len(names)) if i not in set(selected)], dtype=np.int64)
    if not np.array_equal(seeded[:, protected], original[:, protected]):
        raise RuntimeError('Protected sequence hinges changed')
    unclipped = original.astype(np.float64)
    unclipped[:, selected] = unconstrained
    arrays = {'initial_hinge_q': original.copy(), 'unconstrained_hinge_q': unclipped,
              'seeded_hinge_q': seeded.copy(), 'joint_names': np.asarray(names),
              'selected_indices': selected.copy(), 'lower_limits': lower.copy(),
              'upper_limits': upper.copy(), 'fps': np.asarray(float(fps)),
              'temporal_energy_multipliers': full_weights.copy(),
              'temporal_config_json': np.asarray(json.dumps(config, sort_keys=True))}
    changes = seeded[:, selected].astype(np.float64)-reference
    clipped = (unconstrained < low) | (unconstrained > high)
    per_joint = []
    for i, column in enumerate(selected):
        per_joint.append({'joint_name': names[column], 'hinge_column': int(column),
            'temporal_energy_multiplier': float(weights[i]),
            'maximum_abs_change_rad': float(np.max(np.abs(changes[:, i]))),
            'rms_change_rad': float(np.sqrt(np.mean(changes[:, i]**2))),
            'clipped_sample_count': int(clipped[:, i].sum()),
            'maximum_projection_rad': float(np.max(np.abs(projected[:, i]-unconstrained[:, i]))),
            'unconstrained_stationarity_residual_max_abs': float(stationarity[i]),
            'lower_limit_rad': float(low[i]), 'upper_limit_rad': float(high[i])})
    report = {'schema': 'soma_global_hinge_sequence_warm_start.v1',
        'stage': 'pre_sweep_initial_state_only', 'source_targets_modified': False,
        'post_solve_output_filter': False, 'normal_coupled_sweeps_required_afterward': True,
        'controls': controls, 'temporal_config': config, 'fps': float(fps),
        'frame_count': len(original), 'bound_hinge_count': len(names),
        'selected_joint_names': [names[i] for i in selected], 'selected_indices': selected.tolist(),
        'protected_hinges_bit_exact': True, 'returned_dtype': str(seeded.dtype),
        'returned_selected_hinges_in_native_limits': True, 'per_joint': per_joint,
        'clipped_sample_count': int(clipped.sum()),
        'unconstrained_minimum': 'Dense SPD linear solve, subject to floating-point accuracy.',
        'exact_constrained_minimum_claimed': False,
        'projection': 'Elementwise native hinge-limit clipping, then inward native-dtype rounding if needed.',
        'matrix_shape': list(gram.shape), 'matrix_sha256_by_energy_multiplier': matrices,
        'array_sha256': {key: _hash_array(value) for key, value in arrays.items()},
        'quadratic': {'before': _energy(reference, reference, fps, config, weights, fidelity),
            'unconstrained': _energy(unconstrained, reference, fps, config, weights, fidelity),
            'projected_float64': _energy(projected, reference, fps, config, weights, fidelity),
            'returned_seed': _energy(seeded[:, selected], reference, fps, config, weights, fidelity)},
        'maximum_native_rounding_rad': float(np.max(np.abs(seeded[:, selected].astype(np.float64)-projected))),
        'limitations': [
            'Only fidelity to forward hinge states and the temporal quadratic enter this starting-point solve.',
            'Original human, material contact, collision, object and joint-limit objectives must run in subsequent normal SOMA sweeps.',
            'Limit clipping does not solve the coupled constrained quadratic exactly and may increase seed energy.',
            'Seed smoothness does not establish final motion quality, lag, grasp, clearance or physical feasibility.',
            'This function receives hinges only; caller must independently bind selected columns and preserve free root/object/source data.']}
    if not all(math.isfinite(stage['total']) for stage in report['quadratic'].values()):
        raise ValueError('Sequence seed quadratic energy overflowed')
    return seeded, report, arrays
