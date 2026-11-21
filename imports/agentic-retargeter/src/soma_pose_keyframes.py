"""Explicit hash-bound native hinge milestones for fresh SOMA solves.

These are sparse soft costs, not hard pins. Free roots are intentionally outside
this contract. A repaired mounted key remains an unaccepted hypothesis.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
            or not isinstance(value['path'], str) or not value['path']
            or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in value['sha256'])):
        raise ValueError('pose_keyframes requires path and lowercase SHA256')
    return dict(value)


def load(config, *, model_xml, source_bvh, frame_count, fps, joint_limits, base_path='.'):
    """Validate exact artifact bytes, original model/source and named hinge limits.

    joint_limits maps original scalar hinge names to [lower, upper]. The caller
    must exclude free/ball/slide joints rather than presenting them as hinges.
    """
    config = validate_config(config)
    path = (Path(base_path)/config['path']).resolve(strict=True)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != config['sha256']:
        raise ValueError('Pose keyframe artifact hash changed')
    spec = json.loads(data)
    required = {'schema', 'model_xml_sha256', 'source_bvh_sha256', 'frame_count', 'fps', 'keyframes'}
    if not isinstance(spec, dict) or set(spec)-required-{'provenance'} or required-set(spec):
        raise ValueError('Invalid pose keyframe artifact fields')
    if spec['schema'] != 'soma_native_pose_keyframes.v1':
        raise ValueError('Invalid pose keyframe schema')
    if (type(frame_count) is not int or frame_count < 1
            or type(spec['frame_count']) is not int or spec['frame_count'] != frame_count
            or isinstance(spec['fps'], bool) or not isinstance(spec['fps'], (int, float))
            or not math.isfinite(spec['fps']) or spec['fps'] <= 0
            or not math.isclose(spec['fps'], fps, abs_tol=1e-8, rel_tol=0)):
        raise ValueError('Pose keyframe source clock differs')
    if spec['model_xml_sha256'] != sha(model_xml) or spec['source_bvh_sha256'] != sha(source_bvh):
        raise ValueError('Pose keyframe original model/source binding differs')
    if 'provenance' in spec and not isinstance(spec['provenance'], dict):
        raise ValueError('Pose keyframe provenance must be an explicit object')
    keys = spec['keyframes']
    if not isinstance(keys, list) or not keys:
        raise ValueError('At least one explicit pose keyframe is required')
    seen, rows = set(), []
    for key in keys:
        if not isinstance(key, dict) or set(key) != {'frame', 'joints', 'weight', 'sigma_rad'}:
            raise ValueError('Pose keyframe requires frame, joints, weight, sigma_rad')
        f, joints = key['frame'], key['joints']
        if type(f) is not int or not 0 <= f < frame_count:
            raise ValueError('Pose keyframe outside original source timeline')
        if not isinstance(joints, dict) or not joints:
            raise ValueError('Pose keyframe must name native scalar hinges')
        for name in ('weight', 'sigma_rad'):
            value = key[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Pose keyframe weights/sigma must be finite positive')
        gain = math.sqrt(key['weight'])/key['sigma_rad']
        if not math.isfinite(gain) or gain > np.finfo(np.float32).max:
            raise ValueError('Pose keyframe gain exceeds float32 range')
        for name, angle in joints.items():
            if not isinstance(name, str) or name not in joint_limits:
                raise ValueError('Pose keyframe joint is not a bound scalar hinge: '+str(name))
            if (f, name) in seen:
                raise ValueError('Duplicate/conflicting pose keyframe frame and hinge')
            if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle):
                raise ValueError('Pose keyframe angle must be finite native radians')
            lower, upper = joint_limits[name]
            if not lower <= angle <= upper:
                raise ValueError('Pose keyframe exceeds native joint limits: '+name)
            seen.add((f, name))
            rows.append(dict(frame=f, joint=name, target_rad=float(angle),
                             weight=float(key['weight']), sigma_rad=float(key['sigma_rad']), gain=gain))
    names = sorted({row['joint'] for row in rows})
    targets = np.zeros((frame_count, len(names)), dtype=np.float32)
    gains = np.zeros_like(targets)
    for row in rows:
        j = names.index(row['joint']); f = row['frame']
        targets[f, j], gains[f, j] = row['target_rad'], row['gain']
    targets.flags.writeable = gains.flags.writeable = False
    report = dict(schema='soma_native_pose_keyframe_binding.v1', path=str(path), sha256=config['sha256'],
        model_xml_sha256=spec['model_xml_sha256'], source_bvh_sha256=spec['source_bvh_sha256'],
        frame_count=frame_count, fps=float(fps), joint_names=names, key_count=len(rows), keys=rows,
        provenance=spec.get('provenance', {}), soft_constraints=True, motion_accepted=False,
        root_targets=False, original_source_objectives_retained=True,
        residual_definition='sqrt(weight)/sigma_rad * (native_hinge_rad - authored_key_rad); zero away from explicit key frames')
    return dict(config=dict(path=str(path), sha256=config['sha256']), report=report,
                joint_names=names, targets=targets, gains=gains)


def measure(loaded, values):
    """Measure all authored keys in dense source-time selected-hinge arrays."""
    q = np.asarray(values)
    if q.shape != loaded['targets'].shape or not np.isfinite(q).all():
        raise ValueError('Pose keyframe measurement requires all source times/named hinges')
    rows = []
    for key in loaded['report']['keys']:
        actual = float(q[key['frame'], loaded['joint_names'].index(key['joint'])])
        # Match executed float32 target, while preserving authored radians too.
        target = float(np.float32(key['target_rad']))
        rows.append(dict(frame=key['frame'], joint=key['joint'], authored_target_rad=key['target_rad'],
            executed_target_rad=target, actual_rad=actual, error_rad=actual-target, abs_error_rad=abs(actual-target)))
    return dict(rows=rows, max_abs_error_rad=max(row['abs_error_rad'] for row in rows),
                rms_error_rad=float(np.sqrt(np.mean([row['error_rad']**2 for row in rows]))),
                soft_constraints=True, motion_accepted=False)
