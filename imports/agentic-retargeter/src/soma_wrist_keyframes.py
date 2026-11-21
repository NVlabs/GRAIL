"""Hash-bound sparse authored object-relative wrist milestones.

Additive soft objectives only: existing hand trajectories, body targets and
object costs retain their original definitions. Local translations are metres
(original object scale already applied), never unscaled asset coordinates.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_hash(rotations, positions, scale):
    digest = hashlib.sha256(b'soma_wrist_reference.v1')
    for value in (rotations, positions, np.broadcast_to(scale, (3,))):
        array = np.ascontiguousarray(value, dtype='<f8')
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite wrist reference')
        digest.update(repr(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def validate_config(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
            or not isinstance(value['path'], str) or not value['path']
            or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in value['sha256'])):
        raise ValueError('wrist_keyframes requires path and lowercase SHA256')
    return dict(value)


def load(config, *, model_xml, source_bvh, frame_count, fps, hand_targets, base_path='.'):
    config = validate_config(config)
    path = (Path(base_path)/config['path']).resolve(strict=True)
    if sha(path) != config['sha256']:
        raise ValueError('Wrist keyframe artifact hash changed')
    spec = json.loads(path.read_text())
    required = {'schema', 'model_xml_sha256', 'source_bvh_sha256', 'frame_count', 'fps',
                'handoff_sha256', 'targets_sha256', 'object_mesh_sha256',
                'object_reference_sha256', 'authored_native', 'keyframes'}
    if (not isinstance(spec, dict) or set(spec)-required-{'provenance'} or required-set(spec)
            or spec['schema'] != 'soma_object_relative_wrist_keyframes.v1'):
        raise ValueError('Invalid wrist keyframe artifact fields/schema')
    if (type(frame_count) is not int or frame_count < 1 or type(spec['frame_count']) is not int
            or spec['frame_count'] != frame_count or isinstance(spec['fps'], bool)
            or not isinstance(spec['fps'], (int, float)) or not math.isfinite(spec['fps'])
            or spec['fps'] <= 0 or not math.isclose(spec['fps'], fps, abs_tol=1e-8, rel_tol=0)):
        raise ValueError('Wrist keyframe source clock differs')
    if spec['model_xml_sha256'] != sha(model_xml) or spec['source_bvh_sha256'] != sha(source_bvh):
        raise ValueError('Wrist keyframe original model/source binding differs')
    for name in ('handoff_sha256', 'targets_sha256', 'object_mesh_sha256'):
        if spec[name] != hand_targets['report'][name]:
            raise ValueError('Wrist keyframe original hand/object lineage differs: '+name)
    if spec['object_reference_sha256'] != reference_hash(hand_targets['object_reference_rotations'],
            hand_targets['object_reference_positions'], hand_targets['object_scale']):
        raise ValueError('Wrist keyframe solver object reference differs')
    authored = validate_config(spec['authored_native'])
    native_path = (path.parent/authored['path']).resolve(strict=True)
    if sha(native_path) != authored['sha256']:
        raise ValueError('Wrist keyframe authored native changed')
    if 'provenance' in spec and not isinstance(spec['provenance'], dict):
        raise ValueError('Wrist keyframe provenance must be an object')
    keys = spec['keyframes']
    if not isinstance(keys, list) or not keys:
        raise ValueError('Wrist keyframes require explicit sparse keys')
    hands, rows, seen = {}, [], set()
    fields = {'frame', 'hand', 'wrist_body', 'position_object', 'rotation_object',
              'translation_weight', 'rotation_weight', 'translation_sigma_m', 'rotation_sigma_rad'}
    for key in keys:
        if not isinstance(key, dict) or set(key) != fields:
            raise ValueError('Invalid sparse wrist key fields')
        f, side = key['frame'], key['hand']
        if type(f) is not int or not 0 <= f < frame_count:
            raise ValueError('Wrist keyframe outside original timeline')
        if side not in hand_targets['hands'] or key['wrist_body'] != hand_targets['hands'][side]['wrist_body']:
            raise ValueError('Wrist keyframe hand/body is not enabled in bound hand targets')
        if (f, side) in seen:
            raise ValueError('Duplicate wrist keyframe hand/frame')
        seen.add((f, side))
        p, r = np.asarray(key['position_object'], dtype=float), np.asarray(key['rotation_object'], dtype=float)
        if p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all():
            raise ValueError('Wrist keyframe pose must be finite')
        if not np.allclose(r.T@r, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(r), 1., atol=1e-7, rtol=0):
            raise ValueError('Wrist keyframe rotation must be SO(3)')
        gains = []
        for prefix, sigma in (('translation', 'translation_sigma_m'), ('rotation', 'rotation_sigma_rad')):
            for name in (prefix+'_weight', sigma):
                v = key[name]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise ValueError('Wrist keyframe weight/sigma must be finite positive')
            gain = math.sqrt(key[prefix+'_weight'])/key[sigma]
            if not math.isfinite(gain) or gain > np.finfo(np.float32).max:
                raise ValueError('Wrist keyframe gain exceeds float32')
            gains.append(gain)
        if side not in hands:
            hands[side] = dict(wrist_body=key['wrist_body'],
                wrist_position_object=np.zeros((frame_count, 3)),
                wrist_rotation_object=np.repeat(np.eye(3)[None], frame_count, axis=0),
                gains=np.zeros((frame_count, 2), dtype=np.float32))
        hands[side]['wrist_position_object'][f] = p
        hands[side]['wrist_rotation_object'][f] = r
        hands[side]['gains'][f] = gains
        rows.append(dict(key))
    for hand in hands.values():
        for name in ('wrist_position_object', 'wrist_rotation_object', 'gains'):
            hand[name].flags.writeable = False
    report = dict(schema='soma_object_relative_wrist_keyframe_binding.v1',
        path=str(path), sha256=config['sha256'], frame_count=frame_count, fps=float(fps),
        keys=rows, key_count=len(rows), enabled_hands=list(hands),
        input_files=dict(keys=dict(path=str(path), sha256=config['sha256']),
            authored_native=dict(path=str(native_path), sha256=authored['sha256'])),
        lineage={name:spec[name] for name in required if name.endswith('_sha256')},
        provenance=spec.get('provenance', {}), motion_accepted=False, soft_constraints=True,
        target_rule='Live-object relative at explicit frame/hand pairs; exactly zero elsewhere.',
        original_hand_targets_retained=True, root_or_object_pins=False)
    return dict(config=dict(path=str(path), sha256=config['sha256']), report=report,
                hands=hands, timestamps=np.arange(frame_count)/fps)


def measure(loaded, positions, rotations):
    """Replay diagnostics from object-relative mounted-FK wrist states."""
    sides = loaded['report']['enabled_hands']; frames = len(loaded['timestamps'])
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    if (positions.shape != (frames, len(sides), 3) or rotations.shape != (frames, len(sides), 3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()):
        raise ValueError('Invalid actual relative wrist state arrays')
    rows = []
    for key in loaded['report']['keys']:
        f, s = key['frame'], sides.index(key['hand'])
        error = positions[f, s]-key['position_object']
        angle = Rotation.from_matrix(rotations[f, s] @ np.asarray(key['rotation_object']).T).magnitude()
        rows.append(dict(frame=f, hand=key['hand'], translation_error_m=float(np.linalg.norm(error)),
                         rotation_error_rad=float(angle)))
    return dict(rows=rows, max_translation_error_m=max(x['translation_error_m'] for x in rows),
        max_rotation_error_rad=max(x['rotation_error_rad'] for x in rows),
        soft_constraints=True, motion_accepted=False)


def measurements_match(actual, recorded):
    """Allow floating SO(3) roundoff across pinned runtime/adapter SciPy builds.

    Input state/file hashes remain exact. This 1e-10 tolerance applies solely to
    recomputed scalar error diagnostics, not poses, key gains or geometry gates.
    """
    if not isinstance(recorded, dict) or set(recorded) != set(actual):
        return False
    if recorded['soft_constraints'] is not True or recorded['motion_accepted'] is not False:
        return False
    def same(a, b):
        return (not isinstance(b, bool) and isinstance(b, (float, int))
                and math.isfinite(b) and math.isclose(a, b, rel_tol=0., abs_tol=1e-10))
    for name in ('max_translation_error_m', 'max_rotation_error_rad'):
        if not same(actual[name], recorded[name]):
            return False
    if not isinstance(recorded['rows'], list) or len(recorded['rows']) != len(actual['rows']):
        return False
    for a, b in zip(actual['rows'], recorded['rows']):
        if (not isinstance(b, dict) or set(b) != set(a) or type(b['frame']) is not int
                or b['frame'] != a['frame'] or b['hand'] != a['hand']
                or not same(a['translation_error_m'], b['translation_error_m'])
                or not same(a['rotation_error_rad'], b['rotation_error_rad'])):
            return False
    return True
