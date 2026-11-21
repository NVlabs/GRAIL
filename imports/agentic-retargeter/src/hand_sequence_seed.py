"""Coherent free-hand warm starts from explicitly selected sequence keys.

This experimental helper alters candidate initialization only. Source qpos,
object motion, timing and contact metadata remain byte-identical. A fixed grip
is propagated only inside an explicitly declared hold; support and sliding
phases are never silently converted into a no-slip relation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _blend_rotation(a, b, weight):
    if weight <= 0:
        return a.copy()
    if weight >= 1:
        return b.copy()
    return Slerp([0., 1.], Rotation.from_matrix(np.stack((a, b))))([weight]).as_matrix()[0]


def _smoothstep(t):
    return t*t*(3-2*t)


def normalized_roles(config, side, count):
    roles = list(config.get('phase_labels', {}).get(side, ['unknown']*count))
    if len(roles) != count:
        raise ValueError('phase_labels must match all original frames')
    for phase in config.get('contact_phases', {}).get(side, []):
        if phase.get('state') == 'contact' and phase.get('role'):
            first, end = phase['frames']
            roles[first:end] = [phase['role'].replace('-', '_')]*(end-first)
    return roles


def build_seed(arrays, config, selection, *, transition_frames=8):
    """Return candidate arrays and scope evidence without changing any inputs.

    ``selection`` maps side to [{start,end,frame}], with inclusive endpoints.
    Every selected frame must lie inside its explicit hold. Optional per-key
    wrist_position_object, wrist_rotation_object and finger_qpos can carry an
    external candidate, but all three must be supplied together; the provenance
    is included verbatim and requires independent geometry review.
    """
    if type(transition_frames) is not int or not 0 <= transition_frames <= 120:
        raise ValueError('transition_frames must be a bounded nonnegative integer')
    if config.get('is_static_obj', False) and any(selection.values()):
        raise ValueError('Static objects have no contact-hold seed')
    count = len(arrays['timestamps'])
    output = {key: value.copy() for key, value in arrays.items()}
    report = dict(schema_version=1, scope='free-hand warm-start only', motion_accepted=False,
                  transition_frames=transition_frames, hands={})
    strict_hold = {'hold', 'grasp', 'stable_hold'}
    blendable = {'no_contact', 'unknown', 'contact_acquisition', 'approach', 'occluded_approach', 'release'}
    for side, records in selection.items():
        pkey, rkey, qkey = [f'{side}_{suffix}' for suffix in
                          ('wrist_position_object', 'wrist_rotation_object', 'finger_qpos')]
        if any(key not in arrays for key in (pkey, rkey, qkey)):
            raise ValueError('Selected hand not present in warm trajectory')
        roles = normalized_roles(config, side, count)
        active = np.asarray(config.get('contact_mask', {}).get(side, arrays.get(f'{side}_contact_active')))
        if active.shape != (count,) or active.dtype.kind != 'b':
            raise ValueError('Exact original contact mask required')
        touched = set()
        events = []
        for row in records:
            first, last, frame = [row[k] for k in ('start', 'end', 'frame')]
            if (any(type(x) is not int for x in (first,last,frame))
                    or not 0 <= first <= frame <= last < count):
                raise ValueError('Selected key and inclusive interval must be valid source frames')
            if not active[first:last+1].all() or any(v not in strict_hold for v in roles[first:last+1]):
                raise ValueError('Only explicit stable hold roles can receive a fixed object-relative seed')
            if touched.intersection(range(first,last+1)):
                raise ValueError('Selected hold intervals overlap')
            target_keys = ('wrist_position_object', 'wrist_rotation_object', 'finger_qpos')
            supplied = [key in row for key in target_keys]
            if any(supplied) and not all(supplied):
                raise ValueError('External candidate must supply all wrist/finger values')
            if all(supplied):
                p, r, q = [np.asarray(row[key], dtype=np.float64) for key in target_keys]
            else:
                p, r, q = [arrays[key][frame].copy() for key in (pkey,rkey,qkey)]
            if (p.shape != (3,) or r.shape != (3,3) or q.shape != arrays[qkey].shape[1:]
                    or not all(np.isfinite(v).all() for v in (p,r,q))
                    or not np.allclose(r.T@r,np.eye(3),atol=2e-5) or not np.isclose(np.linalg.det(r),1,atol=2e-5)):
                raise ValueError('Invalid rigid hand candidate')
            changes = []
            for index in range(max(0,first-transition_frames), min(count,last+transition_frames+1)):
                if first <= index <= last:
                    weight = 1.
                elif roles[index] not in blendable or index in touched:
                    continue
                elif index < first:
                    weight = _smoothstep((index-(first-transition_frames))/(transition_frames or 1))
                else:
                    weight = _smoothstep(((last+transition_frames)-index)/(transition_frames or 1))
                if weight <= 0:
                    continue
                output[pkey][index] = (1-weight)*arrays[pkey][index]+weight*p
                output[rkey][index] = _blend_rotation(arrays[rkey][index],r,weight)
                output[qkey][index] = (1-weight)*arrays[qkey][index]+weight*q
                touched.add(index)
                changes.append(dict(frame=index,weight=float(weight),role=roles[index]))
            events.append(dict(selection=row, changed=changes,
                               rule='constant object-relative pose only inside explicit hold; smooth source-clock transition outside'))
        report['hands'][side] = dict(events=events, edited_frames=sorted(touched),
                                    support_slide_unchanged=all(np.array_equal(output[key][i],arrays[key][i])
                                       for key in (pkey,rkey,qkey) for i,role in enumerate(roles)
                                       if role in ('support','slide','sliding')))
    protected = [key for key in arrays if not any(key == f'{side}_{suffix}' for side in selection
                 for suffix in ('wrist_position_object','wrist_rotation_object','finger_qpos'))]
    if any(not np.array_equal(output[key],arrays[key]) for key in protected):
        raise AssertionError('Warm-start helper altered protected source metadata')
    report['all_source_body_object_timing_metadata_unchanged'] = True
    return output, report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--selection',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--transition-frames',type=int,default=8)
    args=parser.parse_args()
    paths = {key:Path(getattr(args,key)).resolve(strict=True) for key in ('input','config','selection')}
    with np.load(paths['input'],allow_pickle=False) as data:
        arrays={key:data[key].copy() for key in data.files}
    output,report=build_seed(arrays,json.loads(paths['config'].read_text()),json.loads(paths['selection'].read_text()),
                            transition_frames=args.transition_frames)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**output)
    report['inputs']={key:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key,path in paths.items()}
    report['output_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
