"""Optional exact object-local rigid grips on explicitly declared hold intervals.

Only declared hold/grasp/stable_hold frames may share one pose. Support, slide,
acquisition, release and free motion retain live variables. An optional explicit
phase-gated boundary buffer blends those variables into the shared hold pose.
Neither source poses, source clock nor object motion are edited. Shared poses
are candidates, not collision or contact acceptance. Full-frame wrist bound
checks and collision checks remain the caller's responsibility.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from .hand_sequence import _rotation
from .hand_sequence_rotation import so3_log


def contiguous_intervals(mask):
    mask=np.asarray(mask,dtype=bool)
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return list(zip(edges[::2].tolist(),edges[1::2].tolist()))


def _protected_frames(config, count):
    frames=config.get('protected_frames',[])
    if (not isinstance(frames,list)
            or any(type(frame) is not int or not 0 <= frame < count for frame in frames)
            or frames != sorted(set(frames))):
        raise ValueError('protected_frames must be sorted unique original-frame integer indices')
    return frames


def resolve_hold_intervals(mask, *, side, roles=None, config=None):
    """Resolve strict source holds without merging explicitly distinct grasps.

    Optional ``intervals`` maps each hand to ordered half-open records with
    ``start``, ``end_exclusive`` and a nonempty ``grasp_id``. Records must cover
    the strict hold mask exactly, except explicitly protected frames. Protection
    excludes sharing; the caller still freezes those base variables separately.
    With no interval declaration, the historical contiguous-mask rule remains.
    """
    config={} if config is None else config
    mask=np.asarray(mask)
    if mask.ndim != 1 or mask.dtype.kind != 'b':
        raise ValueError('Shared holds require a one-dimensional Boolean mask')
    count=len(mask);protected=_protected_frames(config,count)
    expected=mask.copy();expected[protected]=False
    if roles is not None and len(roles)!=count:
        raise ValueError('Shared hold phase labels must match every original frame')
    declaration=config.get('intervals')
    if 'intervals' not in config:
        return [dict(start=first,end_exclusive=end,grasp_id=None)
                for first,end in contiguous_intervals(expected)]
    if (not isinstance(declaration,dict) or set(declaration)-{'left','right'}
            or side not in declaration or not isinstance(declaration[side],list)):
        raise ValueError('Explicit shared hold intervals must map each bound hand to a list')
    if roles is None:
        raise ValueError('Explicit shared hold intervals require original phase labels')
    covered=np.zeros(count,dtype=bool);records=[];previous_end=0
    for row in declaration[side]:
        if not isinstance(row,dict) or set(row)!={'start','end_exclusive','grasp_id'}:
            raise ValueError('Shared hold interval requires start, end_exclusive and grasp_id only')
        first,end,grasp_id=row['start'],row['end_exclusive'],row['grasp_id']
        if (type(first) is not int or type(end) is not int
                or not 0 <= first < end <= count):
            raise ValueError('Shared hold interval must use valid half-open original-frame integer bounds')
        if first < previous_end:
            raise ValueError('Shared hold intervals must be ordered and nonoverlapping')
        if not isinstance(grasp_id,str) or not grasp_id.strip():
            raise ValueError('Shared hold grasp_id must be a nonempty string')
        if not expected[first:end].all():
            raise ValueError('Shared hold interval crosses non-hold or protected frames')
        if any(role not in ('hold','grasp','stable_hold') for role in roles[first:end]):
            raise ValueError('Shared holds must contain only strict hold roles, never support or slide')
        covered[first:end]=True;previous_end=end
        records.append(dict(start=first,end_exclusive=end,grasp_id=grasp_id))
    if not np.array_equal(covered,expected):
        raise ValueError('Explicit intervals must cover all strict hold frames except protected_frames')
    return records


def _phase_clipped_blends(intervals, mask, roles, frames, allowed_roles, protected):
    """Partition exterior buffers by nearest hold; equal distance goes earlier.

    Each side stops at the first forbidden frame. Effective windows use the
    same quintic construction over their resulting duration; shortening is a
    deterministic policy consequence, not a new per-case numerical setting.
    """
    eligible={};count=len(mask);protected=set(protected)
    for index,row in enumerate(intervals):
        first,end=row['start'],row['end_exclusive']
        for direction,edge in ((-1,first),(1,end-1)):
            for distance in range(1,frames+1):
                frame=edge+direction*distance
                if (not 0 <= frame < count or frame in protected or mask[frame]
                        or roles[frame] not in allowed_roles):
                    break
                winner=(distance,index)
                if frame not in eligible or winner < eligible[frame]:
                    eligible[frame]=winner
    records=[]
    for index,row in enumerate(intervals):
        first,end=row['start'],row['end_exclusive']
        ids=np.asarray(sorted(frame for frame,(_,owner) in eligible.items() if owner==index),dtype=int)
        before=int(np.count_nonzero(ids<first));after=int(np.count_nonzero(ids>=end))
        left_anchor=first-before-1;right_anchor=end+after
        u=np.where(ids<first,(ids-left_anchor)/(before+1),(right_anchor-ids)/(after+1))
        records.append(dict(frames=ids.tolist(),weights=_quintic_blend_weight(u).tolist(),
            nominal_outer_anchor_frames=[first-frames-1,end+frames],
            effective_outer_anchor_frames=[left_anchor,right_anchor],
            requested_frames_per_side=frames,effective_frames_per_side=dict(before=before,after=after),
            clipped_by_sequence_start=first-frames-1<0,clipped_by_sequence_end=end+frames>=count,
            clipped=before<frames or after<frames,phase_labels=[roles[i] for i in ids],
            policy='phase_clipped',partition_rule='nearest strict hold edge; ties go to earlier interval',
            protected_frames=sorted(protected)))
    return records


def _quintic_blend_weight(t):
    """0/1 endpoint weights with zero first and second endpoint derivatives."""
    return t*t*t*(10+t*(-15+6*t))


def _boundary_blend_options(config):
    frames=config.get('boundary_blend_frames',0)
    if type(frames) is not int or not 0 <= frames <= 120:
        raise ValueError('boundary_blend_frames must be an integer in [0,120]')
    roles=config.get('allowed_blend_roles',[])
    if (not isinstance(roles,list) or any(not isinstance(role,str) or not role for role in roles)
            or len(set(roles))!=len(roles)):
        raise ValueError('allowed_blend_roles must be a list of distinct explicit phase labels')
    if frames and not roles:
        raise ValueError('Positive boundary_blend_frames requires explicit allowed_blend_roles')
    if set(roles)&{'hold','grasp','stable_hold'}:
        raise ValueError('allowed_blend_roles must exclude strict hold roles')
    if config.get('boundary_blend_policy','strict') not in ('strict','phase_clipped'):
        raise ValueError('boundary_blend_policy must be strict or phase_clipped')
    return frames,roles


class SharedHoldParameters:
    """One absolute object-local wrist/finger pose per resolved strict hold.

    With ``boundary_blend_frames=k>0``, the k exterior samples before and after
    each hold blend the *live* base pose with the *live* shared pose. The weight
    has endpoints at start-k-1/start and end-1/end+k on the unchanged source
    clock (end is exclusive). Clipped sequence ends simply have fewer samples.
    By default, buffers crossing unapproved roles or another buffer are
    rejected. The opt-in ``phase_clipped`` policy shortens buffers at protected
    frames, disallowed roles and neighboring holds, and partitions competing
    exterior samples deterministically. Contact masks are never extended.

    Position and joints use convex interpolation; rotation follows the shortest
    SO(3) geodesic. The continuous construction is C2 at its joins for a C2 base
    trajectory away from the SO(3) pi cut. It does not make independently sampled
    base poses or a downstream piecewise-linear playback C2. No output is
    clamped: callers must still reject infeasible source-relative wrist boxes.
    """
    def __init__(self,warm,hands,*,config=None,device='cpu',dtype=None):
        import torch
        config=config or {};dtype=dtype or torch.float32
        policy=config.get('warm_policy','require_constant')
        if policy not in ('require_constant','representative_projection'):
            raise ValueError('Shared-hold warm_policy must require_constant or representative_projection')
        tolerance=config.get('constant_tolerance',{})
        pos_tol=float(tolerance.get('position_m',1e-5))
        rot_tol=float(tolerance.get('rotation_rad',1e-5))
        joint_tol=float(tolerance.get('finger_rad',1e-5))
        if not all(np.isfinite(v) and 0 <= v <= .001 for v in (pos_tol,rot_tol,joint_tol)):
            raise ValueError('Shared-hold constant tolerances must be finite in [0,.001]')
        blend_frames,blend_roles=_boundary_blend_options(config)
        blend_policy=config.get('boundary_blend_policy','strict')
        if isinstance(config.get('intervals'),dict) and set(config['intervals'])-set(hands):
            raise ValueError('Explicit shared hold intervals name an unbound hand')
        self.boundary_blends=[]
        self.hands=hands;self.records=[];self.references=[];control=[]
        for hand_index,(side,hand) in enumerate(hands.items()):
            mask=hand['hold'].detach().cpu().numpy()
            roles=hand.get('roles')
            intervals=resolve_hold_intervals(mask,side=side,roles=roles,config=config)
            protected=_protected_frames(config,len(mask))
            previous_blend_end=None
            if blend_frames:
                if roles is None or len(roles)!=len(mask):
                    raise ValueError('Boundary blending requires original per-frame phase labels')
                if any(mask[i] and role not in ('hold','grasp','stable_hold') for i,role in enumerate(roles)):
                    raise ValueError('Shared holds must contain only strict hold roles, never support or slide')
            clipped_blends=(_phase_clipped_blends(intervals,mask,roles,blend_frames,blend_roles,protected)
                            if blend_frames and blend_policy=='phase_clipped' else None)
            for interval_index,interval in enumerate(intervals):
                first,end=interval['start'],interval['end_exclusive']
                blend=None
                if clipped_blends is not None:
                    blend=clipped_blends[interval_index]
                    self.boundary_blends.append((torch.tensor(blend['frames'],device=device,dtype=torch.long),
                                                 torch.tensor(blend['weights'],device=device,dtype=dtype)))
                elif blend_frames:
                    # Include continuous support endpoints when checking overlap:
                    # adjacent sampled buffers alone can hide a crossing between frames.
                    left_anchor=first-blend_frames-1;right_anchor=end+blend_frames
                    if previous_blend_end is not None and left_anchor < previous_blend_end:
                        raise ValueError(f'{side} shared-hold boundary blend windows overlap')
                    previous_blend_end=right_anchor
                    ids=np.r_[np.arange(max(0,first-blend_frames),first),
                              np.arange(end,min(len(mask),end+blend_frames))]
                    if set(ids.tolist())&set(protected):
                        raise ValueError(f'{side} boundary buffer crosses protected frames')
                    if any(roles[i] not in blend_roles for i in ids):
                        invalid=[(int(i),roles[i]) for i in ids if roles[i] not in blend_roles]
                        raise ValueError(f'{side} boundary buffer crosses disallowed phase labels: {invalid}')
                    u=np.where(ids<first,(ids-left_anchor)/(blend_frames+1),
                               (right_anchor-ids)/(blend_frames+1))
                    weights=_quintic_blend_weight(u)
                    blend=dict(frames=ids.tolist(),weights=weights.tolist(),
                               nominal_outer_anchor_frames=[left_anchor,right_anchor],
                               clipped_by_sequence_start=left_anchor<0,
                               clipped_by_sequence_end=right_anchor>=len(mask),
                               phase_labels=[roles[i] for i in ids])
                    self.boundary_blends.append((torch.tensor(ids,device=device,dtype=torch.long),
                                                 torch.tensor(weights,device=device,dtype=dtype)))
                frame=int(config.get('representative_frames',{}).get(side,{}).get(f'{first}:{end}',(first+end-1)//2))
                if not first <= frame < end:
                    raise ValueError('Shared-hold representative frame must belong to its hold interval')
                p=np.asarray(warm[f'{side}_wrist_position_object'],float)
                r=np.asarray(warm[f'{side}_wrist_rotation_object'],float)
                q=np.asarray(warm[f'{side}_finger_qpos'],float)
                pd=np.linalg.norm(p[first:end]-p[frame],axis=1)
                rd=Rotation.from_matrix(r[first:end]@r[frame].T).magnitude()
                qd=np.abs(q[first:end]-q[frame])
                if not all(np.isfinite(x).all() for x in (p,r,q)):
                    raise ValueError('Nonfinite shared-hold warm pose')
                constant=pd.max()<=pos_tol and rd.max()<=rot_tol and qd.max()<=joint_tol
                if not constant and policy=='require_constant':
                    raise ValueError(f'{side} hold {first}:{end} warm pose is not constant; explicit projection required')
                low=hand['low'].detach().cpu().numpy();high=hand['high'].detach().cpu().numpy()
                if np.any(q[frame]<low-1e-6) or np.any(q[frame]>high+1e-6):
                    raise ValueError('Shared-hold seed violates native finger limits')
                fraction=np.clip((q[frame]-low)/(high-low),1e-6,1-1e-6)
                logits=np.log(fraction/(1-fraction))
                control.append(np.r_[p[frame],np.zeros(3),logits])
                self.references.append(torch.tensor(r[frame],device=device,dtype=dtype))
                self.records.append(dict(side=side,hand_index=hand_index,start=first,end_exclusive=end,
                    representative_frame=frame,original_warm_was_constant=bool(constant),
                    explicit_projection=policy=='representative_projection',
                    warm_deviation_from_shared_pose=dict(position_mean_m=float(pd.mean()),position_max_m=float(pd.max()),
                        angle_mean_rad=float(rd.mean()),angle_max_rad=float(rd.max()),
                        finger_mean_abs_rad=float(qd.mean()),finger_max_abs_rad=float(qd.max())),
                    fingerprint=dict(position_object=p[frame].tolist(),rotation_object=r[frame].tolist(),finger_qpos=q[frame].tolist()),
                    native_limit_sigmoid_initialization_error_rad=float(np.max(np.abs(low+(high-low)*fraction-q[frame])))))
                if interval['grasp_id'] is not None:
                    self.records[-1]['grasp_id']=interval['grasp_id']
                if blend is not None:
                    self.records[-1]['boundary_blend']=blend
        controls=np.stack(control) if control else np.empty((0,13))
        self.controls=torch.nn.Parameter(torch.tensor(controls,device=device,dtype=dtype))
        self.metadata=dict(kind='shared_absolute_object_local_strict_holds',warm_policy=policy,
            intervals=self.records,trainable_shared_parameter_count=int(self.controls.numel()),
            constant_tolerance=dict(position_m=pos_tol,rotation_rad=rot_tol,finger_rad=joint_tol),
            phase_roles_allowed=['hold','grasp','stable_hold'],support_or_slide_frozen=False,
            source_prior='Original raw SOMA at every matching source time.',
            wrist_bounds='Whole-checkpoint feasibility across every source frame; shared pose is never pointwise clamped.',
            joins='No posthoc smoothing; original all-frame derivatives and interpolated-frame collision constraints retained.')
        if blend_frames:
            self.metadata['boundary_blend']=dict(frames=blend_frames,allowed_phase_roles=blend_roles,
                weight='6u^5-15u^4+10u^3; k exterior frames, endpoint duration (k+1) source timesteps',
                interpolation='Live object-local base and live shared pose; convex position/joints and shortest SO(3) geodesic',
                continuity='C2 continuous joins only when the live base is C2 and the relative rotation avoids pi',
                overlap_policy='Reject overlapping continuous windows; a shared zero-weight endpoint is allowed',
                native_finger_limits='Convex blend preserves original native limits when both inputs satisfy them',
                source_clock_unchanged=True,contact_mask_unchanged=True,strict_hold_mask_unchanged=True,
                source_wrist_bound_policy_unchanged=True)
            if blend_policy=='phase_clipped':
                self.metadata['boundary_blend'].update(policy=blend_policy,
                    weight='6u^5-15u^4+10u^3 over each effective exterior buffer',
                    overlap_policy='Nearest hold edge owns each frame; ties go to earlier interval',
                    continuity='Shortened discrete buffers are not a C2 trajectory guarantee; all-frame temporal and collision checks remain required.',
                    protected_frames=config.get('protected_frames',[]),
                    effective_lengths=[dict(side=row['side'],start=row['start'],end_exclusive=row['end_exclusive'],
                        **row['boundary_blend']['effective_frames_per_side']) for row in self.records])
        if config.get('intervals') is not None or config.get('protected_frames'):
            self.metadata.update(interval_source='explicit' if config.get('intervals') is not None else 'contiguous_mask',
                                 protected_frames=config.get('protected_frames',[]),
                                 protected_frame_policy='Excluded from sharing and boundary blending; caller freezes live base variables separately.')

    def decode(self,base):
        import torch
        result=base
        for index,record in enumerate(self.records):
            hand=self.hands[record['side']];value=self.controls[index]
            position=value[:3]
            rotation=_rotation(value[None,3:6])[0]@self.references[index]
            fingers=hand['low']+(hand['high']-hand['low'])*value[6:].sigmoid()
            relative_rotation=so3_log(rotation[None]@hand['source_rotation'].transpose(-1,-2))
            encoded=torch.cat((position[None]-hand['source_position'],relative_rotation,
                               fingers[None]-hand['source_finger']),dim=1)
            mask=torch.zeros(base.shape[:2],dtype=torch.bool,device=base.device)
            mask[record['start']:record['end_exclusive'],record['hand_index']]=True
            result=torch.where(mask[:,:,None],encoded[:,None],result)
            if self.boundary_blends:
                ids,weights=self.boundary_blends[index]
                if not len(ids):
                    continue
                hand_index=record['hand_index'];weight=weights[:,None]
                live=base[ids,hand_index]
                base_position=hand['source_position'][ids]+live[:,:3]
                base_rotation=_rotation(live[:,3:6])@hand['source_rotation'][ids]
                base_fingers=hand['source_finger'][ids]+live[:,6:]
                blended_position=(1-weight)*base_position+weight*position[None]
                delta=so3_log(rotation[None]@base_rotation.transpose(-1,-2))
                blended_rotation=_rotation(weight*delta)@base_rotation
                blended_fingers=(1-weight)*base_fingers+weight*fingers[None]
                blended=torch.cat((blended_position-hand['source_position'][ids],
                    so3_log(blended_rotation@hand['source_rotation'][ids].transpose(-1,-2)),
                    blended_fingers-hand['source_finger'][ids]),dim=1)
                result=result.index_put((ids,torch.full_like(ids,hand_index)),blended)
        return result
