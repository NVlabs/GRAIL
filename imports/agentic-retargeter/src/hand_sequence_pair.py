"""Opt-in simultaneous two-hand refinement on every original source frame.

Both free wrists and native finger sets are live in the SAME autograd graph.
Original hand meshes remain in their body frames; bidirectional cross-hand
queries transform points through both current poses. No stale collision partner,
source/body/object edit, posthoc averaging, or mounted-robot claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from .grasp_io import GraspInputs, number
from .hand_sequence import (_samples, _rotation, _difference, masked_mean,
                            DEFAULT_TEMPORAL, _write, _sha, resolve_static_object_policy)
from .hand_sequence_rotation import rotation_velocity, so3_log, rotation_quaternion
from .hand_sequence_seed import normalized_roles


def validate_source_clock(native, warm, fps, count):
    expected=np.arange(count,dtype=np.float64)/fps
    if 'fps' in native and not np.isclose(float(np.asarray(native['fps'])),fps,atol=1e-12,rtol=0):
        raise ValueError('Configured fps differs from original native timing')
    if not np.array_equal(warm['timestamps'],expected):
        raise ValueError('Warm trajectory must retain exact original uniform clock')
    if 'timestamps' in native and not np.array_equal(native['timestamps'],warm['timestamps']):
        raise ValueError('Original timestamps differ; no quiet retiming is supported')


def contact_policy(config, side, count):
    active=np.asarray(config['contact_mask'][side])
    if active.shape!=(count,) or active.dtype.kind!='b':
        raise ValueError('Contact mask must contain original T booleans')
    active=active.copy()
    if config.get('is_static_obj',False):
        active[:]=False
    roles=normalized_roles(config,side,count)
    hold=active&np.asarray([r in ('hold','grasp','stable_hold') for r in roles])
    return active,hold,roles


def anchor_rejection(item,hold):
    first,end=item['start'],item['end']+1
    if not 0 <= first < end <= len(hold) or not hold[first:end].all():
        return 'not_explicit_hold'
    gap=item.get('initial_gap_m',np.inf)
    if not np.isfinite(gap) or gap < 0:
        return 'invalid_initial_gap'
    if gap > .005:
        return 'initial_gap_over_5mm'
    return None


def validate_warm_metadata(warm,native,hands):
    if not np.array_equal(warm.get('object_scale'),native['object_scale']):
        raise ValueError('Warm object_scale does not exactly bind original native scale')
    for side,hand in hands.items():
        if not np.array_equal(warm.get(f'{side}_finger_addresses'),hand['addresses']):
            raise ValueError('Warm native finger addresses mismatch: '+side)


def load_bound_warm(path):
    """Load and hash the same immutable NPZ bytes, not a later file revision."""
    payload=Path(path).read_bytes()
    with np.load(io.BytesIO(payload),allow_pickle=False) as data:
        arrays={key:data[key].copy() for key in data.files}
    return arrays,hashlib.sha256(payload).hexdigest()


def validate_refinement_reference_prior(option):
    """Optional warm-reference trust weights; the raw-source prior is separate."""
    if option is None:
        return None
    if not isinstance(option,dict) or set(option)-{'position','rotation','finger'}:
        raise ValueError('refinement_reference_prior supports position, rotation and finger only')
    weights={key:number(option.get(key,0.),'refinement_reference_prior.'+key)
             for key in ('position','rotation','finger')}
    return weights if any(weights.values()) else None


def refinement_reference_residuals(position,rotation,fingers,reference):
    """Direct pose/joint residuals at matching source times; reference is fixed.

    SO(3) uses the principal logarithm. Its squared norm is coordinate invariant,
    but remains nondifferentiable at exactly pi; the deterministic finite branch
    is not a claim of smoothness across that cut locus.
    """
    return dict(position=position-reference['position'].detach(),
                rotation=so3_log(rotation@reference['rotation'].detach().transpose(-1,-2)),
                finger=fingers-reference['finger'].detach())


def refinement_reference_terms(position,rotation,fingers,reference,weights):
    if weights is None:
        return position.sum()*0,{}
    residuals=refinement_reference_residuals(position,rotation,fingers,reference)
    terms={key:value.square().sum(-1).mean() for key,value in residuals.items()}
    return sum(weights[key]*terms[key] for key in terms),terms


def validate_refinement_reference_arrays(warm,side,count):
    """Check the actual warm states without projecting or changing the reference."""
    for key,shape in [('wrist_position_object',(count,3)),
                      ('wrist_rotation_object',(count,3,3)),('finger_qpos',(count,7))]:
        value=warm[f'{side}_{key}']
        if value.shape!=shape or not np.isfinite(value).all():
            raise ValueError('Invalid refinement reference '+side+' '+key)
    rotation=warm[f'{side}_wrist_rotation_object']
    if (not np.allclose(rotation@rotation.transpose(0,2,1),np.eye(3),atol=2e-5,rtol=0)
            or not np.allclose(np.linalg.det(rotation),1,atol=2e-5,rtol=0)):
        raise ValueError('Refinement reference must contain proper SO(3) wrist rotations')


def validate_refinement_editable_frames(option,hands,count,*,parameterization=None,shared_hold=None):
    """Explicit original-frame edit sets; unspecified hands are never inferred."""
    if option is None:
        return None
    if parameterization:
        raise ValueError('refinement_editable_frames requires per-frame parameters without spline')
    if not isinstance(option,dict) or set(option)!=set(hands):
        raise ValueError('refinement_editable_frames must map every editable hand and no other hands')
    masks={}
    for side,frames in option.items():
        if (not isinstance(frames,list) or any(type(frame) is not int or not 0<=frame<count for frame in frames)
                or frames!=sorted(set(frames))):
            raise ValueError('Editable frames must be sorted unique original-frame integer indices: '+side)
        masks[side]=np.zeros(count,dtype=bool);masks[side][frames]=True
    if not any(mask.any() for mask in masks.values()):
        raise ValueError('refinement_editable_frames cannot freeze all hands at all frames')
    return masks


def validate_shared_hold_editable_frames(masks, hands, config=None):
    """A frozen-frame overlay must never split one declared rigid hold.

    Whole holds may remain frozen or editable. Exterior frozen frames (including
    frame zero) still enter all temporal and collision terms through the existing
    decoder mask. This validates scope; it does not alter phase labels or masks.
    """
    if masks is None:
        return []
    from .hand_sequence_shared_hold import resolve_hold_intervals
    records = []
    for side, hand in hands.items():
        hold = hand['hold'].detach().cpu().numpy()
        for interval in resolve_hold_intervals(hold,side=side,roles=hand.get('roles'),config=config):
            first,end=interval['start'],interval['end_exclusive']
            editable = masks[side][first:end]
            if editable.any() and not editable.all():
                raise ValueError(f'Editable frame mask splits shared hold {side} {first}:{end}')
            records.append(dict(side=side, start=first, end_exclusive=end,
                                editable=bool(editable.all())))
    return records


def apply_refinement_frame_mask(candidate,warm,mask):
    """Select frozen warm values inside the graph, with zero frozen gradients."""
    import torch
    if mask is None:
        return candidate
    if mask.dtype!=torch.bool or candidate.shape!=warm.shape or candidate.shape[:mask.ndim]!=mask.shape:
        raise ValueError('Refinement frame mask must bind matching leading state dimensions')
    return torch.where(mask.reshape(*mask.shape,*([1]*(candidate.ndim-mask.ndim))),candidate,warm.detach())


def frozen_refinement_wrist_bounds(position,rotation,source_position,source_rotation,
                                   editable,position_bound,rotation_bound):
    """Measure inherited frozen poses, avoiding float32 latent re-encoding.

    These values describe the unchanged warm reference, not live controls.
    Preserve even tiny inherited violations rather than broadening the live
    tolerance or claiming every original pose lies inside the source corridor.
    """
    from scipy.spatial.transform import Rotation
    p,r,p0,r0=(np.asarray(value,dtype=np.float64) for value in
                (position,rotation,source_position,source_rotation))
    mask=np.asarray(editable)
    if (p.shape!=p0.shape or p.ndim!=2 or p.shape[1:]!=(3,)
            or r.shape!=r0.shape or r.shape!=(len(p),3,3)
            or mask.shape!=(len(p),) or mask.dtype.kind!='b'
            or not all(np.isfinite(v).all() for v in (p,r,p0,r0))):
        raise ValueError('Frozen wrist bound check requires matching finite original poses and edit mask')
    ids=np.flatnonzero(~mask)
    if len(ids):
        delta=p[ids]-p0[ids]
        rv=Rotation.from_matrix(r[ids]@r0[ids].transpose(0,2,1)).as_rotvec()
        pe=np.maximum(np.abs(delta)-position_bound,0.)
        re=np.maximum(np.abs(rv)-rotation_bound,0.)
        pmax=float(pe.max());rmax=float(re.max())
        violations=ids[(pe.max(1)>1e-6)|(re.max(1)>1e-6)].tolist()
    else:
        pmax=rmax=0.;violations=[]
    return dict(frame_count=len(ids),position_box_excess_m=pmax,
        rotation_vector_box_excess_rad=rmax,within_original_tolerance=not violations,
        violating_original_frames=violations,tolerance=1e-6,
        precision='float64 differences and principal SO(3) log of actual warm matrices',
        reference='same effective source wrist poses as the live optimizer',
        policy='Inherited frozen reference evidence; exact warm pose overlay remains unchanged.')


def export_refinement_frame_values(value,warm,mask):
    """Verify computed frozen values, then preserve their original NPZ precision."""
    import torch
    actual=value.detach().cpu().numpy()
    if mask is None:
        return actual.copy(),None
    frozen=~mask
    expected=torch.as_tensor(warm,device=value.device,dtype=value.dtype)
    selected=torch.as_tensor(frozen,device=value.device)
    if not torch.equal(value[selected],expected[selected]):
        raise ValueError('Frozen refinement frame changed inside objective evaluation')
    result=actual.astype(warm.dtype,copy=True)
    result[frozen]=warm[frozen]
    exact=result[frozen].tobytes()==warm[frozen].tobytes()
    if not exact:
        raise AssertionError('Frozen warm export must remain byte-identical')
    return result,dict(frozen_frame_count=int(frozen.sum()),evaluation_exact_at_compute_dtype=True,
        export_bytes_exact=True,export_dtype=str(warm.dtype),compute_dtype=str(value.dtype))


def update_export_contact_policy(arrays,side,hand):
    # Export the effective policy, never stale warm-start contact metadata.
    arrays[f'{side}_contact_active']=hand['active'].detach().cpu().numpy().copy()
    arrays[f'{side}_phase_labels']=np.asarray(hand['roles'])
    arrays[f'{side}_fixed_anchor_eligible']=hand['hold'].detach().cpu().numpy().copy()


def relative_points(points_object, target_rotation, target_position):
    """Live points in a live target's original body frame; no detach on either."""
    import torch
    return torch.einsum('bji,bnj->bni',target_rotation,points_object-target_position[:,None])


def validate_spaced_contact_selection(option):
    if option is None or option=={}:
        return None
    if not isinstance(option,dict) or set(option)-{'minimum_hand_spacing_m','points_per_digit'}:
        raise ValueError('spaced_contact_selection supports minimum_hand_spacing_m and points_per_digit only')
    count=option.get('points_per_digit',4)
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError('points_per_digit must be an integer 1..16')
    spacing=number(option.get('minimum_hand_spacing_m',.003),'minimum_hand_spacing_m',0,.05)
    return dict(minimum_hand_spacing_m=spacing,points_per_digit=count)


def select_contact_loss(distance,hand_points,active,margin,option=None):
    """Only original pad-member candidates enter attraction selection."""
    if option is None:
        selected=distance.topk(min(4,distance.shape[1]),largest=False,dim=1).values
        return (selected-margin).square().mean(1),selected,None
    from .hand_sequence_spaced_contact import select_spaced_contact
    selection=select_spaced_contact(distance,hand_points,count=option['points_per_digit'],
        min_spacing_m=option['minimum_hand_spacing_m'],contact_mask=active)
    selected=selection['distances'];valid=selection['valid']
    # Missing slots never dilute the mean or count as satisfied contact.
    loss=((selected-margin).square()*valid).sum(1)/selection['selected_count'].clamp_min(1)
    return loss,selected,selection


def phase_stencil_mask(hold_mask,order):
    """Select object coordinates only when every source pose belongs to the mask."""
    if hold_mask.ndim!=1 or type(order) is not int or not 1 <= order < len(hold_mask):
        raise ValueError('Derivative stencil requires one-dimensional mask and order 1..T-1')
    return hold_mask.unfold(0,order+1,1).all(-1)


def relative_temporal_mask(config, active, roles):
    """Choose derivative coordinates without changing contact or no-slip roles.

    Support and slide may use object-relative acceleration/jerk, but never gain
    zero relative velocity or material anchors from this option.
    """
    selected=config.get('object_relative_temporal_roles',['hold','grasp','stable_hold'])
    allowed={'hold','grasp','stable_hold','support','slide','sliding'}
    if (not isinstance(selected,list) or not selected
            or any(not isinstance(role,str) or role not in allowed for role in selected)
            or len(set(selected))!=len(selected)):
        raise ValueError('object_relative_temporal_roles must be distinct explicit hold/support/slide roles')
    active=np.asarray(active)
    if active.dtype.kind!='b' or active.shape!=(len(roles),):
        raise ValueError('Relative temporal roles require the original Boolean contact mask')
    return active & np.asarray([role in selected for role in roles]),list(selected)


def temporal_terms(position,rotation,fingers,object_position,object_rotation,
                   hold_mask,dt,weights,*,phase_coordinates=False,relative_mask=None):
    """SI terms with one coordinate choice per complete original pose stencil.

    In phase mode position/rotation are WORLD hand poses. A derivative uses
    object-local hand poses iff its entire source-pose stencil is selected by
    relative_mask (strict hold by default). Free and incomplete stencils use world coordinates. Angular
    acceleration/jerk depend on 3/4 pose frames, respectively, just like linear
    acceleration/jerk. Every derivative sample enters once. Finger derivatives
    and the separately declared relative hold-velocity guidance are unchanged.
    """
    import torch
    relative_mask=hold_mask if relative_mask is None else relative_mask
    if relative_mask.shape!=hold_mask.shape or relative_mask.dtype!=torch.bool:
        raise ValueError('Relative derivative mask must match original source frames')
    velocity=rotation_velocity(rotation,dt)
    values=[('wrist_acceleration',_difference(position,dt,2),2),
            ('wrist_jerk',_difference(position,dt,3),3),
            ('angular_acceleration',_difference(velocity,dt),2),
            ('angular_jerk',_difference(velocity,dt,2),3)]
    if phase_coordinates:
        object_velocity=rotation_velocity(object_rotation,dt)
        object_values=[_difference(object_position,dt,2),_difference(object_position,dt,3),
                       _difference(object_velocity,dt),_difference(object_velocity,dt,2)]
        values=[(key,torch.where(phase_stencil_mask(relative_mask,order)[:,None],relative,world),order)
                for (key,world,order),relative in zip(values,object_values)]
    values.extend([('finger_acceleration',_difference(fingers,dt,2),2),
                   ('finger_jerk',_difference(fingers,dt,3),3)])
    terms={};loss=position.sum()*0
    for key,value,_ in values:
        term=value.square().mean()
        terms[key]=term;loss=loss+weights.get(key,0)*term
    pairs=hold_mask[1:]&hold_mask[:-1]
    for key,value in [('hold_translation_velocity',_difference(object_position,dt)),
                      ('hold_angular_velocity',rotation_velocity(object_rotation,dt)),
                      ('hold_finger_velocity',_difference(fingers,dt))]:
        term=masked_mean(value.square().mean(-1),pairs)
        terms[key]=term;loss=loss+weights.get(key,0)*term
    return loss,terms


def solve(config,directory='.',output_directory='paired_hand_output'):
    import torch
    from scipy.spatial.transform import Rotation
    from .repair_tensor import NativeTensorFK,WarpSignedDistance
    from .repair_geometry import TriangleMesh
    from .hand_sequence_collision_samples import append_collision_witnesses

    start=time.monotonic();base=Path(directory);out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    ctx=GraspInputs(config['inputs'],base)
    input_config=config
    static_policy=resolve_static_object_policy(config,ctx.R,ctx.t)
    config={**config,'is_static_obj':static_policy['is_static_obj']}
    if not set(ctx.hands) or not set(ctx.hands).issubset({'left','right'}):
        raise ValueError('Refinement requires one or both explicitly bound Dex3 hands')
    warm_path=(base/config['warm_start_npz']).resolve(strict=True)
    warm,warm_sha256=load_bound_warm(warm_path)
    refinement_prior=validate_refinement_reference_prior(config.get('refinement_reference_prior'))
    refinement_metadata=dict(enabled=refinement_prior is not None,weights=refinement_prior,
        reference=dict(path=str(warm_path),sha256=warm_sha256),raw_source_prior_unchanged=True,
        definition='Per-hand mean squared position norm (m²), principal SO(3) angle (rad²), and native finger-vector norm (rad²), at every original timestamp.',
        coordinate_frame='Object-local matching-time states; norm costs are invariant to common rigid placement.',
        rotation_cut_locus='Finite deterministic principal-log branch; squared angle is not differentiable exactly at pi.',
        correction_derivative_regularization=False,soft_prior_is_not_motion_acceptance=True)
    for key,expected in [('source_qpos',ctx.q),('object_R',ctx.R),('object_trans',ctx.t)]:
        if not np.array_equal(warm[key],expected):
            raise ValueError('Warm trajectory source/body/object mismatch: '+key)
    validate_warm_metadata(warm,ctx.native,ctx.hands)
    count=len(ctx.q);fps=number(config.get('fps',24),'fps',1e-6);dt=1/fps
    validate_source_clock(ctx.native,warm,fps,count)
    finger_speed=None
    if config.get('finger_speed') is not None:
        from .hand_sequence_finger_speed import finger_speed_policy, finger_speed_terms, finger_speed_measure
        finger_speed=finger_speed_policy(config['finger_speed'])
    motion_speed=None
    if config.get('motion_speed') is not None:
        from .hand_sequence_motion_speed import motion_speed_policy, motion_speed_terms, motion_speed_measure
        motion_speed=motion_speed_policy(config['motion_speed'],ctx.hands,count)
    phase_fidelity={}
    if config.get('phase_wrist_fidelity') is not None:
        from .hand_sequence_phase_fidelity import phase_fidelity_policy, phase_fidelity_terms
        policies={side:(contact_policy(config,side,count)[0],contact_policy(config,side,count)[2])
                  for side in ctx.hands}
        phase_fidelity=phase_fidelity_policy(config['phase_wrist_fidelity'],policies,
                                            is_static_obj=config.get('is_static_obj',False))
    editable_frames=validate_refinement_editable_frames(config.get('refinement_editable_frames'),ctx.hands,count,
        parameterization=config.get('trajectory_parameterization'),shared_hold=config.get('strict_hold_parameterization'))
    editable_metadata=dict(enabled=editable_frames is not None,
        frames=None if editable_frames is None else {side:np.flatnonzero(mask).tolist() for side,mask in editable_frames.items()},
        reference=dict(path=str(warm_path),sha256=warm_sha256),
        rule='Outside declared original frames, exact warm wrist/finger states enter FK and every objective. All frames, temporal joins and collision collocation remain evaluated.',
        raw_source_prior_and_bounds_unchanged=True)
    if any(f'{side}_wrist_position_object' in warm for side in {'left','right'}-set(ctx.hands)):
        raise ValueError('Warm trajectory includes an unbound editable hand; frozen opposite hand must use native source')
    steps=config.get('paired_steps',200)
    if type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError('paired_steps must be an integer 1..2000')
    device=config.get('device','cuda:0')
    execution_backend=config.get('execution_backend','eager')
    if execution_backend not in ('eager','cuda_graph'):
        raise ValueError('execution_backend must be eager or cuda_graph')
    rotation_backend=config.get('rotation_backend','rodrigues' if execution_backend=='cuda_graph' else 'matrix_exp')
    if rotation_backend not in ('matrix_exp','rodrigues'):
        raise ValueError('rotation_backend must be matrix_exp or rodrigues')
    if execution_backend=='cuda_graph' and (not device.startswith('cuda:') or rotation_backend!='rodrigues'):
        raise ValueError('cuda_graph requires CUDA and the capture-compatible Rodrigues rotation')
    if execution_backend=='cuda_graph' and (config.get('trajectory_parameterization') or config.get('strict_hold_parameterization')):
        raise ValueError('cuda_graph currently supports per-frame parameters only; spline/shared-hold must use eager execution')
    if rotation_backend=='rodrigues':
        from .hand_sequence_acceleration import stable_rotation as rotation_exp
    else:
        rotation_exp=_rotation
    fk=NativeTensorFK(ctx.model,device=device,dtype=torch.float32)
    obj_query=WarpSignedDistance(ctx.object_query.vertices,ctx.object_query.faces,device=device)
    qbase=fk.t(ctx.q.copy());object_R=fk.t(ctx.R);object_t=fk.t(ctx.t)
    if finger_speed is not None:
        finger_speed_intervals=fk.t(np.diff(warm['timestamps']))
    if motion_speed is not None:
        motion_speed_intervals=fk.t(np.diff(warm['timestamps']))
        motion_speed_indices={side:torch.tensor(indices,dtype=torch.long,device=device)
                              for side,indices in motion_speed['intervals'].items()}
    with torch.no_grad():
        source_R,source_p=fk(qbase)
    samples=int(config.get('samples_per_geom',128))
    if not 32 <= samples <= 4096:
        raise ValueError('samples_per_geom must be in [32,4096]')
    weights={**DEFAULT_TEMPORAL,**config.get('temporal_weights',{})}
    if set(weights)!=set(DEFAULT_TEMPORAL):
        raise ValueError('Unknown temporal weight')
    weights={k:number(v,k) for k,v in weights.items()}
    loss_weights={**dict(object_collision=1.,self_collision=1.,cross_hand=1.,scene_collision=0.,contact=4.,fixed_anchor=4.),
                  **config.get('paired_loss_weights',{})}
    loss_weights={k:number(v,k) for k,v in loss_weights.items()}
    from .hand_scene_clearance import HandSceneClearance
    scene_query=HandSceneClearance(ctx.native,device) if loss_weights['scene_collision'] else None
    scene_enabled=scene_query is not None and scene_query.enabled
    prior={**dict(position=.02,rotation=.0001,finger=.00002),**config.get('source_prior',{})}
    prior={k:number(v,k) for k,v in prior.items()}
    space=config.get('temporal_coordinate_space','object')
    if space not in ('world','object','phase'):
        raise ValueError('temporal_coordinate_space must be world, object, or phase')
    margin=number(config.get('clearance_m',.001),'clearance_m',0,.02)
    cross_margin=number(config.get('cross_hand_clearance_m',.002),'cross_hand_clearance_m',0,.03)
    object_peak_weight=number(config.get('global_worst_object_weight',400.),'global_worst_object_weight')
    cross_peak_weight=number(config.get('global_worst_cross_hand_weight',400.),'global_worst_cross_hand_weight')
    position_bound=number(config.get('translation_limit_m',.15),'translation_limit_m',1e-6,1)
    rotation_bound=number(config.get('rotation_limit_rad',1.5),'rotation_limit_rad',1e-6,np.pi)
    spaced_options=validate_spaced_contact_selection(config.get('spaced_contact_selection'))
    witness_config=config.get('collision_witness_samples',{})
    if not isinstance(witness_config,dict) or set(witness_config)-{'left','right'}:
        raise ValueError('collision_witness_samples must map left/right to original-surface witness rows')
    parameters=[];hands={};anchor_source={}
    if config.get('anchor_report'):
        anchor_path=(base/config['anchor_report']).resolve(strict=True)
        anchor_source=json.loads(anchor_path.read_text())
        if anchor_source['binding']!=ctx.binding:
            raise ValueError('Anchor report must bind exact original inputs')
    for side in ('left','right'):
        if side not in ctx.hands:
            continue
        h=ctx.hands[side];wrist=h['wrist'];address=h['addresses']
        if refinement_prior is not None or editable_frames is not None or side in phase_fidelity:
            validate_refinement_reference_arrays(warm,side,count)
        if list(warm[f'{side}_finger_joint_names'])!=h['finger_joints']:
            raise ValueError('Warm finger-joint identity mismatch')
        wp=(object_R.transpose(-1,-2)@(source_p[:,wrist]-object_t)[...,None]).squeeze(-1)
        wr=object_R.transpose(-1,-2)@source_R[:,wrist]
        initial=np.concatenate((warm[f'{side}_wrist_position_object']-wp.cpu().numpy(),
                Rotation.from_matrix(warm[f'{side}_wrist_rotation_object']@wr.cpu().numpy().transpose(0,2,1)).as_rotvec(),
                warm[f'{side}_finger_qpos']-ctx.q[:,address]),1)
        if initial.shape!=(count,13):
            raise ValueError('Paired Dex3 module expects seven named native fingers per hand')
        parameters.append(initial)
        points,bodies,geoms,members,pad_metadata=_samples(ctx,side,samples)
        points,bodies,geoms,witness_metadata=append_collision_witnesses(points,bodies,geoms,witness_config.get(side,[]),
            meshes=ctx.meshes,geom_bodies=ctx.bodies,allowed_geoms=h['robot_geoms'])
        witness_metadata['query_terms']=['object','self_with_original_adjacency_omissions','cross_hand']
        if any(len(ids) and int(np.max(ids))>=witness_metadata['original_sample_count'] for ids in members.values()):
            raise AssertionError('Collision witness points must never enter original contact-pad memberships')
        declared=config.get('contact_region_faces',{}).get(side,{})
        if declared:
            members={d:i for d,i in members.items() if ('thumb' if d=='thumb' else 'long_fingers') in declared}
        region_queries={};schedules=[]
        for digit in members:
            group='thumb' if digit=='thumb' else 'long_fingers'
            faces=np.asarray(declared.get(group,h['region_faces']))
            mesh=TriangleMesh(ctx.object.vertices,ctx.object.faces[faces])
            region_queries[digit]=WarpSignedDistance(mesh.vertices,mesh.faces,device=device)
        for schedule in config.get('region_schedule',{}).get(side,[]):
            first,end=schedule['frames'];queries={}
            if not 0 <= first < end <= count:
                raise ValueError('Invalid half-open region schedule')
            for digit in members:
                group='thumb' if digit=='thumb' else 'long_fingers'
                faces=np.asarray(schedule.get('contact_region_faces',{}).get(group,schedule['region_faces']))
                mesh=TriangleMesh(ctx.object.vertices,ctx.object.faces[faces])
                queries[digit]=WarpSignedDistance(mesh.vertices,mesh.faces,device=device)
            schedules.append((first,end,queries))
        active_np,hold_np,roles=contact_policy(config,side,count)
        relative_np,relative_roles=relative_temporal_mask(config,active_np,roles)
        mesh_queries=[];self_queries=[];omissions=[]
        for target in h['robot_geoms']:
            target_body=ctx.bodies[target]
            query=WarpSignedDistance(ctx.meshes[target].vertices,ctx.meshes[target].faces,device=device)
            mesh_queries.append((target,target_body,query))
            allowed=[]
            for other in h['robot_geoms']:
                b=ctx.bodies[other]
                if b==target_body or ctx.model.body_parentid[b]==target_body or ctx.model.body_parentid[target_body]==b:
                    omissions.append([int(other),int(target)])
                else:
                    allowed.append(other)
            ids=np.flatnonzero(np.isin(geoms,allowed))
            if len(ids):
                self_queries.append((target_body,fk.t(ids,torch.long),query))
        anchor_rows=[];anchor_rejections=[]
        for item in anchor_source.get('hands',{}).get(side,{}).get('fixed_anchors',[]):
            reason=anchor_rejection(item,hold_np)
            if reason:
                anchor_rejections.append(dict(reason=reason,anchor=item));continue
            geom=int(item['geom']);body=int(item['body'])
            if geom not in h['robot_geoms'] or ctx.bodies[geom]!=body:
                raise ValueError('Anchor hand geometry/body identity mismatch')
            face=int(item['object_face']);bary=np.asarray(item['object_barycentric'])
            if (not 0 <= face < len(ctx.object.faces) or bary.shape!=(3,) or not np.isclose(bary.sum(),1)
                    or np.any(bary < -1e-7)):
                raise ValueError('Invalid original triangle material identity')
            object_point=bary@ctx.object.triangles[face]
            if not np.allclose(object_point,item['object_point'],atol=2e-7,rtol=0):
                raise ValueError('Anchor material target disagrees with original geometry')
            anchor_rows.append(dict(item,object_point=object_point.tolist()))
        hands[side]=dict(hand=h,source_position=wp,source_rotation=wr,source_finger=qbase[:,address].clone(),
                  points=fk.t(points),bodies=fk.t(bodies,torch.long),point_geoms=geoms,members={d:fk.t(i,torch.long) for d,i in members.items()},
                  low=fk.t(ctx.model.jnt_range[h['joints'],0]),high=fk.t(ctx.model.jnt_range[h['joints'],1]),
                  active=torch.tensor(active_np,device=device),hold=torch.tensor(hold_np,device=device),roles=roles,
                  relative_temporal=torch.tensor(relative_np,device=device),relative_temporal_roles=relative_roles,
                  region_queries=region_queries,schedules=schedules,mesh_queries=mesh_queries,self_queries=self_queries,
                  anchors=anchor_rows,anchor_rejections=anchor_rejections,pad_metadata=pad_metadata,omissions=omissions,collision_witness_samples=witness_metadata)
        hands[side]['anchor_tensors']=[(item,fk.t(item['body_local_point']),fk.t(item['object_point']))
                                      for item in anchor_rows]
        hands[side]['address_tensor']=fk.t(address,torch.long)
        if refinement_prior is not None or side in phase_fidelity:
            hands[side]['refinement_reference']={key:fk.t(warm[f'{side}_{name}']).detach().clone()
                for key,name in [('position','wrist_position_object'),('rotation','wrist_rotation_object'),('finger','finger_qpos')]}
        if side in phase_fidelity:
            hands[side]['phase_fidelity_mask']=fk.t(phase_fidelity[side]['mask'],torch.bool)
        if editable_frames is not None:
            hands[side]['editable_frames']=fk.t(editable_frames[side],torch.bool)
            hands[side]['frozen_warm']={key:fk.t(warm[f'{side}_{name}']).detach().clone()
                for key,name in [('position','wrist_position_object'),('rotation','wrist_rotation_object'),('finger','finger_qpos')]}
    # When only one hand is editable, preserve and collide against the other
    # actual native hand, including all its original mesh geoms.
    frozen_hands={}
    import mujoco
    for side in sorted({'left','right'}-set(hands)):
        wrist=int(ctx.model.body(f'{side}_wrist_yaw_link').id)
        points=[];bodies=[];geoms=[];queries=[];omissions=[];frozen_meshes={};frozen_bodies={}
        for geom in range(ctx.model.ngeom):
            body=int(ctx.model.geom_bodyid[geom]);parent=body
            while parent and parent!=wrist:
                parent=int(ctx.model.body_parentid[parent])
            if parent!=wrist:
                continue
            if ctx.model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:
                omissions.append(int(geom));continue
            model=ctx.model;mesh_id=int(model.geom_dataid[geom])
            va,vn=int(model.mesh_vertadr[mesh_id]),int(model.mesh_vertnum[mesh_id])
            fa,fn=int(model.mesh_faceadr[mesh_id]),int(model.mesh_facenum[mesh_id])
            geom_rotation=Rotation.from_quat(model.geom_quat[geom][[1,2,3,0]]).as_matrix()
            mesh=TriangleMesh(model.mesh_vert[va:va+vn]@geom_rotation.T+model.geom_pos[geom],model.mesh_face[fa:fa+fn])
            frozen_meshes[geom]=mesh;frozen_bodies[geom]=body
            nv,nf=min(samples//2,len(mesh.vertices)),min(samples-samples//2,len(mesh.faces))
            pts=np.concatenate((mesh.vertices[np.linspace(0,len(mesh.vertices)-1,nv,dtype=int)],
                mesh.triangles[np.linspace(0,len(mesh.faces)-1,nf,dtype=int)].mean(1)))
            points.extend(pts);bodies.extend([body]*len(pts));geoms.extend([geom]*len(pts))
            queries.append((geom,body,WarpSignedDistance(mesh.vertices,mesh.faces,device=device)))
        if not queries:
            raise ValueError('Missing original frozen opposite-hand meshes')
        points,bodies,geoms,witness_metadata=append_collision_witnesses(np.asarray(points),np.asarray(bodies),np.asarray(geoms),
            witness_config.get(side,[]),meshes=frozen_meshes,geom_bodies=frozen_bodies,allowed_geoms=set(frozen_meshes))
        witness_metadata['query_terms']=['cross_hand_against_editable_hand']
        frozen_hands[side]=dict(wrist=wrist,points=fk.t(points),bodies=fk.t(bodies,torch.long),
                               point_geoms=geoms,mesh_queries=queries,primitive_omissions=omissions,collision_witness_samples=witness_metadata)
    fractions=config.get('collision_collocation_fractions',[])
    if (not isinstance(fractions,list) or len(fractions)>3 or len(set(fractions))!=len(fractions)
            or any(type(v) not in (float,int) or not 0 < v < 1 for v in fractions)):
        raise ValueError('collision_collocation_fractions needs up to three distinct fractions strictly inside (0,1)')
    fractions=sorted(float(v) for v in fractions)
    collocation_weight=number(config.get('collision_collocation_weight',1.),'collision_collocation_weight')
    spline=None
    parameterization=config.get('trajectory_parameterization',{})
    if parameterization:
        kind=parameterization.get('kind')
        if kind not in ('cubic_bspline','phase_cubic_bspline'):
            raise ValueError('Unknown trajectory parameterization')
        from .hand_sequence_spline import HandSplineParameters
        spline_type=HandSplineParameters
        extra={}
        if kind=='phase_cubic_bspline':
            from .hand_sequence_phase_spline import PhaseHandSplineParameters
            spline_type=PhaseHandSplineParameters
            extra=dict(transition_frames=parameterization.get('transition_frames',6),
                       allowed_transition_roles=parameterization.get('allowed_transition_roles'),
                       warm_fit_scope=parameterization.get('warm_fit_scope','strong_contact'))
        spline=spline_type(warm['timestamps'],warm,hands,device=device,dtype=fk.dtype,
                    control_count=parameterization.get('control_count',17),position_bound=position_bound,
                    finger_space=parameterization.get('finger_space','logit_spline'),**extra)
        trainable=spline.controls
    else:
        trainable=torch.nn.Parameter(fk.t(np.stack(parameters,1)))

    finger_speed_initialization=None
    if config.get('finger_speed_initialization') is not None and config.get('finger_speed_initialization') is not False:
        from .hand_sequence_finger_speed_projection import projection_policy, initialize_finger_speed
        policy=projection_policy(config['finger_speed_initialization'])
        if config.get('strict_hold_parameterization') or editable_frames is not None:
            raise ValueError('Finger speed QP initialization does not allow hold/frozen-frame overlays')
        if finger_speed is None or policy['limit_rad_s']!=finger_speed['limit_rad_s']:
            raise ValueError('Finger speed QP initialization requires the same enabled soft gate')
        finger_speed_initialization=initialize_finger_speed(spline,warm['timestamps'],policy)

    shared_hold=None
    if config.get('strict_hold_parameterization'):
        from .hand_sequence_shared_hold import SharedHoldParameters
        editable_metadata['shared_hold_scope'] = validate_shared_hold_editable_frames(
            editable_frames,hands,config['strict_hold_parameterization'])
        shared_hold=SharedHoldParameters(warm,hands,config=config['strict_hold_parameterization'],device=device,dtype=fk.dtype)
    trainables=[trainable]+([shared_hold.controls] if shared_hold is not None and shared_hold.controls.numel() else [])
    feasible_step=None;translation_projector=None
    if config.get('feasible_spline_step') is not None and config.get('feasible_spline_step') is not False:
        from .hand_sequence_feasible_step import (validate_feasible_spline_step,
            decoded_feasibility,backtrack_proposed_step)
        feasible_step=validate_feasible_spline_step(config['feasible_spline_step'])
        if spline is None or execution_backend!='eager':
            raise ValueError('feasible_spline_step requires an eager spline parameterization')
        if feasible_step['kind']=='affine_projection_backtracking':
            if shared_hold is not None or editable_frames is not None:
                raise ValueError('Affine translation projection does not support shared-hold or frozen-frame overlays')
            from .hand_sequence_translation_projection import AffineTranslationProjector
            translation_projector=AffineTranslationProjector(spline,position_bound)
    from .hand_sequence_shared_hold_feasible_step import (validate_feasible_shared_hold_step,
        require_feasible_shared_hold_start)
    feasible_shared_step=validate_feasible_shared_hold_step(config.get('feasible_shared_hold_step'),
        shared_hold=shared_hold,spline=spline,execution_backend=execution_backend,
        feasible_spline_step=feasible_step)
    if feasible_shared_step is not None:
        from .hand_sequence_feasible_step import decoded_feasibility,backtrack_proposed_step
        shared_hold.metadata['wrist_bounds']='Every accepted control update is checked against unchanged original-frame source wrist boxes; shared outputs are never clamped.'
    step_policy=feasible_step if feasible_step is not None else feasible_shared_step
    step_report_key='feasible_shared_hold_step' if feasible_shared_step is not None else 'feasible_spline_step'
    feasible_step_history=[]
    enforce_finger_speed=False
    if config.get('enforce_finger_speed_feasibility') is not None and config.get('enforce_finger_speed_feasibility') is not False:
        from .hand_sequence_finger_speed_feasibility import validate_speed_feasibility, decoded_finger_speed_status
        enforce_finger_speed=validate_speed_feasibility(config['enforce_finger_speed_feasibility'],
            initialization=finger_speed_initialization,soft_policy=finger_speed,backtracking=feasible_step)
    parameter_mask=None;frozen_parameters=None;frozen_wrist_bounds={}
    if editable_frames is not None:
        parameter_mask=fk.t(np.stack([editable_frames[side] for side in hands],axis=1),torch.bool)
        frozen_parameters=fk.t(np.stack(parameters,axis=1)).detach().clone()
        frozen_wrist_bounds={side:frozen_refinement_wrist_bounds(
            warm[f'{side}_wrist_position_object'],warm[f'{side}_wrist_rotation_object'],
            h['source_position'].detach().cpu().numpy(),h['source_rotation'].detach().cpu().numpy(),
            editable_frames[side],position_bound,rotation_bound) for side,h in hands.items()}

    def dense_parameters():
        base_parameters=spline.dense() if spline is not None else trainable
        dense=shared_hold.decode(base_parameters) if shared_hold is not None else base_parameters
        return apply_refinement_frame_mask(dense,frozen_parameters,parameter_mask)

    def bounds_status():
        with torch.no_grad():
            value=dense_parameters()
            if step_policy is not None:
                status=decoded_feasibility(value,hands,position_bound,rotation_bound)
                if enforce_finger_speed:
                    speed_status=decoded_finger_speed_status(value,hands,warm['timestamps'])
                    status['finger_speed']=speed_status
                    status['feasible']=status['feasible'] and speed_status['feasible']
                    status['finite']=status['finite'] and speed_status['finite']
                return status
            # Frozen wrist states are replaced by actual warm matrices before
            # FK/objectives/export. Their re-encoded latent float32 rotvecs are
            # not live output and must not determine editable-frame feasibility.
            bounded=value if parameter_mask is None else torch.where(parameter_mask[:,:,None],value,torch.zeros_like(value))
            position_excess=float(torch.relu(bounded[:,:,:3].abs()-position_bound).max())
            rotation_excess=float(torch.relu(bounded[:,:,3:6].abs()-rotation_bound).max())
            status=dict(feasible=position_excess<=1e-6 and rotation_excess<=1e-6,
                        position_box_excess_m=position_excess,rotation_vector_box_excess_rad=rotation_excess,
                        bounds_are_componentwise_boxes=True)
            if parameter_mask is not None:
                status.update(feasibility_scope='editable_frames_with_exact_frozen_warm_overlay',
                    frozen_reference_bounds=frozen_wrist_bounds,
                    all_frames_within_bounds=status['feasible'] and all(
                        item['within_original_tolerance'] for item in frozen_wrist_bounds.values()),
                    frozen_policy='Exact warm wrist/finger overlay in every FK/objective and original frozen values at export; inherited bound deviations are reported, not repaired or accepted.')
            return status


    def frozen_states(rots,poss,objr,objp):
        result={}
        inverse=objr.transpose(-1,-2)
        for side,h in frozen_hands.items():
            body_rotation=torch.einsum('bij,bnjk->bnik',inverse,rots)
            body_position=torch.einsum('bij,bnj->bni',inverse,poss-objp[:,None])
            posed=torch.einsum('bnij,nj->bni',body_rotation[:,h['bodies']],h['points'])+body_position[:,h['bodies']]
            result[side]=dict(points=posed,body_rotation=body_rotation,body_position=body_position)
        return result

    def cross_clearance(states):
        length=len(next(iter(states.values()))['points'])
        values=[];minimum=torch.full((length,),float('inf'),dtype=fk.dtype,device=device)
        for source_side,target_side in [('left','right'),('right','left')]:
            source=states[source_side]['points'];target=states[target_side]
            target_hand=hands[target_side] if target_side in hands else frozen_hands[target_side]
            for geom,body,query in target_hand['mesh_queries']:
                distance=query.distance(relative_points(source,target['body_rotation'][:,body],target['body_position'][:,body]))
                minimum=torch.minimum(minimum,distance.min(1).values)
                values.append(torch.relu(cross_margin-distance).topk(min(12,source.shape[1]),dim=1).values)
        merged=torch.cat(values,1)
        mean_loss=(400*merged.square().mean(1)+20*merged.max(1).values.square()).mean()
        return loss_weights['cross_hand']*mean_loss+cross_peak_weight*merged.max().square(),mean_loss,minimum

    def collocation(q,states,fraction):
        # Interpolate only for collision collocation. Output source frames and
        # timestamps remain unchanged. Native root is interpolated on SO(3).
        qmid=(1-fraction)*q[:-1]+fraction*q[1:]
        if fk.free_address is not None:
            from .repair_tensor import _quaternion_matrix
            a=fk.free_address
            root_R=_quaternion_matrix(q[:,a+3:a+7],torch)
            rm=rotation_exp(fraction*so3_log(root_R[1:]@root_R[:-1].transpose(-1,-2)))@root_R[:-1]
            qmid=qmid.clone();qmid[:,a+3:a+7]=rotation_quaternion(rm)
        rots,poss=fk(qmid)
        objr=rotation_exp(fraction*so3_log(object_R[1:]@object_R[:-1].transpose(-1,-2)))@object_R[:-1]
        objp=(1-fraction)*object_t[:-1]+fraction*object_t[1:]
        middle=frozen_states(rots,poss,objr,objp)
        loss=qmid.sum()*0;metrics={}
        for side,h in hands.items():
            endpoint=states[side];wrist=h['hand']['wrist'];ids=h['bodies']
            rr0=endpoint['wrist_rotation_object']
            rr=rotation_exp(fraction*so3_log(rr0[1:]@rr0[:-1].transpose(-1,-2)))@rr0[:-1]
            pp0=endpoint['wrist_position_object'];pp=(1-fraction)*pp0[:-1]+fraction*pp0[1:]
            native=torch.einsum('bnij,nj->bni',rots[:,ids],h['points'])+poss[:,ids]
            local=relative_points(native,rots[:,wrist],poss[:,wrist])
            posed=torch.einsum('bij,bnj->bni',rr,local)+pp[:,None]
            body_rotation=torch.einsum('bij,bnjk->bnik',rr@rots[:,wrist].transpose(-1,-2),rots)
            body_position=torch.einsum('bij,bnj->bni',rr@rots[:,wrist].transpose(-1,-2),poss-poss[:,wrist,None])+pp[:,None]
            signed=obj_query.distance(posed);violation=torch.relu(margin-signed)
            object_loss=(400*violation.square().mean(1)+20*violation.max(1).values.square()).mean()
            loss=loss+loss_weights['object_collision']*object_loss+object_peak_weight*violation.max().square()
            if scene_enabled:
                scene_loss,scene_min=scene_query.penalty(posed,objr,objp,margin)
                loss=loss+loss_weights['scene_collision']*(scene_loss.mean()+object_peak_weight*torch.relu(margin-scene_min).max().square())
                metrics[side+'_scene_min']=scene_min
            self_values=[]
            for body,source_ids,query in h['self_queries']:
                distance=query.distance(relative_points(native[:,source_ids],rots[:,body],poss[:,body]))
                self_values.append(torch.relu(margin-distance).topk(min(12,len(source_ids)),dim=1).values)
            if self_values:
                loss=loss+loss_weights['self_collision']*torch.cat(self_values,1).topk(12,dim=1).values.square().mean()
            middle[side]=dict(points=posed,body_rotation=body_rotation,body_position=body_position)
            metrics[side+'_object_min']=signed.min(1).values
        cross,_,minimum=cross_clearance(middle)
        metrics['cross_min']=minimum
        return loss+cross,metrics


    def evaluate():
        x=dense_parameters()
        q=qbase.clone()
        for index,(side,h) in enumerate(hands.items()):
            finger=h['source_finger']+x[:,index,6:]
            if editable_frames is not None:
                finger=apply_refinement_frame_mask(finger,h['frozen_warm']['finger'],h['editable_frames'])
            q[:,h['address_tensor']]=finger
        rots,poss=fk(q);states=frozen_states(rots,poss,object_R,object_t);components={};loss=x.sum()*0
        for index,(side,h) in enumerate(hands.items()):
            wrist=h['hand']['wrist'];ids=h['bodies'];address=h['address_tensor']
            rr=rotation_exp(x[:,index,3:6])@h['source_rotation'];pp=h['source_position']+x[:,index,:3]
            if editable_frames is not None:
                pp=apply_refinement_frame_mask(pp,h['frozen_warm']['position'],h['editable_frames'])
                rr=apply_refinement_frame_mask(rr,h['frozen_warm']['rotation'],h['editable_frames'])
            native_points=torch.einsum('bnij,nj->bni',rots[:,ids],h['points'])+poss[:,ids]
            local=relative_points(native_points,rots[:,wrist],poss[:,wrist])
            posed=torch.einsum('bij,bnj->bni',rr,local)+pp[:,None]
            # All descendant transforms are current in the same graph.
            body_rotation=torch.einsum('bij,bnjk->bnik',rr@rots[:,wrist].transpose(-1,-2),rots)
            body_position=torch.einsum('bij,bnj->bni',rr@rots[:,wrist].transpose(-1,-2),poss-poss[:,wrist,None])+pp[:,None]
            signed=obj_query.distance(posed);violation=torch.relu(margin-signed)
            object_loss=(400*violation.square().mean(1)+20*violation.max(1).values.square()).mean()
            loss=loss+loss_weights['object_collision']*object_loss
            components[side+'_object']=object_loss
            object_peak=violation.max().square()
            loss=loss+object_peak_weight*object_peak;components[side+'_object_global_peak']=object_peak
            if scene_enabled:
                scene_loss,scene_min=scene_query.penalty(posed,object_R,object_t,margin)
                scene_peak=torch.relu(margin-scene_min).max().square()
                loss=loss+loss_weights['scene_collision']*(scene_loss.mean()+object_peak_weight*scene_peak)
                components[side+'_scene']=scene_loss.mean()
                components[side+'_scene_global_peak']=scene_peak
            contact_frame=torch.zeros(count,dtype=fk.dtype,device=device);gaps={};contact_selection={}
            for digit,ids in h['members'].items():
                distance=h['region_queries'][digit].distance(posed[:,ids]).abs()
                for first,end,queries in h['schedules']:
                    replacement=queries[digit].distance(posed[first:end,ids]).abs()
                    distance=torch.cat((distance[:first],replacement,distance[end:]),0)
                digit_loss,selected,selection=select_contact_loss(distance,posed[:,ids],h['active'],margin,spaced_options)
                gaps[digit]=selected
                contact_frame=contact_frame+digit_loss
                if selection is not None:
                    selection['global_sample_indices']=torch.where(selection['valid'],ids[selection['indices'].clamp_min(0)],-1)
                    contact_selection[digit]=selection
            contact=masked_mean(contact_frame,h['active'])
            loss=loss+loss_weights['contact']*contact;components[side+'_contact']=contact
            self_values=[]
            for body,ids,query in h['self_queries']:
                distance=query.distance(relative_points(native_points[:,ids],rots[:,body],poss[:,body]))
                self_values.append(torch.relu(margin-distance).topk(min(12,len(ids)),dim=1).values)
            self_loss=torch.cat(self_values,1).topk(12,dim=1).values.square().mean() if self_values else loss*0
            loss=loss+loss_weights['self_collision']*self_loss;components[side+'_self']=self_loss
            source_loss=(prior['position']*x[:,index,:3].square().sum(1)+prior['rotation']*x[:,index,3:6].square().sum(1)
                         +prior['finger']*x[:,index,6:].square().sum(1)).mean()
            loss=loss+source_loss;components[side+'_source']=source_loss
            if refinement_prior is not None:
                trust,trust_terms=refinement_reference_terms(pp,rr,q[:,address],h['refinement_reference'],refinement_prior)
                loss=loss+trust;components[side+'_refinement_reference']=trust
                components.update({side+'_refinement_reference_'+key:value for key,value in trust_terms.items()})
            if side in phase_fidelity:
                fidelity,fidelity_terms=phase_fidelity_terms(pp,rr,h['refinement_reference'],
                    h['phase_fidelity_mask'],phase_fidelity[side]['weights'])
                loss=loss+fidelity;components[side+'_phase_wrist_fidelity']=fidelity
                components.update({side+'_phase_wrist_fidelity_'+key:value for key,value in fidelity_terms.items()})
            pw=torch.einsum('bij,bj->bi',object_R,pp)+object_t;rw=object_R@rr
            temporal,terms=temporal_terms(pw if space in ('world','phase') else pp,rw if space in ('world','phase') else rr,q[:,address],
                                         pp,rr,h['hold'],dt,weights,phase_coordinates=space=='phase',relative_mask=h['relative_temporal'])
            loss=loss+temporal;components[side+'_temporal']=temporal
            if finger_speed is not None:
                speed_loss,speed_term=finger_speed_terms(q[:,address],finger_speed_intervals,finger_speed)
                loss=loss+speed_loss;components[side+'_finger_speed']=speed_loss
                components[side+'_finger_speed_normalized_excess']=speed_term
            if motion_speed is not None and side in motion_speed_indices:
                speed_loss,speed_terms=motion_speed_terms(
                    pw if motion_speed['coordinate_space']=='world' else pp,
                    rw if motion_speed['coordinate_space']=='world' else rr,
                    motion_speed_intervals,motion_speed,motion_speed_indices[side])
                loss=loss+speed_loss;components[side+'_motion_speed']=speed_loss
                components.update({side+'_motion_speed_'+key:value for key,value in speed_terms.items()})
            # Optional world-motion budgets penalize only excess, not motion itself.
            budget_loss=loss*0
            for label,value in [('linear_acceleration',_difference(pw,dt,2)),
                                ('angular_acceleration',_difference(rotation_velocity(rw,dt),dt))]:
                budget=config.get('world_motion_budgets',{}).get(label)
                if budget:
                    cap=number(budget['maximum'],label,0);weight=number(budget['weight'],label+'_weight')
                    budget_loss=budget_loss+weight*torch.relu(value.norm(dim=-1)-cap).square().mean()
            loss=loss+budget_loss;components[side+'_world_budget']=budget_loss
            anchor_loss=loss*0
            for item,point,target_point in h['anchor_tensors']:
                body=item['body']
                actual=torch.einsum('bij,j->bi',body_rotation[:,body],point)+body_position[:,body]
                residual=(actual-target_point).square().sum(-1)
                first,end=item['start'],item['end']+1
                anchor_loss=anchor_loss+residual[first:end].mean()
            if h['anchors']:
                anchor_loss=anchor_loss/len(h['anchors'])
            loss=loss+loss_weights['fixed_anchor']*anchor_loss;components[side+'_anchor']=anchor_loss
            states[side]=dict(wrist_position_object=pp,wrist_rotation_object=rr,finger_qpos=q[:,address],
                              points=posed,body_rotation=body_rotation,body_position=body_position,
                              sampled_object_min=signed.min(1).values,sampled_object_argmin=signed.argmin(1),
                              contact_frame=contact_frame,gaps=gaps,temporal_terms=terms,contact_selection=contact_selection)
            if scene_enabled:
                states[side]['sampled_scene_min']=scene_min
        cross_loss,cross_mean,cross_min=cross_clearance(states)
        loss=loss+cross_loss;components['cross_hand']=cross_mean;components['cross_hand_total']=cross_loss
        collocation_metrics={}
        for fraction in fractions:
            collocation_loss,detail=collocation(q,states,fraction)
            loss=loss+collocation_weight*collocation_loss/len(fractions)
            components[f'collocation_{fraction}']=collocation_loss
            collocation_metrics[str(fraction)]=detail
        states['_collocation']=collocation_metrics
        return loss,components,states,cross_min

    def project():
        if spline is not None:
            return  # Never clamp dense outputs; optional control backtracking is separate.
        if feasible_shared_step is not None:
            from .hand_sequence_shared_hold_feasible_step import project_base_controls
            project_base_controls(trainable,hands,position_bound,rotation_bound)
            return  # The live shared/blended decoder is still filtered as a whole.
        x=trainable
        with torch.no_grad():
            x[:,:,:3].clamp_(-position_bound,position_bound);x[:,:,3:6].clamp_(-rotation_bound,rotation_bound)
            for index,h in enumerate(hands.values()):
                x[:,index,6:]=torch.minimum(torch.maximum(h['source_finger']+x[:,index,6:],h['low']),h['high'])-h['source_finger']

    def export(path):
        arrays={key:value.copy() for key,value in warm.items() if not key.startswith(('parameterization_','contact_selection_','motion_speed_')) and not key.endswith('_scene_min_signed_m')}
        source_deviation={};selection_summary={};refinement_deviation={};frozen_checks={};motion_speed_metrics={}
        with torch.no_grad():
            loss,components,states,cross=evaluate()
            for side,h in hands.items():
                frozen_checks[side]={}
                for key in ('wrist_position_object','wrist_rotation_object','finger_qpos'):
                    arrays[f'{side}_{key}'],check=export_refinement_frame_values(states[side][key],warm[f'{side}_{key}'],
                        editable_frames[side] if editable_frames is not None else None)
                    if check is not None:
                        frozen_checks[side][key]=check
                update_export_contact_policy(arrays,side,h)
                arrays[f'{side}_sampled_object_min_signed_m']=states[side]['sampled_object_min'].cpu().numpy().copy()
                if scene_enabled:
                    arrays[f'{side}_sampled_scene_min_signed_m']=states[side]['sampled_scene_min'].cpu().numpy().copy()
                ids=states[side]['sampled_object_argmin'].cpu().numpy()
                arrays[f'{side}_sampled_object_argmin_sample']=ids
                arrays[f'{side}_sampled_object_argmin_geom']=h['point_geoms'][ids]
                arrays[f'{side}_per_frame_contact_loss_m2']=states[side]['contact_frame'].cpu().numpy().copy()
                selection_summary[side]={}
                for digit,selection in states[side]['contact_selection'].items():
                    for key in ['global_sample_indices','valid','distances','selected_count','missing_count','coverage_complete','active','initial_eligible_count','invalid_input_count']:
                        arrays[f'contact_selection_{side}_{digit}_{key}']=selection[key].cpu().numpy().copy()
                    valid=selection['valid'];distances=selection['distances']
                    selection_summary[side][digit]=dict(active_frames=int(selection['active'].sum()),
                        spatially_complete_active_frames=int(selection['coverage_complete'].sum()),
                        missing_selected_points=int(selection['missing_count'].sum()),invalid_input_points=int(selection['invalid_input_count'].sum()),
                        maximum_selected_gap_m=float(distances[valid].max()) if bool(valid.any()) else None,
                        spacing_scope=selection['spacing_scope'],object_target_spacing_ensured=False,
                        spatial_coverage_is_not_contact_acceptance=True)
                pos_error=torch.linalg.vector_norm(states[side]['wrist_position_object']-h['source_position'],dim=-1)
                rot_error=torch.linalg.vector_norm(so3_log(states[side]['wrist_rotation_object']@h['source_rotation'].transpose(-1,-2)),dim=-1)
                finger_error=states[side]['finger_qpos']-h['source_finger']
                source_deviation[side]=dict(wrist_position_mean_m=float(pos_error.mean()),wrist_position_max_m=float(pos_error.max()),
                    wrist_angle_mean_rad=float(rot_error.mean()),wrist_angle_max_rad=float(rot_error.max()),
                    finger_rms_rad=float(finger_error.square().mean().sqrt()),finger_max_abs_rad=float(finger_error.abs().max()))
                if refinement_prior is not None:
                    residual=refinement_reference_residuals(states[side]['wrist_position_object'],
                        states[side]['wrist_rotation_object'],states[side]['finger_qpos'],h['refinement_reference'])
                    delta_p=residual['position'].norm(dim=-1);delta_r=residual['rotation'].norm(dim=-1);delta_q=residual['finger']
                    refinement_deviation[side]=dict(wrist_position_mean_m=float(delta_p.mean()),wrist_position_max_m=float(delta_p.max()),
                        wrist_angle_mean_rad=float(delta_r.mean()),wrist_angle_max_rad=float(delta_r.max()),
                        finger_rms_rad=float(delta_q.square().mean().sqrt()),finger_max_abs_rad=float(delta_q.abs().max()))
                if motion_speed is not None and side in motion_speed['intervals']:
                    p=arrays[f'{side}_wrist_position_object'];r=arrays[f'{side}_wrist_rotation_object']
                    if motion_speed['coordinate_space']=='world':
                        p=np.einsum('bij,bj->bi',ctx.R,p)+ctx.t;r=ctx.R@r
                    motion_speed_metrics[side],speed_arrays=motion_speed_measure(p,r,arrays['timestamps'],
                        motion_speed,motion_speed['intervals'][side])
                    arrays.update({f'motion_speed_{side}_{key}':value for key,value in speed_arrays.items()})

            if spline is not None:
                arrays['parameterization_spline_controls']=spline.controls.cpu().numpy().copy()
                arrays['parameterization_spline_basis']=spline.basis.cpu().numpy().copy()
                for side,reference in spline.references.items():
                    arrays[f'parameterization_{side}_rotation_reference']=reference.cpu().numpy().copy()
                if hasattr(spline,'export_arrays'):
                    arrays.update(spline.export_arrays())
            if shared_hold is not None:
                arrays['parameterization_shared_hold_controls']=shared_hold.controls.cpu().numpy().copy()
                arrays['parameterization_shared_hold_references']=np.asarray([v.cpu().numpy() for v in shared_hold.references])
            arrays['paired_sampled_cross_min_signed_m']=cross.cpu().numpy().copy()
            for fraction,detail in states['_collocation'].items():
                for label,value in detail.items():
                    arrays[f'collocation_{fraction}_{label}_signed_m']=value.cpu().numpy().copy()
        np.savez_compressed(path,**arrays)
        result=dict(objective=float(loss),wrist_bounds=bounds_status(),source_deviation=source_deviation,spaced_contact_selection=selection_summary,components={k:float(v) for k,v in components.items()},
                    sampled_cross_min_signed_m=float(cross.min()))
        if scene_enabled:
            result['sampled_scene_min_signed_m']={side:float(states[side]['sampled_scene_min'].min()) for side in hands}
            result['collocation_scene_min_signed_m']={fraction:{side:float(detail[side+'_scene_min'].min())
                for side in hands} for fraction,detail in states['_collocation'].items()}
        if refinement_prior is not None:
            result['refinement_reference_deviation']=refinement_deviation
        if editable_frames is not None:
            result['frozen_refinement_frames']=frozen_checks
        if finger_speed is not None:
            result['finger_speed']={side:finger_speed_measure(arrays[side+'_finger_qpos'],
                arrays['timestamps'],finger_speed,ctx.hands[side]['finger_joints'])for side in hands}
        if motion_speed is not None:
            result['motion_speed']=motion_speed_metrics
        return result

    project()
    shared_start_report=None
    if feasible_shared_step is not None:
        from .hand_sequence_shared_hold_feasible_step import repair_shared_hold_boundary_start
        shared_start_report=repair_shared_hold_boundary_start(trainable,shared_hold,hands,
            position_bound,rotation_bound,bounds_status,feasible_shared_step,editable_frames)
        (out/'shared_hold_initialization.json').write_text(json.dumps(shared_start_report,indent=2)+'\n')
        require_feasible_shared_hold_start(bounds_status())
    elif feasible_step is not None and not bounds_status()['feasible']:
        raise ValueError('Feasible spline stepping requires an initially finite state inside original wrist boxes and native finger limits')
    initial_metrics=export(out/'initial_projected.npz')
    initial_rate=number(config.get('paired_learning_rate',.002),'paired_learning_rate',1e-8,.1)
    schedule=config.get('learning_rate_schedule',{'kind':'constant'})
    if schedule.get('kind') not in ('constant','cosine'):
        raise ValueError('Learning-rate schedule must be constant or cosine')
    minimum_rate=number(schedule.get('minimum_rate',initial_rate),'minimum_rate',0,initial_rate)
    optimizer=torch.optim.Adam(trainables,lr=initial_rate)
    rejected_bounds=[]

    best_value=float('inf');best=None;best_iteration=0;history=[]
    optimize_start=time.monotonic()
    replay=None;capture_seconds=0.
    if execution_backend=='cuda_graph':
        from .hand_sequence_acceleration import CudaObjectiveReplay
        capture_start=time.monotonic()
        replay=CudaObjectiveReplay(evaluate,trainables)
        torch.cuda.synchronize(device)
        capture_seconds=time.monotonic()-capture_start
    iteration_start=time.monotonic()
    for iteration in range(steps):
        if schedule.get('kind')=='cosine':
            current_rate=minimum_rate+(initial_rate-minimum_rate)*.5*(1+np.cos(np.pi*iteration/max(1,steps-1)))
            optimizer.param_groups[0]['lr']=float(current_rate)
        if replay is None:
            optimizer.zero_grad();loss,components,states,cross=evaluate()
        else:
            loss,components,states,cross=replay.replay()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite paired objective at iteration '+str(iteration+1))
        scalar=float(loss.detach())
        feasibility=bounds_status()
        if step_policy is not None and not feasibility['feasible']:
            raise RuntimeError('A backtracked step left the original feasible set: '+step_report_key)
        if not feasibility['feasible']:
            rejected_bounds.append(dict(iteration=iteration,**feasibility))
        if feasibility['feasible'] and scalar < best_value:
            best_value=scalar;best=[value.detach().clone() for value in trainables];best_iteration=iteration
        if replay is None:
            loss.backward()
        if any(value.grad is None or not bool(torch.isfinite(value.grad).all()) for value in trainables):
            raise ValueError('Nonfinite paired gradient at iteration '+str(iteration+1))
        torch.nn.utils.clip_grad_norm_(trainables,10.)
        previous=[value.detach().clone() for value in trainables] if step_policy is not None else None
        optimizer.step();project()
        if step_policy is not None:
            projection=translation_projector.project() if translation_projector is not None else None
            filtered=backtrack_proposed_step(trainables,previous,feasibility,bounds_status,step_policy)
            if projection is not None:
                filtered['translation_projection']=projection
            feasible_step_history.append(dict(iteration=iteration+1,**filtered))
        if iteration%25==0 or iteration+1==steps:
            row=dict(iteration=iteration+1,objective_before_update=scalar,learning_rate=float(optimizer.param_groups[0]['lr']),
                     wrist_bounds_before_update=feasibility,
                     components={k:float(v.detach()) for k,v in components.items()},
                     sampled_cross_min_signed_m=float(cross.detach().min()))
            if step_policy is not None:
                row[step_report_key]=feasible_step_history[-1]
            history.append(row);print(json.dumps(row),flush=True)
        if (iteration+1)%50==0:
            export(out/f'checkpoint_{iteration+1:04d}.npz')
    if device.startswith('cuda:'):
        torch.cuda.synchronize(device)
    iteration_seconds=time.monotonic()-iteration_start
    last_metrics=export(out/'last.npz')
    if last_metrics['wrist_bounds']['feasible'] and last_metrics['objective'] < best_value:
        best_value=last_metrics['objective'];best=[value.detach().clone() for value in trainables];best_iteration=steps
    if best is None:
        _write(out/'report.json',dict(schema_version=1,status='no_feasible_parameterized_checkpoint',
                    motion_accepted=False,all_gradients_finite=True,rejected_wrist_bounds=rejected_bounds,
                    initial_metrics=initial_metrics,last_metrics=last_metrics,
                    refinement_reference_prior=refinement_metadata,
                    refinement_editable_frames=editable_metadata,
                    parameterization=spline.metadata if spline is not None else None,
                    strict_hold_parameterization=shared_hold.metadata if shared_hold is not None else None,
                    input_config=input_config,static_object_policy=static_policy))
        raise ValueError('No checkpoint satisfies configured source-relative wrist boxes; invalid spline outputs not selected')
    with torch.no_grad():
        for value,saved in zip(trainables,best):
            value.copy_(saved)
    best_metrics=export(out/'hand_sequence.npz')
    if device.startswith('cuda:'):
        torch.cuda.synchronize(device)
    report=dict(schema_version=1,mode='paired_temporal',frame_count=count,fps=fps,motion_accepted=False,
                scope='Both free wrists and fingers optimized jointly; original source body/object untouched.',
                binding=ctx.binding,input_config=input_config,static_object_policy=static_policy,
                scene_clearance=(dict(scene_query.metadata,weight=loss_weights['scene_collision'],
                    global_peak_weight=object_peak_weight,contact_phase_masked=False,
                    collocation_fractions=fractions) if scene_query is not None else
                    dict(enabled=False,reason='scene_collision weight is zero; legacy configuration')),
                temporal_coordinate_space=space,temporal_weights=weights,
                source_prior=prior,loss_weights=loss_weights,spaced_contact_selection=spaced_options,global_worst_object_weight=object_peak_weight,
                global_worst_cross_hand_weight=cross_peak_weight,steps=steps,best_objective_iteration=best_iteration,
                best_objective_is_not_motion_acceptance=True,all_gradients_finite=True,history=history,
                trajectory_parameterization=spline.metadata if spline is not None else dict(kind='per_frame'),
                strict_hold_parameterization=shared_hold.metadata if shared_hold is not None else None,
                rejected_wrist_bounds=rejected_bounds,learning_rate_schedule=dict(schedule,initial_rate=initial_rate,minimum_rate=minimum_rate),
                initial_metrics=initial_metrics,last_metrics=last_metrics,best_metrics=best_metrics,
                collision_collocation_fractions=fractions,collision_collocation_weight=collocation_weight,
                collocation_rule='Linear object-local wrist translation and native finger hinges; principal-SO3 wrist/object/root interpolation; output clock unchanged.',
                frozen_opposite_hands={side:dict(original_mesh_geoms=[int(row[0]) for row in h['mesh_queries']],
                    original_native_primitive_omissions=h['primitive_omissions'],source_qpos_unchanged=True,collision_witness_samples=h['collision_witness_samples']) for side,h in frozen_hands.items()},
                optimization_seconds=time.monotonic()-optimize_start,elapsed_seconds=time.monotonic()-start,
                execution=dict(backend=execution_backend,rotation_backend=rotation_backend,
                    capture_seconds=capture_seconds,iteration_seconds=iteration_seconds,
                    capture_included_in_optimization_seconds=True,
                    checks='Every iteration retains finite loss/gradient checks, wrist bounds and best-checkpoint selection; all original frames and configured collision samples retained.'),
                source_warm_start=dict(path=str(warm_path),sha256=warm_sha256),hands={},
                code_sha256={name:_sha(Path(__file__).with_name(name)) for name in ['hand_sequence_pair.py','hand_sequence_rotation.py','hand_sequence.py','grasp_distal_pad.py','hand_sequence_collision_samples.py','hand_scene_clearance.py','static_scene.py']+(['hand_sequence_spline.py'] if spline is not None else [])+(['hand_sequence_shared_hold.py'] if shared_hold is not None else [])+(['hand_sequence_spaced_contact.py'] if spaced_options is not None else [])},
                limitations=['Signed queries and original-surface sampling are incomplete collision proxies.',
                             'Self objective omits same-body and adjacent body pairs; independent full audits retain them.',
                             'Cross-hand objective omits no hand pairs but samples source surfaces; target original meshes are complete.',
                             'Arm reachability, non-hand body/scene collision and physics are outside this hand-only scope.',
                             'Hand/scene clearance is sampled guidance when enabled; absent scene data remains unknown, not clear.'])
    report['refinement_reference_prior']=refinement_metadata
    if enforce_finger_speed:
        report['enforce_finger_speed_feasibility']=dict(enabled=True,limit_rad_s=3.,positive_tolerance_rad_s=0.,
            policy='Additional decode-only original-interval native finger gate in the existing control backtracking. Reduce or restore infeasible proposals; no posthoc output projection/clipping or retiming.',
            source_clock_reference=dict(path=str(warm_path),sha256=warm_sha256),source_interval_count=count-1,
            original_wrist_boxes_and_native_position_limits_unchanged=True,
            scope='Native discrete original-frame speed only; not continuous-time velocity, collision, dynamics or motion acceptance.',
            motion_accepted=False)
        report['code_sha256']['hand_sequence_finger_speed_feasibility.py']=_sha(Path(__file__).with_name('hand_sequence_finger_speed_feasibility.py'))
    if finger_speed_initialization is not None:
        report['finger_speed_initialization']=finger_speed_initialization
        report['code_sha256']['hand_sequence_finger_speed_projection.py']=_sha(Path(__file__).with_name('hand_sequence_finger_speed_projection.py'))
    if finger_speed is not None:
        report['finger_speed']=dict(config=finger_speed,
            normalization='Per-hand half mean squared normalized excess across all original intervals and native scalar finger joints; hand costs are summed.',
            source_clock_reference=dict(path=str(warm_path),sha256=warm_sha256),source_interval_count=count-1,
            scope='All original intervals, including free motion, acquisition, hold, support, slide and joins. No phase mask or angle wrapping.',
            policy='Soft authored animation gate, not a hard constraint or hardware limit. Exactly zero loss/gradient below gate. Native joint limits and source clock unchanged.',
            motion_accepted=False)
        report['code_sha256']['hand_sequence_finger_speed.py']=_sha(Path(__file__).with_name('hand_sequence_finger_speed.py'))
    if motion_speed is not None:
        report['motion_speed']=dict(config=motion_speed,
            normalization='Per-hand half mean of summed squared normalized linear/angular excess on explicitly selected original intervals; hand costs are summed.',
            source_clock_reference=dict(path=str(warm_path),sha256=warm_sha256),source_interval_count=count-1,
            scope='Index i means original adjacent frames [i,i+1]. No phase masking; endpoint joins remain if explicitly selected. Diagnostics include selected and all original intervals.',
            policy='Soft authored animation guidance, not a hard speed or hardware limit. Below-limit loss/gradient is zero. No retiming or native-limit changes.',
            rotation_cut_locus='Principal SO(3) increments; finite deterministic branch at pi, not differentiable across the cut locus.',
            motion_accepted=False)
        report['code_sha256']['hand_sequence_motion_speed.py']=_sha(Path(__file__).with_name('hand_sequence_motion_speed.py'))
    if phase_fidelity:
        report['phase_wrist_fidelity']=dict(
            hands={side:dict(weights=item['weights'],roles=item['roles'],
                original_frames=np.flatnonzero(item['mask']).tolist())for side,item in phase_fidelity.items()},
            reference=dict(path=str(warm_path),sha256=warm_sha256),
            normalization='Per-hand mean over selected active original frames; position m² and principal SO(3) rad².',
            policy='Additive warm-wrist pose prior; no finger term, derivative pin, retiming, contact-mask change or altered source boxes.',
            static_contact_policy='Empty active mask contributes exactly zero.',motion_accepted=False)
        report['code_sha256']['hand_sequence_phase_fidelity.py']=_sha(Path(__file__).with_name('hand_sequence_phase_fidelity.py'))
    report['refinement_editable_frames']=editable_metadata
    if step_policy is not None:
        report[step_report_key]=dict(config=step_policy,history=feasible_step_history,
            total_backtracks=sum(r['backtracks'] for r in feasible_step_history),
            rejected_trial_count=sum(r['rejected_trial_count'] for r in feasible_step_history),
            zero_steps=sum(r['zero_step'] for r in feasible_step_history),
            restored_previous_steps=sum(r['restored_previous'] for r in feasible_step_history),
            reduced_steps=sum(r['accepted_fraction']<1 for r in feasible_step_history),
            all_accepted_states_feasible=all(r['accepted_status']['feasible'] for r in feasible_step_history),
            decode_only_seconds=sum(r['elapsed_seconds'] for r in feasible_step_history),
            constraint_policy='Original-frame componentwise source-relative wrist translation and rotvec boxes, native finger bounds, finite parameters. Existing1e-6 tolerance unchanged; rotation magnitude diagnostic only.',
            adam_state_policy='Retain full-step moments and counter; filter proposed parameters only. No objective, FK or collision evaluation in backtracking.',
            limitation='Original-frame feasibility is not subframe collision, reachability, dynamics or motion acceptance.')
        report['code_sha256']['hand_sequence_feasible_step.py']=_sha(Path(__file__).with_name('hand_sequence_feasible_step.py'))
        if feasible_shared_step is not None:
            report[step_report_key].update(
                initial_feasibility_required=True,
                initialization=shared_start_report,
                parameter_policy='One common fraction for all live per-frame and shared controls; restore exact previous controls if all trials fail. No pointwise projection or shared-output clamping.',
                decoded_scope='Every original source frame after shared-hold decoding, any declared boundary blending and the original editable-frame mask.',
                strict_holds_remain_shared=True,support_slide_frames_remain_live=True,
                frozen_frame_policy='Existing objective/export masks unchanged; frozen values participate in the full-frame feasibility check.')
            report['code_sha256']['hand_sequence_shared_hold_feasible_step.py']=_sha(Path(__file__).with_name('hand_sequence_shared_hold_feasible_step.py'))
        if translation_projector is not None:
            report['feasible_spline_step']['translation_projection']=dict(
                policy='Project only Adam position controls to unchanged original-frame affine wrist boxes, then apply the same decode/backtracking filter. Other controls and Adam moments untouched.',
                failure_policy='An unsuccessful QP leaves the original Adam proposal intact; ordinary feasible backtracking remains authoritative.',
                projected_steps=sum(r['translation_projection']['applied'] for r in feasible_step_history),
                failed_steps=sum(not r['translation_projection']['success'] for r in feasible_step_history),
                seconds=sum(r['translation_projection']['elapsed_seconds'] for r in feasible_step_history))
            report['code_sha256']['hand_sequence_translation_projection.py']=_sha(Path(__file__).with_name('hand_sequence_translation_projection.py'))
    if rotation_backend=='rodrigues':
        report['code_sha256']['hand_sequence_acceleration.py']=_sha(Path(__file__).with_name('hand_sequence_acceleration.py'))
    report['code_sha256']['repair_tensor.py']=_sha(Path(__file__).with_name('repair_tensor.py'))
    if parameterization.get('kind')=='phase_cubic_bspline':
        report['code_sha256']['hand_sequence_phase_spline.py']=_sha(Path(__file__).with_name('hand_sequence_phase_spline.py'))
    for side,h in hands.items():
        report['hands'][side]=dict(phase_labels=h['roles'],contact_active_frames=int(h['active'].sum()),
                                  object_relative_temporal_roles=h['relative_temporal_roles'],
                                  fixed_anchor_eligible_frames=int(h['hold'].sum()),fixed_anchors=h['anchors'],
                                  anchor_rejections=h['anchor_rejections'],distal_pad=h['pad_metadata'],collision_witness_samples=h['collision_witness_samples'],
                                  sampled_self_adjacent_same_body_omissions=h['omissions'],
                                  phase_coordinate_stencils={str(order):dict(pose_frames=order+1,
                                      object_local=int(phase_stencil_mask(h['relative_temporal'],order).sum()),
                                      world=count-order-int(phase_stencil_mask(h['relative_temporal'],order).sum())) for order in (2,3)} if space=='phase' else None)
    report['temporal_coordinate_rule']='Each acceleration/jerk stencil enters once: 3/4 poses wholly active in the explicitly selected contact roles use object coordinates, all other stencils use world coordinates. Fingers and strict-hold-only velocity guidance unchanged.' if space=='phase' else 'Uniform '+space+' coordinates.'
    report['outputs']={p.name:dict(sha256=_sha(p),bytes=p.stat().st_size) for p in out.glob('*.npz')}
    _write(out/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config');parser.add_argument('--output',required=True)
    args=parser.parse_args();path=Path(args.config).resolve(strict=True)
    solve(json.loads(path.read_text()),path.parent,args.output)


if __name__=='__main__':
    main()
