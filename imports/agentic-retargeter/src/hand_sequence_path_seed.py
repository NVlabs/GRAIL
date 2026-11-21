"""CPU dynamic-programming warm paths for explicit support/sliding phases.

Discrete candidates remain unreviewed. This module never creates fixed material
anchors, edits a hold, changes the source clock, or accepts a motion. Geometry
failures remain in every selected key's provenance and dominate transition cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from .hand_sequence_seed import normalized_roles

DEFAULT_COSTS = dict(missing_anchor=5., incomplete_coverage=10., gap=1., source=.05,
                     strict_rejected=2., translation_velocity=1., rotation_velocity=.25,
                     finger_velocity=.25, translation_scale_m_s=.2,
                     rotation_scale_rad_s=2., finger_scale_rad_s=2., gap_scale_m=.005)
NON_HOLD = {'support', 'slide', 'sliding'}


def _array(value,shape,label):
    x=np.asarray(value,dtype=np.float64)
    if x.shape!=shape or not np.isfinite(x).all():
        raise ValueError('Invalid '+label)
    return x


def validate_candidate(candidate,names,limits):
    if candidate.get('finger_joint_names')!=list(names):
        raise ValueError('Candidate native finger order mismatch')
    p=_array(candidate['wrist_position_object'],(3,),'wrist position')
    r=_array(candidate['wrist_rotation_object'],(3,3),'wrist rotation')
    q=_array(candidate['finger_qpos'],(len(names),),'finger pose')
    if not np.allclose(r.T@r,np.eye(3),atol=2e-5,rtol=0) or not np.isclose(np.linalg.det(r),1,atol=2e-5,rtol=0):
        raise ValueError('Candidate wrist is not a proper rotation')
    if np.any(q<limits[:,0]-1e-6) or np.any(q>limits[:,1]+1e-6):
        raise ValueError('Candidate exceeds native finger limits')
    return p,r,q


def local_cost(candidate,weights):
    """Five-dimensional lexicographic cost; transitions affect only last term."""
    if candidate.get('boundary_reference',False):
        return np.zeros(5)
    quality=candidate['quality']
    collision=quality['collision_evidence'];coverage=quality['five_mm_spaced_coverage']
    fields=[]
    for category in ('hand_object','nonadjacent_self'):
        for field in ('hits','unresolved'):
            v=collision[category][field]
            if type(v) is not int or v<0:raise ValueError('Invalid or missing collision evidence')
            fields.append(v)
    gap=coverage.get('available_anchor_mean_gap_m')
    if gap is None:gap=.1
    if not np.isfinite(gap)or gap<0:raise ValueError('Invalid contact gap')
    missing=coverage['missing_count']
    if type(missing)is not int or missing<0:raise ValueError('Invalid coverage count')
    source=quality.get('source_pose_deviation',{}).get('normalized_score',0.)
    if not np.isfinite(source)or source<0:raise ValueError('Invalid source deviation')
    soft=(weights['missing_anchor']*missing+weights['incomplete_coverage']*int(not coverage['coverage_complete'])
          +weights['gap']*(gap/weights['gap_scale_m'])**2+weights['source']*source
          +weights['strict_rejected']*int(quality.get('raw_strict_rejected')is not False))
    return np.asarray([*fields,soft],dtype=np.float64)


def transition_cost(a,b,seconds,weights):
    if not seconds>0:raise ValueError('Key times must increase')
    p=np.linalg.norm(np.asarray(b['wrist_position_object'])-a['wrist_position_object'])/seconds
    r=Rotation.from_matrix(np.asarray(b['wrist_rotation_object'])@np.asarray(a['wrist_rotation_object']).T).magnitude()/seconds
    q=np.sqrt(np.mean((np.asarray(b['finger_qpos'])-a['finger_qpos'])**2))/seconds
    cost=(weights['translation_velocity']*(p/weights['translation_scale_m_s'])**2
          +weights['rotation_velocity']*(r/weights['rotation_scale_rad_s'])**2
          +weights['finger_velocity']*(q/weights['finger_scale_rad_s'])**2)
    return float(cost),dict(translation_velocity_m_s=float(p),rotation_velocity_rad_s=float(r),finger_rms_velocity_rad_s=float(q))


def choose_path(layers,timestamps,weights):
    """Choose one candidate from every explicit source key using exact DP."""
    if not layers or any(not row['candidates']for row in layers):raise ValueError('Each key needs candidate evidence')
    frames=[row['frame']for row in layers]
    if any(type(x)is not int for x in frames)or any(b<=a for a,b in zip(frames,frames[1:])):raise ValueError('Ordered distinct key frames required')
    costs=[local_cost(c,weights)for c in layers[0]['candidates']];back=[[-1]*len(costs)];summaries=[]
    for li in range(1,len(layers)):
        prev=layers[li-1];current=layers[li];next_cost=[];pred=[]
        seconds=float(timestamps[current['frame']]-timestamps[prev['frame']])
        for candidate in current['candidates']:
            unary=local_cost(candidate,weights);options=[]
            for pi,parent in enumerate(prev['candidates']):
                transition,_=transition_cost(parent,candidate,seconds,weights)
                value=costs[pi]+unary;value=value.copy();value[-1]+=transition;options.append(value)
            best=min(range(len(options)),key=lambda k:tuple(options[k])+(k,));next_cost.append(options[best]);pred.append(best)
        costs=next_cost;back.append(pred)
    winner=min(range(len(costs)),key=lambda k:tuple(costs[k])+(k,));objective=costs[winner].tolist();path=[winner]
    for li in range(len(layers)-1,0,-1):path.append(back[li][path[-1]])
    path.reverse();chosen=[layers[i]['candidates'][j]for i,j in enumerate(path)]
    for i in range(1,len(chosen)):
        value,detail=transition_cost(chosen[i-1],chosen[i],float(timestamps[frames[i]]-timestamps[frames[i-1]]),weights)
        summaries.append(dict(from_frame=frames[i-1],to_frame=frames[i],weighted_cost=value,**detail))
    return chosen,dict(candidate_indices=path,lexicographic_total_cost=objective,transitions=summaries,
                       local_costs=[local_cost(c,weights).tolist()for c in chosen],
                       ordering=['hand_object_hits','hand_object_unresolved','nonadjacent_self_hits','nonadjacent_self_unresolved','contact_source_transition_cost'])


def build_path_seed(arrays,config,plan,joint_limits,*,cost_weights=None):
    """Return new arrays and evidence; source fields and every hold stay exact.

    Plan maps side to nonoverlapping phase records with inclusive start/end,
    role, and key_layers [{frame,candidates:[...]}]. Surrounding frames are
    immutable reference endpoints; previously selected holds are never replaced.
    """
    weights={**DEFAULT_COSTS,**(cost_weights or {})}
    if set(weights)!=set(DEFAULT_COSTS)or any(type(v)not in(int,float)or not np.isfinite(v)or v<0 for v in weights.values()):raise ValueError('Invalid path cost weights')
    if any(weights[k]<=0 for k in ['gap_scale_m','translation_scale_m_s','rotation_scale_rad_s','finger_scale_rad_s']):raise ValueError('Cost scales must be positive')
    if config.get('is_static_obj',False)and any(plan.values()):raise ValueError('Static object has no contact warm path')
    t=np.asarray(arrays['timestamps']);count=len(t);fps=float(config['fps'])
    if t.shape!=(count,)or not np.array_equal(t,np.arange(count,dtype=np.float64)/fps):raise ValueError('Original uniform source clock required')
    out={k:v.copy()for k,v in arrays.items()};report=dict(schema_version=1,scope='UNREVIEWED_VARYING_FREE_HAND_WARM_PATH',motion_accepted=False,
        costs=weights,interpolation='PCHIP object-local translation/native hinges; SO3 Slerp at exact original timestamps; no extrapolation.',
        creates_fixed_material_anchors=False,hands={})
    for side,phases in plan.items():
        h=config['inputs']['hands'][side];names=h['finger_joints'];limits=_array(joint_limits[side],(len(names),2),'native limits')
        if np.any(limits[:,0]>limits[:,1]):raise ValueError('Invalid native limits')
        if list(arrays[f'{side}_finger_joint_names'])!=names:raise ValueError('Warm native joint order mismatch')
        keys=[f'{side}_{v}'for v in ['wrist_position_object','wrist_rotation_object','finger_qpos']]
        roles=normalized_roles(config,side,count);active=np.asarray(config['contact_mask'][side])
        if active.shape!=(count,)or active.dtype.kind!='b':raise ValueError('Exact contact mask required')
        touched=set();phase_reports=[]
        for phase in sorted(phases,key=lambda v:v['start']):
            first,last=phase['start'],phase['end'];role=phase['role'];layers=[]
            if type(first)is not int or type(last)is not int or not 0<=first<=last<count or role not in NON_HOLD:raise ValueError('Invalid non-hold phase')
            canonical=lambda r:'slide'if r=='sliding'else r
            if not active[first:last+1].all()or any(canonical(r)!=canonical(role)for r in roles[first:last+1]):raise ValueError('Phase contradicts original support/slide roles')
            if touched.intersection(range(first,last+1)):raise ValueError('Non-hold phases overlap')
            for layer in phase['key_layers']:
                frame=layer['frame']
                if type(frame)is not int or not first<=frame<=last:raise ValueError('Candidate key outside declared phase')
                if not layer['candidates']:raise ValueError('No candidates at source key')
                for c in layer['candidates']:
                    validate_candidate(c,names,limits)
                    if c.get('source_frame')!=frame:raise ValueError('Candidate source frame mismatch')
                    if c.get('boundary_reference',False):raise ValueError('Bank candidate cannot claim boundary status')
                layers.append(dict(frame=frame,candidates=layer['candidates']))
            layers.sort(key=lambda l:l['frame'])
            if not layers:raise ValueError('Non-hold phase needs independently selected key candidates')
            for boundary in [first-1,last+1]:
                if not 0<=boundary<count:continue
                candidate=dict(candidate_id=f'warm_boundary/{side}/{boundary}',source_frame=boundary,boundary_reference=True,
                    wrist_position_object=out[keys[0]][boundary].tolist(),wrist_rotation_object=out[keys[1]][boundary].tolist(),
                    finger_qpos=out[keys[2]][boundary].tolist(),finger_joint_names=names,
                    provenance=dict(source='Input warm arrays or preceding processed non-hold phase',role=roles[boundary],frame=boundary,motion_accepted=False))
                validate_candidate(candidate,names,limits);layers.append(dict(frame=boundary,candidates=[candidate]))
            layers.sort(key=lambda l:l['frame'])
            if layers[0]['frame']>first or layers[-1]['frame']<last:raise ValueError('Boundary phase needs an explicit endpoint key; no extrapolation')
            chosen,path=choose_path(layers,t,weights);kt=np.asarray([t[l['frame']]for l in layers]);target=t[first:last+1]
            positions=np.asarray([c['wrist_position_object']for c in chosen]);rotations=np.asarray([c['wrist_rotation_object']for c in chosen]);fingers=np.asarray([c['finger_qpos']for c in chosen])
            if len(kt)<2:raise ValueError('Varying path requires at least two distinct times')
            newp=PchipInterpolator(kt,positions,axis=0,extrapolate=False)(target)
            newq=PchipInterpolator(kt,fingers,axis=0,extrapolate=False)(target)
            newr=Slerp(kt,Rotation.from_matrix(rotations))(target).as_matrix()
            # Restore supplied key values exactly before only roundoff-scale limit clipping.
            for layer,c in zip(layers,chosen):
                f=layer['frame']
                if first<=f<=last:newp[f-first]=c['wrist_position_object'];newr[f-first]=c['wrist_rotation_object'];newq[f-first]=c['finger_qpos']
            if np.any(newq<limits[:,0]-1e-6)or np.any(newq>limits[:,1]+1e-6):raise ValueError('Interpolation violated native limits')
            clipped=np.clip(newq,limits[:,0],limits[:,1]);clip_change=float(np.max(np.abs(newq-clipped)));newq=clipped
            if not all(np.isfinite(v).all()for v in [newp,newr,newq]):raise ValueError('Nonfinite interpolated path')
            variability=max(float(np.max(np.ptp(newp,axis=0))),float(np.max(np.ptp(newq,axis=0))),float(Rotation.from_matrix(newr@newr[0].T).magnitude().max()))
            if last>first and variability<1e-10:raise ValueError('Degenerate constant non-hold path; retain original warm path instead of inventing motion')
            for key,value in zip(keys,[newp,newr,newq]):out[key][first:last+1]=value
            key_errors=[]
            for layer,c in zip(layers,chosen):
                f=layer['frame']
                if first<=f<=last:
                    dp=float(np.max(np.abs(out[keys[0]][f]-c['wrist_position_object'])));dr=float(np.max(np.abs(out[keys[1]][f]-c['wrist_rotation_object'])));dq=float(np.max(np.abs(out[keys[2]][f]-c['finger_qpos'])))
                    if max(dp,dr,dq)>2e-6:raise AssertionError('Actual key consistency failed')
                    key_errors.append(dict(frame=f,translation_max_abs_m=dp,rotation_matrix_max_abs=dr,finger_max_abs_rad=dq))
            touched.update(range(first,last+1));phase_reports.append(dict(start=first,end=last,role=role,key_frames=[l['frame']for l in layers],
                selected_keys=chosen,path=path,key_consistency=key_errors,roundoff_limit_clipping_max_rad=clip_change,
                varying_output=True,variability_measure=variability,constant_grip_propagation=False,creates_fixed_material_anchors=False,
                all_candidate_provenance=phase['key_layers'],motion_accepted=False))
        untouched=[i for i in range(count)if i not in touched]
        if any(not np.array_equal(out[k][untouched],arrays[k][untouched])for k in keys):raise AssertionError('Changed protected phase frames')
        hold_ids=[i for i,r in enumerate(roles)if r in ('hold','grasp','stable_hold')]
        if any(not np.array_equal(out[k][hold_ids],arrays[k][hold_ids])for k in keys):raise AssertionError('Changed a hold')
        if touched:
            qs=out[keys[2]][sorted(touched)]
            if np.any(qs<limits[:,0]-1e-6)or np.any(qs>limits[:,1]+1e-6):raise AssertionError('Exported native limit violation')
        report['hands'][side]=dict(phases=phase_reports,edited_frames=sorted(touched),holds_byte_identical=True,outside_phases_byte_identical=True)
    editable={f'{side}_{suffix}'for side in plan for suffix in ['wrist_position_object','wrist_rotation_object','finger_qpos']}
    if any(not np.array_equal(out[k],v)for k,v in arrays.items()if k not in editable):raise AssertionError('Protected source/body/object/timing changed')
    report['all_source_body_object_timing_metadata_byte_identical']=True
    report['limitations']=['Discrete key quality and interpolation do not certify the intervening path.','All selected strict geometry failures remain unresolved; no acceptance or fixed anchors.','Independent original-mesh and full-sequence audits are required after optimization.']
    return out,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input','config','plan','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();paths={key:Path(getattr(args,key)).resolve(strict=True)for key in ['input','config','plan']}
    from .grasp_io import GraspInputs
    config=json.loads(paths['config'].read_text());ctx=GraspInputs(config['inputs'],paths['config'].parent)
    with np.load(paths['input'],allow_pickle=False)as data:arrays={k:data[k].copy()for k in data.files}
    for key,expected in [('source_qpos',ctx.q),('object_R',ctx.R),('object_trans',ctx.t),('object_scale',ctx.native['object_scale'])]:
        if not np.array_equal(arrays[key],expected):raise ValueError('Original warm binding mismatch: '+key)
    plan=json.loads(paths['plan'].read_text())
    if plan.get('binding')!=ctx.binding:raise ValueError('Path plan must bind exact original inputs')
    limits={s:ctx.model.jnt_range[h['joints']].copy()for s,h in ctx.hands.items()}
    output,report=build_path_seed(arrays,config,plan['hands'],limits,cost_weights=plan.get('cost_weights'))
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    np.savez_compressed(target,**output);report['inputs']={key:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for key,p in paths.items()}
    report['output_sha256']=hashlib.sha256(target.read_bytes()).hexdigest();target.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
