"""Read-only all-frame original-surface sampled object-distance witnesses.

This reproduces the solver's sampled winding-distance heuristic and identifies
its worst frame/geom/material point. It is deliberately separate from exact
original-mesh intersection auditing and never certifies penetration depth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from .grasp_io import GraspInputs
from .hand_sequence import _samples,_sha,_write
from .hand_sequence_seed import normalized_roles


def audit(config,trajectory,directory='.',device='cpu',samples_per_geom=128):
    import torch
    from .repair_tensor import NativeTensorFK,WarpSignedDistance
    ctx=GraspInputs(config['inputs'],directory)
    with np.load(trajectory,allow_pickle=False) as data:
        arrays={key:data[key].copy() for key in data.files}
    for key,value in [('source_qpos',ctx.q),('object_R',ctx.R),('object_trans',ctx.t)]:
        if not np.array_equal(arrays[key],value):
            raise ValueError('Input binding mismatch: '+key)
    fk=NativeTensorFK(ctx.model,device=device,dtype=torch.float32)
    query=WarpSignedDistance(ctx.object_query.vertices,ctx.object_query.faces,device=device)
    q=ctx.q.copy()
    for side,hand in ctx.hands.items():
        q[:,hand['addresses']]=arrays[f'{side}_finger_qpos']
    report=dict(schema_version=1,trajectory=str(Path(trajectory).resolve()),trajectory_sha256=_sha(trajectory),
                binding=ctx.binding,samples_per_geom=samples_per_geom,device=device,motion_accepted=False,
                signed_distance_is_heuristic=True,full_mesh_collision_certified=False,hands={},query=query.metadata(),original_object_query=ctx.object_query_provenance)
    with torch.no_grad():
        rotations,positions=fk(fk.t(q))
        for side,hand in ctx.hands.items():
            points,bodies,geoms,members,pad=_samples(ctx,side,samples_per_geom)
            ids=fk.t(bodies,torch.long);wrist=hand['wrist']
            native=torch.einsum('bnij,nj->bni',rotations[:,ids],fk.t(points))+positions[:,ids]
            local=torch.einsum('bji,bnj->bni',rotations[:,wrist],native-positions[:,wrist,None])
            posed=torch.einsum('bij,bnj->bni',fk.t(arrays[f'{side}_wrist_rotation_object']),local)+fk.t(arrays[f'{side}_wrist_position_object'])[:,None]
            result=query.query(posed)
            if not bool(result['found'].all()) or not bool(torch.isfinite(result['distances']).all()):
                raise ValueError('Sampled query incomplete; do not manufacture a witness')
            distance=result['distances'].cpu().numpy();face=result['face_ids'].cpu().numpy()
            posed=posed.cpu().numpy();roles=normalized_roles(config,side,len(q))
            rows=[]
            for frame in range(len(q)):
                index=int(distance[frame].argmin());query_face=int(face[frame,index])
                rows.append(dict(frame=frame,phase=roles[frame],contact_active=bool(config['contact_mask'][side][frame]),
                                 minimum_signed_m=float(distance[frame,index]),sample_index=index,
                                 robot_geom=int(geoms[index]),robot_body=int(bodies[index]),
                                 robot_point_body_local=points[index].tolist(),robot_point_object_local=posed[frame,index].tolist(),
                                 original_object_face=int(ctx.object_query_face_ids[query_face]),
                                 negative_sample_count=int((distance[frame]<0).sum()),sample_count=distance.shape[1]))
            report['hands'][side]=dict(worst=min(rows,key=lambda row:row['minimum_signed_m']),frames=rows,
                                      total_samples_per_frame=len(points),original_collision_geometry_preserved=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--trajectory',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--samples-per-geom',type=int,default=128)
    args=parser.parse_args();path=Path(args.config).resolve(strict=True)
    result=audit(json.loads(path.read_text()),args.trajectory,path.parent,args.device,args.samples_per_geom)
    _write(args.output,result)


if __name__=='__main__':
    main()
