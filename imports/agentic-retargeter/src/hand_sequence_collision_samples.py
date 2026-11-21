"""Append counterexample samples on immutable original native hand triangles.

These additional query points improve sampled collision coverage only. They do
not change original geometry, contact-pad memberships or material anchors, and
are not a whole-mesh or continuous-time collision certificate.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def append_collision_witnesses(points,bodies,geoms,rows,*,meshes,geom_bodies,allowed_geoms):
    if not isinstance(rows,list) or len(rows)>4096:
        raise ValueError('Collision witness samples must be a list of at most 4096 rows per hand')
    before=len(points);new_points=[];new_bodies=[];new_geoms=[];records=[]
    for index,row in enumerate(rows):
        if not isinstance(row,dict) or set(row)-{'geom','face','barycentric','body'} or not {'geom','face','barycentric'}<=set(row):
            raise ValueError('Collision witness row requires geom, face, barycentric and optional body only')
        geom=row['geom'];face=row['face']
        if type(geom) is not int or geom not in allowed_geoms or geom not in meshes or geom not in geom_bodies:
            raise ValueError('Collision witness geom must belong to the declared original hand meshes')
        mesh=meshes[geom];body=int(geom_bodies[geom])
        if 'body' in row and (type(row['body']) is not int or row['body']!=body):
            raise ValueError('Collision witness body does not match original geom/body identity')
        if type(face) is not int or not 0 <= face < len(mesh.faces):
            raise ValueError('Collision witness face must index original mesh-local triangle order')
        bary=np.asarray(row['barycentric'],dtype=float)
        if bary.shape!=(3,) or not np.isfinite(bary).all() or np.any(bary<0) or np.any(bary>1) or not np.isclose(bary.sum(),1,atol=1e-8,rtol=0):
            raise ValueError('Collision witness barycentric coordinates must be finite and in the unit simplex')
        if not mesh.normal_valid[face]:
            raise ValueError('Collision witness must reference a nondegenerate original triangle')
        # Normalize only accepted floating-point sum tolerance; record both.
        normalized=bary/bary.sum();point=normalized@mesh.triangles[face]
        new_points.append(point);new_bodies.append(body);new_geoms.append(geom)
        records.append(dict(sample_index=before+index,geom=geom,body=body,face=face,
            original_vertex_indices=mesh.faces[face].tolist(),input_barycentric=bary.tolist(),
            barycentric=normalized.tolist(),body_local_point=point.tolist(),collision_queries_only=True))
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    metadata=dict(input_sha256=digest,original_sample_count=before,added_sample_count=len(rows),
        total_sample_count=before+len(rows),samples=records,
        contact_memberships_unchanged=True,anchor_selection_unchanged=True,original_geometry_unchanged=True,
        frame_policy='Evaluate at every original frame and configured collision collocation fraction.',
        limitation='Additional original-surface point queries are incomplete; retain independent whole-triangle audits.')
    if not rows:
        return points,bodies,geoms,metadata
    return (np.concatenate((points,np.asarray(new_points)),0),
            np.concatenate((bodies,np.asarray(new_bodies,dtype=np.asarray(bodies).dtype))),
            np.concatenate((geoms,np.asarray(new_geoms,dtype=np.asarray(geoms).dtype))),metadata)
