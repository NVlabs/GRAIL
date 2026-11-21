"""Additional original-surface clearance samples without diluting old costs.

An authored group adds weight * mean(hinge(clearance)^2) to an existing mesh
pair. Every point is bound to an exact compiled native robot triangle. These
queries retain the existing winding-distance heuristic and never certify
collision freedom, especially for open target surfaces.
"""
from __future__ import annotations

import hashlib
import numpy as np


def mesh_fingerprint(vertices, faces):
    """Canonical body-local geometry identity, including exact face order."""
    v, f = np.asarray(vertices, dtype='<f8'), np.asarray(faces, dtype='<i8')
    h = hashlib.sha256(b'soma_body_local_original_mesh.v1\0')
    for a in (v, f):
        h.update(np.asarray(a.shape, dtype='<i8').tobytes())
        h.update(a.tobytes(order='C'))
    return h.hexdigest()


def native_geom_fingerprint(vertices, faces, position, quaternion):
    """Exact native mesh/pose identity without platform-dependent matrix products.

    MuJoCo's ordered compiled vertices, triangles and geom pose are hashed before
    transforming to body coordinates. No quantization or tolerance is applied.
    Original asset/model byte hashes remain bound by the enclosing run inputs.
    """
    h = hashlib.sha256(b'soma_native_raw_geom_pose.v2\0')
    for values, dtype in ((vertices, '<f8'), (faces, '<i8'),
                          (position, '<f8'), (quaternion, '<f8')):
        a = np.asarray(values, dtype=dtype)
        h.update(np.asarray(a.shape, dtype='<i8').tobytes())
        h.update(a.tobytes(order='C'))
    return h.hexdigest()


def compile_witness_groups(groups, *, meshes, geom_to_mesh, existing_pairs,
                           native_identities=None):
    """Return extra point groups; original samples/normalization stay intact."""
    if not isinstance(groups, list) or not groups:
        raise ValueError('witness_groups must be a nonempty explicit list')
    allowed = {frozenset(pair) for pair in existing_pairs}
    result, seen = [], set()
    for group in groups:
        required = {'source_geom', 'target', 'mesh_sha256', 'weight', 'samples'}
        if (not isinstance(group, dict) or required-set(group)
                or set(group)-required-{'mesh_identity_schema'}):
            raise ValueError('Witness group requires source_geom, target, mesh_sha256, weight, samples')
        source_geom, target = group['source_geom'], group['target']
        if type(source_geom) is not int or source_geom not in geom_to_mesh:
            raise ValueError('Witness source must be an existing original robot mesh geom')
        a = geom_to_mesh[source_geom]
        if target == 'object':
            b = 0
        elif type(target) is int and target in geom_to_mesh:
            b = geom_to_mesh[target]
        else:
            raise ValueError('Witness target must be object or an existing original robot geom')
        if a == b or frozenset((a, b)) not in allowed or (a, b) in seen:
            raise ValueError('Witness group cannot add excluded or duplicate directed mesh pairs')
        weight = group['weight']
        if type(weight) not in (int, float) or not np.isfinite(weight) or weight <= 0:
            raise ValueError('Witness group weight must be finite positive')
        vertices, faces, _ = meshes[a]
        schema = group.get('mesh_identity_schema', 'compiled_body_mesh.v1')
        if schema == 'compiled_body_mesh.v1':
            digest = mesh_fingerprint(vertices, faces)
        elif schema == 'native_raw_geom_pose.v2':
            if native_identities is None or source_geom not in native_identities:
                raise ValueError('Native witness identity requires the original compiled mesh and geom pose')
            digest = native_identities[source_geom]
        else:
            raise ValueError('Unknown witness mesh identity schema')
        if group['mesh_sha256'] != digest:
            raise ValueError('Witness compiled original mesh identity changed')
        samples = group['samples']
        if not isinstance(samples, list) or not samples:
            raise ValueError('Witness group needs original triangle samples')
        points, sample_ids = [], set()
        for sample in samples:
            if not isinstance(sample, dict) or set(sample) != {'triangle_id', 'barycentric'}:
                raise ValueError('Witness sample requires triangle_id and barycentric')
            face = sample['triangle_id']
            if type(face) is not int or not 0 <= face < len(faces):
                raise ValueError('Witness triangle is outside the bound original mesh')
            bary = np.asarray(sample['barycentric'], dtype=float)
            if (bary.shape != (3,) or not np.isfinite(bary).all()
                    or np.any(bary < 0) or np.any(bary > 1)
                    or not np.isclose(bary.sum(), 1., atol=1e-8, rtol=0)):
                raise ValueError('Witness barycentric coordinates must be a convex original-face point')
            key = (face, *bary.tolist())
            if key in sample_ids:
                raise ValueError('Duplicate witness material sample')
            sample_ids.add(key)
            points.append(bary @ vertices[faces[face]])
        points = np.asarray(points)
        if not np.isfinite(points.astype(np.float32)).all():
            raise ValueError('Witness points exceed solver coordinate range')
        result.append(dict(source=a, target=b, points=points,
            query_weights=np.full(len(points), float(weight)/len(points)),
            report=dict(source_geom=source_geom, target=target, mesh_sha256=digest,
                mesh_identity_schema=schema,
                weight=float(weight), sample_count=len(points), samples=samples,
                coordinate_space='original compiled native mesh in body-local metres',
                source_geometry_preserved=True, old_queries_and_weights_preserved=True,
                temporal_scope='All original frames; unchanged collision enable schedule',
                limitation='Existing winding query over entire target; added coverage is not a safe-side or clearance proof.')))
        seen.add((a, b))
    return result
