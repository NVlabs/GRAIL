"""Independent readback of every native static triangle and USD transform."""
import argparse
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def validate(native, usd):
    from native_scene_contract import static_scene_payload
    scene, _ = static_scene_payload(native)
    stage = Usd.Stage.Open(str(usd))
    assert stage and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetDefaultPrim().GetPath().pathString == '/StaticContext'
    assert not stage.GetRootLayer().subLayerPaths
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 1 and bodies[0].GetPath().pathString == '/StaticContext'
    body = UsdPhysics.RigidBodyAPI(bodies[0])
    assert body.GetRigidBodyEnabledAttr().Get() is True
    assert body.GetKinematicEnabledAttr().Get() is True
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(meshes) == len(colliders) == len(scene['meshes'])
    rows = []
    for index, record in enumerate(scene['meshes']):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f'/StaticContext/NativeMesh_{index:04d}'))
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        source_vertices = native[record['vertices_key']]
        source_faces = native[record['faces_key']]
        assert np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3)
        assert vertices.shape == source_vertices.shape and np.array_equal(faces, source_faces)
        error = float(np.max(np.abs(vertices - source_vertices)))
        assert error <= 1e-6, ('World vertex precision exceeds admitted USD error', error)
        world = np.asarray(UsdGeom.Xformable(mesh.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        assert np.array_equal(world, np.eye(4)), 'Unexpected context world transformation'
        assert mesh.GetSubdivisionSchemeAttr().Get() == 'none'
        assert UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get() == 'none'
        assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get() is True
        rows.append({'native_name': record.get('name'), 'vertices': len(vertices),
                     'triangles': len(faces), 'faces_exact': True,
                     'world_vertex_max_error_m': error, 'world_transform_identity': True})
    return {'status': 'passed', 'meshes': rows, 'mesh_count': len(rows),
            'vertex_count': sum(row['vertices'] for row in rows),
            'triangle_count': sum(row['triangles'] for row in rows),
            'one_kinematic_root': True, 'collision_approximation': 'none',
            'physics_runtime_verified': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--usd', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = validate(dict(np.load(args.native, allow_pickle=False)), args.usd)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
