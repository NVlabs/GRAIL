"""CPU candidate: exact native world triangle context, without inferred geometry.

This module changes data conversion only. PhysX cooking/runtime cost is untested.
Static supervision uses unsigned triangle-surface distance, including for open
meshes; it makes no unproved solid-interior/penetration classification.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class NativeMesh:
    metadata: dict
    vertices: np.ndarray
    faces: np.ndarray

    def as_dict(self):
        return {'native_metadata': self.metadata,
                'vertex_count': len(self.vertices), 'triangle_count': len(self.faces),
                'vertices_array_sha256': hashlib.sha256(self.vertices.tobytes()).hexdigest(),
                'vertices_dtype': str(self.vertices.dtype),
                'faces_array_sha256': hashlib.sha256(self.faces.tobytes()).hexdigest(),
                'faces_dtype': str(self.faces.dtype)}


def native_meshes(native):
    from native_scene_contract import static_scene_payload
    if 'static_scene_json' not in native:
        raise ValueError('Explicit static_scene_json required; legacy absence remains unresolved')
    scene, _ = static_scene_payload(native)
    meshes = []
    keys = set()
    for record in scene['meshes']:
        vertices_key, faces_key = record['vertices_key'], record['faces_key']
        if vertices_key in keys or faces_key in keys:
            raise ValueError('Repeated static mesh array key')
        keys.update((vertices_key, faces_key))
        vertices, faces = np.asarray(native[vertices_key]), np.asarray(native[faces_key])
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all()):
            raise ValueError('Static vertices must be finite, nonempty [N,3]')
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Static faces must be valid original triangle indices')
        meshes.append(NativeMesh(dict(record), vertices.copy(), faces.copy()))
    # Embedded positions are already in the native world frame. Do not re-center,
    # scale, merge vertices, simplify, infer a slab, or add a mesh from its name.
    return tuple(meshes), scene


def static_context_record(meshes):
    return {'kind': 'exact_native_world_triangle_meshes_v2',
            'meshes': [mesh.as_dict() for mesh in meshes],
            'coordinate_policy': 'native_world_vertices_no_transform',
            'collision_approximation': 'none', 'geometry_inference': False}


def write_static_context_usd(path, meshes):
    from pxr import Usd, UsdGeom, UsdPhysics
    path = Path(path)
    if path.exists():
        raise ValueError('Refusing to replace a static USD')
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/StaticContext').GetPrim()
    stage.SetDefaultPrim(root)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)
    for index, source in enumerate(meshes):
        mesh = UsdGeom.Mesh.Define(stage, f'/StaticContext/NativeMesh_{index:04d}')
        mesh.CreatePointsAttr(source.vertices.astype(np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(source.faces))
        mesh.CreateFaceVertexIndicesAttr(source.faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([(0.5, 0.35, 0.2)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        mesh.GetPrim().SetCustomDataByKey('contactfirst:nativeMeshRecord',
                                        json.dumps(source.as_dict(), sort_keys=True))
    stage.GetRootLayer().Save()


def points_to_meshes(points, meshes, chunk_size=4096):
    """Certified nearest triangles, all supplied vertices/faces, no bbox contact."""
    from gear_sonic.rp2 import surface_relation
    if not np.isinf(surface_relation._FAR_FIELD_M):
        raise ValueError('Set RP2_SURFACE_FAR_FIELD_M=inf before importing pinned surface_relation')
    closest_points_to_mesh = surface_relation.closest_points_to_mesh
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    result = np.full(len(flat), np.inf, np.float32)
    for source in meshes:
        for start in range(0, len(flat), chunk_size):
            stop = min(start + chunk_size, len(flat))
            values = closest_points_to_mesh(flat[start:stop], source.vertices, source.faces,
                                           compute_signed_distance=False)[0]
            result[start:stop] = np.minimum(result[start:stop], values)
    return result.reshape(points.shape[:-1])


def add_static_triangle_contacts(fields, audit, *, meshes, fk, model, temporal):
    from gear_sonic.rp2 import g1_collision_geometry as geom
    from gear_sonic.rp2.discrete_contact import discretize_contact_distances
    if not meshes:
        return fields, audit
    distance = fields['reference_contact_distance_m'].copy()
    valid = fields['reference_contact_geometry_valid'].copy()
    axis = fields['reference_contact_entity_classes'].index('static_context')
    for index, group in enumerate(fields['reference_contact_body_groups']):
        points = geom._group_surface_world(fk, model, group)
        distance[:, axis, index] = points_to_meshes(points, meshes).min(axis=1)
        valid[:, axis, index] = True
    provenance = fields['reference_contact_provenance']
    state = discretize_contact_distances(
        distance, geometry_valid=valid,
        required_threshold_m=provenance['required_threshold_m'],
        forbidden_threshold_m=provenance['forbidden_threshold_m'],
        minimum_required_frames=temporal.minimum_required_frames,
        close_gap_frames=temporal.close_gap_frames,
        transition_neutral_frames=temporal.transition_neutral_frames)
    fields.update(reference_contact_distance_m=distance,
                  reference_contact_geometry_valid=valid,
                  reference_contact_state=state.astype(np.int8))
    provenance['entity_geometry']['static_context'] = {
        **static_context_record(meshes), 'valid': True,
        'label_policy': 'minimum_unsigned_exact_triangle_surface_distance',
        'solid_inside_classification': 'not_assumed_for_arbitrary_native_mesh'}
    provenance['distance_query'] = (
        'pinned_active_object_triangle_distance; all_training_body_surface_points_to_'
        'exact_native_static_triangles_unsigned; absolute_distance_to_ground_plane')
    audit.update(valid=int(np.count_nonzero(valid)), required=int(np.count_nonzero(state == 1)),
                 neutral=int(np.count_nonzero(state == 0)), forbidden=int(np.count_nonzero(state == -1)))
    audit['state_fractions'] = {key: audit[key] / state.size for key in ('required', 'neutral', 'forbidden')}
    audit['valid_fraction'] = audit['valid'] / state.size
    audit['static_mesh_count'] = len(meshes)
    audit['static_triangle_count'] = sum(len(mesh.faces) for mesh in meshes)
    return fields, audit
