"""Finite unsigned original-object surface guidance for mounted joint fitting.

This is not a solid-distance field or a nonpenetration constraint. In particular,
deep containment and crossings between the finite robot samples require the
independent original-triangle screen, including any frozen robot geometry.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit.pose import _integer, _keys
from .grasp_io import number
from .repair_geometry import TriangleMesh
from .shared_grasp_scene import _surface_barrier_query


def _left_rotation_jacobian(vector):
    """Map additive rotation-vector changes to world angular changes."""
    x, y, z = vector
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle2 = float(np.dot(vector, vector))
    if angle2 < 1e-8:
        a = .5-angle2/24+angle2*angle2/720
        b = 1/6-angle2/120+angle2*angle2/5040
    else:
        angle = np.sqrt(angle2)
        a = (1-np.cos(angle))/angle2
        b = (angle-np.sin(angle))/(angle2*angle)
    return np.eye(3)+a*cross+b*(cross@cross)


class ObjectClearance:
    """Original robot samples queried against the original scaled object mesh.

    The first ``len(dofs)`` columns are native joint coordinates. When enabled,
    the next six columns are additive world translation and a rotation vector
    left-multiplying the supplied object rotation. No object decision variables
    or geometry are created or changed by this helper.
    """

    def __init__(self, ctx, joints, config=None):
        import mujoco
        config = {} if config is None else config
        _keys(config, set(), {"enabled", "weight", "margin_m", "samples_per_geom"},
              "object_clearance")
        enabled = config.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("object_clearance.enabled must be a boolean")
        self.weight = number(config.get("weight", 20.), "object clearance weight")
        self.margin = number(config.get("margin_m", .003), "object clearance margin", 0, .02)
        samples = _integer(config.get("samples_per_geom", 32), "object clearance samples", 4)
        if samples > 512:
            raise ValueError("object_clearance.samples_per_geom must not exceed 512")
        self.model, self.object = ctx.model, ctx.object
        self.active = enabled and self.weight > 0 and self.margin > 0
        self.robot = []
        self._cache_key, self._cached = None, None
        self._bounds = (self.object.vertices.min(0), self.object.vertices.max(0))
        self.report = dict(
            enabled=enabled, effective_enabled=self.active, weight=self.weight,
            margin_m=self.margin, samples_per_geom=samples, explicit_config=dict(config),
            considered_mesh_geoms=[], controlled_mesh_geoms=[],
            uncontrollable_mesh_geoms=[], omitted_nonmesh_geoms=[], robot_meshes=[],
            robot_samples_built=self.active,
            original_triangle_audit_required=True, motion_accepted=False,
            solid_sign_certified=False,
            semantics="Unsigned finite surface barrier, not penetration depth or collision acceptance.",
            limitations=[
                "Does not recover the safe side of existing intersections or deep containment.",
                "Exact-on-surface samples have no unique unsigned gradient and receive zero guidance.",
                "Finite robot samples can miss edge/triangle crossings; no subframe audit is performed here.",
                "Frozen robot meshes, nonmesh geoms and object/scene collisions require separate audit/repair.",
            ],
            object_mesh=dict(
                original_faces=len(self.object.faces), query_faces=len(self.object.faces),
                omitted_faces=0, degenerate_faces=int((~self.object.normal_valid).sum()),
                original_scaled_sha256=hashlib.sha256(
                    self.object.vertices.tobytes()+self.object.faces.tobytes()).hexdigest()))
        if not self.active:
            self.report["inactive_reason"] = "disabled" if not enabled else "zero_weight_or_margin"
        m = self.model
        moving_bodies = {int(m.jnt_bodyid[j]) for j in joints}
        for geom in range(m.ngeom):
            if m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                self.report["omitted_nonmesh_geoms"].append(geom)
                continue
            self.report["considered_mesh_geoms"].append(geom)
            body = int(m.geom_bodyid[geom]); ancestor = body
            affected = False
            while ancestor:
                if ancestor in moving_bodies:
                    affected = True
                    break
                ancestor = int(m.body_parentid[ancestor])
            self.report["controlled_mesh_geoms" if affected else "uncontrollable_mesh_geoms"].append(geom)
            if not affected or not self.active:
                continue
            mid = m.geom_dataid[geom]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            rotation = Rotation.from_quat(m.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()
            mesh = TriangleMesh(m.mesh_vert[va:va+vn]@rotation.T+m.geom_pos[geom],
                                m.mesh_face[fa:fa+fn])
            if not len(mesh.faces):
                raise ValueError(f"Original robot mesh geom {geom} has no triangles")
            vertices = mesh.vertices[np.unique(mesh.faces)]
            nv = min(samples//2, len(vertices))
            nf = min(samples-samples//2, len(mesh.faces))
            points = np.concatenate((vertices[np.linspace(0, len(vertices)-1, nv, dtype=int)],
                mesh.triangles[np.linspace(0, len(mesh.faces)-1, nf, dtype=int)].mean(1)))
            self.robot.append((geom, body, points))
            self.report["robot_meshes"].append(dict(
                geom=geom, body=body, samples=len(points), original_faces=len(mesh.faces),
                degenerate_faces=int((~mesh.normal_valid).sum()),
                original_sha256=hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest()))
        self.report["residual_count"] = sum(len(points) for _, _, points in self.robot)
        if self.active and not self.robot:
            self.active = False
            self.report.update(effective_enabled=False, inactive_reason="no_controllable_original_meshes")

    def _measure(self, data, object_rotation, object_translation, offset):
        rotation = Rotation.from_rotvec(offset[3:]).as_matrix()@object_rotation
        translation = np.asarray(object_translation)+offset[:3]
        key = rotation.tobytes()+translation.tobytes()+b"".join(
            data.xpos[b].tobytes()+data.xmat[b].tobytes() for _, b, _ in self.robot)
        if key == self._cache_key:
            return self._cached
        records = []
        lower, upper = self._bounds
        for geom, body, points in self.robot:
            world = points@data.xmat[body].reshape(3, 3).T+data.xpos[body]
            local = (world-translation)@rotation
            box_delta = np.maximum(np.maximum(lower-local, local-upper), 0)
            relevant = np.linalg.norm(box_delta, axis=1) < self.margin
            distance = np.full(len(world), np.inf); local_gradient = np.zeros_like(world)
            if relevant.any():
                distance[relevant], local_gradient[relevant] = _surface_barrier_query(
                    local[relevant], self.object, self.margin)
            gradient = local_gradient@rotation.T
            records.append((geom, body, world, world-translation, distance, gradient))
        self._cache_key, self._cached = key, records
        return records

    def evaluate(self, data, object_rotation, object_translation, offset, dofs,
                 nvariables, move_object=False, jacobian=False):
        import mujoco
        if not self.active:
            return np.zeros((0, nvariables)) if jacobian else np.zeros(0)
        if nvariables < len(dofs)+(6 if move_object else 0):
            raise ValueError("Object clearance variable count does not cover declared joint/object columns")
        offset = np.asarray(offset)
        residuals, rows = [], []
        rotation_jacobian = _left_rotation_jacobian(offset[3:]) if move_object and jacobian else None
        for _, body, world, centered, distance, gradient in self._measure(
                data, object_rotation, object_translation, offset):
            # Give each geom the same weight regardless of its sample count.
            scale = self.weight/np.sqrt(len(world))
            residuals.append(scale*np.maximum(self.margin-distance, 0))
            if jacobian:
                row = np.zeros((len(world), nvariables))
                for index in np.flatnonzero((distance < self.margin) & (distance > 1e-12)):
                    jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
                    mujoco.mj_jac(self.model, data, jp, jr, world[index], body)
                    row[index, :len(dofs)] = -scale*gradient[index]@jp[:, dofs]
                    if move_object:
                        start = len(dofs)
                        row[index, start:start+3] = scale*gradient[index]
                        row[index, start+3:start+6] = scale*np.cross(
                            centered[index], gradient[index])@rotation_jacobian
                rows.append(row)
        return np.vstack(rows) if jacobian else np.concatenate(residuals)

    def diagnostics(self, data, object_rotation, object_translation, offset):
        if not self.active:
            return dict(effective_enabled=False)
        records = self._measure(data, object_rotation, object_translation, np.asarray(offset))
        return dict(effective_enabled=True,
            samples_within_margin=sum(int((d < self.margin).sum()) for _, _, _, _, d, _ in records),
            unresolved_zero_distance_samples=sum(int((d <= 1e-12).sum()) for _, _, _, _, d, _ in records),
            max_sampled_barrier_m=max(float(np.maximum(self.margin-d, 0).max(initial=0))
                                     for _, _, _, _, d, _ in records))
