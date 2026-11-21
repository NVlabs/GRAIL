"""CPU original-surface barriers for mounted grasp propagation.

Unsigned distances are deliberate: packed scene surfaces need not be closed or
consistently wound. These finite samples discourage approaching a surface from
either side. They cannot determine a safe side, resolve deep existing crossings,
or certify collision freedom. The independent original-mesh audit remains required.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit.pose import _integer, _keys
from .grasp_io import number
from .repair_geometry import TriangleMesh, _nearest_in_triangles
from .static_scene import static_scene_fingerprint, static_scene_payload


def _surface_barrier_query(points, mesh, radius):
    """Exact nearest original surface inside the barrier's finite support.

    A complete BVH radius search can return infinity outside this support: such
    samples contribute exactly zero. Avoid finding distant witnesses for every
    point inside a large table's bounding box. Degenerate faces remain included.
    """
    distances = np.full(len(points), np.inf)
    gradients = np.zeros_like(points)
    radius2 = radius*radius
    for index, point in enumerate(points):
        stack = [0]; candidates = []
        while stack:
            node = stack.pop()
            lower, upper, left, right, ids = mesh._tree[node]
            delta = np.maximum(np.maximum(lower-point, point-upper), 0)
            if np.dot(delta, delta) > radius2*(1+1e-12):
                continue
            if ids is None:
                stack.extend((left, right))
            else:
                candidates.append(ids)
        if not candidates:
            continue
        ids = np.concatenate(candidates)
        delta = np.maximum(np.maximum(mesh._lower[ids]-point, point-mesh._upper[ids]), 0)
        ids = ids[np.einsum("ij,ij->i", delta, delta) <= radius2*(1+1e-12)]
        if not len(ids):
            continue
        nearest, _, d2 = _nearest_in_triangles(point, mesh.triangles[ids],
                                              mesh.normal_valid[ids], mesh.normals[ids])
        winner = np.lexsort((ids, d2))[0]
        distance = float(np.sqrt(d2[winner]))
        if distance < radius:
            distances[index] = distance
            if distance > 1e-12:
                gradients[index] = (point-nearest[winner])/distance
    return distances, gradients


class SceneClearance:
    """Fixed robot samples, exact original scene triangles and native Jacobians."""

    def __init__(self, model, native, joints, config=None):
        import mujoco
        config = {} if config is None else config
        _keys(config, set(), {"enabled", "weight", "margin_m", "samples_per_geom"},
              "scene_clearance")
        enabled = config.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("scene_clearance.enabled must be a boolean")
        self.weight = number(config.get("weight", 20.), "scene clearance weight")
        self.margin = number(config.get("margin_m", .003), "scene clearance margin", 0, .02)
        samples = _integer(config.get("samples_per_geom", 32), "scene clearance samples", 4)
        if samples > 512:
            raise ValueError("scene_clearance.samples_per_geom must not exceed 512")
        manifest, arrays = static_scene_payload(native)
        self.active = enabled and self.weight > 0 and self.margin > 0 and manifest["status"] == "PRESENT"
        self.robot, self.scene, self.scene_bounds = [], [], []
        self._cache_key, self._cached = None, None
        self.report = dict(
            enabled=enabled, effective_enabled=self.active, scene_status=manifest["status"],
            weight=self.weight, margin_m=self.margin, samples_per_geom=samples,
            explicit_config=dict(config), scene_sha256=static_scene_fingerprint(manifest, arrays),
            controlled_mesh_geoms=[], uncontrollable_mesh_geoms=[], omitted_nonmesh_geoms=[],
            scene_meshes=[], robot_meshes=[], original_triangle_audit_required=True,
            motion_accepted=False, solid_sign_certified=False,
            semantics="Unsigned finite surface barrier, not penetration depth or collision acceptance.",
            limitations=[
                "Does not recover the safe side of existing intersections or deep containment.",
                "Exact-on-surface samples have no unique unsigned gradient and receive zero guidance.",
                "Finite robot samples can miss edge/triangle crossings; no subframe audit is performed here.",
                "Moving-object/scene collision and frozen robot meshes require separate audit/repair.",
            ])
        if manifest["status"] != "PRESENT":
            self.report["inactive_reason"] = "no_packed_scene" if manifest["status"] == "UNKNOWN" else "explicit_empty_scene"
            return
        if not self.active:
            self.report["inactive_reason"] = "disabled" if not enabled else "zero_weight_or_margin"
        moving_bodies = set(int(model.jnt_bodyid[j]) for j in joints)
        for geom in range(model.ngeom):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                self.report["omitted_nonmesh_geoms"].append(geom)
                continue
            body = int(model.geom_bodyid[geom]); parent = body
            affected = False
            while parent:
                if parent in moving_bodies:
                    affected = True
                    break
                parent = int(model.body_parentid[parent])
            key = "controlled_mesh_geoms" if affected else "uncontrollable_mesh_geoms"
            self.report[key].append(geom)
            if not affected:
                continue
            mid = model.geom_dataid[geom]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            rotation = Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()
            mesh = TriangleMesh(model.mesh_vert[va:va+vn] @ rotation.T + model.geom_pos[geom],
                                model.mesh_face[fa:fa+fn])
            if not len(mesh.faces):
                raise ValueError(f"Original robot mesh geom {geom} has no triangles")
            vertices = mesh.vertices[np.unique(mesh.faces)]
            nv, nf = min(samples // 2, len(vertices)), min(samples - samples // 2, len(mesh.faces))
            points = np.concatenate((vertices[np.linspace(0, len(vertices)-1, nv, dtype=int)],
                                     mesh.triangles[np.linspace(0, len(mesh.faces)-1, nf, dtype=int)].mean(1)))
            self.robot.append((geom, body, points))
            self.report["robot_meshes"].append(dict(
                geom=geom, body=body, samples=len(points), original_faces=len(mesh.faces),
                degenerate_faces=int((~mesh.normal_valid).sum()),
                original_sha256=hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest()))
        for descriptor in manifest["meshes"]:
            mesh = TriangleMesh(arrays[descriptor["vertices_key"]], arrays[descriptor["faces_key"]])
            self.scene.append(mesh)
            self.scene_bounds.append((mesh.vertices.min(0), mesh.vertices.max(0)))
            degenerate = int((~mesh.normal_valid).sum())
            self.report["scene_meshes"].append(dict(
                name=descriptor["name"], original_faces=len(mesh.faces), query_faces=len(mesh.faces),
                degenerate_faces=degenerate, omitted_faces=0,
                geometry_status="INCOMPLETE_DEGENERATE_GEOMETRY" if degenerate else "ORIGINAL_TRIANGLES",
                note="Unsigned queries retain degenerate segments/points; a negative intersection audit is inconclusive when faces are degenerate."))
        self.report["residual_count"] = sum(len(points) for _, _, points in self.robot)*len(self.scene)
        if not self.robot:
            self.active = False
            self.report.update(effective_enabled=False, inactive_reason="no_scene_controllable_original_meshes")

    def _measure(self, data):
        # The solver calls residual and Jacobian at the same native pose. Reuse
        # nearest witnesses without caching across a changed finger/body pose.
        key = b"".join(data.xpos[b].tobytes()+data.xmat[b].tobytes() for _, b, _ in self.robot)
        if key == self._cache_key:
            return self._cached
        records = []
        for geom, body, points in self.robot:
            world = points @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            for scene_id, mesh in enumerate(self.scene):
                # AABB distance is a safe lower bound on unsigned surface distance.
                lower, upper = self.scene_bounds[scene_id]
                box_delta = np.maximum(np.maximum(lower-world, world-upper), 0)
                relevant = np.linalg.norm(box_delta, axis=1) < self.margin
                distance = np.full(len(world), np.inf); gradient = np.zeros_like(world)
                if relevant.any():
                    distance[relevant], gradient[relevant] = _surface_barrier_query(world[relevant], mesh, self.margin)
                records.append((geom, body, scene_id, world, distance, gradient))
        self._cache_key, self._cached = key, records
        return records

    def evaluate(self, model, data, dofs, variable_count, jacobian=False):
        import mujoco
        if not self.active:
            return np.zeros((0, variable_count)) if jacobian else np.zeros(0)
        residuals, rows = [], []
        # Normalize each geom/scene pair independently; increasing mesh samples
        # does not implicitly multiply its objective weight.
        for _, body, _, world, distance, gradient in self._measure(data):
            scale = self.weight/np.sqrt(len(world))
            active = distance < self.margin
            residuals.append(scale*np.maximum(self.margin-distance, 0))
            if jacobian:
                row = np.zeros((len(world), variable_count))
                for index in np.flatnonzero(active & (distance > 1e-12)):
                    jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
                    mujoco.mj_jac(model, data, jp, jr, world[index], body)
                    row[index, :len(dofs)] = -scale*gradient[index] @ jp[:, dofs]
                rows.append(row)
        return np.vstack(rows) if jacobian else np.concatenate(residuals)

    def diagnostics(self, data):
        if not self.active:
            return dict(effective_enabled=False)
        records = self._measure(data)
        return dict(effective_enabled=True,
                    samples_within_margin=sum(int((d < self.margin).sum()) for _, _, _, _, d, _ in records),
                    unresolved_zero_distance_samples=sum(int((d <= 1e-12).sum()) for _, _, _, _, d, _ in records),
                    max_sampled_barrier_m=max(float(np.maximum(self.margin-d, 0).max(initial=0))
                                             for _, _, _, _, d, _ in records))
