"""Vendored multi-mesh raycaster (originally from simple-raycaster).

Provides warp-based raycasting against multiple dynamic meshes for height-map
observations.  Only the USD path is kept; MuJoCo and voxelization helpers are
omitted since they are unused in gr00t RL training.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import torch
import trimesh
import warp as wp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def trimesh2wp(mesh: trimesh.Trimesh, device: str) -> wp.Mesh:
    """Convert a ``trimesh.Trimesh`` to a ``wp.Mesh``."""
    return wp.Mesh(
        points=wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(mesh.faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternion rotation to a vector.

    Args:
        quat: Quaternion in (w, x, y, z).  Shape ``(..., 4)``.
        vec: Vector in (x, y, z).  Shape ``(..., 3)``.

    Returns:
        Rotated vector, shape ``(..., 3)``.
    """
    xyz = quat[..., 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return vec - quat[..., 0:1] * t + xyz.cross(t, dim=-1)


# ---------------------------------------------------------------------------
# USD utilities
# ---------------------------------------------------------------------------


def _find_matching_prims(prim_path_regex: str, stage: "Usd.Stage"):
    if not prim_path_regex.startswith("^"):
        prim_path_regex = "^" + prim_path_regex
    if not prim_path_regex.endswith("$"):
        prim_path_regex = prim_path_regex + "$"
    pattern = re.compile(prim_path_regex)
    results = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if pattern.match(prim_path) is not None:
            results.append(prim)
    return results


def _get_mesh_prims_subtree(prim, predicate: Callable | None = None):
    if predicate is None:
        predicate = lambda _: True  # noqa: E731
    if prim.IsInstance():
        prim = prim.GetPrototype()
    mesh_prims = []
    all_prims = [prim]
    while all_prims:
        child_prim = all_prims.pop(0)
        if child_prim.GetTypeName() == "Mesh" and predicate(child_prim):
            mesh_prims.append(child_prim)
        all_prims += child_prim.GetChildren()
    return mesh_prims


def _usd2trimesh(prim):
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get())
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    return trimesh.Trimesh(vertices, faces.reshape(-1, 3))


def _get_trimesh_from_prim(prim):
    from pxr import Usd, UsdGeom

    mesh_prims = _get_mesh_prims_subtree(prim)
    if not mesh_prims:
        raise ValueError(f"No mesh primitives found in {prim.GetPath().pathString}")

    trimesh_list = []
    time = Usd.TimeCode.Default()
    parent_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)

    for mesh_prim in mesh_prims:
        mesh = _usd2trimesh(mesh_prim)
        transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time)
        if not mesh_prim.IsInPrototype():
            transform = transform * parent_transform.GetInverse()
        transform_np = np.array(transform).transpose()
        mesh.apply_transform(transform_np)
        trimesh_list.append(mesh)

    trimesh_combined: trimesh.Trimesh = trimesh.util.concatenate(trimesh_list)
    trimesh_combined.merge_vertices()
    return trimesh_combined


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _raycast_kernel(
    meshes: wp.array(dtype=wp.uint64),
    ray_starts: wp.array(dtype=wp.vec3, ndim=3),
    ray_dirs: wp.array(dtype=wp.vec3, ndim=3),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=3),
):
    i, mesh_id, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, mesh_id, ray_id] = max_dist
        return
    mesh = meshes[mesh_id]
    ray_start = ray_starts[i, mesh_id, ray_id]
    ray_dir = ray_dirs[i, mesh_id, ray_id]
    result = wp.mesh_query_ray(mesh, ray_start, ray_dir, max_dist)
    t = max_dist
    if result.result and result.t >= min_dist:
        t = result.t
    hit_distances[i, mesh_id, ray_id] = t


@wp.kernel(enable_backward=False)
def _transform_and_raycast_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=1),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=1),
    ray_starts_w: wp.array(dtype=wp.vec3, ndim=2),
    ray_dirs_w: wp.array(dtype=wp.vec3, ndim=2),
    cam_ids: wp.array(dtype=wp.int32, ndim=1),
    mesh_ids: wp.array(dtype=wp.int32, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=2),
):
    i, ray_id = wp.tid()
    cam_id = cam_ids[i]
    mesh_id = mesh_ids[i]

    quat_wxyz = mesh_quat_w[mesh_id]
    quat_xyzw = wp.quat(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
    ray_start_b = wp.quat_rotate_inv(quat_xyzw, ray_starts_w[cam_id, ray_id] - mesh_pos_w[mesh_id])
    ray_dir_b = wp.quat_rotate_inv(quat_xyzw, ray_dirs_w[cam_id, ray_id])

    result = wp.mesh_query_ray(meshes[mesh_id], ray_start_b, ray_dir_b, max_dist)
    t = max_dist
    if result.result and result.t >= min_dist:
        t = result.t
    hit_distances[i, ray_id] = t


# ---------------------------------------------------------------------------
# MultiMeshRaycaster
# ---------------------------------------------------------------------------


class MultiMeshRaycaster:
    """Raycaster that supports multiple dynamic meshes via warp."""

    def __init__(
        self,
        meshes: list[wp.Mesh | trimesh.Trimesh],
        device: str,
        mesh_names: list[str] | None = None,
    ):
        self.meshes = [
            mesh if isinstance(mesh, wp.Mesh) else trimesh2wp(mesh, device) for mesh in meshes
        ]
        self.meshes_array = wp.array(
            [mesh.id for mesh in self.meshes], device=device, dtype=wp.uint64
        )
        if mesh_names is not None and len(mesh_names) != len(meshes):
            raise ValueError("`mesh_names` length must match number of meshes.")
        self.mesh_names = list(mesh_names) if mesh_names is not None else None
        self.device = device

    def add_mesh(self, mesh, mesh_name: str | None = None):
        if isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh2wp(mesh, self.device)
        self.meshes.append(mesh)
        self.meshes_array = wp.array(
            [m.id for m in self.meshes], device=self.device, dtype=wp.uint64
        )
        if self.mesh_names is not None:
            if mesh_name is None:
                raise ValueError(
                    "`mesh_name` must be provided when the raycaster tracks mesh names."
                )
            self.mesh_names.append(mesh_name)

    @property
    def n_meshes(self):
        return len(self.meshes)

    def get_mesh_ids(
        self, mesh_filters: list[list[str]], device: str | torch.device
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Return per-camera mesh assignment tensors for ``raycast_fused``."""
        if self.mesh_names is None:
            raise ValueError("Mesh filters require `mesh_names` to be set.")
        n_cam = len(mesh_filters)
        mesh_names = self.mesh_names
        mesh_ids: list[list[int]] = [[] for _ in range(n_cam)]
        for cam_idx, patterns in enumerate(mesh_filters):
            if not patterns:
                continue
            for pattern in patterns:
                if pattern is None:
                    continue
                regex = re.compile(pattern)
                for mesh_idx, mesh_name in enumerate(mesh_names):
                    if regex.fullmatch(mesh_name):
                        mesh_ids[cam_idx].append(mesh_idx)

        n_mesh_per_cam = [len(ids) for ids in mesh_ids]
        cam_ids = [[cam_idx] * len(ids) for cam_idx, ids in enumerate(mesh_ids)]

        mesh_ids_flattened = torch.tensor(sum(mesh_ids, []), device=device, dtype=torch.int32)
        cam_ids_flattened = torch.tensor(sum(cam_ids, []), device=device, dtype=torch.int32)
        return n_mesh_per_cam, mesh_ids_flattened, cam_ids_flattened

    def raycast_fused(
        self,
        mesh_pos_w: torch.Tensor,
        mesh_quat_w: torch.Tensor,
        ray_starts_w: torch.Tensor,
        ray_dirs_w: torch.Tensor,
        n_mesh_per_cam: list[int],
        mesh_ids_flattened: torch.Tensor,
        cam_ids_flattened: torch.Tensor,
        min_dist: float = 0.0,
        max_dist: float = 100.0,
    ):
        """Fused per-camera raycast against posed meshes.

        Returns:
            hit_positions: ``(N, n_rays, 3)`` world-frame hit positions.
            hit_distances: ``(N, n_rays)`` distances.
        """
        n_rays = ray_dirs_w.shape[1]
        total_length = mesh_ids_flattened.shape[0]

        hit_distances = torch.empty(total_length, n_rays, device=ray_starts_w.device)
        wp.launch(
            _transform_and_raycast_kernel,
            dim=(total_length, n_rays),
            inputs=[
                self.meshes_array,
                wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                wp.from_torch(ray_starts_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(ray_dirs_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(cam_ids_flattened, dtype=wp.int32, return_ctype=True),
                wp.from_torch(mesh_ids_flattened, dtype=wp.int32, return_ctype=True),
                min_dist,
                max_dist,
            ],
            outputs=[wp.from_torch(hit_distances, dtype=wp.float32)],
            device=self.device,
            record_tape=False,
        )

        hit_distances = torch.split(hit_distances, n_mesh_per_cam, dim=0)
        hit_distances = [hd.min(dim=0).values for hd in hit_distances]
        hit_distances = torch.stack(hit_distances, dim=0)
        hit_positions = ray_starts_w + hit_distances.unsqueeze(-1) * ray_dirs_w
        return hit_positions, hit_distances

    @classmethod
    def from_prim_paths(
        cls,
        paths: list[str],
        stage: "Usd.Stage",
        device: str,
        simplify_factor: float = 0.0,
    ):
        if isinstance(simplify_factor, float):
            simplify_factor = [simplify_factor] * len(paths)
        if len(paths) != len(simplify_factor):
            raise ValueError("`simplify_factor` must match `paths` length.")

        meshes_wp = []
        mesh_names = []
        for path, factor in zip(paths, simplify_factor):
            prims = _find_matching_prims(path, stage)
            if not prims:
                raise ValueError(f"No prims found for path {path}")
            for prim in prims:
                mesh_combined = _get_trimesh_from_prim(prim)
                if factor > 0.0:
                    mesh_combined = mesh_combined.simplify_quadric_decimation(factor)
                meshes_wp.append(trimesh2wp(mesh_combined, device))
                mesh_names.append(prim.GetPath().pathString)
        return cls(meshes_wp, device, mesh_names=mesh_names)
