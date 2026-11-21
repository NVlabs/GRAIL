"""Original-triangle component-union occupancy with piecewise Torch gradients.

Only exact-coordinate aliases are used to establish topology. Query triangles,
face identities and winding remain original. Open, inconsistently wound, or
non-positive-volume components are rejected, never silently discarded/repaired.

Inside is negative. The minimum of per-shell signed distances is an occupancy
field for their solid union; inside overlapping shells it is NOT necessarily the
exact distance to the external union boundary. Watertightness/winding checks do
not prove absence of self-intersections. Independent collision review is needed.
"""

import math

import numpy as np
import torch
import trimesh
import warp as wp


@wp.kernel
def _winding_closest(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_distance: float,
    closest: wp.array(dtype=wp.vec3),
    signs: wp.array(dtype=float),
    faces: wp.array(dtype=int),
    found: wp.array(dtype=int),
):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, points[i], max_distance, 4.0, 0.5)
    found[i] = int(q.result)
    if q.result:
        closest[i] = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        signs[i] = q.sign
        faces[i] = q.face


class OriginalMeshPenetration:
    """Build reusable original triangle BVHs; CPU is the tested default.

    CUDA uses Warp's Torch interop; both devices require numerical review for
    the target original geometry. No mesh vertices/scales are differentiable. Point and
    object inverse-pose gradients pass through the caller's query tensor.
    """

    def __init__(self, vertices, faces, *, device="cpu", query_scale=1000.0):
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) < 3
            or not np.issubdtype(vertices.dtype, np.floating)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Original vertices must be finite floating [V,3]")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or len(faces) < 4
            or not np.issubdtype(faces.dtype, np.integer)
            or np.issubdtype(faces.dtype, np.bool_)
        ):
            raise ValueError("Original faces must be integer [F,3] triangles")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("Original triangle index out of range")
        if (
            isinstance(query_scale, bool)
            or not isinstance(query_scale, (int, float))
            or not math.isfinite(query_scale)
            or query_scale <= 0
        ):
            raise ValueError("query_scale must be positive and finite")
        self.vertices = vertices.astype(np.float64, copy=True)
        self.faces = faces.astype(np.int64, copy=True)
        tri = self.vertices[self.faces]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
        if not np.isfinite(area).all() or (area == 0).any():
            raise ValueError("Original mesh contains degenerate triangles")
        # Exact aliases are solely a topology map; never replace query geometry.
        canonical_vertices, aliases = np.unique(self.vertices, axis=0, return_inverse=True)
        canonical = trimesh.Trimesh(canonical_vertices, aliases[self.faces], process=False)
        components = trimesh.graph.connected_components(
            canonical.face_adjacency, nodes=np.arange(len(self.faces)), min_len=1
        )
        self.component_face_ids = sorted(
            (np.sort(np.asarray(ids, dtype=np.int64)) for ids in components),
            key=lambda ids: (-len(ids), int(ids[0])),
        )
        self.topology = []
        for index, ids in enumerate(self.component_face_ids):
            shell = trimesh.Trimesh(canonical_vertices, aliases[self.faces[ids]], process=False)
            record = {
                "component": index,
                "original_face_count": len(ids),
                "watertight": bool(shell.is_watertight),
                "winding_consistent": bool(shell.is_winding_consistent),
                "signed_volume_m3": float(shell.volume),
            }
            if (
                not record["watertight"]
                or not record["winding_consistent"]
                or not math.isfinite(record["signed_volume_m3"])
                or record["signed_volume_m3"] <= 0
            ):
                raise ValueError(
                    f"Undefined solid component; no repair or omission allowed: {record}"
                )
            self.topology.append(record)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU/CUDA Warp devices are supported")
        self.warp_device = "cpu" if self.device.type == "cpu" else f"cuda:{self.device.index or 0}"
        self.query_scale = float(query_scale)
        self.max_mesh_coordinate = float(np.abs(self.vertices).max())
        scaled = (self.vertices * self.query_scale).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled original vertices exceed query precision/range")
        self._meshes = []
        self._face_maps = []
        for ids in self.component_face_ids:
            self._meshes.append(
                wp.Mesh(
                    points=wp.array(scaled, dtype=wp.vec3, device=self.warp_device),
                    indices=wp.array(
                        self.faces[ids].astype(np.int32).ravel(),
                        dtype=wp.int32,
                        device=self.warp_device,
                    ),
                    support_winding_number=True,
                )
            )
            self._face_maps.append(torch.as_tensor(ids, dtype=torch.long, device=self.device))
        if self.device.type == "cuda":
            # BVHs are built on Warp's initialization stream. Subsequent queries
            # explicitly share the caller's Torch stream.
            wp.synchronize_device(self.warp_device)

    def query(self, query_local, *, return_details=False):
        """Return signed union field with gradients through query coordinates.

        Shapes are arbitrary [...,3]. Closest face/point and winding sign are
        detached per query; Torch differentiates signed Euclidean point distance.
        This is piecewise valid away from nearest-feature/sign-switch boundaries.
        At exact surface coincidence the Euclidean norm has a zero subgradient.
        """
        if (
            not torch.is_tensor(query_local)
            or query_local.ndim < 2
            or query_local.shape[-1] != 3
            or query_local.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("Queries must be float32/64 [...,3] Torch tensors")
        if query_local.device.type != self.device.type or (
            self.device.type == "cuda"
            and (query_local.device.index or 0) != (self.device.index or 0)
        ):
            raise ValueError("Query device must match the original mesh device")
        if not torch.isfinite(query_local).all():
            raise ValueError("Query points must be finite")
        shape = query_local.shape[:-1]
        flat = query_local.reshape(-1, 3)
        if not len(flat):
            signed = query_local.sum(dim=-1) * 0
            return (
                {
                    "signed_distance_m": signed,
                    "inside": signed < 0,
                    "component_signed_distance_m": signed.unsqueeze(0).expand(
                        len(self._meshes), *shape
                    ),
                }
                if return_details
                else signed
            )
        detached = (flat.detach().to(torch.float32) * self.query_scale).contiguous()
        if not torch.isfinite(detached).all():
            raise ValueError("Scaled points exceed query precision/range")
        search = (
            (float(flat.detach().abs().max()) + self.max_mesh_coordinate + 1.0)
            * self.query_scale
            * 4
        )
        if not math.isfinite(search) or search > np.finfo(np.float32).max:
            raise ValueError("Query search radius exceeds Warp float range")
        points = wp.from_torch(detached, dtype=wp.vec3)
        component_distances, component_closest, component_faces = [], [], []
        for mesh, face_map in zip(self._meshes, self._face_maps):
            closest = wp.empty(len(flat), dtype=wp.vec3, device=self.warp_device)
            signs = wp.empty(len(flat), dtype=float, device=self.warp_device)
            faces = wp.empty(len(flat), dtype=int, device=self.warp_device)
            found = wp.empty(len(flat), dtype=int, device=self.warp_device)
            if self.device.type == "cuda":
                stream = wp.stream_from_torch(torch.cuda.current_stream(query_local.device))
                with wp.ScopedStream(stream):
                    wp.launch(
                        _winding_closest,
                        dim=len(flat),
                        inputs=[mesh.id, points, search],
                        outputs=[closest, signs, faces, found],
                        device=self.warp_device,
                    )
            else:
                wp.launch(
                    _winding_closest,
                    dim=len(flat),
                    inputs=[mesh.id, points, search],
                    outputs=[closest, signs, faces, found],
                    device=self.warp_device,
                )
            if not bool((wp.to_torch(found) != 0).all()):
                raise ValueError(
                    "Original mesh nearest query failed; refusing undefined sign/distance"
                )
            constant_closest = (
                wp.to_torch(closest).to(query_local.dtype) / self.query_scale
            ).detach()
            constant_sign = wp.to_torch(signs).to(query_local.dtype).clone().detach()
            signed = torch.linalg.vector_norm(flat - constant_closest, dim=-1) * constant_sign
            component_distances.append(signed.reshape(shape))
            if return_details:
                component_closest.append(constant_closest.reshape(*shape, 3))
                component_faces.append(face_map[wp.to_torch(faces).long()].reshape(shape))
        all_distances = torch.stack(component_distances)
        field, winner = all_distances.min(dim=0)
        if not return_details:
            return field
        return {
            "signed_distance_m": field,
            "inside": field < 0,
            "component_signed_distance_m": all_distances,
            "winning_component": winner,
            "closest_original_surface_point_m": torch.stack(component_closest),
            "closest_original_face_id": torch.stack(component_faces),
        }

    def penetration_loss(self, query_local):
        """Half mean plus half worst squared inside-depth; exterior is unpenalized."""
        signed = self.query(query_local)
        if not signed.numel():
            return signed.sum() * 0
        depth = torch.relu(-signed)
        return 0.5 * (depth.square().mean() + depth.square().max())
