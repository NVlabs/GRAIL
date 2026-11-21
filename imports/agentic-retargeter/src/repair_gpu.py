"""Optional persistent Warp queries for original-triangle repair proposals.

This is a nearest-surface proposal backend, not a collision acceptance test or
Newton dynamics solver. Warp searches the original topology using float32 local
vertices. The selected original face is then evaluated in float64. Equidistant
face IDs and normals can differ from CPU queries. Winding-facet halfspaces are
not inside/outside labels or penetration depths, including for closed meshes.

Importing this module does not initialize Warp or CUDA. Construct the explicit
backend in a worker; missing Warp/devices raise rather than silently changing
backends. BVHs and query buffers persist for the lifetime of the instance.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping

import numpy as np

_KERNELS = {}
_BROADPHASE_KERNELS = {}


def _query_kernel(wp):
    if wp in _KERNELS:
        return _KERNELS[wp]

    @wp.kernel
    def query_nearest(
        points: wp.array(dtype=wp.vec3),
        mesh_ids: wp.array(dtype=wp.uint64),
        radius: float,
        faces: wp.array(dtype=int),
    ):
        i = wp.tid()
        query = wp.mesh_query_point_no_sign(mesh_ids[i], points[i], radius)
        faces[i] = -1
        if query.result:
            faces[i] = query.face

    _KERNELS[wp] = query_nearest
    return query_nearest


def _broadphase_kernels(wp):
    if wp in _BROADPHASE_KERNELS:
        return _BROADPHASE_KERNELS[wp]

    @wp.kernel
    def count_candidates(
        bvh: wp.uint64,
        lower: wp.array(dtype=wp.vec3),
        upper: wp.array(dtype=wp.vec3),
        target_begin: wp.array(dtype=int),
        target_end: wp.array(dtype=int),
        counts: wp.array(dtype=int),
    ):
        i = wp.tid()
        query = wp.bvh_query_aabb(bvh, lower[i], upper[i])
        index = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
        count = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
        while wp.bvh_query_next(query, index):
            if index >= target_begin[i] and index < target_end[i]:
                count += 1
        counts[i] = count

    @wp.kernel
    def fill_candidates(
        bvh: wp.uint64,
        lower: wp.array(dtype=wp.vec3),
        upper: wp.array(dtype=wp.vec3),
        target_begin: wp.array(dtype=int),
        target_end: wp.array(dtype=int),
        offsets: wp.array(dtype=int),
        target_faces: wp.array(dtype=int),
    ):
        i = wp.tid()
        query = wp.bvh_query_aabb(bvh, lower[i], upper[i])
        index = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
        offset = offsets[i]
        while wp.bvh_query_next(query, index):
            if index >= target_begin[i] and index < target_end[i]:
                target_faces[offset] = index - target_begin[i]
                offset += 1

    _BROADPHASE_KERNELS[wp] = count_candidates, fill_candidates
    return count_candidates, fill_candidates


def _rigid_matrices(value, count):
    transforms = np.asarray(value, dtype=np.float64)
    if (
        transforms.shape != (count, 4, 4)
        or not np.isfinite(transforms).all()
        or not np.allclose(transforms[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-12)
        or not np.allclose(
            transforms[:, :3, :3] @ transforms[:, :3, :3].swapaxes(1, 2),
            np.eye(3),
            rtol=0,
            atol=1e-8,
        )
        or not np.allclose(np.linalg.det(transforms[:, :3, :3]), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError(
            "instances need proper rigid 4 x 4 transforms; bake scale in vertices"
        )
    return transforms


def _affine_matrices(value, count):
    """World bounds accept native affine arithmetic without normalizing it."""
    transforms = np.asarray(value, dtype=np.float64)
    if (
        transforms.shape != (count, 4, 4)
        or not np.isfinite(transforms).all()
        or not np.allclose(transforms[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-12)
    ):
        raise ValueError("broadphase transforms must be finite affine 4 x 4 matrices")
    return transforms


def _outward32(value, direction):
    with np.errstate(over="ignore", invalid="ignore"):
        cast = np.asarray(value).astype(np.float32)
        result = np.nextafter(cast, np.float32(direction))
    if not np.isfinite(result).all():
        raise ValueError("world bounds exceed supported float32 range")
    return result


def _points(value, label):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(label + " must be finite N x 3")
    return result


def _refine(points, triangles):
    """Closest points on selected original faces, recomputed in float64."""
    candidates, weights = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, b] - triangles[:, a]
        denominator = np.einsum("ij,ij->i", edge, edge)
        fraction = np.clip(
            np.einsum("ij,ij->i", points - triangles[:, a], edge) / denominator,
            0,
            1,
        )
        candidates.append(triangles[:, a] + fraction[:, None] * edge)
        bary = np.zeros((len(points), 3))
        bary[:, a], bary[:, b] = 1 - fraction, fraction
        weights.append(bary)
    candidates, weights = np.stack(candidates, 1), np.stack(weights, 1)
    choice = np.argmin(np.sum((candidates - points[:, None]) ** 2, axis=2), axis=1)
    row = np.arange(len(points))
    closest, bary = candidates[row, choice], weights[row, choice]
    e0, e1 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = np.cross(e0, e1)
    normal = cross / np.linalg.norm(cross, axis=1)[:, None]
    relative = points - triangles[:, 0]
    projection = points - np.sum(relative * normal, axis=1)[:, None] * normal
    relative = projection - triangles[:, 0]
    area2 = np.sum(cross * cross, axis=1)
    v = np.sum(np.cross(relative, e1) * cross, axis=1) / area2
    w = np.sum(np.cross(e0, relative) * cross, axis=1) / area2
    projected_bary = np.column_stack((1 - v - w, v, w))
    interior = np.all(projected_bary >= 0, axis=1)
    closest[interior], bary[interior] = projection[interior], projected_bary[interior]
    return closest, bary, normal


class WarpMeshQueries:
    """Persistent original-triangle BVHs with batched rigid mesh instances.

    ``meshes`` maps unique names to objects exposing ``vertices`` and ``faces``
    (for example ``TriangleMesh``), or to ``(vertices, faces)`` tuples. Geometry
    must be nondegenerate after float32 conversion. Nearest-query instances
    require proper rigid matrices with scale baked into bound vertices. The
    independent broadphase accepts finite affine world transforms without
    changing them, since it uses transformed triangle bounds directly.
    Input arrays are copied. This mutable worker object is not thread safe.
    """

    def __init__(self, meshes: Mapping, *, device="cuda:0"):
        started = time.perf_counter()
        if not isinstance(meshes, Mapping) or not meshes:
            raise ValueError("meshes must be a nonempty name mapping")
        # Validate before importing or initializing the optional runtime.
        originals = {}
        for name, value in meshes.items():
            if not isinstance(name, str) or not name:
                raise ValueError("mesh names must be nonempty strings")
            vertices, faces = (
                (value.vertices, value.faces) if hasattr(value, "vertices") else value
            )
            vertices = _points(vertices, "vertices").copy()
            faces = np.asarray(faces)
            if (
                faces.ndim != 2
                or faces.shape[1] != 3
                or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0
                or faces.max() >= len(vertices)
                or faces.max() > np.iinfo(np.int32).max
            ):
                raise ValueError("faces must be nonempty valid integer F x 3")
            faces = faces.astype(np.int32, copy=True)
            original_triangles = vertices[faces]
            e0 = original_triangles[:, 1] - original_triangles[:, 0]
            e1 = original_triangles[:, 2] - original_triangles[:, 0]
            original_area = np.linalg.norm(np.cross(e0, e1), axis=1)
            original_scale = np.linalg.norm(e0, axis=1) * np.linalg.norm(e1, axis=1)
            if np.any(original_area <= 64 * np.finfo(float).eps * original_scale):
                raise ValueError(
                    "Original mesh has degenerate faces; use CPU audit with explicit incomplete status"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                cast = vertices.astype(np.float32)
                triangles = cast[faces].astype(np.float64)
                area = np.linalg.norm(
                    np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    ),
                    axis=1,
                )
            if not np.isfinite(cast).all() or not np.all(area > 0):
                raise ValueError(
                    "Warp query mesh is nonfinite or degenerate in float32; use CPU audit"
                )
            sha = hashlib.sha256(
                vertices.tobytes() + faces.astype(np.int64).tobytes()
            ).hexdigest()
            vertices.setflags(write=False)
            faces.setflags(write=False)
            originals[name] = {
                "vertices": vertices,
                "faces": faces,
                "sha256": sha,
                "max_float32_vertex_error": float(
                    np.max(np.abs(cast.astype(np.float64) - vertices))
                ),
            }
        try:
            import warp as wp
        except ImportError as exc:
            raise RuntimeError(
                "Warp is optional; install it in the configured worker"
            ) from exc
        self.wp = wp
        self.device = wp.get_device(device)
        self.kernel = _query_kernel(wp)
        self.meshes = originals
        self.gpu_meshes = {}
        for name, mesh in originals.items():
            vertices = wp.array(
                mesh["vertices"].astype(np.float32), dtype=wp.vec3, device=self.device
            )
            faces = wp.array(mesh["faces"].ravel(), dtype=wp.int32, device=self.device)
            self.gpu_meshes[name] = wp.Mesh(points=vertices, indices=faces)
        self.capacity = 0
        self.calls = 0
        self.last_timing = {}
        self.last_broadphase_timing = {}
        self._broadphase = None
        wp.synchronize_device(self.device)
        self.build_seconds = time.perf_counter() - started

    def _reserve(self, count):
        if count <= self.capacity:
            return
        self.capacity = max(count, max(256, self.capacity * 2))
        wp = self.wp
        self._points = wp.empty(self.capacity, dtype=wp.vec3, device=self.device)
        self._mesh_ids = wp.empty(self.capacity, dtype=wp.uint64, device=self.device)
        self._faces = wp.empty(self.capacity, dtype=wp.int32, device=self.device)
        # Persistent pinned host buffers avoid allocating/uploading many tiny arrays.
        self._host_points = wp.empty(
            self.capacity, dtype=wp.vec3, device="cpu", pinned=self.device.is_cuda
        )
        self._host_ids = wp.empty(
            self.capacity, dtype=wp.uint64, device="cpu", pinned=self.device.is_cuda
        )
        self._host_faces = wp.empty(
            self.capacity, dtype=wp.int32, device="cpu", pinned=self.device.is_cuda
        )

    def query(self, points, mesh_name, *, max_distance=np.inf):
        """Convenience query in a single mesh's original local coordinates."""
        points = _points(points, "points")
        return self.query_instances(
            points,
            [mesh_name],
            np.eye(4)[None],
            np.zeros(len(points), dtype=np.int32),
            max_distance=max_distance,
        )

    def query_instances(
        self,
        points_world,
        instance_mesh_names,
        instance_transforms,
        point_instances,
        *,
        max_distance=np.inf,
    ):
        """Query N points against K instances without rebuilding any mesh BVH.

        Each ``point_instances[i]`` selects one target instance. Result arrays
        align with input rows. Misses have ``found=False``, ``face_ids=-1`` and
        NaN witnesses/gradients; they do not imply clearance. Gradients are with
        respect to query translation for the selected nearest-surface branch;
        distance gradients are unknown at zero distance and feature switches.
        ``normal_valid`` describes a winding normal, never an outward normal.

        The returned material ``points_local``/``barycentric`` can be frozen in
        an outer repair iteration. A safe-side constraint must separately bind
        a nonintersecting reference pose. No joint update occurs in this class.
        """
        started = time.perf_counter()
        points = _points(points_world, "points_world")
        names = list(instance_mesh_names)
        if not names or any(name not in self.meshes for name in names):
            raise ValueError("every instance needs a known original mesh name")
        transforms = _rigid_matrices(instance_transforms, len(names))
        instances = np.asarray(point_instances)
        if (
            instances.shape != (len(points),)
            or not np.issubdtype(instances.dtype, np.integer)
            or (
                len(instances)
                and (instances.min() < 0 or instances.max() >= len(names))
            )
        ):
            raise ValueError("point_instances must be valid integer N instance indices")
        radius = float(max_distance)
        if np.isnan(radius) or radius <= 0 or np.isneginf(radius):
            raise ValueError("max_distance must be positive")
        count = len(points)
        result = {
            "found": np.zeros(count, dtype=bool),
            "face_ids": np.full(count, -1, dtype=np.int64),
            "normal_valid": np.zeros(count, dtype=bool),
            "distance_gradient_valid": np.zeros(count, dtype=bool),
            "distances": np.full(count, np.nan),
            "signed_halfspace": np.full(count, np.nan),
        }
        for key in (
            "points",
            "points_local",
            "barycentric",
            "normals",
            "normals_local",
            "distance_gradient",
            "squared_distance_gradient",
        ):
            result[key] = np.full((count, 3), np.nan)
        if not count:
            return result
        rotations, translations = (
            transforms[instances, :3, :3],
            transforms[instances, :3, 3],
        )
        # Convert to mesh-local coordinates in float64, avoiding float32 loss at
        # distant world origins before the GPU nearest-feature search.
        local = np.einsum("nji,nj->ni", rotations, points - translations)
        with np.errstate(over="ignore", invalid="ignore"):
            local32 = local.astype(np.float32)
        if not np.isfinite(local32).all():
            raise ValueError("query coordinates exceed float32 range")
        self._reserve(count)
        mesh_ids = np.array(
            [self.gpu_meshes[name].id for name in names], dtype=np.uint64
        )
        self._host_points.numpy()[:count] = local32
        self._host_ids.numpy()[:count] = mesh_ids[instances]
        wp = self.wp
        prepared = time.perf_counter()
        wp.copy(self._points, self._host_points, count=count)
        wp.copy(self._mesh_ids, self._host_ids, count=count)
        # A small broadening avoids turning a boundary roundoff into a miss.
        search_radius = min(
            1e30, radius + 16 * np.finfo(np.float32).eps * max(1, np.abs(local).max())
        )
        wp.launch(
            self.kernel,
            dim=count,
            inputs=[self._points, self._mesh_ids, float(search_radius)],
            outputs=[self._faces],
            device=self.device,
        )
        wp.copy(self._host_faces, self._faces, count=count)
        wp.synchronize_device(self.device)
        face_ids = self._host_faces.numpy()[:count].copy()
        searched = time.perf_counter()
        found = face_ids >= 0
        for name in dict.fromkeys(names):
            applicable = np.array([value == name for value in names])[instances]
            selected = np.flatnonzero(applicable & found)
            if not len(selected):
                continue
            mesh = self.meshes[name]
            triangles = mesh["vertices"][mesh["faces"][face_ids[selected]]]
            closest, bary, normal = _refine(local[selected], triangles)
            distance = np.linalg.norm(local[selected] - closest, axis=1)
            keep = distance <= radius
            selected, closest, bary, normal, distance = (
                selected[keep],
                closest[keep],
                bary[keep],
                normal[keep],
                distance[keep],
            )
            world = (
                np.einsum("nij,nj->ni", rotations[selected], closest)
                + translations[selected]
            )
            normals = np.einsum("nij,nj->ni", rotations[selected], normal)
            delta = points[selected] - world
            result["found"][selected] = True
            result["face_ids"][selected] = face_ids[selected]
            result["normal_valid"][selected] = True
            result["points"][selected] = world
            result["points_local"][selected] = closest
            result["barycentric"][selected] = bary
            result["normals"][selected] = normals
            result["normals_local"][selected] = normal
            result["distances"][selected] = distance
            result["signed_halfspace"][selected] = np.einsum("ij,ij->i", delta, normals)
            result["squared_distance_gradient"][selected] = 2 * delta
            valid = distance > 1e-12
            result["distance_gradient_valid"][selected[valid]] = True
            result["distance_gradient"][selected[valid]] = (
                delta[valid] / distance[valid, None]
            )
        self.calls += 1
        self.last_timing = {
            "query_count": count,
            "prepare_seconds": prepared - started,
            "search_transfer_seconds": searched - prepared,
            "refine_seconds": time.perf_counter() - searched,
            "total_seconds": time.perf_counter() - started,
        }
        return result

    def candidate_pairs_batch(
        self,
        pairs,
        transforms,
        *,
        tolerance=1e-9,
        max_candidates=10000000,
        world_bounds=None,
    ):
        """Enumerate conservative original world-triangle AABB candidate pairs.

        ``pairs`` is an explicit list of distinct named mesh pairs. ``transforms``
        maps each used name to its current finite affine local-to-world matrix;
        matrices are never normalized or rigidized. Return
        one ``(face_ids_a, face_ids_b)`` tuple per pair, including empty arrays.
        IDs are unique within each pair. Feed each tuple to the independent CPU
        ``mesh_intersections(..., candidate_pairs=[tuple])`` narrowphase.

        Bounds are computed on the original float64 transformed vertices and
        rounded outward to float32. One aggregate GPU BVH is refitted when the
        pose changes, without modifying the immutable local nearest-query BVHs.
        Two passes count then fill candidates; exceeding the explicit allocation
        limit raises, never truncates or returns an incomplete negative result.
        This function itself makes no collision or clear-space determination.

        An audit caller which already computed exact float64 transformed
        triangle bounds may pass ``world_bounds[name] = (lower, upper)`` to avoid
        recomputing them. These are an internal trusted-computation interface,
        not independently bound external data; their shape/finiteness is checked
        here but the caller owns their exact correspondence to the given pose.
        """
        started = time.perf_counter()
        pairs = list(pairs)
        seen = set()
        for pair in pairs:
            if (
                len(pair) != 2
                or pair[0] == pair[1]
                or any(name not in self.meshes for name in pair)
            ):
                raise ValueError("candidate pair needs two distinct known meshes")
            key = frozenset(pair)
            if key in seen:
                raise ValueError("duplicate candidate pair")
            seen.add(key)
        if not pairs:
            return []
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            type(max_candidates) is not int
            or not 0 < max_candidates < np.iinfo(np.int32).max
        ):
            raise ValueError("max_candidates must be a positive int32 allocation limit")
        names = tuple(sorted({name for pair in pairs for name in pair}))
        matrices = _affine_matrices([transforms[name] for name in names], len(names))
        bounds, ranges = {}, {}
        offset = 0
        for name, matrix in zip(names, matrices):
            mesh = self.meshes[name]
            if world_bounds is None:
                # Same original-vertex transform as the independent CPU audit.
                world = mesh["vertices"] @ matrix[:3, :3].T + matrix[:3, 3]
                triangles = world[mesh["faces"]]
                bounds[name] = triangles.min(axis=1), triangles.max(axis=1)
            else:
                lo, hi = [
                    np.asarray(value, dtype=np.float64) for value in world_bounds[name]
                ]
                if (
                    lo.shape != (len(mesh["faces"]), 3)
                    or hi.shape != lo.shape
                    or not np.isfinite(lo).all()
                    or not np.isfinite(hi).all()
                    or np.any(lo > hi)
                ):
                    raise ValueError(
                        "world_bounds must be finite original-face lower/upper arrays"
                    )
                bounds[name] = lo, hi
            ranges[name] = offset, offset + len(mesh["faces"])
            offset += len(mesh["faces"])
        if offset >= np.iinfo(np.int32).max:
            raise ValueError("too many original triangles for GPU index range")
        lower = _outward32(np.concatenate([bounds[name][0] for name in names]), -np.inf)
        upper = _outward32(np.concatenate([bounds[name][1] for name in names]), np.inf)
        boxes = {name: (lo.min(0), hi.max(0)) for name, (lo, hi) in bounds.items()}
        active = [
            bool(
                np.all(boxes[a][1] >= boxes[b][0] - tolerance)
                and np.all(boxes[b][1] >= boxes[a][0] - tolerance)
            )
            for a, b in pairs
        ]
        if not any(active):
            self.last_broadphase_timing = {
                "pair_count": len(pairs),
                "active_pair_count": 0,
                "conservative_candidates": 0,
                "total_seconds": time.perf_counter() - started,
            }
            return [
                (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
                for _ in pairs
            ]
        active_pairs = [pair for pair, keep in zip(pairs, active) if keep]
        qlower = _outward32(
            np.concatenate([bounds[a][0] - tolerance for a, _ in active_pairs]), -np.inf
        )
        qupper = _outward32(
            np.concatenate([bounds[a][1] + tolerance for a, _ in active_pairs]), np.inf
        )
        source_sizes = np.array(
            [len(bounds[a][0]) if keep else 0 for (a, _), keep in zip(pairs, active)]
        )
        source_offsets = np.concatenate(([0], np.cumsum(source_sizes)))
        begins = np.repeat(
            np.array([ranges[b][0] for _, b in pairs], dtype=np.int32), source_sizes
        )
        ends = np.repeat(
            np.array([ranges[b][1] for _, b in pairs], dtype=np.int32), source_sizes
        )
        wp = self.wp
        count_kernel, fill_kernel = _broadphase_kernels(wp)
        cache_key = (names, tuple(map(tuple, pairs)), tuple(active))
        cache = self._broadphase
        if cache is None or cache["key"] != cache_key:

            def device_array(values, dtype):
                return wp.array(values, dtype=dtype, device=self.device)

            cache = {
                "key": cache_key,
                "lower": device_array(lower, wp.vec3),
                "upper": device_array(upper, wp.vec3),
                "qlower": device_array(qlower, wp.vec3),
                "qupper": device_array(qupper, wp.vec3),
                "begin": device_array(begins, wp.int32),
                "end": device_array(ends, wp.int32),
                "counts": wp.empty(len(begins), dtype=wp.int32, device=self.device),
                "offsets": wp.empty(len(begins), dtype=wp.int32, device=self.device),
                "host_counts": wp.empty(
                    len(begins),
                    dtype=wp.int32,
                    device="cpu",
                    pinned=self.device.is_cuda,
                ),
                "output_capacity": 0,
            }
            cache["bvh"] = wp.Bvh(cache["lower"], cache["upper"])
            self._broadphase = cache
        else:
            for key, array in (
                ("lower", lower),
                ("upper", upper),
                ("qlower", qlower),
                ("qupper", qupper),
            ):
                cache[key].assign(array)
            cache["bvh"].refit()
        prepared = time.perf_counter()
        inputs = [
            cache["bvh"].id,
            cache["qlower"],
            cache["qupper"],
            cache["begin"],
            cache["end"],
        ]
        wp.launch(
            count_kernel,
            dim=len(begins),
            inputs=inputs,
            outputs=[cache["counts"]],
            device=self.device,
        )
        wp.copy(cache["host_counts"], cache["counts"])
        wp.synchronize_device(self.device)
        counts = cache["host_counts"].numpy().copy()
        cumulative = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        total = int(cumulative[-1])
        if total > max_candidates:
            raise OverflowError(
                f"GPU broadphase requires {total} candidates, exceeds {max_candidates}; no result returned"
            )
        counted = time.perf_counter()
        cache["offsets"].assign(cumulative[:-1].astype(np.int32))
        if total > cache["output_capacity"]:
            cache["output_capacity"] = max(total, 256, cache["output_capacity"] * 2)
            cache["output"] = wp.empty(
                cache["output_capacity"], dtype=wp.int32, device=self.device
            )
            cache["host_output"] = wp.empty(
                cache["output_capacity"],
                dtype=wp.int32,
                device="cpu",
                pinned=self.device.is_cuda,
            )
        result = []
        if total:
            wp.launch(
                fill_kernel,
                dim=len(begins),
                inputs=inputs + [cache["offsets"]],
                outputs=[cache["output"]],
                device=self.device,
            )
            wp.copy(cache["host_output"], cache["output"], count=total)
            wp.synchronize_device(self.device)
            target_ids = cache["host_output"].numpy()[:total].copy()
            for index, size in enumerate(source_sizes):
                lo, hi = source_offsets[index : index + 2]
                source_ids = np.repeat(np.arange(size, dtype=np.int32), counts[lo:hi])
                result.append((source_ids, target_ids[cumulative[lo] : cumulative[hi]]))
        else:
            result = [
                (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
                for _ in pairs
            ]
        self.last_broadphase_timing = {
            "pair_count": len(pairs),
            "active_pair_count": sum(active),
            "original_triangles": offset,
            "query_triangles": len(begins),
            "conservative_candidates": total,
            "prepare_refit_seconds": prepared - started,
            "count_transfer_seconds": counted - prepared,
            "fill_transfer_seconds": time.perf_counter() - counted,
            "total_seconds": time.perf_counter() - started,
        }
        return result

    def metadata(self):
        return {
            "backend": "warp_original_triangle_queries",
            "warp_version": self.wp.__version__,
            "device": str(self.device),
            "search_dtype": "float32_local_coordinates",
            "broadphase_dtype": "float32_outward_rounded_original_world_AABBs",
            "witness_dtype": "float64_original_vertices",
            "mesh_sha256": {name: mesh["sha256"] for name, mesh in self.meshes.items()},
            "max_float32_vertex_error": {
                name: mesh["max_float32_vertex_error"]
                for name, mesh in self.meshes.items()
            },
            "mesh_count": len(self.meshes),
            "build_seconds": self.build_seconds,
            "calls": self.calls,
            "buffer_capacity": self.capacity,
            "acceptance": False,
            "scope": "nearest-surface proposals and conservative world-AABB pairs; exact triangle narrowphase remains external; no containment or continuous-time test",
        }
