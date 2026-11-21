"""Offline mesh-surface relation targets for human-object supervision."""

from __future__ import annotations

import os
import numpy as np
from scipy.spatial import cKDTree

from gear_sonic.rp2.source_relation import world_vectors_to_heading_frame


_FAR_FIELD_M = float(os.environ.get("RP2_SURFACE_FAR_FIELD_M", "0.25"))

def _kd_workers() -> int:
    """cKDTree worker count bounded by the CPUs this process may actually use.

    `workers=-1` means os.cpu_count() (188 on cpu_av nodes) and cKDTree spawns
    that many Python threads PER QUERY CALL: a 1-motion cProfile (2026-09-01)
    showed 1.47M Thread.start calls costing 170 of 320 s inside a 2-CPU cgroup.
    """
    import os as _os

    env = _os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        return max(1, len(_os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, _os.cpu_count() or 1)



def exact_closest_points_to_mesh(
    points_local: np.ndarray,
    vertices_local: np.ndarray,
    faces: np.ndarray,
    *,
    point_chunk: int = 64,
    triangle_chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact unsigned point-to-triangle distances for validation.

    This deliberately uses a chunked brute-force query. Production extraction
    uses a separately implemented certified centroid-radius search; validators
    use this independent path on a bounded frame subset to verify it.
    """
    points = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    vertices = np.asarray(vertices_local, dtype=np.float64)
    triangles = vertices[np.asarray(faces, dtype=np.int64)]
    if not len(points) or not len(triangles):
        raise ValueError("Exact mesh query requires points and triangles")
    closest = np.zeros_like(points)
    minimum_distance_squared = np.full(len(points), np.inf, dtype=np.float64)
    epsilon = 1.0e-16

    for point_start in range(0, len(points), int(point_chunk)):
        point_end = min(len(points), point_start + int(point_chunk))
        point_batch = points[point_start:point_end]
        batch_best_distance = np.full(len(point_batch), np.inf, dtype=np.float64)
        batch_best_point = np.zeros_like(point_batch)
        point = point_batch[:, None, :]
        for triangle_start in range(0, len(triangles), int(triangle_chunk)):
            triangle_end = min(
                len(triangles), triangle_start + int(triangle_chunk)
            )
            triangle = triangles[triangle_start:triangle_end]
            a = triangle[None, :, 0]
            b = triangle[None, :, 1]
            c = triangle[None, :, 2]
            ab = b - a
            ac = c - a

            candidate_points = []
            candidate_distances = []
            for start, end in ((a, b), (b, c), (c, a)):
                edge = end - start
                denominator = np.sum(edge * edge, axis=-1)
                fraction = np.sum((point - start) * edge, axis=-1) / np.maximum(
                    denominator, epsilon
                )
                fraction = np.clip(fraction, 0.0, 1.0)
                segment_point = start + fraction[..., None] * edge
                candidate_points.append(segment_point)
                candidate_distances.append(
                    np.sum((point - segment_point) ** 2, axis=-1)
                )

            normal = np.cross(ab, ac)
            normal_squared = np.sum(normal * normal, axis=-1)
            plane_numerator = np.sum((point - a) * normal, axis=-1)
            projected = point - (
                plane_numerator
                / np.maximum(normal_squared, epsilon)
            )[..., None] * normal
            v0 = ab
            v1 = ac
            v2 = projected - a
            d00 = np.sum(v0 * v0, axis=-1)
            d01 = np.sum(v0 * v1, axis=-1)
            d11 = np.sum(v1 * v1, axis=-1)
            d20 = np.sum(v2 * v0, axis=-1)
            d21 = np.sum(v2 * v1, axis=-1)
            barycentric_denominator = d00 * d11 - d01 * d01
            barycentric_u = (
                d11 * d20 - d01 * d21
            ) / np.maximum(barycentric_denominator, epsilon)
            barycentric_v = (
                d00 * d21 - d01 * d20
            ) / np.maximum(barycentric_denominator, epsilon)
            inside = (
                (normal_squared > epsilon)
                & (barycentric_denominator > epsilon)
                & (barycentric_u >= -1.0e-9)
                & (barycentric_v >= -1.0e-9)
                & ((barycentric_u + barycentric_v) <= 1.0 + 1.0e-9)
            )
            plane_distance = (
                plane_numerator * plane_numerator
            ) / np.maximum(normal_squared, epsilon)
            plane_distance = np.where(inside, plane_distance, np.inf)
            candidate_points.append(projected)
            candidate_distances.append(plane_distance)

            distances = np.stack(candidate_distances, axis=0)
            points_on_surface = np.stack(candidate_points, axis=0)
            candidate_index = np.argmin(distances, axis=0)
            triangle_distance = np.take_along_axis(
                distances,
                candidate_index[None],
                axis=0,
            )[0]
            triangle_point = np.take_along_axis(
                points_on_surface,
                candidate_index[None, ..., None],
                axis=0,
            )[0]
            best_triangle = np.argmin(triangle_distance, axis=1)
            row = np.arange(len(point_batch))
            chunk_distance = triangle_distance[row, best_triangle]
            chunk_point = triangle_point[row, best_triangle]
            improve = chunk_distance < batch_best_distance
            batch_best_distance[improve] = chunk_distance[improve]
            batch_best_point[improve] = chunk_point[improve]

        minimum_distance_squared[point_start:point_end] = batch_best_distance
        closest[point_start:point_end] = batch_best_point
    return (
        np.sqrt(np.maximum(minimum_distance_squared, 0.0)).astype(np.float32),
        closest.astype(np.float32),
    )


def _closest_points_for_candidates(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest points for per-query candidate triangles."""
    point = np.asarray(points, dtype=np.float64)[:, None, :]
    triangle = np.asarray(triangles, dtype=np.float64)
    a = triangle[:, :, 0]
    b = triangle[:, :, 1]
    c = triangle[:, :, 2]
    epsilon = 1.0e-16

    candidate_points = []
    candidate_distances = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        denominator = np.sum(edge * edge, axis=-1)
        fraction = np.sum((point - start) * edge, axis=-1) / np.maximum(
            denominator, epsilon
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        segment_point = start + fraction[..., None] * edge
        candidate_points.append(segment_point)
        candidate_distances.append(
            np.sum((point - segment_point) ** 2, axis=-1)
        )

    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_squared = np.sum(normal * normal, axis=-1)
    plane_numerator = np.sum((point - a) * normal, axis=-1)
    projected = point - (
        plane_numerator / np.maximum(normal_squared, epsilon)
    )[..., None] * normal
    v2 = projected - a
    d00 = np.sum(ab * ab, axis=-1)
    d01 = np.sum(ab * ac, axis=-1)
    d11 = np.sum(ac * ac, axis=-1)
    d20 = np.sum(v2 * ab, axis=-1)
    d21 = np.sum(v2 * ac, axis=-1)
    denominator = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / np.maximum(denominator, epsilon)
    v = (d00 * d21 - d01 * d20) / np.maximum(denominator, epsilon)
    inside = (
        (normal_squared > epsilon)
        & (denominator > epsilon)
        & (u >= -1.0e-9)
        & (v >= -1.0e-9)
        & ((u + v) <= 1.0 + 1.0e-9)
    )
    plane_distance = (
        plane_numerator * plane_numerator
    ) / np.maximum(normal_squared, epsilon)
    candidate_points.append(projected)
    candidate_distances.append(np.where(inside, plane_distance, np.inf))

    distances = np.stack(candidate_distances, axis=0)
    surface_points = np.stack(candidate_points, axis=0)
    feature_index = np.argmin(distances, axis=0)
    triangle_distance = np.take_along_axis(
        distances, feature_index[None], axis=0
    )[0]
    triangle_point = np.take_along_axis(
        surface_points,
        feature_index[None, ..., None],
        axis=0,
    )[0]
    candidate_index = np.argmin(triangle_distance, axis=1)
    row = np.arange(len(points))
    return (
        np.sqrt(np.maximum(triangle_distance[row, candidate_index], 0.0)),
        triangle_point[row, candidate_index],
        candidate_index,
    )


def _certified_face_candidates(
    points: np.ndarray,
    upper_distance: np.ndarray,
    centroids: np.ndarray,
    face_radius: np.ndarray,
) -> list[np.ndarray]:
    """Return a candidate set guaranteed to contain every closer triangle.

    For any point ``x`` on a triangle with centroid ``c`` and radius ``r``,
    ``||p-c|| <= ||p-x|| + r``. Grouping similarly sized triangles therefore
    lets a centroid radius query retain every triangle that could improve the
    current upper bound without expanding every query by the single largest
    triangle in the mesh.
    """
    safe_radius = np.maximum(np.asarray(face_radius), 1.0e-12)
    radius_bucket = np.floor(np.log2(safe_radius)).astype(np.int64)
    candidates: list[list[np.ndarray]] = [[] for _ in range(len(points))]
    epsilon = 1.0e-10

    for bucket in np.unique(radius_bucket):
        face_ids = np.flatnonzero(radius_bucket == bucket)
        bucket_tree = cKDTree(centroids[face_ids])
        bucket_max_radius = float(np.max(safe_radius[face_ids]))
        matches = bucket_tree.query_ball_point(
            points,
            r=np.asarray(upper_distance) + bucket_max_radius + epsilon,
            workers=_kd_workers(),
        )
        for point_index, local_faces in enumerate(matches):
            if len(local_faces):
                candidates[point_index].append(
                    face_ids[np.asarray(local_faces, dtype=np.int64)]
                )

    result = []
    for point_candidates in candidates:
        if not point_candidates:
            raise RuntimeError("Certified mesh query produced no candidates")
        result.append(np.concatenate(point_candidates))
    return result


def _refine_variable_face_candidates(
    points: np.ndarray,
    triangles: np.ndarray,
    candidate_faces: list[np.ndarray],
    *,
    group_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine variable-size face sets in similarly sized padded groups."""
    num_points = len(points)
    distance = np.empty(num_points, dtype=np.float64)
    surface_point = np.empty((num_points, 3), dtype=np.float64)
    selected_face = np.empty(num_points, dtype=np.int64)
    order = np.argsort(
        np.asarray([len(candidate) for candidate in candidate_faces])
    )

    for start in range(0, num_points, int(group_size)):
        point_ids = order[start : start + int(group_size)]
        maximum_count = max(len(candidate_faces[index]) for index in point_ids)
        padded_faces = np.empty(
            (len(point_ids), maximum_count), dtype=np.int64
        )
        for row, point_index in enumerate(point_ids):
            faces_for_point = candidate_faces[point_index]
            padded_faces[row] = faces_for_point[0]
            padded_faces[row, : len(faces_for_point)] = faces_for_point
        group_distance, group_point, group_choice = (
            _closest_points_for_candidates(
                points[point_ids],
                triangles[padded_faces],
            )
        )
        row = np.arange(len(point_ids))
        distance[point_ids] = group_distance
        surface_point[point_ids] = group_point
        selected_face[point_ids] = padded_faces[row, group_choice]
    return distance, surface_point, selected_face


def _oriented_face_normals(
    vertices: np.ndarray,
    face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient watertight components and mark faces with a trustworthy sign."""
    triangles = vertices[face_indices]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normal_norm = np.linalg.norm(face_normals, axis=-1, keepdims=True)
    face_normals /= np.maximum(normal_norm, 1.0e-12)
    num_faces = len(face_indices)
    parent = np.arange(num_faces, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(face_indices):
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            if start == end:
                continue
            key = (min(start, end), max(start, end))
            direction = 1 if start < end else -1
            edge_occurrences.setdefault(key, []).append(
                (face_index, direction)
            )

    for occurrences in edge_occurrences.values():
        first_face = occurrences[0][0]
        for face_index, _ in occurrences[1:]:
            union(first_face, face_index)
    roots = np.asarray([find(index) for index in range(num_faces)])
    component_faces: dict[int, list[int]] = {}
    for face_index, root in enumerate(roots):
        component_faces.setdefault(int(root), []).append(face_index)

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_faces)]
    invalid_components: set[int] = set()
    for occurrences in edge_occurrences.values():
        root = int(roots[occurrences[0][0]])
        if len(occurrences) != 2:
            invalid_components.add(root)
            continue
        (left_face, left_direction), (right_face, right_direction) = occurrences
        relative_sign = -left_direction * right_direction
        adjacency[left_face].append((right_face, relative_sign))
        adjacency[right_face].append((left_face, relative_sign))

    orientation = np.zeros(num_faces, dtype=np.int8)
    signable = np.zeros(num_faces, dtype=bool)
    for root, members in component_faces.items():
        if root in invalid_components:
            continue
        stack = [members[0]]
        orientation[members[0]] = 1
        consistent = True
        while stack:
            face_index = stack.pop()
            for neighbor, relative_sign in adjacency[face_index]:
                expected = int(orientation[face_index]) * relative_sign
                if orientation[neighbor] == 0:
                    orientation[neighbor] = expected
                    stack.append(neighbor)
                elif orientation[neighbor] != expected:
                    consistent = False
        if not consistent or np.any(orientation[members] == 0):
            continue
        member_array = np.asarray(members, dtype=np.int64)
        oriented_triangles = triangles[member_array].copy()
        flip = orientation[member_array] < 0
        oriented_triangles[flip] = oriented_triangles[flip][:, [0, 2, 1]]
        signed_volume = np.sum(
            np.einsum(
                "ij,ij->i",
                oriented_triangles[:, 0],
                np.cross(
                    oriented_triangles[:, 1],
                    oriented_triangles[:, 2],
                ),
            )
        ) / 6.0
        if abs(float(signed_volume)) <= 1.0e-12:
            continue
        component_sign = orientation[member_array].astype(np.float64)
        if signed_volume < 0.0:
            component_sign *= -1.0
        face_normals[member_array] *= component_sign[:, None]
        signable[member_array] = True
    return face_normals, signable


def closest_points_to_mesh(
    points_local: np.ndarray,
    vertices_local: np.ndarray,
    faces: np.ndarray,
    *,
    candidate_count: int = 64,
    exact_face_threshold: int = 512,
    point_chunk: int = 512,
    compute_signed_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Certified shape-aware closest-point and signed-distance mesh query.

    Small meshes are queried exactly. For dense meshes, nearest face-centroid
    candidates first provide an upper bound. Radius-bucketed centroid queries
    then retain every face that could beat that bound before exact
    point-to-triangle refinement.
    """
    original_shape = np.asarray(points_local).shape[:-1]
    points = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    vertices = np.asarray(vertices_local, dtype=np.float64)
    face_indices = np.asarray(faces, dtype=np.int64)
    triangles = vertices[face_indices]
    if not len(points) or not len(triangles):
        raise ValueError("Mesh query requires points and triangles")

    centroids = triangles.mean(axis=1)
    face_radius = np.max(
        np.linalg.norm(triangles - centroids[:, None, :], axis=-1),
        axis=1,
    )
    if compute_signed_distance:
        _, signable_faces = _oriented_face_normals(vertices, face_indices)
        signable_face_indices = face_indices[signable_faces]
        parity_inside = (
            _inside_mesh(points, vertices, signable_face_indices)
            if len(signable_face_indices)
            else np.zeros(len(points), dtype=bool)
        )
    else:
        signable_faces = np.zeros(len(triangles), dtype=bool)
        parity_inside = np.zeros(len(points), dtype=bool)

    use_all_faces = len(triangles) <= int(exact_face_threshold)
    tree = None if use_all_faces else cKDTree(centroids)
    count = min(max(1, int(candidate_count)), len(triangles))
    output_distance = np.empty(len(points), dtype=np.float64)
    output_point = np.empty_like(points)
    output_face = np.empty(len(points), dtype=np.int64)
    output_signed = np.empty(len(points), dtype=np.float64)

    for start in range(0, len(points), int(point_chunk)):
        end = min(len(points), start + int(point_chunk))
        point_batch = points[start:end]
        if use_all_faces:
            candidate_faces = np.broadcast_to(
                np.arange(len(triangles), dtype=np.int64),
                (len(point_batch), len(triangles)),
            )
        else:
            _, candidate_faces = tree.query(
                point_batch, k=count, workers=_kd_workers()
            )
            if count == 1:
                candidate_faces = candidate_faces[:, None]
        distance, surface_point, local_choice = (
            _closest_points_for_candidates(
                point_batch,
                triangles[candidate_faces],
            )
        )
        row = np.arange(len(point_batch))
        selected_face = candidate_faces[row, local_choice]
        if not use_all_faces:
            # FAR-FIELD SHORTCUT (2026-09-01). Certification (bucketed radius
            # queries + exact refinement over every face that could beat the
            # k-nearest-centroid bound) is the dominant cost, and it only
            # changes the answer when a closer face was hidden behind a large
            # neighbour -- a sub-face-radius effect. Links farther than
            # RP2_SURFACE_FAR_FIELD_M (default 0.25 m; contact thresholds are
            # 0.02/0.06 m) keep the k-candidate result. For teleop, 13 of the
            # 14 links are far-field for the whole clip.
            near = distance <= _FAR_FIELD_M
            if np.any(near):
                near_idx = np.flatnonzero(near)
                certified_faces = _certified_face_candidates(
                    point_batch[near_idx],
                    distance[near_idx],
                    centroids,
                    face_radius,
                )
                n_distance, n_point, n_face = (
                    _refine_variable_face_candidates(
                        point_batch[near_idx],
                        triangles,
                        certified_faces,
                    )
                )
                distance = distance.copy(); surface_point = surface_point.copy()
                selected_face = np.array(selected_face, copy=True)
                distance[near_idx] = n_distance
                surface_point[near_idx] = n_point
                selected_face[near_idx] = n_face
        if compute_signed_distance:
            signed_distance = np.where(
                signable_faces[selected_face]
                & parity_inside[start:end],
                -distance,
                distance,
            )
        else:
            signed_distance = distance
        output_distance[start:end] = distance
        output_point[start:end] = surface_point
        output_face[start:end] = selected_face
        output_signed[start:end] = signed_distance

    return (
        output_distance.reshape(original_shape).astype(np.float32),
        output_point.reshape(original_shape + (3,)).astype(np.float32),
        output_signed.reshape(original_shape).astype(np.float32),
        output_face.reshape(original_shape),
    )


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4:
        raise ValueError(f"Expected WXYZ quaternion(s), got {value.shape}")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("Object quaternion contains a zero norm")
    w, x, y, z = np.moveaxis(value / norm, -1, 0)
    matrix = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def world_points_to_object_local(
    points_world: np.ndarray,
    object_position: np.ndarray,
    object_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32)
    position = np.asarray(object_position, dtype=np.float32)
    rotation = _quat_wxyz_to_matrix(object_quaternion_wxyz)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points [T, L, 3], got {points.shape}")
    if position.shape != (points.shape[0], 3):
        raise ValueError(
            f"Object position shape {position.shape} does not match {points.shape}"
        )
    return np.einsum(
        "tlj,tji->tli", points - position[:, None, :], rotation
    ).astype(np.float32)


def object_points_to_world(
    points_local: np.ndarray,
    object_position: np.ndarray,
    object_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_local, dtype=np.float32)
    position = np.asarray(object_position, dtype=np.float32)
    rotation = _quat_wxyz_to_matrix(object_quaternion_wxyz)
    return (
        np.einsum("tij,tlj->tli", rotation, points) + position[:, None, :]
    ).astype(np.float32)


def _ray_intersection_counts(
    points: np.ndarray,
    triangles: np.ndarray,
    direction: np.ndarray,
    *,
    point_chunk: int,
    triangle_chunk: int,
) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1.0e-12
    counts = np.zeros(len(points), dtype=np.int32)
    epsilon = 1.0e-8
    for point_start in range(0, len(points), point_chunk):
        point_end = min(len(points), point_start + point_chunk)
        point_batch = points[point_start:point_end].astype(np.float64)
        count = np.zeros(len(point_batch), dtype=np.int32)
        for triangle_start in range(0, len(triangles), triangle_chunk):
            triangle_end = min(len(triangles), triangle_start + triangle_chunk)
            triangle = triangles[triangle_start:triangle_end].astype(np.float64)
            vertex = triangle[:, 0]
            edge_1 = triangle[:, 1] - vertex
            edge_2 = triangle[:, 2] - vertex
            h = np.cross(np.broadcast_to(direction, edge_2.shape), edge_2)
            determinant = np.sum(edge_1 * h, axis=-1)
            valid_triangle = np.abs(determinant) > epsilon
            if not np.any(valid_triangle):
                continue
            inverse = np.zeros_like(determinant)
            inverse[valid_triangle] = 1.0 / determinant[valid_triangle]
            s = point_batch[:, None, :] - vertex[None, :, :]
            u = inverse[None, :] * np.sum(s * h[None, :, :], axis=-1)
            q = np.cross(s, edge_1[None, :, :])
            v = inverse[None, :] * np.sum(
                q * direction.reshape(1, 1, 3), axis=-1
            )
            distance = inverse[None, :] * np.sum(
                q * edge_2[None, :, :], axis=-1
            )
            hits = (
                valid_triangle[None, :]
                & (u >= -1.0e-7)
                & (v >= -1.0e-7)
                & ((u + v) <= 1.0 + 1.0e-7)
                & (distance > epsilon)
            )
            count += hits.sum(axis=1).astype(np.int32)
        counts[point_start:point_end] = count
    return counts


def _inside_mesh(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    point_chunk: int = 128,
    triangle_chunk: int = 512,
) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(faces):
        return inside
    aabb_min = np.asarray(vertices).min(axis=0) - 1.0e-6
    aabb_max = np.asarray(vertices).max(axis=0) + 1.0e-6
    candidates = np.all((points >= aabb_min) & (points <= aabb_max), axis=-1)
    if not np.any(candidates):
        return inside
    candidate_points = np.asarray(points[candidates], dtype=np.float32)
    triangles = np.asarray(vertices, dtype=np.float32)[
        np.asarray(faces, dtype=np.int64)
    ]
    votes = np.zeros(len(candidate_points), dtype=np.int32)
    for direction in (
        np.asarray([1.0, 0.371, 0.173]),
        np.asarray([0.137, 1.0, 0.419]),
        np.asarray([0.223, 0.311, 1.0]),
    ):
        counts = _ray_intersection_counts(
            candidate_points,
            triangles,
            direction,
            point_chunk=point_chunk,
            triangle_chunk=triangle_chunk,
        )
        votes += (counts % 2 == 1).astype(np.int32)
    inside[candidates] = votes >= 2
    return inside


def points_inside_mesh_ray_parity(
    points_local: np.ndarray,
    vertices_local: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Independent ray-parity sign query intended for bounded validation."""
    original_shape = np.asarray(points_local).shape[:-1]
    return _inside_mesh(
        np.asarray(points_local, dtype=np.float32).reshape(-1, 3),
        np.asarray(vertices_local, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
    ).reshape(original_shape)


def _stabilize_surface_indices(
    points_local: np.ndarray,
    surface_samples: np.ndarray,
    nearest_distance: np.ndarray,
    nearest_index: np.ndarray,
    *,
    hysteresis_m: float,
) -> np.ndarray:
    """Keep the prior surface anchor while it remains nearly as good."""
    output = nearest_index.copy()
    if hysteresis_m <= 0.0 or points_local.shape[0] < 2:
        return output
    for link_index in range(points_local.shape[1]):
        previous = int(output[0, link_index])
        for frame in range(1, points_local.shape[0]):
            previous_distance = float(
                np.linalg.norm(
                    points_local[frame, link_index] - surface_samples[previous]
                )
            )
            if previous_distance <= float(
                nearest_distance[frame, link_index] + hysteresis_m
            ):
                output[frame, link_index] = previous
            else:
                previous = int(output[frame, link_index])
    return output


def build_mesh_surface_relations(
    link_positions_world: np.ndarray,
    object_positions: np.ndarray,
    object_quaternions_wxyz: np.ndarray,
    heading_xy: np.ndarray,
    *,
    vertices_local: np.ndarray,
    faces: np.ndarray,
    surface_samples_local: np.ndarray,
    anchor_hysteresis_m: float = 0.005,
    surface_epsilon_m: float = 0.001,
) -> dict[str, np.ndarray | str | float]:
    """Build SDF-like nearest-surface targets without changing tensor width.

    The signed-distance convention is positive outside and negative inside for
    independently orientable watertight components. Open or non-manifold
    components retain unsigned positive distance because their interior is not
    geometrically defined. Penetrating source links receive a zero desired
    displacement, which projects the desired G1 landmark onto the stored
    surface anchor instead of teaching the policy to reproduce penetration.
    """
    links_world = np.asarray(link_positions_world, dtype=np.float32)
    local = world_points_to_object_local(
        links_world, object_positions, object_quaternions_wxyz
    )
    (
        nearest_distance,
        surface_point_local,
        signed_distance,
        _,
    ) = closest_points_to_mesh(
        local,
        np.asarray(vertices_local, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
    )
    surface_point_world = object_points_to_world(
        surface_point_local, object_positions, object_quaternions_wxyz
    )

    signed_distance[np.abs(signed_distance) <= float(surface_epsilon_m)] = 0.0
    inside = signed_distance < 0.0

    displacement_world = surface_point_world - links_world
    displacement_world[inside] = 0.0
    relation = world_vectors_to_heading_frame(displacement_world, heading_xy)
    valid = (
        np.isfinite(relation).all(axis=-1)
        & np.isfinite(surface_point_local).all(axis=-1)
        & np.isfinite(signed_distance)
    )
    return {
        "relation_vector": np.where(
            valid[..., None], relation, 0.0
        ).astype(np.float32),
        "relation_mask": valid.astype(np.float32),
        "surface_point_local": np.where(
            valid[..., None], surface_point_local, 0.0
        ).astype(np.float32),
        "signed_distance": np.where(valid, signed_distance, 0.0).astype(
            np.float32
        ),
        "source_penetration_mask": (inside & valid).astype(np.float32),
        "semantics": (
            "nearest_triangle_surface_displacement_heading_frame_"
            "watertight_signed_distance_positive_outside_"
            "open_mesh_unsigned_source_penetration_projected_v3"
        ),
        "anchor_hysteresis_m": 0.0,
        "surface_epsilon_m": float(surface_epsilon_m),
    }
