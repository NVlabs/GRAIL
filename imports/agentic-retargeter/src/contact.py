"""Original-triangle contact evidence and independent CPU geometry diagnostics.

Distances are unsigned, in the input length unit (normally metres). No mesh is
remeshed, simplified, repaired or reordered. These NumPy routines favour an
auditable FP64 reference over GPU throughput; use them for contact regions and
independent checks, not an all-pairs full-sequence loss. A small unsigned gap is
neither proof of contact nor proof of collision freedom.
"""

import hashlib
from dataclasses import dataclass

import numpy as np


def _mesh(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite (V, 3)")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or raw_faces.dtype.kind not in "iu":
        raise ValueError("faces must be integer (F, 3)")
    faces = raw_faces.astype(np.int64, copy=False)
    if not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces must be nonempty and index the original vertices")
    return vertices, faces


def mesh_digest(vertices, faces):
    """Hash exact geometry/order, independent of host byte order and input dtype."""
    vertices, faces = _mesh(vertices, faces)
    digest = hashlib.sha256(b"grail-original-surface-v1\0")
    for array, dtype in ((vertices, "<f8"), (faces, "<i8")):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ContactPatch:
    """A reviewed *region hypothesis*, never an observed hidden 3-D point.

    The inclusive phase and source frames use the unresampled source timeline.
    ``evidence`` should describe visible hand/object regions and uncertainty;
    ``source_sha256`` is the source-video file digest from the case manifest.
    Face IDs address the exact mesh whose ``mesh_sha256`` is supplied.
    """

    name: str
    side: str
    start_frame: int
    end_frame: int
    face_ids: tuple[int, ...]
    evidence_frames: tuple[int, ...]
    evidence: str
    source_sha256: str
    mesh_sha256: str

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("face_ids", "evidence_frames"):
            data[key] = tuple(data[key])
        return cls(**data)

    def validate(self, vertices, faces, frame_count, source_sha256):
        _, faces = _mesh(vertices, faces)
        if self.side not in {"left", "right"} or not self.name or not self.evidence.strip():
            raise ValueError("contact needs a name, left/right side and visual evidence")
        indices = (self.start_frame, self.end_frame, *self.evidence_frames, *self.face_ids)
        if any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in indices):
            raise ValueError("frame and original face IDs must be integers")
        if not 0 <= self.start_frame <= self.end_frame < frame_count:
            raise ValueError("contact phase lies outside the source timeline")
        if not self.evidence_frames or any(not 0 <= i < frame_count for i in self.evidence_frames):
            raise ValueError("contact needs source evidence frames inside the timeline")
        if not self.face_ids or min(self.face_ids) < 0 or max(self.face_ids) >= len(faces):
            raise ValueError("contact region must reference existing original faces")
        if len(set(self.face_ids)) != len(self.face_ids):
            raise ValueError("contact region contains duplicate original face IDs")
        for value in (self.source_sha256, self.mesh_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("bindings must be lowercase SHA-256 digests")
        if self.source_sha256 != source_sha256:
            raise ValueError("contact evidence is bound to another source video")
        if self.mesh_sha256 != mesh_digest(vertices, faces):
            raise ValueError("original mesh changed; review the region instead of reusing face IDs")


def surface_samples(vertices, faces, face_ids, barycentric):
    """Resolve exact authored-face samples without losing their material identity."""
    vertices, faces = _mesh(vertices, faces)
    ids = np.asarray(face_ids)
    barycentric = np.asarray(barycentric, dtype=np.float64)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("face_ids must be an integer vector")
    if len(ids) and (ids.min() < 0 or ids.max() >= len(faces)):
        raise ValueError("invalid original face ID")
    if barycentric.shape != (len(ids), 3) or not np.isfinite(barycentric).all():
        raise ValueError("barycentric must be finite (N, 3)")
    if np.any(barycentric < 0) or not np.allclose(barycentric.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("barycentric weights must be nonnegative and sum to one")
    return np.einsum("ni,nij->nj", barycentric, vertices[faces[ids]])


@dataclass(frozen=True)
class NearestSurface:
    distance: np.ndarray
    points: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray
    normals: np.ndarray


def _closest_pairs(points, triangles):
    """Every point/triangle pair: plane projection plus all three bounded edges.

    The projection is considered only inside a nondegenerate triangle. Including
    all edge minima also handles zero-area triangles and avoids a closest-face
    switch silently returning a point on the wrong triangle.
    """
    a, b, c = np.moveaxis(triangles, 1, 0)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm2 = np.einsum("ij,ij->i", normal, normal)
    relative = points[:, None] - a
    denom = np.where(norm2 > 0, norm2, 1)
    plane = (
        points[:, None] - (np.einsum("pfi,fi->pf", relative, normal) / denom)[..., None] * normal
    )
    relative_plane = plane - a
    # Cross-product barycentrics avoid the cancellation in dot-product Gram determinants.
    v = np.einsum("pfi,fi->pf", np.cross(relative_plane, ac), normal) / denom
    w = np.einsum("pfi,fi->pf", np.cross(ab, relative_plane), normal) / denom
    bary = np.stack((1 - v - w, v, w), axis=-1)
    distance2 = np.sum((points[:, None] - plane) ** 2, axis=-1)
    distance2[(norm2 == 0)[None, :] | (bary < 0).any(axis=-1)] = np.inf
    closest = plane
    basis = np.eye(3)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start, stop = triangles[:, i], triangles[:, j]
        edge = stop - start
        length2 = np.sum(edge * edge, axis=-1)
        fraction = np.clip(
            np.einsum("pfi,fi->pf", points[:, None] - start, edge)
            / np.where(length2 > 0, length2, 1),
            0,
            1,
        )
        candidate = start + fraction[..., None] * edge
        candidate_distance2 = np.sum((points[:, None] - candidate) ** 2, axis=-1)
        better = candidate_distance2 < distance2
        closest = np.where(better[..., None], candidate, closest)
        candidate_bary = (1 - fraction[..., None]) * basis[i] + fraction[..., None] * basis[j]
        bary = np.where(better[..., None], candidate_bary, bary)
        distance2 = np.minimum(distance2, candidate_distance2)
    return distance2, closest, bary


def nearest_surface(points, vertices, faces, face_ids=None, point_chunk=32, face_chunk=1024):
    """Exact FP64 unsigned nearest points on all requested original triangles.

    No nearest-vertex/centroid approximation is used. Ties choose the lowest
    original face ID. Complexity is O(points × faces); chunks bound memory.
    Degenerate faces resolve to their edges, with zero normal.
    """
    vertices, faces = _mesh(vertices, faces)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite (N, 3)")
    ids = np.arange(len(faces)) if face_ids is None else np.asarray(face_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
        raise ValueError("face_ids must be a nonempty integer vector")
    if ids.min() < 0 or ids.max() >= len(faces) or min(point_chunk, face_chunk) < 1:
        raise ValueError("invalid original face IDs or chunk sizes")
    ids = np.unique(ids)
    distance2 = np.full(len(points), np.inf)
    closest = np.empty_like(points)
    barycentric = np.empty_like(points)
    closest_ids = np.empty(len(points), dtype=np.int64)
    for first in range(0, len(points), point_chunk):
        stop = min(first + point_chunk, len(points))
        p = points[first:stop]
        for offset in range(0, len(ids), face_chunk):
            chunk_ids = ids[offset : offset + face_chunk]
            d2, cp, bary = _closest_pairs(p, vertices[faces[chunk_ids]])
            col = d2.argmin(axis=1)
            row = np.arange(len(p))
            better = d2[row, col] < distance2[first:stop]
            selected = first + row[better]
            distance2[selected] = d2[row[better], col[better]]
            closest[selected] = cp[row[better], col[better]]
            barycentric[selected] = bary[row[better], col[better]]
            closest_ids[selected] = chunk_ids[col[better]]
    triangle = vertices[faces[closest_ids]]
    normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    lengths = np.linalg.norm(normals, axis=-1)
    normals /= np.where(lengths > 0, lengths, 1)[:, None]
    return NearestSurface(np.sqrt(distance2), closest, closest_ids, barycentric, normals)


def _components(vertices, faces):
    """Topology-only exact-position welding; original triangle arrays stay intact."""
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded = inverse[faces]
    parent = np.arange(len(faces))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edge_faces = {}
    for i, face in enumerate(welded):
        for a, b in zip(face, np.roll(face, -1)):
            edge = (min(a, b), max(a, b))
            if edge in edge_faces:
                parent[find(i)] = find(edge_faces[edge][0][0])
            edge_faces.setdefault(edge, []).append((i, int(a < b)))
    labels = np.array([find(i) for i in range(len(faces))])
    components = []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        edges = [hits for hits in edge_faces.values() if labels[hits[0][0]] == label]
        closed = all(len(hits) == 2 and hits[0][1] != hits[1][1] for hits in edges)
        triangle = vertices[faces[ids]]
        area = np.linalg.norm(
            np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1
        )
        closed = closed and bool((area > 0).all())
        components.append((ids, closed))
    return components


def _winding(points, triangles, chunk=256):
    winding = np.zeros(len(points))
    for point_start in range(0, len(points), 32):
        point_stop = min(point_start + 32, len(points))
        for first in range(0, len(triangles), chunk):
            relative = (
                triangles[None, first : first + chunk] - points[point_start:point_stop, None, None]
            )
            a, b, c = np.moveaxis(relative, 2, 0)
            la, lb, lc = [np.linalg.norm(v, axis=-1) for v in (a, b, c)]
            numerator = np.einsum("nfi,nfi->nf", a, np.cross(b, c))
            denominator = (
                la * lb * lc
                + np.sum(a * b, -1) * lc
                + np.sum(b * c, -1) * la
                + np.sum(c * a, -1) * lb
            )
            winding[point_start:point_stop] += np.arctan2(numerator, denominator).sum(axis=1) / (
                2 * np.pi
            )
    return winding


def classify_exposure(vertices, faces, face_ids, barycentric, boundary_tolerance=1e-8):
    """Identify samples buried in other closed, consistently wound components.

    This is a conservative target filter, not a solid-validity certificate.
    Self-intersecting components and same-component concavity are not certified;
    open/nonmanifold components make remaining exposure unknown. Samples near
    another component's boundary are also unknown. Exact-position welding is
    used for topology only, so face IDs and authored tessellation never change.
    """
    vertices, faces = _mesh(vertices, faces)
    points = surface_samples(vertices, faces, face_ids, barycentric)
    ids = np.asarray(face_ids)
    if not np.isfinite(boundary_tolerance) or boundary_tolerance <= 0:
        raise ValueError("boundary_tolerance must be positive")
    components = _components(vertices, faces)
    labels = np.empty(len(faces), dtype=int)
    for index, (component_ids, _) in enumerate(components):
        labels[component_ids] = index
    buried = np.zeros(len(points), dtype=bool)
    boundary = np.zeros(len(points), dtype=bool)
    witness = np.full(len(points), -1, dtype=int)
    for index, (component_ids, closed) in enumerate(components):
        use = np.flatnonzero(labels[ids] != index)
        if not closed or not len(use):
            continue
        distance = nearest_surface(points[use], vertices, faces, component_ids).distance
        on_boundary = distance <= boundary_tolerance
        boundary[use] |= on_boundary
        inside = (
            np.abs(_winding(points[use], vertices[faces[component_ids]])) > 0.5
        ) & ~on_boundary
        buried[use[inside]] = True
        witness[use[inside]] = index
    open_components = sum(not closed for _, closed in components)
    status = np.full(len(points), "outside_other_closed_components", dtype="U40")
    if open_components:
        status[:] = "unknown_open_geometry"
    status[boundary] = "unknown_boundary"
    status[buried] = "buried_in_other_closed_component"
    return {
        "status": status,
        "buried": buried,
        "witness_component": witness,
        "closed_components": len(components) - open_components,
        "open_or_inconsistent_components": open_components,
        "limitations": "No self-intersection or same-component exposure certification.",
    }


def unique_edges(faces):
    faces = np.asarray(faces, dtype=np.int64)
    return np.unique(
        np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1),
        axis=0,
    )


def edge_triangle_crossings(
    source_vertices, source_faces, target_vertices, target_faces, tolerance=1e-9, witness_limit=20
):
    """Count non-coplanar edge intersections, including target triangle seams.

    Segment endpoints are excluded to avoid reporting a touch as a crossing.
    Coincident target faces at one edge location count once. Coplanar overlap,
    tangent vertex touches and containment require separate checks; this is not
    a signed penetration depth. All original edges/faces participate.
    """
    sv, sf = _mesh(source_vertices, source_faces)
    tv, tf = _mesh(target_vertices, target_faces)
    if not np.isfinite(tolerance) or tolerance <= 0 or witness_limit < 0:
        raise ValueError("invalid crossing tolerance or witness limit")
    edges = unique_edges(sf)
    triangle = tv[tf]
    lower, upper = triangle.min(axis=1), triangle.max(axis=1)
    count, crossed_edges, witnesses = 0, 0, []
    for edge in edges:
        start, end = sv[edge]
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= 2 * tolerance:
            continue
        candidates = np.flatnonzero(
            (
                (upper >= np.minimum(start, end) - tolerance)
                & (lower <= np.maximum(start, end) + tolerance)
            ).all(axis=1)
        )
        if not len(candidates):
            continue
        a, b, c = np.moveaxis(triangle[candidates], 1, 0)
        e1, e2 = b - a, c - a
        p = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, p)
        scale = length * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        valid = np.abs(det) > 32 * np.finfo(float).eps * scale
        inverse = np.divide(1, det, out=np.zeros_like(det), where=valid)
        relative = start - a
        u = np.einsum("ij,ij->i", relative, p) * inverse
        q = np.cross(relative, e1)
        v = (q @ direction) * inverse
        fraction = np.einsum("ij,ij->i", e2, q) * inverse
        bary_tolerance = 64 * np.finfo(float).eps
        valid &= (
            (u >= -bary_tolerance)
            & (v >= -bary_tolerance)
            & (u + v <= 1 + bary_tolerance)
            & (fraction * length > tolerance)
            & ((1 - fraction) * length > tolerance)
        )
        hits = np.flatnonzero(valid)
        hits = hits[np.argsort(fraction[hits], kind="stable")]
        last = -np.inf
        edge_count = 0
        for hit in hits:
            position = fraction[hit] * length
            if position - last <= tolerance:
                continue
            last = position
            edge_count += 1
            if len(witnesses) < witness_limit:
                witnesses.append(
                    {
                        "source_edge_vertex_ids": edge.tolist(),
                        "target_original_face_id": int(candidates[hit]),
                        "edge_fraction": float(fraction[hit]),
                        "target_barycentric": [
                            float(1 - u[hit] - v[hit]),
                            float(u[hit]),
                            float(v[hit]),
                        ],
                        "point": (start + fraction[hit] * direction).tolist(),
                    }
                )
        count += edge_count
        crossed_edges += bool(edge_count)
    return {
        "intersection_locations": count,
        "intersecting_edges": crossed_edges,
        "source_edges_tested": len(edges),
        "target_faces_tested": len(tf),
        "witnesses": witnesses,
        "limitations": "Excludes coplanar overlap, endpoint touches and containment; no depth.",
    }


def audit_surfaces(a_vertices, a_faces, b_vertices, b_faces):
    """Independent bidirectional unsigned and original-edge diagnostic.

    Vertex distances have incomplete surface coverage; edge intersections can
    detect triangle-interior crossings missed by either vertex query direction.
    Even both directions cannot certify absence of penetration/containment.
    """
    av, af = _mesh(a_vertices, a_faces)
    bv, bf = _mesh(b_vertices, b_faces)

    def unsigned(points, vertices, faces):
        distances = nearest_surface(points, vertices, faces).distance
        return {
            "query_vertices": len(points),
            "min": float(distances.min()),
            "mean": float(distances.mean()),
            "max": float(distances.max()),
        }

    return {
        "a_to_b_unsigned": unsigned(av, bv, bf),
        "b_to_a_unsigned": unsigned(bv, av, af),
        "a_edges_to_b": edge_triangle_crossings(av, af, bv, bf),
        "b_edges_to_a": edge_triangle_crossings(bv, bf, av, af),
        "signed_penetration": None,
        "limitations": "Unsigned vertex coverage and non-coplanar crossings only; "
        "no containment, force closure or physical validation.",
    }
