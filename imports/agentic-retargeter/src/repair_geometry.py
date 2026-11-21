"""Original-triangle geometry evidence for bounded native motion repair.

No geometry is decimated, repaired or treated as a closed solid. Intersection
queries cover triangle surfaces at the supplied pose, including coplanar overlap
and endpoint touches. Degenerate faces make a negative audit inconclusive.
Containment without surface intersection and continuous-time collision are not
tested. A nearest facet's oriented halfspace is an optimizer aid, *not* a signed
distance to a solid or a penetration depth (even for closed meshes near corners).
"""

from __future__ import annotations

import heapq
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class _TreeTopology:
    """Read-only face partition shared by all affine poses of one mesh."""

    nodes: tuple
    leaf_nodes: np.ndarray
    leaf_offsets: np.ndarray
    face_order: np.ndarray
    levels: tuple

    @classmethod
    def build(cls, lower, upper):
        nodes, levels, leaf_nodes, leaf_faces = [], [], [], []
        center = (lower + upper) * 0.5

        def partition(ids, depth=0):
            index = len(nodes)
            nodes.append(None)
            if len(ids) <= 16:
                ids.setflags(write=False)
                leaf_nodes.append(index)
                leaf_faces.append(ids)
                nodes[index] = (-1, -1, ids)
            else:
                while len(levels) <= depth:
                    levels.append([])
                levels[depth].append(index)
                local = center[ids]
                axis = int(np.argmax(np.ptp(local, axis=0)))
                cut = len(ids) // 2
                order = np.argpartition(local[:, axis], cut)
                left = partition(ids[order[:cut]], depth + 1)
                right = partition(ids[order[cut:]], depth + 1)
                nodes[index] = (left, right, None)
            return index

        if len(lower):
            partition(np.arange(len(lower)))
        order = np.concatenate(leaf_faces) if leaf_faces else np.empty(0, dtype=int)
        offsets = np.cumsum([0] + [len(ids) for ids in leaf_faces[:-1]])[
            : len(leaf_faces)
        ]
        leaves = np.asarray(leaf_nodes, dtype=int)
        branches = tuple(
            tuple(
                np.asarray(values, dtype=int)
                for values in (
                    level,
                    [nodes[i][0] for i in level],
                    [nodes[i][1] for i in level],
                )
            )
            for level in reversed(levels)
        )
        for array in (
            order,
            offsets,
            leaves,
            *(a for level in branches for a in level),
        ):
            array.setflags(write=False)
        return cls(tuple(nodes), leaves, offsets, order, branches)

    def refit(self, lower, upper, *, with_bounds=False, lazy_nodes=False):
        # Reduce actual posed triangle bounds, not transformed local boxes. The
        # same partition therefore remains exact under shear, reflection, or a
        # singular affine transform. All faces occur in exactly one leaf.
        lo = np.empty((len(self.nodes), 3))
        hi = np.empty_like(lo)
        if len(self.nodes):
            lo[self.leaf_nodes] = np.minimum.reduceat(
                lower[self.face_order], self.leaf_offsets
            )
            hi[self.leaf_nodes] = np.maximum.reduceat(
                upper[self.face_order], self.leaf_offsets
            )
            for nodes, left, right in self.levels:
                lo[nodes] = np.minimum(lo[left], lo[right])
                hi[nodes] = np.maximum(hi[left], hi[right])
        lo.setflags(write=False)
        hi.setflags(write=False)
        tree = (
            _PosedTreeNodes(self, lo, hi)
            if lazy_nodes and self.nodes
            else tuple((lo[i], hi[i], *node) for i, node in enumerate(self.nodes))
        )
        return (tree, lo, hi) if with_bounds else tree


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _PosedTreeNodes(Sequence):
    """Immutable sequence of tuple-shaped nodes backed by this pose's bounds.

    Original meshes retain materialized tuples. Poses share only the immutable
    face partition and own their bounds; building another pose cannot invalidate
    a retained node/view. Optional array-based traversal need not allocate a
    Python tuple and two array views for every node it never visits.
    """

    topology: _TreeTopology
    lower: np.ndarray
    upper: np.ndarray

    def __len__(self):
        return len(self.topology.nodes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        # Match tuple indexing, including Boolean and NumPy integer indices.
        # Passing a Boolean straight to NumPy would add an array dimension.
        index = operator.index(index)
        return (self.lower[index], self.upper[index], *self.topology.nodes[index])


@dataclass(frozen=True)
class TriangleMesh:
    """Immutable original vertices/faces; affine poses reuse a face partition."""

    vertices: np.ndarray
    faces: np.ndarray
    triangles: np.ndarray = field(init=False, repr=False)
    normals: np.ndarray = field(init=False, repr=False)
    normal_valid: np.ndarray = field(init=False, repr=False)
    _lower: np.ndarray = field(init=False, repr=False)
    _upper: np.ndarray = field(init=False, repr=False)
    # Private readonly node sequence: original meshes and empty poses use
    # tuples; nonempty poses construct tuple-shaped nodes only when accessed.
    _tree: Sequence[tuple] = field(init=False, repr=False)
    # Explicit immutable arrays for optional traversal backends. Tuple nodes
    # continue to share these bounds; callers need not inspect NumPy .base.
    _node_lower: np.ndarray = field(init=False, repr=False)
    _node_upper: np.ndarray = field(init=False, repr=False)
    _topology: _TreeTopology = field(init=False, repr=False)

    def __post_init__(self):
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        faces = np.asarray(self.faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("vertices must be a finite N x 3 array")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
        ):
            raise ValueError("faces must be an integer F x 3 array")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("face vertex index out of bounds")
        self._set_geometry(vertices, np.array(faces, dtype=np.int64, copy=True))

    def _set_geometry(self, vertices, faces, topology=None):
        posed = topology is not None
        tri = vertices[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(cross, axis=1)
        scale = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1) * np.linalg.norm(
            tri[:, 2] - tri[:, 0], axis=1
        )
        valid = length > 64 * np.finfo(float).eps * scale
        normals = np.divide(
            cross, length[:, None], out=np.zeros_like(cross), where=valid[:, None]
        )
        # Every face has exactly three vertices. Elementwise extrema avoid
        # short strided reductions without changing the original float64 bounds.
        lower = np.minimum(np.minimum(tri[:, 0], tri[:, 1]), tri[:, 2])
        upper = np.maximum(np.maximum(tri[:, 0], tri[:, 1]), tri[:, 2])
        if topology is None:
            topology = _TreeTopology.build(lower, upper)
        tree, node_lower, node_upper = topology.refit(
            lower, upper, with_bounds=True, lazy_nodes=posed
        )
        for name, value in (
            ("vertices", vertices),
            ("faces", faces),
            ("triangles", tri),
            ("normals", normals),
            ("normal_valid", valid),
            ("_lower", lower),
            ("_upper", upper),
            ("_topology", topology),
            ("_tree", tree),
            ("_node_lower", node_lower),
            ("_node_upper", node_upper),
        ):
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    def transformed(self, matrix: np.ndarray) -> TriangleMesh:
        """Return an affine pose, retaining the immutable original face partition.

        Exact bounds and winding normals are recomputed from posed vertices,
        including reflections/nonuniform scale. No outward normal is inferred.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("transform must be a finite 4 x 4 affine matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError("projective transforms are not supported")
        vertices = self.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.isfinite(vertices).all():
            raise ValueError("transformed vertices must be finite")
        posed = object.__new__(TriangleMesh)
        posed._set_geometry(vertices, self.faces, self._topology)
        return posed


def _tolerance(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("tolerance must be finite and positive")
    return value


def _overlap(lo_a, hi_a, lo_b, hi_b, tolerance):
    return np.all((hi_a >= lo_b - tolerance) & (hi_b >= lo_a - tolerance), axis=-1)


def _candidate_pairs(a, b, tolerance):
    """Dual original-triangle AABB tree; each leaf pair visited once."""
    if not a._tree or not b._tree:
        return
    pending = [(0, 0)]
    batches_a, batches_b, count = [], [], 0
    while pending:
        ai, bi = pending.pop()
        al, ah, ac1, ac2, af = a._tree[ai]
        bl, bh, bc1, bc2, bf = b._tree[bi]
        if (
            ah[0] < bl[0] - tolerance
            or bh[0] < al[0] - tolerance
            or ah[1] < bl[1] - tolerance
            or bh[1] < al[1] - tolerance
            or ah[2] < bl[2] - tolerance
            or bh[2] < al[2] - tolerance
        ):
            continue
        if af is not None and bf is not None:
            keep = _overlap(
                a._lower[af, None],
                a._upper[af, None],
                b._lower[None, bf],
                b._upper[None, bf],
                tolerance,
            )
            ia, ib = np.nonzero(keep)
            if len(ia):
                batches_a.append(af[ia])
                batches_b.append(bf[ib])
                count += len(ia)
                # Keep the same exhaustive candidate set, while amortizing the
                # vectorized narrowphase over more than one small tree leaf.
                if count >= 4096:
                    yield np.concatenate(batches_a), np.concatenate(batches_b)
                    batches_a, batches_b, count = [], [], 0
        elif bf is not None or (af is None and np.max(ah - al) >= np.max(bh - bl)):
            pending.extend(((ac1, bi), (ac2, bi)))
        else:
            pending.extend(((ai, bc1), (ai, bc2)))
    if count:
        yield np.concatenate(batches_a), np.concatenate(batches_b)


def _barycentric(points, tri):
    e0, e1, relative = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], points - tri[:, 0]
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    cross = np.cross(e0, e1)
    denominator = dot(cross, cross)
    v = np.divide(
        dot(np.cross(relative, e1), cross),
        denominator,
        out=np.zeros(len(tri)),
        where=denominator > 0,
    )
    w = np.divide(
        dot(np.cross(e0, relative), cross),
        denominator,
        out=np.zeros(len(tri)),
        where=denominator > 0,
    )
    return np.column_stack((1 - v - w, v, w))


def _inside(points, tri, tolerance):
    bary = _barycentric(points, tri)
    # Convert the spatial tolerance to a barycentric bound through each altitude.
    area2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    edges = np.stack(
        (tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2], tri[:, 1] - tri[:, 0]), axis=1
    )
    limits = tolerance * np.linalg.norm(edges, axis=2) / area2[:, None]
    return np.all(bary >= -limits - 64 * np.finfo(float).eps, axis=1)


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _coplanar_witness(a, b, normal, tolerance):
    """Clipped projected triangle polygon; detects overlap and boundary touches."""
    drop = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != drop]
    clip = b[:, keep]
    orientation = 1 if _cross2(clip[1] - clip[0], clip[2] - clip[0]) >= 0 else -1
    polygon = [vertex.copy() for vertex in a]
    for index in range(3):
        start, end = clip[index], clip[(index + 1) % 3]
        edge = end - start
        length = np.linalg.norm(edge)
        previous = polygon[-1] if polygon else None
        result = []
        for current in polygon:
            dp = orientation * _cross2(edge, previous[keep] - start) / length
            dc = orientation * _cross2(edge, current[keep] - start) / length
            previous_inside, current_inside = dp >= -tolerance, dc >= -tolerance
            if previous_inside != current_inside:
                # Clip to the tolerance-expanded line, retaining touches.
                fraction = (dp + tolerance) / (dp - dc)
                result.append(previous + fraction * (current - previous))
            if current_inside:
                result.append(current)
            previous = current
        polygon = result
        if not polygon:
            return None
    return np.mean(polygon, axis=0)


def _triangle_hits(a, b, na, nb, tolerance):
    """Vectorized two-direction edge/triangle tests for candidate face pairs."""
    da = np.einsum("nkj,nj->nk", a - b[:, :1], nb)
    db = np.einsum("nkj,nj->nk", b - a[:, :1], na)
    coplanar = (np.max(np.abs(da), axis=1) <= tolerance) & (
        np.max(np.abs(db), axis=1) <= tolerance
    )
    possible = (
        (da.min(1) <= tolerance)
        & (da.max(1) >= -tolerance)
        & (db.min(1) <= tolerance)
        & (db.max(1) >= -tolerance)
    )
    hits = np.zeros(len(a), dtype=bool)
    points = np.zeros((len(a), 3))
    kinds = np.full(len(a), "touch", dtype="U8")
    proper = (
        (da.min(1) < -tolerance)
        & (da.max(1) > tolerance)
        & (db.min(1) < -tolerance)
        & (db.max(1) > tolerance)
    )
    kinds[proper] = "crossing"
    for source, target, distance in ((a, b, da), (b, a, db)):
        for edge in range(3):
            nxt = (edge + 1) % 3
            ids = np.flatnonzero(possible & ~coplanar & ~hits)
            if not len(ids):
                continue
            d0, d1 = distance[ids, edge], distance[ids, nxt]
            denominator = d0 - d1
            segment_length = np.linalg.norm(
                source[ids, nxt] - source[ids, edge], axis=1
            )
            fraction = np.divide(
                d0, denominator, out=np.zeros_like(d0), where=denominator != 0
            )
            fraction_tolerance = tolerance / np.maximum(segment_length, tolerance)
            crosses = (
                (denominator != 0)
                & (fraction >= -fraction_tolerance)
                & (fraction <= 1 + fraction_tolerance)
            )
            on_plane = np.abs(d0) <= tolerance
            fraction[on_plane] = 0
            valid = crosses | on_plane
            point = source[ids, edge] + np.clip(fraction, 0, 1)[:, None] * (
                source[ids, nxt] - source[ids, edge]
            )
            valid &= _inside(point, target[ids], tolerance)
            found = ids[valid]
            hits[found], points[found] = True, point[valid]
    for index in np.flatnonzero(coplanar):
        point = _coplanar_witness(a[index], b[index], na[index], tolerance)
        if point is not None:
            hits[index], points[index], kinds[index] = True, point, "coplanar"
    return hits, points, kinds


def _validated_candidates(batches, a, b, tolerance):
    """Validate and bound external batches; completeness is the provider contract."""
    for batch in batches:
        if len(batch) != 2:
            raise ValueError("candidate batch must contain paired face-ID arrays")
        ai, bi = (np.asarray(ids) for ids in batch)
        if (
            ai.ndim != 1
            or bi.shape != ai.shape
            or not np.issubdtype(ai.dtype, np.integer)
            or not np.issubdtype(bi.dtype, np.integer)
        ):
            raise ValueError("candidate face IDs must be paired 1D integer arrays")
        if len(ai) and (
            ai.min() < 0
            or ai.max() >= len(a.faces)
            or bi.min() < 0
            or bi.max() >= len(b.faces)
        ):
            raise ValueError("candidate face ID outside original mesh")
        for start in range(0, len(ai), 4096):
            ia, ib = ai[start : start + 4096], bi[start : start + 4096]
            keep = _overlap(
                a._lower[ia], a._upper[ia], b._lower[ib], b._upper[ib], tolerance
            )
            if np.any(keep):
                yield ia[keep], ib[keep]


def mesh_intersections(
    mesh_a: TriangleMesh,
    mesh_b: TriangleMesh,
    *,
    witness_limit: int = 16,
    tolerance: float = 1e-9,
    candidate_pairs=None,
) -> dict:
    """Audit explicit original surfaces; counts refer to *face pairs*, not points.

    All face pairs are considered; limiting witnesses does not stop the audit.
    Witnesses use the lowest original face-ID pairs, independent of tree layout.
    ``complete`` is false when degenerate triangles were excluded. A complete
    negative result covers these sampled surfaces, not containment/solid volume.

    ``candidate_pairs`` is an internal acceleration hook: an iterable of paired
    integer face-ID arrays. The provider must include every overlapping original
    triangle AABB pair exactly once. Conservative extras are filtered here in
    float64. Missing/duplicate pairs cannot be certified without repeating the
    broadphase; only independently validated providers should use this hook.
    """
    tolerance = _tolerance(tolerance)
    if (
        isinstance(witness_limit, bool)
        or not isinstance(witness_limit, (int, np.integer))
        or witness_limit < 0
    ):
        raise ValueError("witness_limit must be a nonnegative integer")
    counts = {"crossing": 0, "touch": 0, "coplanar": 0}
    # The bounded witness set is canonical by original face IDs, so a refitted
    # tree and a freshly partitioned tree return the same repair constraints.
    witness_a = np.empty(0, dtype=int)
    witness_b = np.empty(0, dtype=int)
    witness_points = np.empty((0, 3))
    witness_kinds = np.empty(0, dtype="U8")
    candidates = 0
    batches = (
        _candidate_pairs(mesh_a, mesh_b, tolerance)
        if candidate_pairs is None
        else _validated_candidates(candidate_pairs, mesh_a, mesh_b, tolerance)
    )
    for ai, bi in batches:
        candidates += len(ai)
        valid = mesh_a.normal_valid[ai] & mesh_b.normal_valid[bi]
        ai, bi = ai[valid], bi[valid]
        if not len(ai):
            continue
        hit, points, kinds = _triangle_hits(
            mesh_a.triangles[ai],
            mesh_b.triangles[bi],
            mesh_a.normals[ai],
            mesh_b.normals[bi],
            tolerance,
        )
        for kind in counts:
            counts[kind] += int(np.sum(hit & (kinds == kind)))
        indices = np.flatnonzero(hit) if witness_limit else np.empty(0, dtype=int)
        if len(witness_a) == witness_limit and witness_limit:
            # Only a lexicographically better pair can enter the bounded set.
            better = (ai[indices] < witness_a[-1]) | (
                (ai[indices] == witness_a[-1]) & (bi[indices] < witness_b[-1])
            )
            indices = indices[better]
        if len(indices):
            all_a = np.concatenate((witness_a, ai[indices]))
            all_b = np.concatenate((witness_b, bi[indices]))
            keep = np.lexsort((all_b, all_a))[:witness_limit]
            witness_a, witness_b = all_a[keep], all_b[keep]
            witness_points = np.concatenate((witness_points, points[indices]))[keep]
            witness_kinds = np.concatenate((witness_kinds, kinds[indices]))[keep]
    witnesses = []
    if len(witness_a):
        bary_a = _barycentric(witness_points, mesh_a.triangles[witness_a])
        bary_b = _barycentric(witness_points, mesh_b.triangles[witness_b])
        for index in range(len(witness_a)):
            witnesses.append(
                {
                    "face_a": int(witness_a[index]),
                    "face_b": int(witness_b[index]),
                    "point": witness_points[index].tolist(),
                    "kind": str(witness_kinds[index]),
                    "barycentric_a": bary_a[index].tolist(),
                    "barycentric_b": bary_b[index].tolist(),
                }
            )
    degenerates = {
        "a": np.flatnonzero(~mesh_a.normal_valid).tolist(),
        "b": np.flatnonzero(~mesh_b.normal_valid).tolist(),
    }
    total = sum(counts.values())
    return {
        "intersecting": bool(total),
        "intersection_pair_count": total,
        "counts": counts,
        "witnesses": witnesses,
        "candidate_pair_count": candidates,
        "degenerate_faces": degenerates,
        "complete": not any(degenerates.values()),
        "tolerance": tolerance,
        "scope": "original triangle surfaces at the supplied pose; no containment or continuous-time test",
    }


def mesh_intersection_probe(
    mesh_a: TriangleMesh,
    mesh_b: TriangleMesh,
    *,
    tolerance: float = 1e-9,
    candidate_pairs=None,
) -> dict:
    """Decide surface intersection, stopping after a confirmed positive witness.

    Use for candidate screening or a per-pose collision flag. A negative decision
    still exhausts every candidate pair and rejects degenerate geometry. This
    deliberately does not return total intersection counts or canonical repair
    witnesses; use ``mesh_intersections`` when those are required. An early
    positive witness proves intersection, never clearance or penetration depth.
    The candidate-provider coverage contract is identical to mesh_intersections.
    """
    tolerance = _tolerance(tolerance)
    complete = bool(mesh_a.normal_valid.all() and mesh_b.normal_valid.all())
    batches = (_candidate_pairs(mesh_a, mesh_b, tolerance) if candidate_pairs is None
               else _validated_candidates(candidate_pairs, mesh_a, mesh_b, tolerance))
    examined = 0
    for ai, bi in batches:
        examined += len(ai)
        valid = mesh_a.normal_valid[ai] & mesh_b.normal_valid[bi]
        ai, bi = ai[valid], bi[valid]
        if not len(ai):
            continue
        hit, _, _ = _triangle_hits(mesh_a.triangles[ai], mesh_b.triangles[bi],
                                  mesh_a.normals[ai], mesh_b.normals[bi], tolerance)
        if hit.any():
            i = int(np.flatnonzero(hit)[0])
            return dict(intersecting=True, complete=complete,
                        decision_complete=True, pair_enumeration_complete=False,
                        candidate_pairs_examined=examined,
                        witness_face_ids=[int(ai[i]), int(bi[i])],
                        scope="Existence of an original-triangle surface intersection; no total counts")
    return dict(intersecting=False, complete=complete,
                decision_complete=complete, pair_enumeration_complete=True,
                candidate_pairs_examined=examined, witness_face_ids=None,
                scope="Exhaustive original-triangle surface decision; no containment/continuous guarantee")


def audit_pairs(
    meshes: Mapping[str, TriangleMesh], pairs: Sequence[Sequence[str]], **kwargs
) -> dict:
    """Audit exactly the named pairs; absent/self/duplicate pairs are rejected."""
    seen, rows = set(), []
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("each pair must name two distinct meshes")
        a, b = pair
        if a not in meshes or b not in meshes:
            raise ValueError(f"unknown mesh in pair: {pair}")
        identity = frozenset((a, b))
        if identity in seen:
            raise ValueError(f"duplicate mesh pair: {pair}")
        seen.add(identity)
        rows.append(
            {
                "mesh_a": a,
                "mesh_b": b,
                **mesh_intersections(meshes[a], meshes[b], **kwargs),
            }
        )
    return {
        "pairs": rows,
        "intersecting": any(row["intersecting"] for row in rows),
        "complete": all(row["complete"] for row in rows),
        "pair_count": len(rows),
    }


def _nearest_in_triangles(point, triangles, valid_normals, normals):
    count = len(triangles)
    # First choose the nearest edge, so degenerate triangles remain measurable.
    candidates, barys = [], []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, j] - triangles[:, i]
        denominator = np.einsum("ij,ij->i", edge, edge)
        numerator = np.einsum("ij,ij->i", point - triangles[:, i], edge)
        fraction = np.clip(
            np.divide(
                numerator, denominator, out=np.zeros(count), where=denominator > 0
            ),
            0,
            1,
        )
        candidates.append(triangles[:, i] + fraction[:, None] * edge)
        bary = np.zeros((count, 3))
        bary[:, i], bary[:, j] = 1 - fraction, fraction
        barys.append(bary)
    candidates = np.stack(candidates, axis=1)
    barys = np.stack(barys, axis=1)
    distance = np.sum((candidates - point) ** 2, axis=2)
    best = np.argmin(distance, axis=1)
    nearest, bary = (
        candidates[np.arange(count), best].copy(),
        barys[np.arange(count), best].copy(),
    )
    signed = np.einsum("ij,ij->i", point - triangles[:, 0], normals)
    projection = point - signed[:, None] * normals
    pb = _barycentric(projection, triangles)
    interior = valid_normals & np.all(pb >= 0, axis=1)
    nearest[interior], bary[interior] = projection[interior], pb[interior]
    return nearest, bary, np.sum((nearest - point) ** 2, axis=1)


def closest_points(points: np.ndarray, mesh: TriangleMesh) -> dict[str, np.ndarray]:
    """Exact nearest original triangles using AABB branch-and-bound.

    ``signed_halfspace`` = dot(query - closest point, winding facet normal).
    Positive means on the normal-facing side of that facet. Normals on degenerate
    faces are zero with ``normal_valid=False``. A repair constraint must explicitly
    establish the intended side; this function does not infer inside/outside.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite N x 3 array")
    if not len(mesh.faces):
        raise ValueError("closest point query requires at least one triangle")
    closest = np.empty_like(points)
    barycentric = np.empty_like(points)
    face_ids = np.empty(len(points), dtype=np.int64)
    distances2 = np.empty(len(points))
    for index, point in enumerate(points):
        best, best_face = np.inf, -1
        queue = [(0.0, 0)]
        while queue:
            lower_bound, node_index = heapq.heappop(queue)
            if lower_bound > best:
                break
            _lo, _hi, left, right, ids = mesh._tree[node_index]
            if ids is not None:
                candidates, bary, d2 = _nearest_in_triangles(
                    point,
                    mesh.triangles[ids],
                    mesh.normal_valid[ids],
                    mesh.normals[ids],
                )
                # Resolve ties by original face id, independent of tree partition.
                order = np.lexsort((ids, d2))
                winner = int(order[0])
                face = int(ids[winner])
                if d2[winner] < best or (d2[winner] == best and face < best_face):
                    best, best_face = float(d2[winner]), face
                    closest[index], barycentric[index] = (
                        candidates[winner],
                        bary[winner],
                    )
            else:
                for child in (left, right):
                    cl, ch = mesh._tree[child][:2]
                    delta = np.maximum(np.maximum(cl - point, point - ch), 0)
                    bound = float(np.sum(delta * delta))
                    # Conservative roundoff allowance prevents a box bound from
                    # pruning an equally close lower face ID (e.g. collapsed or
                    # duplicate faces) because dot/reduction rounding differs.
                    bound = max(0.0, bound - 8 * np.finfo(float).eps * bound)
                    if bound <= best:
                        heapq.heappush(queue, (bound, child))
        face_ids[index], distances2[index] = best_face, best
    normals = mesh.normals[face_ids].copy()
    return {
        "points": closest,
        "face_ids": face_ids,
        "barycentric": barycentric,
        "distances": np.sqrt(distances2),
        "normals": normals,
        "normal_valid": mesh.normal_valid[face_ids].copy(),
        "signed_halfspace": np.einsum("ij,ij->i", points - closest, normals),
    }
