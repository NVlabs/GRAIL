"""Optional float64 nearest-original-triangle queries with compiled traversal.

No geometry simplification or solid inside/outside inference. The reference is
repair_geometry.closest_points; this module is opt-in and does not change it.
"""

from functools import lru_cache

import numpy as np


def _reduction_order(operation):
    """Detect NumPy's local three-column grouping; fail if it is unsupported."""
    rng = np.random.default_rng(20261024)
    a, b = rng.normal(size=(257, 3)), rng.normal(size=(257, 3))
    possible = {0, 1, 2}
    for count in (1, 2, 3, 8, 16, 32, 257):
        x, y = a[:count], b[:count]
        values = x*y
        expected = (np.einsum("ij,ij->i", x, y) if operation == "einsum"
                    else np.sum(values, axis=1))
        variants = [(values[:, 0]+values[:, 1])+values[:, 2],
                    (values[:, 0]+values[:, 2])+values[:, 1],
                    values[:, 0]+(values[:, 1]+values[:, 2])]
        possible.intersection_update(i for i, v in enumerate(variants)
                                     if np.array_equal(v, expected))
    if not possible:
        raise RuntimeError("NumPy reduction grouping unsupported by exact optional contact query")
    return min(possible)


@lru_cache(maxsize=1)
def _compiled_query():
    from numba import njit

    dot_order, sum_order = _reduction_order("einsum"), _reduction_order("sum")

    @njit(cache=False)
    def dot(a, b):
        # NumPy's float64 einsum ij,ij->i reduction uses (0+2)+1 for
        # three columns. Preserve that order: material identity at exact
        # shared edges can change under a different floating-point grouping.
        if dot_order == 0:
            return (a[0]*b[0] + a[1]*b[1]) + a[2]*b[2]
        if dot_order == 1:
            return (a[0]*b[0] + a[2]*b[2]) + a[1]*b[1]
        return a[0]*b[0] + (a[1]*b[1] + a[2]*b[2])

    @njit(cache=False)
    def norm_squared(a):
        # The reference squared-distance uses np.sum, whose order differs
        # from einsum above. Do not merge these operations with fastmath.
        if sum_order == 0:
            return (a[0]*a[0] + a[1]*a[1]) + a[2]*a[2]
        if sum_order == 1:
            return (a[0]*a[0] + a[2]*a[2]) + a[1]*a[1]
        return a[0]*a[0] + (a[1]*a[1] + a[2]*a[2])

    @njit(cache=False)
    def cross(a, b):
        return np.array((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                         a[0]*b[1]-a[1]*b[0]))

    @njit(cache=False)
    def nearest(point, triangle, valid, normal):
        best = np.inf
        answer, bary = np.zeros(3), np.zeros(3)
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = triangle[second]-triangle[first]
            denominator = dot(edge, edge)
            fraction = 0.
            if denominator > 0:
                fraction = min(1., max(0., dot(point-triangle[first], edge)/denominator))
            candidate = triangle[first] + fraction*edge
            delta = candidate-point
            distance = norm_squared(delta)
            if distance < best:
                best = distance
                answer = candidate
                bary = np.zeros(3)
                bary[first], bary[second] = 1-fraction, fraction
        if valid:
            signed = dot(point-triangle[0], normal)
            projection = point-signed*normal
            e0, e1, relative = triangle[1]-triangle[0], triangle[2]-triangle[0], projection-triangle[0]
            area = cross(e0, e1)
            denominator = dot(area, area)
            v, w = 0., 0.
            if denominator > 0:
                v = dot(cross(relative, e1), area)/denominator
                w = dot(cross(e0, relative), area)/denominator
            u = 1-v-w
            if u >= 0 and v >= 0 and w >= 0:
                answer = projection
                bary = np.array((u, v, w))
                delta = answer-point
                best = norm_squared(delta)
        return answer, bary, best

    @njit(cache=False)
    def box_bound(point, lower, upper):
        total = 0.
        for axis in range(3):
            delta = max(0., lower[axis]-point[axis], point[axis]-upper[axis])
            total += delta*delta
        return max(0., total - (8*np.finfo(np.float64).eps)*total)

    @njit(cache=False)
    def query(points, triangles, valid, normals, lower, upper, left, right, offsets, counts, faces):
        closest, barycentric = np.empty_like(points), np.empty_like(points)
        face_ids = np.empty(len(points), dtype=np.int64)
        distances2 = np.empty(len(points))
        pending = np.empty(len(left)+1, dtype=np.int64)
        bounds = np.empty(len(left)+1)
        for index in range(len(points)):
            point = points[index]
            best, best_face = np.inf, -1
            pending[0], bounds[0], size = 0, 0., 1
            while size:
                size -= 1
                node, bound = pending[size], bounds[size]
                if bound > best:
                    continue
                if left[node] < 0:
                    for at in range(offsets[node], offsets[node]+counts[node]):
                        face = faces[at]
                        candidate, bary, d2 = nearest(point, triangles[face], valid[face], normals[face])
                        if d2 < best or (d2 == best and face < best_face):
                            best, best_face = d2, face
                            closest[index], barycentric[index] = candidate, bary
                else:
                    a, b = left[node], right[node]
                    ba = box_bound(point, lower[a], upper[a])
                    bb = box_bound(point, lower[b], upper[b])
                    # Visit the nearer child first; retain equality for face-ID ties.
                    if ba < bb:
                        a, b, ba, bb = b, a, bb, ba
                    if ba <= best:
                        pending[size], bounds[size] = a, ba
                        size += 1
                    if bb <= best:
                        pending[size], bounds[size] = b, bb
                        size += 1
            face_ids[index], distances2[index] = best_face, best
        return closest, barycentric, face_ids, distances2

    return query


class OriginalTriangleQuery:
    """Reusable flattened traversal over an immutable TriangleMesh instance."""

    def __init__(self, mesh):
        from .repair_geometry import TriangleMesh

        if not isinstance(mesh, TriangleMesh):
            raise ValueError("mesh must be an original TriangleMesh")
        if not len(mesh.faces):
            raise ValueError("closest point query requires at least one triangle")
        self.mesh = mesh
        nodes = mesh._topology.nodes
        self.left = np.array([node[0] for node in nodes], dtype=np.int64)
        self.right = np.array([node[1] for node in nodes], dtype=np.int64)
        self.offsets = np.zeros(len(nodes), dtype=np.int64)
        self.counts = np.zeros(len(nodes), dtype=np.int64)
        for node, offset in zip(mesh._topology.leaf_nodes, mesh._topology.leaf_offsets):
            self.offsets[node] = offset
            self.counts[node] = len(nodes[node][2])
        for array in (self.left, self.right, self.offsets, self.counts):
            array.setflags(write=False)

    def query(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be a finite N x 3 array")
        mesh = self.mesh
        closest, bary, ids, d2 = _compiled_query()(
            np.ascontiguousarray(points), mesh.triangles, mesh.normal_valid, mesh.normals,
            mesh._node_lower, mesh._node_upper, self.left, self.right,
            self.offsets, self.counts, mesh._topology.face_order)
        normals = mesh.normals[ids].copy()
        return dict(points=closest, face_ids=ids, barycentric=bary, distances=np.sqrt(d2),
                    normals=normals, normal_valid=mesh.normal_valid[ids].copy(),
                    signed_halfspace=np.einsum("ij,ij->i", points-closest, normals))


def closest_points(points, mesh):
    """Stateless opt-in equivalent; reuse OriginalTriangleQuery for repeated calls."""
    return OriginalTriangleQuery(mesh).query(points)
