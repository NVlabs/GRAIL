"""Geometric material-contact patches on original surfaces, never probabilities.

Patch members are explicitly authored; no anatomy or finger participation is
inferred. Missing close, distinct pairs remain a coverage shortfall.
"""
from collections import OrderedDict
import time

import numpy as np

from .contact_edit.pose import _integer, _keys
from .grasp_io import number
from .repair_geometry import closest_points


class AnchorQuerySession:
    """Explicit, per-bank CPU query backend; never caches posed object queries.

    Robot-local projections are keyed by complete geometry and float64 query
    bytes using ExactSurfaceQueryCache. Compiled traversals retain immutable
    TriangleMesh instances only; new transformed meshes get separate traversals.
    """

    def __init__(self, config):
        from .contact_query_cache import ExactSurfaceQueryCache
        _keys(config, {'backend'}, {'max_cache_entries', 'max_mesh_entries'}, 'anchor_queries')
        backend = config['backend']
        if backend not in ('python_cached', 'numba_cached'):
            raise ValueError('anchor_queries.backend must be python_cached or numba_cached')
        cache_count = _integer(config.get('max_cache_entries', 256), 'max_cache_entries', 1)
        mesh_count = _integer(config.get('max_mesh_entries', 64), 'max_mesh_entries', 1)
        if cache_count > 4096 or mesh_count > 256:
            raise ValueError('anchor query caches require <=4096 results and <=256 meshes')
        self.config = dict(backend=backend, max_cache_entries=cache_count, max_mesh_entries=mesh_count)
        self.engines = OrderedDict()
        self.local_cache = ExactSurfaceQueryCache(self._query, max_entries=cache_count)
        self.metrics = dict(robot_projection_calls=0, object_projection_calls=0,
                            robot_projection_seconds=0., object_projection_seconds=0.,
                            compiled_mesh_builds=0)

    @staticmethod
    def _immutable_mesh(mesh):
        from .repair_geometry import TriangleMesh
        if (not isinstance(mesh, TriangleMesh) or mesh.vertices.flags.writeable
                or mesh.faces.flags.writeable):
            raise ValueError('Anchor query reuse requires immutable original TriangleMesh geometry')

    def _query(self, points, mesh):
        if self.config['backend'] == 'python_cached':
            return closest_points(points, mesh)
        from .contact_query_numba import OriginalTriangleQuery
        key = id(mesh)
        if key not in self.engines:
            self.engines[key] = OriginalTriangleQuery(mesh)
            self.metrics['compiled_mesh_builds'] += 1
            if len(self.engines) > self.config['max_mesh_entries']:
                self.engines.popitem(last=False)
        # The engine retains the mesh, preventing id reuse while cached.
        self.engines.move_to_end(key)
        return self.engines[key].query(points)

    def robot_surface(self, points, mesh):
        self._immutable_mesh(mesh)
        started = time.monotonic()
        result = self.local_cache(points, mesh)
        self.metrics['robot_projection_seconds'] += time.monotonic()-started
        self.metrics['robot_projection_calls'] += 1
        return result

    def object_surface(self, points, mesh):
        self._immutable_mesh(mesh)
        started = time.monotonic()
        # Current posed points are queried on every call; no candidate/pose key
        # can accidentally return a previous candidate's nearest material face.
        result = self._query(points, mesh)
        self.metrics['object_projection_seconds'] += time.monotonic()-started
        self.metrics['object_projection_calls'] += 1
        return result

    def counters(self):
        return dict(self.metrics, robot_cache_hits=self.local_cache.hits,
                    robot_cache_misses=self.local_cache.misses)

    def stats(self):
        return dict(config=dict(self.config), device='cpu', precision='float64',
                    **self.counters(), robot_cache=self.local_cache.stats(),
                    compiled_mesh_entries=len(self.engines),
                    reference='repair_geometry.closest_points',
                    exact_triangle_ties='Lowest original face index; existing float64 reduction order.',
                    robot_cache_key='Full float64 local query bytes plus original vertices/faces dtype, shape and bytes.',
                    object_query_results_cached=False,
                    compilation_included_in_query_seconds=self.config['backend'] == 'numba_cached',
                    query_seconds_included_in_anchor_extraction=True,
                    geometry_modified=False, selection_policy_modified=False,
                    motion_accepted=False)


def create_anchor_query_session(config):
    """Absent config preserves the original uncached Python query path."""
    return None if config is None else AnchorQuerySession(config)


def validate_contact_patch(config, hands):
    """Validate the opt-in patch policy and explicit per-group geom partition."""
    if config is None:
        return None
    _keys(config, {"points_per_member", "max_gap_m", "min_spacing_m", "members"},
          set(), "contact_patch")
    count = _integer(config["points_per_member"], "points_per_member", 1)
    if count > 64:
        raise ValueError("points_per_member must be <= 64")
    gap = number(config["max_gap_m"], "max_gap_m", 0, .1)
    spacing = number(config["min_spacing_m"], "min_spacing_m", 0, .1)
    members = config["members"]
    if not isinstance(members, dict) or set(members) != set(hands):
        raise ValueError("contact_patch members must cover exactly the configured hands")
    for hand, groups in members.items():
        expected = hands[hand]["contact_groups"]
        if not isinstance(groups, dict) or set(groups) != set(expected):
            raise ValueError("contact_patch members must cover exactly the hand's contact groups")
        for group, named in groups.items():
            if not isinstance(named, dict) or not named:
                raise ValueError("Each contact group needs explicit patch members")
            used = set()
            for member, geoms in named.items():
                if not isinstance(member, str) or not member.strip():
                    raise ValueError("Patch members need nonempty names")
                if (not isinstance(geoms, list) or not geoms
                        or any(type(g) is not int for g in geoms)
                        or len(set(geoms)) != len(geoms)
                        or not set(geoms) <= set(expected[group])
                        or used.intersection(geoms)):
                    raise ValueError("Patch member geoms must form a disjoint partition of their group")
                used.update(geoms)
            if used != set(expected[group]):
                raise ValueError("Patch member geoms must cover their complete contact group")
    return dict(points_per_member=count, max_gap_m=gap, min_spacing_m=spacing,
                members=members)


def _points(value, name):
    a = np.asarray(value, dtype=float)
    if a.size == 0:
        a = a.reshape(0, 3)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite points with shape [N,3]")
    return a


def select_patch_indices(hand_points, object_points, gaps, *, points_per_member,
                         max_gap_m, min_spacing_m, reserved_hand_points=None,
                         reserved_object_points=None):
    """Closest seed then farthest-point sampling, spaced on BOTH surfaces.

    All points use one metric coordinate frame. Reserved points prevent repeated
    material contacts across members/groups. No out-of-range point pads a patch.
    """
    hand = _points(hand_points, "hand_points")
    obj = _points(object_points, "object_points")
    gaps = np.asarray(gaps, dtype=float)
    if obj.shape != hand.shape or gaps.shape != (len(hand),):
        raise ValueError("Patch point and gap arrays must have matching lengths")
    if not np.isfinite(gaps).all() or np.any(gaps < 0):
        raise ValueError("Patch gaps must be finite and nonnegative")
    count = _integer(points_per_member, "points_per_member", 1)
    if count > 64:
        raise ValueError("points_per_member must be <= 64")
    max_gap = number(max_gap_m, "max_gap_m", 0, .1)
    spacing = number(min_spacing_m, "min_spacing_m", 0, .1)
    # Even an explicitly zero spacing must not duplicate material points.
    separation = max(spacing, 1e-9)
    available = gaps <= max_gap
    eligible = int(available.sum())
    min_distance = np.full(len(hand), np.inf)
    for field, reserved in ((hand, reserved_hand_points), (obj, reserved_object_points)):
        if reserved is not None:
            for point in _points(reserved, "reserved_points"):
                distance = np.linalg.norm(field-point, axis=1)
                available &= distance >= separation
                min_distance = np.minimum(min_distance, distance)
    selected = []
    while len(selected) < count and available.any():
        ids = np.flatnonzero(available)
        if not selected:
            index = int(ids[np.argmin(gaps[ids])])
        else:
            # Stable tie breaking preserves reproducibility on duplicate meshes.
            index = int(ids[np.argmax(min_distance[ids])])
        selected.append(index)
        distance = np.minimum(np.linalg.norm(hand-hand[index], axis=1),
                              np.linalg.norm(obj-obj[index], axis=1))
        min_distance = np.minimum(min_distance, distance)
        available &= distance >= separation
        available[index] = False
    complete = len(selected) == count
    return np.asarray(selected, dtype=int), dict(
        status="COMPLETE" if complete else "SHORTFALL", coverage_complete=complete,
        requested_count=count, selected_count=len(selected), eligible_count=eligible,
        missing_count=count-len(selected), max_gap_m=max_gap, min_spacing_m=spacing,
        spacing_scope="both posed hand and object surfaces, including reserved pairs",
        selection="closest seed, then farthest distinct pair; no radius restriction",
        evidence="GEOMETRIC_PROXIMITY_HYPOTHESIS", motion_accepted=False)


def extract_contact_patch(*, group, member, member_geoms, points, point_geoms,
                          geom_transforms, meshes, body_names, object_mesh,
                          object_face_ids, points_per_member, max_gap_m,
                          min_spacing_m, reserved_hand_points=None,
                          reserved_object_points=None, anchor_queries=None):
    """Export exact material pairs; region face IDs map to ORIGINAL object faces.

    ``points`` are body-local samples, ``meshes`` original body-local triangles,
    and each geom transform maps that body frame to the scaled object frame.
    ``object_mesh`` is the eligible region (not a replacement object surface).
    Optional ``anchor_queries`` is one explicit AnchorQuerySession shared across
    candidate calls. It changes query execution only, never patch selection.
    """
    if anchor_queries is not None and not isinstance(anchor_queries, AnchorQuerySession):
        raise ValueError('anchor_queries must be an explicit AnchorQuerySession')
    query_before = anchor_queries.counters() if anchor_queries is not None else None
    robot_query = anchor_queries.robot_surface if anchor_queries is not None else closest_points
    object_query = anchor_queries.object_surface if anchor_queries is not None else closest_points
    local = _points(points, "points")
    geoms = np.asarray(point_geoms)
    face_ids = np.asarray(object_face_ids)
    if geoms.shape != (len(local),) or geoms.dtype.kind not in "iu":
        raise ValueError("point_geoms must be integer geom IDs for every point")
    if (face_ids.shape != (len(object_mesh.faces),) or face_ids.dtype.kind not in "iu"
            or np.any(face_ids < 0) or len(np.unique(face_ids)) != len(face_ids)):
        raise ValueError("object_face_ids must map region triangles to unique original faces")
    ids = np.flatnonzero(np.isin(geoms, member_geoms))
    exact_local = np.empty((len(ids), 3))
    posed = np.empty_like(exact_local)
    robot_faces = np.empty(len(ids), dtype=int)
    robot_bary = np.empty_like(exact_local)
    robot_normals = np.empty_like(exact_local)
    for geom in member_geoms:
        subset = np.flatnonzero(geoms[ids] == geom)
        if not len(subset):
            continue
        matrix = np.asarray(geom_transforms[geom], dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
            raise ValueError("Patch geom transforms must be finite rigid body-to-object transforms")
        surface = robot_query(local[ids[subset]], meshes[geom])
        exact_local[subset] = surface["points"]
        posed[subset] = surface["points"] @ matrix[:3, :3].T + matrix[:3, 3]
        robot_faces[subset] = surface["face_ids"]
        robot_bary[subset] = surface["barycentric"]
        robot_normals[subset] = surface["normals"]
    nearest = object_query(posed, object_mesh)
    selected, report = select_patch_indices(
        posed, nearest["points"], nearest["distances"], points_per_member=points_per_member,
        max_gap_m=max_gap_m, min_spacing_m=min_spacing_m,
        reserved_hand_points=reserved_hand_points, reserved_object_points=reserved_object_points)
    anchors = []
    for ordinal, index in enumerate(selected):
        geom = int(geoms[ids[index]])
        anchors.append(dict(
            name=f"{group}/{member}/{ordinal}", group=group, member=member, geom=geom,
            body=body_names[geom], body_local_point=exact_local[index].tolist(),
            robot_face=int(robot_faces[index]), robot_barycentric=robot_bary[index].tolist(),
            body_local_normal=robot_normals[index].tolist(), normal_weight=0,
            object_face=int(face_ids[nearest["face_ids"][index]]),
            object_barycentric=nearest["barycentric"][index].tolist(),
            gap_m=float(nearest["distances"][index]), position_weight=1/len(selected),
            hand_point_object=posed[index].tolist(), object_point=nearest["points"][index].tolist(),
            evidence="GEOMETRIC_PROXIMITY_HYPOTHESIS"))
    report.update(group=group, member=member, member_geoms=list(member_geoms),
                  selected_max_gap_m=max((a["gap_m"] for a in anchors), default=None))
    if anchor_queries is not None:
        report['query_execution'] = dict(backend=anchor_queries.config['backend'],
            **{key: value-query_before[key] for key, value in anchor_queries.counters().items()})
    return anchors, report
