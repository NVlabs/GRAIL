"""Portable hand and contact inspection primitives (NumPy only).

All coordinates and thresholds use metres. Measured maps are *unsigned surface
proximity*, not observed contact, penetration, pressure, or physical ground
truth. Predictions retain their provenance and exact original-vertex binding;
missing frame coverage never becomes a zero-valued prediction.
"""

import hashlib
from collections.abc import Mapping

import numpy as np


def _points(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{name} must be numeric (N, 3)")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _mesh(vertices, triangles):
    vertices = _points(vertices, "vertices")
    triangles = np.asarray(triangles)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu":
        raise ValueError("triangles must be integer (F, 3)")
    if not len(triangles) or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangles must be nonempty and index original vertices")
    return vertices, triangles.astype(np.int64, copy=False)


def mesh_digest(vertices, triangles):
    """SHA-256 of original local vertices/triangles, including their order.

    Matches :mod:`agentic_retargeter.contact`; normalizes byte order and numeric
    representation, without transforming, welding, or reordering any geometry.
    Pose/world coordinates must not be used as a stable prediction binding.
    """
    vertices, triangles = _mesh(vertices, triangles)
    digest = hashlib.sha256(b"grail-original-surface-v1\0")
    for array, dtype in ((vertices, "<f8"), (triangles, "<i8")):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()


def _rotation(value, name):
    raw = np.asarray(value)
    if raw.shape != (3, 3) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a proper 3x3 rotation")
    rotation = raw.astype(np.float64)
    if (
        not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be a proper rotation (no scale, shear or reflection)")
    return rotation


def canonical_vertices(world_vertices, wrist_matrix, canonical_rotation=None):
    """Express world vertices in a rigid wrist-local display coordinate frame.

    ``wrist_matrix`` maps wrist-local column vectors into world coordinates.
    Optional ``canonical_rotation`` rotates wrist-local vectors into the display
    frame; it must be a proper 3x3 rotation. No automatic left/right mirroring,
    pose normalization, geometry modification, or finger remapping is performed.
    """
    vertices = _points(world_vertices, "world_vertices")
    raw = np.asarray(wrist_matrix)
    if raw.shape != (4, 4) or raw.dtype.kind not in "iuf":
        raise ValueError("wrist_matrix must be a rigid 4x4 transform")
    matrix = raw.astype(np.float64)
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0
    ):
        raise ValueError("wrist_matrix must be a finite homogeneous rigid transform")
    rotation = _rotation(matrix[:3, :3], "wrist_matrix rotation")
    result = (vertices - matrix[:3, 3]) @ rotation
    if canonical_rotation is not None:
        result = result @ _rotation(canonical_rotation, "canonical_rotation").T
    if not np.isfinite(result).all():
        raise ValueError("canonical vertices overflowed")
    return result


def _pair_distance_squared(points, triangles):
    """Exact point/triangle distances, including triangle interiors and edges."""
    a, b, c = np.moveaxis(triangles, 1, 0)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm2 = np.einsum("fi,fi->f", normal, normal)
    denominator = np.where(norm2 > 0, norm2, 1)
    relative = points[:, None] - a
    projection = relative - (
        np.einsum("pfi,fi->pf", relative, normal) / denominator
    )[..., None] * normal
    bary_b = np.einsum("pfi,fi->pf", np.cross(projection, ac), normal) / denominator
    bary_c = np.einsum("pfi,fi->pf", np.cross(ab, projection), normal) / denominator
    interior = (bary_b >= 0) & (bary_c >= 0) & (bary_b + bary_c <= 1)
    distance2 = np.sum((relative - projection) ** 2, axis=-1)
    distance2[~interior | (norm2 == 0)[None, :]] = np.inf
    # Edges supply the correct boundary minimum and handle degenerate faces.
    for start, stop in ((a, b), (b, c), (c, a)):
        edge = stop - start
        length2 = np.einsum("fi,fi->f", edge, edge)
        offset = points[:, None] - start
        fraction = np.clip(
            np.einsum("pfi,fi->pf", offset, edge) / np.where(length2 > 0, length2, 1),
            0,
            1,
        )
        distance2 = np.minimum(
            distance2, np.sum((offset - fraction[..., None] * edge) ** 2, axis=-1)
        )
    return distance2


def proximity_map(
    query, target_vertices, target_triangles, threshold_m, *, point_chunk=64, face_chunk=512
):
    """Unsigned distance to original target triangles and a bounded proximity map.

    ``scores = clip(1 - distances_m / threshold_m, 0, 1)``. Call separately in
    each direction to obtain hand-to-object and object-to-hand maps. These are
    per-query samples; they cannot establish continuous collision clearance or
    distinguish an inside surface from an outside one. Complexity is O(V*F),
    with chunks bounding memory; Blender integrations may use an exact BVH.
    """
    points = _points(query, "query")
    vertices, faces = _mesh(target_vertices, target_triangles)
    threshold = _number(threshold_m, "threshold_m")
    if threshold <= 0:
        raise ValueError("threshold_m must be positive")
    for name, value in (("point_chunk", point_chunk), ("face_chunk", face_chunk)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    distances2 = np.full(len(points), np.inf)
    for first in range(0, len(points), point_chunk):
        stop = min(first + point_chunk, len(points))
        for face_start in range(0, len(faces), face_chunk):
            pair_distances = _pair_distance_squared(
                points[first:stop], vertices[faces[face_start : face_start + face_chunk]]
            )
            distances2[first:stop] = np.minimum(
                distances2[first:stop], pair_distances.min(axis=1)
            )
    distances = np.sqrt(distances2)
    if not np.isfinite(distances).all():
        raise ValueError("surface distances overflowed")
    return {
        "kind": "unsigned_surface_proximity",
        "distances_m": distances,
        "scores": np.clip(1 - distances / threshold, 0, 1),
        "threshold_m": threshold,
    }


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _scores(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector")
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{name} must not contain booleans")
    scores = raw.astype(np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{name} must contain finite scores in [0, 1]")
    return scores


def discrepancy(measured_scores, predicted_scores):
    """Signed score residual: predicted minus measured, in [-1, 1].

    Requires corresponding original vertices and two available maps. This
    numeric difference compares representations; it is not prediction accuracy
    against ground truth, and it does not define missing predictions as zero.
    """
    measured = _scores(measured_scores, "measured_scores")
    predicted = _scores(predicted_scores, "predicted_scores")
    if measured.shape != predicted.shape:
        raise ValueError("measured and predicted scores must use corresponding vertices")
    return predicted - measured


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("mesh bindings must be lowercase SHA-256 digests")
    return value


def _keys(value, required, name):
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise ValueError(f"{name} must have exactly these keys: {', '.join(required)}")


def _frame(value):
    frame = _number(value, "source_frame")
    if frame < 0:
        raise ValueError("source_frame must be nonnegative")
    return frame


def validate_predictions(payload, mesh_bindings, source_frame):
    """Validate a v1 prediction document completely, then select one exact frame.

    ``mesh_bindings`` supplies current original mesh identities as
    ``{name: {"sha256": digest, "vertex_count": count}}``. The JSON document is::

        {"schema_version": 1,
         "provenance": {"kind": "model_prediction", "producer": "model name",
                        "reference": "prediction run or evidence record"},
         "mesh_bindings": {"object": "<sha256>"},
         "maps": [{"source_frame": 0, "mesh": "object", "values": [0.1, ...]}]}

    Provenance kind can also be ``authored_hypothesis``. Every map uses one
    score per original vertex, in original order, with values in [0, 1]. A
    document can cover a subset of meshes/frames; each binding must match a
    known mesh. Duplicate mesh/frame maps are rejected. There is no nearest
    frame fallback, interpolation, or extrapolation. ``payload=None`` means no
    predictions are supplied. Returned maps are independent NumPy arrays;
    missing meshes and provenance are explicit.
    """
    selected_frame = _frame(source_frame)
    if not isinstance(mesh_bindings, Mapping) or not mesh_bindings:
        raise ValueError("mesh_bindings must contain current original mesh identities")
    for name, binding in mesh_bindings.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("mesh names must be nonempty strings")
        _keys(binding, ("sha256", "vertex_count"), f"mesh binding {name}")
        _digest(binding["sha256"])
        count = binding["vertex_count"]
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("vertex_count must be a positive integer")
    result = {
        "source_frame": selected_frame,
        "maps": {},
        "missing_meshes": list(mesh_bindings),
        "provenance": None,
    }
    if payload is None:
        return result
    _keys(payload, ("schema_version", "provenance", "mesh_bindings", "maps"), "prediction document")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("prediction schema_version must be integer 1")
    provenance = payload["provenance"]
    _keys(provenance, ("kind", "producer", "reference"), "prediction provenance")
    if provenance["kind"] not in ("model_prediction", "authored_hypothesis"):
        raise ValueError("provenance kind must be model_prediction or authored_hypothesis")
    if any(not isinstance(provenance[key], str) or not provenance[key].strip() for key in ("producer", "reference")):
        raise ValueError("prediction provenance needs nonempty producer and reference")
    bindings = payload["mesh_bindings"]
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("prediction mesh_bindings must be a nonempty mapping")
    for name, digest in bindings.items():
        if name not in mesh_bindings:
            raise ValueError(f"prediction refers to unknown mesh {name!r}")
        if _digest(digest) != mesh_bindings[name]["sha256"]:
            raise ValueError(f"original mesh hash mismatch for {name!r}")
    if not isinstance(payload["maps"], list):
        raise ValueError("prediction maps must be a list")
    seen = set()
    for entry in payload["maps"]:
        _keys(entry, ("source_frame", "mesh", "values"), "prediction map")
        frame = _frame(entry["source_frame"])
        name = entry["mesh"]
        if not isinstance(name, str) or name not in bindings:
            raise ValueError("prediction map must refer to a bound original mesh")
        key = (name, frame)
        if key in seen:
            raise ValueError(f"duplicate prediction map for mesh {name!r} at frame {frame}")
        seen.add(key)
        values = _scores(entry["values"], "prediction values")
        if len(values) != mesh_bindings[name]["vertex_count"]:
            raise ValueError(f"prediction vertex count mismatch for mesh {name!r}")
        if frame == selected_frame:
            result["maps"][name] = values.copy()
    result["missing_meshes"] = [name for name in mesh_bindings if name not in result["maps"]]
    result["provenance"] = dict(provenance)
    return result
