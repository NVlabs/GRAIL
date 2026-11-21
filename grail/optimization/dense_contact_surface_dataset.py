"""Verified optional original-face-owned contact quadrature.

This module is imported only when an interface explicitly declares a dedicated
surface dataset. It reads canonical geometry only, never a pose or camera.
"""

import hashlib
import re
from pathlib import Path

import numpy as np

DESCRIPTOR_KEYS = {"path", "sha256", "spatial_weight_key", "evidence_key", "max_samples"}
REQUIRED_ARRAYS = {
    "sample_ids",
    "original_face_ids",
    "original_vertex_ids",
    "barycentric",
    "subtriangle_corner_barycentric",
    "canonical_points_m",
    "represented_area_m2",
    "eligible_original_face_ids",
    "original_face_subdivision_n",
}


def validate_descriptor(spec):
    if not isinstance(spec, dict) or set(spec) != DESCRIPTOR_KEYS:
        raise ValueError(
            "Dedicated surface dataset requires exactly " + str(sorted(DESCRIPTOR_KEYS))
        )
    path = spec["path"]
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise ValueError("Dedicated surface path must be relative to its prediction directory")
    if not isinstance(spec["sha256"], str) or re.fullmatch("[0-9a-f]{64}", spec["sha256"]) is None:
        raise ValueError("Dedicated surface dataset requires a lowercase SHA256")
    if any(
        not isinstance(spec[k], str) or not spec[k] for k in ["spatial_weight_key", "evidence_key"]
    ):
        raise ValueError("Dedicated surface field keys must be nonempty strings")
    if spec["spatial_weight_key"] == spec["evidence_key"]:
        raise ValueError("Spatial compatibility and evidence require distinct keys")
    if type(spec["max_samples"]) is not int or spec["max_samples"] < 1:
        raise ValueError("Dedicated surface max_samples must be a positive integer")


def _hash(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def assign_representatives(points, representatives):
    """Bound memory while retaining NumPy argmin's deterministic tie order."""
    assigned, squared = [], []
    for start in range(0, len(points), 512):
        distance = np.sum((points[start : start + 512, None] - representatives[None]) ** 2, axis=-1)
        nearest = np.argmin(distance, axis=1)
        assigned.append(nearest)
        squared.append(distance[np.arange(len(nearest)), nearest])
    return np.concatenate(assigned), np.sqrt(np.concatenate(squared))


def load_surface_dataset(prediction_dir, spec, vertices, faces, declared_geometry):
    """Regenerate every canonical cell, including area and original ownership.

    n² cells tile each original face; no welded node can import excluded-face
    area. Saved points are verified diagnostics, not fixed-world targets.
    """
    validate_descriptor(spec)
    path = Path(prediction_dir) / spec["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Dedicated surface dataset file hash mismatch")
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    actual = {"vertices_sha256": _hash(v, "<f8"), "faces_sha256": _hash(f, "<i8")}
    if any(declared_geometry.get(k) != value for k, value in actual.items()):
        raise ValueError("Dedicated dataset requires exact declared original canonical hashes")
    with np.load(path, allow_pickle=False) as archive:
        if not REQUIRED_ARRAYS.issubset(archive.files):
            raise ValueError("Dedicated surface dataset lacks required canonical arrays")
        a = {k: archive[k].copy() for k in REQUIRED_ARRAYS}
    ids = a["eligible_original_face_ids"]
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or not len(ids)
        or np.any(ids < 0)
        or np.any(ids >= len(f))
        or np.any(ids[1:] <= ids[:-1])
    ):
        raise ValueError("Dedicated eligible faces must be sorted unique original triangle IDs")
    count = len(a["original_face_ids"])
    expected_shapes = {
        "sample_ids": (count,),
        "original_face_ids": (count,),
        "original_vertex_ids": (count, 3),
        "barycentric": (count, 3),
        "subtriangle_corner_barycentric": (count, 3, 3),
        "canonical_points_m": (count, 3),
        "represented_area_m2": (count,),
        "original_face_subdivision_n": (len(ids),),
    }
    if count < 1 or any(a[k].shape != shape for k, shape in expected_shapes.items()):
        raise ValueError("Dedicated surface array shape mismatch")
    integers = [
        "sample_ids",
        "original_face_ids",
        "original_vertex_ids",
        "original_face_subdivision_n",
    ]
    if any(a[k].dtype.kind not in "iu" for k in integers) or any(
        not np.isfinite(a[k]).all() for k in a
    ):
        raise ValueError("Dedicated surface arrays must have finite values and integer IDs")
    if not np.array_equal(a["sample_ids"], np.arange(count)):
        raise ValueError("Dedicated sample IDs must be contiguous original-dataset indices")
    if np.any(a["original_face_ids"] < 0) or np.any(a["original_face_ids"] >= len(f)):
        raise ValueError("Dedicated original face ID out of range")
    if not np.array_equal(a["original_vertex_ids"], f[a["original_face_ids"]]):
        raise ValueError("Dedicated samples do not reference original face vertex IDs")
    owners, corners, subdivisions = [], [], []
    for fid in ids:
        tri = v[f[fid]]
        area = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / 2
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Dedicated eligible face is degenerate")
        n = max(
            1, int(np.ceil(np.linalg.norm(tri - np.roll(tri, -1, axis=0), axis=1).max() / 0.002))
        )
        subdivisions.append(n)
        point = lambda i, j: np.array([1 - (i + j) / n, i / n, j / n])
        for i in range(n):
            for j in range(n - i):
                corners.append([point(i, j), point(i + 1, j), point(i, j + 1)])
                owners.append(int(fid))
                if i + j <= n - 2:
                    corners.append([point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)])
                    owners.append(int(fid))
    owners = np.asarray(owners, dtype=np.int64)
    corners = np.asarray(corners, dtype=np.float64)
    if not np.array_equal(a["original_face_ids"], owners) or not np.array_equal(
        a["original_face_subdivision_n"], subdivisions
    ):
        raise ValueError("Dedicated cells must cover every eligible face in deterministic n² order")
    if not np.allclose(a["subtriangle_corner_barycentric"], corners, rtol=0, atol=2e-15):
        raise ValueError("Dedicated cell barycentrics differ from canonical subdivision")
    bary = corners.mean(1)
    if not np.allclose(a["barycentric"], bary, rtol=0, atol=2e-15) or np.any(a["barycentric"] <= 0):
        raise ValueError("Dedicated samples must be interior canonical cell centroids")
    tri = v[f[owners]]
    xyz_corners = np.einsum("ncv,nvd->ncd", corners, tri)
    area = (
        np.linalg.norm(
            np.cross(xyz_corners[:, 1] - xyz_corners[:, 0], xyz_corners[:, 2] - xyz_corners[:, 0]),
            axis=1,
        )
        / 2
    )
    points = np.einsum("nv,nvd->nd", bary, tri)
    edge = np.linalg.norm(xyz_corners - np.roll(xyz_corners, -1, axis=1), axis=-1).max()
    if edge > 0.002 + 1e-12 or not np.allclose(a["canonical_points_m"], points, rtol=0, atol=2e-12):
        raise ValueError("Dedicated canonical coordinates, scale or 2mm edge bound mismatch")
    if np.any(a["represented_area_m2"] <= 0) or not np.allclose(
        a["represented_area_m2"], area, rtol=1e-10, atol=1e-16
    ):
        raise ValueError("Dedicated represented area differs from original canonical cells")
    total_by_face = np.bincount(owners, weights=a["represented_area_m2"], minlength=len(f))[ids]
    parent_tri = v[f[ids]]
    parent_area = (
        np.linalg.norm(
            np.cross(parent_tri[:, 1] - parent_tri[:, 0], parent_tri[:, 2] - parent_tri[:, 0]),
            axis=1,
        )
        / 2
    )
    if not np.allclose(total_by_face, parent_area, rtol=1e-10, atol=1e-16):
        raise ValueError("Dedicated per-face represented area does not conserve original area")
    return dict(
        face_ids=owners,
        barycentric=a["barycentric"],
        area=a["represented_area_m2"],
        points=points,
        cell_corners=xyz_corners,
        path=path,
        provenance={
            "path": str(path),
            "sha256": spec["sha256"],
            "canonical_object": actual,
            "eligible_original_face_ids_sha256": _hash(ids, "<i8"),
            "original_faces": len(ids),
            "samples": count,
            "represented_area_m2": float(a["represented_area_m2"].sum()),
            "max_canonical_cell_edge_m": float(edge),
            "max_per_face_area_error_m2": float(np.max(abs(total_by_face - parent_area))),
            "sample_face_ids_sha256": _hash(owners, "<i8"),
            "sample_barycentric_sha256": _hash(a["barycentric"], "<f8"),
            "sample_area_m2_sha256": _hash(a["represented_area_m2"], "<f8"),
            "area_definition": "Full area of each original-face-owned <=2mm canonical subtriangle at its interior centroid; no welding.",
        },
    )
