"""CPU evidence for a native human refinement candidate, never an acceptance vote.

Geometry must be the native model's world-space output at the explicit source
``frames``. Original vertex/face order is retained. Scene object vertices are
object-local, before ``object_scale``; the camera is OpenCV camera-to-world.
Contact input is a ``grail.contact_review`` or ``grail.fixed_contact`` document
(v2 vertex pairs, or historical v1 triangle anchors). ``evidence.sequence`` and
``evidence.source_video_sha256`` bind it independently to the current source.

Collision queries require explicit ``evidence.collision_frames``. They use all
original edges and triangles, with the existing CPU routine's coplanarity and
containment limitations. This module does not introduce a collision proxy.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np

from ..contact import _components, _mesh, edge_triangle_crossings

SCHEMA = "grail.hoi_refinement.review.v1"


def _array(value, label, *, integer=False):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in ("iu" if integer else "fiu"):
        raise ValueError(f"{label} must contain {'integer' if integer else 'numeric'} values")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result.astype(np.int64 if integer else np.float64, copy=False)


def _not_performed(reason):
    return {"status": "NOT_PERFORMED", "reason": reason}


def _stats(values):
    values = np.asarray(values, dtype=float).ravel()
    if not len(values):
        return {"count": 0, "min": None, "mean": None, "max": None, "p95": None}
    return {
        "count": len(values),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "p95": float(np.quantile(values, 0.95)),
    }


def _geometry(data, name, frame_count):
    vertices = _array(data["vertices"], f"{name}.vertices")
    joints = _array(data["joints"], f"{name}.joints")
    frames = _array(data["frames"], f"{name}.frames")
    if frames.ndim != 1 or not len(frames) or (np.diff(frames) <= 0).any():
        raise ValueError(f"{name}.frames must be a nonempty strictly increasing vector")
    if frames[0] < 0 or frames[-1] > frame_count - 1:
        raise ValueError(f"{name}.frames lie outside the native source timeline")
    for field, value in (("vertices", vertices), ("joints", joints)):
        if value.ndim != 3 or value.shape[0] != len(frames) or value.shape[2] != 3:
            raise ValueError(f"{name}.{field} must be [len(frames), N, 3]")
        if not value.shape[1]:
            raise ValueError(f"{name}.{field} must be nonempty")
    _, faces = _mesh(vertices[0], data["faces"])
    return {**data, "vertices": vertices, "joints": joints, "faces": faces, "frames": frames}


def _model_type(data):
    value = np.asarray(data["model_type"])
    if value.shape != () or value.dtype.kind not in "US" or not str(value.item()):
        raise ValueError("model_type must be an explicit scalar string")
    return str(value.item())


def _motion(data, name):
    trans = _array(data["trans"], f"{name}.trans")
    poses = _array(data["poses"], f"{name}.poses")
    if trans.ndim != 2 or trans.shape[1] != 3 or not len(trans):
        raise ValueError(f"{name}.trans must be [T, 3]")
    if poses.ndim < 2 or len(poses) != len(trans):
        raise ValueError(f"{name}.poses must retain the native time dimension")
    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{name}.fps must be positive and finite")
    _model_type(data)
    rotation = _array(data["obj_R"], f"{name}.obj_R")
    translation = _array(data["obj_t"], f"{name}.obj_t")
    if rotation.shape != (len(trans), 3, 3) or translation.shape != trans.shape:
        raise ValueError(f"{name} object track must cover the full native timeline")
    if not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-5):
        raise ValueError(f"{name}.obj_R must contain rotations")
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError(f"{name}.obj_R must contain proper rotations")
    return len(trans), fps, poses, trans, rotation, translation


def _binding(parent, geometry, object_vertices, object_faces, evidence):
    if not evidence.get("source_video_sha256") or not evidence.get("sequence"):
        return None
    from grail.optimization.fixed_contact import build_binding

    return build_binding(
        sequence=evidence["sequence"],
        source_video_sha256=evidence["source_video_sha256"],
        frame_count=len(parent["trans"]),
        fps=float(parent["fps"]),
        human_model_type=_model_type(parent),
        human_vertex_count=geometry["vertices"].shape[1],
        human_faces=geometry["faces"],
        object_vertices=object_vertices,
        object_faces=object_faces,
    )


def _contacts(spec, binding, object_vertices, object_faces, human_faces):
    """Reuse the existing strict parser rather than invent another contact format."""
    import torch

    from grail.optimization.contact_review import compile_review
    from grail.optimization.fixed_contact import FixedContactLoss

    if spec.get("schema") in ("grail.contact_review.v1", "grail.contact_review.v2"):
        version = spec["schema"].rsplit(".", 1)[-1]
        spec = {
            "schema": f"grail.fixed_contact.{version}",
            "binding": binding,
            "prediction": {"method": "review compiler", "evidence": "supplied video review"},
            "contacts": compile_review(spec, binding),
            "video_review": spec,
        }
    # from_file hashes and validates one immutable snapshot, including the
    # embedded reviewed intervals. No persistent artifact is created here.
    with tempfile.TemporaryDirectory(prefix="grail-contact-review-") as directory:
        path = Path(directory) / "contacts.json"
        path.write_text(json.dumps(spec, allow_nan=False), encoding="utf-8")
        fixed = FixedContactLoss.from_file(
            path,
            binding=binding,
            object_vertices=object_vertices,
            object_faces=object_faces,
            human_faces=human_faces,
            device="cpu",
            dtype=torch.float64,
        )
    return fixed.export_contacts(), copy.deepcopy(spec.get("video_review"))


def _velocity(points, frames, fps):
    spacing = np.diff(frames)
    adjacent = spacing <= 1
    rate = (fps / spacing[adjacent]).reshape((-1,) + (1,) * (points.ndim - 1))
    velocity = np.diff(points, axis=0)[adjacent] * rate
    return {
        "status": "PERFORMED" if adjacent.any() else "NOT_PERFORMED",
        "reason": None if adjacent.any() else "No evaluated steps within one source-frame interval",
        "frame_pairs": np.stack((frames[:-1][adjacent], frames[1:][adjacent]), axis=-1).tolist(),
        "speed_m_per_s": _stats(np.linalg.norm(velocity, axis=-1)),
        "omitted_steps_exceeding_one_source_frame": int((~adjacent).sum()),
    }


def _normal_review(human, objects, human_faces, object_faces, pair):
    import torch

    from ..hoi_contact_normals import local_vertex_normals, material_normal_dots

    try:
        human_tensor = torch.as_tensor(human, dtype=torch.float64)
        object_tensor = torch.as_tensor(objects, dtype=torch.float64)
        if "object_vertex_id" in pair:
            human_normal = local_vertex_normals(
                human_tensor, np.ascontiguousarray(human_faces), [pair["human_vertex_id"]]
            )
            object_normal = local_vertex_normals(
                object_tensor, np.ascontiguousarray(object_faces), [pair["object_vertex_id"]]
            )
            dots = (human_normal * object_normal).sum(-1).clamp(-1, 1)[:, 0].numpy()
        else:
            dots = material_normal_dots(
                human_tensor,
                object_tensor,
                np.ascontiguousarray(human_faces),
                np.ascontiguousarray(object_faces),
                [{**pair, "weight": 1.0}],
            )[:, 0].numpy()
    except ValueError as exc:
        return _not_performed(str(exc))
    return {
        "status": "PERFORMED",
        "outward_normal_dot": _stats(dots),
        "per_frame_dot": dots.tolist(),
        "opposed_hemisphere_count": int((dots < 0).sum()),
        "convention": "-1 is opposing; directions follow original face winding",
        "object_normal_definition": (
            "area-weighted incident triangles at original vertex"
            if "object_vertex_id" in pair
            else "original triangle face normal"
        ),
    }


def _contact_review(pairs, review, before, after, object_before, object_after, object_faces, fps):
    frames = before["frames"]
    reports = []
    for pair in pairs:
        start, stop = pair["active_frames"]
        selected = (frames >= start) & (frames < stop)
        active = frames[selected]
        record = {
            "id": pair["id"],
            "body_part": pair["body_part"],
            "human_vertex_id": pair["human_vertex_id"],
            "active_frames": pair["active_frames"],
            "authored_confidence": pair["confidence"],
            "evaluated_active_frames": active.tolist(),
            "missing_active_frame_count": stop - start - int((active == np.floor(active)).sum()),
            "evaluated_active_subframe_count": int((active != np.floor(active)).sum()),
            "status": "PERFORMED" if len(active) else "NOT_PERFORMED",
        }
        if "object_vertex_id" in pair:
            record["object_vertex_id"] = pair["object_vertex_id"]
        else:
            record.update(
                object_face_id=pair["object_face_id"], object_barycentric=pair["object_barycentric"]
            )
        if not len(active):
            record["reason"] = "No evaluated frames within this authored active interval"
            reports.append(record)
            continue
        gaps = []
        for label, geometry, objects in (
            ("before", before, object_before),
            ("after", after, object_after),
        ):
            human = geometry["vertices"][selected]
            objects = objects[selected]
            if "object_vertex_id" in pair:
                anchor = objects[:, pair["object_vertex_id"]]
            else:
                triangle = objects[:, object_faces[pair["object_face_id"]]]
                anchor = np.einsum("v,fvc->fc", pair["object_barycentric"], triangle)
            relative = human[:, pair["human_vertex_id"]] - anchor
            gap = np.linalg.norm(relative, axis=-1)
            gaps.append(gap)
            record[label] = {
                "fixed_material_gap_m": _stats(gap),
                "per_frame_gap_m": gap.tolist(),
                "normal": _normal_review(human, objects, geometry["faces"], object_faces, pair),
                "contact_relative_velocity": _velocity(relative, active, fps),
            }
        record["gap_change_m"] = _stats(gaps[1] - gaps[0])
        reports.append(record)
    return {
        "status": "PERFORMED",
        "pairs": reports,
        "video_intervals_present": review is not None,
        "reviewed_hand_intervals": None if review is None else copy.deepcopy(review["hands"]),
        "limitations": "Fixed authored material pairs only; distances do not establish contact. "
        "Zero-confidence pairs retain their authored confidence and are diagnostic only.",
    }


def _topology_counts(vertices, faces):
    """Count edge-connected shells after exact-coordinate welding, including nonmanifold edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    _, aliases = np.unique(vertices, axis=0, return_inverse=True)
    welded = aliases.reshape(-1)[faces]
    start = welded.ravel()
    end = welded[:, [1, 2, 0]].ravel()
    low = np.minimum(start, end)
    high = np.maximum(start, end)
    order = np.lexsort((high, low))
    low, high = low[order], high[order]
    neighbors = (low[1:] == low[:-1]) & (high[1:] == high[:-1])
    face_ids = order // 3
    # A chain connects all faces sharing an edge, including three or more.
    rows, cols = face_ids[1:][neighbors], face_ids[:-1][neighbors]
    graph = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(faces), len(faces))
    )
    count, labels = connected_components(graph, directed=False)
    groups = np.flatnonzero(np.r_[True, ~neighbors])
    hits = np.diff(np.r_[groups, len(order)])
    direction = (start < end)[order]
    # A closed oriented shell has exactly one forward hit and one reverse hit per edge.
    bad_edge = (hits != 2) | (np.add.reduceat(direction, groups) != 1)
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    bad_shells = np.union1d(labels[face_ids[groups[bad_edge]]], labels[~(area > 0)])
    return count, len(bad_shells)


def _topology(vertices, faces):
    if len(faces) >= 100_000:
        component_count, open_count = _topology_counts(vertices, faces)
    else:
        components = _components(vertices, faces)
        component_count = len(components)
        open_count = sum(not closed for _, closed in components)
    return {
        "original_vertices": len(vertices),
        "original_faces": len(faces),
        "components": component_count,
        "open_or_inconsistent_components": open_count,
        "solid_state": "UNKNOWN_OPEN_GEOMETRY" if open_count else "CLOSED_COMPONENTS_UNCERTIFIED",
        "limitations": "Exact-coordinate aliases establish topology only; no mesh repair. "
        "Closed components are not a self-intersection or solid-validity certificate.",
    }


def _collisions(before, after, object_before, object_after, object_faces, queried):
    if queried is None:
        return _not_performed("No explicit evidence.collision_frames supplied")
    queried = _array(queried, "collision_frames")
    frames = before["frames"]
    if queried.ndim != 1 or len(set(queried.tolist())) != len(queried):
        raise ValueError("collision_frames must be a unique source-frame vector")
    if not np.isin(queried, frames).all():
        raise ValueError("collision_frames must be a subset of native evaluated frames")
    if not len(queried):
        return _not_performed("The explicit collision frame query is empty")
    records, inherited, introduced, resolved = [], [], [], []
    for frame in queried:
        index = int(np.searchsorted(frames, frame))
        record = {"frame": float(frame)}
        counts = []
        for label, geometry, objects in (
            ("before", before, object_before),
            ("after", after, object_after),
        ):
            human, obj = geometry["vertices"][index], objects[index]
            if (
                label == "after"
                and np.array_equal(before["vertices"][index], human)
                and np.array_equal(object_before[index], obj)
            ):
                # An unedited Blender round trip can reproduce both original meshes exactly.
                record[label] = copy.deepcopy(record["before"])
                counts.append(counts[0])
                continue
            human_to_object = edge_triangle_crossings(human, geometry["faces"], obj, object_faces)
            object_to_human = edge_triangle_crossings(obj, object_faces, human, geometry["faces"])
            count = (
                human_to_object["intersection_locations"]
                + object_to_human["intersection_locations"]
            )
            counts.append(count)
            record[label] = {
                "human_edges_to_object": human_to_object,
                "object_edges_to_human": object_to_human,
                "human_topology": _topology(human, geometry["faces"]),
                "object_topology": _topology(obj, object_faces),
            }
        if counts[0] and counts[1]:
            inherited.append(float(frame))
        elif counts[1]:
            introduced.append(float(frame))
        elif counts[0]:
            resolved.append(float(frame))
        record["crossing_location_count_change"] = counts[1] - counts[0]
        records.append(record)
    return {
        "status": "PERFORMED",
        "queried_frames": queried.tolist(),
        "frames": records,
        "frames_with_crossings_before_and_after": inherited,
        "frames_with_new_crossings": introduced,
        "frames_with_crossings_no_longer_detected": resolved,
        "limitations": "Whole original meshes, bidirectional non-coplanar edge/triangle queries "
        "at listed source frames only. Endpoint touches, coplanar overlap, containment, "
        "self-collision, between-frame crossings and signed penetration depth are not checked. "
        "No detected crossing is not collision clearance; surviving crossings may differ.",
    }


def _camera(scene):
    c2w = _array(scene["camera_matrix_world"], "camera_matrix_world")
    intrinsics = scene.get("camera_intrinsics", scene.get("camera_intrinsicsK"))
    if intrinsics is None:
        raise ValueError("camera_intrinsics is required for source projection")
    intrinsics = _array(intrinsics, "camera_intrinsics")
    if c2w.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("Camera matrix must be 4x4 and intrinsics 3x3")
    if not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("camera_matrix_world must be affine")
    if not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("camera_matrix_world must have a rigid rotation")
    if not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-5):
        raise ValueError("camera_matrix_world must have a proper rotation")
    if not np.allclose(intrinsics[2], [0, 0, 1]) or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0:
        raise ValueError("camera_intrinsics must follow OpenCV projection convention")
    return c2w, intrinsics


def _project(points, camera):
    c2w, intrinsics = camera
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    homogeneous = local @ intrinsics.T
    valid = local[..., 2] > 1e-8
    pixels = np.zeros_like(local[..., :2])
    np.divide(homogeneous[..., :2], homogeneous[..., 2, None], out=pixels, where=valid[..., None])
    return pixels, valid


def _source_projection(before, after, camera, scene):
    if camera is None:
        return _not_performed("No calibrated camera supplied")
    source, valid_source = _project(before["joints"], camera)
    candidate, valid_candidate = _project(after["joints"], camera)
    common = valid_source & valid_candidate
    movement = np.linalg.norm(candidate - source, axis=-1)
    result = {
        "status": "PERFORMED",
        "frames": before["frames"].tolist(),
        "joint_displacement_px": _stats(movement[common]),
        "valid_joint_pairs": int(common.sum()),
        "missing_joint_pairs": int(common.size - common.sum()),
        "per_frame_displacement_px": [_stats(row[mask]) for row, mask in zip(movement, common)],
        "limitations": "Projection movement relative to the supplied parent native joints; "
        "not an observed RGB keypoint error or confidence score. Occlusion is not tested.",
    }
    if "image_width" in scene and "image_height" in scene:
        width, height = float(scene["image_width"]), float(scene["image_height"])
        if not np.isfinite([width, height]).all() or min(width, height) <= 0:
            raise ValueError("Image dimensions must be positive and finite")
        for label, pixels, valid in (
            ("before", source, valid_source),
            ("after", candidate, valid_candidate),
        ):
            inside = valid & (pixels[..., 0] >= 0) & (pixels[..., 0] < width)
            inside &= (pixels[..., 1] >= 0) & (pixels[..., 1] < height)
            result[f"{label}_in_image_joint_count"] = int(inside.sum())
    return result


def _ray_hits(origins, directions, vertices, faces):
    """Nearest positive FP64 ray/triangle hits, retaining original face IDs.

    The Moller--Trumbore test is two-sided and includes triangle seams. Queries
    are chunked by original faces and never repair, subsample or proxy geometry.
    Parallel/coplanar rays have no unique surface hit and are left missing.
    """
    distances = np.full(len(origins), np.inf)
    ids = np.full(len(origins), -1, dtype=np.int64)
    for offset in range(0, len(faces), 4096):
        triangle = vertices[faces[offset : offset + 4096]]
        a, b, c = np.moveaxis(triangle, 1, 0)
        edge_a, edge_b = b - a, c - a
        scale = np.linalg.norm(edge_a, axis=-1) * np.linalg.norm(edge_b, axis=-1)
        for index, (origin, direction) in enumerate(zip(origins, directions)):
            p = np.cross(direction, edge_b)
            det = np.einsum("ij,ij->i", edge_a, p)
            valid = np.abs(det) > 32 * np.finfo(float).eps * scale
            inverse = np.divide(1, det, out=np.zeros_like(det), where=valid)
            relative = origin - a
            u = np.einsum("ij,ij->i", relative, p) * inverse
            q = np.cross(relative, edge_a)
            v = (q @ direction) * inverse
            distance = np.einsum("ij,ij->i", edge_b, q) * inverse
            tolerance = 64 * np.finfo(float).eps
            valid &= (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1 + tolerance)
            valid &= distance > 1e-9
            distance = np.where(valid, distance, np.inf)
            nearest = int(distance.argmin())
            if distance[nearest] < distances[index]:
                distances[index] = distance[nearest]
                ids[index] = offset + nearest
    return distances, ids


def _digest(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _fixed_depth(
    data,
    evidence,
    scene,
    camera,
    before,
    after,
    object_before,
    object_after,
    object_faces,
    frame_count,
):
    """Compare nearest original-surface depths on an immutable pixel-ray NPZ.

    Evidence keys: path, sha256, source_video_sha256, scene_sha256, units='m',
    depth_kind='ray_distance', calibration='background_gt', calibration_sha256.
    NPZ arrays: frames[N], origins[N,3], directions[N,3], target_distance[N],
    optional group[N]. Directions are unit world rays; distances are metres.
    Calibration is a supplied hash-bound artifact, never recomputed here.
    """
    if data is None:
        return _not_performed("No bound frozen pixel-ray depth cache supplied")
    if camera is None or not evidence.get("source_video_sha256") or not scene.get("_sha256"):
        return _not_performed(
            "Fixed-ray review requires independent source/scene hashes and calibrated camera"
        )
    for key, actual in (
        ("source_video_sha256", evidence["source_video_sha256"]),
        ("scene_sha256", scene["_sha256"]),
    ):
        if _digest(data.get(key), key) != _digest(actual, key):
            raise ValueError(f"Fixed depth {key} differs from the current input")
    if evidence.get("scene_sha256", scene["_sha256"]) != scene["_sha256"]:
        raise ValueError("Depth evidence scene binding disagrees with the current scene")
    if (data.get("units"), data.get("depth_kind"), data.get("calibration")) != (
        "m",
        "ray_distance",
        "background_gt",
    ):
        raise ValueError(
            "Fixed depth requires metre ray-distance targets and background_gt calibration"
        )
    _digest(data.get("calibration_sha256"), "calibration_sha256")
    payload = Path(data["path"]).expanduser().read_bytes()
    expected_hash = _digest(data.get("sha256", data.get("npz_sha256")), "fixed depth sha256")
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("Fixed depth NPZ hash mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        frames = _array(archive["frames"], "depth frames", integer=True)
        origins = _array(archive["origins"], "depth origins")
        directions = _array(archive["directions"], "depth directions")
        targets = _array(archive["target_distance"], "target_distance")
        groups = archive["group"].tolist() if "group" in archive else None
    if (
        frames.ndim != 1
        or not len(frames)
        or origins.shape != (len(frames), 3)
        or directions.shape != origins.shape
        or targets.shape != frames.shape
    ):
        raise ValueError("Frozen depth arrays must be nonempty and have compatible row counts")
    if groups is not None and (
        not isinstance(groups, list)
        or len(groups) != len(frames)
        or any(not isinstance(g, (str, int, float)) for g in groups)
        or any(isinstance(g, (int, float)) and not np.isfinite(g) for g in groups)
    ):
        raise ValueError("Optional depth groups must contain one string or numeric label per row")
    if (frames < 0).any() or (frames >= frame_count).any() or (targets <= 0).any():
        raise ValueError("Frozen depth frames/targets lie outside the timeline or forward rays")
    if not np.allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-6, rtol=0):
        raise ValueError("Frozen depth directions must be unit world rays")
    if not np.allclose(origins, camera[0][:3, 3], atol=1e-7, rtol=0):
        raise ValueError("Frozen depth origins differ from the calibrated camera center")
    if ((directions @ camera[0][:3, :3])[:, 2] <= 0).any():
        raise ValueError("Frozen depth rays must point forward in the calibrated OpenCV camera")
    selected = np.isin(frames, before["frames"])
    indices = np.flatnonzero(selected)
    if not len(indices):
        return {
            **_not_performed("No frozen rays at native evaluated source frames"),
            "unevaluated_ray_count": len(frames),
        }
    distances, hit_faces, hit_surfaces = {}, {}, {}
    for label, geometry, objects in (
        ("before", before, object_before),
        ("after", after, object_after),
    ):
        distances[label] = np.full(len(indices), np.inf)
        hit_faces[label] = np.full(len(indices), -1, dtype=np.int64)
        hit_surfaces[label] = np.full(len(indices), "missing", dtype="U7")
        for frame in np.unique(frames[selected]):
            row_mask = frames[selected] == frame
            native_index = int(np.searchsorted(before["frames"], frame))
            for surface, vertices, faces in (
                ("human", geometry["vertices"][native_index], geometry["faces"]),
                ("object", objects[native_index], object_faces),
            ):
                depth, face_id = _ray_hits(
                    origins[selected][row_mask], directions[selected][row_mask], vertices, faces
                )
                rows = np.flatnonzero(row_mask)
                better = depth < distances[label][rows]
                chosen = rows[better]
                distances[label][chosen] = depth[better]
                hit_faces[label][chosen] = face_id[better]
                hit_surfaces[label][chosen] = surface
    common = np.isfinite(distances["before"]) & np.isfinite(distances["after"])
    results = {
        "status": "PERFORMED",
        "cache_sha256": expected_hash,
        "calibration_sha256": data["calibration_sha256"],
        "queried_ray_count": len(indices),
        "unevaluated_ray_count": int((~selected).sum()),
        "common_hit_count": int(common.sum()),
        "new_missing_count": int(
            (np.isfinite(distances["before"]) & ~np.isfinite(distances["after"])).sum()
        ),
        "inherited_missing_count": int(
            (~np.isfinite(distances["before"]) & ~np.isfinite(distances["after"])).sum()
        ),
        "new_hit_count": int(
            (~np.isfinite(distances["before"]) & np.isfinite(distances["after"])).sum()
        ),
        "rows": [],
        "limitations": "Exact same frozen pixel rays and targets before/after; nearest original "
        "human/object triangle hits. Calibration provenance is supplied, not re-estimated. "
        "Parallel/coplanar and missing hits remain missing; no confidence is inferred.",
    }
    for label in ("before", "after"):
        hit = np.isfinite(distances[label])
        errors = np.abs(distances[label][common] - targets[selected][common])
        results[label] = {
            "hit_count": int(hit.sum()),
            "missing_count": int((~hit).sum()),
            "coverage": float(hit.mean()),
            "common_hit_absolute_error_m": _stats(errors),
        }
    for local, row in enumerate(indices):
        record = {
            "row": int(row),
            "frame": int(frames[row]),
            "target_distance_m": float(targets[row]),
        }
        if groups is not None:
            record["group"] = groups[row]
        for label in ("before", "after"):
            record[label] = {
                "surface": str(hit_surfaces[label][local]),
                "original_face_id": (
                    int(hit_faces[label][local]) if hit_faces[label][local] >= 0 else None
                ),
                "distance_m": (
                    float(distances[label][local]) if np.isfinite(distances[label][local]) else None
                ),
            }
        results["rows"].append(record)
    return results


def _object_world(geometry, vertices, rotation, translation):
    frames = geometry["frames"]
    integers = frames == np.floor(frames)
    if "obj_R" in geometry and "obj_t" in geometry:
        sampled_rotation = _array(geometry["obj_R"], "evaluated obj_R")
        sampled_translation = _array(geometry["obj_t"], "evaluated obj_t")
        if sampled_rotation.shape != (len(frames), 3, 3) or sampled_translation.shape != (
            len(frames),
            3,
        ):
            raise ValueError("Evaluated object transforms must cover exactly the queried frames")
        if not np.allclose(
            sampled_rotation @ sampled_rotation.swapaxes(-1, -2), np.eye(3), atol=1e-5
        ) or not np.allclose(np.linalg.det(sampled_rotation), 1, atol=1e-5):
            raise ValueError("Evaluated object transforms must contain proper rotations")
        ids = frames[integers].astype(np.int64)
        if not np.allclose(
            sampled_rotation[integers], rotation[ids], atol=1e-7, rtol=0
        ) or not np.allclose(sampled_translation[integers], translation[ids], atol=1e-7, rtol=0):
            raise ValueError("Evaluated object transforms disagree with the native source track")
    elif integers.all():
        ids = frames.astype(np.int64)
        sampled_rotation, sampled_translation = rotation[ids], translation[ids]
    else:
        raise ValueError("Subframe review requires obj_R and obj_t from the native evaluator")
    return np.einsum("fij,vj->fvi", sampled_rotation, vertices) + sampled_translation[:, None]


def review_candidate(parent, candidate, before, after, scene, contact_spec=None, evidence=None):
    """Return a JSON-safe UNREVIEWED report comparing evaluated native geometry.

    Invalid timeline/topology/source bindings raise ``ValueError``. Missing
    optional evidence yields explicit ``NOT_PERFORMED`` checks. There is no
    threshold-based acceptance, automatic contact selection or physical claim.
    """
    evidence = {} if evidence is None else evidence
    count, fps, poses, trans, rotation, translation = _motion(parent, "parent")
    new_count, new_fps, new_poses, new_trans, new_rotation, new_translation = _motion(
        candidate, "candidate"
    )
    if (new_count, new_fps, _model_type(candidate), new_poses.shape) != (
        count,
        fps,
        _model_type(parent),
        poses.shape,
    ):
        raise ValueError("Candidate must preserve native model, pose shape and source timeline")
    before, after = _geometry(before, "before", count), _geometry(after, "after", count)
    if not np.array_equal(before["frames"], after["frames"]):
        raise ValueError("Before/after native geometry must evaluate the same source frames")
    if not np.array_equal(before["faces"], after["faces"]):
        raise ValueError("Candidate must preserve exact original human topology/order")
    for name in ("vertices", "joints"):
        if before[name].shape != after[name].shape:
            raise ValueError(f"Candidate must preserve native {name} shape and order")
    vertices, object_faces = _mesh(scene["object_vertices"], scene["object_faces"])
    scale = _array(scene.get("object_scale", 1.0), "object_scale")
    if scale.shape not in ((), (3,)) or (scale <= 0).any():
        raise ValueError("object_scale must be a positive scalar or XYZ vector")
    for label, motion in (
        ("parent", parent),
        ("candidate", candidate),
        ("before", before),
        ("after", after),
    ):
        if "obj_scale" in motion:
            native_scale = _array(motion["obj_scale"], f"{label}.obj_scale")
            if native_scale.shape not in ((), (3,)) or not np.array_equal(
                np.broadcast_to(native_scale, (3,)), np.broadcast_to(scale, (3,))
            ):
                raise ValueError(f"{label} object scale disagrees with original scene geometry")
    vertices = vertices * scale
    frames = before["frames"]
    object_before = _object_world(before, vertices, rotation, translation)
    object_after = _object_world(after, vertices, new_rotation, new_translation)
    binding = _binding(parent, before, vertices, object_faces, evidence)
    if contact_spec is None:
        contacts = _not_performed("No fixed material contact specification supplied")
    elif binding is None:
        contacts = _not_performed(
            "Independent evidence.sequence and source_video_sha256 are required"
        )
    else:
        pairs, reviewed = _contacts(contact_spec, binding, vertices, object_faces, before["faces"])
        contacts = _contact_review(
            pairs, reviewed, before, after, object_before, object_after, object_faces, fps
        )
    camera = _camera(scene) if "camera_matrix_world" in scene else None
    return {
        "schema": SCHEMA,
        "status": "UNREVIEWED",
        "source_frame_count": count,
        "evaluated_frames": frames.tolist(),
        "unevaluated_frame_count": count - int((frames == np.floor(frames)).sum()),
        "evaluated_subframe_count": int((frames != np.floor(frames)).sum()),
        "binding": binding,
        "checks": {
            "native_geometry": {
                "status": "PERFORMED",
                "human_vertex_count": before["vertices"].shape[1],
                "human_face_count": len(before["faces"]),
                "joint_count": before["joints"].shape[1],
                "pose_control_change": _stats(np.abs(new_poses - poses)),
                "root_translation_change_m": _stats(np.linalg.norm(new_trans - trans, axis=-1)),
                "human_vertex_displacement_m": _stats(
                    np.linalg.norm(after["vertices"] - before["vertices"], axis=-1)
                ),
                "object_track_unchanged": bool(
                    np.array_equal(rotation, new_rotation)
                    and np.array_equal(translation, new_translation)
                ),
            },
            "fixed_material_contacts": contacts,
            "original_mesh_intersections": _collisions(
                before,
                after,
                object_before,
                object_after,
                object_faces,
                evidence.get("collision_frames"),
            ),
            "source_projection": _source_projection(before, after, camera, scene),
            "fixed_ray_depth": _fixed_depth(
                evidence.get("fixed_depth"),
                evidence,
                scene,
                camera,
                before,
                after,
                object_before,
                object_after,
                object_faces,
                count,
            ),
            "temporal": {
                "status": "PERFORMED",
                "before_joint_velocity": _velocity(before["joints"], frames, fps),
                "after_joint_velocity": _velocity(after["joints"], frames, fps),
                "joint_velocity_change": _velocity(after["joints"] - before["joints"], frames, fps),
                "limitations": "Finite differences of native geometry at listed source frames and "
                "subframes, with step at most one source-frame interval. No dynamics claim.",
            },
        },
        "limitations": [
            "Metrics do not accept the candidate; independent visual/source review is required.",
            "Before/after comparisons retain inherited measurements and newly introduced changes.",
            "No force, stability, grasp success or physical validation is performed.",
        ],
    }
