"""File validation and constant-depth trials, independent of Blender's UI."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def new_path(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Choose a new output path; already exists: {path}")
    return path


def write_json(path, value):
    path = new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def integer_list(value, label, allow_empty=False):
    if not value.strip() and allow_empty:
        return []
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(f"{label}: use comma-separated integers") from error
    if len(values) != len(set(values)):
        raise ValueError(f"{label}: duplicate values are not allowed")
    return values


def validate_scene(scene):
    """Validate before kit.setup creates any Blender datablocks."""
    required = ("object_vertices", "object_faces", "camera_K", "camera_R",
                "camera_t", "image_width", "image_height")
    missing = set(required) - set(scene)
    if missing:
        raise ValueError(f"Scene NPZ missing fields: {sorted(missing)}")
    if "object_scale" in scene:
        raise ValueError("Scene must contain unscaled object vertices; scale comes from native obj_scale")
    for name in required:
        a = np.asarray(scene[name])
        if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError(f"Scene {name} must be finite numeric data")
    vertices, faces = scene["object_vertices"], scene["object_faces"]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Scene object_vertices must have shape [V,3], V >= 3")
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or faces.dtype.kind not in "iu" or np.any(faces < 0)
            or np.any(faces >= len(vertices))):
        raise ValueError("Scene object_faces must contain valid integer triangles")
    k, r, t = (np.asarray(scene[name], dtype=float) for name in
               ("camera_K", "camera_R", "camera_t"))
    if k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,):
        raise ValueError("Camera shapes must be K[3,3], R[3,3], t[3]")
    if not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("Camera K bottom row must be [0,0,1]")
    if min(k[0, 0], k[1, 1]) <= 0 or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8:
        raise ValueError("Camera needs positive focal lengths and zero skew")
    if (not np.allclose(r.T @ r, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-5, rtol=0)):
        raise ValueError("camera_R must be a proper world-to-CV-camera rotation")
    for name in ("image_width", "image_height"):
        a = np.asarray(scene[name])
        if a.shape != () or float(a) != int(a) or int(a) < 1:
            raise ValueError(f"Scene {name} must be a positive scalar integer")
    return scene


def constant_depth_candidate(motion, scene, *, axis_world, offset_m, bound_m,
                             contact_frame, prefix_end, translation_tolerance,
                             rotation_tolerance, evidence):
    """A declared constant root shift, with measured stationary-prefix guard.

    This does not estimate calibration, detect contact, repair object drift,
    or certify the supplied source evidence. All other arrays retain bytes.
    """
    validate_scene(scene)
    frames = len(motion["poses"])
    if (isinstance(contact_frame, bool) or isinstance(prefix_end, bool)
            or not isinstance(contact_frame, int) or not isinstance(prefix_end, int)
            or not 0 <= contact_frame <= prefix_end < frames or prefix_end < 1):
        raise ValueError("Select a valid contact frame inside a stationary prefix [0,end] of at least two frames")
    if not evidence.strip():
        raise ValueError("Record the source evidence for contact timing, stationary prefix, and calibrated camera")
    values = [offset_m, bound_m, translation_tolerance, rotation_tolerance]
    if not np.isfinite(values).all() or not 0 < abs(offset_m) <= bound_m:
        raise ValueError("Depth offset must be nonzero, finite, and within the declared positive bound")
    if translation_tolerance < 0 or rotation_tolerance < 0:
        raise ValueError("Stationarity tolerances must be finite and nonnegative")
    axis = np.asarray(axis_world, dtype=float)
    expected_axis = np.asarray(scene["camera_R"], dtype=float).T @ np.array([0., 0., 1.])
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.allclose(axis, expected_axis, atol=1e-5, rtol=0):
        raise ValueError("Declared depth axis must equal camera_R.T @ [0,0,1] (positive CV camera depth)")
    translations = np.asarray(motion["obj_t"][:prefix_end + 1], dtype=float)
    rotations = np.asarray(motion["obj_R"][:prefix_end + 1], dtype=float)
    translation_error = float(np.linalg.norm(translations - translations[0], axis=1).max())
    relative = rotations @ rotations[0].T
    # atan2 remains well-conditioned for small rotations; unlike acos(trace),
    # it does not mistake float32 identity roundoff for measurable drift.
    skew = np.stack((relative[:, 2, 1] - relative[:, 1, 2],
                     relative[:, 0, 2] - relative[:, 2, 0],
                     relative[:, 1, 0] - relative[:, 0, 1]), axis=-1) / 2
    angle = np.arctan2(np.linalg.norm(skew, axis=1), (np.trace(relative, axis1=1, axis2=2) - 1) / 2)
    rotation_error = float(angle.max())
    if translation_error > translation_tolerance or rotation_error > rotation_tolerance:
        raise ValueError(f"Object stationary-prefix check failed: {translation_error:.6g} m, {rotation_error:.6g} rad. Correct/review inherited drift separately")
    candidate = {key: value.copy() for key, value in motion.items()}
    candidate["trans"] = (motion["trans"].astype(np.float64) + offset_m * axis).astype(motion["trans"].dtype)
    if not np.isfinite(candidate["trans"]).all():
        raise ValueError("Depth offset overflows native translation dtype")
    if np.array_equal(candidate["trans"], motion["trans"]):
        raise ValueError("Depth offset is below the source dtype precision; no native change would be exported")
    for name in motion:
        if name != "trans" and candidate[name].tobytes() != motion[name].tobytes():
            raise AssertionError(f"Unexpected frozen field change: {name}")
    report = dict(schema_version=1, status="UNREVIEWED", stage="Stage 1 constant camera-depth trial",
                  temporal_application="constant offset to every original source frame",
                  velocity_effect="A constant translation does not repair depth velocity; velocities are preserved up to native dtype roundoff",
                  editable_fields=["trans"], frozen_fields=sorted(set(motion) - {"trans"}),
                  contact_source_frame=contact_frame, camera_convention="Xcv = R @ Xworld + t",
                  camera_depth_axis_world=axis.tolist(), offset_m=float(offset_m), bound_m=float(bound_m),
                  stationary_prefix=dict(window=[0, prefix_end], max_translation_m=translation_error,
                                         max_rotation_rad=rotation_error,
                                         translation_tolerance_m=float(translation_tolerance),
                                         rotation_tolerance_rad=float(rotation_tolerance)),
                  source_evidence=evidence, source_evidence_independently_verified=False,
                  contact_collision_acceptance=False,
                  limitations=["No contact, source-image, depth-ray, support, or collision acceptance",
                               "No camera calibration or inferred contact event",
                               "Canonical NPZ only; Stage 2 plan finalization cannot apply this root shift",
                               "Derived caches in extra arrays are preserved and must be rebuilt"],
                  created_utc=datetime.now(timezone.utc).isoformat())
    return candidate, report


def export_review_note(candidate_path, note_path, *, reviewer, status, scope, evidence, notes):
    candidate = Path(candidate_path).expanduser().resolve()
    if not candidate.is_file():
        raise ValueError("Choose the actual exported candidate NPZ to review")
    if status not in {"UNREVIEWED", "PARTIAL", "REJECTED", "ACCEPTED_FOR_STATED_SCOPE"}:
        raise ValueError("Invalid explicit review status")
    if not reviewer.strip() or not scope.strip() or not notes.strip():
        raise ValueError("Review note requires reviewer, stated scope, and findings/unresolved issues")
    if status != "UNREVIEWED" and not evidence.strip():
        raise ValueError("Record the actual review evidence for this explicit decision")
    note = dict(schema_version=1, kind="manual independent review note", status=status,
                candidate=dict(path=str(candidate), sha256=digest(candidate)), reviewer=reviewer,
                scope=scope, evidence=evidence, findings=notes, decision_source="explicit user entry",
                created_utc=datetime.now(timezone.utc).isoformat(),
                candidate_report_unchanged=True, plugin_inferred_approval=False)
    write_json(note_path, note)
    return note
