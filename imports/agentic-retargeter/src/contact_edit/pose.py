"""Bounded editing of one native MuJoCo pose against fixed material contacts.

``solve_pose(config, config_dir='.', candidate_guard=None)`` returns a PoseResult
containing an independent trajectory dictionary and a diagnostic report. Paths
are relative to config_dir. ``run(config_path, out_dir, candidate_guard=None)``
also saves a new candidate directory; the CLI is::

    python pose.py --config edit.json --out new_candidate

Required config keys: schema_version=1, model_xml, trajectory_npz,
object_mesh_npz, frame, editable_joints and contacts. editable_joints maps native
limited hinge names to {"delta_bounds_rad": [lower, upper]}; each source-relative
corridor must include zero. A contact has name, body_name, point_local,
normal_local, object_face_id, object_barycentric, and optionally
object_normal_sign (+1/-1), position_weight (>0), normal_weight (>=0, default 0).
Normal opposition is diagnostic unless explicitly weighted. No targets are inferred.

The trajectory NPZ requires qpos[T,nq], timestamps[T], fps, obj_R[T,3,3],
obj_t[T,3], and object_scale (scalar or XYZ). Original object vertices/faces come
from the mesh NPZ; scale is applied before computing triangle normals. Unknown
metadata is retained. Recognized derived fields and optional ``derived_fields``
are removed after changes; ``fk_cache_valid=False`` invalidates ALL input derived
state, including custom caches that callers should list in derived_fields.
Candidate NPZs carry their own UNREVIEWED/REJECTED_BY_GUARD status and export
provenance. Input acceptance/selection fields move to explicitly parent-prefixed
metadata; a parent's accepted status never describes the edited candidate.

Optional solver keys: max_evaluations (default 100, at most 10000), pose_weight
(default 1e-5). The least-squares objective is sum(position_weight * gap**2 +
normal_weight * ||pad_normal + object_normal||**2) plus a source-pose prior.
Optional expected_input_sha256 maps model_xml/trajectory_npz/object_mesh_npz to
exact file hashes. candidate_guard receives a COPY of the proposed trajectory
and must return a JSON-serializable dict with a boolean ``passed``. Passing only
leaves the candidate UNREVIEWED; rejection labels it REJECTED_BY_GUARD.

This module performs no source-image, mesh-collision, contact-completeness,
temporal-continuity, force-closure, or dynamics validation. It is a pose tool;
editing one animation frame can create discontinuities. Native FK/Jacobian and
material-anchor patterns are adapted from GRAIL-agentic-kinematic's
grail/agentic_kinematic/{robot,contact}.py and retargeting/native.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class PoseResult:
    trajectory: dict[str, np.ndarray]
    report: dict


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, indent=2).encode() + b"\n"


def _finite(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"{name} must be a numeric array with shape {shape}")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _weight(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _keys(value, required, optional, name):
    if not isinstance(value, dict) or set(value) - required - optional or required - set(value):
        raise ValueError(f"{name}: required keys {sorted(required)}, optional {sorted(optional)}")


def _unit(value, name):
    result = _finite(value, (3,), name)
    if not np.isclose(np.linalg.norm(result), 1.0, atol=1e-8, rtol=0):
        raise ValueError(f"{name} must be a unit normal; normalization is not implicit")
    return result


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _equal(a, b):
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _model_dependencies(path):
    # Same include/asset binding pattern as agentic_kinematic.robot. Nested
    # include paths are relative to their containing XML; assets use main XML.
    pending, paths, roots = [path], set(), []
    while pending:
        current = pending.pop().resolve(strict=True)
        if current in paths:
            continue
        paths.add(current)
        root = ET.parse(current).getroot()
        roots.append(root)
        pending.extend(current.parent / node.attrib["file"] for node in root.iter("include"))
    compiler = {}
    for root in roots:
        for node in root.iter("compiler"):
            compiler.update(node.attrib)
    for root in roots:
        for node in root.iter():
            if node.tag not in {"mesh", "texture", "hfield", "skin"} or "file" not in node.attrib:
                continue
            directory = compiler.get("assetdir", "")
            if node.tag == "mesh":
                directory = compiler.get("meshdir", directory)
            elif node.tag == "texture":
                directory = compiler.get("texturedir", directory)
            name = node.attrib["file"]
            if compiler.get("strippath", "false") == "true":
                name = Path(name).name
            paths.add((path.parent / directory / name).resolve(strict=True))
    return sorted(paths)


_NATIVE_FIELDS = {
    "qpos", "timestamps", "fps", "obj_R", "obj_t", "object_scale", "dof_names",
    "qposadr", "source_path", "source_sha256", "compiled_model_sha256", "scene_scale",
    "height_scale", "height_offset", "pelvis_rotation", "contact_config_sha256", "export_kind",
}
_DERIVED_FIELDS = {
    "xpos", "xmat", "xquat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat",
    "qvel", "qacc", "cvel", "subtree_com", "body_pos", "body_rot", "body_quat",
    "joint_pos", "joint_rot", "joint_quat", "dof_pos", "root_pos", "root_rot",
    "root_quat", "root_vel", "root_ang_vel", "local_body_pos", "global_body_pos",
    "body_quat_wxyz", "geom_pos", "geom_R", "native_task_errors",
}
_CANDIDATE_STATE_FIELDS = {
    "status", "candidate_status", "review_status", "approval_status", "selection_status",
    "validation_status", "acceptance_status", "selected", "accepted", "reviewed", "selection",
    "export_kind", "pose_edit_parent_sha256", "pose_edit_config_canonical_sha256",
    "pose_edit_config_file_sha256",
}


def _is_derived(key):
    return key in _DERIVED_FIELDS or key.startswith(("fk_", "parent_native_task_errors")) or "cache" in key.lower() or key.endswith(
        ("_positions", "_rotations", "_orientations", "_velocities", "_velocity", "_acceleration")
    )


def solve_pose(config, config_dir=".", *, candidate_guard=None):
    """Optimize one mounted native pose; never mutate inputs or accept a grasp.

    Only declared limited hinges at config['frame'] may vary. All other qpos
    values and all native object/timeline metadata remain bitwise identical.
    The optional guard sees the complete candidate, so it can perform caller-
    specific original-mesh, source, or transition checks before using the result.
    """
    import mujoco
    import scipy
    from scipy.optimize import least_squares

    _keys(config, {"schema_version", "model_xml", "trajectory_npz", "object_mesh_npz",
                   "frame", "editable_joints", "contacts"},
          {"solver", "derived_fields", "expected_input_sha256"}, "config")
    if _integer(config["schema_version"], "schema_version", 1) != 1:
        raise ValueError("Unsupported schema_version")
    paths = {}
    for key in ("model_xml", "trajectory_npz", "object_mesh_npz"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
        paths[key] = (Path(config_dir) / config[key]).resolve(strict=True)
    input_hashes = {key: _sha(path) for key, path in paths.items()}
    expected = config.get("expected_input_sha256", {})
    if not isinstance(expected, dict) or set(expected) - set(paths):
        raise ValueError("expected_input_sha256 has unknown input names")
    for key, digest in expected.items():
        if input_hashes[key] != digest:
            raise ValueError(f"Input hash mismatch: {key}")
    model_hashes = {str(path): _sha(path) for path in _model_dependencies(paths["model_xml"])}
    model = mujoco.MjModel.from_xml_path(str(paths["model_xml"]))
    if model.neq:
        raise ValueError("Models with equality constraints require a constraint-aware solver")
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, compiled)
    compiled_sha = hashlib.sha256(compiled.tobytes()).hexdigest()
    source = _load_npz(paths["trajectory_npz"])
    required = {"qpos", "timestamps", "fps", "obj_R", "obj_t", "object_scale"}
    if required - set(source):
        raise ValueError(f"Missing trajectory fields: {sorted(required - set(source))}")
    qpos = source["qpos"]
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos) or qpos.dtype.kind != "f":
        raise ValueError("qpos must be a nonempty floating-point [T, model.nq] array")
    _finite(qpos, qpos.shape, "qpos")
    frames = len(qpos)
    frame = _integer(config["frame"], "frame")
    if frame >= frames:
        raise ValueError("frame lies outside the source timeline")
    fps_array = _finite(source["fps"], (), "fps")
    fps = _weight(float(fps_array), "fps", positive=True)
    timestamps = _finite(source["timestamps"], (frames,), "timestamps")
    if frames > 1 and not np.allclose(np.diff(timestamps), 1 / fps, atol=1e-9, rtol=1e-6):
        raise ValueError("timestamps must match the original uniform source fps")
    obj_R = _finite(source["obj_R"], (frames, 3, 3), "obj_R")
    obj_t = _finite(source["obj_t"], (frames, 3), "obj_t")
    # Native float32 reconstruction rotations can retain a few ulps of error
    # after float64 serialization. Accept that roundoff without projecting or
    # otherwise changing the original, frozen object trajectory.
    if not np.allclose(obj_R @ obj_R.swapaxes(-1, -2), np.eye(3), atol=1e-6, rtol=0) or not np.allclose(
        np.linalg.det(obj_R), 1, atol=1e-6, rtol=0
    ):
        raise ValueError("obj_R must contain proper rotation matrices")
    raw_scale = source["object_scale"]
    if raw_scale.shape not in ((), (3,)):
        raise ValueError("object_scale must be a scalar or XYZ vector")
    scale = np.broadcast_to(_finite(raw_scale, raw_scale.shape, "object_scale"), (3,))
    if np.any(scale <= 0):
        raise ValueError("object_scale must be positive")
    if "compiled_model_sha256" in source and (
        source["compiled_model_sha256"].shape != () or source["compiled_model_sha256"].item() != compiled_sha
    ):
        raise ValueError("Trajectory compiled_model_sha256 does not match actual model")
    if ("dof_names" in source) != ("qposadr" in source):
        raise ValueError("dof_names and qposadr must be provided together")
    if "dof_names" in source:
        scalar = np.flatnonzero(np.isin(model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]))
        expected_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in scalar]
        names, addresses = source["dof_names"], source["qposadr"]
        if names.ndim != 1 or names.dtype.kind not in "US" or len(set(names.tolist())) != len(names):
            raise ValueError("dof_names must contain unique native names")
        if addresses.shape != names.shape or addresses.dtype.kind not in "iu":
            raise ValueError("qposadr must be an integer vector matching dof_names")
        binding = dict(zip(names.tolist(), addresses.tolist()))
        if binding != dict(zip(expected_names, model.jnt_qposadr[scalar].tolist())):
            raise ValueError("Native name/address binding differs from the actual model")
    for jid, kind in enumerate(model.jnt_type):
        address = int(model.jnt_qposadr[jid])
        if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            start = address + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
            if not np.allclose(np.linalg.norm(qpos[:, start:start + 4], axis=1), 1, atol=1e-7, rtol=0):
                raise ValueError("Native source quaternion must have unit length")
        elif model.jnt_limited[jid]:
            if not model.jnt_range[jid, 0] <= qpos[frame, address] <= model.jnt_range[jid, 1]:
                raise ValueError("Source pose lies outside native joint limits; no clipping is performed")
    edits = config["editable_joints"]
    if not isinstance(edits, dict) or not edits:
        raise ValueError("editable_joints must be a nonempty name-to-corridor mapping")
    qids, dids, lower, upper = [], [], [], []
    for name, corridor in edits.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Editable joints must have nonempty native names")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[jid]:
            raise ValueError(f"Editable joint must be an actual limited hinge: {name}")
        _keys(corridor, {"delta_bounds_rad"}, set(), name)
        delta = _finite(corridor["delta_bounds_rad"], (2,), f"{name} delta_bounds_rad")
        if not delta[0] <= 0 <= delta[1] or delta[0] >= delta[1]:
            raise ValueError("A source correction corridor must include zero and have positive width")
        address = int(model.jnt_qposadr[jid])
        bounds = qpos[frame, address] + delta
        lo, hi = max(bounds[0], model.jnt_range[jid, 0]), min(bounds[1], model.jnt_range[jid, 1])
        if not lo < hi:
            raise ValueError(f"Empty native-limit/correction-corridor intersection: {name}")
        qids.append(address)
        dids.append(int(model.jnt_dofadr[jid]))
        lower.append(lo)
        upper.append(hi)
    qids, dids = np.asarray(qids), np.asarray(dids)
    lower, upper = np.asarray(lower), np.asarray(upper)
    initial = qpos[frame, qids].astype(np.float64)
    mesh = _load_npz(paths["object_mesh_npz"])
    if not {"vertices", "faces"} <= set(mesh):
        raise ValueError("Mesh NPZ requires original vertices and faces")
    vertices, faces = mesh["vertices"], mesh["faces"]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Original vertices must have shape [V,3]")
    vertices = _finite(vertices, vertices.shape, "original vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in "iu":
        raise ValueError("Original faces must be a nonempty integer [F,3] array")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Original triangle indices are outside the mesh")
    contacts = config["contacts"]
    if not isinstance(contacts, list) or not contacts:
        raise ValueError("contacts must contain explicit material anchors")
    anchors, contact_names = [], set()
    for item in contacts:
        _keys(item, {"name", "body_name", "point_local", "normal_local", "object_face_id", "object_barycentric"},
              {"object_normal_sign", "position_weight", "normal_weight"}, "contact")
        name = item["name"]
        if not isinstance(name, str) or not name or name in contact_names:
            raise ValueError("Contacts need nonempty unique names")
        contact_names.add(name)
        if not isinstance(item["body_name"], str):
            raise ValueError("body_name must be a native body name")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"])
        if body <= 0:
            raise ValueError(f"Unknown or world contact body: {item['body_name']}")
        face = _integer(item["object_face_id"], "object_face_id")
        if face >= len(faces):
            raise ValueError("object_face_id outside original mesh")
        bary = _finite(item["object_barycentric"], (3,), "object_barycentric")
        if np.any(bary < 0) or not np.isclose(bary.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError("Barycentric weights must be nonnegative and sum to one")
        triangle = vertices[faces[face]] * scale
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if not length > 1e-14:
            raise ValueError("Authored original triangle is degenerate; its normal is undefined")
        sign = item.get("object_normal_sign", 1)
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ValueError("object_normal_sign must be +1 or -1")
        anchors.append(dict(
            name=name, body=body, local=_finite(item["point_local"], (3,), "point_local"),
            pad_normal=_unit(item["normal_local"], "normal_local"),
            target=obj_R[frame] @ (bary @ triangle) + obj_t[frame],
            normal=obj_R[frame] @ (sign * normal / length),
            wp=np.sqrt(_weight(item.get("position_weight", 1.0), "position_weight", positive=True)),
            wn=np.sqrt(_weight(item.get("normal_weight", 0.0), "normal_weight")),
            face=face, bary=bary,
        ))
    solver_config = config.get("solver", {})
    _keys(solver_config, set(), {"max_evaluations", "pose_weight"}, "solver")
    max_evaluations = _integer(solver_config.get("max_evaluations", 100), "max_evaluations", 1)
    if max_evaluations > 10000:
        raise ValueError("max_evaluations exceeds the bounded pose-tool limit of 10000")
    pose_weight = np.sqrt(_weight(solver_config.get("pose_weight", 1e-5), "pose_weight"))
    derived = config.get("derived_fields", [])
    if not isinstance(derived, list) or any(not isinstance(k, str) for k in derived) or len(set(derived)) != len(derived):
        raise ValueError("derived_fields must be a unique string list")
    if set(derived) & _NATIVE_FIELDS:
        raise ValueError("derived_fields cannot remove native pose/object/timeline/source metadata")
    if set(derived) - set(source):
        raise ValueError("derived_fields references absent trajectory fields")
    data = mujoco.MjData(model)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def evaluate(values, with_jacobian=False):
        data.qpos[:] = qpos[frame]
        data.qpos[qids] = values
        mujoco.mj_forward(model, data)
        residuals, jacobians = [], []
        for anchor in anchors:
            rotation = data.xmat[anchor["body"]].reshape(3, 3)
            point = data.xpos[anchor["body"]] + rotation @ anchor["local"]
            normal = rotation @ anchor["pad_normal"]
            residuals.extend((anchor["wp"] * (point - anchor["target"]), anchor["wn"] * (normal + anchor["normal"])))
            if with_jacobian:
                mujoco.mj_jac(model, data, jacp, jacr, point, anchor["body"])
                x, y, z = normal
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                jacobians.extend((anchor["wp"] * jacp[:, dids], -anchor["wn"] * skew @ jacr[:, dids]))
        residuals.append(pose_weight * (values - initial))
        if with_jacobian:
            jacobians.append(pose_weight * np.eye(len(initial)))
            return np.vstack(jacobians)
        return np.concatenate(residuals)

    def diagnostics(values):
        evaluate(values)
        result = []
        for anchor in anchors:
            rotation = data.xmat[anchor["body"]].reshape(3, 3)
            point = data.xpos[anchor["body"]] + rotation @ anchor["local"]
            normal = rotation @ anchor["pad_normal"]
            result.append({"name": anchor["name"], "object_face_id": anchor["face"],
                           "object_barycentric": anchor["bary"].tolist(),
                           "gap_m": float(np.linalg.norm(point - anchor["target"])),
                           "normal_opposition": float(-normal @ anchor["normal"]),
                           "pad_point_world": point.tolist(), "pad_normal_world": normal.tolist(),
                           "object_point_world": anchor["target"].tolist(),
                           "object_normal_world": anchor["normal"].tolist()})
        return result

    before = diagnostics(initial)
    optimization = least_squares(evaluate, initial, jac=lambda x: evaluate(x, True),
                                 bounds=(lower, upper), max_nfev=max_evaluations,
                                 ftol=1e-12, xtol=1e-12, gtol=1e-12)
    candidate = {key: value.copy() for key, value in source.items()}
    candidate["qpos"][frame, qids] = optimization.x
    # Audit the actual stored dtype, including float32 rounding at joint bounds.
    saved_values = candidate["qpos"][frame, qids].astype(np.float64)
    if not np.isfinite(saved_values).all() or np.any(saved_values < lower) or np.any(saved_values > upper):
        raise ValueError("Stored candidate violates declared bounds; no clipping is performed")
    frozen_mask = np.ones(qpos.shape, dtype=bool)
    frozen_mask[frame, qids] = False
    if not np.array_equal(candidate["qpos"][frozen_mask], qpos[frozen_mask]):
        raise RuntimeError("Frozen native coordinates changed")
    changed = not np.array_equal(candidate["qpos"], qpos)
    invalidated = []
    if changed:
        invalidated = sorted(k for k in candidate if k not in _NATIVE_FIELDS and (_is_derived(k) or k in derived))
        for key in invalidated:
            del candidate[key]
        candidate["fk_cache_valid"] = np.asarray(False)
    # A stored parent review/export label must not appear to approve this edit.
    # Preserve histories without overwriting any existing parent-prefixed field.
    parent_provenance = {}
    for key in sorted(_CANDIDATE_STATE_FIELDS & set(candidate)):
        parent_key = "parent_" + key
        while parent_key in candidate:
            parent_key = "parent_" + parent_key
        candidate[parent_key] = candidate.pop(key)
        parent_provenance[key] = parent_key
    config_canonical_sha = hashlib.sha256(_json_bytes(config)).hexdigest()
    candidate["pose_edit_parent_sha256"] = np.asarray(input_hashes["trajectory_npz"])
    candidate["pose_edit_config_canonical_sha256"] = np.asarray(config_canonical_sha)

    def set_candidate_status(status):
        candidate["status"] = np.asarray(status)
        candidate["export_kind"] = np.asarray("contact_edit_kit_native_pose_" + status)
        candidate["selected"] = np.asarray(False)

    set_candidate_status("UNREVIEWED")
    preserved = all(_equal(candidate[parent_provenance.get(key, key)], value) for key, value in source.items()
                    if key not in invalidated and key not in {"qpos", "fk_cache_valid"})
    guard = {"performed": False, "passed": None}
    if candidate_guard is not None:
        decision = candidate_guard({key: value.copy() for key, value in candidate.items()})
        if not isinstance(decision, dict) or type(decision.get("passed")) is not bool:
            raise ValueError("candidate_guard must return a dict containing boolean passed")
        _json_bytes(decision)
        guard = {**decision, "performed": True}
    status = "REJECTED_BY_GUARD" if guard["passed"] is False else "UNREVIEWED"
    set_candidate_status(status)
    report = {
        "status": status,
        "selected": False, "frame": frame, "frame_count": frames, "fps": fps,
        "optimizer": {"success": bool(optimization.success), "status": int(optimization.status),
                      "message": str(optimization.message), "evaluations": int(optimization.nfev),
                      "max_evaluations": max_evaluations, "cost": float(optimization.cost)},
        "editable_joints": {name: {"qpos_address": int(qid), "lower_rad": float(lo), "upper_rad": float(hi),
                                  "source_rad": float(old), "candidate_rad": float(new)}
                            for name, qid, lo, hi, old, new in zip(edits, qids, lower, upper, initial, saved_values)},
        "contacts_before": before, "contacts_after": diagnostics(saved_values),
        "checks": {"frozen_qpos_exact": True, "native_limits_and_correction_corridors": True,
                   "preserved_object_timeline_and_metadata_exact": preserved},
        "qpos_changed": changed, "dropped_derived_fields": invalidated,
        "parent_provenance_fields": parent_provenance,
        "derived_state": "ALL_INPUT_DERIVED_STATE_INVALID" if changed else "UNCHANGED",
        "candidate_guard": guard,
        "omitted_checks": ["source image alignment", "original-mesh penetration/intersection",
                           "contact completeness", "adjacent-frame continuity", "force closure", "dynamics"],
        "inputs": {key: {"path": str(paths[key]), "sha256": digest} for key, digest in input_hashes.items()},
        "model_dependency_sha256": model_hashes, "compiled_model_sha256": compiled_sha,
        "config_canonical_sha256": config_canonical_sha,
        "implementation_sha256": _sha(__file__),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "mujoco": mujoco.__version__},
    }
    for path, digest in model_hashes.items():
        if _sha(path) != digest:
            raise ValueError("Model dependency changed while solving")
    for key, path in paths.items():
        if _sha(path) != input_hashes[key]:
            raise ValueError("Input changed while solving")
    _json_bytes(report)
    return PoseResult(candidate, report)


def run(config_path, out_dir, *, candidate_guard=None):
    """Read config and exclusively create a native candidate plus audit report."""
    config_path, output = Path(config_path).resolve(strict=True), Path(out_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    config_bytes = config_path.read_bytes()

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    config = json.loads(config_bytes, object_pairs_hook=unique_pairs)
    result = solve_pose(config, config_path.parent, candidate_guard=candidate_guard)
    if config_path.read_bytes() != config_bytes:
        raise ValueError("Config changed while solving")
    result.trajectory["pose_edit_config_file_sha256"] = np.asarray(hashlib.sha256(config_bytes).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    with (output / "candidate.npz").open("xb") as stream:
        np.savez_compressed(stream, **result.trajectory)
    reread = _load_npz(output / "candidate.npz")
    if set(reread) != set(result.trajectory) or not all(_equal(reread[k], v) for k, v in result.trajectory.items()):
        raise RuntimeError("Native candidate failed exact serialization roundtrip")
    result.report["serialized_exact"] = True
    result.report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    result.report["candidate_sha256"] = _sha(output / "candidate.npz")
    (output / "config.json").write_bytes(config_bytes)
    (output / "report.json").write_bytes(_json_bytes(result.report))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="New output directory; existing paths are rejected")
    args = parser.parse_args()
    result = run(args.config, args.out)
    print(json.dumps({"output": str(Path(args.out).resolve()), "status": result.report["status"],
                      "optimizer_success": result.report["optimizer"]["success"],
                      "contacts": result.report["contacts_after"],
                      "omitted_checks": result.report["omitted_checks"]}, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
