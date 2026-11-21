"""NumPy-only contracts for bounded native joint-curve edits.

Source frames are zero based.  Only the strict interior of each declared
window may change; endpoints, all other channels, and source metadata remain
exact.  Blender evaluates the curves separately and supplies their samples.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np


NUMERIC_TOLERANCE = 1e-6
INVALIDATED_FIELDS = frozenset({
    "body_pos", "body_quat_wxyz", "geom_pos", "geom_R",
    "qvel", "qacc", "qacc_warmstart",
    "body_R", "body_rot", "body_rotmat", "body_quat", "body_positions",
    "body_world_pos", "body_world_R", "body_world_quat", "body_world_transforms",
    "joint_world_pos", "joint_world_R", "joint_world_quat", "joint_world_transforms",
    "xpos", "xmat", "xquat", "xipos", "ximat", "xanchor", "xaxis",
    "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "cam_xpos", "cam_xmat",
    "light_xpos", "light_xdir", "subtree_com", "cvel", "cacc",
    "native_task_errors", "native_task_errors_status",
    "parent_native_task_errors", "parent_native_task_errors_status",
    "parent_native_task_errors_scope", "solver_diagnostics",
})
CANDIDATE_STATE_FIELDS = frozenset({
    "status", "native_status", "candidate_status", "review_status", "approval_status",
    "selection_status", "validation_status", "acceptance_status", "selected", "accepted",
    "reviewed", "selection", "export_kind", "contact_edit_review_status",
    "contact_edit_provenance_json", "pose_edit_parent_sha256",
    "pose_edit_config_canonical_sha256", "pose_edit_config_file_sha256",
})
PROTECTED_FIELDS = frozenset({
    "qpos", "fps", "timestamps", "dof_names", "qposadr", "joint_names",
    "joint_qposadr", "joint_dofadr", "joint_ranges", "joint_limited",
    "hinge_qposadr", "hinge_dofadr", "finger_seed", "finger_qposadr",
    "body_names", "native_wrist_targets", "focal_length_px", "frame_width",
    "frame_height", "scene_scale", "height_offset", "height_scale",
    "pelvis_rotation", "descriptor_sha256", "raw_native_sha256",
    "fk_cache_valid", "export_kind", "native_status",
})
PROTECTED_PREFIXES = ("source_", "object_", "obj_", "human_", "camera_", "contact_",
                      "material_", "model_", "compiled_model_", "target_", "depth_")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ eagerly without pickle, returning independent arrays."""
    archive = np.load(Path(path), allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ archive")
    with archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("NPZ archive has duplicate array names")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _integer(value, description: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{description} must be an integer")
    return int(value)


def _finite_number(value, description: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{description} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{description} must be a finite number")
    return result


def _native_qpos(native: Mapping) -> np.ndarray:
    if not isinstance(native, Mapping) or "qpos" not in native or "fps" not in native:
        raise ValueError("Native archive requires qpos and scalar fps")
    qpos = np.asarray(native["qpos"])
    if qpos.ndim != 2 or not all(qpos.shape) or qpos.dtype.kind != "f" or not np.isfinite(qpos).all():
        raise ValueError("Native qpos must be a finite floating array of shape [frames, nq]")
    fps = np.asarray(native["fps"])
    if fps.shape != () or fps.dtype.kind not in "ifu" or not np.isfinite(fps).all() or float(fps) <= 0:
        raise ValueError("Native fps must be a finite positive scalar")
    if "timestamps" in native:
        timestamps = np.asarray(native["timestamps"])
        if timestamps.shape != (len(qpos),) or timestamps.dtype.kind not in "fiu" or not np.isfinite(timestamps).all():
            raise ValueError("Native timestamps must be a finite vector matching source frames")
        steps = np.diff(timestamps.astype(np.float64))
        if np.any(steps <= 0) or not np.allclose(steps, 1. / float(fps), atol=NUMERIC_TOLERANCE, rtol=1e-6):
            raise ValueError("Native timestamps must increase with spacing consistent with fps")
    return qpos


def _model_joints(model: Mapping) -> list[dict]:
    required = ("joint_names", "jnt_type", "jnt_limited", "jnt_range", "jnt_qposadr")
    if not isinstance(model, Mapping) or any(key not in model for key in required):
        raise ValueError("Compiled model requires " + ", ".join(required))
    names = np.asarray(model["joint_names"])
    if names.ndim != 1 or names.dtype.kind not in "US" or len(names) == 0:
        raise ValueError("Model joint_names must be a nonempty string vector")
    names = names.astype(str)
    if len(set(names)) != len(names):
        raise ValueError("Model joint names must be unique")
    arrays = {key: np.asarray(model[key]) for key in required[1:]}
    for key in ("jnt_type", "jnt_limited", "jnt_qposadr"):
        if arrays[key].shape != names.shape or arrays[key].dtype.kind not in "iub":
            raise ValueError(f"Model {key} must be an integer vector matching joint_names")
    if arrays["jnt_range"].shape != (len(names), 2) or arrays["jnt_range"].dtype.kind not in "fiu":
        raise ValueError("Model jnt_range must have shape [joints, 2]")
    if not np.isin(arrays["jnt_type"], [0, 1, 2, 3]).all():
        raise ValueError("Model has an unknown native joint type")
    if not np.isin(arrays["jnt_limited"], [0, 1]).all():
        raise ValueError("Model jnt_limited must contain only zero or one")
    addresses = arrays["jnt_qposadr"]
    if np.any(addresses < 0) or len(set(addresses.tolist())) != len(addresses):
        raise ValueError("Model joint qpos addresses must be nonnegative and unique")
    return [dict(joint=name, joint_index=index, address=int(addresses[index]),
                 joint_type=int(arrays["jnt_type"][index]),
                 limited=bool(arrays["jnt_limited"][index]),
                 limits=arrays["jnt_range"][index].astype(float).tolist())
            for index, name in enumerate(names)]


def _editable_joint(joint: dict) -> dict:
    if joint["joint_type"] != 3 or not joint["limited"]:
        raise ValueError(f"Joint {joint['joint']!r} must be an explicitly limited native hinge")
    low, high = joint["limits"]
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(f"Joint {joint['joint']!r} has invalid native limits")
    return joint


def _validate_native_binding(native: Mapping, joints: list[dict]) -> None:
    """Check optional source joint metadata against the supplied model."""
    by_name = {joint["joint"]: joint for joint in joints}
    scalar = [joint for joint in joints if joint["joint_type"] in (2, 3)]

    def names(key):
        values = np.asarray(native[key])
        if values.ndim != 1 or values.dtype.kind not in "US":
            raise ValueError(f"Native {key} must be a string vector")
        values = values.astype(str).tolist()
        if len(set(values)) != len(values):
            raise ValueError(f"Native {key} must not contain duplicate names")
        return values

    def addresses(key):
        values = np.asarray(native[key])
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(f"Native {key} must be an integer vector")
        return values.tolist()

    scalar_names = [joint["joint"] for joint in scalar]
    if "dof_names" in native:
        dof_names = names("dof_names")
        if "qposadr" in native:
            actual = addresses("qposadr")
            if set(dof_names) != set(scalar_names) or len(actual) != len(dof_names):
                raise ValueError("Native dof_names/qposadr binding does not cover compiled model scalar joints")
            if any(address != by_name[name]["address"] for name, address in zip(dof_names, actual)):
                raise ValueError("Native dof_names/qposadr binding mismatches compiled model")
        elif dof_names != scalar_names:
            raise ValueError("Native dof_names order mismatches compiled model scalar joints")
    elif "qposadr" in native and addresses("qposadr") != [joint["address"] for joint in scalar]:
        raise ValueError("Native qposadr mismatches compiled model scalar joints")
    if "joint_names" in native and names("joint_names") != [joint["joint"] for joint in joints]:
        raise ValueError("Native joint_names mismatches compiled model")
    for key, expected in (
        ("joint_qposadr", [joint["address"] for joint in joints]),
        ("hinge_qposadr", [joint["address"] for joint in joints if joint["joint_type"] == 3]),
    ):
        if key in native and addresses(key) != expected:
            raise ValueError(f"Native {key} mismatches compiled model")


def _declared_derived_fields(plan: Mapping) -> list[str]:
    fields = plan.get("derived_fields", [])
    if not isinstance(fields, list) or any(not isinstance(name, str) or not name for name in fields):
        raise ValueError("Plan derived_fields must be a list of nonempty field names")
    if len(fields) != len(set(fields)):
        raise ValueError("Plan derived_fields must not contain duplicates")
    for name in fields:
        if name in PROTECTED_FIELDS or name in CANDIDATE_STATE_FIELDS or name.startswith(PROTECTED_PREFIXES) or name.endswith("_targets"):
            raise ValueError(f"Cannot invalidate protected source field {name!r}")
    return fields


def check_sample_limits(model: Mapping, addresses, values,
                        tolerance: float = NUMERIC_TOLERANCE) -> None:
    """Reject nonfinite/illegal samples with shape ``[..., len(addresses)]``.

    Addresses must identify unique limited hinges.  This also accepts dense
    subframe samples; the caller chooses their times.  No clipping is applied.
    """
    tolerance = _finite_number(tolerance, "Limit tolerance")
    if tolerance < 0:
        raise ValueError("Limit tolerance must be nonnegative")
    try:
        addresses = [_integer(value, "Qpos address") for value in addresses]
    except TypeError as exc:
        raise ValueError("Addresses must be a sequence") from exc
    if not addresses or len(set(addresses)) != len(addresses):
        raise ValueError("Sample addresses must be nonempty and unique")
    by_address = {joint["address"]: joint for joint in _model_joints(model)}
    selected = []
    for address in addresses:
        if address not in by_address:
            raise ValueError(f"Unknown model qpos address {address}")
        selected.append(_editable_joint(by_address[address]))
    samples = np.asarray(values)
    if samples.ndim < 1 or samples.shape[-1] != len(addresses) or samples.size == 0 or samples.dtype.kind not in "fiu" or not np.isfinite(samples).all():
        raise ValueError("Curve samples must be finite with one final-axis value per address")
    low, high = np.asarray([joint["limits"] for joint in selected]).T
    violations = (samples < low - tolerance) | (samples > high + tolerance)
    if np.any(violations):
        index = tuple(np.argwhere(violations)[0])
        joint = selected[index[-1]]
        raise ValueError(f"Original native limit exceeded for {joint['joint']!r}: {samples[index]} outside {joint['limits']}")


def resolve_channels(native: Mapping, model: Mapping, plan: Mapping) -> list[dict]:
    """Validate a version-1 plan and return explicit normalized channel records.

    Each record has ``joint``, ``joint_index``, ``address``, ``window``,
    ``keys``, ``interpolation``, ``auto_smoothing`` and ``limits``. There is no implicit arm or
    finger scope.  Fractional source-frame keys are allowed inside the window.
    """
    qpos = _native_qpos(native)
    if not isinstance(plan, Mapping) or _integer(plan.get("schema_version"), "schema_version") != 1:
        raise ValueError("Expected curve plan schema_version 1")
    channels = plan.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError("Plan channels must be a nonempty list")
    _declared_derived_fields(plan)
    joints = _model_joints(model)
    _validate_native_binding(native, joints)
    by_name = {joint["joint"]: joint for joint in joints}
    result, seen = [], set()
    for channel in channels:
        if not isinstance(channel, Mapping):
            raise ValueError("Each channel must be an object")
        name = channel.get("joint")
        if not isinstance(name, str) or name not in by_name:
            raise ValueError(f"Unknown native joint {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate channel for joint {name!r}")
        seen.add(name)
        joint = _editable_joint(by_name[name])
        address = joint["address"]
        if address >= qpos.shape[1]:
            raise ValueError(f"Joint {name!r} address is outside native qpos")
        window = channel.get("window")
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(f"Channel {name!r} requires a two-integer source window")
        start, end = [_integer(value, "Window endpoint") for value in window]
        if not 0 <= start < end < len(qpos):
            raise ValueError(f"Channel {name!r} window is outside source frames or empty")
        interpolation = channel.get("interpolation")
        if interpolation not in ("BEZIER", "LINEAR"):
            raise ValueError("Interpolation must be BEZIER or LINEAR")
        auto_smoothing = channel.get("auto_smoothing", "NONE")
        if auto_smoothing not in ("NONE", "CONT_ACCEL"):
            raise ValueError("auto_smoothing must be NONE or CONT_ACCEL")
        keys = channel.get("keys")
        if not isinstance(keys, (list, tuple)) or len(keys) < 2:
            raise ValueError(f"Channel {name!r} requires at least its two endpoint keys")
        parsed = []
        for key in keys:
            if not isinstance(key, (list, tuple)) or len(key) != 2:
                raise ValueError("Each curve key must be [source_frame, radians]")
            parsed.append([_finite_number(key[0], "Key frame"), _finite_number(key[1], "Key value")])
        times = np.asarray(parsed)[:, 0]
        if np.any(np.diff(times) <= 0) or times[0] != start or times[-1] != end:
            raise ValueError(f"Channel {name!r} keys must increase strictly and include both window endpoints")
        for frame, value in (parsed[0], parsed[-1]):
            if abs(value - float(qpos[int(frame), address])) > NUMERIC_TOLERANCE:
                raise ValueError(f"Channel {name!r} authored endpoint does not match baseline at frame {int(frame)}")
        check_sample_limits(model, [address], np.asarray(parsed)[:, 1:])
        result.append(dict(joint=name, joint_index=joint["joint_index"], address=address,
                           window=[start, end], keys=parsed, interpolation=interpolation,
                           auto_smoothing=auto_smoothing,
                           limits=joint["limits"]))
    return result


def _array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def merge_curve_samples(native: Mapping, model: Mapping, plan: Mapping,
                        sampled_qpos) -> tuple[dict[str, np.ndarray], dict]:
    """Merge evaluated integer-frame samples after validating the edit boundary.

    Supply a baseline copy with only strict window interiors replaced by actual
    evaluated curve values.  A changed endpoint or frozen sample is an error,
    even when numerically small; this function never repairs one silently.
    """
    channels = resolve_channels(native, model, plan)
    baseline = _native_qpos(native)
    samples = np.asarray(sampled_qpos)
    if samples.shape != baseline.shape or samples.dtype.kind != "f" or not np.isfinite(samples).all():
        raise ValueError("Sampled qpos must be a finite floating array matching native qpos")
    editable = np.zeros(baseline.shape, dtype=bool)
    for channel in channels:
        start, end = channel["window"]
        editable[start + 1:end, channel["address"]] = True
    if not np.array_equal(samples[~editable], baseline[~editable]):
        frame, address = np.argwhere((samples != baseline) & ~editable)[0]
        raise ValueError(f"Frozen or out-of-window sample changed at source frame {frame}, qpos address {address}; endpoints must remain exact")
    addresses = [channel["address"] for channel in channels]
    check_sample_limits(model, addresses, samples[:, addresses])
    qpos = baseline.copy()
    qpos[editable] = samples[editable]
    check_sample_limits(model, addresses, qpos[:, addresses])
    changed = qpos != baseline
    actual_changed = bool(np.any(changed))
    declared = _declared_derived_fields(plan)
    invalidated = sorted(set(native) & (INVALIDATED_FIELDS | set(declared))) if actual_changed else []
    candidate = {key: np.array(value, copy=True) for key, value in native.items() if key not in invalidated}
    candidate["qpos"] = qpos
    prior_validity = np.asarray(native.get("fk_cache_valid", []))
    cache_validity = bool(prior_validity) if prior_validity.shape == () and prior_validity.dtype.kind == "b" else None
    if actual_changed:
        candidate["fk_cache_valid"] = np.asarray(False)
        cache_validity = False
    # A parent's approval describes its own motion, never this exported edit.
    # Preserve all prior histories without overwriting existing parent fields.
    parent_provenance = {}
    for key in sorted(CANDIDATE_STATE_FIELDS & set(candidate)):
        parent_key = "parent_" + key
        while parent_key in candidate:
            parent_key = "parent_" + parent_key
        candidate[parent_key] = candidate.pop(key)
        parent_provenance[key] = parent_key
    candidate["status"] = np.asarray("UNREVIEWED")
    candidate["selected"] = np.asarray(False)
    report = dict(schema_version=1, status="UNREVIEWED", review_status="UNREVIEWED", selected=False,
                  source_frame_count=len(baseline), fps=float(np.asarray(native["fps"])),
                  channels=channels, changed_values=int(np.count_nonzero(changed)),
                  changed_frames=np.flatnonzero(np.any(changed, axis=1)).tolist(),
                  maximum_delta_rad=float(np.max(np.abs(qpos - baseline))),
                  source_qpos_sha256=_array_hash(baseline), candidate_qpos_sha256=_array_hash(qpos),
                  invalidated_fields=invalidated, declared_derived_fields=declared,
                  parent_provenance_fields=parent_provenance,
                  actual_qpos_changed=actual_changed, fk_cache_valid=cache_validity,
                  unknown_cached_data_status=("All motion-derived caches are invalid after this edit, including custom arrays retained because they were not declared in derived_fields. Recompute before use."
                                              if actual_changed else "Qpos unchanged; existing cache validity was not established or changed."),
                  frozen_samples_exact=True,
                  endpoint_samples_exact=True, source_metadata_preserved=True,
                  validation_scope="Integer curve samples and declared boundaries; dense subframes and contact review are separate")
    candidate["export_kind"] = np.asarray("native_curve_edit_UNREVIEWED")
    candidate["contact_edit_review_status"] = np.asarray("UNREVIEWED")
    candidate["contact_edit_provenance_json"] = np.asarray(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return candidate, report
