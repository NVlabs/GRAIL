"""Hash-bound native hand-sequence targets for whole-body SOMA.

This is a NumPy-only input contract, not a contact detector or motion gate.
Object-local wrist translations already use scaled metric object coordinates;
they must NEVER be multiplied by object_scale again. A shared rigid world-frame
conversion changes world targets and object poses, leaving local targets intact.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


_WEIGHTS = dict(translation_weight=1., rotation_weight=1., finger_weight=1.,
                translation_sigma_m=.02, rotation_sigma_rad=.2, finger_sigma_rad=.2)
_OPTIONAL = {"enabled_hands", "world_to_solver", "expected_object_mesh_sha256", "source_hand_weight_multiplier"}
_REQUIRED = {"handoff_json", "handoff_sha256", "targets_npz", "targets_sha256"}
_CONTACT_PHASES = {"contact_acquisition", "hold", "support", "slide"}
_OTHER_PHASES = {"no_contact", "unknown", "release", "approach"}


def _hash(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _finite(value, shape, label):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{label} must be finite numeric {shape}")
    return raw.astype(np.float64, copy=True)


def _rotations(value, shape, label):
    result = _finite(value, shape, label)
    if (not np.allclose(result @ np.swapaxes(result, -1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(result), 1., atol=1e-5, rtol=0)):
        raise ValueError(f"{label} must contain proper rotations")
    return result


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def validate_robot_hand_target_config(value):
    """Require both artifact hashes; weight values are energy multipliers.

    Residual scales are sqrt(weight)/sigma, where sigma is in metres/radians.
    No config means no objective; callers should not call this validator when
    the feature is absent. All enabled pose terms remain strictly positive.
    """
    if (not isinstance(value, dict) or _REQUIRED - set(value)
            or set(value) - _REQUIRED - _OPTIONAL - set(_WEIGHTS)):
        raise ValueError("Invalid native robot hand target configuration fields")
    result = {**_WEIGHTS, "source_hand_weight_multiplier": 0., **value}
    for name in ("handoff_json", "targets_npz"):
        if not isinstance(result[name], str) or not result[name]:
            raise ValueError(f"{name} must be a nonempty path")
    for name in ("handoff_sha256", "targets_sha256", "expected_object_mesh_sha256"):
        if name in result:
            _hash(result[name], name)
    for name in _WEIGHTS:
        number = result[name]
        if (isinstance(number, bool) or not isinstance(number, Real)
                or not math.isfinite(number) or number <= 0):
            raise ValueError(f"{name} must be finite and positive")
        result[name] = float(number)
    multiplier = result["source_hand_weight_multiplier"]
    if (isinstance(multiplier, bool) or not isinstance(multiplier, Real)
            or not math.isfinite(multiplier) or multiplier < 0):
        raise ValueError("source_hand_weight_multiplier must be finite and nonnegative")
    result["source_hand_weight_multiplier"] = float(multiplier)
    if "enabled_hands" in result:
        hands = result["enabled_hands"]
        if (not isinstance(hands, (tuple, list)) or not hands
                or len(set(hands)) != len(hands) or set(hands) - {"left", "right"}):
            raise ValueError("enabled_hands must be unique left/right names")
        result["enabled_hands"] = list(hands)
    transform = _finite(result.get("world_to_solver", np.eye(4)), (4, 4), "world_to_solver")
    _rotations(transform[:3, :3], (3, 3), "world_to_solver rotation")
    if not np.array_equal(transform[3], [0, 0, 0, 1]):
        raise ValueError("world_to_solver must be a rigid homogeneous transform")
    result["world_to_solver"] = transform.tolist()
    return result


def _bound_file(path, digest, base_path):
    path = (Path(base_path) / path).resolve(strict=True)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Native hand target artifact hash changed: {path.name}")
    return path, data


def load_robot_hand_targets(config, *, fps, frame_count, object_positions,
                            object_rotations, available_joint_names,
                            available_body_names, object_scale=None,
                            base_path="."):
    """Load immutable targets and verify their clock/model/object alignment.

    ``object_positions``/``object_rotations`` are the ORIGINAL collision object
    trajectory in solver coordinates, never the currently optimized trajectory.
    Both exact NPZ and JSON hashes must be supplied by the owning experiment.
    ``world_to_solver`` must be explicitly supplied for a nonidentity frame
    conversion. We reject implicit per-frame alignment, scaling, or retiming.

    Returns a dict containing hands (side-keyed dictionaries), timestamps,
    object_reference_positions/rotations, object_scale, config and report.
    Hand arrays are source-frame indexed and read-only. ``contact_mask`` marks
    where object-relative pose tracking applies; outside this mask use the
    converted world target. These masks do not certify physical contact.
    """
    config = validate_robot_hand_target_config(config)
    if (type(frame_count) is not int or frame_count < 1 or isinstance(fps, bool)
            or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid native hand target source clock")
    json_path, json_bytes = _bound_file(config["handoff_json"], config["handoff_sha256"], base_path)
    npz_path, npz_bytes = _bound_file(config["targets_npz"], config["targets_sha256"], base_path)
    metadata = json.loads(json_bytes)
    if (metadata.get("schema") != "unaccepted_hand_sequence_soma_handoff.v1"
            or type(metadata.get("frames")) is not int or metadata["frames"] != frame_count
            or isinstance(metadata.get("fps"), bool)
            or not np.isclose(metadata.get("fps", 0), fps, atol=1e-8, rtol=0)):
        raise ValueError("Native hand target schema/frame count/FPS differs")
    if metadata.get("hand_only") is not True or metadata.get("motion_accepted") is not False:
        raise ValueError("Expected explicitly unaccepted hand-only input")
    binding = metadata.get("binding", {})
    for key in ("native_sha256", "object_mesh_sha256", "hands_sha256"):
        _hash(binding.get(key), key)
    if ("expected_object_mesh_sha256" in config
            and binding["object_mesh_sha256"] != config["expected_object_mesh_sha256"]):
        raise ValueError("Native hand target original object mesh differs")
    declared_hands = metadata.get("hands")
    if not isinstance(declared_hands, dict) or not declared_hands or set(declared_hands) - {"left", "right"}:
        raise ValueError("Native hand target hands must be explicit left/right entries")
    enabled = config.get("enabled_hands", list(declared_hands))
    if set(enabled) - set(declared_hands):
        raise ValueError("Enabled hand has no native target")
    # Parse the already hashed bytes, avoiding a time-of-check/file-replacement
    # race and disabling pickled object arrays.
    import io
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {"timestamps", "object_R", "object_trans", "object_scale", "source_qpos_reference_only"}
    suffixes = {"wrist_position_world", "wrist_rotation_world", "wrist_position_object",
                "wrist_rotation_object", "finger_qpos", "finger_joint_names"}
    required |= {side+"_"+suffix for side in declared_hands for suffix in suffixes}
    if set(arrays) != required:
        raise ValueError("Unexpected or missing native hand target arrays")
    times = _finite(arrays["timestamps"], (frame_count,), "timestamps")
    if not np.allclose(times, np.arange(frame_count)/fps, atol=1e-7, rtol=0):
        raise ValueError("Native hand target timestamps differ from original source clock")
    source_qpos = arrays["source_qpos_reference_only"]
    if (source_qpos.ndim != 2 or len(source_qpos) != frame_count or source_qpos.shape[1] < 7
            or source_qpos.dtype.kind not in "fiu" or not np.isfinite(source_qpos).all()):
        raise ValueError("Invalid source_qpos_reference_only")
    original_r = _rotations(arrays["object_R"], (frame_count, 3, 3), "object_R")
    original_p = _finite(arrays["object_trans"], (frame_count, 3), "object_trans")
    scale = _finite(arrays["object_scale"], (3,), "object_scale")
    if (scale <= 0).any() or not np.array_equal(scale, binding.get("object_scale")):
        raise ValueError("Native hand target object scale binding differs")
    if object_scale is not None and not np.allclose(scale, np.broadcast_to(object_scale, (3,)), atol=1e-10, rtol=0):
        raise ValueError("Native hand target and solver object scales differ")
    transform = np.asarray(config["world_to_solver"])
    rotation, translation = transform[:3, :3], transform[:3, 3]
    solver_r = rotation @ original_r
    solver_p = original_p @ rotation.T + translation
    expected_r = _rotations(object_rotations, (frame_count, 3, 3), "solver object rotations")
    expected_p = _finite(object_positions, (frame_count, 3), "solver object positions")
    if (not np.allclose(solver_r, expected_r, atol=2e-6, rtol=0)
            or not np.allclose(solver_p, expected_p, atol=2e-6, rtol=0)):
        raise ValueError("Native hand target object trajectory/frame differs from solver reference")
    def model_names(values, label):
        values = list(values)
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError(f"{label} must contain named original model elements")
        suffixes = [name.rsplit("/", 1)[-1] for name in values]
        if len(set(suffixes)) != len(suffixes):
            raise ValueError(f"Ambiguous original model {label}")
        return set(suffixes)
    joints = model_names(available_joint_names, "joints")
    bodies = model_names(available_body_names, "bodies")
    hands = {}
    for side in declared_hands:
        item = declared_hands[side]
        names = item.get("finger_joints")
        expected_names = {f"{side}_hand_{digit}_{i}_joint"
                          for digit, count in (("thumb", 3), ("middle", 2), ("index", 2))
                          for i in range(count)}
        if (not isinstance(names, list) or len(names) != 7 or set(names) != expected_names
                or arrays[side+"_finger_joint_names"].shape != (7,)
                or arrays[side+"_finger_joint_names"].dtype.kind not in "US"
                or arrays[side+"_finger_joint_names"].tolist() != names):
            raise ValueError(f"{side} finger joint names/order differ")
        wrist = item.get("wrist_body")
        if wrist != f"{side}_wrist_yaw_link":
            raise ValueError(f"{side} wrist body differs from native Dex3 binding")
        if side in enabled and (set(names) - joints or wrist not in bodies):
            raise ValueError(f"{side} target joints/wrist are missing from solver model")
        phases = item.get("phase_labels")
        mask = np.asarray(item.get("contact_mask"))
        if (not isinstance(phases, list) or len(phases) != frame_count
                or any(not isinstance(x, str) for x in phases)
                or set(phases) - _CONTACT_PHASES - _OTHER_PHASES
                or mask.shape != (frame_count,) or mask.dtype.kind != "b"
                or not np.array_equal(mask, [p in _CONTACT_PHASES for p in phases])):
            raise ValueError(f"{side} source phases/contact mask differ")
        p = _finite(arrays[side+"_wrist_position_object"], (frame_count, 3), side+" wrist position")
        r = _rotations(arrays[side+"_wrist_rotation_object"], (frame_count, 3, 3), side+" wrist rotation")
        q = _finite(arrays[side+"_finger_qpos"], (frame_count, 7), side+" finger_qpos")
        world_p = _finite(arrays[side+"_wrist_position_world"], (frame_count, 3), side+" world position")
        world_r = _rotations(arrays[side+"_wrist_rotation_world"], (frame_count, 3, 3), side+" world rotation")
        if (not np.allclose(np.einsum("tij,tj->ti", original_r, p)+original_p, world_p, atol=2e-6, rtol=0)
                or not np.allclose(original_r @ r, world_r, atol=2e-6, rtol=0)):
            raise ValueError(f"{side} local/world wrist target consistency failed")
        if side in enabled:
            hands[side] = dict(wrist_body=wrist, finger_joints=tuple(names),
                wrist_position_object=_readonly(p), wrist_rotation_object=_readonly(r),
                finger_qpos=_readonly(q), contact_mask=_readonly(mask), phase_labels=tuple(phases),
                wrist_position_world=_readonly(world_p @ rotation.T+translation),
                wrist_rotation_world=_readonly(rotation @ world_r))
    report = dict(schema="soma_native_robot_hand_targets.v1", status="UNREVIEWED_SOFT_TARGETS",
        handoff_json=str(json_path), handoff_sha256=config["handoff_sha256"],
        targets_npz=str(npz_path), targets_sha256=config["targets_sha256"],
        source_native_sha256=binding["native_sha256"], object_mesh_sha256=binding["object_mesh_sha256"],
        source_hands_sha256=binding["hands_sha256"],
        input_files={"handoff_json":dict(path=str(json_path),sha256=config["handoff_sha256"]),
                     "targets_npz":dict(path=str(npz_path),sha256=config["targets_sha256"])},
        frames=frame_count, fps=float(fps), enabled_hands=list(hands),
        source_hand_weight_multiplier=config["source_hand_weight_multiplier"],
        object_relative_frames={side:int(h["contact_mask"].sum()) for side,h in hands.items()},
        residual_scales={"translation":math.sqrt(config["translation_weight"])/config["translation_sigma_m"],
                         "rotation":math.sqrt(config["rotation_weight"])/config["rotation_sigma_rad"],
                         "finger":math.sqrt(config["finger_weight"])/config["finger_sigma_rad"]},
        fixed_anchors_loaded=False, motion_accepted=False,
        object_local_translation_units="metres, already includes original object scale",
        target_rule="Object-relative on declared contact mask; fixed solver-world targets otherwise.")
    return dict(hands=hands, timestamps=_readonly(times), object_reference_positions=_readonly(expected_p),
        object_reference_rotations=_readonly(expected_r), object_scale=_readonly(scale), config=config, report=report)
