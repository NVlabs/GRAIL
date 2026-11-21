"""Source-bound, native-rate regression checks for one Stage 2 iteration.

The original ``base`` defines the immutable fields and rotation corridor. The
reviewed ``seed`` defines the temporal baseline only; it never replaces that
source prior. Geometry is independently evaluated world-space native output,
with explicit source-frame coordinates. Anatomical joint IDs must be supplied
by the caller for the selected native model; root translation is not a wrist.

PASS means only that these regression gates passed. Contact, source evidence,
collision and visual review remain separate requirements.
"""

from __future__ import annotations

import numpy as np

from .interpolation import (
    quaternion_multiply,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
    validate_plan,
)
from .native import DERIVED_FIELDS, validate_motion

METRIC_ORDERS = {
    "joint_rotation_step_rad": 0,
    "joint_rotation_speed_rad_s": 1,
    "joint_rotation_acceleration_rad_s2": 2,
    "joint_rotation_jerk_rad_s3": 3,
    "joint_boundary_velocity_jump_rad_s": 1,
    "wrist_speed_m_s": 1,
    "wrist_acceleration_m_s2": 2,
    "wrist_jerk_m_s3": 3,
    "wrist_boundary_velocity_jump_m_s": 1,
    "foot_displacement_m": 0,
    "foot_sliding_speed_m_s": 1,
}


def _settings(settings, joint_count):
    settings = {} if settings is None else settings
    allowed = {
        "preserve_smoothness",
        "relative_tolerance",
        "absolute_tolerance",
        "wrist_joint_ids",
        "foot_joint_ids",
    }
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError("Unknown iteration settings")
    result = dict(settings)
    if type(result.setdefault("preserve_smoothness", True)) is not bool:
        raise ValueError("preserve_smoothness must be a boolean")
    for name in ("relative_tolerance", "absolute_tolerance"):
        values = result.setdefault(name, {})
        if not isinstance(values, dict) or set(values) - METRIC_ORDERS.keys():
            raise ValueError(f"{name} must be a dictionary of known metric names")
        for metric, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name}.{metric} must be finite and nonnegative")
    for name in ("wrist_joint_ids", "foot_joint_ids"):
        ids = result.setdefault(name, [])
        if (
            not isinstance(ids, list)
            or any(type(index) is not int or not 0 <= index < joint_count for index in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"{name} must contain unique valid native joint indices")
    return result


def _relative_rotations(left, right):
    """Shortest SO(3) increments in the common parent-joint coordinate basis."""
    left = rotvec_to_quaternion(left)
    right = rotvec_to_quaternion(right)
    conjugate = left.copy()
    conjugate[..., 1:] *= -1
    result = quaternion_to_rotvec(quaternion_multiply(right, conjugate))
    if not np.isfinite(result).all():
        raise ValueError("Native rotations exceed the numerical range of SO(3) review")
    return result


def _precision(*arrays):
    return max(
        (
            np.finfo(np.asarray(array).dtype).eps
            if np.asarray(array).dtype.kind == "f"
            else np.finfo(np.float64).eps
        )
        for array in arrays
    )


def _source_contract(base, motion, plan, label):
    """Check original frozen controls and per-joint, source-relative corridors."""
    metadata = validate_motion(motion)
    original = {
        name: np.asarray(value) for name, value in base.items() if name not in DERIVED_FIELDS
    }
    current = {
        name: np.asarray(value) for name, value in motion.items() if name not in DERIVED_FIELDS
    }
    if original.keys() != current.keys():
        raise ValueError(f"{label} has different canonical fields from the original source")
    for name, value in original.items():
        other = current[name]
        if value.dtype != other.dtype or value.shape != other.shape:
            raise ValueError(f"{label} changed source field shape/dtype: {name}")
        if name != "poses" and value.tobytes() != other.tobytes():
            raise ValueError(f"{label} changed frozen source field: {name}")
    poses = original["poses"].reshape(metadata["frame_count"], -1, 3)
    edited = current["poses"].reshape(poses.shape)
    editable = np.zeros(poses.shape[:2], dtype=bool)
    for window in plan["windows"]:
        selection = slice(int(window["start"]) + 1, int(window["end"]))
        joint = window["joint"]
        editable[selection, joint] = True
        angles = np.linalg.norm(
            _relative_rotations(poses[selection, joint], edited[selection, joint]), axis=-1
        )
        epsilon = 32 * _precision(poses, edited)
        if np.any(angles > window["max_angle_rad"] + epsilon):
            raise ValueError(f"{label} exceeds the original-source rotation corridor")
    if poses[~editable].tobytes() != edited[~editable].tobytes():
        raise ValueError(f"{label} changed frozen source rotations outside editable windows")
    angles = np.linalg.norm(_relative_rotations(poses, edited), axis=-1)
    return {
        "reference": "original_base",
        "max_joint_rotation_difference_rad": float(angles.max(initial=0)),
        "frozen_fields_equal": True,
        "within_declared_corridors": True,
    }


def _geometry(data, name, count):
    if data is None:
        return None, f"{name} native geometry was not supplied"
    if not isinstance(data, dict):
        raise ValueError(f"{name} geometry must be a mapping")
    missing = {"frames", "joints", "vertices"} - data.keys()
    if missing:
        return None, f"{name} geometry lacks {', '.join(sorted(missing))}"
    arrays = {}
    for field in ("frames", "joints", "vertices"):
        value = np.asarray(data[field])
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{name}.{field} must be finite numeric data")
        arrays[field] = value
    frames = arrays["frames"]
    if frames.ndim != 1 or np.any(np.diff(frames) <= 0):
        raise ValueError(f"{name}.frames must be a strictly increasing vector")
    if np.any(frames < 0) or np.any(frames > count - 1):
        raise ValueError(f"{name}.frames lie outside the source timeline")
    for field in ("joints", "vertices"):
        value = arrays[field]
        if value.ndim != 3 or value.shape[0] != len(frames) or value.shape[-1] != 3:
            raise ValueError(f"{name}.{field} must have shape [len(frames), N, 3]")
        if not value.shape[1]:
            raise ValueError(f"{name}.{field} must contain native geometry")
    return arrays, None


def _window_frames(window, count):
    return np.arange(max(0, int(window["start"]) - 3), min(count, int(window["end"]) + 4))


def _stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(values).all():
        raise ValueError("Native derivatives exceed the numerical range of iteration review")
    return {
        "count": int(values.size),
        "max": float(values.max()) if values.size else None,
        "mean": float(np.sum(values / values.size)) if values.size else None,
    }


def _rotation_series(poses, frames, joint, fps, joins):
    selected = poses[frames, joint]
    step = _relative_rotations(selected[:-1], selected[1:])
    velocity = step * fps
    result = {
        "joint_rotation_step_rad": np.linalg.norm(step, axis=-1),
        "joint_rotation_speed_rad_s": np.linalg.norm(velocity, axis=-1),
        "joint_rotation_acceleration_rad_s2": np.linalg.norm(
            np.diff(velocity, axis=0) * fps, axis=-1
        ),
        "joint_rotation_jerk_rad_s3": np.linalg.norm(
            np.diff(velocity, n=2, axis=0) * fps**2, axis=-1
        ),
    }
    if joins:
        offsets = np.asarray(joins) - frames[0]
        result["joint_boundary_velocity_jump_rad_s"] = np.linalg.norm(
            velocity[offsets] - velocity[offsets - 1], axis=-1
        )
    return result


def _position_series(positions, fps, joins, first_frame):
    positions = np.asarray(positions, dtype=np.float64)
    velocity = np.diff(positions, axis=0) * fps
    result = {
        "wrist_speed_m_s": np.linalg.norm(velocity, axis=-1),
        "wrist_acceleration_m_s2": np.linalg.norm(np.diff(velocity, axis=0) * fps, axis=-1),
        "wrist_jerk_m_s3": np.linalg.norm(np.diff(velocity, n=2, axis=0) * fps**2, axis=-1),
    }
    if joins:
        offsets = np.asarray(joins) - first_frame
        result["wrist_boundary_velocity_jump_m_s"] = np.linalg.norm(
            velocity[offsets] - velocity[offsets - 1], axis=-1
        )
    return result


def _comparison(name, before, after, settings, epsilon):
    before, after = _stats(before), _stats(after)
    if before["max"] is None or after["max"] is None:
        return {
            "seed": before,
            "candidate": after,
            "passed": None,
            "reason": "Insufficient samples",
        }
    limit = (
        before["max"] * (1 + settings["relative_tolerance"].get(name, 0))
        + settings["absolute_tolerance"].get(name, 0)
        + epsilon
    )
    if not np.isfinite(limit):
        raise ValueError(f"Tolerance overflows the {name} gate")
    return {
        "seed": before,
        "candidate": after,
        "allowed_max": float(limit),
        "numerical_epsilon": float(epsilon),
        "passed": bool(after["max"] <= limit),
    }


def _joint_comparison(name, before, after, settings, epsilon, joint_ids):
    """A busy wrist/foot cannot hide a regression on another declared joint."""
    result = _comparison(name, before, after, settings, epsilon)
    if before.ndim != 2:
        return result
    details = [
        {
            "joint_id": joint,
            **_comparison(name, before[:, offset], after[:, offset], settings, epsilon),
        }
        for offset, joint in enumerate(joint_ids)
    ]
    result["per_joint"] = details
    if any(item["passed"] is False for item in details):
        result["passed"] = False
    elif any(item["passed"] is None for item in details):
        result["passed"] = None
    return result


def review_iteration(base, seed, candidate, seed_geometry, candidate_geometry, plan, settings=None):
    """Return JSON-safe PASS/FAIL/NOT_PERFORMED regression evidence.

    ``settings`` accepts ``preserve_smoothness`` (default true), explicit
    ``wrist_joint_ids``/``foot_joint_ids``, and nonnegative metric-name maps
    ``relative_tolerance``/``absolute_tolerance``. Gates compare each candidate
    per-joint maximum to ``seed_max * (1 + relative) + absolute + numerical_epsilon``.
    Defaults add numerical precision only. Foot displacement has a zero seed
    baseline, so it requires an explicit absolute tolerance to allow movement.

    Every declared window, including unchanged ones, is checked with three
    source-frame neighbors on either side. All required integer native geometry
    samples must exist; fractional samples cannot replace them. Native controls
    provide SO(3) differences, and evaluated native wrist positions provide
    linear derivatives. Malformed data, settings or source-contract violations
    raise ValueError. Missing evidence gives NOT_PERFORMED with passed=None;
    measured failures retain FAIL even when some checks could not run.
    """
    metadata = validate_motion(base)
    validate_plan(base, plan)
    settings = _settings(settings, metadata["joint_count"])
    source = {
        "seed": _source_contract(base, seed, plan, "seed"),
        "candidate": _source_contract(base, candidate, plan, "candidate"),
    }
    count, fps = metadata["frame_count"], metadata["fps"]
    before, before_missing = _geometry(seed_geometry, "seed", count)
    after, after_missing = _geometry(candidate_geometry, "candidate", count)
    unavailable = [reason for reason in (before_missing, after_missing) if reason]
    if before is not None and after is not None:
        for field in ("joints", "vertices"):
            if before[field].shape[1:] != after[field].shape[1:]:
                raise ValueError(f"Candidate changed native {field} count or shape")
        for name in ("wrist_joint_ids", "foot_joint_ids"):
            if any(index >= before["joints"].shape[1] for index in settings[name]):
                raise ValueError(f"{name} is outside the evaluated native joints")
        if "faces" in seed_geometry or "faces" in candidate_geometry:
            if not (
                "faces" in seed_geometry
                and "faces" in candidate_geometry
                and np.array_equal(seed_geometry["faces"], candidate_geometry["faces"])
            ):
                raise ValueError("Candidate changed native geometry topology")
    if not settings["wrist_joint_ids"]:
        unavailable.append("Explicit native wrist_joint_ids are required for wrist review")
    if not plan["windows"]:
        unavailable.append("No declared edit windows to review")
    if not settings["preserve_smoothness"]:
        unavailable.append("Smoothness protection was explicitly disabled")
    seed_poses = np.asarray(seed["poses"]).reshape(count, -1, 3)
    candidate_poses = np.asarray(candidate["poses"]).reshape(count, -1, 3)
    angular_precision = 32 * _precision(seed_poses, candidate_poses)
    windows, failures = [], []
    for index, window in enumerate(plan["windows"]):
        frames = _window_frames(window, count)
        joins = [int(frame) for frame in (window["start"], window["end"]) if 0 < frame < count - 1]
        joint = window["joint"]
        record = {
            "window": index,
            "joint": joint,
            "start": window["start"],
            "end": window["end"],
            "evaluated_frames": frames.tolist(),
            "boundary_frames": joins,
            "changed_from_seed": not np.array_equal(
                seed_poses[frames, joint], candidate_poses[frames, joint]
            ),
            "metrics": {},
        }
        values_before = _rotation_series(seed_poses, frames, joint, fps, joins)
        values_after = _rotation_series(candidate_poses, frames, joint, fps, joins)
        precision = {name: angular_precision for name in values_before}
        missing = {}
        for name, geometry in (("seed", before), ("candidate", after)):
            if geometry is None:
                missing[name] = frames.tolist()
            else:
                absent = frames[~np.isin(frames, geometry["frames"])]
                if len(absent):
                    missing[name] = absent.tolist()
        if missing:
            record["geometry"] = {"status": "NOT_PERFORMED", "missing_integer_frames": missing}
            unavailable.append(f"Window {index} lacks required native integer geometry samples")
        else:
            left = before["joints"][np.searchsorted(before["frames"], frames)]
            right = after["joints"][np.searchsorted(after["frames"], frames)]
            declared_ids = settings["wrist_joint_ids"] + settings["foot_joint_ids"]
            selected = declared_ids or [joint]
            if any(index >= left.shape[1] for index in selected):
                selected = list(range(left.shape[1]))
            position_precision = (
                32
                * _precision(left, right)
                * max(
                    1.0,
                    float(np.max(np.abs(left[:, selected]))),
                    float(np.max(np.abs(right[:, selected]))),
                )
            )
            left, right = left.astype(np.float64), right.astype(np.float64)
            if settings["wrist_joint_ids"]:
                wrist_ids = settings["wrist_joint_ids"]
                values_before.update(_position_series(left[:, wrist_ids], fps, joins, frames[0]))
                values_after.update(_position_series(right[:, wrist_ids], fps, joins, frames[0]))
            if settings["foot_joint_ids"]:
                foot_ids = settings["foot_joint_ids"]
                displacement = np.linalg.norm(right[:, foot_ids] - left[:, foot_ids], axis=-1)
                values_before["foot_displacement_m"] = np.zeros_like(displacement)
                values_after["foot_displacement_m"] = displacement
                values_before["foot_sliding_speed_m_s"] = np.linalg.norm(
                    np.diff(left[:, foot_ids], axis=0) * fps, axis=-1
                )
                values_after["foot_sliding_speed_m_s"] = np.linalg.norm(
                    np.diff(right[:, foot_ids], axis=0) * fps, axis=-1
                )
            precision.update(
                {name: position_precision for name in values_before if name not in precision}
            )
            record["geometry"] = {"status": "PERFORMED"}
        for name, value in values_before.items():
            ids = settings["foot_joint_ids" if name.startswith("foot_") else "wrist_joint_ids"]
            comparison = _joint_comparison(
                name,
                value,
                values_after[name],
                settings,
                precision[name] * fps ** METRIC_ORDERS[name],
                ids,
            )
            comparison["enforced"] = (
                settings["preserve_smoothness"] or name == "foot_displacement_m"
            )
            record["metrics"][name] = comparison
            if comparison["passed"] is None:
                unavailable.append(f"Window {index}: {name} has insufficient source samples")
            elif not comparison["passed"] and comparison["enforced"]:
                failures.append(
                    {
                        "window": index,
                        "metric": name,
                        "candidate_max": comparison["candidate"]["max"],
                        "allowed_max": comparison["allowed_max"],
                        "failing_joint_ids": [
                            item["joint_id"]
                            for item in comparison.get("per_joint", [])
                            if item["passed"] is False
                        ],
                    }
                )
        windows.append(record)
    status = "FAIL" if failures else "NOT_PERFORMED" if unavailable else "PASS"
    return {
        "status": status,
        "passed": False if failures else None if unavailable else True,
        "failures": failures,
        "metrics": {
            "source_fps": fps,
            "source_frame_count": count,
            "source_prior": source,
            "wrist_joint_ids": list(settings["wrist_joint_ids"]),
            "foot_joint_ids": list(settings["foot_joint_ids"]),
            "windows": windows,
        },
        "limitations": list(dict.fromkeys(unavailable))
        + [
            "This gate does not accept contact, source alignment, collision or visual quality.",
            "Derivatives use source-rate samples and shortest SO(3) increments; no continuous-time or dynamics guarantee.",
            "Sequence endpoints have no exterior frozen neighbor for a boundary velocity jump.",
            *(
                [
                    "Foot sliding is a 3D native joint-speed proxy; no ground/contact constraint is inferred."
                ]
                if settings["foot_joint_ids"]
                else [
                    "Foot preservation was not assessed because no native foot_joint_ids were declared."
                ]
            ),
        ],
    }
