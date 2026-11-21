"""Optional, deterministic smoothing of SOMA's calibrated finger priors.

The input is the official pipeline's ``input_targets`` array: world xyz + xyzw,
ordered by ``mapped_joints``. Only finger rows in real source frames change.
Every frame uses its ORIGINAL wrist pose, so moving the wrist is not confused
with finger articulation. This creates a derivative source *prior*, not new
contacts and not a filter of the solved robot motion.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Real
import re

import numpy as np
from scipy.spatial.transform import Rotation


_FINGER = re.compile(r"^(Left|Right)Hand(?:Thumb|Index|Middle|Ring|Pinky)(?:[1-9]|End)$")
_DEFAULTS = {
    "enabled": False,
    "position_sigma_frames": 0.,
    "rotation_sigma_frames": 0.,
    "truncate": 3.,
    "position_weight_multiplier": 1.,
    "rotation_weight_multiplier": 1.,
}


def validate_hand_target_config(value=None):
    """Return explicit validated settings; absence is an exact no-op."""
    if value is None:
        return dict(_DEFAULTS)
    if not isinstance(value, dict) or set(value) - _DEFAULTS.keys():
        raise ValueError("Unknown hand target configuration field")
    config = dict(_DEFAULTS, **value)
    if not isinstance(config["enabled"], bool):
        raise ValueError("hand target enabled must be boolean")
    for key in config.keys() - {"enabled"}:
        number = config[key]
        if (isinstance(number, bool) or not isinstance(number, Real)
                or not math.isfinite(number) or number < 0):
            raise ValueError(f"hand target {key} must be finite and nonnegative")
        config[key] = float(number)
    if config["truncate"] <= 0:
        raise ValueError("hand target truncate must be positive")
    return config


def _names(names):
    names = list(names)
    if not names or len(set(names)) != len(names) or not all(isinstance(x, str) for x in names):
        raise ValueError("mapped joint names must be unique nonempty strings")
    return names


def hand_target_weight_multipliers(names, config=None):
    """Multipliers for the existing mapping's residual weights, in row order.

    SOMA passes mapping weights directly to Newton residual objectives. These
    are therefore residual multipliers, whose squares scale loss energies.
    Wrist and body entries remain exactly one. A zero original weight stays
    zero; this does not add any missing target objective.
    """
    names, config = _names(names), validate_hand_target_config(config)
    position, rotation = np.ones(len(names)), np.ones(len(names))
    if config["enabled"]:
        indices = [i for i, name in enumerate(names) if _FINGER.fullmatch(name)]
        position[indices] = config["position_weight_multiplier"]
        rotation[indices] = config["rotation_weight_multiplier"]
    return position, rotation


def _window(frame_count, sigma, truncate):
    """Symmetric Gaussian windows with explicit nearest endpoint padding."""
    if sigma == 0:
        return [(np.asarray([i]), np.ones(1)) for i in range(frame_count)]
    radius = max(1, int(math.ceil(sigma * truncate)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-.5 * np.square(offsets / sigma))
    weights /= weights.sum()
    return [(np.clip(i + offsets, 0, frame_count - 1), weights) for i in range(frame_count)]


def _smooth_rotations(rotations, windows):
    """Gaussian intrinsic SO(3) means; quaternion signs never affect the fit."""
    smoothed = []
    maximum_iterations = 32
    unconverged = []
    for frame, (indices, weights) in enumerate(windows):
        current = rotations[frame]
        for _ in range(maximum_iterations):
            tangent = (current.inv() * rotations[indices]).as_rotvec()
            step = np.sum(tangent * weights[:, None], axis=0)
            if np.linalg.norm(step) < 1e-10:
                break
            current = current * Rotation.from_rotvec(step)
        else:
            unconverged.append(frame)
        smoothed.append(current.as_quat())
    if unconverged:
        raise ValueError(f"SO(3) hand target smoothing did not converge at frames {unconverged}")
    return Rotation.from_quat(smoothed)


def _hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _derivatives(position, rotation, fps):
    velocity = np.diff(position, axis=0) * fps
    acceleration = np.diff(position, n=2, axis=0) * fps**2
    angular_velocity = (rotation[:-1].inv() * rotation[1:]).as_rotvec() * fps if len(position) > 1 else np.empty((0, 3))
    # All rotations here are relative to the unchanged wrist frame. These
    # finite increment statistics are diagnostics, not a robot acceptance test.
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if x.size else None
    return {"position_velocity_rms_m_s": rms(velocity),
            "position_acceleration_rms_m_s2": rms(acceleration),
            "rotation_increment_rms_rad_s": rms(angular_velocity)}


def smooth_hand_targets(targets, names, fps, config=None, *, first_source_frame=0):
    """Return a new target array and provenance report without input mutation.

    Warmup rows before ``first_source_frame`` stay byte-identical. The Gaussian
    averages only actual source frames, never synthetic initialization poses.
    XYZ is smoothed in each original wrist frame. Relative finger SO(3) poses
    use intrinsic rotation means and are composed with that same wrist.

    This intentionally relaxes noisy calibrated fingertip/link priors; smoothed
    targets need not correspond to one exactly reachable anatomical hand pose.
    Contact targets, source BVH, source body/wrists, scale and timing are not
    inputs to edit here. Their invariants must also be checked at integration.
    """
    config = validate_hand_target_config(config)
    names = _names(names)
    targets = np.asarray(targets)
    if (targets.ndim != 3 or targets.shape[1:] != (len(names), 7) or len(targets) < 1
            or not np.issubdtype(targets.dtype, np.floating) or not np.isfinite(targets).all()):
        raise ValueError("hand targets must be finite floating [frames, mapped joints, 7] xyz+xyzw")
    if (isinstance(fps, bool) or not isinstance(fps, Real)
            or not math.isfinite(fps) or fps <= 0):
        raise ValueError("hand target fps must be finite and positive")
    if (isinstance(first_source_frame, bool) or not isinstance(first_source_frame, (int, np.integer))
            or not 0 <= first_source_frame < len(targets)):
        raise ValueError("first_source_frame must index actual source targets")
    norms = np.linalg.norm(targets[:, :, 3:], axis=-1)
    if not np.allclose(norms, 1., atol=1e-4, rtol=0.):
        raise ValueError("hand target quaternions must be unit xyzw")
    result = targets.copy()
    finger_indices = [i for i, name in enumerate(names) if _FINGER.fullmatch(name)]
    report = {"schema": "soma_hand_target_derivative.v1", "config": config,
        "enabled": config["enabled"], "fps": float(fps), "first_source_frame": int(first_source_frame),
        "source_frame_count": len(targets) - first_source_frame,
        "original_target_sha256": _hash(targets), "finger_target_count": len(finger_indices),
        "finger_names": [names[i] for i in finger_indices], "per_target": [],
        "method": "Symmetric Gaussian wrist-local xyz and intrinsic SO(3) mean; nearest endpoint padding",
        "contact_targets_modified": False,
        "limitations": ["Derivative source priors, not anatomical reconstruction or measured contacts.",
            "Independent target smoothing does not enforce hand bone lengths or robot reachability.",
            "Offline symmetric smoothing uses future source frames.",
            "Soft reference objectives and reduced target jitter do not establish motion quality."]}
    first = int(first_source_frame)
    source = targets[first:]
    if config["enabled"]:
        for index in finger_indices:
            side = _FINGER.fullmatch(names[index]).group(1)
            wrist_name = side + "Hand"
            if wrist_name not in names:
                raise ValueError(f"Finger target {names[index]} lacks mapped wrist {wrist_name}")
            wrist = names.index(wrist_name)
            wrist_rotation = Rotation.from_quat(source[:, wrist, 3:])
            original_rotation = Rotation.from_quat(source[:, index, 3:])
            local_rotation = wrist_rotation.inv() * original_rotation
            local_position = wrist_rotation.inv().apply(source[:, index, :3] - source[:, wrist, :3])
            derived_position = local_position.copy()
            derived_rotation = local_rotation
            if config["position_sigma_frames"] > 0:
                windows = _window(len(source), config["position_sigma_frames"], config["truncate"])
                derived_position = np.asarray([np.sum(local_position[indices] * weights[:, None], axis=0)
                                               for indices, weights in windows])
                result[first:, index, :3] = wrist_rotation.apply(derived_position) + source[:, wrist, :3]
            if config["rotation_sigma_frames"] > 0:
                windows = _window(len(source), config["rotation_sigma_frames"], config["truncate"])
                derived_rotation = _smooth_rotations(local_rotation, windows)
                quaternions = (wrist_rotation * derived_rotation).as_quat()
                # Retain original quaternion hemisphere for numerical export
                # continuity without treating sign flips as physical rotations.
                sign = np.sum(quaternions * source[:, index, 3:], axis=1) < 0
                quaternions[sign] *= -1
                result[first:, index, 3:] = quaternions
            position_deviation = np.linalg.norm(result[first:, index, :3] - source[:, index, :3], axis=1)
            rotation_deviation = (original_rotation.inv() * Rotation.from_quat(result[first:, index, 3:])).magnitude()
            report["per_target"].append({"name": names[index], "wrist": wrist_name,
                "maximum_position_deviation_m": float(position_deviation.max()),
                "rms_position_deviation_m": float(np.sqrt(np.mean(position_deviation**2))),
                "maximum_rotation_deviation_deg": float(np.degrees(rotation_deviation.max())),
                "rms_rotation_deviation_deg": float(np.degrees(np.sqrt(np.mean(rotation_deviation**2)))),
                "wrist_local_before": _derivatives(local_position, local_rotation, float(fps)),
                "wrist_local_after": _derivatives(derived_position, derived_rotation, float(fps))})
    protected = [i for i in range(len(names)) if i not in finger_indices]
    report.update(derived_target_sha256=_hash(result),
        actual_targets_changed=not np.array_equal(result, targets),
        protected_body_wrist_targets_unchanged=np.array_equal(result[:, protected], targets[:, protected]),
        warmup_targets_unchanged=np.array_equal(result[:first], targets[:first]))
    return result, report
