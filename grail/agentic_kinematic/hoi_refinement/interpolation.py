"""Bounded local joint corrections on an immutable native source timeline."""

from __future__ import annotations

import numpy as np

from .native import validate_motion


def rotvec_to_quaternion(value):
    value = np.asarray(value, dtype=np.float64)
    angle = np.linalg.norm(value, axis=-1, keepdims=True)
    return np.concatenate((np.cos(angle / 2), 0.5 * np.sinc(angle / (2 * np.pi)) * value), axis=-1)


def quaternion_to_rotvec(value):
    value = np.asarray(value, dtype=np.float64)
    value = value / np.linalg.norm(value, axis=-1, keepdims=True)
    value = np.where(value[..., :1] < 0, -value, value)
    length = np.linalg.norm(value[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(length, value[..., :1])
    return value[..., 1:] * np.divide(
        angle, length, out=np.full_like(length, 2.0), where=length > 1e-12
    )


def quaternion_multiply(left, right):
    left, right = np.asarray(left), np.asarray(right)
    scalar = left[..., :1] * right[..., :1] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1, keepdims=True
    )
    vector = (
        left[..., :1] * right[..., 1:]
        + right[..., :1] * left[..., 1:]
        + np.cross(left[..., 1:], right[..., 1:])
    )
    return np.concatenate((scalar, vector), axis=-1)


def slerp(left, right, amount):
    """Shortest-arc unit quaternion SLERP, stable for nearly equal rotations."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    dot = np.sum(left * right, axis=-1, keepdims=True)
    right = np.where(dot < 0, -right, right)
    dot = np.clip(np.abs(dot), 0, 1)
    theta = np.arccos(dot)
    fraction = np.asarray(amount)[..., None]
    sine = np.sin(theta)
    safe = np.where(sine > 1e-8, sine, 1)
    curved = (np.sin((1 - fraction) * theta) * left + np.sin(fraction * theta) * right) / safe
    linear = (1 - fraction) * left + fraction * right
    result = np.where(dot > 0.9995, linear, curved)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def rotvec_to_matrix(value):
    value = np.asarray(value, dtype=np.float64)
    angle = np.linalg.norm(value, axis=-1)
    cross = np.zeros(value.shape[:-1] + (3, 3))
    cross[..., 0, 1], cross[..., 0, 2] = -value[..., 2], value[..., 1]
    cross[..., 1, 0], cross[..., 1, 2] = value[..., 2], -value[..., 0]
    cross[..., 2, 0], cross[..., 2, 1] = -value[..., 1], value[..., 0]
    return (
        np.eye(3)
        + np.sinc(angle / np.pi)[..., None, None] * cross
        + (0.5 * np.sinc(angle / (2 * np.pi)) ** 2)[..., None, None] * (cross @ cross)
    )


def matrix_to_quaternion(value):
    # A symmetric eigensystem avoids division by tiny traces at 180 degrees.
    matrices = np.asarray(value, dtype=np.float64)
    output = []
    for r in matrices.reshape(-1, 3, 3):
        matrix = np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[0, 1] + r[1, 0],
                    r[0, 2] + r[2, 0],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[0, 1] + r[1, 0],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[1, 2] + r[2, 1],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[0, 2] + r[2, 0],
                    r[1, 2] + r[2, 1],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        _, vectors = np.linalg.eigh(matrix)
        output.append(vectors[:, -1][[3, 0, 1, 2]])
    return np.asarray(output).reshape(matrices.shape[:-2] + (4,))


def validate_plan(motion, plan):
    """Validate source-frame windows and correction bounds; return the plan."""
    metadata = validate_motion(motion)
    if (
        not isinstance(plan, dict)
        or set(plan) != {"schema_version", "windows"}
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
        or not isinstance(plan["windows"], list)
    ):
        raise ValueError("Edit plan must have schema_version=1 and a windows list")
    intervals = {}
    for window in plan["windows"]:
        if set(window) != {"joint", "start", "end", "max_angle_rad", "keys"}:
            raise ValueError("Unexpected edit window fields")
        joint = window["joint"]
        if type(joint) is not int or joint not in metadata["editable_body_joints"]:
            raise ValueError(f"Joint {joint!r} is not an editable native body joint")
        for name in ("start", "end", "max_angle_rad"):
            if (
                isinstance(window[name], bool)
                or not isinstance(window[name], (int, float))
                or not np.isfinite(window[name])
            ):
                raise ValueError(f"Window {name} must be finite")
        start, end, limit = window["start"], window["end"], window["max_angle_rad"]
        # Export retains the integer source timeline. Fractional boundaries
        # would let replay interpolate an edited neighbour across a protected
        # endpoint, even though authored corrections are zero outside it.
        if not float(start).is_integer() or not float(end).is_integer():
            raise ValueError("Window start/end must be integer source frames")
        if not 0 <= start < end <= metadata["frame_count"] - 1:
            raise ValueError("Window must lie on the preserved source timeline")
        if not 0 < limit < np.pi:
            raise ValueError("max_angle_rad must be positive and below pi")
        for previous_start, previous_end in intervals.setdefault(joint, []):
            if max(start, previous_start) < min(end, previous_end):
                raise ValueError("Edit windows for a joint must not overlap")
        intervals[joint].append((start, end))
        keys = window["keys"]
        if not isinstance(keys, list) or len(keys) < 2:
            raise ValueError("Each window needs at least its two zero endpoint keys")
        previous = -np.inf
        for key in keys:
            if set(key) != {"frame", "delta_rotvec"}:
                raise ValueError("Unexpected edit key fields")
            frame = key["frame"]
            if (
                isinstance(frame, bool)
                or not isinstance(frame, (int, float))
                or not np.isfinite(frame)
                or not start <= frame <= end
                or frame <= previous
            ):
                raise ValueError("Key frames must be finite, strictly sorted and within the window")
            delta = np.asarray(key["delta_rotvec"])
            if (
                delta.shape != (3,)
                or delta.dtype.kind not in "fiu"
                or not np.isfinite(delta).all()
                or np.linalg.norm(delta) > limit + 1e-12
            ):
                raise ValueError("Correction must be a finite rotvec within max_angle_rad")
            previous = frame
        if (
            keys[0]["frame"] != start
            or keys[-1]["frame"] != end
            or np.any(np.asarray(keys[0]["delta_rotvec"]) != 0)
            or np.any(np.asarray(keys[-1]["delta_rotvec"]) != 0)
        ):
            raise ValueError("Window endpoints must have exactly zero correction keys")
        quaternions = rotvec_to_quaternion([key["delta_rotvec"] for key in keys])
        if np.any(np.sum(quaternions[:-1] * quaternions[1:], axis=-1) < 0):
            raise ValueError("Shortest correction arc crosses pi and exceeds max_angle_rad")
    return plan


def _apply_sampled_corrections(poses, plan, frames):
    result = poses.copy()
    joints = result.reshape(len(frames), -1, 3)
    for window in plan["windows"]:
        active = np.flatnonzero((frames > window["start"]) & (frames < window["end"]))
        if not len(active):
            continue
        times = np.asarray([key["frame"] for key in window["keys"]])
        values = np.asarray([key["delta_rotvec"] for key in window["keys"]])
        right = np.searchsorted(times, frames[active], side="right")
        left = right - 1
        fraction = (frames[active] - times[left]) / (times[right] - times[left])
        delta = slerp(
            rotvec_to_quaternion(values[left]), rotvec_to_quaternion(values[right]), fraction
        )
        nonzero = np.linalg.norm(delta[:, 1:], axis=-1) > 0
        selected = active[nonzero]
        if len(selected):
            joint = window["joint"]
            joints[selected, joint] = quaternion_to_rotvec(
                quaternion_multiply(rotvec_to_quaternion(joints[selected, joint]), delta[nonzero])
            )
    return result


def apply_edits(motion, plan):
    """Compose original local rotations with bounded corrections, Rparent @ Rdelta."""
    validate_plan(motion, plan)
    result = {name: np.asarray(value).copy() for name, value in motion.items()}
    result["poses"] = _apply_sampled_corrections(
        result["poses"], plan, np.arange(len(result["poses"]), dtype=np.float64)
    )
    return result


def sample_motion(motion, plan, frames):
    """Evaluate source-frame/subframe native controls and continuous corrections.

    Parent rotations use shortest-arc SLERP; translations are linear. Integer
    source frames are copied exactly. Fractional samples across changing identity
    or native scales are rejected instead of inventing an averaged human shape.
    """
    validate_plan(motion, plan)
    count = len(motion["poses"])
    frames = np.asarray(frames, dtype=np.float64)
    if (
        frames.ndim != 1
        or not len(frames)
        or not np.isfinite(frames).all()
        or np.any(frames < 0)
        or np.any(frames > count - 1)
    ):
        raise ValueError("Sample frames must be nonempty finite source coordinates within [0,T-1]")
    left, right = np.floor(frames).astype(int), np.ceil(frames).astype(int)
    amount = frames - left
    result = {name: np.asarray(value).copy() for name, value in motion.items()}
    array = np.asarray(motion["poses"])
    shape = (len(frames),) + array.shape[1:]
    source = array.reshape(count, -1, 3)
    quaternion = slerp(
        rotvec_to_quaternion(source[left]), rotvec_to_quaternion(source[right]), amount[:, None]
    )
    sampled = quaternion_to_rotvec(quaternion).astype(array.dtype).reshape(shape)
    sampled[amount == 0] = array[left[amount == 0]]
    result["poses"] = sampled
    for name in ("trans", "obj_t"):
        if name not in motion:
            continue
        array = np.asarray(motion[name])
        result[name] = (
            (1 - amount[:, None]) * array[left] + amount[:, None] * array[right]
        ).astype(array.dtype)
        result[name][amount == 0] = array[left[amount == 0]]
    rotations = np.asarray(motion["obj_R"])
    quaternion = slerp(
        matrix_to_quaternion(rotations[left]), matrix_to_quaternion(rotations[right]), amount
    )
    result["obj_R"] = rotvec_to_matrix(quaternion_to_rotvec(quaternion)).astype(rotations.dtype)
    result["obj_R"][amount == 0] = rotations[left[amount == 0]]
    for name in ("identity_coeffs", "scale_params"):
        if name not in motion:
            continue
        array = np.asarray(motion[name])
        if array.ndim == 2 and array.shape[0] == count and count > 1:
            if np.any(array[left] != array[right]):
                raise ValueError(
                    f"Fractional sampling across time-varying {name} is undefined; use native integer frames"
                )
            result[name] = array[left].copy()
    result["poses"] = _apply_sampled_corrections(result["poses"], plan, frames)
    return result
