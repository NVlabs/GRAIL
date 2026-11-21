"""Hard, source-reviewed static object intervals for native HOI fitting.

An interval fixes both rotation and translation to a specified raw pose. It is
not inferred from absent hand contact: objects can move for other reasons.
"""

from __future__ import annotations


def validate_stationary_intervals(intervals, frame_count=None):
    if not isinstance(intervals, list):
        raise ValueError("object_stationary_intervals must be a list")
    if frame_count is not None and (
        isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1
    ):
        raise ValueError("frame_count must be a positive integer")
    fields = {"start_frame", "end_frame", "anchor_frame", "evidence_frames", "evidence"}
    previous_end = -1
    result = []
    for interval in intervals:
        if not isinstance(interval, dict) or set(interval) != fields:
            raise ValueError("Static interval requires exactly start/end/anchor/evidence fields")
        for key in ("start_frame", "end_frame", "anchor_frame"):
            value = interval[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Static interval {key} must be a nonnegative integer")
        start, end, anchor = (interval[k] for k in ("start_frame", "end_frame", "anchor_frame"))
        if not previous_end < start <= anchor <= end:
            raise ValueError("Static intervals must be ordered, disjoint and contain their anchor")
        if frame_count is not None and end >= frame_count:
            raise ValueError("Static interval exceeds source frame count")
        evidence = interval["evidence"]
        frames = interval["evidence_frames"]
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Static interval requires source evidence")
        if (
            not isinstance(frames, list)
            or not frames
            or any(
                isinstance(f, bool) or not isinstance(f, int) or not start <= f <= end
                for f in frames
            )
        ):
            raise ValueError("Static evidence frames must lie within their interval")
        if len(set(frames)) != len(frames):
            raise ValueError("Static evidence frames must be unique")
        result.append({**interval, "evidence_frames": list(frames)})
        previous_end = end
    return result


def build_stationary_pose_reference(poses, intervals):
    import torch

    if not torch.is_tensor(poses) or poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Raw object poses must have shape [T,4,4]")
    if not poses.is_floating_point() or not torch.isfinite(poses).all():
        raise ValueError("Raw object poses must be finite floating point")
    intervals = validate_stationary_intervals(intervals, frame_count=len(poses))
    mask = torch.zeros(len(poses), dtype=torch.bool, device=poses.device)
    rotation = poses[:, :3, :3].detach().clone()
    translation = poses[:, :3, 3].detach().clone()
    for interval in intervals:
        start, end, anchor = (interval[k] for k in ("start_frame", "end_frame", "anchor_frame"))
        mask[start : end + 1] = True
        rotation[start : end + 1] = poses[anchor, :3, :3].detach()
        translation[start : end + 1] = poses[anchor, :3, 3].detach()
    return mask, rotation, translation


def apply_stationary_pose(rotation, translation, reference):
    """Remove static-frame pose freedom before any loss or mesh evaluation."""
    import torch

    if reference is None:
        return rotation, translation
    mask, fixed_rotation, fixed_translation = reference
    return (
        torch.where(mask[:, None, None], fixed_rotation, rotation),
        torch.where(mask[:, None], fixed_translation, translation),
    )


def audit_stationary_pose(initial_object, saved_object, intervals):
    """Check serialized native R/t against raw anchors; no residual-only claims."""
    import numpy as np

    raw_r = np.asarray(initial_object["obj_R"])
    raw_t = np.asarray(initial_object["obj_t"]).reshape(-1, 3)
    rot = np.asarray(saved_object["obj_R"])
    trans = np.asarray(saved_object["obj_t"]).reshape(-1, 3)
    if rot.shape != raw_r.shape or trans.shape != raw_t.shape:
        raise ValueError("Static audit requires matching native frame counts")
    intervals = validate_stationary_intervals(intervals, frame_count=len(raw_r))
    checks = []
    for interval in intervals:
        start, end, anchor = (interval[k] for k in ("start_frame", "end_frame", "anchor_frame"))
        expected_r = np.broadcast_to(raw_r[anchor], (end - start + 1, 3, 3))
        expected_t = np.broadcast_to(raw_t[anchor], (end - start + 1, 3))
        checks.append(
            {
                **interval,
                "rotation_exact": bool(np.array_equal(rot[start : end + 1], expected_r)),
                "translation_exact": bool(np.array_equal(trans[start : end + 1], expected_t)),
                "translation_max_error_m": float(
                    np.linalg.norm(trans[start : end + 1] - expected_t, axis=-1).max()
                ),
            }
        )
    return {
        "all_exact": all(c["rotation_exact"] and c["translation_exact"] for c in checks),
        "intervals": checks,
    }
