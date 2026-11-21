"""Authored native pose milestones with bounded local smooth correction seeds.

This edits an additive native-hinge correction, not a collision-free path. The
quintic correction is C2 before baking; exported LINEAR motion is not C2.
"""

import numpy as np


def apply(qpos, mask, lower, upper, timestamps, model_snapshot, keyframes):
    """Return seed, refinement mask, receipt; freeze authored key coordinates.

    Each key is ``{"frame": int, "window": [start, end],
    "joints": {native_joint_name: absolute_angle_radians}}``. ``model_snapshot``
    uses the native I/O joint binding. Only native hinges may change. Windows
    may touch but may not overlap on the same joint. Corridor failures are errors
    rather than clipped edits. Temporal/contact/whole-mesh gates remain required.
    """
    from .contact_edit import native_io
    native_joints = native_io._model_joints(model_snapshot)
    hinges = {j["joint"]: j for j in native_joints if j["joint_type"] == 3}
    joint_addresses = {name: int(j["address"]) for name, j in hinges.items()}
    q = np.asarray(qpos, dtype=float)
    editable = np.asarray(mask)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    if (q.ndim != 2 or len(q) < 3 or not np.isfinite(q).all()
            or editable.shape != q.shape or editable.dtype != bool
            or lo.shape != q.shape or hi.shape != q.shape
            or np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any()):
        raise ValueError("Invalid pose, editable mask, or corridor arrays")
    if ts.shape != (len(q),) or not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError("timestamps must be finite and strictly increasing")
    if not isinstance(keyframes, list) or not keyframes:
        raise ValueError("Declare at least one authored pose keyframe")
    out, refinement_mask = q.copy(), editable.copy()
    occupied, rows = {}, []
    for key in keyframes:
        if not isinstance(key, dict) or set(key) != {"frame", "window", "joints"}:
            raise ValueError("Pose keyframe requires frame, window, joints")
        f, window, joints = key["frame"], key["window"], key["joints"]
        if (type(f) is not int or not isinstance(window, list) or len(window) != 2
                or any(type(i) is not int for i in window)
                or not 0 <= window[0] < f < window[1] < len(q)):
            raise ValueError("Pose keyframe must be strictly inside its window")
        if not isinstance(joints, dict) or not joints:
            raise ValueError("Declare native hinge angles for each pose keyframe")
        start, end = window
        frames = np.arange(start + 1, end)
        u = np.where(frames <= f, (ts[frames]-ts[start])/(ts[f]-ts[start]),
                     (ts[end]-ts[frames])/(ts[end]-ts[f]))
        bump = u**3*(10+u*(-15+6*u))
        corrections = {}
        for name, value in joints.items():
            if name not in joint_addresses:
                raise ValueError("Pose keyframe joint is not a permitted native hinge: " + name)
            a = joint_addresses[name]
            if type(a) is not int or not 0 <= a < q.shape[1]:
                raise ValueError("Invalid native hinge address: " + name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("Pose key angle must be finite radians: " + name)
            native = hinges[name]
            if native["limited"] and not native["limits"][0] <= value <= native["limits"][1]:
                raise ValueError("Pose keyframe exceeds native hinge limits: " + name)
            if not editable[frames, a].all():
                raise ValueError("Pose keyframe window crosses protected samples: " + name)
            for old_start, old_end in occupied.get(a, []):
                if max(start, old_start) < min(end, old_end):
                    raise ValueError("Overlapping pose keyframe windows on joint: " + name)
            occupied.setdefault(a, []).append((start, end))
            delta = float(value)-q[f, a]
            out[frames, a] = q[frames, a]+bump*delta
            out[f, a] = float(value)
            refinement_mask[f, a] = False
            corrections[name] = delta
        rows.append(dict(frame=f, window=window, joints=dict(joints), correction_rad=corrections))
    if (out < lo).any() or (out > hi).any():
        raise ValueError("Authored pose keyframe/correction exceeds declared corridor")
    if not np.array_equal(out[~editable], q[~editable]):
        raise RuntimeError("Authored correction changed protected samples")
    return out, refinement_mask, dict(
        schema_version=1, mode="AUTHORED_NATIVE_POSE_KEYS", keyframes=rows,
        preserved_exact_key_coordinates=True, motion_accepted=False,
        correction_interpolation="C2 quintic additive correction before baking; LINEAR export is not C2",
        requires=["native limits", "temporal projection feasibility", "contact", "whole-mesh collision", "source review"],
    )
