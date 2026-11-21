"""Conservative Stage 2 skip policy for a completely stationary source object.

This is a numerical equality check over the entire bound pose sequence, not the
interaction-onset heuristic or a classifier for noisy tracking trajectories.
"""

from __future__ import annotations

import math

import numpy as np


def canonicalize_rigid_poses(object_poses):
    """Validate rigid poses and normalize only float32 homogeneous-row noise.

    Legacy float32 camera inversion can leave the bookkeeping bottom row a few
    ULPs from [0, 0, 0, 1]. The fixed 1e-6 representation bound is independent of
    all motion tolerances. No rotation projection, translation adjustment or
    homogeneous division is performed; input data and physical R/t are retained.
    """
    if hasattr(object_poses, "detach"):
        object_poses = object_poses.detach().cpu().numpy()
    poses = np.asarray(object_poses, dtype=np.float64)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or not len(poses)
        or not np.isfinite(poses).all()
    ):
        raise ValueError("Require a finite nonempty Nx4x4 rigid-pose sequence")
    row_error = np.max(np.abs(poses[:, 3] - [0.0, 0.0, 0.0, 1.0]), axis=1)
    rotations = poses[:, :3, :3]
    if (
        row_error.max() > 1e-6
        or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4, rtol=0)
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4, rtol=0)
    ):
        raise ValueError("Invalid rigid poses: projective bottom row or improper rotation")
    diagnostic = dict(
        max_homogeneous_row_error=float(row_error.max()),
        corrected_frames=np.flatnonzero(row_error > 0).tolist(),
        homogeneous_row_tolerance=1e-6,
        rotation_orthogonality_tolerance=1e-4,
    )
    result = poses.copy()
    result[:, 3] = [0.0, 0.0, 0.0, 1.0]
    return result, diagnostic


def stage2_skip_decision(cfg, object_poses=None):
    """Return a JSON-safe decision without opening contact maps or model assets.

    ``skip_stage2_static_obj`` defaults to true. Set it to false for explicitly
    requested contact repair against static support. Translation and rotation
    tolerances are numerical defaults, configurable with
    ``stage2_static_translation_tolerance_m`` and
    ``stage2_static_rotation_tolerance_deg``. They do not relax source fidelity.

    Supplied poses take precedence over ``is_static_obj``: a moving or invalid
    trajectory never qualifies for skipping. The explicit flag is used alone
    only when no poses are available. Never pass the old motion-onset proxy as
    that flag; it ignores rotation, early motion and slow accumulated movement.
    """
    enabled = cfg.get("skip_stage2_static_obj", True)
    explicit_static = cfg.get("is_static_obj", False)
    if type(enabled) is not bool or type(explicit_static) is not bool:
        raise ValueError("Stage 2 static policy flags must be booleans")
    translation_tolerance = cfg.get("stage2_static_translation_tolerance_m", 1e-6)
    rotation_tolerance = cfg.get("stage2_static_rotation_tolerance_deg", 1e-4)
    for name, value in (("translation", translation_tolerance), ("rotation", rotation_tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Stage 2 static {name} tolerance must be a finite nonnegative number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Stage 2 static {name} tolerance must be a finite nonnegative number")
    decision = dict(
        skip=False,
        reason="policy_disabled" if not enabled else "object_poses_unavailable",
        enabled=enabled,
        explicit_static=explicit_static,
        frame_count=None,
        max_translation_m=None,
        max_rotation_deg=None,
        translation_tolerance_m=float(translation_tolerance),
        rotation_tolerance_deg=float(rotation_tolerance),
    )
    if not enabled:
        return decision
    if object_poses is None:
        if explicit_static:
            decision.update(skip=True, reason="explicitly_static_object")
        return decision

    try:
        poses, representation = canonicalize_rigid_poses(object_poses)
    except (TypeError, ValueError):
        decision["reason"] = "invalid_object_poses"
        return decision
    decision["frame_count"] = len(poses)
    decision["pose_representation"] = representation
    rotations = poses[:, :3, :3]
    if len(poses) < 2:
        decision["reason"] = "insufficient_object_poses"
        return decision

    translation = np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1)
    # For rotations, ||R_i - R_0||_F = sqrt(8) sin(theta / 2). Unlike
    # acos(trace(R_i R_0.T)), identical float32 matrices give exactly zero.
    rotation_delta = np.linalg.norm(rotations - rotations[0], axis=(1, 2))
    angles = np.rad2deg(2 * np.arcsin(np.clip(rotation_delta / np.sqrt(8), 0, 1)))
    max_translation = float(translation.max())
    max_rotation = float(angles.max())
    static = max_translation <= translation_tolerance and max_rotation <= rotation_tolerance
    decision.update(
        skip=static,
        reason="constant_object_pose" if static else "object_pose_moves",
        max_translation_m=max_translation,
        max_rotation_deg=max_rotation,
    )
    return decision
