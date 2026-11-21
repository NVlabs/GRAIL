"""Typed dataclasses for HOI optimization data and predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class HOIData:
    """Ground-truth scene data assembled by HOIOptimizer.init_data()."""

    @dataclass
    class Camera:
        pose: torch.Tensor  # (4, 4) camera-to-world
        frame_height: int
        frame_width: int
        focal_length: float

    @dataclass
    class Human:
        faces: torch.Tensor  # (F, 3)
        masks: list  # per-frame binary masks, len=L
        motion_data: dict  # body-model-specific (poses, trans, scale, betas, ...)
        motion_data_global_init: dict  # global-frame reference motion
        body_keypoints_seq: torch.Tensor  # (L, J_body, 3) with confidence
        hand_keypoints_seq: torch.Tensor  # (L, J_hand, 3) with confidence
        foot_contact_probs: torch.Tensor | None  # (L, 4) or None

    @dataclass
    class Object:
        scale: torch.Tensor  # (3,)
        verts: torch.Tensor  # (V, 3) canonical mesh vertices
        faces: torch.Tensor  # (F, 3)
        masks: list  # per-frame binary masks, len=L
        verts_seq: torch.Tensor  # (L, V, 3) per-frame transformed vertices
        poses: torch.Tensor  # (L, 4, 4) SE(3) poses
        verts_tracking_seq: torch.Tensor  # (L, V, 2) projected 2D vertices

    frame_num: int
    inter_start_idx: int
    inter_end_idx: int
    human: Human
    obj: Object
    camera: Camera
    images_path: list  # list of image file paths
    depth_maps: list  # list of per-frame depth tensors
    is_static_obj: bool
    obj_sdf: Any = None
    static_objects: dict | None = None
    # Bound original contact geometry is independent of remeshed tracking verts.
    fixed_contact_context: dict | None = None


@dataclass
class OptParams:
    """Optimization parameters (residuals applied on top of initial estimates)."""

    human_trans_global: torch.Tensor  # (3,) — global translation offset
    human_trans_res: torch.Tensor  # (L, 3) — per-frame translation residuals
    human_pose_res: torch.Tensor  # (L, J_body, 6) — body pose residuals in 6D
    hand_pose_res: torch.Tensor  # (L, J_hand, 6) — hand pose residuals in 6D
    obj_R_res: torch.Tensor  # (L, 6) — object rotation residuals in 6D
    obj_t_res: torch.Tensor  # (L, 3) — object translation residuals


@dataclass
class HOIPrediction:
    """Predicted HOI state from HOIOptimizer.forward()."""

    @dataclass
    class Human:
        trans: torch.Tensor  # (L, 3)
        root_pose: torch.Tensor  # (L, 3)
        pose: torch.Tensor  # (L, J*3)
        verts_seq: torch.Tensor  # (L, V, 3)
        body_joints_seq: torch.Tensor  # (L, J_body, 3)
        body_keypoints_seq: torch.Tensor  # (L, J_body, 2)
        hand_joints_seq: torch.Tensor  # (L, J_hand, 3)
        hand_keypoints_seq: torch.Tensor  # (L, J_hand, 2)
        pose_res: torch.Tensor  # (L, J_body, 6)
        trans_res: torch.Tensor  # (L, 1, 3)
        motion_data: dict  # body-model-specific, used by get_optimized_data

    @dataclass
    class Object:
        trans: torch.Tensor  # (L, 3)
        R: torch.Tensor  # (L, 3, 3)
        verts_seq: torch.Tensor  # (L, V, 3)

    human: Human
    obj: Object
