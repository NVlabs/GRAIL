"""
Abstract human body model interface and concrete implementations
(SOMA, SMPL-X, G1-proportioned SMPL-X).

Provides a polymorphic HumanModel interface so that HOIOptimizer can be
model-agnostic.  All NOVA / SMPL-X branching lives here.
"""

import os
from abc import ABC, abstractmethod
from glob import glob
from typing import List, Tuple

import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)

# ============================================================================
# Abstract Base Class
# ============================================================================


class HumanModel(ABC):
    """Unified interface for human body models."""

    # ---- lifecycle ---------------------------------------------------------

    @abstractmethod
    def __init__(self, cfg: dict, device: str): ...

    # ---- mesh generation ---------------------------------------------------

    @abstractmethod
    def generate_mesh(
        self, motion_data: dict, *, output_joints: bool = False, require_grad: bool = False
    ):
        """Returns (verts, faces) or (verts, faces, joints)."""
        ...

    # ---- body-segment queries ----------------------------------------------

    @abstractmethod
    def get_verts_segment(
        self, human_verts: torch.Tensor, segment_names: list[str]
    ) -> torch.Tensor: ...

    @abstractmethod
    def get_segment_indices(self, segment_names: list[str]): ...

    # ---- shape / height ----------------------------------------------------

    @abstractmethod
    def load_shape_params(self, cfg: dict, character_name: str, device: str):
        """Load and return model-specific shape parameters."""
        ...

    @abstractmethod
    def get_shape_from_motion_data(self, motion_data: dict):
        """Extract model-specific shape params from motion data dict."""
        ...

    @abstractmethod
    def compute_tpose_height(self, shape_params) -> float: ...

    # ---- motion transforms -------------------------------------------------

    @abstractmethod
    def transform_global_motion(
        self,
        global_motion_data,
        incam_motion_data,
        *,
        cam_R,
        cam_t,
        align_frame=0,
        use_global=False,
        gt_shape_params=None,
        gt_height=None,
        device="cuda",
    ): ...

    # ---- joint extraction --------------------------------------------------

    def get_native_joints(self, motion_data: dict, *, require_grad: bool = False) -> torch.Tensor:
        """Full model joints in world coordinates, not a 2D-observation subset."""
        return self.generate_mesh(motion_data, output_joints=True, require_grad=require_grad)[2]

    @abstractmethod
    def get_body_joints(self, motion_data: dict, *, require_grad: bool = False) -> torch.Tensor: ...

    @abstractmethod
    def get_hand_joints(self, motion_data: dict, *, require_grad: bool = False) -> torch.Tensor: ...

    @abstractmethod
    def get_foot_joint_indices(self) -> tuple[int, int]:
        """(left_ankle_idx, right_ankle_idx) in body-joint space."""
        ...

    @abstractmethod
    def extract_foot_poses(self, motion_data: dict) -> torch.Tensor:
        """Extract foot joint rotations (ankles + feet) as (L, 4, 3) axis-angle."""
        ...

    # ---- pose residuals ----------------------------------------------------

    @property
    @abstractmethod
    def num_body_joints(self) -> int: ...

    @property
    @abstractmethod
    def num_hand_joints(self) -> int: ...

    @abstractmethod
    def apply_pose_residuals(
        self, motion_data: dict, body_res: torch.Tensor, hand_res: torch.Tensor, frame_num: int
    ) -> dict: ...

    # ---- GT keypoint parsing -----------------------------------------------

    @abstractmethod
    def extract_gt_keypoints(self, motion_data: dict):
        """Returns (body_keypoints, hand_keypoints)."""
        ...

    # ---- pose extraction (logging) -----------------------------------------

    @abstractmethod
    def extract_root_body_pose(self, motion_data: dict):
        """Returns (root_pose, body_pose)."""
        ...


# ============================================================================
# NOVA Implementation
# ============================================================================


class SomaHumanModel(HumanModel):

    def __init__(self, cfg: dict, device: str):
        from grail.models.soma_model import (
            SOMA_BODY_JOINT_INDICES,
            SOMA_BODY_POSE_INDICES,
            SOMA_LEFT_HAND_JOINT_INDICES,
            SOMA_LEFT_HAND_POSE_INDICES,
            SOMA_RIGHT_HAND_JOINT_INDICES,
            SOMA_RIGHT_HAND_POSE_INDICES,
            setup_soma_model,
        )

        self.device = device

        model_path = cfg.get("soma_model_path")
        if model_path is None:
            raise ValueError("soma_model_path must be set in the config YAML.")
        self.model = setup_soma_model(model_path=model_path, device=device)

        self.BODY_POSE_INDICES = SOMA_BODY_POSE_INDICES
        self.LEFT_HAND_POSE_INDICES = SOMA_LEFT_HAND_POSE_INDICES
        self.RIGHT_HAND_POSE_INDICES = SOMA_RIGHT_HAND_POSE_INDICES
        self.BODY_JOINT_INDICES = SOMA_BODY_JOINT_INDICES
        self.LEFT_HAND_JOINT_INDICES = SOMA_LEFT_HAND_JOINT_INDICES
        self.RIGHT_HAND_JOINT_INDICES = SOMA_RIGHT_HAND_JOINT_INDICES

        self.model_type = "soma"

    # -- mesh ----------------------------------------------------------------

    def generate_mesh(self, motion_data, *, output_joints=False, require_grad=False):
        from grail.models.soma_model import generate_soma_mesh

        return generate_soma_mesh(
            self.model,
            motion_data,
            output_joints=output_joints,
            require_grad=require_grad,
            device=self.device,
        )

    # -- segments ------------------------------------------------------------

    def get_verts_segment(self, human_verts, segment_names):
        from grail.models.soma_model import get_soma_verts_segment

        return get_soma_verts_segment(
            human_verts,
            segment_name=segment_names,
            soma_model=self.model,
        )

    def get_segment_indices(self, segment_names):
        from grail.models.soma_model import get_soma_segment_indices

        return get_soma_segment_indices(
            segment_names,
            soma_model=self.model,
        )

    # -- shape / height ------------------------------------------------------

    def get_shape_from_motion_data(self, motion_data):
        return (motion_data.get("identity_coeffs", None), motion_data.get("scale_params", None))

    def load_shape_params(self, cfg, character_name, device):
        from grail.models.soma_model import load_soma_beta

        hm_cfg = cfg.get("human_model", cfg)
        pattern = os.path.join(
            hm_cfg["soma_shape_params_path"],
            f"*{character_name}*.pkl",
        )
        files = glob(pattern)

        if not files:
            return None
        else:
            return load_soma_beta(files[0], device=device)

    def compute_tpose_height(self, shape_params):
        from grail.models.soma_model import (
            get_tpose_human_height as get_tpose_human_height_soma,
        )

        if isinstance(shape_params, dict):
            identity_coeffs = shape_params["identity_coeffs"]
            scale_params = shape_params["scale_params"]
        elif isinstance(shape_params, tuple):
            identity_coeffs, scale_params = shape_params
        else:
            raise ValueError(f"Unexpected shape_params type: {type(shape_params)}")
        return get_tpose_human_height_soma(
            self.model,
            identity_coeffs=identity_coeffs,
            scale_params=scale_params,
            device=self.device,
        )

    # -- motion transform ----------------------------------------------------

    def transform_global_motion(
        self,
        global_motion_data,
        incam_motion_data,
        *,
        cam_R,
        cam_t,
        align_frame=0,
        use_global=False,
        gt_shape_params=None,
        gt_height=None,
        device="cuda",
    ):
        from grail.models.soma_model import (
            transform_global_motion as transform_global_motion_soma,
        )

        if gt_shape_params is not None:
            gt_identity_coeffs, gt_scale_params = gt_shape_params

            if gt_height is not None:
                cur_tpose_height = self.compute_tpose_height(gt_shape_params)
                if abs(float(cur_tpose_height) / gt_height - 1.0) > 0.05:
                    print(
                        f"  Warning: cached SOMA shape height ({float(cur_tpose_height):.3f}m) "
                        f"disagrees with character height ({gt_height:.3f}m); "
                        f"overriding scale_params[..., 0]."
                    )
                    height_ratio = gt_height / float(cur_tpose_height)
                    gt_scale_params = gt_scale_params.clone()
                    gt_scale_params[..., 0] *= height_ratio
        else:
            if gt_height is None:
                raise ValueError("Need either gt_shape_params or gt_height to set SOMA body scale.")
            pred_identity_coeffs = global_motion_data.get("identity_coeffs", None)
            pred_scale_params = global_motion_data.get("scale_params", None)
            if pred_identity_coeffs is None or pred_scale_params is None:
                raise ValueError("pred_identity_coeffs or pred_scale_params is None")

            est_tpose_height = self.compute_tpose_height((pred_identity_coeffs, pred_scale_params))
            height_ratio = gt_height / float(est_tpose_height)
            gt_identity_coeffs = pred_identity_coeffs
            gt_scale_params = pred_scale_params.clone()
            gt_scale_params[..., 0] *= height_ratio

        return transform_global_motion_soma(
            self.model,
            global_motion_data,
            incam_motion_data,
            cam_R=cam_R,
            cam_t=cam_t,
            align_frame=align_frame,
            use_global=use_global,
            gt_identity_coeffs=gt_identity_coeffs,
            gt_scale_params=gt_scale_params,
            device=device,
        )

    # -- joints --------------------------------------------------------------

    def get_body_joints(self, motion_data, *, require_grad=False):
        _, _, joints = self.generate_mesh(
            motion_data,
            output_joints=True,
            require_grad=require_grad,
        )
        return joints[:, self.BODY_JOINT_INDICES, :]

    def get_hand_joints(self, motion_data, *, require_grad=False):
        _, _, joints = self.generate_mesh(
            motion_data, output_joints=True, require_grad=require_grad
        )
        left = joints[:, self.LEFT_HAND_JOINT_INDICES, :]
        right = joints[:, self.RIGHT_HAND_JOINT_INDICES, :]
        return torch.cat([left, right], dim=1)

    def get_foot_joint_indices(self):
        return (21, 26)

    def extract_foot_poses(self, motion_data):
        poses = motion_data["poses"]
        if poses.dim() == 2:
            poses = poses.reshape(poses.shape[0], 77, 3)
        # poses[:, k, :] = SOMA_JOINT_NAMES[k] directly (no offset needed)
        # 69=LeftFoot, 70=LeftToeBase, 74=RightFoot, 75=RightToeBase
        foot_indices = [69, 70, 74, 75]
        return poses[:, foot_indices, :]  # (L, 4, 3)

    # -- pose residuals ------------------------------------------------------

    @property
    def num_body_joints(self):
        return 1 + len(self.BODY_POSE_INDICES)  # 1 root + 26 body = 27

    @property
    def num_hand_joints(self):
        return len(self.LEFT_HAND_POSE_INDICES) + len(self.RIGHT_HAND_POSE_INDICES)  # 25 + 25 = 50

    def apply_pose_residuals(self, motion_data, body_res, hand_res, frame_num):
        poses = motion_data["poses"]
        if poses.dim() == 2 and poses.shape[-1] == 231:
            poses = poses.reshape(frame_num, 77, 3)

        global_orient = poses[:, 0, :]  # (L, 3)
        body_pose = poses[:, 1:, :]  # (L, 76, 3)

        n_body = len(self.BODY_POSE_INDICES)
        n_lh = len(self.LEFT_HAND_POSE_INDICES)
        n_rh = len(self.RIGHT_HAND_POSE_INDICES)

        root_res = body_res[:, 0:1, :]  # (L, 1, 6)
        body_pose_res = body_res[:, 1:, :]  # (L, n_body, 6)

        root_mat = axis_angle_to_matrix(global_orient.reshape(-1, 3))
        root_res_mat = rotation_6d_to_matrix(root_res.reshape(-1, 6))
        pred_global = matrix_to_axis_angle(torch.bmm(root_res_mat, root_mat)).reshape(frame_num, 3)

        body_j = body_pose[:, self.BODY_POSE_INDICES, :]
        body_j_mat = axis_angle_to_matrix(body_j.reshape(-1, 3))
        body_res_mat = rotation_6d_to_matrix(body_pose_res.reshape(-1, 6))
        pred_body = matrix_to_axis_angle(torch.bmm(body_res_mat, body_j_mat)).reshape(
            frame_num, n_body, 3
        )

        lh_res = hand_res[:, :n_lh, :]
        lh_j = body_pose[:, self.LEFT_HAND_POSE_INDICES, :]
        lh_mat = axis_angle_to_matrix(lh_j.reshape(-1, 3))
        lh_res_mat = rotation_6d_to_matrix(lh_res.reshape(-1, 6))
        pred_lh = matrix_to_axis_angle(torch.bmm(lh_res_mat, lh_mat)).reshape(frame_num, n_lh, 3)

        rh_res = hand_res[:, n_lh:, :]
        rh_j = body_pose[:, self.RIGHT_HAND_POSE_INDICES, :]
        rh_mat = axis_angle_to_matrix(rh_j.reshape(-1, 3))
        rh_res_mat = rotation_6d_to_matrix(rh_res.reshape(-1, 6))
        pred_rh = matrix_to_axis_angle(torch.bmm(rh_res_mat, rh_mat)).reshape(frame_num, n_rh, 3)

        updated = body_pose.clone()
        updated[:, self.BODY_POSE_INDICES, :] = pred_body
        updated[:, self.LEFT_HAND_POSE_INDICES, :] = pred_lh
        updated[:, self.RIGHT_HAND_POSE_INDICES, :] = pred_rh

        motion_data["poses"] = torch.cat(
            [pred_global.unsqueeze(1), updated],
            dim=1,
        )
        return motion_data

    # -- GT keypoints --------------------------------------------------------

    def extract_gt_keypoints(self, motion_data):
        vitpose = motion_data["vitpose"]  # (L, 77, 3)
        body_kp = vitpose[:, self.BODY_JOINT_INDICES, :]
        hand_indices = self.LEFT_HAND_JOINT_INDICES + self.RIGHT_HAND_JOINT_INDICES
        hand_kp = vitpose[:, hand_indices, :]
        return body_kp, hand_kp

    # -- logging helpers -----------------------------------------------------

    def extract_root_body_pose(self, motion_data):
        return motion_data["poses"][:, 0, :], motion_data["poses"][:, 1:, :]


# ============================================================================
# SMPL-X Implementation
# ============================================================================


class SmplxHumanModel(HumanModel):

    def __init__(self, cfg: dict, device: str):
        from grail.models.smplx_model import (
            setup_smplx_model,
            setup_smplxlite_coco17_model,
        )

        self.device = device

        smplx_path = cfg.get("smplx_model_path", None)
        coco17_path = cfg.get("smplxlite_coco17_model_path", None)

        self.model = setup_smplx_model(
            smplx_path,
            flat_hand_mean=True,
            device=device,
        )
        self.coco17_model = setup_smplxlite_coco17_model(
            coco17_path,
            device=device,
        )

        self.model_type = "smplx"

    # -- mesh ----------------------------------------------------------------

    def generate_mesh(self, motion_data, *, output_joints=False, require_grad=False):
        from grail.models.smplx_model import generate_smplx_mesh

        return generate_smplx_mesh(
            self.model,
            motion_data,
            output_joints=output_joints,
            require_grad=require_grad,
            device=self.device,
        )

    # -- segments ------------------------------------------------------------

    def get_verts_segment(self, human_verts, segment_names):
        from grail.models.smplx_model import get_smplx_verts_segment

        return get_smplx_verts_segment(
            human_verts,
            segment_name=segment_names,
        )

    def get_segment_indices(self, segment_names):
        from grail.models.smplx_model import get_smplx_segment_indices

        return get_smplx_segment_indices(segment_names)

    # -- shape / height ------------------------------------------------------

    def get_shape_from_motion_data(self, motion_data):
        return motion_data.get("betas", None)

    def load_shape_params(self, cfg, character_name, device):
        from grail.models.smplx_model import load_smplx_beta

        hm_cfg = cfg.get("human_model", cfg)
        pattern = os.path.join(
            hm_cfg["smplx_shape_params_path"],
            f"*{character_name}*.npz",
        )
        files = glob(pattern)
        if not files:
            return None
        else:
            return load_smplx_beta(files[0], device=device)

    def compute_tpose_height(self, shape_params):
        from grail.models.smplx_model import (
            get_tpose_human_height as get_tpose_human_height_smplx,
        )

        if isinstance(shape_params, dict):
            betas = shape_params["betas"]
            scale = shape_params.get("scale", 1.0)
        elif isinstance(shape_params, tuple):
            betas, scale = shape_params
        elif isinstance(shape_params, torch.Tensor):
            betas = shape_params
            scale = 1.0
        else:
            raise ValueError(f"Unexpected shape_params type: {type(shape_params)}")

        return scale * get_tpose_human_height_smplx(
            self.model,
            betas=betas,
            device=self.device,
        )

    # -- motion transform ----------------------------------------------------

    def transform_global_motion(
        self,
        global_motion_data,
        incam_motion_data,
        *,
        cam_R,
        cam_t,
        align_frame=0,
        use_global=False,
        gt_shape_params=None,
        gt_height=None,
        device="cuda",
    ):
        from grail.models.smplx_model import (
            transform_global_motion as transform_global_motion_smplx,
        )

        if gt_shape_params is not None:
            gt_beta, gt_scale = gt_shape_params

            if gt_height is not None:
                gt_tpose_height = self.compute_tpose_height(gt_shape_params)

                if abs(gt_tpose_height / gt_height - 1.0) > 0.05:
                    gt_scale = gt_scale * gt_height / gt_tpose_height
        else:
            if gt_height is not None:
                pred_beta = global_motion_data.get("betas", None)
                if pred_beta is None:
                    raise ValueError("pred_beta is None")
                est_tpose_height = self.compute_tpose_height((pred_beta, 1.0))
                gt_beta = pred_beta
                gt_scale = gt_height / est_tpose_height
            else:
                raise ValueError("gt_height is None")

        return transform_global_motion_smplx(
            self.model,
            global_motion_data,
            incam_motion_data,
            cam_R=cam_R,
            cam_t=cam_t,
            align_frame=align_frame,
            use_global=use_global,
            gt_beta=gt_beta,
            gt_scale=gt_scale,
            device=device,
        )

    # -- joints --------------------------------------------------------------

    def get_native_joints(self, motion_data, *, require_grad=False):
        # SMPL-X appends face/mesh landmarks after its 55 articulated joints.
        # The all-joint motion prior uses the skeleton, not those regressors.
        joints = self.generate_mesh(motion_data, output_joints=True, require_grad=require_grad)[2]
        if joints.ndim != 3 or joints.shape[1] < 55 or joints.shape[-1] != 3:
            raise ValueError("SMPL-X native skeleton requires at least 55 XYZ joints")
        return joints[:, :55]

    def get_body_joints(self, motion_data, *, require_grad=False):
        from grail.models.smplx_model import get_coco17_joints

        return get_coco17_joints(
            self.coco17_model,
            motion_data,
            require_grad=require_grad,
            device=self.device,
        )

    def get_hand_joints(self, motion_data, *, require_grad=False):
        _, _, all_joints = self.generate_mesh(
            motion_data, output_joints=True, require_grad=require_grad
        )
        return torch.cat(
            [
                all_joints[:, 20:21, :],  # left wrist
                all_joints[:, 25:40, :],  # left finger joints
                all_joints[:, 21:22, :],  # right wrist
                all_joints[:, 40:55, :],  # right finger joints
            ],
            dim=1,
        )

    def get_foot_joint_indices(self):
        return (15, 16)

    def extract_foot_poses(self, motion_data):
        poses = motion_data["poses"]  # (L, 165)
        # SMPL-X body joints (0-indexed after root): L_Ankle=6, R_Ankle=7, L_Foot=9, R_Foot=10
        foot_joint_indices = [6, 7, 9, 10]
        foot_poses = []
        for idx in foot_joint_indices:
            start = 3 + idx * 3
            foot_poses.append(poses[:, start : start + 3])
        return torch.stack(foot_poses, dim=1)  # (L, 4, 3)

    # -- pose residuals ------------------------------------------------------

    @property
    def num_body_joints(self):
        return 22  # 1 root + 21 body

    @property
    def num_hand_joints(self):
        return 30  # 15 left + 15 right

    def apply_pose_residuals(self, motion_data, body_res, hand_res, frame_num):
        body_pose = motion_data["poses"]  # (L, 165)
        left_hand = motion_data["left_hand_pose"]  # (L, 45)
        right_hand = motion_data["right_hand_pose"]  # (L, 45)

        root_res = body_res[:, 0:1, :]  # (L, 1, 6)
        bpose_res = body_res[:, 1:, :]  # (L, 21, 6)

        # root
        root = body_pose[:, :3]
        root_mat = axis_angle_to_matrix(root.reshape(-1, 3))
        root_res_mat = rotation_6d_to_matrix(root_res.reshape(-1, 6))
        pred_root = matrix_to_axis_angle(torch.bmm(root_res_mat, root_mat)).reshape(frame_num, 3)

        # body (21 joints)
        bp = body_pose[:, 3:66]
        bp_mat = axis_angle_to_matrix(bp.reshape(-1, 3))
        bp_res_mat = rotation_6d_to_matrix(bpose_res.reshape(-1, 6))
        pred_bp = matrix_to_axis_angle(torch.bmm(bp_res_mat, bp_mat)).reshape(frame_num, 63)

        updated_poses = torch.cat(
            [pred_root, pred_bp, body_pose[:, 66:]],
            dim=1,
        )

        # left hand (15 joints)
        lh_res = hand_res[:, :15, :]
        lh_mat = axis_angle_to_matrix(left_hand.reshape(-1, 3))
        lh_res_mat = rotation_6d_to_matrix(lh_res.reshape(-1, 6))
        pred_lh = matrix_to_axis_angle(torch.bmm(lh_res_mat, lh_mat)).reshape(frame_num, 45)

        # right hand (15 joints)
        rh_res = hand_res[:, 15:, :]
        rh_mat = axis_angle_to_matrix(right_hand.reshape(-1, 3))
        rh_res_mat = rotation_6d_to_matrix(rh_res.reshape(-1, 6))
        pred_rh = matrix_to_axis_angle(torch.bmm(rh_res_mat, rh_mat)).reshape(frame_num, 45)

        motion_data["poses"] = updated_poses
        motion_data["left_hand_pose"] = pred_lh
        motion_data["right_hand_pose"] = pred_rh
        return motion_data

    # -- GT keypoints --------------------------------------------------------

    def extract_gt_keypoints(self, motion_data):
        body_kp = motion_data["vitpose"]  # (L, 17, 3)
        hand_kp = motion_data["hand_keypoints_2d"]  # (L, 32, 3)
        return body_kp, hand_kp

    # -- logging helpers -----------------------------------------------------

    def extract_root_body_pose(self, motion_data):
        return motion_data["poses"][:, :3], motion_data["poses"][:, 3:66]


# ============================================================================
# Factory
# ============================================================================


def create_human_model(cfg, device="cuda"):
    """Create a HumanModel instance based on config.

    Args:
        cfg: Config dict with ``body_model`` key (``"smplx"`` | ``"soma"`` | ``"soma03"`` | ``"g1_smplx"``).
        device: Target torch device.

    Returns:
        HumanModel instance.
    """
    body_model_type = cfg.get("body_model", "soma03")
    if body_model_type == "soma03":
        from grail.models.soma03_model import Soma03HumanModel

        return Soma03HumanModel(cfg, device)
    elif body_model_type == "soma":
        return SomaHumanModel(cfg, device)
    elif body_model_type == "g1_smplx":
        from grail.models.g1_smplx_human import G1SmplxHumanModel

        return G1SmplxHumanModel(cfg, device)
    elif body_model_type == "smplx":
        return SmplxHumanModel(cfg, device)
    else:
        raise ValueError(f"Unknown body_model: {body_model_type}")
