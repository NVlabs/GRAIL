import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from grail.core.torch_utils import tensor_to, tensor_to_numpy
from grail.core.video import get_video_fps_and_frame_count
from grail.pose_est.utils import smooth_axis_angle_sequence, smooth_pose_sequence


def compute_global_rotation(pose_axis_anges, joint_idx):
    """
    calculating joints' global rotation for SMPL-X (22 body joints)
    Args:
        pose_axis_anges (np.array): SMPLX's local pose (22,3)
    Returns:
        np.array: (3, 3)
    """
    global_rotation = np.eye(3)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    while joint_idx != -1:
        joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation


def fuse_smplx_mano_predictions(
    smplx_params,
    mano_params,
    left_hand_pose_relaxed,
    right_hand_pose_relaxed,
    body_keypoints_2d=None,  # (17, 3) vitpose for this frame, COCO-17 format
    wrist_dist_threshold=30.0,  # pixel distance threshold for wrist sanity check
    wrist_rot_threshold=1.5,  # radians (~86°); reject hand if local wrist rotation magnitude exceeds this
):
    M = np.diag([-1, 1, 1])  # Preparing for the left hand switch

    # Assuming that your data are stored in smplx_params and hamer_mano_params
    full_body_pose = torch.concatenate(
        (
            smplx_params["global_orient"],
            smplx_params["body_pose"].reshape(21, 3),
        ),
        dim=0,
    )

    hand_keypoints_2d = torch.zeros((32, 3))

    right_detected = False
    left_detected = False
    right_hand_idx = None
    left_hand_idx = None
    right_bbox_conf_min = 0.0
    left_bbox_conf_min = 0.0

    for i in range(len(mano_params)):
        if mano_params[i]["is_right"] > 0.5:
            right_detected = True

            bbox_conf = mano_params[i]["bbox_conf"]
            if bbox_conf > right_bbox_conf_min:
                right_bbox_conf_min = bbox_conf
                right_hand_idx = i
        else:
            left_detected = True
            bbox_conf = mano_params[i]["bbox_conf"]
            if bbox_conf > left_bbox_conf_min:
                left_bbox_conf_min = bbox_conf
                left_hand_idx = i

    # map from mano hand order to smplx hand order
    mano_to_smplx_hand_order = [
        0,  # wrist
        5,
        6,
        7,  # index
        9,
        10,
        11,  # middle
        17,
        18,
        19,  # pinky
        13,
        14,
        15,  # ring
        1,
        2,
        3,  # thumb
    ]

    # Wrist distance sanity check: reject WiLoR hand if its wrist is too far
    # from the body wrist keypoint (vitpose COCO-17: left_wrist=9, right_wrist=10)
    if body_keypoints_2d is not None and wrist_dist_threshold > 0:
        body_kpts = body_keypoints_2d
        if not isinstance(body_kpts, torch.Tensor):
            body_kpts = torch.tensor(body_kpts, dtype=torch.float32)

        # Check left hand
        if left_detected:
            body_left_wrist = body_kpts[9]  # (3,) = (x, y, conf)
            if body_left_wrist[2] > 0.3:  # only check when body wrist is reliable
                left_kpts = tensor_to(
                    mano_params[left_hand_idx]["wilor_preds"]["pred_keypoints_2d"]
                ).reshape(-1, 2)
                left_wrist_kpt = left_kpts[0]  # MANO wrist = index 0
                dist = torch.norm(left_wrist_kpt - body_left_wrist[:2]).item()
                if dist > wrist_dist_threshold:
                    left_detected = False

        # Check right hand
        if right_detected:
            body_right_wrist = body_kpts[10]  # (3,) = (x, y, conf)
            if body_right_wrist[2] > 0.3:  # only check when body wrist is reliable
                right_kpts = tensor_to(
                    mano_params[right_hand_idx]["wilor_preds"]["pred_keypoints_2d"]
                ).reshape(-1, 2)
                right_wrist_kpt = right_kpts[0]  # MANO wrist = index 0
                dist = torch.norm(right_wrist_kpt - body_right_wrist[:2]).item()
                if dist > wrist_dist_threshold:
                    right_detected = False

    if left_detected:
        left_mano_params = mano_params[left_hand_idx]["wilor_preds"]
        # hand_keypoints_2d["left_hand"] = tensor_to(
        #     left_mano_params["pred_keypoints_2d"]
        # ).reshape(-1, 2)

        left_hand_keypoints_2d = tensor_to(left_mano_params["pred_keypoints_2d"]).reshape(-1, 2)[
            mano_to_smplx_hand_order
        ]
        # add conf
        left_hand_keypoints_2d = torch.cat(
            [left_hand_keypoints_2d, torch.ones((left_hand_keypoints_2d.shape[0], 1))], dim=1
        )
        hand_keypoints_2d[:16] = left_hand_keypoints_2d

        left_elbow_global_rot = compute_global_rotation(full_body_pose, 18)  # left elbow IDX: 18

        left_wrist_global_rot = left_mano_params["global_orient"][0]
        # axis angle to matrix
        left_wrist_global_rot = R.from_rotvec(left_wrist_global_rot).as_matrix()
        # left_wrist_global_rot = M @ left_wrist_global_rot @ M
        left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot
        left_wrist_pose_vec = R.from_matrix(left_wrist_pose).as_rotvec()

        # Reject if local wrist rotation is too large (likely bad prediction)
        left_wrist_rot_mag = np.linalg.norm(left_wrist_pose_vec)
        if wrist_rot_threshold > 0 and left_wrist_rot_mag > wrist_rot_threshold:
            # Fall back to relaxed hand
            smplx_params["left_hand_pose"] = tensor_to(left_hand_pose_relaxed).reshape(1, 15, 3)
            hand_keypoints_2d[:16] = torch.zeros((16, 3))
        else:
            smplx_params["body_pose"][0, 57:60] = tensor_to(left_wrist_pose_vec).reshape(1, 3)
            smplx_params["left_hand_pose"] = tensor_to(left_mano_params["hand_pose"]).reshape(
                1, 15, 3
            )

    else:
        smplx_params["left_hand_pose"] = tensor_to(left_hand_pose_relaxed).reshape(1, 15, 3)
        hand_keypoints_2d[:16] = torch.zeros((16, 3))

    if right_detected:
        right_mano_params = mano_params[right_hand_idx]["wilor_preds"]

        right_hand_keypoints_2d = tensor_to(right_mano_params["pred_keypoints_2d"]).reshape(-1, 2)[
            mano_to_smplx_hand_order
        ]
        # add conf
        right_hand_keypoints_2d = torch.cat(
            [right_hand_keypoints_2d, torch.ones((right_hand_keypoints_2d.shape[0], 1))], dim=1
        )
        hand_keypoints_2d[16:] = right_hand_keypoints_2d

        right_elbow_global_rot = compute_global_rotation(full_body_pose, 19)  # left elbow IDX: 19

        right_wrist_global_rot = right_mano_params["global_orient"][0]
        right_wrist_global_rot = R.from_rotvec(right_wrist_global_rot).as_matrix()
        right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ right_wrist_global_rot
        right_wrist_pose_vec = R.from_matrix(right_wrist_pose).as_rotvec()

        # Reject if local wrist rotation is too large (likely bad prediction)
        right_wrist_rot_mag = np.linalg.norm(right_wrist_pose_vec)
        if wrist_rot_threshold > 0 and right_wrist_rot_mag > wrist_rot_threshold:
            # Fall back to relaxed hand
            smplx_params["right_hand_pose"] = tensor_to(right_hand_pose_relaxed).reshape(1, 15, 3)
            hand_keypoints_2d[16:] = torch.zeros((16, 3))
        else:
            smplx_params["body_pose"][0, 60:63] = tensor_to(right_wrist_pose_vec).reshape(1, 3)
            smplx_params["right_hand_pose"] = tensor_to(right_mano_params["hand_pose"]).reshape(
                1, 15, 3
            )

    else:
        smplx_params["right_hand_pose"] = tensor_to(right_hand_pose_relaxed).reshape(1, 15, 3)
        hand_keypoints_2d[16:] = torch.zeros((16, 3))

    return smplx_params, hand_keypoints_2d


def interp_smplx_mano_predictions(smplx_params):
    """
    Interpolate missing hand poses in GENMO predictions.

    Args:
        smplx_params (dict): Dictionary containing SMPLX parameters with keys:
            - 'left_hand_pose': tensor of shape (frame_num, 15, 3)
            - 'right_hand_pose': tensor of shape (frame_num, 15, 3)
            - 'body_pose': tensor of shape (frame_num, 21, 3)

    Returns:
        dict: Updated smplx_params with interpolated hand poses
    """
    import numpy as np
    import torch

    frame_num = smplx_params["body_pose"].shape[0]
    smplx_params["body_pose"] = smplx_params["body_pose"].reshape(frame_num, -1, 3)

    # Load relaxed hand poses as fallback
    data_path = os.path.join(os.path.dirname(__file__), "..", "constants", "smplx_handposes.npz")
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        left_hand_pose_relaxed, right_hand_pose_relaxed = hand_poses["relaxed"]

    left_hand_pose_relaxed = tensor_to(left_hand_pose_relaxed).reshape(15, 3)
    right_hand_pose_relaxed = tensor_to(right_hand_pose_relaxed).reshape(15, 3)

    def is_relaxed_pose(hand_pose, relaxed_pose, tolerance=1e-5):
        """Check if hand pose is close to relaxed pose (indicating missing detection)"""
        if hand_pose is None:
            return True
        return torch.allclose(hand_pose, relaxed_pose, atol=tolerance)

    def interpolate_pose_sequence(poses, valid_mask):
        """Interpolate poses for invalid frames using linear interpolation"""
        poses_interp = poses.clone()

        for i in range(frame_num):
            if not valid_mask[i]:
                # Find previous and next valid frames
                prev_valid = None
                next_valid = None

                # Search backwards for previous valid frame
                for j in range(i - 1, -1, -1):
                    if valid_mask[j]:
                        prev_valid = j
                        break

                # Search forwards for next valid frame
                for j in range(i + 1, frame_num):
                    if valid_mask[j]:
                        next_valid = j
                        break

                # Interpolate based on available valid frames
                if prev_valid is not None and next_valid is not None:
                    # Linear interpolation between prev and next
                    alpha = (i - prev_valid) / (next_valid - prev_valid)
                    poses_interp[i] = (1 - alpha) * poses[prev_valid] + alpha * poses[next_valid]
                elif prev_valid is not None:
                    # Use previous valid frame
                    poses_interp[i] = poses[prev_valid]
                elif next_valid is not None:
                    # Use next valid frame
                    poses_interp[i] = poses[next_valid]
                # If no valid frames exist, keep original (likely relaxed pose)

        return poses_interp

    # Process left hand pose
    if "left_hand_pose" in smplx_params:
        left_hand_poses = smplx_params["left_hand_pose"]
        left_valid_mask = torch.zeros(frame_num, dtype=torch.bool)

        # Identify valid (non-relaxed) hand poses
        for i in range(frame_num):
            if not is_relaxed_pose(left_hand_poses[i], left_hand_pose_relaxed):
                left_valid_mask[i] = True

        # Interpolate invalid frames
        if left_valid_mask.any():  # Only interpolate if we have some valid poses
            smplx_params["left_hand_pose"] = interpolate_pose_sequence(
                left_hand_poses, left_valid_mask
            )

            # Also interpolate left wrist rotation (body_pose[57:60])
            left_wrist_poses = smplx_params["body_pose"][:, 19, :]  # wrist is joint 19
            smplx_params["body_pose"][:, 19, :] = interpolate_pose_sequence(
                left_wrist_poses, left_valid_mask
            )

    # Process right hand pose
    if "right_hand_pose" in smplx_params:
        right_hand_poses = smplx_params["right_hand_pose"]
        right_valid_mask = torch.zeros(frame_num, dtype=torch.bool)

        # Identify valid (non-relaxed) hand poses
        for i in range(frame_num):
            if not is_relaxed_pose(right_hand_poses[i], right_hand_pose_relaxed):
                right_valid_mask[i] = True

        # Interpolate invalid frames
        if right_valid_mask.any():  # Only interpolate if we have some valid poses
            smplx_params["right_hand_pose"] = interpolate_pose_sequence(
                right_hand_poses, right_valid_mask
            )

            # Also interpolate right wrist rotation (body_pose[60:63] -> joint 20)

            right_wrist_poses = smplx_params["body_pose"].reshape(frame_num, -1, 3)[
                :, 20, :
            ]  # wrist is joint 20
            smplx_params["body_pose"][:, 20, :] = interpolate_pose_sequence(
                right_wrist_poses, right_valid_mask
            )
            # body_pose = smplx_params['body_pose'].reshape(frame_num, -1, 3)
            # right_wrist_poses = body_pose[:, 20, :]
            # right_wrist_poses = interpolate_pose_sequence(
            #     right_wrist_poses, right_valid_mask
            # )
            # body_pose[:, 20, :] = right_wrist_poses
            # smplx_params['body_pose'] = body_pose.reshape(frame_num, -1)

    # Apply Savitzky-Golay smoothing to hand poses and wrist rotations
    # Use quaternion-based smoothing for rotations to avoid discontinuities
    if "left_hand_pose" in smplx_params and smplx_params["left_hand_pose"] is not None:
        smplx_params["left_hand_pose"] = smooth_axis_angle_sequence(smplx_params["left_hand_pose"])
        # Smooth left wrist rotation
        left_wrist_poses = smplx_params["body_pose"][:, 19, :]
        smplx_params["body_pose"][:, 19, :] = smooth_axis_angle_sequence(left_wrist_poses)

    # Smooth right hand pose
    if "right_hand_pose" in smplx_params and smplx_params["right_hand_pose"] is not None:
        smplx_params["right_hand_pose"] = smooth_axis_angle_sequence(
            smplx_params["right_hand_pose"]
        )
        # Smooth right wrist rotation
        right_wrist_poses = smplx_params["body_pose"][:, 20, :]
        smplx_params["body_pose"][:, 20, :] = smooth_axis_angle_sequence(right_wrist_poses)

    return smplx_params


def run_human_pose_est_smplx(
    video_path, cache_dir, smplx_model_path=None, *, human_model_config=None
):
    """
    Predict human pose using GENMO
    """
    from grail.adapters.gem_smpl import infer_human_pose as infer_human_pose_smplx
    from grail.adapters.wilor import infer_hand_pose

    cfg = dict(human_model_config or {})
    smplx_model_path = smplx_model_path or cfg.get("smplx_model_path")
    source_fps, _ = get_video_fps_and_frame_count(video_path)
    smplx_pred = infer_human_pose_smplx(
        video_path,
        cache_dir,
        is_static_cam=bool(cfg.get("static_cam", True)),
        human_crop_repair=cfg.get("gem_smpl_human_crop_repair", False),
        human_crop_masks=cfg.get("gem_smpl_human_crop_masks"),
        human_crop_receipt=cfg.get("gem_smpl_human_crop_receipt"),
        human_crop_config=cfg.get("gem_smpl_human_crop_config"),
    )  # WARNING: static_cam=True is used to avoid droid slam failure
    smplx_global = smplx_pred["smpl_params_global"]
    data_path = os.path.join(os.path.dirname(__file__), "..", "constants", "smplx_handposes.npz")
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        left_hand_pose_relaxed, right_hand_pose_relaxed = hand_poses["relaxed"]
        hand_pose_relaxed = np.concatenate(
            (left_hand_pose_relaxed, right_hand_pose_relaxed)
        ).reshape(1, -1)

    """
    Predict hand pose using WiLoR
    """
    mano_preds = infer_hand_pose(video_path)

    """
    Fuse GENMO and WiLoR predictions
    """
    frame_num = smplx_global["body_pose"].shape[0]
    if frame_num != len(mano_preds):
        raise ValueError("Frame number mismatch between GENMO and WiLoR")

    # Create fused predictions with hand poses for each frame
    fused_smplx_global = {}
    fused_smplx_incam = {}

    # Initialize arrays to store fused results
    fused_smplx_global["body_pose"] = smplx_global["body_pose"].clone()
    fused_smplx_global["global_orient"] = smplx_global["global_orient"].clone()
    fused_smplx_global["transl"] = smplx_global["transl"].clone()
    fused_smplx_global["betas"] = smplx_global["betas"].clone()
    fused_smplx_global["left_hand_pose"] = torch.zeros((frame_num, 15, 3))
    fused_smplx_global["right_hand_pose"] = torch.zeros((frame_num, 15, 3))

    fused_smplx_incam = {}
    fused_smplx_incam["body_pose"] = smplx_pred["smpl_params_incam"]["body_pose"].clone()
    fused_smplx_incam["global_orient"] = smplx_pred["smpl_params_incam"]["global_orient"].clone()
    fused_smplx_incam["transl"] = smplx_pred["smpl_params_incam"]["transl"].clone()
    fused_smplx_incam["betas"] = smplx_pred["smpl_params_incam"]["betas"].clone()
    fused_smplx_incam["left_hand_pose"] = torch.zeros((frame_num, 15, 3))
    fused_smplx_incam["right_hand_pose"] = torch.zeros((frame_num, 15, 3))

    # Process each frame for fusion
    # hand_keypoints_2d = {
    #     "left_hand": [None] * frame_num,
    #     "right_hand": [None] * frame_num,
    # }
    hand_keypoints_2d = torch.zeros((frame_num, 32, 3))
    for i in range(frame_num):
        # Extract frame-specific predictions
        frame_global = {k: v[i : i + 1] for k, v in fused_smplx_global.items()}
        frame_incam = {k: v[i : i + 1] for k, v in fused_smplx_incam.items()}

        frame_mano_preds = mano_preds[i]

        # Fuse predictions for this frame
        frame_vitpose = smplx_pred["vitpose"][i] if smplx_pred.get("vitpose") is not None else None
        fused_frame_incam, hand_keypoints_2d_frame = fuse_smplx_mano_predictions(
            frame_incam,
            frame_mano_preds,
            left_hand_pose_relaxed,
            right_hand_pose_relaxed,
            body_keypoints_2d=frame_vitpose,
        )

        fused_smplx_incam["body_pose"][i : i + 1] = fused_frame_incam["body_pose"]
        fused_smplx_incam["left_hand_pose"][i : i + 1] = fused_frame_incam["left_hand_pose"]
        fused_smplx_incam["right_hand_pose"][i : i + 1] = fused_frame_incam["right_hand_pose"]

        fused_smplx_global["body_pose"][i : i + 1] = fused_frame_incam["body_pose"]
        fused_smplx_global["left_hand_pose"][i : i + 1] = fused_frame_incam["left_hand_pose"]
        fused_smplx_global["right_hand_pose"][i : i + 1] = fused_frame_incam["right_hand_pose"]

        hand_keypoints_2d[i : i + 1] = hand_keypoints_2d_frame

        # hand_keypoints_2d["left_hand"][i] = hand_keypoints_2d_frame["left_hand"]
        # hand_keypoints_2d["right_hand"][i] = hand_keypoints_2d_frame["right_hand"]

    # Interpolate missing hand poses in GENMO predictions
    fused_smplx_global = interp_smplx_mano_predictions(fused_smplx_global)
    fused_smplx_incam = interp_smplx_mano_predictions(fused_smplx_incam)

    # Process incam motion
    body_pose = fused_smplx_incam["body_pose"].reshape(-1, 63)
    betas = fused_smplx_incam["betas"].reshape(-1, 10)
    global_orient = fused_smplx_incam["global_orient"].reshape(-1, 3)
    trans = fused_smplx_incam["transl"].reshape(-1, 3)
    left_hand_pose = fused_smplx_incam["left_hand_pose"].reshape(-1, 45)
    right_hand_pose = fused_smplx_incam["right_hand_pose"].reshape(-1, 45)

    incam_poses = torch.zeros((frame_num, 165))
    incam_poses[:, :3] = global_orient
    incam_poses[:, 3 : 3 + 21 * 3] = body_pose
    incam_poses[:, 3 + 21 * 3 : 3 + 21 * 3 + 45] = left_hand_pose
    incam_poses[:, 3 + 21 * 3 + 45 : 3 + 21 * 3 + 45 + 45] = right_hand_pose

    incam_betas = betas[0, :10]
    incam_trans = trans

    # Get foot contact probs if available (from GENMO prediction)
    foot_contact_probs = smplx_pred.get("foot_contact_probs", None)
    if foot_contact_probs is not None:
        # Shape: (1, L, 4) -> (L, 4) for [L_Ankle, L_foot, R_Ankle, R_foot]
        foot_contact_probs = foot_contact_probs.squeeze(0)

    motion_incam = dict(
        poses=incam_poses,
        betas=incam_betas,
        trans=incam_trans,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        mocap_frame_rate=source_fps,
        gender="neutral",
        vitpose=smplx_pred["vitpose"],
        hand_keypoints_2d=hand_keypoints_2d,
        foot_contact_probs=foot_contact_probs,  # (L, 4) or None
    )

    # Process global motion
    body_pose = fused_smplx_global["body_pose"].reshape(-1, 63)
    betas = fused_smplx_global["betas"].reshape(-1, 10)
    global_orient = fused_smplx_global["global_orient"].reshape(-1, 3)
    trans = fused_smplx_global["transl"].reshape(-1, 3)
    left_hand_pose = fused_smplx_global["left_hand_pose"].reshape(-1, 45)
    right_hand_pose = fused_smplx_global["right_hand_pose"].reshape(-1, 45)

    global_poses = torch.zeros((frame_num, 165))
    global_poses[:, :3] = global_orient
    global_poses[:, 3 : 3 + 21 * 3] = body_pose
    global_poses[:, 3 + 21 * 3 : 3 + 21 * 3 + 45] = left_hand_pose
    global_poses[:, 3 + 21 * 3 + 45 : 3 + 21 * 3 + 45 + 45] = right_hand_pose

    global_betas = betas[0, :10]
    global_trans = trans

    motion_global = dict(
        poses=global_poses,
        betas=global_betas,
        trans=global_trans,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        mocap_frame_rate=source_fps,
        gender="neutral",
    )

    motion_incam["model"] = "smplx"
    motion_global["model"] = "smplx"
    if "estimator_provenance" in smplx_pred:
        for motion_space in (motion_incam, motion_global):
            motion_space["estimator_provenance"] = dict(smplx_pred["estimator_provenance"])

    # Compute predicted body height from HMR-predicted betas (using raw SMPLX).
    # Stored upstream so downstream consumers (e.g. G1SmplxHumanModel, which
    # bakes zeroed shapedirs into v_template) can't recompute it.
    if smplx_model_path is None:
        raise ValueError(
            "smplx_model_path is required to compute predicted_body_height. "
            "Pass it from the pipeline config (human_model.smplx_model_path)."
        )
    from grail.models.smplx_model import get_tpose_human_height, setup_smplx_model

    smplx_model_dir = os.path.dirname(os.path.dirname(smplx_model_path))
    smplx_model = setup_smplx_model(model_path=smplx_model_dir, flat_hand_mean=True, device="cpu")
    predicted_body_height = float(
        get_tpose_human_height(smplx_model, betas=incam_betas.cpu(), device="cpu")
    )
    motion_incam["predicted_body_height"] = predicted_body_height
    motion_global["predicted_body_height"] = predicted_body_height

    motion = dict(motion_global=motion_global, motion_incam=motion_incam)

    return motion


def run_human_pose_est_soma03(video_path, cache_dir, human_model_config):
    """Estimate native SOMA03 motion before any fixed G1 identity substitution."""
    from grail.models.soma03_model import Soma03HumanModel

    cfg = dict(human_model_config or {})
    backend = cfg.get("human_pose_backend", "gem_x")
    if backend != "gem_x":
        raise ValueError(
            f"Unsupported SOMA03 human_pose_backend: {backend!r}; use the public GEM-X backend"
        )
    from grail.adapters.gem_x_soma import infer_human_pose, prediction_to_motion

    for field in ("gem_x_source_root", "gem_x_config_path", "gem_x_python"):
        if not cfg.get(field):
            raise ValueError(f"human_model.{field} is required for human_pose_backend=gem_x")
    prediction = infer_human_pose(
        video_path,
        cache_dir,
        source_root=cfg["gem_x_source_root"],
        config_path=cfg["gem_x_config_path"],
        python=cfg["gem_x_python"],
        device=cfg.get("gem_x_device", "cuda"),
        is_static_cam=cfg.get("static_cam", True),
        human_crop_repair=cfg.get("gem_x_human_crop_repair", True),
        human_crop_masks=cfg.get("gem_x_human_crop_masks"),
        human_crop_receipt=cfg.get("gem_x_human_crop_receipt"),
        human_crop_config=cfg.get("gem_x_human_crop_config"),
        postprocess=cfg.get("gem_x_postprocess", False),
        soma_assets=cfg.get("gem_x_soma_assets"),
    )
    motion = prediction_to_motion(prediction)

    # Source height is derived from the predicted identity, never the fixed G1
    # identity. Downstream placement consumes this stored height when swapping
    # identities; it must not mutate the native fixed-global-scale channel.
    source_cfg = dict(cfg)
    source_cfg.pop("soma_identity_params_path", None)
    source_model = Soma03HumanModel(source_cfg, device="cpu")
    source_height = float(source_model.compute_tpose_height(motion["motion_incam"]))
    if not np.isfinite(source_height) or source_height <= 0:
        raise ValueError("Predicted SOMA03 source identity has an invalid body height")
    for space in ("motion_incam", "motion_global"):
        motion[space]["predicted_body_height"] = source_height
        motion[space]["estimator_provenance"]["predicted_body_height_m"] = source_height
        motion[space]["estimator_provenance"]["predicted_height_identity"] = "native_estimator"
    return motion


def run_human_pose_est(
    video_path,
    model="soma03",
    cache_dir="cache",
    *,
    human_model_config=None,
):
    """Predict with the selected native estimator; never convert/fall back silently."""
    if model not in ("soma03", "smplx", "g1_smplx"):
        raise ValueError(
            "Human pose prediction requires soma03 (GEM-X) or explicit smplx/g1_smplx (GEM-SMPL)."
        )
    if human_model_config is None:
        from grail.core.config import load_recon_config

        config_path = None if model == "soma03" else "configs/recon_4dhoi/manip_smplx.yaml"
        cfg, _ = load_recon_config(config_path)
        human_model_config = cfg["human_model"]
        if model == "smplx":
            human_model_config = {**human_model_config, "body_model": "smplx"}
    if human_model_config.get("body_model", "soma03") != model:
        raise ValueError("Human model configuration and requested backend must agree")
    if model in ("smplx", "g1_smplx"):
        # G1 shape is a downstream reconstruction choice. Raw prediction is the
        # estimator's SMPL-X body, never a SOMA conversion or baked G1 body.
        return run_human_pose_est_smplx(
            video_path, cache_dir=cache_dir, human_model_config=human_model_config
        )
    return run_human_pose_est_soma03(
        video_path, cache_dir=cache_dir, human_model_config=human_model_config
    )


if __name__ == "__main__":
    from grail.cli.estimate_soma_motion import main

    raise SystemExit(main())
