"""Observation functions for the manager-based RL environment MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.envs.manager_env.mdp import commands, utils
from gear_sonic.trl.utils import torch_transform
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_apply_yaw,
    quat_conjugate,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Joint ordering constants (Mujoco order for compatibility)
G1_MUJOCO_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Index mappings for 29 DOF
isaaclab_to_mujoco_dof = [joint_utils.G1_ISAACLab_ORDER.index(i) for i in G1_MUJOCO_ORDER]
mujoco_to_isaaclab = [G1_MUJOCO_ORDER.index(i) for i in joint_utils.G1_ISAACLab_ORDER]


@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    command = None
    command_vel = None
    command_max = None
    command_multi_future = None
    command_multi_future_joint_pos = None
    command_multi_future_joint_body_pos = None
    command_multi_future_joint_body_diff_pos = None
    command_multi_future_joint_body_abs_pos = None
    command_multi_future_lower_body = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    root_pos_multi_future = None
    root_quat_multi_future = None
    joint_pos_multi_future = None
    smpl_pose = None
    smpl_body_pose = None
    smpl_joints_multi_future = None
    smpl_joints_lower_multi_future = None
    smpl_root_ori_b = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_pos_b_xy = None
    motion_anchor_ori_w_mf = None
    motion_anchor_ori_heading_mf = None
    motion_anchor_ori_b = None
    motion_anchor_ori_w = None
    motion_anchor_yaw_b = None
    robot_anchor_ori_w = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    # Body-only observations (for pre-trained action_transform_module)
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    vr_3point_target = None
    vr_3point_target_compliant = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    head_orn_target_multi_future = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    body_pos = None
    body_pos_diff_l = None
    # HOI manipulation task specific
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    # Object future motion observations
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    grab_contact_flag = None
    # Hand-object transform
    hand_object_transform_6d = None
    # Object-to-root observations (current state in body frame)
    object_pos_b = None
    object_ori_b_6d = None
    # Table observations
    table_pos_b = None
    table_ori_b = None
    # BPS object shape encoding (static per object, no noise)
    object_bps = None
    # RP2 full-scene multi-object observations.
    object_valid_mask = None
    object_ref_state_b = None
    object_ref_state_b_multi_future = None
    object_bps_all = None
    active_articulation_joint_state = None
    active_articulation_joint_future = None

    # Policy output from last step (latent residual + primitives, e.g., 64+2=66 dims)
    last_meta_action = None

    # Dual-arm observations
    finger_tips_force_dual_arm = None
    grab_contact_flag_dual_arm = None

    # Terrain observations
    height_map_flat = None


@configclass
class PolicyAtmCfg(ObsGroup):
    """Observations for action_transform_module (ATM).

    This observation group provides body-only observations (29 DOF) for use with
    pre-trained action_transform_module. When using a 43 DOF robot (29 body + 14 hand),
    this group extracts only the body joint observations matching ATM's expected input.

    NOTE: Order must match the observations the pretrained ATM was trained with.
    """

    # Order matches PolicyCfg: base_ang_vel, joint_pos, joint_vel, actions, gravity_dir
    base_ang_vel = None
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    gravity_dir = None


@configclass
class TeacherCfg(ObsGroup):
    """Teacher observations for distillation.

    This provides privileged state observations that the teacher policy uses.
    The student policy learns to imitate the teacher's actions using vision instead.
    """

    # # Basic proprioception
    # command = None
    # motion_anchor_pos_b = None
    # motion_anchor_ori_b = None
    # base_lin_vel = None
    # base_ang_vel = None
    # joint_pos = None
    # joint_vel = None
    # actions = None
    # # Object and manipulation observations
    # target_object_pos = None
    # hand_object_transform = None
    # finger_tips_force = None
    # grab_contact_flag = None
    # # Object future motion observations
    # object_pos_b_multi_future = None
    # object_ori_b_multi_future = None
    # # Body multi-future observations
    # command_multi_future = None
    # motion_anchor_ori_b_mf = None

    # observation terms (order preserved)
    command = None
    command_vel = None
    command_max = None
    command_multi_future = None
    command_multi_future_joint_pos = None
    command_multi_future_joint_body_pos = None
    command_multi_future_joint_body_diff_pos = None
    command_multi_future_joint_body_abs_pos = None
    command_multi_future_lower_body = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    root_pos_multi_future = None
    root_quat_multi_future = None
    joint_pos_multi_future = None
    smpl_pose = None
    smpl_body_pose = None
    smpl_joints_multi_future = None
    smpl_joints_lower_multi_future = None
    smpl_root_ori_b = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_pos_b_xy = None
    motion_anchor_ori_w_mf = None
    motion_anchor_ori_b = None
    motion_anchor_ori_w = None
    motion_anchor_yaw_b = None
    robot_anchor_ori_w = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    # Body-only observations (for pre-trained action_transform_module)
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    vr_3point_target = None
    vr_3point_target_compliant = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    head_orn_target_multi_future = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    body_pos = None
    body_pos_diff_l = None
    # HOI manipulation task specific
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    # Object future motion observations
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    grab_contact_flag = None
    # Policy output from last step (latent residual + primitives, e.g., 64+2=66 dims)
    last_meta_action = None


@configclass
class PrivilegedCfg(ObsGroup):
    """Privileged observations for the critic network (asymmetric actor-critic)."""

    command = None
    command_max = None
    command_multi_future = None
    command_multi_future_lower_body = None
    command_multi_future_lower_body_joint_pos = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_ori_heading_mf = None
    body_pos = None
    body_ori = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    vr_3point_target = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    head_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    # HOI manipulation task specific
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    # Object future motion observations
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    # Hand-object transform
    hand_object_transform_6d = None
    # Object-to-root observations (current state in body frame)
    object_pos_b = None
    object_ori_b_6d = None
    # RP2 full-scene multi-object observations.
    object_valid_mask = None
    object_ref_state_b = None
    object_ref_state_b_multi_future = None
    object_bps_all = None
    active_articulation_joint_state = None
    active_articulation_joint_future = None
    # Table observations
    table_pos_b = None
    table_ori_b = None
    # Staged training
    task_stage = None

    # g1 token obs
    ref_root_pos_future_b = None
    ref_root_ori_future_b = None
    diff_body_pos_future_local = None
    diff_body_ori_future_local = None
    diff_body_lin_vel_future_local = None
    diff_body_ang_vel_future_local = None
    grab_contact_flag = None

    # Dual-arm observations
    finger_tips_force_dual_arm = None
    grab_contact_flag_dual_arm = None

    # Terrain observations
    height_map_flat = None


@configclass
class DiscriminatorCfg:
    """Observation specifications for the discriminator."""

    disc_obs = None
    ref_disc_obs = None


@configclass
class TokenizerCfg(ObsGroup):
    """Observations for the tokenizer (SONIC/ATM encoder input)."""

    encoder_index = None
    command_multi_future_nonflat = None
    motion_anchor_ori_w = None
    command_z_multi_future_nonflat = None
    command_z = None
    motion_anchor_ori_b = None
    motion_anchor_ori_heading_b = None
    motion_anchor_ori_b_nonflat = None
    motion_anchor_ori_b_mf_nonflat = None
    motion_anchor_ori_w_mf_nonflat = None
    command_multi_future_egocentric_joint_transforms = None
    command_multi_future_egocentric_joint_transforms_nonflat = None
    command_multi_future_egocentric_joint_positions = None
    command_multi_future_egocentric_joint_positions_nonflat = None
    command_multi_future_egocentric_joint_rotations = None
    command_multi_future_egocentric_joint_rotations_nonflat = None
    command_multi_future_root_transforms = None
    command_multi_future_root_transforms_nonflat = None
    motion_anchor_ori_heading_mf_nonflat = None
    motion_anchor_ori_refheading_mf_nonflat = None
    heading_diff_robot_ref = None
    motion_anchor_ori_refheading = None
    motion_anchor_ori_heading = None
    command_multi_future_lower_body = None
    vr_3point_local_target = None
    vr_3point_local_orn_target = None
    vr_3point_local_target_compliant = None
    smpl_joints_multi_future_nonflat = None
    smpl_joints_multi_future_local_nonflat = None
    smpl_joints_multi_future_local_flatten = None
    smpl_lower_body_joints_multi_future_local_nonflat = None
    smpl_lower_body_joints_multi_future_local_flatten = None
    smpl_joints_lower_multi_future_local_flatten = None
    smpl_transl_z_multi_future_nonflat = None
    smpl_root_ori_b_multi_future = None
    smpl_root_ori_b_multi_future_flatten = None
    smpl_root_ori_refheading_multi_future = None
    smpl_root_ori_heading_multi_future = None
    smpl_elbow_wrist_pose_multi_future = None
    smpl_wrist_pose_multi_future = None
    joint_pos_multi_future_wrist = None
    joint_pos_multi_future_wrist_flatten = None
    joint_pos_multi_future_wrist_for_smpl = None
    # SOMA skeleton observations
    soma_joints_multi_future_local_nonflat = None
    soma_root_ori_b_multi_future = None
    joint_pos_multi_future_wrist_for_soma = None
    # Object goal observations (position and orientation in robot body frame)
    object_pos_b = None
    object_ori_b = None
    object_ori_b_6d = None
    object_valid_mask = None
    object_ref_state_b = None
    object_ref_state_b_multi_future = None
    object_bps_all = None
    ref_root_pos_future_b = None
    ref_root_ori_future_b = None
    diff_body_pos_future_local = None
    diff_body_ori_future_local = None
    diff_body_lin_vel_future_local = None
    diff_body_ang_vel_future_local = None
    compliance = None
    height_map_flat = None
    height_map_z_flat = None
    # Depth camera observations
    depth_image = None
    depth_image_multiframe = None
    depth_image_history = None
    # HOI encoder observations (for end-to-end SONIC-HOI training)
    command = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    target_object_pos = None
    hand_object_transform_6d = None
    finger_tips_force = None
    table_pos_b = None
    table_ori_b = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    command_multi_future = None


@configclass
class HeightMapCfg(ObsGroup):
    """Height map observation group for terrain-aware locomotion."""

    height_map = None


@configclass
class CameraRGBCfg(ObsGroup):
    """Camera RGB image observation group.

    This is a separate observation group for vision observations.
    It will be passed through the wrapper as 'camera_rgb' key in obs_dict,
    not concatenated with other policy observations.
    """

    camera_rgb = None


@configclass
class ResidualAction(ObsGroup):
    """Observation group for residual action feedback."""

    residual_action = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: PolicyCfg = None
    critic: PrivilegedCfg = None
    disc: DiscriminatorCfg = None
    tokenizer: TokenizerCfg = None
    policy_atm: PolicyAtmCfg = None  # Body-only obs for pre-trained action_transform_module
    height_map: HeightMapCfg = None
    teacher: TeacherCfg = None  # Teacher observations for distillation
    camera_rgb: CameraRGBCfg = None  # Separate vision observation group
    residual_action: ResidualAction = None


def command_max(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get maximum reference body positions in heading-local frame.

    Returns:
        torch.Tensor: Flattened body positions, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    return command.command_max


def command_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference body velocities.

    Returns:
        torch.Tensor: Flattened body velocities, shape (num_envs, vel_dim).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_vel


def command_max_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get difference between reference and robot body positions.

    Returns:
        torch.Tensor: Flattened position differences, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff


def command_max_diff_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences for multiple future frames.

    Returns:
        torch.Tensor: Flattened multi-future differences,
            shape (num_envs, num_future_frames * num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_multi_future


def command_z(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference root z-height.

    Returns:
        torch.Tensor: Root z-height, shape (num_envs, 1).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_z


def command_z_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference root z-heights for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, 1).

    Returns:
        torch.Tensor: Z-heights, shape (num_envs, num_future_frames) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_z_multi_future.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_z_multi_future


def command_max_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get maximum reference body positions for multiple future frames in heading-local frame.

    Returns:
        torch.Tensor: Flattened multi-future body positions,
            shape (num_envs, num_future_frames * num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_multi_future


def command_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference body positions in body-local frame for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_bodies * 3).

    Returns:
        torch.Tensor: Body positions, shape (num_envs, num_future_frames * num_bodies * 3)
            when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future.reshape(command.num_envs, command.num_future_frames, -1)
    else:
        return command.command_multi_future


def command_multi_future_joint_pos(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference joint positions for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_joints).

    Returns:
        torch.Tensor: Joint positions, shape (num_envs, num_future_frames * num_joints)
            when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future_joint_pos.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future_joint_pos


def command_multi_future_joint_body_pos(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference joint body positions for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_joints * 3).

    Returns:
        torch.Tensor: Joint body positions, shape (num_envs, num_future_frames * num_joints * 3)
            when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future_joint_body_pos.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future_joint_body_pos


def command_multi_future_joint_body_diff_pos(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference-minus-robot joint body position differences for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_joints * 3).

    Returns:
        torch.Tensor: Joint body position differences,
            shape (num_envs, num_future_frames * num_joints * 3) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future_joint_body_diff_pos.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future_joint_body_diff_pos


def command_multi_future_joint_body_abs_pos(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get absolute reference joint body positions for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_joints * 3).

    Returns:
        torch.Tensor: Absolute joint body positions,
            shape (num_envs, num_future_frames * num_joints * 3) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future_joint_body_abs_pos.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future_joint_body_abs_pos


def command_multi_future_lower_body(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference lower-body positions for multiple future frames in body-local frame.

    Returns:
        torch.Tensor: Flattened lower-body positions,
            shape (num_envs, num_future_frames * lower_body_dim).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_multi_future_lower_body


def command_multi_future_lower_body_joint_pos(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Get reference lower-body joint positions for multiple future frames.

    Returns:
        torch.Tensor: Flattened lower-body joint positions,
            shape (num_envs, num_future_frames * lower_body_joints).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_multi_future_lower_body_joint_pos


def command_max_diff_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in world frame.

    Returns:
        torch.Tensor: Flattened position differences, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_w


def command_max_diff_w_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in world frame for multiple future frames.

    Returns:
        torch.Tensor: Flattened multi-future differences,
            shape (num_envs, num_future_frames * num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_w_multi_future


def command_max_diff_l(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in body-local frame.

    Returns:
        torch.Tensor: Flattened position differences, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_l


def command_max_diff_l_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in body-local frame for multiple future frames.

    Returns:
        torch.Tensor: Flattened multi-future differences,
            shape (num_envs, num_future_frames * num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_l_multi_future


def command_num_frames(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-env number of valid future frames. Shape [num_envs, 1]."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if getattr(command, "per_env_num_frames", None) is not None:
        return command.per_env_num_frames.float().reshape(-1, 1)
    return torch.full(
        (command.num_envs, 1), float(command.num_future_frames), device=command.device
    )


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot anchor (pelvis) orientation in world frame as 6D rotation.

    Returns:
        torch.Tensor: First two columns of rotation matrix, shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_gravity_dir(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Compute gravity direction in the reference motion anchor frame.

    Transforms the world-frame down vector into the reference anchor's local frame.

    Returns:
        torch.Tensor: Gravity direction vector, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    gravity_dir = quat_apply(quat_inv(command.anchor_quat_w), command.down_dir)
    return gravity_dir


def gravity_dir(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Compute gravity direction in the robot anchor (pelvis) frame.

    Transforms the world-frame down vector into the robot's local frame.
    Provides the policy with tilt/orientation information.

    Returns:
        torch.Tensor: Gravity direction vector, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    gravity_dir = quat_apply(quat_inv(command.robot_anchor_quat_w), command.down_dir)
    return gravity_dir


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot anchor (pelvis) linear velocity in world frame.

    Returns:
        torch.Tensor: Linear velocity xyz, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot anchor (pelvis) angular velocity in world frame.

    Returns:
        torch.Tensor: Angular velocity xyz, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot body positions in robot anchor (pelvis) local frame.

    Transforms all tracked body positions from world frame into the robot's
    anchor frame using subtract_frame_transforms.

    Returns:
        torch.Tensor: Flattened body positions, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_pos_diff_l(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in world frame.

    Computes (reference_body_pos - robot_body_pos) for each tracked body.

    NOTE: Despite the ``_l`` suffix, the subtraction is done in world frame
    (body_pos_relative_w - robot_body_pos_w).

    Returns:
        torch.Tensor: Flattened position differences, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = commands._get_body_indexes(command, command.cfg.body_names)  # noqa: SLF001
    body_pos_diff_l = (
        command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
    )
    return body_pos_diff_l.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot body orientations in robot anchor (pelvis) local frame as 6D rotation.

    Returns:
        torch.Tensor: First two columns of each body's rotation matrix,
            shape (num_envs, num_bodies * 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str, mask_out_z=False) -> torch.Tensor:
    """Get reference motion anchor position relative to robot anchor in robot-local frame.

    Computes the position of the reference motion's root relative to the robot's
    root, expressed in the robot's local coordinate frame.

    Args:
        command_name: Name of the tracking command term.
        mask_out_z: If True, return only xy components (discard z).

    Returns:
        torch.Tensor: Position offset, shape (num_envs, 3) or (num_envs, 2) if mask_out_z.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    if mask_out_z:
        pos = pos[:, :2]

    return pos.view(env.num_envs, -1)


def motion_anchor_yaw_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get yaw (heading) difference between reference anchor and robot anchor.

    Extracts only the heading component of the relative orientation between the
    reference motion root and the robot root.

    Returns:
        torch.Tensor: Heading quaternion, shape (num_envs, 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    yaw = torch_transform.get_heading_q(ori)
    return yaw.view(env.num_envs, -1)


def motion_anchor_ori_heading_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference root orientation with heading removed as 6D rotation.

    Removes the reference anchor's own heading (yaw) from the reference root
    quaternion, preserving pitch/roll relative to gravity.

    Returns:
        torch.Tensor: 6D rotation (first two columns of rotation matrix),
            shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.motion_lib.get_root_quat_w(
        command.motion_ids, command.motion_start_time_steps + command.time_steps
    )
    root_heading_inv = quat_inv(command.anchor_heading_quat).view(env.num_envs, 4)
    deheaded_ref_rot = quat_mul(root_heading_inv, ref_root_quat)
    mat = matrix_from_quat(deheaded_ref_rot)
    deheaded_ref_root_mat = mat[..., :2].reshape(mat.shape[0], -1)
    return deheaded_ref_root_mat


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str, non_flatten=False) -> torch.Tensor:
    """Get reference anchor orientation relative to robot anchor as 6D rotation.

    Computes the orientation difference between the reference motion root and the
    robot root, expressed in the robot's local frame.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, repeat across future frames and return
            shape (num_envs, num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation representation, shape (num_envs, 6) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,  # robot
        command.robot_anchor_quat_w,
        command.anchor_pos_w,  # reference
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    ori = mat[..., :2].reshape(mat.shape[0], -1)
    if non_flatten:
        return ori.unsqueeze(1).repeat(1, command.num_future_frames, 1)
    else:
        return ori


def motion_anchor_ori_refheading(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Motion anchor orientation canonicalized by its own heading.

    Removes the heading component from the anchor orientation, preserving
    pitch/roll relative to gravity.

    Returns:
        torch.Tensor: 6D rotation matrix representation,
            shape (num_envs, 6) or (num_envs, num_future_frames, 6) if non_flatten
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ori = command.anchor_ori_refheading  # (num_envs, 6)
    if non_flatten:
        return ori.unsqueeze(1).repeat(1, command.num_future_frames, 1)
    return ori


def motion_anchor_ori_heading(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Motion anchor orientation canonicalized by robot heading (yaw).

    Uses the robot's heading for canonicalization, preserving the reference
    motion's pitch/roll relative to gravity while removing heading difference.

    Returns:
        torch.Tensor: 6D rotation matrix representation,
            shape (num_envs, 6) or (num_envs, num_future_frames, 6) if non_flatten
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ori = command.anchor_ori_heading  # (num_envs, 6)
    if non_flatten:
        return ori.unsqueeze(1).repeat(1, command.num_future_frames, 1)
    return ori


def motion_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference motion anchor orientation in world frame as 6D rotation.

    Returns:
        torch.Tensor: First two columns of rotation matrix, shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def object_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object position in robot body frame.

    Returns the object's position relative to the robot's anchor (pelvis),
    fully transformed into the robot's local coordinate system.

    Returns:
        torch.Tensor: Object position in body frame, shape (num_envs, 3)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    object_pos_w = command.target_object_root_pos
    object_quat_w = command.target_object_root_quat

    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,  # Robot anchor (pelvis) position
        command.robot_anchor_quat_w,  # Robot anchor orientation
        object_pos_w,
        object_quat_w,
    )
    return pos_b.view(env.num_envs, -1)  # (num_envs, 3)


def object_bps(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """BPS (Basis Point Set) shape encoding for the current object per environment.

    Provides a compact fixed-dimensional representation of object geometry.
    Precomputed offline (see scripts/motion/compute_bps.py) and looked up
    by motion key at load time. Static within an episode.

    Returns:
        torch.Tensor: BPS encoding, shape (num_envs, d) where d = num_basis points.
            Returns zeros if no BPS file was loaded.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.target_object_bps


def object_interaction_mask(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-object reference interaction mask, shape (num_envs, num_objects)."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    mask = command.object_interaction_mask
    if mask is None:
        max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
        return torch.zeros(env.num_envs, max_objects, device=env.device)
    return mask


def object_motion_mask(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-object motion-derived activity mask, shape (num_envs, num_objects)."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    mask = command.object_motion_mask
    if mask is None:
        max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
        return torch.zeros(env.num_envs, max_objects, device=env.device)
    return mask


def _max_num_objects(command: commands.TrackingCommand) -> int:
    return int(getattr(command.motion_lib, "max_num_objects", 1) or 1)


def _object_valid_mask(command: commands.TrackingCommand) -> torch.Tensor:
    """Return a padded per-env object-valid mask in motion/USD order."""
    max_objects = _max_num_objects(command)
    if max_objects <= 1:
        return torch.ones(command.num_envs, 1, device=command.device)

    cached = getattr(command, "_object_valid_mask_cache", None)
    if (
        cached is not None
        and cached.shape == (command.num_envs, max_objects)
        and cached.device == command.device
    ):
        return cached.float()

    valid = torch.zeros(command.num_envs, max_objects, device=command.device)
    motion_ids = command.motion_ids.detach().cpu().tolist()
    for env_idx, motion_id in enumerate(motion_ids):
        motion_key = command.motion_lib.curr_motion_keys[int(motion_id)]
        if hasattr(command, "_object_stems_for_motion_key"):
            stems = command._object_stems_for_motion_key(  # noqa: SLF001
                motion_key, max_objects=max_objects
            )
            count = len(stems)
        else:
            count = max_objects
        if count <= 0:
            count = max_objects
        valid[env_idx, : min(max_objects, count)] = 1.0
    return valid


def _object_mask_or_zeros(
    command: commands.TrackingCommand,
    mask: torch.Tensor | None,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if mask is None:
        return torch.zeros(*shape, device=command.device)
    mask = mask.float().to(command.device)
    expected = 1
    for dim in shape:
        expected *= int(dim)
    if mask.numel() == expected:
        return mask.view(*shape)
    if len(shape) == 2 and mask.shape == (shape[0],):
        return mask.view(shape[0], 1).expand(*shape)

    out = torch.zeros(*shape, device=command.device)
    flat_mask = mask.reshape(shape[0], -1)
    count = min(flat_mask.shape[1], shape[-1])
    out[:, :count] = flat_mask[:, :count]
    return out


def _future_object_mask_or_zeros(
    command: commands.TrackingCommand,
    mask_name: str,
) -> torch.Tensor:
    max_objects = _max_num_objects(command)
    getter = getattr(command.motion_lib, f"get_{mask_name}", None)
    if getter is None:
        return torch.zeros(
            command.num_envs, command.num_future_frames, max_objects, device=command.device
        )
    mask = getter(command.future_motion_ids, command.future_time_steps)
    if mask is None:
        return torch.zeros(
            command.num_envs, command.num_future_frames, max_objects, device=command.device
        )
    return mask.float().to(command.device).view(
        command.num_envs, command.num_future_frames, max_objects
    )


def object_valid_mask(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Padded object-valid mask, shape ``(num_envs, max_num_objects)``."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return _object_valid_mask(command)


def object_ref_state_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Flattened per-object current reference state in the robot anchor frame.

    Per object token layout:
    ``[valid, interaction_mask, motion_mask, pos_b(3), ori_b_6d(6), lin_vel_b(3), ang_vel_b(3)]``.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    max_objects = _max_num_objects(command)

    object_pos_w = command.object_root_pos
    object_quat_w = command.object_root_quat
    robot_pos_w = command.robot_anchor_pos_w[:, None, :].expand(-1, max_objects, -1)
    robot_quat_w = command.robot_anchor_quat_w[:, None, :].expand(-1, max_objects, -1)

    pos_b, ori_b = subtract_frame_transforms(
        robot_pos_w.reshape(-1, 3),
        robot_quat_w.reshape(-1, 4),
        object_pos_w.reshape(-1, 3),
        object_quat_w.reshape(-1, 4),
    )
    pos_b = pos_b.view(env.num_envs, max_objects, 3)
    ori_6d = matrix_from_quat(ori_b).view(env.num_envs, max_objects, 3, 3)[
        ..., :2
    ].reshape(env.num_envs, max_objects, 6)

    current_steps = command.motion_start_time_steps + command.time_steps
    try:
        lin_vel_w = command.motion_lib.get_object_lin_vel(command.motion_ids, current_steps)
        ang_vel_w = command.motion_lib.get_object_ang_vel(command.motion_ids, current_steps)
    except Exception:  # noqa: BLE001
        lin_vel_w = torch.zeros(env.num_envs, max_objects, 3, device=env.device)
        ang_vel_w = torch.zeros(env.num_envs, max_objects, 3, device=env.device)

    robot_quat_inv = quat_inv(robot_quat_w.reshape(-1, 4))
    lin_vel_b = quat_apply(robot_quat_inv, lin_vel_w.reshape(-1, 3)).view(
        env.num_envs, max_objects, 3
    )
    ang_vel_b = quat_apply(robot_quat_inv, ang_vel_w.reshape(-1, 3)).view(
        env.num_envs, max_objects, 3
    )

    valid = _object_valid_mask(command)
    interaction = _object_mask_or_zeros(
        command, command.object_interaction_mask, (env.num_envs, max_objects)
    )
    motion = _object_mask_or_zeros(command, command.object_motion_mask, (env.num_envs, max_objects))
    state = torch.cat(
        [
            valid[..., None],
            interaction[..., None],
            motion[..., None],
            pos_b,
            ori_6d,
            lin_vel_b,
            ang_vel_b,
        ],
        dim=-1,
    )
    state = state * valid[..., None]
    return state.reshape(env.num_envs, -1)


def object_ref_state_b_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Flattened per-object future reference state in the current robot anchor frame.

    Per future/object token layout:
    ``[valid, interaction_mask, motion_mask, pos_b(3), ori_b_6d(6)]``.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    max_objects = _max_num_objects(command)
    num_future = command.num_future_frames

    object_pos_w = command.object_root_pos_multi_future
    object_quat_w = command.object_root_quat_multi_future
    robot_pos_w = command.robot_anchor_pos_w[:, None, None, :].expand(
        -1, num_future, max_objects, -1
    )
    robot_quat_w = command.robot_anchor_quat_w[:, None, None, :].expand(
        -1, num_future, max_objects, -1
    )

    pos_b, ori_b = subtract_frame_transforms(
        robot_pos_w.reshape(-1, 3),
        robot_quat_w.reshape(-1, 4),
        object_pos_w.reshape(-1, 3),
        object_quat_w.reshape(-1, 4),
    )
    pos_b = pos_b.view(env.num_envs, num_future, max_objects, 3)
    ori_6d = matrix_from_quat(ori_b).view(env.num_envs, num_future, max_objects, 3, 3)[
        ..., :2
    ].reshape(env.num_envs, num_future, max_objects, 6)

    valid = _object_valid_mask(command)[:, None, :].expand(-1, num_future, -1)
    interaction = _future_object_mask_or_zeros(command, "object_interaction_mask")
    motion = _future_object_mask_or_zeros(command, "object_motion_mask")
    state = torch.cat(
        [
            valid[..., None],
            interaction[..., None],
            motion[..., None],
            pos_b,
            ori_6d,
        ],
        dim=-1,
    )
    state = state * valid[..., None]
    return state.reshape(env.num_envs, -1)


def object_bps_all(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Flattened per-object BPS features in motion/USD order."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    max_objects = _max_num_objects(command)
    if max_objects <= 1:
        return command.object_bps

    cached = getattr(command, "_object_bps_all_cache", None)
    if cached is not None and cached.shape[:2] == (env.num_envs, max_objects):
        return cached.reshape(env.num_envs, -1)

    bps_lookup = getattr(command.motion_lib, "_bps_lookup", None) or {}
    bps_dim = int(getattr(command.motion_lib, "_bps_dim", 0) or 0)
    if bps_dim <= 0:
        bps = command.motion_lib.get_object_bps(command.motion_ids)
        if bps.ndim == 3 and bps.shape[1] == max_objects:
            return bps.reshape(env.num_envs, -1)
        return torch.zeros(env.num_envs, max_objects, 1, device=env.device).reshape(
            env.num_envs, -1
        )

    out = torch.zeros(env.num_envs, max_objects, bps_dim, device=env.device)
    motion_ids = command.motion_ids.detach().cpu().tolist()
    for env_idx, motion_id in enumerate(motion_ids):
        motion_key = command.motion_lib.curr_motion_keys[int(motion_id)]
        stems = command._object_stems_for_motion_key(  # noqa: SLF001
            motion_key, max_objects=max_objects
        )
        for obj_idx, stem in enumerate(stems[:max_objects]):
            vec = bps_lookup.get(stem)
            if vec is not None:
                out[env_idx, obj_idx] = torch.as_tensor(
                    vec, dtype=torch.float32, device=env.device
                )
    out = out * _object_valid_mask(command)[..., None]
    return out.reshape(env.num_envs, -1)


def object_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object orientation in robot body frame (quaternion representation).

    Returns the object's orientation relative to the robot's anchor,
    represented as quaternion (xyzw format).

    Returns:
        torch.Tensor: Object orientation in body frame, shape (num_envs, 4)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    object_pos_w = command.target_object_root_pos
    object_quat_w = command.target_object_root_quat

    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        object_pos_w,
        object_quat_w,
    )

    return ori_b.view(env.num_envs, -1)  # (num_envs, 4)


def object_ori_b_6d(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object orientation in robot body frame (6D rotation representation).

    Returns the object's orientation relative to the robot's anchor,
    represented as the first two columns of the rotation matrix.

    Returns:
        torch.Tensor: Object orientation in body frame, shape (num_envs, 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    object_pos_w = command.target_object_root_pos
    object_quat_w = command.target_object_root_quat

    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        object_pos_w,
        object_quat_w,
    )

    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)  # (num_envs, 6)


def object_pos_b_multi_future(
    env: ManagerBasedEnv, command_name: str, debug_vis: bool = False
) -> torch.Tensor:
    """Object future positions in robot body frame.

    Returns the object's future positions relative to the robot's anchor (pelvis),
    fully transformed into the robot's local coordinate system.

    Returns:
        torch.Tensor: Object future positions in body frame, shape (num_envs, num_future_frames * 3)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    object_pos_w = command.target_object_root_pos_multi_future

    # Transform to body frame using robot anchor
    # Expand robot anchor to match future frames
    robot_anchor_pos = command.robot_anchor_pos_w[:, None, :].expand_as(
        object_pos_w
    )  # (num_envs, num_future_frames, 3)
    robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(
        -1, command.num_future_frames, -1
    )  # (num_envs, num_future_frames, 4)

    # Position relative to anchor
    pos_rel = object_pos_w - robot_anchor_pos  # (num_envs, num_future_frames, 3)

    # Rotate to body frame
    pos_b = quat_apply(quat_inv(robot_anchor_quat.reshape(-1, 4)), pos_rel.reshape(-1, 3))
    pos_b = pos_b.reshape(env.num_envs, command.num_future_frames, 3)
    # Debug visualization: draw pos_b transformed back to world frame
    if debug_vis:
        if not hasattr(env, "_debug_draw_object_future"):
            try:
                from isaacsim.util.debug_draw import _debug_draw  # noqa: PLC0415

                env._debug_draw_object_future = (  # noqa: SLF001
                    _debug_draw.acquire_debug_draw_interface()
                )  # noqa: SLF001
            except Exception:  # noqa: BLE001
                env._debug_draw_object_future = None  # noqa: SLF001

        if env._debug_draw_object_future is not None:  # noqa: SLF001
            env._debug_draw_object_future.clear_points()  # noqa: SLF001
            env._debug_draw_object_future.clear_lines()  # noqa: SLF001

            # Transform pos_b back to world frame for visualization
            pos_rel_recovered = quat_apply(
                robot_anchor_quat.reshape(-1, 4), pos_b.reshape(-1, 3)
            ).reshape(env.num_envs, command.num_future_frames, 3)
            pos_w_from_b = pos_rel_recovered + robot_anchor_pos

            num_frames = command.num_future_frames
            for env_idx in range(env.num_envs):
                prev_point = None
                for frame_idx in range(num_frames):
                    # Color gradient: green (early) -> yellow (late)
                    t = frame_idx / max(num_frames - 1, 1)
                    color = (t, 1.0, 0.0, 1.0)  # (R, G, B, A): green -> yellow

                    pos = pos_w_from_b[env_idx, frame_idx]
                    point = [(pos[0].item(), pos[1].item(), pos[2].item())]

                    # Draw sphere for each future position
                    size = 15 + int(10 * t)  # Larger for later frames
                    env._debug_draw_object_future.draw_points(  # noqa: SLF001
                        point, [color], [size]
                    )  # noqa: SLF001

                    # Draw line connecting consecutive frames
                    if prev_point is not None:
                        env._debug_draw_object_future.draw_lines(  # noqa: SLF001
                            [prev_point], [point[0]], [color], [2]
                        )
                    prev_point = point[0]

                # Also draw robot anchor position (cyan)
                anchor_pos = robot_anchor_pos[env_idx, 0]
                anchor_point = [(anchor_pos[0].item(), anchor_pos[1].item(), anchor_pos[2].item())]
                env._debug_draw_object_future.draw_points(  # noqa: SLF001
                    anchor_point, [(0.0, 1.0, 1.0, 1.0)], [30]
                )

    return pos_b.reshape(env.num_envs, -1)  # (num_envs, num_future_frames * 3)


def object_ori_b_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object future orientations in robot body frame (quaternion representation).

    Returns the object's future orientations relative to the robot's anchor,
    represented as quaternions (xyzw format).

    Returns:
        torch.Tensor: Object future orientations in body frame, shape (num_envs, num_future_frames * 4)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    object_quat_w = command.target_object_root_quat_multi_future

    # Get robot anchor orientation expanded to match
    robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(
        -1, command.num_future_frames, -1
    )

    # Compute relative orientation: q_rel = q_anchor^{-1} * q_object
    ori_b = quat_mul(quat_inv(robot_anchor_quat.reshape(-1, 4)), object_quat_w.reshape(-1, 4))
    ori_b = ori_b.reshape(env.num_envs, command.num_future_frames, 4)

    return ori_b.reshape(env.num_envs, -1)  # (num_envs, num_future_frames * 4)


def object_pos_delta_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future REFERENCE object positions as deltas from current SIMULATED object position.

    Following OmniGrasp/MimicManipulator: (^p_obj_{t+k} - p_obj_t)
    - ^p (with hat) = reference/target future positions
    - p (no hat) = simulated current position (where object actually is)

    This tells the policy: "where should the object go relative to where it IS now"

    Positions are computed in robot body frame before taking the delta.

    Returns:
        torch.Tensor: Position deltas, shape (num_envs, num_future_frames * 3)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    # Get SIMULATED current object position (where it actually is)
    object_pos_current_w, _, _, _, _ = utils.get_target_object_current_state(env, command)

    # Get REFERENCE future object positions (where it should go)
    object_pos_future_w = command.target_object_root_pos_multi_future

    # Get robot anchor for body frame transform
    robot_anchor_pos = command.robot_anchor_pos_w
    robot_anchor_quat = command.robot_anchor_quat_w
    robot_anchor_quat_inv = quat_inv(robot_anchor_quat)

    # Transform SIMULATED current object position to body frame
    pos_rel_current = object_pos_current_w - robot_anchor_pos
    pos_b_current = quat_apply(robot_anchor_quat_inv, pos_rel_current)  # (num_envs, 3)

    # Transform REFERENCE future object positions to body frame
    robot_anchor_pos_expanded = robot_anchor_pos[:, None, :].expand_as(object_pos_future_w)
    pos_rel_future = (
        object_pos_future_w - robot_anchor_pos_expanded
    )  # (num_envs, num_future_frames, 3)
    robot_anchor_quat_inv_expanded = robot_anchor_quat_inv[:, None, :].expand(
        -1, command.num_future_frames, -1
    )
    pos_b_future = quat_apply(
        robot_anchor_quat_inv_expanded.reshape(-1, 4), pos_rel_future.reshape(-1, 3)
    ).reshape(env.num_envs, command.num_future_frames, 3)

    # Compute delta: reference_future - simulated_current (both in body frame)
    pos_delta = pos_b_future - pos_b_current[:, None, :]  # (num_envs, num_future_frames, 3)

    return pos_delta.reshape(env.num_envs, -1)  # (num_envs, num_future_frames * 3)


def object_ori_delta_multi_future_6d(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future REFERENCE object orientations as deltas from current SIMULATED object orientation (6D).

    Following OmniGrasp/MimicManipulator: (^q_obj_{t+k} ⊖ q_obj_t) = q_obj_t^{-1} * ˆq_obj_{t+k}
    - ^q (with hat) = reference/target future orientations
    - q (no hat) = simulated current orientation (where object actually is)

    This tells the policy: "how should the object rotate relative to its current orientation"

    The delta is computed as relative rotation from simulated current to reference future.
    Output is 6D rotation representation (first two columns of rotation matrix).

    Returns:
        torch.Tensor: Orientation deltas in 6D, shape (num_envs, num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    # Get SIMULATED current object orientation (where it actually is)
    _, object_quat_current_w, _, _, _ = utils.get_target_object_current_state(env, command)

    # Get REFERENCE future object orientations (where it should go)
    object_quat_future_w = command.target_object_root_quat_multi_future

    # Compute delta: q_simulated_current^{-1} * q_reference_future
    object_quat_current_inv = quat_inv(object_quat_current_w)  # (num_envs, 4)
    object_quat_current_inv_expanded = object_quat_current_inv[:, None, :].expand(
        -1, command.num_future_frames, -1
    )  # (num_envs, num_future_frames, 4)

    ori_delta = quat_mul(
        object_quat_current_inv_expanded.reshape(-1, 4), object_quat_future_w.reshape(-1, 4)
    )  # (num_envs * num_future_frames, 4)

    # Convert to 6D rotation representation
    mat = matrix_from_quat(ori_delta)  # (num_envs * num_future_frames, 3, 3)
    ori_6d = mat[..., :2].reshape(env.num_envs, command.num_future_frames, 6)

    return ori_6d.reshape(env.num_envs, -1)  # (num_envs, num_future_frames * 6)


def active_entity_base_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference base position for the one active rigid or articulated entity."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.active_entity_root_pos,
        command.active_entity_root_quat,
    )
    return pos_b


def active_entity_base_ori_b_6d(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Reference base orientation for the active entity in the robot frame."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.active_entity_root_pos,
        command.active_entity_root_quat,
    )
    return matrix_from_quat(ori_b)[..., :2].reshape(env.num_envs, 6)


def active_entity_base_pos_delta_multi_future(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Future reference base position minus current simulated base position."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    current_pos_w, _, _, _, _ = utils.get_active_entity_base_current_state(env, command)
    future_pos_w = command.active_entity_root_pos_multi_future
    anchor_pos_w = command.robot_anchor_pos_w
    anchor_quat_inv = quat_inv(command.robot_anchor_quat_w)

    current_pos_b = quat_apply(anchor_quat_inv, current_pos_w - anchor_pos_w)
    future_pos_b = quat_apply(
        anchor_quat_inv[:, None, :]
        .expand(-1, command.num_future_frames, -1)
        .reshape(-1, 4),
        (future_pos_w - anchor_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape(env.num_envs, command.num_future_frames, 3)
    return (future_pos_b - current_pos_b[:, None, :]).reshape(env.num_envs, -1)


def active_entity_base_ori_delta_multi_future_6d(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Future reference base rotation relative to the simulated active base."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    _, current_quat_w, _, _, _ = utils.get_active_entity_base_current_state(env, command)
    future_quat_w = command.active_entity_root_quat_multi_future
    relative = quat_mul(
        quat_inv(current_quat_w)[:, None, :]
        .expand(-1, command.num_future_frames, -1)
        .reshape(-1, 4),
        future_quat_w.reshape(-1, 4),
    )
    return matrix_from_quat(relative)[..., :2].reshape(env.num_envs, -1)


def table_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Table position in robot body frame.

    Returns the table's position relative to the robot's anchor (pelvis),
    transformed into the robot's local coordinate system.

    Returns:
        torch.Tensor: Table position in body frame, shape (num_envs, 3)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    if "table" not in env.scene.rigid_objects:
        return torch.zeros(env.num_envs, 3, device=env.device)

    table = env.scene["table"]
    table_pos_w = table.data.root_pos_w  # (num_envs, 3)
    table_quat_w = table.data.root_quat_w  # (num_envs, 4)

    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        table_pos_w,
        table_quat_w,
    )
    return pos_b.view(env.num_envs, -1)  # (num_envs, 3)


def table_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Table orientation in robot body frame (6D rotation representation).

    Returns the table's orientation relative to the robot's anchor,
    using the first two columns of the rotation matrix.

    Returns:
        torch.Tensor: Table orientation in body frame, shape (num_envs, 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    if "table" not in env.scene.rigid_objects:
        return torch.zeros(env.num_envs, 6, device=env.device)

    table = env.scene["table"]
    table_pos_w = table.data.root_pos_w
    table_quat_w = table.data.root_quat_w

    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        table_pos_w,
        table_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)  # (num_envs, 6)


def motion_anchor_ori_b_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference-vs-robot root orientation difference for multiple future frames in body-local frame.

    Uses the robot's full orientation (including pitch/roll) for normalization.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation differences,
            shape (num_envs, num_future_frames * 6) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_dif_l_multi_future.reshape(
            env.num_envs, command.num_future_frames, -1
        )
    else:
        return command.root_rot_dif_l_multi_future.reshape(env.num_envs, -1)


def motion_anchor_ori_w_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference root orientations in world frame for multiple future frames as 6D rotation.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation representations,
            shape (num_envs, num_future_frames * 6) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_w_multi_future.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return command.root_rot_w_multi_future.reshape(env.num_envs, -1)


# =============================================================================
# Egocentric joint transforms and root transforms relative to first frame
# =============================================================================


def command_multi_future_egocentric_joint_transforms(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Egocentric joint transforms (positions + rotations) for reference frames.

    For each future frame, joint positions and rotations are relative to that frame's
    projected root (heading/yaw only rotation, z=0).

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, num_bodies_full * 9] (3 pos + 6 rot per body, flattened)
        else:
            [num_envs, num_future_frames * num_bodies_full * 9]
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    transforms = command.egocentric_joint_transforms_multi_future
    if non_flatten:
        return transforms.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return transforms.reshape(env.num_envs, -1)


def command_multi_future_egocentric_joint_positions(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Egocentric joint positions for reference frames.

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, num_bodies_full * 3]
        else:
            [num_envs, num_future_frames * num_bodies_full * 3]
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    positions = command.egocentric_joint_positions_multi_future
    if non_flatten:
        return positions.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return positions.reshape(env.num_envs, -1)


def command_multi_future_egocentric_joint_rotations(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Egocentric joint rotations (6D) for reference frames.

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, num_bodies_full * 6]
        else:
            [num_envs, num_future_frames * num_bodies_full * 6]
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    rotations = command.egocentric_joint_rotations_multi_future
    if non_flatten:
        return rotations.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return rotations.reshape(env.num_envs, -1)


def command_multi_future_root_transforms(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Root transforms (position + rotation) relative to the first reference frame.

    Position: Delta from first frame's projected root position, in first frame's heading frame.
    Rotation: Relative rotation from first frame's heading quaternion (6D representation).

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, 1 * 9] (3 pos + 6 rot)
        else:
            [num_envs, num_future_frames * 1 * 9]
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    transforms = command.root_transforms_relative_to_first_frame
    if non_flatten:
        return transforms.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return transforms.reshape(env.num_envs, -1)


# =============================================================================
# Motionbricks observations (global and first-frame-heading variants)
# =============================================================================


def command_multi_future_global_joint_transforms_with_pelvis_rotation_and_height(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """
    Global joint transforms with per-frame root XY subtracted. No heading canonicalization.
    Pelvis (body 0) has XY=(0,0), Z=height, rotation=raw global.

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, num_bodies_full * 9]
        else:
            [num_envs, num_future_frames * num_bodies_full * 9]
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    transforms = command.global_joint_transforms_with_pelvis_rotation_and_height_multi_future
    if non_flatten:
        return transforms.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return transforms.reshape(env.num_envs, -1)


def command_multi_future_global_root_xy_delta_wrt_first_frame(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """
    Root xy in global frame minus first frame root xy: root_xy[t] - root_xy[0].

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, 2]
        else:
            [num_envs, num_future_frames * 2]
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    xy_delta = command.global_root_xy_delta_wrt_first_frame_multi_future
    if non_flatten:
        return xy_delta.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return xy_delta.reshape(env.num_envs, -1)


def command_multi_future_global_root_xy_delta_wrt_previous_frame(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """
    Per-frame root xy delta in global frame: root_xy[t] - root_xy[t-1].

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, 2]
        else:
            [num_envs, num_future_frames * 2]
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    xy_delta = command.global_root_xy_delta_wrt_previous_frame_multi_future
    if non_flatten:
        return xy_delta.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return xy_delta.reshape(env.num_envs, -1)


def command_multi_future_firstframe_heading_canonicalized_global_joint_transforms(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """
    Joint transforms in first-frame heading frame with per-frame root XY subtracted.
    Pelvis (body 0) has XY=(0,0), Z=height in heading frame, rotation=heading-relative.

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, num_bodies_full * 9]
        else:
            [num_envs, num_future_frames * num_bodies_full * 9]
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    transforms = command.firstframe_heading_canonicalized_global_joint_transforms_multi_future
    if non_flatten:
        return transforms.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return transforms.reshape(env.num_envs, -1)


def command_multi_future_firstframe_heading_canonicalized_global_root_xy_delta(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """
    Root XY delta in first frame's heading frame.

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, 2]
        else:
            [num_envs, num_future_frames * 2]
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    xy = command.firstframe_heading_canonicalized_global_root_xy_delta_multi_future
    if non_flatten:
        return xy.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return xy.reshape(env.num_envs, -1)


def motion_anchor_ori_heading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Motion anchor orientation normalized by robot heading (yaw) only, for multi-future frames.

    Unlike motion_anchor_ori_b_mf which normalizes by the robot's full orientation (including
    pitch/roll), this version only normalizes by the robot's heading (yaw). This preserves
    the reference motion's pitch/roll relative to gravity while removing the heading difference.

    Returns:
        torch.Tensor: Orientation as 6D rotation matrix representation (first 2 columns of
            rotation matrix), shape (num_envs, num_future_frames, 6) if non_flatten else
            (num_envs, num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_dif_heading_multi_future.reshape(
            env.num_envs, command.num_future_frames, -1
        )
    else:
        return command.root_rot_dif_heading_multi_future.reshape(env.num_envs, -1)


def motion_anchor_ori_refheading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Motion anchor orientation canonicalized by the first target frame's heading.

    Instead of using the robot's heading, this uses the heading of the first (immediate)
    future frame from the reference motion. The trajectory is expressed in the reference
    motion's own heading frame.

    Returns:
        torch.Tensor: Orientation as 6D rotation matrix representation,
            shape (num_envs, num_future_frames, 6) if non_flatten else
            (num_envs, num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_dif_refheading_multi_future.reshape(
            env.num_envs, command.num_future_frames, -1
        )
    else:
        return command.root_rot_dif_refheading_multi_future.reshape(env.num_envs, -1)


def heading_diff_robot_ref(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Relative heading from robot to reference motion's first target frame.

    Computes the rotation from the robot's heading to the reference motion's first
    future frame heading as a single-frame 6D rotation.

    Returns:
        torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.heading_diff_robot_ref


### 3 point force based tracking
def vr_3point_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point tracking target positions in heading-local frame.

    Transforms the reference 3-point body positions (left wrist, right wrist, head)
    relative to the robot anchor, then de-heads (removes yaw) to get a
    heading-invariant representation.

    Returns:
        torch.Tensor: Flattened 3-point positions, shape (num_envs, 9).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    pos_b = command.vr_3point_body_pos_w - command.robot_anchor_pos_w[:, None, :]

    # transform pos_b in to deheaded root frame
    root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    deheaded_pos_b = quat_apply_yaw(quat_inv(root_quat), pos_b)
    return deheaded_pos_b.view(env.num_envs, -1)


def _phase_to_weight_pyramid(phase, start=0.25, end=0.75):
    """Compute pyramid-shaped weight from a [0,1] phase value.

    Ramps linearly from 0 to 1 over [0, start], holds at 1 over [start, end],
    then ramps linearly from 1 to 0 over [end, 1].
    """
    weight = torch.zeros_like(phase)
    mask1 = phase < start
    mask2 = phase > end
    weight[mask1] = 1 / start * phase[mask1]
    weight[mask2] = 1 / (1 - end) * (1 - phase[mask2])
    weight[~(mask1 | mask2)] = 1.0
    return weight


def vr_3point_target_compliant_multi_future(
    env: ManagerBasedEnv, motion_command_name: str, force_command_name: str
) -> torch.Tensor:
    """Get force-compliant VR 3-point targets for multiple future frames in heading-local frame.

    Modifies the reference 3-point positions by subtracting external force displacements
    (scaled by compliance/stiffness) to create compliant tracking targets. The force
    profile uses a pyramid-shaped phase weighting across future frames.

    Returns:
        torch.Tensor: Flattened compliant targets,
            shape (num_envs, num_future_frames * num_points * 3).
    """
    motion_command: commands.TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: commands.ForceTrackingCommand = env.command_manager.get_term(force_command_name)

    future_phases = (
        (
            force_command.force_push_counter[:, None]
            + motion_command.future_time_steps_init
            - force_command.force_update_frequency
        )
        / force_command.force_duration_per_env[:, None]
    ).clamp(min=0.0, max=1.0)
    # future_phases shape: (num_envs, num_future_frames)
    future_forces = (
        force_command.body_force_magnitude_buf[:, None, None, None]
        * _phase_to_weight_pyramid(future_phases)[:, :, None, None]
        * (force_command.body_force_dir_buf[:, force_command.force_push_ids_rel])[:, None, :, :]
        * force_command.max_force
    )

    # future_forces [E, 5, 2, 3]
    ext_force_disp = future_forces * force_command.eef_stiffness_buf[:, None, :, None]

    mod_3point = motion_command.vr_3point_body_pos_w_multi_future
    mod_3point -= ext_force_disp
    pos_b = mod_3point - motion_command.robot_anchor_pos_w[:, None, None, :]
    # transform pos_b in to deheaded root frame
    root_quat = motion_command.robot_anchor_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, motion_command.num_future_frames, len(motion_command.cfg.vr_3point_body), 1
    )
    deheaded_pos_b = quat_apply_yaw(quat_inv(root_quat), pos_b)
    return deheaded_pos_b.view(env.num_envs, -1)


def vr_3point_target_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point tracking target positions for multiple future frames in heading-local frame.

    Returns:
        torch.Tensor: Flattened multi-future 3-point positions,
            shape (num_envs, num_future_frames * num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    pos_b = command.vr_3point_body_pos_w_multi_future - command.robot_anchor_pos_w[:, None, None, :]
    # transform pos_b in to deheaded root frame
    root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, command.num_future_frames, len(command.cfg.vr_3point_body), 1
    )
    deheaded_pos_b = quat_apply_yaw(quat_inv(root_quat), pos_b)
    return deheaded_pos_b.view(env.num_envs, -1)


def head_orn_target_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference head orientation for multiple future frames, de-headed by robot yaw.

    Returns:
        torch.Tensor: Flattened head quaternions,
            shape (num_envs, num_future_frames * 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    head_quat = command.head_orn_w_multi_future
    root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, command.num_future_frames, 1
    )
    # deheaded_head_quat = quat_mul(head_quat, get_heading_q(quat_inv(root_quat)))
    deheaded_head_quat = quat_mul(torch_transform.get_heading_q(quat_inv(root_quat)), head_quat)
    return deheaded_head_quat.view(env.num_envs, -1)


def vr_3point_orn_target_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point body orientations for multiple future frames, de-headed by robot yaw.

    Returns:
        torch.Tensor: Flattened body quaternions,
            shape (num_envs, num_future_frames * num_points * 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    vr_3point_quat = command.vr_3point_body_quat_w_multi_future
    root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, command.num_future_frames, len(command.cfg.vr_3point_body), 1
    )
    # deheaded_vr_3point_quat = quat_mul(vr_3point_quat, get_heading_q(quat_inv(root_quat)))
    deheaded_vr_3point_quat = quat_mul(
        torch_transform.get_heading_q(quat_inv(root_quat)), vr_3point_quat
    )
    return deheaded_vr_3point_quat.view(env.num_envs, -1)


def vr_3point_local_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point positions in reference motion anchor local frame.

    NOTE: "local" here means relative to the reference motion root, not the robot root.
    This gives the policy the reference pose structure independent of global position.

    Returns:
        torch.Tensor: Flattened 3-point positions, shape (num_envs, num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_target_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point positions in reference anchor local frame for multiple future frames.

    All future frames use the current reference anchor orientation for canonicalization.

    Returns:
        torch.Tensor: Flattened multi-future local positions,
            shape (num_envs, num_future_frames * num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, command.num_future_frames, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_diff = (
        command.vr_3point_body_pos_w_multi_future - command.anchor_pos_w[:, None, None, :]
    )  # All w.r.t current anchor
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_target_compliant(
    env: ManagerBasedEnv,
    motion_command_name: str,
    force_command_name: str,
    zero_out_head_position: bool = False,
) -> torch.Tensor:
    """Get force-compliant VR 3-point targets in reference anchor local frame.

    Subtracts external force displacements (in robot-local frame) from the reference
    3-point positions to create compliant tracking targets.

    Args:
        motion_command_name: Name of the motion tracking command.
        force_command_name: Name of the force tracking command.
        zero_out_head_position: If True, zero out the head position component.

    Returns:
        torch.Tensor: Flattened compliant targets, shape (num_envs, num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: commands.ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    ext_force_disp_w = (
        force_command.last_force_applied * force_command.eef_stiffness_buf[:, :, None]
    )  # delta x external force world frame
    root_quat = command.robot_anchor_quat_w[:, None, :].repeat(
        1, len(command.cfg.vr_3point_body), 1
    )  # robot root pos not ref motion root pose as force is applied on robot, ref motion root frame is meaningless
    ext_force_disp_l = quat_apply(
        quat_inv(root_quat), ext_force_disp_w
    )  # delta x external force local frame
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    ref_3point_root -= ext_force_disp_l  # [E, 3, 3] -- only a single frame VR 3 point is returned

    # Conditionally zero out the head position (index -1)
    if zero_out_head_position:
        ref_3point_root[:, -1, :] = 0.0  # Head position

    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point body orientations in reference anchor local frame.

    Returns:
        torch.Tensor: Flattened 3-point quaternions, shape (num_envs, num_points * 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_quat = command.vr_3point_body_quat_w
    ref_3point_root = quat_mul(quat_inv(ref_root_quat), ref_3point_quat)
    return ref_3point_root.view(env.num_envs, -1)


def vr_wrists_local_pos_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get wrist positions in reference anchor local frame (wrists only, no head).

    Returns:
        torch.Tensor: Flattened 2-wrist positions, shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    ref_2point_diff = command.vr_3point_body_pos_w[:, :2] - command.anchor_pos_w[:, None, :]
    ref_2point_root = quat_apply(quat_inv(ref_root_quat), ref_2point_diff)
    return ref_2point_root.view(env.num_envs, -1)


def vr_wrists_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get wrist orientations in reference anchor local frame (wrists only, no head).

    Returns:
        torch.Tensor: Flattened 2-wrist quaternions, shape (num_envs, 8).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    ref_2point_quat = command.vr_3point_body_quat_w[:, :2]
    ref_2point_root = quat_mul(quat_inv(ref_root_quat), ref_2point_quat)
    return ref_2point_root.view(env.num_envs, -1)


def vr_head_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get head orientation in reference anchor local frame.

    Returns:
        torch.Tensor: Head quaternion in anchor frame, shape (num_envs, 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w
    head_quat = command.vr_3point_body_quat_w[:, 2]
    ref_head_quat_root = quat_mul(quat_inv(ref_root_quat), head_quat)
    return ref_head_quat_root.view(env.num_envs, -1)


def get_command_obs(env: ManagerBasedEnv, command_name: str, obs_name: str) -> torch.Tensor:
    """Get an arbitrary attribute from the tracking command by name.

    Generic accessor for command observations not covered by dedicated functions.

    Args:
        command_name: Name of the tracking command term.
        obs_name: Attribute name on the command object.

    Returns:
        torch.Tensor: The requested observation tensor.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return getattr(command, obs_name)


def smpl_pose_multi_future_select_joints(
    env: ManagerBasedEnv, command_name: str, joints_idx: list
) -> torch.Tensor:
    """Extract SMPL joint axis-angle poses for selected joints across future frames.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of SMPL joints to select.

    Returns:
        torch.Tensor: Selected joint poses, shape (..., len(joints_idx) * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    smpl_pose = command.smpl_pose_multi_future
    smpl_pose_selected = smpl_pose.view(*smpl_pose.shape[:-1], -1, 3)[..., joints_idx, :].view(
        *smpl_pose.shape[:-1], -1
    )
    return smpl_pose_selected


def joint_pos_multi_future_select_joints(
    env: ManagerBasedEnv, command_name: str, joints_idx: list, non_flatten=True
) -> torch.Tensor:
    """Extract reference joint positions for selected joints across future frames.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of joints to select from the full joint position vector.
        non_flatten: If True, return shape (num_envs, num_future_frames, len(joints_idx)).

    Returns:
        torch.Tensor: Selected joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos = command.joint_pos_multi_future
    # Flatten case: (num_envs, num_future_frames * num_joints)
    joint_pos_reshaped = joint_pos.view(env.num_envs, command.num_future_frames, -1)
    joint_pos_selected = joint_pos_reshaped[..., joints_idx]
    if not non_flatten:
        joint_pos_selected = joint_pos_selected.view(env.num_envs, -1)
    return joint_pos_selected


def joint_pos_multi_future_select_joints_for_smpl(
    env: ManagerBasedEnv, command_name: str, joints_idx: list
) -> torch.Tensor:
    """Extract reference joint positions for selected joints across SMPL-aligned future frames.

    Uses smpl_num_future_frames (which may differ from num_future_frames) for
    alignment with SMPL motion data.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of joints to select.

    Returns:
        torch.Tensor: Selected joint positions,
            shape (num_envs, smpl_num_future_frames, len(joints_idx)).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos = command.joint_pos_multi_future_for_smpl
    # Flatten case: (num_envs, num_future_frames * num_joints)
    joint_pos_reshaped = joint_pos.view(env.num_envs, command.smpl_num_future_frames, -1)
    joint_pos_selected = joint_pos_reshaped[..., joints_idx]
    return joint_pos_selected


def joint_pos_multi_future_select_joints_for_smpl(  # noqa: F811
    env: ManagerBasedEnv, command_name: str, joints_idx: list
) -> torch.Tensor:
    """Extract reference joint positions for selected joints across SMPL-aligned future frames.

    NOTE: This is a duplicate definition that shadows the previous one. Only this
    version is active at runtime.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of joints to select.

    Returns:
        torch.Tensor: Selected joint positions,
            shape (num_envs, smpl_num_future_frames, len(joints_idx)).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos = command.joint_pos_multi_future_for_smpl
    # Flatten case: (num_envs, num_future_frames * num_joints)
    joint_pos_reshaped = joint_pos.view(env.num_envs, command.smpl_num_future_frames, -1)
    joint_pos_selected = joint_pos_reshaped[..., joints_idx]
    return joint_pos_selected


def smpl_elbow_wrist_pose_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Extract SMPL elbow and wrist joint poses for multiple future frames.

    Selects indices 54:66 from the SMPL pose vector, corresponding to
    left elbow, left wrist, right elbow, right wrist (4 joints x 3 axis-angle).

    Returns:
        torch.Tensor: Elbow/wrist poses, shape (..., 12).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    smpl_pose = command.smpl_pose_multi_future
    elbow_wrist_pose = smpl_pose[..., 54:66]
    return elbow_wrist_pose


def smpl_root_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get SMPL root orientation relative to robot anchor as 6D rotation.

    Returns:
        torch.Tensor: 6D rotation representation, shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        None,
        command.smpl_root_quat_w,
    )
    # diff = quat_mul(command.smpl_root_quat_w, quat_inv(command.anchor_quat_w))
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def smpl_root_ori_b_mf(env: ManagerBasedEnv, command_name: str, non_flatten=True) -> torch.Tensor:
    """Get SMPL root orientation difference in body-local frame for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation differences.
    """
    # ZL: non-flatten set to true temporarily to match previous jobs, will be set to false in the future.
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        rot = command.smpl_root_quat_w_dif_l_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        rot = command.smpl_root_quat_w_dif_l_multi_future.reshape(env.num_envs, -1)
    return rot


def smpl_root_ori_refheading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=True
) -> torch.Tensor:
    """SMPL root orientation canonicalized by the first SMPL future frame's heading.

    Uses the heading of the first SMPL future frame instead of the robot's orientation
    for canonicalization.

    Returns:
        torch.Tensor: 6D rotation matrix representation,
            shape (num_envs, smpl_num_future_frames, 6) if non_flatten else
            (num_envs, smpl_num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.smpl_root_quat_w_dif_refheading_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        return command.smpl_root_quat_w_dif_refheading_multi_future.reshape(env.num_envs, -1)


def smpl_root_ori_heading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=True
) -> torch.Tensor:
    """SMPL root orientation canonicalized by robot heading (yaw) only.

    Uses the robot's heading (yaw) instead of full orientation for canonicalization,
    preserving the SMPL root's pitch/roll relative to gravity.

    Returns:
        torch.Tensor: 6D rotation matrix representation,
            shape (num_envs, smpl_num_future_frames, 6) if non_flatten else
            (num_envs, smpl_num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.smpl_root_quat_w_dif_heading_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        return command.smpl_root_quat_w_dif_heading_multi_future.reshape(env.num_envs, -1)


def smpl_joints_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get SMPL joint positions relative to SMPL root for multiple future frames.

    Canonicalizes joint positions using the first SMPL frame's root orientation.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, num_joints * 3).

    Returns:
        torch.Tensor: Root-relative SMPL joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_joints = command.smpl_joints_multi_future
    ref_root_quat = command.smpl_root_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, command.num_future_frames, ref_joints.shape[-2], 1
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)


def smpl_joints_multi_future_local(
    env: ManagerBasedEnv, command_name: str, non_flatten=False, joints_idx=None
) -> torch.Tensor:
    """Get SMPL joint positions relative to each frame's own root orientation.

    Unlike smpl_joints_multi_future which uses only the first frame's root,
    this version canonicalizes each future frame independently using its own
    root orientation. Optionally select a subset of joints.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, num_joints * 3).
        joints_idx: Optional list of joint indices to select.

    Returns:
        torch.Tensor: Per-frame root-relative SMPL joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    ref_joints = command.smpl_joints_multi_future
    ref_root_quat = command.smpl_root_quat_w_multi_future.unsqueeze(-2).repeat(
        1, 1, ref_joints.shape[-2], 1
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if joints_idx is not None:
        ref_joints_root = ref_joints_root[..., joints_idx, :]
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)


def smpl_joints_lower_multi_future_local(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get lower-body SMPL joint positions relative to each frame's own root orientation.

    Selects joints [0,1,2,4,5,7,8,10,11] (hips, knees, ankles) and canonicalizes
    each future frame using its own root orientation.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, 9 * 3).

    Returns:
        torch.Tensor: Per-frame root-relative lower-body SMPL joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    ref_joints = command.smpl_joints_multi_future[
        :, :, [0, 1, 2, 4, 5, 7, 8, 10, 11], :
    ]  # only first 12 joints (lower body)
    ref_root_quat = command.smpl_root_quat_w_multi_future.unsqueeze(-2).repeat(
        1, 1, ref_joints.shape[-2], 1
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)


def smpl_transl_z_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get SMPL root translation z-height for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, preserve per-frame dimension.

    Returns:
        torch.Tensor: Z-heights, shape (num_envs, smpl_num_future_frames) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_transl_z = command.smpl_transl_z_multi_future
    if non_flatten:
        return ref_transl_z
    else:
        return ref_transl_z.view(env.num_envs, -1)


def soma_joints_multi_future_local(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """SOMA skeleton joint positions canonicalized to each frame's root orientation.

    Mirrors smpl_joints_multi_future_local but uses SOMA skeleton (26 joints).
    Root quat has Y→Z up conversion and base rotation removal applied.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    ref_joints = command.soma_joints_multi_future  # (batch, num_frames, 26, 3)
    ref_root_quat = command.soma_root_quat_w_multi_future.unsqueeze(-2).repeat(
        1, 1, ref_joints.shape[-2], 1  # (batch, num_frames, 26, 4)
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)


def soma_root_ori_b_mf(env: ManagerBasedEnv, command_name: str, non_flatten=True):
    """SOMA root orientation relative to robot anchor, as 6D rotation matrix.

    Returns:
        (num_envs, soma_num_future_frames, 6) if non_flatten
        (num_envs, soma_num_future_frames * 6) otherwise
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        rot = command.soma_root_quat_w_dif_l_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        rot = command.soma_root_quat_w_dif_l_multi_future.reshape(env.num_envs, -1)
    return rot


def compliance(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get end-effector compliance (stiffness) scaled for observation input.

    Returns:
        torch.Tensor: Scaled stiffness values, shape (num_envs, num_eef).
    """
    command: commands.ForceTrackingCommand = env.command_manager.get_term(command_name)
    return command.eef_stiffness_buf * 10.0  # rescaling observation


def ext_forces(
    env: ManagerBasedEnv, force_command_name: str, motion_command_name: str
) -> torch.Tensor:
    """Get last-applied external forces in heading-local frame.

    Transforms the world-frame external forces into a heading-invariant frame
    by removing the robot's yaw rotation.

    Returns:
        torch.Tensor: Flattened forces, shape (num_envs, num_force_bodies * 3).
    """
    force_command: commands.ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    motion_command: commands.TrackingCommand = env.command_manager.get_term(motion_command_name)
    ext_force_w = force_command.last_force_applied
    root_quat = motion_command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, ext_force_w.shape[1], 1
    )
    deheaded_force = quat_apply_yaw(quat_inv(root_quat), ext_force_w)
    return deheaded_force.view(env.num_envs, -1)


# =============================================================================
# Body-only observation functions (for pre-trained action_transform_module)
# Uses get_body_joint_indices from joint_utils.py


def joint_pos_wo_hand(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    """Get joint positions excluding hand joints (29 DOF body only)."""
    asset = env.scene[asset_cfg.name]
    body_indices = joint_utils.get_body_joint_indices(asset)
    return asset.data.joint_pos[:, body_indices] - asset.data.default_joint_pos[:, body_indices]


def joint_vel_wo_hand(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    """Get joint velocities excluding hand joints (29 DOF body only)."""
    asset = env.scene[asset_cfg.name]
    body_indices = joint_utils.get_body_joint_indices(asset)
    return asset.data.joint_vel[:, body_indices] - asset.data.default_joint_vel[:, body_indices]


def last_action_wo_hand(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    """Get last actions excluding hand joints."""
    asset = env.scene[asset_cfg.name]
    body_indices = joint_utils.get_body_joint_indices(asset)
    return env.action_manager.action[:, body_indices]


def last_meta_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Get last meta action (policy output: latent residual + finger primitives).

    This returns the policy's output from the previous step, not the joint-level
    actions that were applied to the simulation. For a student policy using
    latent residual mode, this is typically:
    - 64 dims: latent residual (tokenizer space)
    - 2 dims: finger primitive actions (left + right hand)
    Total: 66 dims

    The buffer is stored on env by the ManagerEnvWrapper and initialized to zeros
    on reset.
    """
    if hasattr(env, "_last_meta_action"):
        return env._last_meta_action  # noqa: SLF001
    else:
        # Fallback: return zeros if buffer not initialized (shouldn't happen)
        meta_action_dim = 66  # Default: 64 latent + 2 primitives
        return torch.zeros(env.num_envs, meta_action_dim, dtype=torch.float32, device=env.device)


def ref_root_pos_future_b(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Get reference root positions for future frames in robot anchor frame.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * 3).

    Returns:
        torch.Tensor: Reference root positions,
            shape (num_envs, num_future_frames, 3) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_pos_future_w = command.anchor_pos_w_multi_future.view(
        command.num_envs, command.num_future_frames, -1
    )
    robot_root_pos_w = command.robot_anchor_pos_w.unsqueeze(1)
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = robot_root_quat_w.expand(-1, command.num_future_frames, -1)

    ref_root_pos_future_b = quat_apply_inverse(
        robot_root_quat_w, ref_root_pos_future_w - robot_root_pos_w
    )
    if flatten:
        return ref_root_pos_future_b.reshape(command.num_envs, -1)
    return ref_root_pos_future_b.reshape(command.num_envs, command.num_future_frames, -1)


def ref_root_ori_future_b(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Get reference root orientations for future frames in robot anchor frame as 6D rotation.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * 6).

    Returns:
        torch.Tensor: 6D rotation representations,
            shape (num_envs, num_future_frames, 6) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat_future_w = command.anchor_quat_w_multi_future.view(
        command.num_envs, command.num_future_frames, -1
    )
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = robot_root_quat_w.expand(-1, command.num_future_frames, -1)

    ref_root_quat_future_b = quat_mul(
        quat_conjugate(robot_root_quat_w),
        ref_root_quat_future_w,
    )
    ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
    ref_root_ori_future_b = ref_root_ori_future_b[:, :, :2, :]
    if flatten:
        return ref_root_ori_future_b.reshape(command.num_envs, -1)
    return ref_root_ori_future_b.reshape(command.num_envs, command.num_future_frames, -1)


def diff_body_pos_future_local(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Compute reference-minus-robot body position differences in their respective heading frames.

    Reference body positions are expressed in the reference motion's heading (yaw-only,
    z=0 projected) root frame; robot body positions are expressed in the robot's heading
    root frame. The difference captures the tracking error per body per future frame.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * num_bodies * 3).

    Returns:
        torch.Tensor: Position differences,
            shape (num_envs, num_future_frames, num_bodies * 3) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_body_pos_future_w = command.body_pos_w_multi_future.view(
        command.num_envs, command.num_future_frames, command.num_bodies, -1
    )

    ref_root_pos_w = command.anchor_pos_w.unsqueeze(1).unsqueeze(2)
    # shape: (num_envs, 1, 1, 3)
    ref_root_quat_w = command.anchor_quat_w.unsqueeze(1).unsqueeze(2)
    # shape: (num_envs, 1, 1, 4)
    ref_root_pos_w = ref_root_pos_w.clone()
    ref_root_pos_w[..., 2] = 0.0
    ref_root_quat_w = torch_transform.get_heading_q(ref_root_quat_w)
    ref_root_quat_w = ref_root_quat_w.expand(
        command.num_envs, command.num_future_frames, len(command.cfg.body_names), -1
    )

    robot_body_pos_w = command.robot_body_pos_w.view(command.num_envs, command.num_bodies, -1)

    robot_root_pos_w = command.robot_anchor_pos_w.unsqueeze(1)
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_pos_w = robot_root_pos_w.clone()
    robot_root_pos_w[..., 2] = 0.0
    robot_root_quat_w = torch_transform.get_heading_q(robot_root_quat_w)
    robot_root_quat_w = robot_root_quat_w.expand(command.num_envs, len(command.cfg.body_names), -1)

    robot_body_pos_local = quat_apply_inverse(
        robot_root_quat_w, robot_body_pos_w - robot_root_pos_w
    )
    ref_body_pos_future_local = quat_apply_inverse(
        ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w
    )

    diff = ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1)
    if flatten:
        return diff.reshape(command.num_envs, -1)
    return diff.reshape(command.num_envs, command.num_future_frames, -1)


def diff_body_ori_future_local(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Compute reference-minus-robot body orientation differences in their respective heading frames.

    Both reference and robot body orientations are first canonicalized by their respective
    heading (yaw-only) root frames, then the relative rotation between them is computed
    as a 6D rotation representation.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * num_bodies * 6).

    Returns:
        torch.Tensor: 6D orientation differences,
            shape (num_envs, num_future_frames, num_bodies * 6) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_body_quat_future_w = command.body_quat_w_multi_future.view(
        command.num_envs, command.num_future_frames, command.num_bodies, -1
    )

    ref_root_quat_w = command.anchor_quat_w.unsqueeze(1).unsqueeze(2)
    ref_root_quat_w = torch_transform.get_heading_q(ref_root_quat_w)
    ref_root_quat_w = ref_root_quat_w.expand(
        command.num_envs, command.num_future_frames, len(command.cfg.body_names), -1
    )

    robot_body_quat_w = command.robot_body_quat_w.view(command.num_envs, command.num_bodies, -1)

    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = torch_transform.get_heading_q(robot_root_quat_w)
    robot_root_quat_w = robot_root_quat_w.expand(command.num_envs, len(command.cfg.body_names), -1)

    robot_body_quat_local = quat_mul(
        quat_conjugate(robot_root_quat_w),
        robot_body_quat_w,
    ).unsqueeze(1)
    ref_body_quat_future_local = quat_mul(
        quat_conjugate(ref_root_quat_w),
        ref_body_quat_future_w,
    )

    diff_body_quat_future = quat_mul(
        quat_conjugate(robot_body_quat_local).expand_as(ref_body_quat_future_local),
        ref_body_quat_future_local,
    )
    diff_body_ori_future_local = matrix_from_quat(diff_body_quat_future)
    diff = diff_body_ori_future_local[:, :, :, :2, :]
    if flatten:
        return diff.reshape(command.num_envs, -1)
    return diff.reshape(command.num_envs, command.num_future_frames, -1)


def diff_body_lin_vel_future_local(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Compute reference-minus-robot body linear velocity differences in heading frames.

    Both reference and robot velocities are expressed in their respective heading
    (yaw-only) root frames. Output is clamped to [-25, 25] for numerical stability.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * num_bodies * 3).

    Returns:
        torch.Tensor: Velocity differences,
            shape (num_envs, num_future_frames, num_bodies * 3) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_body_lin_vel_future_w = command.motion_lib.get_body_lin_vel_w(
        command.future_motion_ids, command.future_time_steps
    ).view(command.num_envs, command.num_future_frames, -1, 3)

    ref_root_quat_w = command.anchor_quat_w.unsqueeze(1).unsqueeze(2)
    # shape: (num_envs, 1, 1, 4)
    ref_root_quat_w = torch_transform.get_heading_q(ref_root_quat_w)
    ref_root_quat_w = ref_root_quat_w.expand(
        command.num_envs, command.num_future_frames, command.num_bodies, -1
    )
    ref_body_lin_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_lin_vel_future_w)

    robot_body_lin_vel_w = command.robot_body_lin_vel_w
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = torch_transform.get_heading_q(robot_root_quat_w).expand(
        command.num_envs, command.num_bodies, -1
    )
    robot_body_lin_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_lin_vel_w)

    diff = ref_body_lin_vel_future_local - robot_body_lin_vel_local.unsqueeze(1)
    diff.clamp_(min=-25, max=25)
    if flatten:
        return diff.reshape(command.num_envs, -1)
    return diff.reshape(command.num_envs, command.num_future_frames, -1)


def diff_body_ang_vel_future_local(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Compute reference-minus-robot body angular velocity differences in heading frames.

    Both reference and robot angular velocities are expressed in their respective heading
    (yaw-only) root frames. Output is clamped to [-25, 25] for numerical stability.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * num_bodies * 3).

    Returns:
        torch.Tensor: Angular velocity differences,
            shape (num_envs, num_future_frames, num_bodies * 3) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_body_ang_vel_future_w = command.motion_lib.get_body_ang_vel_w(
        command.future_motion_ids, command.future_time_steps
    ).view(command.num_envs, command.num_future_frames, -1, 3)

    ref_root_quat_w = command.anchor_quat_w.unsqueeze(1).unsqueeze(2)
    # shape: (num_envs, 1, 1, 4)
    ref_root_quat_w = torch_transform.get_heading_q(ref_root_quat_w)
    ref_root_quat_w = ref_root_quat_w.expand(
        command.num_envs, command.num_future_frames, command.num_bodies, -1
    )
    ref_body_ang_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_ang_vel_future_w)

    robot_body_ang_vel_w = command.robot_body_ang_vel_w
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = torch_transform.get_heading_q(robot_root_quat_w).expand(
        command.num_envs, command.num_bodies, -1
    )
    robot_body_ang_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_ang_vel_w)

    diff = ref_body_ang_vel_future_local - robot_body_ang_vel_local.unsqueeze(1)
    diff.clamp_(min=-25, max=25)
    if flatten:
        return diff.reshape(command.num_envs, -1)
    return diff.reshape(command.num_envs, command.num_future_frames, -1)


def height_map(env: ManagerBasedEnv, command_name, random=False) -> torch.Tensor:
    """Get height map observation from the environment."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    if not hasattr(command, "scan_dot_pos_w"):
        # Height map disabled - return zeros with expected shape
        n = int(command.cfg.height_map_size / command.cfg.height_map_resolution) + 1
        return torch.zeros(command.num_envs, n, n, 3, device=command.device)

    if not hasattr(command, "scan_dot_pos_w"):
        # Height map disabled - return zeros with expected shape
        n = int(command.cfg.height_map_size / command.cfg.height_map_resolution) + 1
        return torch.zeros(command.num_envs, n, n, 3, device=command.device)

    robot_root_pos_w, robot_root_quat_w = (
        command.robot.data.root_pos_w,
        command.robot.data.root_quat_w,
    )
    scan_dot_pos_w = command.scan_dot_pos_w

    robot_root_quat_w_yaw = (
        torch_transform.get_heading_q(robot_root_quat_w)
        .unsqueeze(1)
        .unsqueeze(2)
        .expand(-1, command.num_rays_x, command.num_rays_y, -1)
    )
    obs = quat_apply_inverse(
        robot_root_quat_w_yaw, scan_dot_pos_w - robot_root_pos_w.unsqueeze(1).unsqueeze(2)
    )
    obs.nan_to_num_(0.0)
    if random:
        obs = torch.rand_like(obs)
    return obs


def height_map_flat(env: "ManagerBasedEnv", command_name, random=False) -> torch.Tensor:
    """Get flattened height map observation for use as encoder input.
    Returns shape (B, 363) from (B, 11, 11, 3)."""
    hm = height_map(env, command_name, random)
    return hm.reshape(hm.shape[0], -1)


def height_map_z_flat(env: "ManagerBasedEnv", command_name, random=False) -> torch.Tensor:
    """Get flattened height-only (z) map for decoder prediction target.

    Returns shape ``(B, N*N)`` where N = height_map_size / height_map_resolution + 1
    (default 11 → 121).  Values are z-heights in the robot body frame.
    """
    hm = height_map(env, command_name, random)  # (B, Nx, Ny, 3)
    return hm[..., 2].reshape(hm.shape[0], -1)  # (B, Nx*Ny)


def depth_image(
    env: "ManagerBasedEnv",
    camera_name: str = "depth_camera",
    depth_range: tuple = (0.0, 2.5),
    normalize: bool = True,
) -> torch.Tensor:
    """Get depth image from a RayCasterCamera sensor, returned as (B, 1, H, W).

    If the sensor has ``distance_to_image_plane_processed`` (i.e. DepthCamera with
    crop/normalize pipeline), uses that directly. Otherwise falls back to raw output.
    """
    camera = env.scene[camera_name]
    if "distance_to_image_plane_processed" in camera.data.output:
        # Already cropped + normalized by the sensor
        depth = camera.data.output["distance_to_image_plane_processed"]  # (B, H', W', 1)
        depth = depth.squeeze(-1)  # (B, H', W')
    else:
        depth = camera.data.output["distance_to_image_plane"]  # (B, H, W, 1)
        depth = depth.squeeze(-1)  # (B, H, W)
        depth = depth.clamp(depth_range[0], depth_range[1])
        if normalize:
            depth = (depth - depth_range[0]) / (depth_range[1] - depth_range[0])
    return depth.unsqueeze(1)  # (B, 1, H, W)


def depth_image_flat(
    env: "ManagerBasedEnv",
    camera_name: str = "depth_camera",
    depth_range: tuple = (0.0, 2.5),
    normalize: bool = True,
) -> torch.Tensor:
    """Get flattened single-frame depth image. Returns (B, H*W)."""
    di = depth_image(env, camera_name, depth_range, normalize)
    return di.reshape(di.shape[0], -1)


def depth_image_history(
    env: "ManagerBasedEnv",
    camera_name: str = "depth_camera",
    depth_range: tuple = (0.0, 2.5),
    normalize: bool = True,
    history_skip_frames: int = 5,
    num_output_frames: int = 8,
    obs_resize: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Get temporal depth image sequence from the sensor's history buffer.

    Reads from ``distance_to_image_plane_history`` maintained by DepthCamera
    at the sensor level. The history frames are already cropped + normalized
    by the sensor pipeline.

    Args:
        obs_resize: If set, bilinear resize each frame to (H, W) before flattening.
            Use (16, 16) for legacy checkpoints trained on 16x16 depth.
            None means use sensor output size directly (e.g. 18x32).

    Returns shape (B, num_output_frames * H_out * W_out) flattened for tokenizer.
    """
    camera = env.scene[camera_name]
    # Sensor-level history: (B, history_length, H', W', 1) — already cropped + normalized
    history = camera.data.output["distance_to_image_plane_history"]
    history = history.squeeze(-1)  # (B, history_length, H', W')
    history_length = history.shape[1]

    # Sample frames: take num_output_frames evenly spaced from the end
    # history[:, -1] is the newest frame (matching InstinctLab convention)
    indices = [history_length - 1 - i * history_skip_frames for i in range(num_output_frames)]
    indices.reverse()  # oldest first
    sampled = history[:, indices]  # (B, num_output_frames, H', W')

    # Optional resize for legacy checkpoint compatibility
    if obs_resize is not None and (sampled.shape[2], sampled.shape[3]) != tuple(obs_resize):
        import torch.nn.functional as F

        B, T, H, W = sampled.shape
        sampled = sampled.reshape(B * T, 1, H, W)
        sampled = F.interpolate(sampled, size=obs_resize, mode="bilinear", align_corners=False)
        sampled = sampled.reshape(B, T, obs_resize[0], obs_resize[1])

    return sampled.reshape(sampled.shape[0], -1)


def residual_joint_pos_action(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Compute the action-space representation of reference joint positions.

    Converts the reference motion's joint positions into the normalized action space
    using the action manager's offset and scale.

    Returns:
        torch.Tensor: Normalized residual actions, shape (num_envs, num_action_joints).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    action_manager = env.action_manager.get_term("joint_pos")
    action_offset = action_manager._offset  # noqa: SLF001
    action_scale = action_manager._scale  # noqa: SLF001

    motion_joint_pos = command.joint_pos
    # motion_joint_names = command.robot.joint_names
    # action_joint_names = action_manager._joint_names
    action_joint_pos = motion_joint_pos[:, action_manager._joint_ids]  # noqa: SLF001
    residual_action = (action_joint_pos - action_offset) / action_scale
    return residual_action


# =============================================================================
# Task stage observation (for staged training)


def get_task_stage(env: ManagerBasedEnv) -> torch.Tensor:
    """Get current task stage."""
    if hasattr(env, "task_stage"):
        return env.task_stage.float().unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, dtype=torch.float, device=env.device)


def get_target_object_pos_in_base_frame(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Get object position in robot base frame."""
    if (
        asset_cfg.name == "active_articulation"
        or (asset_cfg.name == "object" and "object" not in env.scene.rigid_objects)
    ):
        command: commands.TrackingCommand = env.command_manager.get_term("motion")
        obj_pos_w, _, _, _, _ = utils.get_target_object_current_state(env, command)
    else:
        asset = env.scene[asset_cfg.name]
        obj_pos_w = asset.data.root_pos_w
    robot = env.scene["robot"]

    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # Transform object position to robot base frame
    obj_pos_rel = obj_pos_w - robot_pos_w
    obj_pos_b = quat_apply(quat_inv(robot_quat_w), obj_pos_rel)

    return obj_pos_b


def active_articulation_joint_state(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    max_joints: int = 4,
    velocity_scale: float = 0.1,
    normalize_by_joint_limits: bool = False,
) -> torch.Tensor:
    """Return padded ``q``, scaled ``qdot``, reference error, and a validity mask.

    This term is intentionally opt-in. Its fixed width lets a versioned policy
    mix mechanisms with different joint counts without enabling RP2's padded
    world-object observation path.
    """
    if max_joints <= 0:
        raise ValueError(f"max_joints must be positive, got {max_joints}")
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos, joint_vel, reference_pos, _, joint_valid = (
        utils.get_active_articulation_joint_tracking_state(env, command)
    )
    num_joints = joint_pos.shape[1]
    if num_joints > max_joints:
        raise ValueError(
            "Active articulation exceeds the configured observation capacity: "
            f"usd={num_joints}, max_joints={max_joints}"
        )

    output = torch.zeros(
        env.num_envs,
        4,
        max_joints,
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    )
    joint_valid_float = joint_valid.to(dtype=joint_pos.dtype)
    if normalize_by_joint_limits and num_joints:
        articulation = utils.get_active_articulation(env)
        limits = articulation.data.soft_joint_pos_limits
        center = 0.5 * (limits[..., 0] + limits[..., 1])
        half_range = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1e-3)
        normalized_pos = (joint_pos - center) / half_range
        normalized_vel = joint_vel / half_range
        normalized_error = (reference_pos - joint_pos) / half_range
    else:
        normalized_pos = joint_pos
        normalized_vel = joint_vel
        normalized_error = reference_pos - joint_pos
    output[:, 0, :num_joints] = normalized_pos * joint_valid_float
    output[:, 1, :num_joints] = (
        normalized_vel * float(velocity_scale) * joint_valid_float
    )
    output[:, 2, :num_joints] = normalized_error * joint_valid_float
    output[:, 3, :num_joints] = joint_valid_float
    return output.flatten(start_dim=1)


def active_articulation_joint_future(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    max_joints: int = 4,
    normalize_by_joint_limits: bool = False,
) -> torch.Tensor:
    """Padded future joint targets relative to the current simulated joint state."""
    if max_joints <= 0:
        raise ValueError(f"max_joints must be positive, got {max_joints}")
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos, _, _, _, _ = utils.get_active_articulation_joint_tracking_state(
        env, command
    )
    future_pos, future_valid = utils.get_active_articulation_joint_future_reference(
        env, command
    )
    num_joints = joint_pos.shape[1]
    if num_joints > max_joints:
        raise ValueError(
            "Active articulation exceeds the configured future observation capacity: "
            f"usd={num_joints}, max_joints={max_joints}"
        )
    output = torch.zeros(
        env.num_envs,
        command.num_future_frames,
        max_joints,
        dtype=future_pos.dtype if future_pos.numel() else torch.float32,
        device=env.device,
    )
    if num_joints:
        future_delta = future_pos - joint_pos[:, None, :]
        if normalize_by_joint_limits:
            articulation = utils.get_active_articulation(env)
            limits = articulation.data.soft_joint_pos_limits
            half_range = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1e-3)
            future_delta = future_delta / half_range[:, None, :]
        output[:, :, :num_joints] = future_delta * future_valid.to(
            dtype=future_pos.dtype
        )
    return output.flatten(start_dim=1)


def _get_hand_pose_w_from_transformer_or_robot(env: ManagerBasedEnv, transformer):
    cfg = getattr(env.cfg, "config", {}) if hasattr(env, "cfg") else {}
    hand_frame_links = list(cfg.get("object_to_hand_frame_links", []))
    robot = env.scene["robot"]

    if hand_frame_links:
        body_names = robot.body_names
        indices = [body_names.index(link) for link in hand_frame_links if link in body_names]
        if indices:
            hand_pos_w = robot.data.body_pos_w[:, indices, :].mean(dim=1)
            hand_quat_w = robot.data.body_quat_w[:, indices[0], :]
            return hand_pos_w, hand_quat_w

    if hasattr(transformer.data, "target_pos_w") and hasattr(transformer.data, "target_quat_w"):
        target_pos_w = transformer.data.target_pos_w
        target_quat_w = transformer.data.target_quat_w
        if target_pos_w.shape[1] > 0:
            return target_pos_w.mean(dim=1), target_quat_w[:, 0, :]

    return None, None


def _get_hand_pose_in_object_frame(transformer) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate a hand frame in object coordinates from the configured target links."""
    target_pos_in_obj = transformer.data.target_pos_source
    if target_pos_in_obj.shape[1] == 0:
        raise ValueError("object_to_hand_frame_transformer has no target frames configured")

    hand_pos_in_obj = target_pos_in_obj.mean(dim=1)
    hand_quat_in_obj = transformer.data.target_quat_source[:, 0, :]
    return hand_pos_in_obj, hand_quat_in_obj


def get_hand_object_transform(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get transform from the configured hand reference frame to the object."""
    transformer = env.scene[asset_cfg.name]
    if "object" not in env.scene.rigid_objects:
        hand_pos_w, hand_quat_w = _get_hand_pose_w_from_transformer_or_robot(env, transformer)
        if hand_pos_w is not None:
            command: commands.TrackingCommand = env.command_manager.get_term("motion")
            object_pos_w, object_quat_w, _, _, _ = utils.get_target_object_current_state(
                env, command
            )
            object_pos_in_hand = quat_apply(quat_inv(hand_quat_w), object_pos_w - hand_pos_w)
            object_quat_in_hand = quat_mul(quat_inv(hand_quat_w), object_quat_w)
            return torch.cat([object_pos_in_hand, object_quat_in_hand], dim=-1)

    hand_pos_in_obj, hand_quat_in_obj = _get_hand_pose_in_object_frame(transformer)

    # Invert transform: object position in hand frame
    object_pos_in_hand = quat_apply(quat_inv(hand_quat_in_obj), -hand_pos_in_obj)
    object_quat_in_hand = quat_inv(hand_quat_in_obj)

    return torch.cat([object_pos_in_hand, object_quat_in_hand], dim=-1)


def hand_object_transform_6d(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get transform from hand to object with 6D rotation.

    Same as get_hand_object_transform but returns 6D rotation representation
    (first two columns of rotation matrix) instead of quaternion.

    Returns:
        torch.Tensor: Object pose in hand frame, shape (num_envs, 9) = 3 pos + 6 ori
    """
    transformer = env.scene[asset_cfg.name]
    if "object" not in env.scene.rigid_objects:
        hand_pos_w, hand_quat_w = _get_hand_pose_w_from_transformer_or_robot(env, transformer)
        if hand_pos_w is not None:
            command: commands.TrackingCommand = env.command_manager.get_term("motion")
            object_pos_w, object_quat_w, _, _, _ = utils.get_target_object_current_state(
                env, command
            )
            object_pos_in_hand = quat_apply(quat_inv(hand_quat_w), object_pos_w - hand_pos_w)
            object_quat_in_hand = quat_mul(quat_inv(hand_quat_w), object_quat_w)
            mat = matrix_from_quat(object_quat_in_hand)
            ori_6d = mat[..., :2].reshape(mat.shape[0], -1)
            return torch.cat([object_pos_in_hand, ori_6d], dim=-1)

    hand_pos_in_obj, hand_quat_in_obj = _get_hand_pose_in_object_frame(transformer)

    # Invert transform: object position in hand frame
    object_pos_in_hand = quat_apply(quat_inv(hand_quat_in_obj), -hand_pos_in_obj)
    object_quat_in_hand = quat_inv(hand_quat_in_obj)

    # Convert quaternion to 6D rotation (first 2 columns of rotation matrix)
    mat = matrix_from_quat(object_quat_in_hand)  # (num_envs, 3, 3)
    ori_6d = mat[..., :2].reshape(mat.shape[0], -1)  # (num_envs, 6)

    return torch.cat([object_pos_in_hand, ori_6d], dim=-1)  # (num_envs, 9)


def get_finger_tips_contact_force(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "motion",
    hand: str = "right_hand",
) -> torch.Tensor:
    """Get contact forces on the configured hand contact links.

    The sensor is configured with:
    - prim_path: Object (1 body)
    - filter_prim_paths_expr: configured hand contact links in a fixed order

    force_matrix_w shape: (N, 1, num_links, 3)
    - Dim 1: object (prim_path)
    - Dim 2: configured hand links in filter_prim_paths_expr order
    """
    command = env.command_manager.get_term(command_name)
    force = utils.get_target_object_contact_forces(
        env,
        command,
        hand=hand,
        fallback_sensor_name=sensor_cfg.name,
    )
    if force is None:
        return torch.zeros(env.num_envs, 0, device=env.device)

    # Shape: (N, num_objects, num_links, 3). Sum across tracked object bodies
    # so the observation shape remains num_links * 3 for single and multi-object scenes.
    if force.ndim == 4:
        force = force.sum(dim=1)
    return force.reshape(env.num_envs, -1)


def bilateral_hand_contact_force_summary(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_to_left_hand_contact_sensor"),
    command_name: str = "motion",
) -> torch.Tensor:
    """Return a fixed-width bilateral contact observation.

    The legacy YAM observation exposes four right-hand contact-force vectors
    (12D).  Represent each hand with its thumb vector and one aggregate vector
    for the remaining finger/palm contacts, then concatenate right and left.
    This preserves the 12D network contract without cancelling opposing thumb
    and support-finger forces in a valid grasp.
    """
    command = env.command_manager.get_term(command_name)

    def hand_force_pair(hand: str, fallback_sensor_name: str) -> torch.Tensor:
        force = utils.get_target_object_contact_forces(
            env,
            command,
            hand=hand,
            fallback_sensor_name=fallback_sensor_name,
        )
        if force is None:
            return torch.zeros(env.num_envs, 6, device=env.device)
        if force.shape[0] != env.num_envs or force.shape[-1] != 3:
            raise ValueError(
                f"Unexpected {hand} contact-force shape {tuple(force.shape)}; "
                f"expected [num_envs, ..., 3]"
            )
        force = force.reshape(env.num_envs, -1, 3)
        thumb_force = force[:, 0]
        if force.shape[1] > 1:
            support_force = force[:, 1:].sum(dim=1)
        else:
            support_force = torch.zeros_like(thumb_force)
        return torch.cat([thumb_force, support_force], dim=-1)

    right_force = hand_force_pair("right_hand", sensor_cfg.name)
    left_force = hand_force_pair("left_hand", left_sensor_cfg.name)
    return torch.cat([right_force, left_force], dim=-1)


def grab_contact_flag(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get contact flag indicating whether current frame should have contact.

    Returns a tensor of shape (num_envs, 1) where 1.0 indicates contact should occur
    at the current frame, 0.0 indicates no contact expected.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    if (
        command._motion_contact_flags is None  # noqa: SLF001
        or len(command._motion_contact_flags) == 0  # noqa: SLF001
    ):  # noqa: SLF001
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)

    frame_indices = (command.motion_start_time_steps + command.time_steps).long()  # (num_envs,)
    motion_ids = command.motion_ids  # (num_envs,)
    contact_flags = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # Process each unique motion_id with vectorized operations
    for motion_id in torch.unique(motion_ids):
        motion_id_int = motion_id.item()
        assert (
            motion_id_int in command._motion_contact_flags  # noqa: SLF001
        ), f"Motion ID {motion_id_int} not found in contact flags"
        env_mask = motion_ids == motion_id
        env_indices = torch.where(env_mask)[0]
        env_frame_indices = torch.clamp(
            frame_indices[env_indices],
            0,
            command._motion_contact_flags[motion_id_int].shape[0] - 1,  # noqa: SLF001
        )
        contact_flags[env_indices] = command._motion_contact_flags[motion_id_int][  # noqa: SLF001
            env_frame_indices
        ].float()
    return contact_flags.unsqueeze(-1)


def get_tiled_camera_image(
    env: ManagerBasedEnv,
    camera_cfg: SceneEntityCfg,
    normalize: bool = True,
    normalize_mean: list[float] = [0.485, 0.456, 0.406],  # ImageNet mean (RGB)
    normalize_std: list[float] = [0.229, 0.224, 0.225],  # ImageNet std (RGB)
    debug_visualize: bool = False,  # Enable real-time visualization with cv2
    debug_env_idx: int = 0,  # Which environment's image to visualize
    debug_show_predicted: bool = False,  # Also show predicted object position (in red)
) -> torch.Tensor:
    """Get RGB image from tiled camera sensor.

    Returns normalized RGB image of shape (num_envs, H, W, 3) - NOT flattened.
    Image values can be normalized with configurable mean and std (default: ImageNet stats).

    Args:
        env: The environment object
        camera_cfg: Camera configuration with sensor name
        normalize: Whether to apply mean/std normalization (default: True)
        normalize_mean: RGB mean values for normalization (default: ImageNet [0.485, 0.456, 0.406])
        normalize_std: RGB std values for normalization (default: ImageNet [0.229, 0.224, 0.225])
        debug_visualize: Whether to display the image in real-time using cv2 (default: False)
        debug_env_idx: Which environment's image to visualize (default: 0)
        debug_show_predicted: Whether to also show predicted object position in red (default: False)
        debug_show_predicted: Whether to also show predicted object position in red (default: False)

    Returns:
        RGB image tensor of shape (num_envs, H, W, 3) - channels last format
    """
    camera = env.scene[camera_cfg.name]

    # Get RGB image from camera - shape is (num_envs, H, W, 4) with RGBA
    rgb_image = camera.data.output["rgb"]

    # Take only RGB channels (drop alpha if present)
    if rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[..., :3]

    # Debug visualization (before normalization)
    if debug_visualize:
        utils.debug_visualize_object_projection(
            env, camera, rgb_image, debug_env_idx, show_predicted=debug_show_predicted
        )

    # Convert to float if needed
    if rgb_image.dtype != torch.float:
        rgb_image = rgb_image.float()

    # Normalize to [0, 1] if in [0, 255] range
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0

    # Apply mean/std normalization if enabled
    if normalize:
        mean = torch.tensor(normalize_mean, device=rgb_image.device, dtype=rgb_image.dtype)
        std = torch.tensor(normalize_std, device=rgb_image.device, dtype=rgb_image.dtype)
        rgb_image = (rgb_image - mean) / std

    # Return shape: (num_envs, H, W, 3) - do NOT flatten
    # The vision encoder will handle permuting to (B, C, H, W) if needed
    return rgb_image


def get_finger_tips_contact_force_dual_arm(
    env: ManagerBasedEnv,
    right_hand_sensor_cfg: SceneEntityCfg,
    left_hand_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get contact forces on finger tips (thumb, index, middle, palm).

    The sensor is configured with:
    - prim_path: Object (1 body)
    - filter_prim_paths_expr: [thumb_2, index_1, middle_1, palm] (4 bodies in fixed order)

    force_matrix_w shape: (N, 1, 4, 3)
    - Dim 1: object (prim_path)
    - Dim 2: finger tips (filter_prim_paths_expr, indices 0-3 in order: thumb, index, middle, palm)
    """
    right_hand_sensor = env.scene[right_hand_sensor_cfg.name]
    left_hand_sensor = env.scene[left_hand_sensor_cfg.name]
    right_force = right_hand_sensor.data.force_matrix_w
    left_force = left_hand_sensor.data.force_matrix_w

    if right_force.ndim == 4:
        right_force = right_force.sum(dim=1)
    if left_force.ndim == 4:
        left_force = left_force.sum(dim=1)

    return torch.cat(
        [
            right_force.reshape(env.num_envs, -1),
            left_force.reshape(env.num_envs, -1),
        ],
        dim=1,
    )


def grab_contact_flag_dual_arm(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get hand contact flag indicating which hand(s) should have contact at current frame.

    Returns a tensor of shape (num_envs, 2) where:
    - [:, 0] indicates left hand contact (1.0 = should contact, 0.0 = no contact expected)
    - [:, 1] indicates right hand contact (1.0 = should contact, 0.0 = no contact expected)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    # Use vectorized hand contact tensor from command
    hand_contact_tensor = command.get_hand_contact_tensor()  # [num_envs, 2] bool
    return hand_contact_tensor.float()  # Convert to float: [num_envs, 2]
