"""Reward functions for the manager-based RL environment MDP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    quat_rotate_inverse,
)
import torch
import numpy as np

from gear_sonic.envs.manager_env.mdp.commands import (
    ForceTrackingCommand,
    TrackingCommand,
    _get_body_indexes,
)
from gear_sonic.envs.manager_env.mdp import utils
from gear_sonic.envs.manager_env.mdp.observations import get_hand_object_transform
from gear_sonic.rp2.discrete_contact import (
    CONTACT_BODY_GROUPS,
    CONTACT_ENTITY_CLASSES,
    canonical_hand_contact_indices,
)
from gear_sonic.rp2.reward_kernels import tolerance_tracking_score

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    tracking_anchor_pos = None
    tracking_anchor_ori = None
    tracking_relative_body_pos = None
    tracking_relative_body_ori = None
    tracking_relative_body_ori_weighted = None
    tracking_body_linvel = None
    tracking_body_angvel = None
    action_rate_l2 = None
    joint_limit = None
    undesired_contacts = None
    undesired_contacts_no_hands = None
    undesired_contacts_no_ankle_hand = None
    undesired_contacts_no_ankle_yam = None
    tracking_body_pos = None
    tracking_body_ori = None
    tracking_vr_3point_global = None
    tracking_vr_3point_local = None
    tracking_vr_3point_force = None
    tracking_vr_2wrists_ori_tight = None
    tracking_vr_2wrists_local_ori = None
    tracking_head_local_ori = None
    feet_contact_duration = None
    jitter_penalty = None
    energy_consumption = None
    anti_shake_ang_vel = None
    feet_contact_requirement = None
    feet_contact_requirement_motion_lib = None
    tracking_vr_5point_local = None
    motion_5point_local_pos = None
    feet_acc = None
    # HOI manipulation rewards
    hand_fingers_object_distance = None
    grasp_finger_direction = None
    grasp_finger_direction_left = None
    grasp_reward = None
    grasp_reward_left = None
    grasp_contact_center_right = None
    grasp_contact_center_left = None
    lift_reward = None
    lift_z_elevation = None
    hand_table_contact_penalty = None
    # Object motion tracking reward (OmniGrasp-style r_t^obj)
    object_tracking_reward = None
    # Mask-aware active-entity object tracking (replaces the term above in ablation C).
    active_entity_object_tracking = None
    rp2_source_relation_tracking = None
    active_articulation_joint_tracking = None
    canonical_required_nonhand_contact = None
    canonical_forbidden_full_body_contact = None
    rp2_support_stability_penalty = None
    rp2_soma_object_relation_reward = None
    rp2_soma_wrist_object_relation_reward = None
    object_lift_contact_reward = None
    # Finger primitive action limit penalty
    finger_primitive_limit = None
    # Meta action (token + residual) smoothness penalty
    meta_action_rate_l2 = None
    # Full latent (decoder input) smoothness penalty
    full_latent_rate_l2 = None
    # Contact-transition-only meta action smoothness penalty
    rp2_contact_phase_meta_action_rate_l2 = None
    # Early termination penalty (penalizes non-timeout terminations)
    is_terminated = None
    # Wrist alignment reward (encourage wrist to face object during approach)
    wrist_look_at_object = None
    # Wrist smoothness penalty around contact transition
    wrist_smoothness_at_contact = None
    # Approach behavior rewards (sim2real: prevent swatting, keep hand open)
    approach_velocity_penalty = None
    open_hand_until_close = None
    # Penalize a closed hand primitive while that hand has no reference contact.
    open_hand_unless_reference_contact_right = None
    open_hand_unless_reference_contact_left = None
    upright_penalty = None
    # Foot slippage penalty (penalize foot velocity while in ground contact)
    foot_slippage_penalty = None
    # Source-neutral, ContactV4-inspired multiplier on existing tracking terms.
    contact_first_grasp_factor = None


def _finger_contact_force_magnitude(sensor: ContactSensor) -> torch.Tensor:
    """Return contact force magnitudes aggregated over source object bodies."""
    force_magnitude = torch.norm(sensor.data.force_matrix_w, dim=-1)
    if force_magnitude.ndim == 3:
        return force_magnitude.sum(dim=1)
    return force_magnitude


def tracking_anchor_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor position tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) position to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel. Smaller values produce
            sharper falloff and stricter tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    sq_dist = (diff * diff).sum(dim=-1)
    return torch.exp(-sq_dist / (std * std))


def tracking_anchor_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor orientation tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) orientation to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return torch.exp(-angular_err.square() / (std * std))


def upright_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_name: str | None = None,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize tilt of bodies away from upright.

    Compute the squared magnitude of the x/y components of the gravity vector
    in each body's local frame, summed across all specified bodies. When a body
    is perfectly upright the local gravity is [0, 0, -1] and the penalty is 0.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        body_name: Single body name (for backwards compatibility).
        body_names: List of body names. If both are None, defaults to ["pelvis"].

    Returns:
        Penalty tensor of shape (num_envs,). Zero when upright, positive when tilted.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]

    if body_names is None:
        body_names = [body_name] if body_name else ["pelvis"]

    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for name in body_names:
        body_idx = robot.body_names.index(name)
        body_quat = robot.data.body_quat_w[:, body_idx]
        g_local = quat_apply(quat_inv(body_quat), command.down_dir)
        total_penalty += g_local[:, 0] ** 2 + g_local[:, 1] ** 2

    return total_penalty


def tracking_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body positions to match the reference motion. The reward is
    the mean squared distance across all tracked bodies, passed through an exponential.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_vr_3point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 3-point tracking reward in world frame using a Gaussian kernel.

    Encourages the robot's 3 VR tracking points (typically left wrist, right wrist,
    head) to match their reference positions in world frame.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_diff = command.robot_vr_3point_pos_w - command.vr_3point_body_pos_w
    per_point_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_point_err.mean(dim=-1) / (std * std))


def tracking_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in world frame.

    Measure the orientation error of 2 wrist bodies against the reference motion,
    similar to tracking_relative_body_ori_error but restricted to wrist links.

    NOTE: The rigid extension defined in vr_3point_body_offset can be skipped for
    orientation error since it does not affect rotations.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_local_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in the anchor's local frame.

    Transform both reference and robot wrist orientations into the anchor (root)
    frame before computing the angular error. This makes the reward invariant to
    global root orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    body_indexes = _get_body_indexes(command, body_names)
    num_bodies = len(body_indexes)

    # reference motion
    ref_wrist_quat_w = command.body_quat_w[:, body_indexes]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, num_bodies, 1)
    ref_wrist_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_wrist_quat_w)

    # robot
    robot_wrist_quat_w = command.robot_body_quat_w[:, body_indexes]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, num_bodies, 1
    )
    robot_wrist_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_wrist_quat_w)

    error = quat_error_magnitude(ref_wrist_quat_local, robot_wrist_quat_local) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def tracking_local_head_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute head orientation tracking reward in the anchor's local frame.

    Transform the head (torso_link) orientation into the anchor's local frame for
    both the reference motion and the robot, then compute the angular error. This
    encourages the robot to match the head-to-root relative orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get the head body index (torso_link)
    head_body_names = ["torso_link"]
    body_indexes = _get_body_indexes(command, head_body_names)

    # reference motion: head orientation in world frame, transformed to anchor's local frame
    ref_head_quat_w = command.body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4)
    ref_head_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_head_quat_w)

    # robot: head orientation in world frame, transformed to anchor's local frame
    robot_head_quat_w = command.robot_body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4)
    robot_head_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_head_quat_w)

    error = quat_error_magnitude(ref_head_quat_local, robot_head_quat_local) ** 2
    return torch.exp(-error.squeeze(-1) / std**2)


def tracking_local_vr_3point_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    point_weights: list[float] | None = None,
):
    """Compute VR 3-point tracking reward in the anchor's local frame.

    Transform tracking points into the anchor (root) local frame before computing
    position error, making the reward invariant to global root position/orientation.
    Supports optional per-point weighting.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        point_weights: Optional weights for each tracking point. Order matches
            vr_3point_body config (typically [left_wrist, right_wrist, head]).
            If None, all points are weighted equally.
            Example: [2, 2, 1] gives wrists 2x importance vs head.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_pos = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    robot_3point_diff = command.robot_vr_3point_pos_w - command.robot_anchor_pos_w[:, None, :]
    robot_3point_pos = quat_apply(quat_inv(robot_root_quat), robot_3point_diff)
    diff = robot_3point_pos - ref_3point_pos
    error = torch.sum(torch.square(diff), dim=-1)  # [num_envs, num_points]

    if point_weights is not None:
        # Weighted mean: sum(w_i * e_i) / sum(w_i)
        weights = torch.tensor(point_weights, dtype=error.dtype, device=error.device)
        weighted_error = (error * weights).sum(dim=-1) / weights.sum()
    else:
        # Simple mean (equal weights)
        weighted_error = error.mean(dim=-1)

    return torch.exp(-weighted_error / std**2)


def tracking_local_vr_5point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 5-point tracking reward in the anchor's local frame.

    Same approach as tracking_local_vr_3point_error but with 5 tracking points
    (e.g., 2 wrists + head + 2 feet).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_5point_diff = command.reward_point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    ref_5point_pos = quat_apply(quat_inv(ref_root_quat), ref_5point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    robot_5point_diff = (
        command.robot_reward_point_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    )
    robot_5point_pos = quat_apply(quat_inv(robot_root_quat), robot_5point_diff)
    diff = robot_5point_pos - ref_5point_pos
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_vr_3point_error_pos_force(
    env: ManagerBasedRLEnv, motion_command_name: str, force_command_name: str, std: float
):
    """Compute VR 3-point tracking reward with force-based compliance correction.

    Add a force-proportional offset to the wrist tracking error so that applied
    external forces shift the tracking target, enabling compliant behavior under
    force perturbations.

    Args:
        env: The environment.
        motion_command_name: Name of the motion tracking command term.
        force_command_name: Name of the force tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    motion_command: TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    diff = motion_command.robot_vr_3point_pos_w - motion_command.vr_3point_body_pos_w
    force_error_wrists = (
        force_command.last_force_applied * force_command.eef_stiffness_buf[:, :, None]
    )
    diff[:, :2] += force_error_wrists
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body orientations to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def _tracked_body_names(command: TrackingCommand, body_indexes: list[int]) -> list[str]:
    return [command.cfg.body_names[i] for i in body_indexes]


def _contact_phase_body_weights(
    command: TrackingCommand,
    body_indexes: list[int],
    base_weights: torch.Tensor,
    contact_relax_body_names: list[str] | str | None = None,
    contact_relax_weight: float = 1.0,
    contact_relax_hand: str = "either",
    contact_relax_gate_mode: str = "binary",
    contact_relax_pre_frames: int = 0,
    contact_relax_post_frames: int = 0,
    contact_relax_sigma_frames: float = 5.0,
) -> torch.Tensor:
    """Optionally reduce selected body tracking weights during contact phases."""
    if not contact_relax_body_names or float(contact_relax_weight) == 1.0:
        return base_weights
    if isinstance(contact_relax_body_names, str):
        relax_names = {
            item.strip()
            for item in contact_relax_body_names.split(",")
            if item.strip()
        }
    else:
        relax_names = {str(item).strip() for item in contact_relax_body_names if str(item).strip()}
    if not relax_names:
        return base_weights

    tracked_names = _tracked_body_names(command, body_indexes)
    relax_mask = torch.tensor(
        [name in relax_names for name in tracked_names],
        device=base_weights.device,
        dtype=base_weights.dtype,
    )
    if not bool((relax_mask > 0).any().item()):
        return base_weights

    gate = _reference_contact_gate(
        command,
        hand=contact_relax_hand,
        gate_mode=contact_relax_gate_mode,
        pre_contact_frames=contact_relax_pre_frames,
        post_contact_frames=contact_relax_post_frames,
        gate_sigma_frames=contact_relax_sigma_frames,
    ).to(base_weights.device, dtype=base_weights.dtype)
    multiplier = torch.ones_like(base_weights)
    multiplier = torch.where(relax_mask > 0, torch.full_like(multiplier, float(contact_relax_weight)), multiplier)
    return base_weights[None, :] * (1.0 - gate[:, None] + gate[:, None] * multiplier[None, :])


def tracking_relative_body_pos_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    contact_relax_body_names: list[str] | str | None = None,
    contact_relax_weight: float = 1.0,
    contact_relax_hand: str = "either",
    contact_relax_gate_mode: str = "binary",
    contact_relax_pre_frames: int = 0,
    contact_relax_post_frames: int = 0,
    contact_relax_sigma_frames: float = 5.0,
) -> torch.Tensor:
    """Compute body position tracking reward using anchor-relative reference positions.

    Use reference body positions that have been shifted to share the robot's anchor
    (root) position, so only the relative pose matters rather than absolute position.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    base_weights = torch.ones(len(tracked), device=per_body_err.device, dtype=per_body_err.dtype)
    weights = _contact_phase_body_weights(
        command,
        tracked,
        base_weights,
        contact_relax_body_names=contact_relax_body_names,
        contact_relax_weight=contact_relax_weight,
        contact_relax_hand=contact_relax_hand,
        contact_relax_gate_mode=contact_relax_gate_mode,
        contact_relax_pre_frames=contact_relax_pre_frames,
        contact_relax_post_frames=contact_relax_post_frames,
        contact_relax_sigma_frames=contact_relax_sigma_frames,
    )
    if weights.ndim == 1:
        weighted_error = (per_body_err * weights).sum(-1) / weights.sum().clamp_min(1e-6)
    else:
        weighted_error = (per_body_err * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
    return torch.exp(-weighted_error / (std * std))


def tracking_relative_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward using anchor-relative reference orientations.

    Use reference body orientations that have been transformed to share the robot's
    anchor (root) orientation, making the reward invariant to global heading.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_relative_w[:, tracked],
        command.robot_body_quat_w[:, tracked],
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_ori_weighted_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    body_weights: dict[str, float] | None = None,
    contact_relax_body_names: list[str] | str | None = None,
    contact_relax_weight: float = 1.0,
    contact_relax_hand: str = "either",
    contact_relax_gate_mode: str = "binary",
    contact_relax_pre_frames: int = 0,
    contact_relax_post_frames: int = 0,
    contact_relax_sigma_frames: float = 5.0,
) -> torch.Tensor:
    """Compute anchor-relative body orientation tracking reward with per-body weights.

    Same as tracking_relative_body_ori_error but allows different bodies to contribute
    differently to the mean error. Useful for relaxing tracking on certain joints
    (e.g., wrists during manipulation).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.
        body_weights: Dict mapping body name to weight multiplier. Bodies not listed
            default to 1.0. E.g. {"left_wrist_yaw_link": 0.1} to relax wrist tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    if body_weights is not None:
        tracked_names = _tracked_body_names(command, body_indexes)
        weights = torch.tensor(
            [body_weights.get(name, 1.0) for name in tracked_names],
            device=error.device,
            dtype=error.dtype,
        )
        weights = _contact_phase_body_weights(
            command,
            body_indexes,
            weights,
            contact_relax_body_names=contact_relax_body_names,
            contact_relax_weight=contact_relax_weight,
            contact_relax_hand=contact_relax_hand,
            contact_relax_gate_mode=contact_relax_gate_mode,
            contact_relax_pre_frames=contact_relax_pre_frames,
            contact_relax_post_frames=contact_relax_post_frames,
            contact_relax_sigma_frames=contact_relax_sigma_frames,
        )
        if weights.ndim == 1:
            weighted_error = (error * weights).sum(-1) / weights.sum().clamp_min(1e-6)
        else:
            weighted_error = (error * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
    elif contact_relax_body_names:
        weights = torch.ones(len(body_indexes), device=error.device, dtype=error.dtype)
        weights = _contact_phase_body_weights(
            command,
            body_indexes,
            weights,
            contact_relax_body_names=contact_relax_body_names,
            contact_relax_weight=contact_relax_weight,
            contact_relax_hand=contact_relax_hand,
            contact_relax_gate_mode=contact_relax_gate_mode,
            contact_relax_pre_frames=contact_relax_pre_frames,
            contact_relax_post_frames=contact_relax_post_frames,
            contact_relax_sigma_frames=contact_relax_sigma_frames,
        )
        if weights.ndim == 1:
            weighted_error = (error * weights).sum(-1) / weights.sum().clamp_min(1e-6)
        else:
            weighted_error = (error * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
    else:
        weighted_error = error.mean(-1)
    return torch.exp(-weighted_error / std**2)


def tracking_body_linvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body linear velocity tracking reward using a Gaussian kernel.

    Encourages tracked body linear velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_lin_vel_w[:, tracked] - command.robot_body_lin_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_body_angvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body angular velocity tracking reward using a Gaussian kernel.

    Encourages tracked body angular velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_ang_vel_w[:, tracked] - command.robot_body_ang_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def feet_contact_duration(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Compute reward for maintaining foot contact within a time threshold.

    Penalize feet that are airborne but whose last contact was recent (within
    threshold), encouraging stable ground contact timing.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor configuration for the feet.
        threshold: Maximum time (seconds) since last contact to still count as
            recently airborne.

    Returns:
        Reward tensor of shape (num_envs,). Count of feet meeting the criterion.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    airborne_mask = sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    time_since_contact = sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return ((time_since_contact < threshold) * airborne_mask).sum(dim=-1)


def feet_contact_requirement(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_ref_feet_height_no_contact: float,
    max_ref_feet_height_when_contact: float,
):
    """Compute penalty for incorrect foot contact relative to reference motion height.

    Penalize two error cases: (1) foot is on the ground when the reference foot is
    above min_ref_feet_height_no_contact (should be airborne), and (2) foot is in
    the air when the reference foot is below max_ref_feet_height_when_contact (should
    be grounded). Skips the first 10 steps and ignores crouching motions
    (ref root height < 0.3m).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        sensor_cfg: Contact sensor configuration for the feet.
        min_ref_feet_height_no_contact: Reference foot height above which the real
            foot should not be in contact.
        max_ref_feet_height_when_contact: Reference foot height below which the real
            foot must be in contact.

    Returns:
        Penalty tensor of shape (num_envs,). Negative values indicate violations.
    """
    # Add validation for threshold ordering
    assert (
        min_ref_feet_height_no_contact > max_ref_feet_height_when_contact
    ), "min_ref_feet_height_no_contact should be greater than max_ref_feet_height_when_contact"

    command: TrackingCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Remove redundant slicing
    current_feet_contact_forces = contact_sensor.data.net_forces_w
    feet_index = contact_sensor.find_bodies(".*ankle_roll.*")[0]
    current_feet_in_the_air = current_feet_contact_forces[:, feet_index, 2] <= 1.0

    body_indexes = _get_body_indexes(command, sensor_cfg.body_names)
    # Fix: Extract only Z-coordinate for height comparison
    ref_feet_height = command.body_pos_w[:, body_indexes, 2]

    feet_not_need_contact = ref_feet_height > min_ref_feet_height_no_contact

    feet_must_contact = ref_feet_height < max_ref_feet_height_when_contact
    feet_wrong_contact = torch.logical_and(feet_not_need_contact, ~current_feet_in_the_air)
    feet_wrong_air = torch.logical_and(feet_must_contact, current_feet_in_the_air)

    feet_wrong_contact_reward = -torch.logical_or(feet_wrong_air, feet_wrong_contact).to(
        torch.float64
    )

    after_10_steps = env.episode_length_buf > 10

    current_reward = torch.sum(feet_wrong_contact_reward, dim=1) * after_10_steps.to(torch.float64)

    current_reward[command.running_ref_root_height < 0.3] = 0.0

    return current_reward


def feet_contact_requirement_motion_lib(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
):
    """Compute penalty for incorrect foot contact using motion library contact labels.

    Use precomputed left/right foot contact labels from the motion library instead
    of height-based heuristics. Penalize cases where the reference says a foot
    should be in contact but the robot's foot is airborne. Skips the first 10 steps
    and ignores crouching motions (ref root height < 0.3m).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        sensor_cfg: Contact sensor configuration for the feet.

    Returns:
        Penalty tensor of shape (num_envs,). Negative values indicate violations.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_feet_contact_forces = contact_sensor.data.net_forces_w
    feet_index = contact_sensor.find_bodies(".*ankle_roll.*")[0]
    current_feet_in_the_air = current_feet_contact_forces[:, feet_index, 2] <= 1.0
    ref_feet_l = command.feet_l  # Left foot contact from reference
    ref_feet_r = command.feet_r  # Right foot contact from reference
    ref_foot_contact = torch.stack([ref_feet_l, ref_feet_r], dim=1)  # Shape: [num_envs, 2]
    feet_wrong_contact_reward = -torch.logical_and(
        ref_foot_contact.squeeze(-1), current_feet_in_the_air
    ).to(torch.float64)

    after_10_steps = env.episode_length_buf > 10
    current_reward = torch.sum(feet_wrong_contact_reward, dim=1) * after_10_steps.to(torch.float64)
    current_reward[command.running_ref_root_height < 0.3] = 0.0
    return current_reward


def jitter_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute total squared body position error across all bodies and coordinates.

    Penalize large deviations from the reference body positions. Unlike the per-body
    mean used in tracking rewards, this sums across all coordinates for a raw
    aggregate error signal.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.

    Returns:
        Penalty tensor of shape (num_envs, num_bodies). Squared error per body.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    return torch.sum(torch.square(command.body_pos_w - command.robot_body_pos_w), dim=-1)


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize excessive angular velocity on selected bodies with a deadzone.

    Discourage high-frequency jitter on small links (wrists, head) while allowing
    normal intentional motion within the threshold. Speeds below the threshold
    incur zero penalty.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        threshold: Angular velocity deadzone (rad/s). No penalty below this.
        body_names: Bodies to penalize. If None, uses all tracked bodies.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # [E, B, 3]
    ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
    # magnitude per body: [E, B]
    speed = torch.linalg.norm(ang_vel, dim=-1)
    # deadzone then square: [E, B]
    excess = torch.relu(speed - threshold)
    penalty = (excess * excess).mean(dim=-1)
    return penalty


# ==================== HOI Manipulation Rewards ====================


def reward_hand_fingers_object_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_to_hand_frame_transformer"),
    exp_coeff: float = -10.0,
    command_name: str = "motion",
) -> torch.Tensor:
    """Compute reward for hand proximity to the object.

    Encourage the hand to approach the object using an exponential distance kernel.
    Only applied when the contact label indicates contact should happen
    (current_frame >= first_contact_frame).

    Args:
        env: The environment.
        asset_cfg: Frame transformer config for the object-to-hand transform.
        exp_coeff: Exponential coefficient for distance-to-reward mapping (negative).
        command_name: Name of the tracking command term.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Check if contact data is available - use precomputed per-env first contact frame
    per_env_first_contact = getattr(command, "_per_env_first_contact", None)

    if per_env_first_contact is not None:
        current_time = command.motion_start_time_steps + command.time_steps
        should_contact = (current_time >= per_env_first_contact).float()
    else:
        # No contact data, always apply reward
        should_contact = torch.ones(env.num_envs, device=env.device)

    hand_object_transform = get_hand_object_transform(env, asset_cfg)
    object_pos_in_hand = hand_object_transform[:, :3]
    object_pos_in_hand_distance = torch.norm(object_pos_in_hand, dim=-1)
    reward = torch.exp(exp_coeff * object_pos_in_hand_distance)
    # Only apply reward when contact should happen
    return reward * should_contact


def _measured_grasp_score(
    env: ManagerBasedRLEnv,
    command: TrackingCommand | None,
    *,
    hand: str,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.1,
    min_contacts: int = 3,
    measured_contact_source: str = "legacy",
    canonical_force_threshold_n: float = 1.0,
    canonical_force_release_threshold_n: float | None = None,
    canonical_contact_persistence_frames: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unchanged measured grasp score in [0, 1] and the finger-contact count for one hand.

    ``legacy`` counts finger-sensor forces above ``contact_threshold`` (N) against ``min_contacts``;
    ``canonical`` uses the cached active-entity hand contact bit. This is the exact measurement that
    ``reward_grasp`` used before the per-hand reference-state policy existed; nothing about the grasp
    definition or its thresholds changes here.
    """
    measured_source = str(measured_contact_source).lower()
    if measured_source == "canonical":
        from gear_sonic.envs.manager_env.mdp.rp2_source_relation import (  # noqa: PLC0415
            canonical_measured_contact_state,
        )

        measured = canonical_measured_contact_state(
            env,
            force_threshold_n=float(canonical_force_threshold_n),
            force_release_threshold_n=canonical_force_release_threshold_n,
            persistence_frames=int(canonical_contact_persistence_frames),
        )
        entity_index, body_indices = canonical_hand_contact_indices(hand)
        canonical_hand_contact = measured[:, entity_index, list(body_indices)].amax(dim=-1)
        return canonical_hand_contact, canonical_hand_contact
    if measured_source != "legacy":
        raise ValueError(
            "measured_contact_source must be 'legacy' or 'canonical', "
            f"got {measured_contact_source!r}"
        )
    force = (
        utils.get_target_object_contact_forces(
            env,
            command,
            hand=hand,
            fallback_sensor_name=sensor_cfg.name,
        )
        if command is not None
        else None
    )
    if force is None:
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        force_magnitude = _finger_contact_force_magnitude(sensor)
    else:
        force_magnitude = torch.norm(force, dim=-1)
    fingers_in_contact = (force_magnitude > contact_threshold).float()
    num_fingers_in_contact = fingers_in_contact.sum(dim=-1)
    reward = torch.clamp(num_fingers_in_contact / min_contacts, max=1.0)
    return reward, num_fingers_in_contact


# Per-hand reference contact state used by the ordinary hand reward policy (2026-10-13):
#   +1 required (reference hand touches the active object), 0 neutral (no requirement), -1 known no-contact.
HAND_REF_REQUIRED, HAND_REF_NEUTRAL, HAND_REF_NEGATIVE = 1, 0, -1


def _legacy_hand_label(
    motion_lib, motion_ids: torch.Tensor, motion_steps: torch.Tensor, hand: str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Legacy binary hand label and its per-frame presence (None when the loaded state has no presence channel).

    Prefers the per-hand ``object_in_contact_*`` family (what the legacy gate reads); otherwise the per-object
    ``object_contact_mask_*`` reduced over objects. A loader-padded 0 is never a label: the presence channel
    distinguishes it from a valid binary 0.
    """
    label = (
        motion_lib.get_object_in_contact(motion_ids, motion_steps, hand=hand)
        if hasattr(motion_lib, "get_object_in_contact")
        else None
    )
    if label is not None:
        valid = (
            motion_lib.get_object_in_contact_valid(motion_ids, motion_steps, hand=hand)
            if hasattr(motion_lib, "get_object_in_contact_valid")
            else None
        )
        return label.float(), (None if valid is None else valid > 0.5)
    mask = (
        motion_lib.get_object_contact_mask(motion_ids, motion_steps, hand=hand)
        if hasattr(motion_lib, "get_object_contact_mask")
        else None
    )
    if mask is None:
        return None, None
    label = mask.reshape(mask.shape[0], -1).amax(dim=-1).float()
    valid = (
        motion_lib.get_object_contact_mask_valid(motion_ids, motion_steps, hand=hand)
        if hasattr(motion_lib, "get_object_contact_mask_valid")
        else None
    )
    return label, (None if valid is None else valid > 0.5)


def _hand_reference_state(
    command: TrackingCommand, hand: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(state, absent)`` for one hand at the current reference frames.

    ``state`` is int8 in {+1 required, 0 neutral, -1 known no-contact}; ``absent`` marks frames with no
    usable supervision at all (``state`` is 0 there and callers apply the documented neutral fallback).
    Precedence, per the audited source semantics (ROOT_CONTACT_LABEL_SEMANTICS 2026-10-13):
      1. canonical three-state ``reference_contact_state`` for the hand body group when the clip carried it
         (canonical 0 is a real geometric NEUTRAL, not a missing label);
      2. a VALID native binary label (presence channel = 1): 1 -> required, 0 -> known no-contact. When the
         clip's native label is marked authoritative (OmniContact wrist contact_info), a valid native 0 is
         known no-contact even where the canonical conversion says neutral; it never demotes a canonical
         required frame;
      3. otherwise absent. Loader padding never counts as a label; a legacy field that is merely missing
         never overrides a canonical forbidden; geometry validity is not label validity. When the loaded
         state predates the presence channels (old optional cache bundle), canonical values are taken as
         they are and legacy zeros cannot be proven negative.
    """
    motion_lib = command.motion_lib
    motion_ids = command.motion_ids
    steps = _clamp_motion_steps(command, command.motion_start_time_steps + command.time_steps)
    n = motion_ids.shape[0]
    device = motion_ids.device
    state = torch.zeros(n, dtype=torch.int8, device=device)
    decided = torch.zeros(n, dtype=torch.bool, device=device)

    reference = (
        motion_lib.get_reference_contact_state(motion_ids, steps)
        if hasattr(motion_lib, "get_reference_contact_state")
        else None
    )
    if reference is not None:
        entity_index, body_indices = canonical_hand_contact_indices(hand)
        canonical = reference[:, entity_index, list(body_indices)].to(torch.int8).amax(dim=-1)
        canonical_valid_t = (
            motion_lib.get_reference_contact_state_valid(motion_ids, steps)
            if hasattr(motion_lib, "get_reference_contact_state_valid")
            else None
        )
        canonical_valid = (
            canonical_valid_t > 0.5
            if canonical_valid_t is not None
            else torch.ones(n, dtype=torch.bool, device=device)
        )
        state = torch.where(canonical_valid, canonical, state)
        decided = decided | canonical_valid

    legacy, legacy_valid = _legacy_hand_label(motion_lib, motion_ids, steps, hand)
    if legacy is not None:
        legacy = legacy.to(device)
        positive = legacy > 0.5
        if legacy_valid is None:
            # no presence information: a 1 is a genuine label, a 0 may be padding -> not provably negative
            negative = torch.zeros(n, dtype=torch.bool, device=device)
        else:
            negative = (legacy <= 0.5) & legacy_valid.to(device)
        native_t = (
            motion_lib.get_contact_label_native_authoritative(motion_ids, steps)
            if hasattr(motion_lib, "get_contact_label_native_authoritative")
            else None
        )
        native = native_t.to(device) > 0.5 if native_t is not None else torch.zeros(n, dtype=torch.bool, device=device)
        # authoritative native 0 -> known no-contact, also over a canonical neutral (never over canonical required)
        override = negative & native & (state != HAND_REF_REQUIRED)
        state = torch.where(override, torch.full_like(state, HAND_REF_NEGATIVE), state)
        decided = decided | override
        # canonical undecided -> the valid binary label decides
        use = ~decided
        state = torch.where(use & positive, torch.full_like(state, HAND_REF_REQUIRED), state)
        state = torch.where(use & negative, torch.full_like(state, HAND_REF_NEGATIVE), state)
        decided = decided | (use & (positive | negative))

    absent = ~decided
    state = torch.where(absent, torch.zeros_like(state), state)
    return state, absent


_HAND_OPENNESS_CACHE: dict = {}


def _hand_openness(env: ManagerBasedRLEnv, hand: str) -> torch.Tensor:
    """Physical hand openness in [0, 1] from the ACTUAL finger joint positions.

    Uses the runtime finger-primitive map published by the env wrapper (per hand: absolute joint indices,
    ``pos_0`` = open, ``pos_1`` = closed in ACTION space, per-joint signs differ between hands) mapped to
    physical joint positions through the resolved joint_pos action scale/offset: openness = 1 - mean over the
    nonzero-span joints of clamp((q - q_open) / (q_closed - q_open), 0, 1). Zero-span joints (e.g. thumb_0
    with pos_0 == pos_1 == 0) are excluded, so they never divide by zero nor earn free credit. Fully open ->
    1, fully closed -> 0. The raw primitive action sign is deliberately not used. Joint indices are cached.
    """
    primitive_map = getattr(env, "_finger_primitive_map", None)
    if not primitive_map:
        raise RuntimeError(
            "hand openness requires the runtime finger primitive map (manager_env.config.use_finger_primitive "
            "with finger_primitive.primitive_action_map); the raw action sign is not a physical openness"
        )
    side = "left" if hand == "left_hand" else "right"
    key = (id(env), side)
    cached = _HAND_OPENNESS_CACHE.get(key)
    if cached is None:
        primitive = None
        for entry in primitive_map:
            name = str(entry.get("action_name", "")).lower()
            dof_names = [str(d).lower() for d in entry.get("dof_names", [])]
            if side in name or (dof_names and all(side in d for d in dof_names)):
                primitive = entry
                break
        if primitive is None:
            raise RuntimeError(f"no finger primitive found for the {side} hand in the runtime map")
        idx = torch.as_tensor(list(primitive["dof_idx"]), dtype=torch.long, device=env.device)
        pos_open_action = torch.as_tensor(primitive["pos_0"], dtype=torch.float32, device=env.device)
        pos_closed_action = torch.as_tensor(primitive["pos_1"], dtype=torch.float32, device=env.device)
        # The primitive endpoints are ACTION-space values: the joint-position action term applies
        # processed = raw * scale + offset before they become physical joint targets (IsaacLab JointAction;
        # G1 finger scale 0.7 / thumb_0 0.30625 from the robot action-scale table). Resolve the term's per-joint
        # scale/offset columns for exactly these joint ids; never hard-code the scale.
        action_term = env.action_manager.get_term("joint_pos")
        joint_ids = action_term._joint_ids  # noqa: SLF001
        if isinstance(joint_ids, slice):
            columns = idx.clone()
        else:
            lookup = {int(j): k for k, j in enumerate(list(joint_ids))}
            missing = [int(j) for j in idx.tolist() if int(j) not in lookup]
            if missing:
                raise RuntimeError(f"finger joints {missing} are not driven by the joint_pos action term")
            columns = torch.as_tensor([lookup[int(j)] for j in idx.tolist()], dtype=torch.long, device=env.device)
        span_action = pos_closed_action - pos_open_action
        keep = span_action.abs() > 1e-6
        if int(keep.sum().item()) == 0:
            raise RuntimeError(f"the {side} finger primitive has no nonzero-span joints")
        cached = (idx[keep], columns[keep], pos_open_action[keep], pos_closed_action[keep])
        _HAND_OPENNESS_CACHE[key] = cached
    idx, columns, pos_open_action, pos_closed_action = cached
    action_term = env.action_manager.get_term("joint_pos")
    scale = action_term._scale  # noqa: SLF001
    offset = action_term._offset  # noqa: SLF001
    scale_cols = scale[:, columns] if torch.is_tensor(scale) else torch.full((1, idx.numel()), float(scale), device=env.device)
    offset_cols = offset[:, columns] if torch.is_tensor(offset) else torch.full((1, idx.numel()), float(offset), device=env.device)
    q_open = scale_cols * pos_open_action + offset_cols      # physical joint position commanded at 'open'
    q_closed = scale_cols * pos_closed_action + offset_cols  # physical joint position commanded at 'closed'
    span = q_closed - q_open
    q = env.scene["robot"].data.joint_pos[:, idx]
    closed_fraction = ((q - q_open) / span).clamp(0.0, 1.0)
    return 1.0 - closed_fraction.mean(dim=-1)


def reward_hand_contact_policy(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    hand: str = "right_hand",
    command_name: str = "motion",
    contact_threshold: float = 0.1,
    min_contacts: int = 3,
    measured_contact_source: str = "legacy",
    canonical_force_threshold_n: float = 1.0,
    canonical_force_release_threshold_n: float | None = None,
    canonical_contact_persistence_frames: int = 1,
    coefficient_required: float = 1.0,
    coefficient_neutral: float = 0.1,
    coefficient_absent: float = 0.1,
    coefficient_open: float = 0.1,
    allowed_roles: list[str] | str | None = None,
    hand_reward_version: str = "required20_neutral2_open2_v1",
    # reward_grasp-era keys that Hydra still merges in from the shared term bundles. They must be NAMED parameters:
    # Isaac Lab's RewardManager validates term params against the signature (a **kwargs catch-all is rejected as a
    # mandatory parameter, GPU job 1305545). They have NO effect here: the reference state is resolved by
    # _hand_reference_state, never by the old binary/temporal gate.
    gate_with_contact_label: bool = False,
    gate_mode: str = "binary",
    pre_contact_frames: int = 0,
    post_contact_frames: int = 0,
    gate_sigma_frames: float = 5.0,
    contact_time_shift_frames: int = 0,
    anti_contact_factor: float | None = None,
    reference_contact_source: str = "legacy",
) -> torch.Tensor:
    """Per-hand ordinary reward policy (user decision 2026-10-13), mutually exclusive per hand per frame.

    The reward_grasp-era gate parameters (gate_with_contact_label, gate_mode, pre/post_contact_frames,
    gate_sigma_frames, contact_time_shift_frames, anti_contact_factor, reference_contact_source) are accepted for
    config compatibility and ignored: the reference state is resolved by _hand_reference_state and never by the old
    binary gate.

    With the term weight W (default 20) the effective coefficients are W x coefficient_*:
      reference REQUIRED            -> coefficient_required (1.0 -> 20) x measured grasp score
      reference NEUTRAL             -> coefficient_neutral  (0.1 -> 2)  x measured grasp score
      supervision truly ABSENT      -> coefficient_absent   (0.1 -> 2)  x measured grasp score (documented fallback)
      reference KNOWN NO-CONTACT    -> coefficient_open     (0.1 -> 2)  x physical hand openness
    Exactly one branch applies to a hand on a frame. No grasp reward in the known-no-contact branch; no openness
    reward (and no closed-hand penalty) in the other branches; a measured grasp of zero gives zero in the grasp
    branches and a closed hand gives zero in the openness branch. The measured grasp definition and thresholds
    are the unchanged ``reward_grasp`` measurement; openness comes from the actual finger joint positions.
    ``hand_reward_version`` only labels the policy in the resolved config.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    grasp, num_fingers_in_contact = _measured_grasp_score(
        env,
        command,
        hand=hand,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        min_contacts=min_contacts,
        measured_contact_source=measured_contact_source,
        canonical_force_threshold_n=canonical_force_threshold_n,
        canonical_force_release_threshold_n=canonical_force_release_threshold_n,
        canonical_contact_persistence_frames=canonical_contact_persistence_frames,
    )
    state, absent = _hand_reference_state(command, hand)
    required = state == HAND_REF_REQUIRED
    negative = state == HAND_REF_NEGATIVE
    neutral = (state == HAND_REF_NEUTRAL) & ~absent
    openness = _hand_openness(env, hand)
    grasp_coefficient = (
        required.float() * float(coefficient_required)
        + neutral.float() * float(coefficient_neutral)
        + absent.float() * float(coefficient_absent)
    )
    reward = grasp_coefficient * grasp + negative.float() * float(coefficient_open) * openness
    if allowed_roles:
        role_gate = utils.get_target_object_role_mask(env, command, allowed_roles).float()
        reward = reward * role_gate
    if hasattr(command, "metrics"):
        side = "left" if hand == "left_hand" else "right"
        command.metrics[f"hand_policy/{side}/required"] = required.float()
        command.metrics[f"hand_policy/{side}/neutral"] = neutral.float()
        command.metrics[f"hand_policy/{side}/absent"] = absent.float()
        command.metrics[f"hand_policy/{side}/known_no_contact"] = negative.float()
        command.metrics[f"hand_policy/{side}/measured_grasp"] = grasp
        command.metrics[f"hand_policy/{side}/openness"] = openness
        command.metrics[f"hand_policy/{side}/sim_contact_count"] = num_fingers_in_contact
        command.metrics[f"hand_policy/{side}/reward"] = reward
    return reward


def reward_grasp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.1,  # Min force to count as contact
    min_contacts: int = 3,  # Minimum number of finger contacts required
    gate_with_contact_label: bool = False,
    gate_mode: str = "binary",
    pre_contact_frames: int = 0,
    post_contact_frames: int = 0,
    gate_sigma_frames: float = 5.0,
    contact_time_shift_frames: int = 0,
    anti_contact_factor: float | None = None,
    reference_contact_source: str = "legacy",
    measured_contact_source: str = "legacy",
    canonical_force_threshold_n: float = 1.0,
    canonical_force_release_threshold_n: float | None = None,
    canonical_contact_persistence_frames: int = 1,
    hand: str = "right_hand",
    command_name: str = "motion",
    allowed_roles: list[str] | str | None = None,
) -> torch.Tensor:
    """Compute reward for successful grasp based on finger contact count.

    Reward is proportional to the number of fingers in contact, reaching 1.0 when
    at least min_contacts fingers touch the object. Optionally gates the reward by
    motion contact labels from object motion data, so that reward activation follows
    dataset timing rather than pure simulation contact.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config for finger-object contact.
        contact_threshold: Minimum force (N) to count a finger as in contact.
        min_contacts: Number of finger contacts required for full reward.
        gate_with_contact_label: If True, only apply reward when the motion data's
            contact label is active for the given hand.
        gate_mode: Optional contact-label smoothing mode. ``binary`` preserves
            the original behavior; ``mask_ramp`` and ``mask_gaussian`` use a
            temporal window around reference contact.
        reference_contact_source: ``legacy`` reads the per-hand object labels
            used by the original GRAIL-SONIC/RP2 configs. ``canonical`` reads
            the active-entity hand group from the unified ternary full-body
            contact reference.
        measured_contact_source: ``legacy`` counts individual finger-sensor
            forces. ``canonical`` uses the same cached active-entity hand bit
            as the joint contact observation and termination.
        hand: Which hand's contact label to use ("left_hand" or "right_hand").
        command_name: Name of the tracking command term.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    if anti_contact_factor is not None and not gate_with_contact_label:
        raise ValueError(
            "reward_grasp: anti_contact_factor requires gate_with_contact_label=True"
        )

    command: TrackingCommand | None = None
    if hasattr(env, "command_manager"):
        command = env.command_manager.get_term(command_name)
    reward, num_fingers_in_contact = _measured_grasp_score(
        env,
        command,
        hand=hand,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        min_contacts=min_contacts,
        measured_contact_source=measured_contact_source,
        canonical_force_threshold_n=canonical_force_threshold_n,
        canonical_force_release_threshold_n=canonical_force_release_threshold_n,
        canonical_contact_persistence_frames=canonical_contact_persistence_frames,
    )

    metric_side = "left" if hand == "left_hand" else "right"
    if command is not None and hasattr(command, "metrics"):
        command.metrics[f"rp2_grasp_sim_contact_count/{metric_side}"] = (
            num_fingers_in_contact
        )
        command.metrics[f"rp2_grasp_sim_contact_reward/{metric_side}"] = reward

    reward_out = reward

    if gate_with_contact_label:
        if command is None:
            command = env.command_manager.get_term(command_name)
        reference_contact = _reference_contact_label(
            command,
            hand,
            reference_contact_source=reference_contact_source,
        )
        if reference_contact is None:
            raise RuntimeError(
                "reward_grasp: gate_with_contact_label=True but no contact label found. "
                f"reference_contact_source={reference_contact_source!r}."
            )

        reference_contact_f = reference_contact.float()
        contact_gate = _reference_contact_gate(
            command,
            hand=hand,
            gate_mode=gate_mode,
            pre_contact_frames=pre_contact_frames,
            post_contact_frames=post_contact_frames,
            gate_sigma_frames=gate_sigma_frames,
            time_shift_frames=contact_time_shift_frames,
            reference_contact_source=reference_contact_source,
        )
        if anti_contact_factor is not None:
            # in_contact=1: normal reward; in_contact=0: penalty for unwanted contact
            reward_out = reward * contact_gate + anti_contact_factor * reward * (
                1.0 - contact_gate
            )
        else:
            reward_out = reward * contact_gate

        if hasattr(command, "metrics"):
            command.metrics[f"rp2_grasp_reference_contact/{metric_side}"] = (
                reference_contact_f
            )
            command.metrics[f"rp2_grasp_reference_contact_gate/{metric_side}"] = (
                contact_gate
            )
            command.metrics[f"rp2_grasp_off_label_contact/{metric_side}"] = reward * (
                1.0 - reference_contact_f
            )

    if allowed_roles:
        if command is None:
            command = env.command_manager.get_term(command_name)
        role_gate = utils.get_target_object_role_mask(
            env, command, allowed_roles
        ).float()
        reward_out = reward_out * role_gate
        if hasattr(command, "metrics"):
            command.metrics[f"rp2_grasp_role_eligible/{metric_side}"] = role_gate

    if command is not None and hasattr(command, "metrics"):
        command.metrics[f"rp2_grasp_reward_gated/{metric_side}"] = reward_out

    return reward_out


def _reference_contact_label(
    command: TrackingCommand,
    hand: str = "right_hand",
    *,
    reference_contact_source: str = "legacy",
) -> torch.Tensor | None:
    """Return the reference contact label for one hand or either hand."""
    current_steps = command.motion_start_time_steps + command.time_steps
    return _reference_contact_label_at(
        command,
        hand=hand,
        motion_steps=current_steps,
        reference_contact_source=reference_contact_source,
    )


def _clamp_motion_steps(command: TrackingCommand, motion_steps: torch.Tensor) -> torch.Tensor:
    motion_steps = torch.clamp(motion_steps, min=0)
    motion_num_steps = getattr(command, "motion_num_steps", None)
    if motion_num_steps is None:
        return motion_steps
    if torch.is_tensor(motion_num_steps):
        max_steps = torch.clamp(motion_num_steps.to(motion_steps.device) - 1, min=0)
    else:
        max_steps = torch.full_like(motion_steps, max(int(motion_num_steps) - 1, 0))
    return torch.minimum(motion_steps, max_steps)


def _reference_contact_label_at(
    command: TrackingCommand,
    hand: str = "right_hand",
    motion_steps: torch.Tensor | None = None,
    *,
    reference_contact_source: str = "legacy",
) -> torch.Tensor | None:
    """Return reference contact labels at explicit motion-library frame indices."""
    if motion_steps is None:
        return _reference_contact_label(
            command,
            hand=hand,
            reference_contact_source=reference_contact_source,
        )
    motion_steps = _clamp_motion_steps(command, motion_steps)
    source = str(reference_contact_source).lower()
    if source == "canonical":
        reference = command.motion_lib.get_reference_contact_state(
            command.motion_ids,
            motion_steps,
        )
        if reference is None:
            return None
        expected_tail = (
            len(CONTACT_ENTITY_CLASSES),
            len(CONTACT_BODY_GROUPS),
        )
        if tuple(reference.shape[1:]) != expected_tail:
            raise ValueError(
                "Canonical reference contact shape is "
                f"{tuple(reference.shape)}, expected [N, {expected_tail[0]}, "
                f"{expected_tail[1]}]"
            )
        entity_index, body_indices = canonical_hand_contact_indices(hand)
        labels = [
            (
                reference[
                    :,
                    entity_index,
                    body_index,
                ]
                > 0.5
            ).float()
            for body_index in body_indices
        ]
        result = labels[0]
        for other in labels[1:]:
            result = torch.maximum(result, other)
        return result
    if source != "legacy":
        raise ValueError(
            "reference_contact_source must be 'legacy' or 'canonical', "
            f"got {reference_contact_source!r}"
        )
    if hand == "either":
        labels = [
            label
            for label in (
                command.motion_lib.get_object_in_contact(
                    command.motion_ids, motion_steps, hand="left_hand"
                ),
                command.motion_lib.get_object_in_contact(
                    command.motion_ids, motion_steps, hand="right_hand"
                ),
            )
            if label is not None
        ]
        if not labels:
            return None
        label = labels[0].float()
        for other in labels[1:]:
            label = torch.maximum(label, other.float())
        return label
    label = command.motion_lib.get_object_in_contact(command.motion_ids, motion_steps, hand=hand)
    return None if label is None else label.float()


def _first_contact_gate(
    command: TrackingCommand,
    pre_contact_frames: int = 0,
) -> torch.Tensor | None:
    """Return a 0..1 ramp ending at the first reference contact frame."""
    first_contact_lookup = getattr(command, "_first_contact_lookup", None)
    if first_contact_lookup is None or int(pre_contact_frames) <= 0:
        return None

    current_frame = command.motion_start_time_steps + command.time_steps
    first_contact = first_contact_lookup[command.motion_ids]
    motion_num_steps = getattr(command, "motion_num_steps", None)
    if motion_num_steps is None:
        has_contact = torch.ones_like(first_contact, dtype=torch.bool)
    else:
        has_contact = first_contact < motion_num_steps

    distance_to_contact = current_frame - first_contact
    ramp = (distance_to_contact.float() + float(pre_contact_frames)) / float(pre_contact_frames)
    ramp = torch.clamp(ramp, min=0.0, max=1.0)
    return ramp * has_contact.float()


def _contact_mask_window_gate(
    command: TrackingCommand,
    hand: str = "right_hand",
    pre_contact_frames: int = 0,
    post_contact_frames: int = 0,
    gate_mode: str = "mask_ramp",
    gate_sigma_frames: float = 5.0,
    time_shift_frames: int = 0,
    reference_contact_source: str = "legacy",
) -> torch.Tensor | None:
    """Dilate/smooth the reference contact label around contact onset/release.

    Positive offsets look ahead, which activates the reward before an upcoming
    reference contact.  Negative offsets keep it active shortly after release.
    """
    pre = max(int(pre_contact_frames), 0)
    post = max(int(post_contact_frames), 0)
    if pre <= 0 and post <= 0 and int(time_shift_frames) == 0:
        return None

    mode = (gate_mode or "mask_ramp").lower()
    current_steps = command.motion_start_time_steps + command.time_steps + int(time_shift_frames)
    current_steps = _clamp_motion_steps(command, current_steps)
    gate = torch.zeros(command.num_envs, dtype=torch.float32, device=command.device)
    sigma = max(float(gate_sigma_frames), 1e-6)
    for offset in range(-post, pre + 1):
        steps = _clamp_motion_steps(command, current_steps + offset)
        label = _reference_contact_label_at(
            command,
            hand=hand,
            motion_steps=steps,
            reference_contact_source=reference_contact_source,
        )
        if label is None:
            continue
        abs_offset = abs(offset)
        if mode in {"mask_gaussian", "gaussian", "smooth_mask"}:
            weight = math.exp(-0.5 * (float(abs_offset) / sigma) ** 2)
        elif offset > 0 and pre > 0:
            weight = 1.0 - float(offset) / float(pre + 1)
        elif offset < 0 and post > 0:
            weight = 1.0 - float(abs_offset) / float(post + 1)
        else:
            weight = 1.0
        gate = torch.maximum(gate, label.float() * float(weight))
    return gate


def _reference_contact_gate(
    command: TrackingCommand,
    hand: str = "right_hand",
    gate_mode: str = "binary",
    pre_contact_frames: int = 0,
    post_contact_frames: int = 0,
    gate_sigma_frames: float = 5.0,
    time_shift_frames: int = 0,
    distance: torch.Tensor | None = None,
    soft_gate_radius: float = 0.15,
    soft_gate_temperature: float = 0.03,
    reference_contact_source: str = "legacy",
) -> torch.Tensor:
    """Reference-contact gate used by optional RP2 contact-transition ablations.

    ``binary`` exactly mirrors the existing behavior.  ``precontact`` adds a
    short ramp before the first reference contact frame.  ``soft_distance`` also
    activates smoothly when the current hand/object distance is already close.
    """
    mode = (gate_mode or "binary").lower()
    current_steps = command.motion_start_time_steps + command.time_steps + int(time_shift_frames)
    label = _reference_contact_label_at(
        command,
        hand,
        _clamp_motion_steps(command, current_steps),
        reference_contact_source=reference_contact_source,
    )
    if label is None:
        per_env_first_contact = getattr(command, "_per_env_first_contact", None)
        if per_env_first_contact is not None:
            current_frame = command.motion_start_time_steps + command.time_steps
            label = (current_frame >= per_env_first_contact).float()
        else:
            label = torch.ones(command.num_envs, device=command.device)
    gate = label.float()

    if mode in {"mask_ramp", "mask_gaussian", "gaussian", "smooth_mask", "dilated", "window"}:
        window_gate = _contact_mask_window_gate(
            command,
            hand=hand,
            pre_contact_frames=pre_contact_frames,
            post_contact_frames=post_contact_frames,
            gate_mode=mode,
            gate_sigma_frames=gate_sigma_frames,
            time_shift_frames=time_shift_frames,
            reference_contact_source=reference_contact_source,
        )
        if window_gate is not None:
            gate = torch.maximum(gate, window_gate)

    if mode in {"precontact", "ramp", "precontact_ramp", "soft_distance"}:
        ramp = _first_contact_gate(command, pre_contact_frames=pre_contact_frames)
        if ramp is not None:
            gate = torch.maximum(gate, ramp)

    if mode in {"distance_only", "soft_distance_only", "sigmoid_distance"} and distance is not None:
        temperature = max(float(soft_gate_temperature), 1e-6)
        gate = torch.sigmoid((float(soft_gate_radius) - distance) / temperature)
    elif mode in {"soft", "soft_distance", "distance"} and distance is not None:
        temperature = max(float(soft_gate_temperature), 1e-6)
        soft = torch.sigmoid((float(soft_gate_radius) - distance) / temperature)
        gate = torch.maximum(gate, soft)

    if mode in {"always", "none"}:
        gate = torch.ones_like(gate)
    return torch.clamp(gate, min=0.0, max=1.0)


def _contact_center_world_from_local(
    command: TrackingCommand,
    contact_center_local: torch.Tensor,
) -> torch.Tensor:
    """Transform an object-local contact center into the current target object's frame."""
    from gear_sonic.isaac_utils import rotations as isaac_utils_rotations  # noqa: PLC0415

    return (
        isaac_utils_rotations.quat_rotate(
            command.target_object_root_quat,
            contact_center_local,
            w_last=False,
        )
        + command.target_object_root_pos
    )


def _lookahead_contact_center_world(
    command: TrackingCommand,
    hand: str,
    fallback_center: torch.Tensor,
    max_lookahead_frames: int,
    min_valid_norm: float,
) -> torch.Tensor:
    """Fill missing current contact centers from the nearest future reference contact."""
    if int(max_lookahead_frames) <= 0:
        return fallback_center

    current_frame = command.motion_start_time_steps + command.time_steps
    motion_num_steps = getattr(command, "motion_num_steps", None)
    if motion_num_steps is None:
        return fallback_center
    max_frame = torch.clamp(motion_num_steps - 1, min=0)

    out = fallback_center
    valid = torch.norm(out, dim=-1) > float(min_valid_norm)
    for offset in range(1, int(max_lookahead_frames) + 1):
        if bool(valid.all().item()):
            break
        future_steps = torch.minimum(current_frame + offset, max_frame)
        future_center_local = command.motion_lib.get_object_contact_center(
            command.motion_ids, future_steps, hand=hand
        )
        if future_center_local is None:
            break
        future_label = command.motion_lib.get_object_in_contact(
            command.motion_ids, future_steps, hand=hand
        )
        future_valid = torch.norm(future_center_local, dim=-1) > float(min_valid_norm)
        if future_label is not None:
            future_valid = future_valid & (future_label.float() > 0.5)
        use_future = (~valid) & future_valid
        if bool(use_future.any().item()):
            future_center_world = _contact_center_world_from_local(command, future_center_local)
            out = torch.where(use_future[:, None], future_center_world, out)
            valid = valid | use_future
    return out


def reward_grasp_finger_direction(
    env: ManagerBasedRLEnv,
    thumb_link: str = "right_hand_thumb_2_link",
    index_link: str = "right_hand_index_1_link",
    middle_link: str | None = "right_hand_middle_1_link",
    hand: str = "right_hand",
    command_name: str = "motion",
    thumb_offset: tuple[float, float, float] | None = None,
    index_offset: tuple[float, float, float] | None = None,
    middle_offset: tuple[float, float, float] | None = None,
    use_contact_center: bool = False,
    gate_mode: str = "binary",
    pre_contact_frames: int = 0,
    post_contact_frames: int = 0,
    gate_sigma_frames: float = 5.0,
    contact_time_shift_frames: int = 0,
    soft_gate_radius: float = 0.15,
    soft_gate_temperature: float = 0.03,
    allowed_roles: list[str] | str | None = None,
) -> torch.Tensor:
    """Compute reward for fingers opposing around the object for a pinch grasp.

    Encourage the thumb and index/middle fingers to be on opposite sides of the
    object by computing the negative dot product of the object-to-thumb and
    object-to-index/middle directions. When fingers oppose, the dot product is
    negative and the reward is positive.

    Only applied when the in_contact label is active for the given hand.

    Args:
        env: The environment.
        thumb_link: Name of the thumb tip link.
        index_link: Name of the index finger tip link.
        middle_link: Name of the middle finger tip link.
        hand: Which hand's in_contact label to use ("left_hand" or "right_hand").
        command_name: Name of the tracking command term.
        thumb_offset: Optional (x, y, z) offset in the thumb link's local frame,
            e.g. to shift from the link origin to the fingertip contact point.
        index_offset: Optional (x, y, z) offset in the index link's local frame.
        middle_offset: Optional (x, y, z) offset in the middle link's local frame.
        use_contact_center: If True, use the motion data's per-hand contact target
            instead of the object center of mass. This is suitable only when that
            target represents a grasp center; a nearest surface point generally
            does not define finger opposition. Falls back to object CoM per-env
            when the target is unavailable or zero.

    Returns:
        Reward tensor of shape (num_envs,). Positive when fingers oppose.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    robot = env.scene["robot"]

    # Reference point: contact center from motion data, or object CoM fallback
    ref_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)
    if use_contact_center:
        side = "left" if hand == "left_hand" else "right"
        contact_center = getattr(command, f"object_contact_center_{side}", None)
        if contact_center is not None:
            valid = torch.norm(contact_center, dim=-1, keepdim=True) > 1e-6
            ref_pos = torch.where(valid, contact_center, ref_pos)

    # Get finger link indices
    body_names = robot.body_names
    if thumb_link not in body_names or index_link not in body_names:
        return torch.zeros(env.num_envs, device=env.device)
    thumb_idx = body_names.index(thumb_link)
    index_idx = body_names.index(index_link)
    middle_idx = body_names.index(middle_link) if middle_link in body_names else None

    # Get finger positions, optionally offset from link origin to fingertip
    thumb_pos = robot.data.body_pos_w[:, thumb_idx, :]  # (num_envs, 3)
    if thumb_offset is not None:
        offset = torch.tensor(thumb_offset, device=env.device, dtype=thumb_pos.dtype).unsqueeze(0).expand_as(thumb_pos)
        thumb_pos = thumb_pos + quat_apply(robot.data.body_quat_w[:, thumb_idx, :], offset)

    index_pos = robot.data.body_pos_w[:, index_idx, :]  # (num_envs, 3)
    if index_offset is not None:
        offset = torch.tensor(index_offset, device=env.device, dtype=index_pos.dtype).unsqueeze(0).expand_as(index_pos)
        index_pos = index_pos + quat_apply(robot.data.body_quat_w[:, index_idx, :], offset)

    # Compute directions from reference point to fingers
    ref_to_thumb = thumb_pos - ref_pos
    ref_to_index = index_pos - ref_pos
    support_dirs = [ref_to_index]
    if middle_idx is not None:
        middle_pos = robot.data.body_pos_w[:, middle_idx, :]  # (num_envs, 3)
        if middle_offset is not None:
            offset = torch.tensor(middle_offset, device=env.device, dtype=middle_pos.dtype).unsqueeze(0).expand_as(middle_pos)
            middle_pos = middle_pos + quat_apply(robot.data.body_quat_w[:, middle_idx, :], offset)
        support_dirs.append(middle_pos - ref_pos)
    ref_to_index_middle = torch.stack(support_dirs, dim=0).mean(dim=0)

    # Normalize directions
    ref_to_thumb = ref_to_thumb / (torch.norm(ref_to_thumb, dim=-1, keepdim=True) + 1e-6)
    ref_to_index_middle = ref_to_index_middle / (
        torch.norm(ref_to_index_middle, dim=-1, keepdim=True) + 1e-6
    )

    # Inner product
    inner_product = torch.sum(ref_to_thumb * ref_to_index_middle, dim=-1)

    # Return negative inner product so that opposite sides give positive reward
    # Only apply when contact should happen
    avg_distance = 0.5 * (
        torch.norm(thumb_pos - ref_pos, dim=-1) + torch.norm(index_pos - ref_pos, dim=-1)
    )
    should_contact = _reference_contact_gate(
        command,
        hand=hand,
        gate_mode=gate_mode,
        pre_contact_frames=pre_contact_frames,
        post_contact_frames=post_contact_frames,
        gate_sigma_frames=gate_sigma_frames,
        time_shift_frames=contact_time_shift_frames,
        distance=avg_distance,
        soft_gate_radius=soft_gate_radius,
        soft_gate_temperature=soft_gate_temperature,
    )
    reward = -inner_product * should_contact
    if allowed_roles:
        reward = reward * utils.get_target_object_role_mask(
            env, command, allowed_roles
        ).float()
    return reward


def reward_wrist_look_at_object(
    env: ManagerBasedRLEnv,
    wrist_link: str = "right_wrist_yaw_link",
    forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hand: str = "right_hand",
    command_name: str = "motion",
) -> torch.Tensor:
    """Reward for wrist orientation pointing toward the object center during approach.

    Computes the dot product between the wrist's local forward axis (rotated to world
    frame) and the unit direction from wrist to object center. Returns a value in [-1, 1]
    where +1 means the wrist is perfectly aimed at the object.

    Gated by the motion data in_contact label: active only BEFORE contact (approach phase).
    Once in_contact is True, the reward is zeroed so it doesn't interfere with manipulation.

    Args:
        env: The environment.
        wrist_link: Name of the wrist rigid body link.
        forward_axis: Local-frame axis that should point toward the object (default: +X,
            verified visually for right_wrist_yaw_link on YAM robot).
        hand: Which hand's in_contact label to use ("right_hand" or "left_hand").
        command_name: Name of the motion command (for in_contact data).

    Returns:
        Reward tensor of shape (num_envs,). Range [-1, 1], use positive weight.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get in_contact label — fail loudly if missing
    in_contact = command.get_in_contact(hand)
    if in_contact is None:
        raise RuntimeError(
            f"reward_wrist_look_at_object: in_contact label not available for hand='{hand}' "
            f"on command='{command_name}'. The motion data must provide per-frame contact labels."
        )
    # Active only BEFORE contact (approach phase)
    approach_mask = 1.0 - in_contact

    robot = env.scene["robot"]

    # Validate wrist link exists
    body_names = robot.body_names
    if wrist_link not in body_names:
        raise RuntimeError(
            f"reward_wrist_look_at_object: wrist_link='{wrist_link}' not found in robot body names. "
            f"Available: {body_names}"
        )
    wrist_idx = body_names.index(wrist_link)

    # Get wrist position and orientation
    wrist_pos = robot.data.body_pos_w[:, wrist_idx, :]    # [num_envs, 3]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx, :]  # [num_envs, 4]

    # Get object position
    obj_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)

    # Rotate local forward axis to world frame
    forward_local = torch.tensor(
        forward_axis, device=env.device, dtype=wrist_pos.dtype
    ).unsqueeze(0).expand(env.num_envs, -1)  # [num_envs, 3]
    forward_world = quat_apply(wrist_quat, forward_local)  # [num_envs, 3]
    forward_world = forward_world / (torch.norm(forward_world, dim=-1, keepdim=True) + 1e-6)

    # Direction from wrist to object
    wrist_to_obj = obj_pos - wrist_pos  # [num_envs, 3]
    wrist_to_obj = wrist_to_obj / (torch.norm(wrist_to_obj, dim=-1, keepdim=True) + 1e-6)

    # Dot product: +1 when wrist forward axis points at object
    alignment = torch.sum(forward_world * wrist_to_obj, dim=-1)  # [num_envs]

    return alignment * approach_mask


def reward_wrist_smoothness_at_contact(
    env: ManagerBasedRLEnv,
    wrist_link: str = "right_wrist_yaw_link",
    command_name: str = "motion",
    window_seconds: float = 1.0,
    ee_radius: float = 0.15,
) -> torch.Tensor:
    """Penalize wrist velocity in a time window around the contact transition.

    Discourages sudden jerky wrist movements at the approach→grasp boundary.
    Active for ±window_seconds/2 around the first contact frame (from reference motion).
    Outside this window, returns zero.

    Combines linear and angular velocity into a unified penalty using an
    equivalent-radius approximation: angular velocity is scaled by ee_radius²
    to convert to equivalent linear velocity² at the end effector tip.
    penalty = lin_vel² + ee_radius² * ang_vel²

    Requires _first_contact_lookup to be populated on the TrackingCommand
    (automatically derived when object motions pkl has contact_points_right_hand).

    Args:
        env: The environment.
        wrist_link: Name of the wrist rigid body link.
        command_name: Name of the motion command.
        window_seconds: Total window duration in seconds, centered on contact transition.
        ee_radius: Distance (meters) from wrist center to end effector tip.
            Used to scale angular velocity to equivalent linear velocity.
            Default 0.15m (~15cm for YAM gripper).

    Returns:
        Penalty tensor of shape (num_envs,). Positive values, use negative weight.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get first contact lookup
    first_contact_lookup = getattr(command, "_first_contact_lookup", None)
    if first_contact_lookup is None:
        raise RuntimeError(
            "reward_wrist_smoothness_at_contact requires _first_contact_lookup but it is None. "
            "The object motions pkl must contain contact_points_right_hand."
        )

    robot = env.scene["robot"]

    # Validate wrist link
    body_names = robot.body_names
    if wrist_link not in body_names:
        raise RuntimeError(
            f"reward_wrist_smoothness_at_contact: wrist_link='{wrist_link}' not found. "
            f"Available: {body_names}"
        )
    wrist_idx = body_names.index(wrist_link)

    # Current frame in motion space
    current_frame = command.motion_start_time_steps + command.time_steps  # [num_envs]
    first_contact = first_contact_lookup[command.motion_ids]  # [num_envs]

    # Compute window in frames: ±half_window around contact
    sim_fps = 1.0 / env.step_dt  # typically 50
    half_window_frames = int(window_seconds * sim_fps / 2.0)

    # Signed distance to contact: negative=before, positive=after
    distance = current_frame - first_contact  # [num_envs]

    # Active only within window
    in_window = ((distance >= -half_window_frames) & (distance <= half_window_frames)).float()

    # Wrist velocity penalty: linear + angular (scaled by ee_radius² for equivalence)
    wrist_lin_vel = robot.data.body_lin_vel_w[:, wrist_idx, :]   # [num_envs, 3]
    wrist_ang_vel = robot.data.body_ang_vel_w[:, wrist_idx, :]   # [num_envs, 3]
    lin_vel_sq = torch.sum(torch.square(wrist_lin_vel), dim=-1)   # [num_envs]
    ang_vel_sq = torch.sum(torch.square(wrist_ang_vel), dim=-1)   # [num_envs]

    penalty = (lin_vel_sq + ee_radius * ee_radius * ang_vel_sq) * in_window

    return penalty


def reward_grasp_contact_center(
    env: ManagerBasedRLEnv,
    finger_links: list[str] | None = None,
    hand: str = "right_hand",
    command_name: str = "motion",
    exp_coeff: float = -10.0,
    gate_mode: str = "binary",
    pre_contact_frames: int = 0,
    post_contact_frames: int = 0,
    gate_sigma_frames: float = 5.0,
    contact_time_shift_frames: int = 0,
    lookahead_contact_center_frames: int = 0,
    contact_region_radius: float = 0.0,
    soft_gate_radius: float = 0.15,
    soft_gate_temperature: float = 0.03,
    min_valid_contact_center_norm: float = 1e-6,
    allowed_roles: list[str] | str | None = None,
) -> torch.Tensor:
    """Compute reward for fingers approaching the reference contact center.

    Use the per-hand object-local contact target from the motion library to guide
    grasping. This is a legacy contact-point mean or, for current RP2 packs, the
    nearest validated source-hand surface point. The reward uses an exponential
    kernel on the average fingertip distance to that target.

    Only applied when the contact label indicates contact should happen.

    Args:
        env: The environment.
        finger_links: List of finger link names to track. Defaults to the
            right hand fingertips if hand="right_hand", left otherwise.
        hand: Which hand's contact center to use ("left_hand" or "right_hand").
        command_name: Name of the tracking command term.
        exp_coeff: Exponential coefficient for distance-to-reward mapping (negative).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    if finger_links is None:
        if hand == "left_hand":
            finger_links = [
                "left_hand_thumb_2_link",
                "left_hand_index_1_link",
                "left_hand_middle_1_link",
            ]
        else:
            finger_links = [
                "right_hand_thumb_2_link",
                "right_hand_index_1_link",
                "right_hand_middle_1_link",
            ]

    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get per-hand contact center in world frame
    if hand == "left_hand":
        contact_center = command.object_contact_center_left
    else:
        contact_center = command.object_contact_center_right

    if contact_center is None:
        return torch.zeros(env.num_envs, device=env.device)

    if int(contact_time_shift_frames) != 0:
        shifted_steps = _clamp_motion_steps(
            command,
            command.motion_start_time_steps
            + command.time_steps
            + int(contact_time_shift_frames),
        )
        shifted_center_local = command.motion_lib.get_object_contact_center(
            command.motion_ids, shifted_steps, hand=hand
        )
        if shifted_center_local is not None:
            shifted_valid = torch.norm(shifted_center_local, dim=-1) > float(
                min_valid_contact_center_norm
            )
            shifted_center = _contact_center_world_from_local(command, shifted_center_local)
            contact_center = torch.where(shifted_valid[:, None], shifted_center, contact_center)

    if int(lookahead_contact_center_frames) > 0:
        contact_center = _lookahead_contact_center_world(
            command,
            hand=hand,
            fallback_center=contact_center,
            max_lookahead_frames=int(lookahead_contact_center_frames),
            min_valid_norm=float(min_valid_contact_center_norm),
        )

    robot = env.scene["robot"]
    body_names = robot.body_names

    # Get finger positions
    finger_indices = [body_names.index(link) for link in finger_links]
    finger_positions = robot.data.body_pos_w[:, finger_indices, :]  # (num_envs, num_fingers, 3)

    # Compute distance from each finger to the contact center
    contact_center_expanded = contact_center.unsqueeze(1)  # (num_envs, 1, 3)
    distances = torch.norm(
        finger_positions - contact_center_expanded, dim=-1
    )  # (num_envs, num_fingers)

    # Average over fingers
    avg_distance = distances.mean(dim=-1)  # (num_envs,)
    if float(contact_region_radius) > 0.0:
        avg_distance = torch.clamp(avg_distance - float(contact_region_radius), min=0.0)

    # Convert distance to reward (closer = higher reward)
    reward = torch.exp(exp_coeff * avg_distance)

    should_contact = _reference_contact_gate(
        command,
        hand=hand,
        gate_mode=gate_mode,
        pre_contact_frames=pre_contact_frames,
        post_contact_frames=post_contact_frames,
        gate_sigma_frames=gate_sigma_frames,
        time_shift_frames=contact_time_shift_frames,
        distance=avg_distance,
        soft_gate_radius=soft_gate_radius,
        soft_gate_temperature=soft_gate_temperature,
    )
    metric_side = "left" if hand == "left_hand" else "right"
    if hasattr(command, "metrics"):
        command.metrics[f"rp2_grasp_contact_center_distance/{metric_side}"] = avg_distance
        command.metrics[f"rp2_grasp_contact_center_gate/{metric_side}"] = should_contact
    reward = reward * should_contact
    if allowed_roles:
        reward = reward * utils.get_target_object_role_mask(
            env, command, allowed_roles
        ).float()
    return reward


def reward_lift_distance(
    env: ManagerBasedRLEnv, exp_coeff: float = -10.0, command_name: str = "motion"
) -> torch.Tensor:
    """Compute reward for lifting the object toward a goal position in the robot's local frame.

    Transform the object position into the robot's root frame and compute exponential
    reward based on distance to a fixed local goal (0.3m forward, 0.5m up).

    Args:
        env: The environment.
        exp_coeff: Exponential coefficient for distance-to-reward mapping (negative).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    obj_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)
    goal_pos = [0.3, 0.0, 0.5]
    # Transform obj_pos to local frame relative to robot root
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    obj_pos_local = quat_rotate_inverse(robot_quat, obj_pos - robot_pos, w_last=True)
    # Calculate distance between obj_pos_local and goal_pos
    distance = torch.norm(obj_pos_local - goal_pos, dim=-1)
    return torch.exp(exp_coeff * -distance)


def reward_lift_z_elevation(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Compute reward proportional to object z elevation above the table.

    Reward increases linearly with object height above 0.8m (assumed table +
    half-object height), clamped to a maximum of 0.3m.

    Args:
        env: The environment.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 0.3].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    obj_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)
    obj_z = obj_pos[:, 2]
    # Reward lifting above table height + half of the object height (assume it is 0.8m)
    return torch.clamp(obj_z - 0.8, min=0, max=0.3)


def reward_hand_table_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute penalty for hand contacting the table surface.

    Discourage the hand from pressing against the table. Uses force_matrix_w
    (filtered forces between table and hand links only), not net_forces_w which
    includes all forces on the table.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config for table-hand contact.

    Returns:
        Penalty tensor of shape (num_envs,) in [0, 1]. Clamped total force magnitude.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # force_matrix_w: [num_envs, num_sensor_bodies, num_filter_bodies, 3]
    contact_force = sensor.data.force_matrix_w
    force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))
    return torch.clamp(force_magnitude, max=1.0)


def active_articulation_joint_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    position_weight: float = 0.8,
    velocity_weight: float = 0.2,
    position_exp_coeff: float = -4.0,
    velocity_exp_coeff: float = -0.25,
) -> torch.Tensor:
    """Track reduced articulation coordinates without exposing extra object slots."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos, joint_vel, reference_pos, reference_vel, joint_valid = (
        utils.get_active_articulation_joint_tracking_state(env, command)
    )
    if joint_pos.shape[1] == 0:
        reward = torch.zeros(env.num_envs, device=env.device)
        if hasattr(command, "metrics"):
            command.metrics["active_articulation/valid_joint_count"] = reward
            command.metrics["active_articulation/joint_tracking_reward"] = reward
        return reward
    articulation = utils.get_active_articulation(env)
    limits = articulation.data.soft_joint_pos_limits
    half_range = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1e-3)
    joint_valid_float = joint_valid.to(dtype=joint_pos.dtype)
    valid_count = joint_valid_float.sum(dim=-1)
    divisor = valid_count.clamp_min(1.0)
    position_error = (
        torch.square((reference_pos - joint_pos) / half_range) * joint_valid_float
    ).sum(dim=-1) / divisor
    velocity_error = (
        torch.square(reference_vel - joint_vel) * joint_valid_float
    ).sum(dim=-1) / divisor
    position_reward = torch.exp(float(position_exp_coeff) * position_error)
    velocity_reward = torch.exp(float(velocity_exp_coeff) * velocity_error)

    if hasattr(command, "metrics"):
        command.metrics["active_articulation/valid_joint_count"] = valid_count
        command.metrics["active_articulation/joint_position_error_normalized"] = (
            torch.sqrt(position_error)
        )
        command.metrics["active_articulation/joint_velocity_error_rms"] = torch.sqrt(
            velocity_error
        )
    reward = (
        float(position_weight) * position_reward
        + float(velocity_weight) * velocity_reward
    )
    reward = reward * (valid_count > 0).to(dtype=reward.dtype)
    if hasattr(command, "metrics"):
        command.metrics["active_articulation/joint_tracking_reward"] = reward
    return reward


def _reference_hold_mask(command: TrackingCommand, num_envs: int, device) -> torch.Tensor:
    """Per-env bool: the reference is held/frozen (e.g. a pretrack hold), so reference motion must not activate.

    Shared interface for adapters: ``command.pretrack.active`` (bool tensor, True while the hold is active; the
    last held physics step can have steps_left == 0 while active is still True) or ``command.reference_hold_active``.
    Recomputed every step (the first free frame restores the reference velocity). Only the reference-moving
    branch is masked; actual hand contact still activates.
    """
    for source in (getattr(getattr(command, "pretrack", None), "active", None), getattr(command, "reference_hold_active", None)):
        if torch.is_tensor(source):
            return source.to(device=device).reshape(-1).bool()
    return torch.zeros(num_envs, dtype=torch.bool, device=device)


def _hands_in_contact(
    env: ManagerBasedRLEnv,
    command: TrackingCommand,
    hands: list[str],
    sensor_cfg: SceneEntityCfg | None,
    left_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """OR over hands of 'this hand touches the target object' (per-hand total force > 1 N on the filtered sensor).

    Each hand is thresholded separately (never a summed left+right force against a shared threshold). The
    forces come from utils.get_target_object_contact_forces, i.e. the per-object filtered hand sensors of the
    scene (object_to_hand_contact_sensor / object_to_left_hand_contact_sensor): robot hand bodies against the
    active object prim, not generic hand-anything contact.
    """
    in_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for hand in hands:
        if hand not in {"left_hand", "right_hand"}:
            raise ValueError(
                "object tracking contact_hands must contain only left_hand/right_hand, "
                f"got {hand!r}"
            )
        if hand == "left_hand":
            fallback_sensor_name = (
                left_sensor_cfg.name if left_sensor_cfg is not None else "object_to_left_hand_contact_sensor"
            )
        else:
            fallback_sensor_name = sensor_cfg.name if sensor_cfg is not None else "object_to_hand_contact_sensor"
        contact_force = utils.get_target_object_contact_forces(
            env,
            command,
            hand=hand,
            fallback_sensor_name=fallback_sensor_name,
        )
        if contact_force is None:
            raise RuntimeError(
                "object tracking could not resolve the requested contact sensor "
                f"for {hand}: {fallback_sensor_name}"
            )
        force_magnitude = torch.norm(contact_force, dim=-1)
        while force_magnitude.ndim > 1:
            force_magnitude = force_magnitude.sum(dim=-1)
        in_contact |= force_magnitude > 1.0
    return in_contact


def object_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg | None = None,
    contact_hands: list[str] | None = None,
    w_op: float = 0.5,  # object position weight
    w_or: float = 0.3,  # object rotation weight
    w_ov: float = 0.05,  # object linear velocity weight
    w_oav: float = 0.05,  # object angular velocity weight
    pos_exp_coeff: float = -100.0,
    ori_exp_coeff: float = -100.0,
    vel_exp_coeff: float = -5.0,
    ang_vel_exp_coeff: float = -5.0,
    track_active_entity_base: bool = False,
    kernel: str = "exp",
    pos_tolerance: float = 0.02,
    pos_sigma: float = 0.05,
    ori_tolerance: float = 0.0872664626,
    ori_sigma: float = 0.2617993878,
    require_contact: bool = True,
    activation: str = "legacy",
    ref_lin_speed_threshold: float = 0.05,
    ref_ang_speed_threshold: float = 0.1,
    object_tracking_version: str | None = None,
) -> torch.Tensor:
    """Compute object tracking reward following the OmniGrasp formulation.

    ``activation="contact_or_refmotion"`` (user default 2026-10-13) replaces the right-sensor-only gate:
    eligible = actual LEFT hand contact OR actual RIGHT hand contact OR the REFERENCE target clearly moving
    (|v_ref| > ref_lin_speed_threshold m/s or |w_ref| > ref_ang_speed_threshold rad/s, read from the same
    reference entity as the position/rotation targets; velocities are used only for activation, never as a
    velocity-error reward). One scalar gate (OR, no double counting); ``require_contact=False`` does not
    bypass it. ``activation="legacy"`` keeps the historical behaviour exactly.

    Weighted sum of exponential rewards for object position, orientation, linear
    velocity, and angular velocity tracking, gated by a contact indicator. Only
    active when the robot is in contact with the object. Contact encouragement
    is handled separately by reward_grasp.

    r_t^obj = (w_op * exp(c_p * ||err_p||) + w_or * exp(c_r * ||err_r||)
             + w_ov * exp(c_v * ||err_v||) + w_oav * exp(c_av * ||err_av||)) * 1{C}

    Args:
        env: The environment.
        command_name: Name of the tracking command containing object trajectory.
        sensor_cfg: Contact sensor config for finger-object contact detection.
        left_sensor_cfg: Left-hand contact sensor used by bilateral gating.
        contact_hands: Optional hands whose simulated contact is OR-reduced.
            ``None`` preserves the original right-sensor-only behavior.
        w_op: Weight for object position tracking.
        w_or: Weight for object orientation tracking.
        w_ov: Weight for object linear velocity tracking.
        w_oav: Weight for object angular velocity tracking.
        pos_exp_coeff: Exponential coefficient for position error.
        ori_exp_coeff: Exponential coefficient for orientation error.
        vel_exp_coeff: Exponential coefficient for linear velocity error.
        ang_vel_exp_coeff: Exponential coefficient for angular velocity error.

    Returns:
        Reward tensor of shape (num_envs,). Zero when not in contact.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get current target object state from scene. In full-scene mode this can
    # resolve to different scene assets per environment.
    if track_active_entity_base:
        (
            current_obj_pos,
            current_obj_quat,
            current_obj_lin_vel,
            current_obj_ang_vel,
            target_valid,
        ) = utils.get_active_entity_base_current_state(env, command)
        target_obj_pos = command.active_entity_root_pos
        target_obj_quat = command.active_entity_root_quat
        target_obj_lin_vel = command.active_entity_root_lin_vel
        target_obj_ang_vel = command.active_entity_root_ang_vel
    else:
        (
            current_obj_pos,
            current_obj_quat,
            current_obj_lin_vel,
            current_obj_ang_vel,
            target_valid,
        ) = utils.get_target_object_current_state(env, command)
        target_obj_pos = command.target_object_root_pos
        target_obj_quat = command.target_object_root_quat
        target_obj_lin_vel = command.target_object_lin_vel
        target_obj_ang_vel = command.target_object_ang_vel

    # Position tracking. Default exp kernel (exp(c*d)) preserves the legacy
    # OmniGrasp behavior; kernel="cauchy" (or "gaussian") uses the RP2 shared
    # tolerance_tracking_score for a gentler kernel with real gradient out to the
    # deviation cutoff (RP2 parity — see gear_sonic/rp2/reward_kernels.py).
    pos_error = torch.norm(target_obj_pos - current_obj_pos, dim=-1)
    if kernel == "exp":
        pos_reward = torch.exp(pos_exp_coeff * pos_error)
    else:
        pos_reward = tolerance_tracking_score(
            pos_error, tolerance=pos_tolerance, sigma=pos_sigma, kernel=kernel
        )

    # Orientation tracking
    ori_error = quat_error_magnitude(target_obj_quat, current_obj_quat)
    if kernel == "exp":
        ori_reward = torch.exp(ori_exp_coeff * ori_error)
    else:
        ori_reward = tolerance_tracking_score(
            ori_error, tolerance=ori_tolerance, sigma=ori_sigma, kernel=kernel
        )

    # Linear velocity tracking
    lin_vel_error = torch.norm(target_obj_lin_vel - current_obj_lin_vel, dim=-1)
    lin_vel_reward = torch.exp(vel_exp_coeff * lin_vel_error)

    # Angular velocity tracking
    ang_vel_error = torch.norm(target_obj_ang_vel - current_obj_ang_vel, dim=-1)
    ang_vel_reward = torch.exp(ang_vel_exp_coeff * ang_vel_error)

    # print("obj_po", target_obj_pos, current_obj_pos)
    # print("d0:", pos_error, pos_reward, ori_error, ori_reward)
    # print("d1:", lin_vel_error, lin_vel_reward, ang_vel_error, ang_vel_reward)

    # Contact indicator 1{C}. Preserve the original right-sensor behavior by
    # default; opt-in bilateral configs use an OR gate so two-hand contact does
    # not double the object-tracking reward.
    activation_mode = str(activation).lower()
    if activation_mode == "contact_or_refmotion":
        hands = list(contact_hands) if contact_hands else ["left_hand", "right_hand"]
        contact_any = _hands_in_contact(env, command, hands, sensor_cfg, left_sensor_cfg)
        ref_moving = (torch.norm(target_obj_lin_vel, dim=-1) > float(ref_lin_speed_threshold)) | (
            torch.norm(target_obj_ang_vel, dim=-1) > float(ref_ang_speed_threshold)
        )
        ref_moving = ref_moving & ~_reference_hold_mask(command, env.num_envs, env.device)
        in_contact = (contact_any | ref_moving).to(pos_reward.dtype)
        if hasattr(command, "metrics"):
            command.metrics["active_entity_tracking/eligible_contact"] = contact_any.float()
            command.metrics["active_entity_tracking/eligible_ref_moving"] = ref_moving.float()
    elif activation_mode != "legacy":
        raise ValueError(f"object_tracking_reward activation must be 'legacy' or 'contact_or_refmotion', got {activation!r}")
    elif not require_contact:
        in_contact = torch.ones(env.num_envs, device=env.device, dtype=pos_reward.dtype)
    elif contact_hands is None:
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        contact_force = sensor.data.force_matrix_w
        force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))
        in_contact = (force_magnitude > 1.0).float()
    else:
        in_contact = _hands_in_contact(env, command, list(contact_hands), sensor_cfg, left_sensor_cfg).float()

    # Tracking reward gated by contact: (tracking) * 1{C}
    tracking_reward = (
        w_op * pos_reward + w_or * ori_reward + w_ov * lin_vel_reward + w_oav * ang_vel_reward
    )
    if hasattr(command, "metrics"):
        command.metrics["active_entity_tracking/position_error_m"] = pos_error
        command.metrics["active_entity_tracking/orientation_error_rad"] = ori_error
        command.metrics["active_entity_tracking/linear_velocity_error_mps"] = (
            lin_vel_error
        )
        command.metrics["active_entity_tracking/angular_velocity_error_radps"] = (
            ang_vel_error
        )
        command.metrics["active_entity_tracking/contact_gate"] = in_contact
        command.metrics["active_entity_tracking/ungated_reward"] = tracking_reward

    # A missing/invalid target (the getter's validity) never earns tracking reward, however it was activated.
    if torch.is_tensor(target_valid):
        valid_f = target_valid.reshape(env.num_envs, -1).all(dim=-1).to(tracking_reward.dtype)
        if hasattr(command, "metrics"):
            command.metrics["active_entity_tracking/target_valid"] = valid_f
        return tracking_reward * in_contact * valid_f
    return tracking_reward * in_contact


class ContactFirstGraspFactor(ManagerTermBase):
    """Multiply source-neutral tracking reward by a bounded physical grasp score.

    The five-source policy has two hand primitives, unlike Inspire's individual
    finger actions. We transfer ContactV4's contact-first objective but use the
    common G1 thumb/index/middle/palm bodies and per-asset USD surface samples.
    Reference wrist-to-surface distance supplies coarse intent for every source;
    no source contact label or hand-pose target enters this reward.
    """

    TRACKING_TERMS = (
        "tracking_anchor_pos", "tracking_anchor_ori", "tracking_relative_body_pos",
        "tracking_relative_body_ori_weighted", "tracking_body_linvel",
        "tracking_body_angvel", "object_tracking_reward",
    )
    HAND_LINKS = (
        ("left_hand_thumb_2_link", "left_hand_index_1_link",
         "left_hand_middle_1_link", "left_hand_palm_link"),
        ("right_hand_thumb_2_link", "right_hand_index_1_link",
         "right_hand_middle_1_link", "right_hand_palm_link"),
    )
    WRIST_LINKS = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        command = env.command_manager.get_term("motion")
        roster = command.asset_compatible_roster
        if roster is None or len(roster["slots"]) != env.num_envs:
            raise ValueError("Contact-first grasp requires a fixed-asset cohort for every environment")
        cache = {}
        surfaces = []
        for slot in roster["slots"]:
            path = slot.get("active_surface_path")
            if not path:
                raise ValueError("Contact-first cohort slot lacks active_surface_path")
            if path not in cache:
                value = np.load(path, allow_pickle=False)
                if value.shape != (512, 3) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid active surface samples: {path}")
                cache[path] = torch.as_tensor(value, device=env.device)
            surfaces.append(cache[path])
        self.surface_local = torch.stack(surfaces, dim=0)
        names = env.scene["robot"].body_names
        self.hand_ids = [[names.index(name) for name in hand] for hand in self.HAND_LINKS]
        wrist_ids = _get_body_indexes(command, list(self.WRIST_LINKS))
        if len(wrist_ids) != 2 or [command.cfg.body_names[index] for index in wrist_ids] != list(self.WRIST_LINKS):
            raise ValueError("Reference wrist links are missing or out of order")
        self.ref_wrist_ids = wrist_ids

    def __call__(
        self,
        env,
        strength: float = 0.40,
        intent_distance_m: float = 0.25,
        intent_softness_m: float = 0.05,
        proximity_sigma_m: float = 0.05,
        contact_distance_m: float = 0.04,
        contact_threshold_n: float = 0.1,
        tracking_terms: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        if not 0.0 <= strength <= 1.0 or min(intent_softness_m, proximity_sigma_m, contact_distance_m) <= 0:
            raise ValueError("Invalid contact-first reward parameters")
        command = env.command_manager.get_term("motion")
        obj_pos, obj_quat, _, _, _ = utils.get_target_object_current_state(env, command)
        n_env, n_surface, _ = self.surface_local.shape
        surface_world = quat_apply(
            obj_quat[:, None, :].expand(n_env, n_surface, 4), self.surface_local
        ) + obj_pos[:, None, :]
        ref_pos = command.target_object_root_pos
        ref_quat = command.target_object_root_quat
        ref_surface = quat_apply(
            ref_quat[:, None, :].expand(n_env, n_surface, 4), self.surface_local
        ) + ref_pos[:, None, :]
        ref_wrists = command.body_pos_w[:, self.ref_wrist_ids, :]
        ref_gap = torch.cdist(ref_wrists, ref_surface).amin(dim=-1)
        intent = torch.sigmoid((intent_distance_m - ref_gap) / intent_softness_m)

        robot_pos = env.scene["robot"].data.body_pos_w
        hand_scores = []
        opposed_scores = []
        for side, sensor_name in enumerate((
            "object_to_left_hand_contact_sensor", "object_to_hand_contact_sensor"
        )):
            pad_pos = robot_pos[:, self.hand_ids[side], :]
            distances = torch.cdist(pad_pos, surface_world)
            nearest = distances.argmin(dim=-1)
            nearest_surface = torch.gather(
                surface_world, 1, nearest[..., None].expand(-1, -1, 3)
            )
            gap = (torch.linalg.vector_norm(pad_pos - nearest_surface, dim=-1) - 0.01).clamp_min(0)
            proximity = torch.exp(-torch.square(gap / proximity_sigma_m))
            direction = (pad_pos - nearest_surface) / torch.linalg.vector_norm(
                pad_pos - nearest_surface, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            opposition = (0.5 * (1.0 - (direction[:, 0:1] * direction[:, 1:3]).sum(dim=-1))).clamp(0, 1)
            pair_proximity = torch.sqrt(proximity[:, 0:1] * proximity[:, 1:3])
            approach = pair_proximity.amax(dim=-1)
            enclosure = (pair_proximity * opposition).amax(dim=-1)
            hand = "left_hand" if side == 0 else "right_hand"
            forces = utils.get_target_object_contact_forces(
                env, command, hand=hand, fallback_sensor_name=sensor_name
            )
            if forces is None:
                raise RuntimeError(f"Missing {hand} object contact forces")
            forces = forces.reshape(n_env, -1, 3)
            if forces.shape[1] != 4:
                raise ValueError(f"{hand} expected thumb/index/middle/palm forces, got {forces.shape}")
            contact = (torch.linalg.vector_norm(forces, dim=-1) > contact_threshold_n).float()
            proxy = contact * (1.0 - gap / contact_distance_m).clamp(0, 1)
            opposed = proxy[:, 0] * (proxy[:, 1:3] * opposition).amax(dim=-1)
            any_finger = proxy[:, :3].amax(dim=-1)
            palm = proxy[:, 3]
            score = 0.05 * approach + 0.10 * enclosure + 0.65 * opposed + 0.15 * any_finger + 0.05 * palm
            score = torch.where(contact.amax(dim=-1) > 0, score, score.clamp(max=0.15))
            hand_scores.append(score.clamp(0, 1))
            opposed_scores.append(opposed)
        hand_scores = torch.stack(hand_scores, dim=-1)
        deficit = ((1.0 - hand_scores) * intent).sum(dim=-1) / intent.sum(dim=-1).clamp_min(1.0)
        factor = (1.0 - strength * deficit).clamp(min=1.0 - strength, max=1.0)

        manager = env.reward_manager
        term_idx = manager._term_names.index("contact_first_grasp_factor")
        # A config that replaces a tracking term must name its replacement here;
        # a missing term raises instead of silently dropping it from the factor.
        names = self.TRACKING_TERMS if tracking_terms is None else tuple(tracking_terms)
        tracked_indices = [manager._term_names.index(name) for name in names]
        if any(index >= term_idx for index in tracked_indices):
            raise RuntimeError("Contact-first factor must run after its tracking terms")
        tracking_per_second = manager._step_reward[:, tracked_indices].sum(dim=-1)
        command.metrics["contact_first/factor"] = factor
        command.metrics["contact_first/intent"] = intent.mean(dim=-1)
        command.metrics["contact_first/opposed"] = torch.stack(opposed_scores, dim=-1).mean(dim=-1)
        return (factor - 1.0) * tracking_per_second


def object_tracking_reward_multi(
    env: ManagerBasedRLEnv,
    command_name: str,
    w_op: float = 0.5,
    w_or: float = 0.0,
    w_ov: float = 0.0,
    w_oav: float = 0.0,
    pos_exp_coeff: float = -100.0,
    ori_exp_coeff: float = -100.0,
    vel_exp_coeff: float = -5.0,
    ang_vel_exp_coeff: float = -5.0,
    mask_name: str = "interaction",
    role_names: list[str] | str | None = None,
    scale_start_step: int | None = None,
    scale_end_step: int | None = None,
    start_scale: float = 1.0,
    end_scale: float = 1.0,
    contact_gate_hand: str | None = None,
    contact_gate_mode: str = "binary",
    contact_gate_pre_frames: int = 0,
    contact_gate_post_frames: int = 0,
    contact_gate_sigma_frames: float = 5.0,
    contact_gate_time_shift_frames: int = 0,
    contact_gate_soft_radius: float = 0.15,
    contact_gate_soft_temperature: float = 0.03,
    track_active_entity_base: bool = False,
    activation: str = "legacy",
    ref_lin_speed_threshold: float = 0.05,
    ref_ang_speed_threshold: float = 0.1,
    contact_sensor_name: str = "object_to_hand_contact_sensor",
    left_contact_sensor_name: str = "object_to_left_hand_contact_sensor",
    object_tracking_version: str | None = None,
) -> torch.Tensor:
    """Mask-aware object tracking reward for RP2 full-scene data.

    ``activation="contact_or_refmotion"`` (user default 2026-10-13): the averaged tracking reward is gated by
    eligible = actual LEFT contact OR actual RIGHT contact OR the reference target of an active object clearly
    moving (|v_ref| > ref_lin_speed_threshold m/s or |w_ref| > ref_ang_speed_threshold rad/s, from the
    hold-aware reference getters, never the simulated object); the old reference-label ``contact_gate_hand``
    is NOT applied in this mode. ``activation="legacy"`` keeps the historical behaviour exactly.

    Tracks every object active in the per-frame interaction/motion mask and
    averages over the active set. If no object is active in an env, the reward is
    zero for that env instead of falling back to object index 0.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    if track_active_entity_base:
        max_objects = 1
        current_pos, current_quat, current_lin_vel, current_ang_vel, valid = (
            utils.get_active_entity_base_current_state(env, command)
        )
        current_pos = current_pos[:, None, :]
        current_quat = current_quat[:, None, :]
        current_lin_vel = current_lin_vel[:, None, :]
        current_ang_vel = current_ang_vel[:, None, :]
        valid = valid[:, None]
        target_pos = command.active_entity_root_pos[:, None, :]
        target_quat = command.active_entity_root_quat[:, None, :]
        target_lin_vel = command.active_entity_root_lin_vel[:, None, :]
        target_ang_vel = command.active_entity_root_ang_vel[:, None, :]
    else:
        max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
        current_pos, current_quat, current_lin_vel, current_ang_vel, valid = (
            utils.get_all_object_current_state(env, command, max_objects=max_objects)
        )
        target_pos = command.object_root_pos[:, :max_objects]
        target_quat = command.object_root_quat[:, :max_objects]

        current_steps = command.motion_start_time_steps + command.time_steps
        target_lin_vel = command.motion_lib.get_object_lin_vel(
            command.motion_ids, current_steps
        )[:, :max_objects]
        target_ang_vel = command.motion_lib.get_object_ang_vel(
            command.motion_ids, current_steps
        )[:, :max_objects]

    if mask_name == "motion":
        mask = command.object_motion_mask
    elif mask_name == "interaction":
        mask = command.object_interaction_mask
    elif mask_name == "active_entity_valid":
        if not track_active_entity_base:
            raise ValueError(
                "active_entity_valid requires track_active_entity_base=true"
            )
        mask = None
    else:
        raise ValueError(
            "object_tracking_reward_multi mask_name must be motion, "
            "interaction, or active_entity_valid"
        )
    if mask is None:
        active = valid
    else:
        mask = mask.to(valid.device)
        if mask.ndim == 1:
            mask = mask.view(-1, 1)
        else:
            mask = mask.reshape(mask.shape[0], -1)
        if mask.shape[1] < max_objects:
            padded = torch.zeros(env.num_envs, max_objects, dtype=mask.dtype, device=valid.device)
            padded[:, : mask.shape[1]] = mask
            mask = padded
        active = (mask[:, :max_objects] > 0.5) & valid
    if role_names:
        if isinstance(role_names, str):
            role_set = {item.strip() for item in role_names.split(",") if item.strip()}
        else:
            role_set = {str(item).strip() for item in role_names if str(item).strip()}
        if role_set:
            role_mask = torch.zeros_like(active)
            for env_idx, roles in enumerate(command._object_role_names_by_env(max_objects)):
                for obj_idx, role in enumerate(roles[:max_objects]):
                    if role in role_set:
                        role_mask[env_idx, obj_idx] = True
            active = active & role_mask

    pos_error = torch.norm(target_pos - current_pos, dim=-1)
    pos_reward = torch.exp(pos_exp_coeff * pos_error)

    ori_error = quat_error_magnitude(
        target_quat.reshape(-1, 4), current_quat.reshape(-1, 4)
    ).view(env.num_envs, max_objects)
    ori_reward = torch.exp(ori_exp_coeff * ori_error)

    lin_vel_error = torch.norm(target_lin_vel - current_lin_vel, dim=-1)
    lin_vel_reward = torch.exp(vel_exp_coeff * lin_vel_error)

    ang_vel_error = torch.norm(target_ang_vel - current_ang_vel, dim=-1)
    ang_vel_reward = torch.exp(ang_vel_exp_coeff * ang_vel_error)

    tracking_reward = (
        w_op * pos_reward + w_or * ori_reward + w_ov * lin_vel_reward + w_oav * ang_vel_reward
    )
    active_f = active.float()
    denom = active_f.sum(dim=-1).clamp_min(1.0)
    reward = (tracking_reward * active_f).sum(dim=-1) / denom

    scale = float(end_scale)
    if scale_start_step is None and scale_end_step is None:
        scale = 1.0
    else:
        step = float(getattr(env, "common_step_counter", 0))
        start_step = float(scale_start_step or 0)
        end_step = float(scale_end_step if scale_end_step is not None else start_step)
        if end_step <= start_step:
            scale = float(end_scale if step >= start_step else start_scale)
        else:
            alpha = min(max((step - start_step) / (end_step - start_step), 0.0), 1.0)
            scale = float(start_scale + alpha * (end_scale - start_scale))
    scale_tensor = torch.full((env.num_envs,), scale, device=env.device)
    command.metrics["rp2_object_tracking_reward_scale"] = scale_tensor
    command.metrics["rp2_object_tracking_reward_active_count"] = active_f.sum(dim=-1)
    command.metrics["active_entity_tracking/position_error_m"] = (
        pos_error * active_f
    ).sum(dim=-1) / denom
    command.metrics["active_entity_tracking/orientation_error_rad"] = (
        ori_error * active_f
    ).sum(dim=-1) / denom
    command.metrics["active_entity_tracking/linear_velocity_error_mps"] = (
        lin_vel_error * active_f
    ).sum(dim=-1) / denom
    command.metrics["active_entity_tracking/angular_velocity_error_radps"] = (
        ang_vel_error * active_f
    ).sum(dim=-1) / denom
    command.metrics["active_entity_tracking/ungated_reward"] = reward
    activation_mode = str(activation).lower()
    if activation_mode == "contact_or_refmotion":
        contact_any = _hands_in_contact(
            env,
            command,
            ["left_hand", "right_hand"],
            SceneEntityCfg(name=contact_sensor_name),
            SceneEntityCfg(name=left_contact_sensor_name),
        )
        ref_moving_per_object = (
            torch.norm(target_lin_vel, dim=-1) > float(ref_lin_speed_threshold)
        ) | (torch.norm(target_ang_vel, dim=-1) > float(ref_ang_speed_threshold))
        ref_moving = (ref_moving_per_object & active).any(dim=-1)
        ref_moving = ref_moving & ~_reference_hold_mask(command, env.num_envs, env.device)
        eligible = contact_any | ref_moving
        command.metrics["active_entity_tracking/eligible_contact"] = contact_any.float()
        command.metrics["active_entity_tracking/eligible_ref_moving"] = ref_moving.float()
        command.metrics["rp2_object_tracking_contact_gate"] = eligible.float()
        return reward * eligible.float() * scale_tensor
    if activation_mode != "legacy":
        raise ValueError(
            f"object_tracking_reward_multi activation must be 'legacy' or 'contact_or_refmotion', got {activation!r}"
        )
    if contact_gate_hand:
        gate = _reference_contact_gate(
            command,
            hand=contact_gate_hand,
            gate_mode=contact_gate_mode,
            pre_contact_frames=contact_gate_pre_frames,
            post_contact_frames=contact_gate_post_frames,
            gate_sigma_frames=contact_gate_sigma_frames,
            time_shift_frames=contact_gate_time_shift_frames,
            soft_gate_radius=contact_gate_soft_radius,
            soft_gate_temperature=contact_gate_soft_temperature,
        )
        command.metrics["rp2_object_tracking_contact_gate"] = gate
        reward = reward * gate
    return reward * scale_tensor


def rp2_wrist_contact_surface_proximity_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    robot_wrist_body_names: list[str] | None = None,
    excess_exp_coeff: float = -10.0,
    allowed_roles: list[str] | str | None = None,
) -> torch.Tensor:
    """Drive labeled-contact wrists onto the object surface.

    For each hand whose reference in_contact label is active, compare the sim
    wrist-to-object-center distance against the (uniformly scaled, calibrated)
    SOMA wrist relation distance at the same frame and reward driving the
    EXCESS to zero. Since the SOMA wrist genuinely touches the object, zero
    excess puts the sim hand on the surface without needing collision
    geometry. Hands whose label is inactive contribute nothing, so the term
    never fights approach/release phases.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    if robot_wrist_body_names is None:
        robot_wrist_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    hands = ["left_hand", "right_hand"]

    object_pos_w, _, _, _, _ = utils.get_target_object_current_state(env, command)
    steps = command.motion_start_time_steps + command.time_steps

    total = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    gates = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    role_eligible = (
        utils.get_target_object_role_mask(env, command, allowed_roles)
        if allowed_roles
        else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    )
    for hand, body_name in zip(hands, robot_wrist_body_names, strict=True):
        if body_name not in command.cfg.body_names:
            continue
        in_contact = command.get_in_contact(hand)
        ref_vec = command.motion_lib.get_object_wrist_relation_vector(
            command.motion_ids, steps, hand=hand
        )
        if in_contact is None or ref_vec is None:
            continue
        if ref_vec.ndim == 3:
            ref_vec = ref_vec[:, 0]
        ref_dist = torch.linalg.vector_norm(ref_vec, dim=-1)
        valid = (in_contact > 0.5) & role_eligible
        mask = command.motion_lib.get_object_wrist_relation_mask(
            command.motion_ids, steps, hand=hand
        )
        if mask is not None:
            mask = mask.reshape(env.num_envs, -1)[:, 0]
            valid = valid & (mask > 0.5)
        body_idx = command.cfg.body_names.index(body_name)
        wrist_pos_w = command.robot_body_pos_w[:, body_idx]
        sim_dist = torch.linalg.vector_norm(object_pos_w - wrist_pos_w, dim=-1)
        excess = torch.clamp(sim_dist - ref_dist, min=0.0)
        score = torch.exp(excess_exp_coeff * excess)
        gate = valid.float()
        total = total + score * gate
        gates = gates + gate
    reward = total / gates.clamp_min(1.0)
    command.metrics["rp2_wrist_contact_proximity/reward"] = reward
    command.metrics["rp2_wrist_contact_proximity/active_hands"] = gates
    return reward


def rp2_soma_object_relation_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    mask_name: str = "interaction",
    object_role_names: list[str] | str | None = None,
    robot_hand_body_names: list[str] | None = None,
    soma_hand_indices: list[int] | None = None,
    dist_exp_coeff: float = -25.0,
    contact_region_radius: float = 0.0,
    soma_joints_are_root_local: bool = False,
) -> torch.Tensor:
    """Opt-in RP2 reward matching SOMA/source hand-object graph edges.

    Current RP2 processing bakes dense per-hand object-local relation vectors
    into the object motion PKL as object_relation_vector_left/right.  This
    reward compares the simulated G1 hand position in the current object's local
    frame against those dense source edges, so it supervises approach/release as
    well as contact frames.  If dense relation tensors are absent, it falls back
    to the older full-SOMA-joint distance proxy.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
    if max_objects <= 0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    current_pos, _, _, _, valid = utils.get_all_object_current_state(
        env, command, max_objects=max_objects
    )

    if mask_name == "motion":
        mask = command.object_motion_mask
    elif mask_name == "relation":
        mask = None
    elif mask_name in {"none", "all"}:
        mask = None
    else:
        mask = command.object_interaction_mask
    if mask is None:
        active = valid
    else:
        mask = mask.to(valid.device)
        if mask.ndim == 1:
            mask = mask.view(-1, 1)
        else:
            mask = mask.reshape(mask.shape[0], -1)
        if mask.shape[1] < max_objects:
            padded = torch.zeros(env.num_envs, max_objects, dtype=mask.dtype, device=valid.device)
            padded[:, : mask.shape[1]] = mask
            mask = padded
        active = (mask[:, :max_objects] > 0.5) & valid

    # Match GRAIL single-object semantics: the sole ``Object`` scene entity is
    # already the target. Role metadata is only needed to disambiguate objects
    # in a genuine multi-object scene.
    if object_role_names and max_objects > 1:
        active = utils.get_object_role_mask(
            env, command, object_role_names, max_objects, valid=active
        )
    if not active.any():
        zero = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        command.metrics["rp2_soma_object_relation_active_count"] = zero
        command.metrics["rp2_soma_object_relation_error"] = zero
        command.metrics["rp2_soma_object_relation_reward_raw"] = zero
        return zero

    if robot_hand_body_names is None:
        robot_hand_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    has_dense_relation = any(
        hasattr(command.motion_lib, attr)
        for attr in (
            "_motion_object_relation_vector_left",
            "_motion_object_relation_vector_right",
        )
    )
    if has_dense_relation:
        from gear_sonic.isaac_utils import rotations as isaac_utils_rotations  # noqa: PLC0415

        current_steps = command.motion_start_time_steps + command.time_steps
        _, current_quat, _, _, valid = utils.get_all_object_current_state(
            env, command, max_objects=max_objects
        )
        reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        error_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        active_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        side_specs = (
            ("left_hand", [robot_hand_body_names[0]]),
            (
                "right_hand",
                [robot_hand_body_names[1] if len(robot_hand_body_names) > 1 else robot_hand_body_names[0]],
            ),
        )
        for hand, body_names in side_specs:
            relation_vector = command.motion_lib.get_object_relation_vector(
                command.motion_ids, current_steps, hand=hand
            )
            if relation_vector is None:
                continue
            if relation_vector.ndim == 2:
                relation_vector = relation_vector.unsqueeze(1)
            if relation_vector.ndim != 3 or relation_vector.shape[-1] != 3:
                raise ValueError(
                    "Expected object relation vectors with shape [batch, objects, 3], "
                    f"got {tuple(relation_vector.shape)} for {hand}"
                )
            relation_vector = relation_vector[:, :max_objects]
            relation_mask = command.motion_lib.get_object_relation_mask(
                command.motion_ids, current_steps, hand=hand
            )
            if relation_mask is None:
                relation_mask = torch.norm(relation_vector, dim=-1) > 1e-6
            else:
                if relation_mask.ndim == 1:
                    relation_mask = relation_mask.unsqueeze(1)
                if relation_mask.ndim != 2:
                    raise ValueError(
                        "Expected object relation masks with shape [batch, objects], "
                        f"got {tuple(relation_mask.shape)} for {hand}"
                    )
                relation_mask = relation_mask[:, :max_objects] > 0.5

            body_indices = _get_body_indexes(command, body_names)
            if not body_indices:
                continue
            hand_pos_w = command.robot_body_pos_w[:, body_indices].mean(dim=1)
            hand_object_vec_w = hand_pos_w[:, None, :] - current_pos
            hand_object_vec_l = isaac_utils_rotations.quat_rotate_inverse(
                current_quat.reshape(-1, 4),
                hand_object_vec_w.reshape(-1, 3),
                w_last=False,
            ).view(env.num_envs, max_objects, 3)
            dist_error = torch.norm(hand_object_vec_l - relation_vector, dim=-1)
            if float(contact_region_radius) > 0.0:
                dist_error = torch.clamp(dist_error - float(contact_region_radius), min=0.0)
            relation_reward = torch.exp(dist_exp_coeff * dist_error)
            side_active = active & valid & relation_mask
            side_active_f = side_active.float()
            reward_sum += (relation_reward * side_active_f).sum(dim=-1)
            error_sum += (dist_error * side_active_f).sum(dim=-1)
            active_sum += side_active_f.sum(dim=-1)

        if active_sum.any():
            denom = active_sum.clamp_min(1.0)
            reward = reward_sum / denom
            command.metrics["rp2_soma_object_relation_active_count"] = active_sum
            command.metrics["rp2_soma_object_relation_error"] = error_sum / denom
            command.metrics["rp2_soma_object_relation_reward_raw"] = reward
            return reward

        zero = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        command.metrics["rp2_soma_object_relation_active_count"] = zero
        command.metrics["rp2_soma_object_relation_error"] = zero
        command.metrics["rp2_soma_object_relation_reward_raw"] = zero
        return zero

    if not all(
        hasattr(command.motion_lib, attr)
        for attr in ("_motion_soma_joints", "_motion_soma_root_quat", "_motion_soma_transl")
    ):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    target_pos = command.object_root_pos[:, :max_objects]
    robot_hand_indices = _get_body_indexes(command, robot_hand_body_names)
    if not robot_hand_indices:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    robot_hands_w = command.robot_body_pos_w[:, robot_hand_indices]

    if soma_hand_indices is None:
        # gear_sonic/data_process/extract_soma_joints_from_bvh.py:
        # LeftHand=9, RightHand=15 in the 26-joint SOMA subset.
        soma_hand_indices = [9, 15]
    soma_joints = command.soma_joints
    max_soma_idx = max(int(idx) for idx in soma_hand_indices)
    if soma_joints.ndim != 3 or soma_joints.shape[1] <= max_soma_idx:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    current_steps = command.motion_start_time_steps + command.time_steps
    soma_transl = command.motion_lib.get_soma_transl(command.motion_ids, current_steps)
    if getattr(command.motion_lib, "soma_y_up", False):
        # Rotate Y-up translation to Z-up with the same +90deg X convention used
        # for SOMA root quaternions: (x, y, z) -> (x, -z, y).
        soma_transl = torch.stack(
            (soma_transl[:, 0], -soma_transl[:, 2], soma_transl[:, 1]), dim=-1
        )
    soma_transl = soma_transl + env.scene.env_origins
    soma_hands_local = soma_joints[:, [int(idx) for idx in soma_hand_indices]]
    if soma_joints_are_root_local:
        # Alternate/legacy SOMA tensors may store joints in the root-local frame.
        # Current RP2 extraction stores root-centered, world-oriented Z-up vectors,
        # so the default path avoids rotating them by the root a second time.
        num_soma_hands = soma_hands_local.shape[1]
        soma_root_quat = command.soma_root_quat_w
        soma_hands_offset = quat_apply(
            soma_root_quat[:, None, :].expand(-1, num_soma_hands, -1).reshape(-1, 4),
            soma_hands_local.reshape(-1, 3),
        ).view(env.num_envs, num_soma_hands, 3)
    else:
        soma_hands_offset = soma_hands_local
    soma_hands_w = soma_transl[:, None, :] + soma_hands_offset

    soma_valid = (
        soma_hands_local.abs().sum(dim=(-1, -2)) > 1e-6
    ) & torch.isfinite(soma_hands_w).all(dim=(-1, -2))
    if not soma_valid.any():
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    robot_dist = torch.norm(
        robot_hands_w[:, :, None, :] - current_pos[:, None, :, :],
        dim=-1,
    ).min(dim=1).values
    soma_dist = torch.norm(
        soma_hands_w[:, :, None, :] - target_pos[:, None, :, :],
        dim=-1,
    ).min(dim=1).values
    dist_error = torch.abs(robot_dist - soma_dist)
    relation_reward = torch.exp(dist_exp_coeff * dist_error)

    active = active & soma_valid[:, None]
    active_f = active.float()
    denom = active_f.sum(dim=-1).clamp_min(1.0)
    reward = (relation_reward * active_f).sum(dim=-1) / denom
    command.metrics["rp2_soma_object_relation_active_count"] = active_f.sum(dim=-1)
    command.metrics["rp2_soma_object_relation_error"] = (
        dist_error * active_f
    ).sum(dim=-1) / denom
    command.metrics["rp2_soma_object_relation_reward_raw"] = reward
    return reward


def rp2_soma_wrist_object_relation_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    object_role_names: list[str] | str | None = None,
    robot_wrist_body_names: list[str] | None = None,
    dist_exp_coeff: float = -20.0,
    relation_tolerance: float = 0.02,
    proximity_distance: float = 0.25,
    proximity_temperature: float = 0.05,
    far_weight: float = 0.05,
) -> torch.Tensor:
    """Track stable SOMA wrist-object edges with a smooth source-proximity gate.

    Reference vectors and wrist-to-surface distances are generated from the
    exact ``LeftHand``/``RightHand`` source joints.  The proximity gate depends
    only on the source motion, so a policy cannot disable this reward by moving
    its simulated wrist away from the object.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    metric_prefix = "rp2_soma_wrist_object_relation"
    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)

    def _zero_metrics() -> torch.Tensor:
        zero = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        if hasattr(command, "metrics"):
            command.metrics[f"{metric_prefix}_active_count"] = zero
            command.metrics[f"{metric_prefix}_error"] = zero
            command.metrics[f"{metric_prefix}_proximity_gate"] = zero
            command.metrics[f"{metric_prefix}_surface_distance"] = zero
            command.metrics[f"{metric_prefix}_reward_raw"] = zero
            for side in ("left", "right"):
                command.metrics[f"{metric_prefix}_{side}_active_count"] = zero
                command.metrics[f"{metric_prefix}_{side}_error"] = zero
                command.metrics[f"{metric_prefix}_{side}_proximity_gate"] = zero
                command.metrics[f"{metric_prefix}_{side}_surface_distance"] = zero
                command.metrics[f"{metric_prefix}_{side}_reward_raw"] = zero
        return zero

    if max_objects <= 0:
        return _zero_metrics()
    required_getters = (
        "get_object_wrist_relation_vector",
        "get_object_wrist_relation_mask",
        "get_object_wrist_surface_distance",
    )
    if not all(hasattr(command.motion_lib, name) for name in required_getters):
        return _zero_metrics()

    current_pos, current_quat, _, _, valid = utils.get_all_object_current_state(
        env, command, max_objects=max_objects
    )
    active = valid
    if object_role_names and max_objects > 1:
        active = utils.get_object_role_mask(
            env, command, object_role_names, max_objects, valid=active
        )
    if not active.any():
        return _zero_metrics()

    if robot_wrist_body_names is None:
        robot_wrist_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    current_steps = command.motion_start_time_steps + command.time_steps
    reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    error_sum = torch.zeros_like(reward_sum)
    gate_sum = torch.zeros_like(reward_sum)
    surface_distance_sum = torch.zeros_like(reward_sum)
    active_sum = torch.zeros_like(reward_sum)
    temperature = max(float(proximity_temperature), 1e-6)
    far = min(max(float(far_weight), 0.0), 1.0)

    from gear_sonic.isaac_utils import rotations as isaac_utils_rotations  # noqa: PLC0415

    side_specs = (
        ("left_hand", robot_wrist_body_names[0]),
        (
            "right_hand",
            robot_wrist_body_names[1]
            if len(robot_wrist_body_names) > 1
            else robot_wrist_body_names[0],
        ),
    )
    for hand, body_name in side_specs:
        relation_vector = command.motion_lib.get_object_wrist_relation_vector(
            command.motion_ids, current_steps, hand=hand
        )
        relation_mask = command.motion_lib.get_object_wrist_relation_mask(
            command.motion_ids, current_steps, hand=hand
        )
        surface_distance = command.motion_lib.get_object_wrist_surface_distance(
            command.motion_ids, current_steps, hand=hand
        )
        if relation_vector is None or relation_mask is None or surface_distance is None:
            continue
        if relation_vector.ndim == 2:
            relation_vector = relation_vector.unsqueeze(1)
        if relation_mask.ndim == 1:
            relation_mask = relation_mask.unsqueeze(1)
        if surface_distance.ndim == 1:
            surface_distance = surface_distance.unsqueeze(1)
        if relation_vector.ndim != 3 or relation_vector.shape[-1] != 3:
            raise ValueError(
                "Expected wrist relation vectors [batch, objects, 3], got "
                f"{tuple(relation_vector.shape)} for {hand}"
            )
        if relation_mask.ndim != 2 or surface_distance.ndim != 2:
            raise ValueError(
                "Expected wrist masks/distances [batch, objects], got "
                f"{tuple(relation_mask.shape)} and {tuple(surface_distance.shape)} for {hand}"
            )

        relation_vector = relation_vector[:, :max_objects]
        relation_mask = relation_mask[:, :max_objects] > 0.5
        surface_distance = surface_distance[:, :max_objects]
        body_indices = _get_body_indexes(command, [body_name])
        if not body_indices:
            continue
        wrist_pos_w = command.robot_body_pos_w[:, body_indices].mean(dim=1)
        wrist_object_vec_w = wrist_pos_w[:, None, :] - current_pos
        wrist_object_vec_l = isaac_utils_rotations.quat_rotate_inverse(
            current_quat.reshape(-1, 4),
            wrist_object_vec_w.reshape(-1, 3),
            w_last=False,
        ).view(env.num_envs, max_objects, 3)

        relation_error = torch.norm(wrist_object_vec_l - relation_vector, dim=-1)
        effective_error = torch.clamp(
            relation_error - max(float(relation_tolerance), 0.0), min=0.0
        )
        relation_reward = torch.exp(float(dist_exp_coeff) * effective_error)
        proximity_gate = torch.sigmoid(
            (float(proximity_distance) - surface_distance) / temperature
        )
        proximity_weight = far + (1.0 - far) * proximity_gate
        side_active = active & valid & relation_mask & torch.isfinite(surface_distance)
        side_active_f = side_active.to(dtype=relation_reward.dtype)
        side_reward_sum = (relation_reward * proximity_weight * side_active_f).sum(dim=-1)
        side_error_sum = (relation_error * side_active_f).sum(dim=-1)
        side_gate_sum = (proximity_gate * side_active_f).sum(dim=-1)
        side_surface_distance_sum = (surface_distance * side_active_f).sum(dim=-1)
        side_active_sum = side_active_f.sum(dim=-1)
        side_denom = side_active_sum.clamp_min(1.0)
        side = hand.split("_", maxsplit=1)[0]
        if hasattr(command, "metrics"):
            command.metrics[f"{metric_prefix}_{side}_active_count"] = side_active_sum
            command.metrics[f"{metric_prefix}_{side}_error"] = side_error_sum / side_denom
            command.metrics[f"{metric_prefix}_{side}_proximity_gate"] = (
                side_gate_sum / side_denom
            )
            command.metrics[f"{metric_prefix}_{side}_surface_distance"] = (
                side_surface_distance_sum / side_denom
            )
            command.metrics[f"{metric_prefix}_{side}_reward_raw"] = (
                side_reward_sum / side_denom
            )
        reward_sum += side_reward_sum
        error_sum += side_error_sum
        gate_sum += side_gate_sum
        surface_distance_sum += side_surface_distance_sum
        active_sum += side_active_sum

    if not active_sum.any():
        return _zero_metrics()
    denom = active_sum.clamp_min(1.0)
    reward = reward_sum / denom
    if hasattr(command, "metrics"):
        command.metrics[f"{metric_prefix}_active_count"] = active_sum
        command.metrics[f"{metric_prefix}_error"] = error_sum / denom
        command.metrics[f"{metric_prefix}_proximity_gate"] = gate_sum / denom
        command.metrics[f"{metric_prefix}_surface_distance"] = surface_distance_sum / denom
        command.metrics[f"{metric_prefix}_reward_raw"] = reward
    return reward


def rp2_support_stability_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    role_names: list[str] | str = "moving_support",
    pos_threshold: float = 0.5,
    ori_threshold: float = 1.5,
    height_threshold: float = 0.3,
    support_body_pos_threshold: float = 0.2,
    w_pos: float = 1.0,
    w_ori: float = 0.5,
    w_height: float = 0.5,
    w_support_body: float = 1.0,
    max_clip: float = 5.0,
    reduction: str = "max",
    support_body_names: list[str] | None = None,
) -> torch.Tensor:
    """RP2 support/chair stability penalty, off by default in config.

    The term measures valid objects with support roles, usually
    ``moving_support`` chairs, against the RP2 reference and optionally includes
    seated body/support-body tracking error. The function returns a positive
    penalty; configs should use a negative reward weight when enabling it.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
    if max_objects <= 1 or not hasattr(command, "object_root_pos"):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    current_pos, current_quat, _, _, valid = utils.get_all_object_current_state(
        env, command, max_objects=max_objects
    )
    role_mask = utils.get_object_role_mask(env, command, role_names, max_objects, valid=valid)
    if not role_mask.any():
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    target_pos = command.object_root_pos[:, :max_objects]
    target_quat = command.object_root_quat[:, :max_objects]
    pos_error = torch.norm(target_pos - current_pos, dim=-1)
    ori_error = quat_error_magnitude(
        target_quat.reshape(-1, 4), current_quat.reshape(-1, 4)
    ).view(env.num_envs, max_objects)
    height_error = torch.abs(target_pos[..., 2] - current_pos[..., 2])

    pos_metric = utils.reduce_masked_object_metric(pos_error, role_mask, reduction=reduction)
    ori_metric = utils.reduce_masked_object_metric(ori_error, role_mask, reduction=reduction)
    height_metric = utils.reduce_masked_object_metric(
        height_error, role_mask, reduction=reduction
    )

    def _scaled(metric: torch.Tensor, threshold: float) -> torch.Tensor:
        if threshold <= 0.0:
            return torch.clamp(metric, min=0.0, max=max_clip)
        return torch.clamp(metric / threshold, min=0.0, max=max_clip)

    penalty = (
        w_pos * _scaled(pos_metric, pos_threshold)
        + w_ori * _scaled(ori_metric, ori_threshold)
        + w_height * _scaled(height_metric, height_threshold)
    )

    if w_support_body != 0.0 and hasattr(command, "body_pos_relative_w"):
        if support_body_names is None:
            support_body_names = getattr(
                command.cfg,
                "rp2_support_body_metric_names",
                [
                    "pelvis",
                    "torso_link",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                ],
            )
        support_indices = _get_body_indexes(command, list(support_body_names))
        if support_indices:
            body_error = torch.norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1)
            support_body_error = body_error[:, support_indices].max(dim=-1).values
            penalty = penalty + w_support_body * _scaled(
                support_body_error, support_body_pos_threshold
            )

    return penalty


def object_lift_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_lift_height: float = 0.15,  # 15 cm max reward height
) -> torch.Tensor:
    """Compute reward for lifting the object above the table after first contact.

    Linear reward based on object height above the table surface, normalized by
    max_lift_height. Zero before the first contact frame from the motion data.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        max_lift_height: Maximum height (meters) for full reward. Heights above
            this are clamped.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    current_time = command.motion_start_time_steps + command.time_steps

    # Use precomputed per-env first contact frame
    per_env_first_contact = getattr(command, "_per_env_first_contact", None)
    if per_env_first_contact is not None:
        should_lift = current_time > per_env_first_contact
    else:
        # No contact data, always apply reward
        should_lift = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # Get object and table positions
    object_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)
    table = env.scene["table"]
    object_z = object_pos[:, 2]  # Object height
    table_z = table.data.root_pos_w[:, 2]  # Table height

    # Linear lift reward: how high object is above table, clipped to max_lift_height
    height_above_table = torch.clamp(object_z - table_z, min=0.0, max=max_lift_height)
    lift_height_reward = height_above_table / max_lift_height  # Normalize to [0, 1]

    # Only apply reward after first contact frame
    lift_reward = torch.where(
        should_lift, lift_height_reward.reshape(-1), torch.zeros_like(lift_height_reward)
    )
    return lift_reward


def reward_finger_primitive_limit(
    env: ManagerBasedRLEnv,
    mode: str = "discrete",
    action_scale: float = 1.0,
    discrete_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize finger primitive actions that exceed valid limits.

    Encourage the policy to output actions within the valid range naturally,
    rather than relying on clamping. This helps with gradient flow during training.

    Args:
        env: The environment.
        mode: "linear" or "discrete". Determines the valid action range.
        action_scale: For linear mode, valid range is [-1/action_scale, 1/action_scale].
        discrete_threshold: For discrete mode, valid range is [-threshold, threshold].

    Returns:
        Penalty tensor of shape (num_envs,). Positive values indicate out-of-limit.
    """
    # Get raw primitive actions stored by wrapper before processing
    primitive_actions = getattr(env, "_finger_primitive_actions_raw", None)

    if primitive_actions is None:
        # Finger primitives not enabled, return zero penalty
        return torch.zeros(env.num_envs, device=env.device)

    # Determine valid range based on mode
    if mode == "linear":
        lower_limit = -1.0 / action_scale
        upper_limit = 1.0 / action_scale
    elif mode == "discrete":
        lower_limit = -discrete_threshold
        upper_limit = discrete_threshold
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'linear' or 'discrete'.")

    below_limit = (-primitive_actions + lower_limit).clamp(min=0.0)  # positive when below lower
    above_limit = (primitive_actions - upper_limit).clamp(min=0.0)  # positive when above upper

    # Sum penalties across all primitive actions
    penalty = (below_limit + above_limit).sum(dim=-1)

    return penalty


def energy_consumption(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute energy consumption penalty based on mechanical power.

    Sum absolute joint torque-velocity products (P = |tau * omega|) across all
    joints. Encourages the policy to minimize energy usage while performing the task.

    Args:
        env: The environment.
        asset_cfg: The asset config for the robot (default: "robot").

    Returns:
        Penalty tensor of shape (num_envs,). Higher values = more energy used.
    """
    # Handle both dict (from YAML) and SceneEntityCfg object
    # Get the robot asset
    # Handle both dict (from YAML) and SceneEntityCfg object
    if isinstance(asset_cfg, dict):
        robot_name = asset_cfg.get("name", "robot")
    else:
        robot_name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[robot_name]

    # Get joint torques (applied torques) and joint velocities
    # joint_acc_pos_w: [num_envs, num_joints] - applied torques
    # joint_vel: [num_envs, num_joints] - joint velocities
    joint_torques = robot.data.applied_torque
    joint_velocities = robot.data.joint_vel

    # Compute instantaneous mechanical power: P = |tau * omega|
    # Taking absolute value since we care about energy magnitude regardless of direction
    power = torch.abs(joint_torques * joint_velocities)

    # Sum over all joints to get total energy consumption
    total_energy = power.sum(dim=-1)

    return total_energy


def meta_action_rate_l2(
    env: ManagerBasedRLEnv,
    token_only: bool = False,
) -> torch.Tensor:
    """Penalize the rate of change of meta actions (token + residual) via L2-squared.

    The meta action is the policy's raw output: latent residual (e.g., 64 dims)
    + finger primitives (e.g., 2 dims). Encourages smooth token trajectories in
    latent space, reducing jitter in the decoded joint-level actions.

    Since full_token = encoded_token + scaled_residual and encoded_token is constant
    for the same proprioception, penalizing the residual rate effectively penalizes
    the full_token rate.

    Args:
        env: The environment.
        token_only: If True, only penalize the latent/token portion (excluding
            finger primitives). If False, penalize the full meta action.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values = higher rate.
    """
    last_meta = getattr(env, "_last_meta_action", None)
    prev_meta = getattr(env, "_prev_meta_action", None)

    if last_meta is None or prev_meta is None:
        return torch.zeros(env.num_envs, device=env.device)

    if token_only:
        tokenizer_dim = getattr(env, "_tokenizer_action_dim", None)
        if tokenizer_dim is None:
            hand_action_dim = getattr(env, "_hand_action_dim", 2)
            tokenizer_dim = last_meta.shape[-1] - hand_action_dim
        diff = last_meta[:, :tokenizer_dim] - prev_meta[:, :tokenizer_dim]
    else:
        diff = last_meta - prev_meta

    return torch.sum(torch.square(diff), dim=-1)


def rp2_contact_phase_meta_action_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    hand: str = "either",
    pre_contact_frames: int = 15,
    post_contact_frames: int = 15,
    token_only: bool = False,
) -> torch.Tensor:
    """Penalize meta-action jumps only around the reference contact transition."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    first_contact_lookup = getattr(command, "_first_contact_lookup", None)
    if first_contact_lookup is None:
        return torch.zeros(env.num_envs, device=env.device)

    last_meta = getattr(env, "_last_meta_action", None)
    prev_meta = getattr(env, "_prev_meta_action", None)
    if last_meta is None or prev_meta is None:
        return torch.zeros(env.num_envs, device=env.device)

    if token_only:
        tokenizer_dim = getattr(env, "_tokenizer_action_dim", None)
        if tokenizer_dim is None:
            hand_action_dim = getattr(env, "_hand_action_dim", 2)
            tokenizer_dim = last_meta.shape[-1] - hand_action_dim
        diff = last_meta[:, :tokenizer_dim] - prev_meta[:, :tokenizer_dim]
    else:
        diff = last_meta - prev_meta
    penalty = torch.sum(torch.square(diff), dim=-1)

    current_frame = command.motion_start_time_steps + command.time_steps
    first_contact = first_contact_lookup[command.motion_ids]
    motion_num_steps = getattr(command, "motion_num_steps", None)
    if motion_num_steps is None:
        has_contact = torch.ones_like(first_contact, dtype=torch.bool)
    else:
        has_contact = first_contact < motion_num_steps

    distance_to_contact = current_frame - first_contact
    in_window = (
        (distance_to_contact >= -int(pre_contact_frames))
        & (distance_to_contact <= int(post_contact_frames))
        & has_contact
    )
    if hand != "either":
        label = _reference_contact_label(command, hand)
        if label is not None:
            approach_gate = _first_contact_gate(command, pre_contact_frames=pre_contact_frames)
            hand_active = label.float() > 0.5
            if approach_gate is not None:
                hand_active = hand_active | (approach_gate > 0.0)
            in_window = in_window & hand_active
    return penalty * in_window.float()


def full_latent_rate_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize the rate of change of the full decoder-input latent via L2-squared.

    The full latent is the post-quantization token sent to the decoder:
        full_latent = FSQ_quantize(encoder_output + scaled_residual)

    Unlike meta_action_rate_l2 which penalizes the policy's residual output, this
    penalizes the actual decoder input including the encoder's contribution.

    NOTE: Buffers (_full_latent, _prev_full_latent) are maintained by
    ManagerEnvWrapper.

    Args:
        env: The environment.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values = higher rate.
    """
    full_latent = getattr(env, "_full_latent", None)
    prev_full_latent = getattr(env, "_prev_full_latent", None)

    if full_latent is None or prev_full_latent is None:
        return torch.zeros(env.num_envs, device=env.device)

    diff = full_latent - prev_full_latent
    return torch.sum(torch.square(diff), dim=-1)


# ==================== Approach Behavior Rewards ====================


def reward_approach_velocity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    wrist_link: str = "right_hand_palm_link",
    sigma: float = 0.25,
    contact_threshold: float = 0.1,
    min_contacts: int = 3,
    disable_after_contact: bool = True,
    command_name: str = "motion",
) -> torch.Tensor:
    """Penalize high wrist velocity near the object to prevent swatting.

    Apply a Gaussian distance-gated velocity penalty: strongest at the object and
    decaying with distance. By default, disabled after actual grasp (physics contact)
    to allow free movement during lifting/manipulation.

    Formula: penalty = velocity^2 * exp(-distance^2 / sigma^2) * contact_gate

    NOTE: Disabling penalty after grasp may incentivize rushing to touch the object
    to escape the penalty. If observed, increase min_contacts or set
    disable_after_contact=False.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config for finger-object contact detection.
        wrist_link: Name of the wrist/palm link to track velocity of.
        sigma: Characteristic distance (meters). Penalty is ~37% at this distance.
        contact_threshold: Minimum contact force (N) to count as touching.
        min_contacts: Minimum finger contacts to count as "grasping".
        disable_after_contact: If True, penalty turns off after grasp. If False,
            penalty stays active during grasp.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get robot and object from scene
    robot = env.scene["robot"]

    # Get wrist body index from robot.body_names (ALL robot bodies)
    body_names = robot.body_names
    if wrist_link not in body_names:
        return torch.zeros(env.num_envs, device=env.device)
    wrist_index = body_names.index(wrist_link)

    # Get wrist position and velocity in world frame
    wrist_pos = robot.data.body_pos_w[:, wrist_index, :]  # [num_envs, 3]
    wrist_vel = robot.data.body_lin_vel_w[:, wrist_index, :]  # [num_envs, 3]

    # Get object position
    object_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)

    # Compute squared distance and squared velocity (L2)
    diff = wrist_pos - object_pos
    distance_sq = torch.sum(torch.square(diff), dim=-1)  # [num_envs]
    velocity_sq = torch.sum(torch.square(wrist_vel), dim=-1)  # [num_envs]

    # Gaussian distance factor: strongest at object, decays with distance
    sigma_sq = sigma * sigma
    distance_factor = torch.exp(-distance_sq / sigma_sq)

    # Contact gating using ACTUAL physics contact (same as grasp_reward)
    if disable_after_contact:
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        force_magnitude = _finger_contact_force_magnitude(sensor)

        # Count fingers in contact (force above threshold)
        fingers_in_contact = (force_magnitude > contact_threshold).float()
        num_fingers_in_contact = fingers_in_contact.sum(dim=-1)

        # Grasping = at least min_contacts fingers touching
        is_grasping = (num_fingers_in_contact >= min_contacts).float()
        contact_gate = 1.0 - is_grasping
    else:
        # No contact gating - penalty stays active even during/after grasp
        contact_gate = 1.0

    # Final penalty: velocity^2 * distance_factor * contact_gate
    penalty = velocity_sq * distance_factor * contact_gate

    return penalty


def reward_open_hand_until_close(
    env: ManagerBasedRLEnv,
    wrist_link: str = "right_hand_palm_link",
    sigma: float = 0.25,
    hand: str = "right",
    command_name: str = "motion",
) -> torch.Tensor:
    """Penalize closing the hand when far from the object.

    Encourage the robot to keep its hand open during approach and only close when
    close to the object (within grasp range). Uses an inverse Gaussian: penalty is
    zero at the object and increases with distance.

    Formula: penalty = hand_closed * (1 - exp(-distance^2 / sigma^2))

    Args:
        env: The environment.
        wrist_link: Name of the wrist/palm link to measure distance from.
        sigma: Characteristic distance (meters). Penalty reaches ~63% at this distance.
        hand: Which hand to check ("left" or "right").

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    # Get hand action state (0 = left, 1 = right due to sorted keys)
    hand_actions = getattr(env, "_finger_primitive_actions_raw", None)
    if hand_actions is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Hand index: sorted(["left", "right"]) = ["left", "right"]
    # Index 0 = left, Index 1 = right
    hand_idx = 1 if hand == "right" else 0

    # Binary closed detection: action >= 0 means closed (discrete mode threshold)
    hand_closed = (hand_actions[:, hand_idx] >= 0).float()  # [num_envs]

    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get robot and object from scene
    robot = env.scene["robot"]

    # Get wrist body index from robot.body_names (ALL robot bodies)
    body_names = robot.body_names
    if wrist_link not in body_names:
        return torch.zeros(env.num_envs, device=env.device)
    wrist_index = body_names.index(wrist_link)

    wrist_pos = robot.data.body_pos_w[:, wrist_index, :]  # [num_envs, 3]

    # Get object position
    object_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)

    # Compute squared distance
    diff = wrist_pos - object_pos
    distance_sq = torch.sum(torch.square(diff), dim=-1)  # [num_envs]

    # Inverse Gaussian: penalty when FAR, not when close
    # At distance=0: factor=0 (no penalty for closing)
    # At distance=sigma: factor=0.63
    # At distance=2*sigma: factor=0.98
    sigma_sq = sigma * sigma
    far_factor = 1.0 - torch.exp(-distance_sq / sigma_sq)

    # Final penalty: closed * far_factor
    penalty = hand_closed * far_factor

    return penalty


def reward_open_hand_until_close_contact_gated(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    wrist_link: str = "right_hand_palm_link",
    sigma: float = 0.25,
    hand: str = "right",
    contact_threshold: float = 0.1,
    min_contacts: int = 3,
    command_name: str = "motion",
) -> torch.Tensor:
    """Penalize closing the hand when far from the object, gated by physics contact.

    Same as reward_open_hand_until_close but also disables the penalty once the
    robot has established a grasp (detected via contact sensor), allowing free
    hand closure during manipulation.

    Formula: penalty = hand_closed * far_factor * not_grasping

    Args:
        env: The environment.
        sensor_cfg: Contact sensor configuration for detecting grasp.
        wrist_link: Name of the wrist/palm link to measure distance from.
        sigma: Characteristic distance (meters). Penalty reaches ~63% at this distance.
        hand: Which hand to check ("left" or "right").
        contact_threshold: Force threshold (N) to consider a finger in contact.
        min_contacts: Minimum number of fingers in contact to consider grasping.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    # Get hand action state (0 = left, 1 = right due to sorted keys)
    hand_actions = getattr(env, "_finger_primitive_actions_raw", None)
    if hand_actions is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Hand index: sorted(["left", "right"]) = ["left", "right"]
    # Index 0 = left, Index 1 = right
    hand_idx = 1 if hand == "right" else 0

    # Binary closed detection: action >= 0 means closed (discrete mode threshold)
    hand_closed = (hand_actions[:, hand_idx] >= 0).float()  # [num_envs]

    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get robot and object from scene
    robot = env.scene["robot"]

    # Get wrist body index from robot.body_names (ALL robot bodies)
    body_names = robot.body_names
    if wrist_link not in body_names:
        return torch.zeros(env.num_envs, device=env.device)
    wrist_index = body_names.index(wrist_link)

    wrist_pos = robot.data.body_pos_w[:, wrist_index, :]  # [num_envs, 3]

    # Get object position
    object_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)

    # Compute squared distance
    diff = wrist_pos - object_pos
    distance_sq = torch.sum(torch.square(diff), dim=-1)  # [num_envs]

    # Inverse Gaussian: penalty when FAR, not when close
    sigma_sq = sigma * sigma
    far_factor = 1.0 - torch.exp(-distance_sq / sigma_sq)

    # Physics-based grasp detection using contact sensor
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force_magnitude = _finger_contact_force_magnitude(sensor)

    # Count fingers in contact (force above threshold)
    fingers_in_contact = (force_magnitude > contact_threshold).float()
    num_fingers_in_contact = fingers_in_contact.sum(dim=-1)

    # Grasping = at least min_contacts fingers touching
    is_grasping = (num_fingers_in_contact >= min_contacts).float()
    not_grasping = 1.0 - is_grasping

    # Final penalty: closed * far_factor * not_grasping
    penalty = hand_closed * far_factor * not_grasping

    return penalty


def reward_open_hand_unless_reference_contact(
    env: ManagerBasedRLEnv,
    hand: str = "right_hand",
    command_name: str = "motion",
) -> torch.Tensor:
    """Penalize a closed hand primitive while the reference has no contact for that hand.

    G1 two-primitive transfer of HiPHI's open-hand-unless-reference-contact term:
    the per-hand source object-contact label gates the penalty, so the hand may
    close only on reference-contact frames. Closed uses the discrete primitive
    threshold (raw action >= 0), matching the finger-primitive conversion.

    Returns:
        Penalty tensor of shape (num_envs,) in {0, 1}. Use a negative weight.
    """
    if hand not in ("left_hand", "right_hand"):
        raise ValueError(f"hand must be 'left_hand' or 'right_hand', got {hand!r}")
    hand_actions = getattr(env, "_finger_primitive_actions_raw", None)
    if hand_actions is None:
        raise RuntimeError("open-hand reward requires finger primitive actions")
    if hand_actions.shape[-1] != 2:
        raise ValueError(f"Expected [left, right] hand primitives, got {tuple(hand_actions.shape)}")
    hand_closed = (hand_actions[:, 1 if hand == "right_hand" else 0] >= 0).float()

    command: TrackingCommand = env.command_manager.get_term(command_name)
    reference_contact = _reference_contact_label(command, hand)
    if reference_contact is None:
        raise RuntimeError(f"open-hand reward found no {hand} reference contact label")
    penalty = hand_closed * (1.0 - reference_contact.float())

    side = "left" if hand == "left_hand" else "right"
    command.metrics[f"open_hand_unless_reference_contact/closed/{side}"] = hand_closed
    command.metrics[f"open_hand_unless_reference_contact/reference_contact/{side}"] = (
        reference_contact.float()
    )
    command.metrics[f"open_hand_unless_reference_contact/penalty/{side}"] = penalty
    return penalty


def reward_foot_slippage_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot velocity while the foot is in ground contact.

    Discourages the robot from shuffling or skating its feet along the ground.
    When a foot has ground contact force above the threshold, any velocity of
    that foot contributes to the penalty.

    Formula: penalty = sum_feet( ||foot_vel|| * (||contact_force|| > threshold) )

    Args:
        env: The environment.
        sensor_cfg: Contact sensor configuration with body_names for feet.
        contact_threshold: Force magnitude (N) above which the foot is considered
            in ground contact. Default 1.0N matches existing foot contact checks.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    robot = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Find foot body indices in both the articulation and the contact sensor
    feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_idx_robot = [robot.body_names.index(n) for n in feet_names]
    feet_idx_sensor = contact_sensor.find_bodies(".*ankle_roll.*")[0]

    # Foot velocities — full 3D
    foot_vel = robot.data.body_lin_vel_w[:, feet_idx_robot, :]  # [num_envs, 2, 3]
    foot_speed = torch.norm(foot_vel, dim=-1)  # [num_envs, 2]

    # Contact mask: is foot touching ground?
    contact_force = contact_sensor.data.net_forces_w[:, feet_idx_sensor, :]  # [num_envs, 2, 3]
    in_contact = (torch.norm(contact_force, dim=-1) > contact_threshold).float()  # [num_envs, 2]

    # Penalty = speed * in_contact, summed over both feet
    return (foot_speed * in_contact).sum(dim=-1)
