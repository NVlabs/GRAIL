"""Termination conditions for motion-tracking and HOI reinforcement learning environments."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
)

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand, _get_body_indexes
from gear_sonic.envs.manager_env.mdp import utils
from gear_sonic.trl.utils.torch_transform import get_heading_q


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = None
    motion_time_out = None
    anchor_pos = None
    anchor_pos_adaptive = None
    anchor_pos_orig = None
    ee_body_pos_adaptive = None
    ee_body_pos_orig = None
    cumm_body_errors = None
    anchor_ori = None
    anchor_ori_full = None
    ee_body_pos = None
    anchor_pos_xy = None
    foot_pos_xyz = None
    cumm_body_pos_error = None
    cumm_body_ori_error = None
    cumm_body_pos_error_local = None
    cumm_body_ori_error_local = None
    # HOI terminations
    object_pos_deviation = None
    object_z_pos_deviation = None
    object_ori_deviation = None  # v344: object orientation deviation termination (radians)
    rp2_support_instability = None
    object_not_lifted = None
    robot_table_contact_before_object = None
    hand_table_contact_termination = None
    grasp_failure_after_contact = None
    grasp_failure_after_contact_left = None
    canonical_required_hand_contact_failure = None
    active_articulation_joint_deviation = None
    physical_fall = None


def physical_root_fall(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_root_height: float = 0.45,
    max_tilt_rad: float = 1.0,
) -> torch.Tensor:
    """Reference-free fall predicate for generative-policy rollouts.

    Tracking error is not a valid termination once a policy is allowed to
    depart from its source motion.  This predicate instead checks only the
    simulated robot: a low root or a root whose local gravity vector tilts by
    more than ``max_tilt_rad`` from upright is considered a fall.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    gravity_w = asset.data.GRAVITY_VEC_W
    quat_apply_fn = (
        math_utils.quat_apply_inverse
        if hasattr(math_utils, "quat_apply_inverse")
        else math_utils.quat_rotate_inverse
    )
    gravity_b = quat_apply_fn(root_quat, gravity_w)
    # Upright => gravity_b.z == -1.  This one-sided comparison also rejects an
    # upside-down base while allowing arbitrary yaw.
    tilted = gravity_b[:, 2] > -torch.cos(
        torch.as_tensor(max_tilt_rad, device=root_pos.device, dtype=root_pos.dtype)
    )
    return root_pos[:, 2].lt(min_root_height) | tilted


def fixed_horizon_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference-free timeout based only on the configured episode horizon."""
    # This predicate is evaluated after IsaacLab advances episode_length_buf.
    # Using the tracking helper's ``- 1`` convention produced 99 transitions
    # for a configured 100-step generative episode in the simulator smoke.
    return env.episode_length_buf >= env.max_episode_length


def exceeded_anchor_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate if anchor (root) position error exceeds a distance threshold.

    Compute the L2 norm between the reference and robot anchor positions in world
    frame and flag environments where the error exceeds ``threshold``.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed position error in meters.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_diff = command.anchor_pos_w - command.robot_anchor_pos_w
    return pos_diff.norm(dim=1).gt(threshold)


def exceeded_anchor_pos_xy(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate if anchor XY (horizontal) position error exceeds a distance threshold.

    Only the X and Y components of the world-frame anchor position are compared,
    ignoring vertical drift.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed horizontal position error in meters.
        body_names: Unused; kept for config compatibility.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    xy_diff = command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
    return xy_diff.norm(dim=1).gt(threshold)


def exceeded_anchor_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    root_height_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if anchor Z-height error exceeds a threshold.

    When ``threshold_adaptive`` is True, use a looser ``down_threshold`` for
    environments whose reference root height is below ``root_height_threshold``
    (e.g. crouching or sitting motions).

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Default maximum allowed height error in meters.
        threshold_adaptive: Enable per-env adaptive thresholding.
        down_threshold: Looser threshold applied when the reference root is low.
        root_height_threshold: Height below which ``down_threshold`` is used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    height_diff = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
    if threshold_adaptive:
        thresh = torch.full_like(height_diff, threshold)
        thresh[command.running_ref_root_height < root_height_threshold] = down_threshold
        return height_diff.gt(thresh)
    return height_diff.gt(threshold)


def exceeded_anchor_tilt(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate if the anchor tilt deviates from the reference gravity projection.

    Compare the Z-component of the gravity vector rotated into the reference and
    robot anchor frames. A large difference indicates excessive torso tilt.

    Args:
        env: The manager-based RL environment.
        asset_cfg: Scene entity config used to obtain the gravity vector.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed absolute difference in projected gravity Z.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: TrackingCommand = env.command_manager.get_term(command_name)
    quat_apply_fn = (
        math_utils.quat_apply_inverse
        if hasattr(math_utils, "quat_apply_inverse")
        else math_utils.quat_rotate_inverse
    )
    ref_grav = quat_apply_fn(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_grav = quat_apply_fn(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (ref_grav[:, 2] - robot_grav[:, 2]).abs().gt(threshold)


def exceeded_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate if the squared anchor orientation error exceeds a threshold.

    Compute the full quaternion error magnitude between reference and robot
    anchor orientations, then compare the squared value against ``threshold``.

    Args:
        env: The manager-based RL environment.
        asset_cfg: Scene entity config (unused but required by termination API).
        command_name: Name of the tracking command term.
        threshold: Maximum allowed squared orientation error (radians^2).

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return angular_err.square().gt(threshold)


def exceeded_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate if any tracked body position error exceeds a distance threshold.

    Compare per-body world-frame positions between the reference motion and the
    robot, and terminate if *any* body exceeds ``threshold``.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed position error per body in meters.
        body_names: Optional list of body names to check. If ``None``, all
            tracked bodies from the command are used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    return pos_diff.norm(dim=-1).gt(threshold).any(dim=-1)


def exceeded_body_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    body_names: list[str] | None = None,
    root_height_threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate if any tracked body Z-height error exceeds a threshold.

    When ``threshold_adaptive`` is True, use a looser ``down_threshold`` for
    environments whose reference root height is below ``root_height_threshold``.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Default maximum allowed height error per body in meters.
        threshold_adaptive: Enable per-env adaptive thresholding.
        down_threshold: Looser threshold applied when the reference root is low.
        body_names: Optional list of body names to check. If ``None``, all
            tracked bodies from the command are used.
        root_height_threshold: Height below which ``down_threshold`` is used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    height_err = (
        command.body_pos_relative_w[:, tracked, 2] - command.robot_body_pos_w[:, tracked, 2]
    ).abs()
    if threshold_adaptive:
        thresh = torch.full_like(height_err, threshold)
        thresh[command.running_ref_root_height < root_height_threshold] = down_threshold
    else:
        thresh = torch.full_like(height_err, threshold)

    triggered_by_body = height_err.gt(thresh)

    # IsaacLab auto-resets a terminated environment before the evaluation
    # callback can inspect it, and CommandTerm.reset clears command metrics.
    # Preserve the pre-reset values on a diagnostic-only attribute so the
    # callback can identify the exact ankle or wrist without disabling or
    # relaxing the termination.
    command._rp2_last_ee_body_termination_debug = {  # noqa: SLF001
        "body_names": [command.cfg.body_names[index] for index in tracked],
        "height_error_m": height_err.detach().clone(),
        "height_threshold_m": thresh.detach().clone(),
        "reference_z_m": command.body_pos_relative_w[:, tracked, 2].detach().clone(),
        "simulated_z_m": command.robot_body_pos_w[:, tracked, 2].detach().clone(),
        "triggered": triggered_by_body.detach().clone(),
    }
    return triggered_by_body.any(dim=-1)


def tracking_time_out(
    env: ManagerBasedRLEnv,
    command_name: str,
    cap_by_episode_length: bool = True,
) -> torch.Tensor:
    """Terminate when the motion ends or an optional episode-length cap is reached.

    Compare the elapsed simulation steps (including the motion start offset)
    against the total length of each environment's assigned motion clip.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        cap_by_episode_length: Also truncate at ``env.max_episode_length``.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    elapsed = command.time_steps + command.motion_start_time_steps + 1
    total = command.motion_lib.get_time_step_total(command.motion_ids)
    motion_finished = elapsed >= total
    if not cap_by_episode_length:
        return motion_finished
    episode_finished = env.episode_length_buf >= env.max_episode_length - 1
    return motion_finished | episode_finished


def _resolve_matching_names(patterns: str | Sequence[str], candidates: Sequence[str]) -> list[str]:
    """Resolve body name patterns (exact or regex) against available names."""
    if isinstance(patterns, str | bytes):
        patterns = [patterns]
    resolved: list[str] = []
    for pattern in patterns:
        if pattern == ".*":
            resolved.extend(candidates)
            continue
        regex = re.compile(pattern)
        resolved.extend([name for name in candidates if regex.fullmatch(name)])
    # preserve order and drop duplicates
    seen = set()
    ordered_unique = []
    for name in resolved:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)
    return ordered_unique


class _CummErrorMixin(ManagerTermBase):
    """Shared logic for cumulative error-based terminations."""

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize cumulative error tracking buffers.

        Args:
            cfg: Termination term config. Expected ``params`` keys:
                ``min_steps``, ``threshold``, ``command_name``.
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        self.min_steps: int = cfg.params.get("min_steps")
        self.threshold: float = cfg.params.get("threshold")
        self.command_name = cfg.params.get("command_name")
        self.command: TrackingCommand = env.command_manager.get_term(self.command_name)

        device = self.command.device
        self.error = torch.zeros(env.num_envs, device=device)
        self._cum_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=device)

    def _update_counters(self) -> torch.Tensor:
        """Accumulate consecutive steps above the threshold and return done mask."""
        exceeded = self.error >= self.threshold
        self._cum_steps[exceeded] += 1
        self._cum_steps[~exceeded] = 0
        return self._cum_steps >= self.min_steps

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset cumulative step counters for the given environments.

        Args:
            env_ids: Environment indices to reset, or ``None`` for all.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._cum_steps[env_ids] = 0


class CummBodyPosError(_CummErrorMixin):
    """Terminate if body position error in world frame exceeds threshold for min_steps."""

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize body index mapping for world-frame position error tracking.

        Args:
            cfg: Termination term config. Additional ``params`` key:
                ``body_names`` -- regex or list of body names to track
                (default ``".*"`` for all).
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        motion_names = self.command.cfg.body_names
        body_names = cfg.params.get("body_names", ".*")
        selected = _resolve_matching_names(body_names, motion_names)
        self.motion_body_indices = [motion_names.index(name) for name in selected]
        # command.body_indexes maps motion body order to robot body indices
        self.robot_body_indices = self.motion_body_indices

    def __call__(self, env, body_names=None, min_steps=None, threshold=None, command_name=None):
        """Compute max world-frame body position error and check cumulative threshold.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        ref_body_pos = self.command.body_pos_w[:, self.motion_body_indices]
        robot_body_pos = self.command.robot_body_pos_w[:, self.robot_body_indices]
        body_pos_error = (ref_body_pos - robot_body_pos).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        return self._update_counters()


class CummBodyOriError(_CummErrorMixin):
    """Terminate if body orientation error in world frame exceeds threshold for min_steps."""

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize body index mapping for world-frame orientation error tracking.

        Args:
            cfg: Termination term config. Additional ``params`` key:
                ``body_names`` -- regex or list of body names to track
                (default ``".*"`` for all).
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        motion_names = self.command.cfg.body_names
        body_names = cfg.params.get("body_names", ".*")
        selected = _resolve_matching_names(body_names, motion_names)
        self.motion_body_indices = [motion_names.index(name) for name in selected]
        # self.robot_body_indices = self.command.body_indexes[self.motion_body_indices].tolist()
        self.robot_body_indices = self.motion_body_indices

    def __call__(self, env, body_names=None, min_steps=None, threshold=None, command_name=None):
        """Compute max world-frame body orientation error and check cumulative threshold.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        ref_body_quat = self.command.body_quat_w[:, self.motion_body_indices]
        robot_body_quat = self.command.robot_body_quat_w[:, self.robot_body_indices]
        quat_diff = quat_mul(quat_conjugate(ref_body_quat), robot_body_quat)
        body_ori_error = axis_angle_from_quat(quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        return self._update_counters()


class CummBodyPosErrorLocal(_CummErrorMixin):
    """Terminate if body position error in the root-yaw frame exceeds threshold for min_steps."""

    def __init__(self, cfg, env):
        """Initialize body index mapping for root-local position error tracking.

        Args:
            cfg: Termination term config. Additional ``params`` key:
                ``body_names`` -- regex or list of body names to track
                (default ``".*"`` for all).
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        body_names = cfg.params.get("body_names", ".*")
        motion_names = self.command.cfg.body_names
        selected = _resolve_matching_names(body_names, motion_names)
        self.motion_body_indices = [motion_names.index(name) for name in selected]
        # self.robot_body_indices = self.command.body_indexes[self.motion_body_indices].tolist()
        self.robot_body_indices = self.motion_body_indices

    def __call__(self, env, body_names=None, min_steps=None, threshold=None, command_name=None):
        """Compute max root-yaw-local body position error and check cumulative threshold.

        Transform body positions into the heading-aligned root frame before
        computing errors, making the check invariant to global position and yaw.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        ref_body_pos = self.command.body_pos_w[:, self.motion_body_indices]
        robot_body_pos = self.command.robot_body_pos_w[:, self.robot_body_indices]

        ref_root_pos = self.command.anchor_pos_w.view(self.num_envs, 1, 3).clone()
        robot_root_pos = self.command.robot_anchor_pos_w.view(self.num_envs, 1, 3).clone()
        ref_root_pos[..., 2] = 0.0
        robot_root_pos[..., 2] = 0.0

        ref_root_quat = get_heading_q(self.command.anchor_quat_w.view(self.num_envs, 1, 4))
        robot_root_quat = get_heading_q(self.command.robot_anchor_quat_w.view(self.num_envs, 1, 4))
        # expand root quaternions to match per-body vectors for quat_apply_inverse
        ref_root_quat = ref_root_quat.expand(ref_body_pos.shape[0], ref_body_pos.shape[1], -1)
        robot_root_quat = robot_root_quat.expand(
            robot_body_pos.shape[0], robot_body_pos.shape[1], -1
        )

        ref_body_local = quat_apply_inverse(ref_root_quat, ref_body_pos - ref_root_pos)
        robot_body_local = quat_apply_inverse(robot_root_quat, robot_body_pos - robot_root_pos)

        body_pos_error = (ref_body_local - robot_body_local).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        return self._update_counters()


class CummBodyOriErrorLocal(_CummErrorMixin):
    """Terminate if body orientation error in the root-yaw frame exceeds threshold for min_steps."""

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize body index mapping for root-local orientation error tracking.

        Args:
            cfg: Termination term config. Additional ``params`` key:
                ``body_names`` -- regex or list of body names to track
                (default ``".*"`` for all).
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        motion_names = self.command.cfg.body_names
        body_names = cfg.params.get("body_names", ".*")
        selected = _resolve_matching_names(body_names, motion_names)
        self.motion_body_indices = [motion_names.index(name) for name in selected]
        # self.robot_body_indices = self.command.body_indexes[self.motion_body_indices].tolist()
        self.robot_body_indices = self.motion_body_indices

    def __call__(self, env, body_names=None, min_steps=None, threshold=None, command_name=None):
        """Compute max root-yaw-local body orientation error and check cumulative threshold.

        Transform body quaternions into the heading-aligned root frame before
        computing axis-angle errors, making the check invariant to global yaw.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        ref_body_quat = self.command.body_quat_w[:, self.motion_body_indices]
        robot_body_quat = self.command.robot_body_quat_w[:, self.robot_body_indices]

        ref_root_quat = get_heading_q(self.command.anchor_quat_w.view(self.num_envs, 1, 4))
        robot_root_quat = get_heading_q(self.command.robot_anchor_quat_w.view(self.num_envs, 1, 4))
        ref_root_quat = ref_root_quat.expand_as(ref_body_quat)
        robot_root_quat = robot_root_quat.expand_as(robot_body_quat)

        ref_body_quat_local = quat_mul(quat_conjugate(ref_root_quat), ref_body_quat)
        robot_body_quat_local = quat_mul(quat_conjugate(robot_root_quat), robot_body_quat)

        quat_diff = quat_mul(quat_conjugate(ref_body_quat_local), robot_body_quat_local)
        body_ori_error = axis_angle_from_quat(quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        return self._update_counters()




# ==================== HOI manipulation terminations ====================


def _target_object_role_eligible(
    env: ManagerBasedRLEnv,
    command: TrackingCommand,
    allowed_roles: Sequence[str] | str | None,
) -> torch.Tensor:
    """Return whether each selected target belongs to an allowed role."""
    if not allowed_roles:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return utils.get_target_object_role_mask(env, command, allowed_roles)


def object_pos_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.2,
    start_after_first_contact: bool = False,
    grace_frames: int = 0,
    track_active_entity_base: bool = False,
    allowed_roles: Sequence[str] | str | None = None,
    position_axes: str = "xyz",
) -> torch.Tensor:
    """Terminate if object position deviates from reference by more than threshold.

    This implements condition (i) from Interaction Early Termination (IET):
    Object points deviate from their references by more than threshold on average.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        threshold: Maximum allowed deviation in meters.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    if track_active_entity_base:
        ref_object_pos = command.active_entity_root_pos
        current_object_pos, _, _, _, valid = (
            utils.get_active_entity_base_current_state(env, command)
        )
    else:
        ref_object_pos = command.target_object_root_pos
        current_object_pos, _, _, _, valid = utils.get_target_object_current_state(
            env, command
        )

    delta = ref_object_pos - current_object_pos
    if position_axes == "xyz":
        pos_error = torch.norm(delta, dim=-1)
    elif position_axes == "xy":
        pos_error = torch.norm(delta[..., :2], dim=-1)
    else:
        raise ValueError(
            "object_pos_deviation position_axes must be 'xyz' or 'xy', "
            f"got {position_axes!r}"
        )
    return valid & _target_object_role_eligible(
        env, command, allowed_roles
    ) & _object_deviation_time_gate(
        env, command, start_after_first_contact, grace_frames
    ) & (pos_error > threshold)


def object_ori_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.8,
    start_after_first_contact: bool = False,
    grace_frames: int = 0,
    track_active_entity_base: bool = False,
    allowed_roles: Sequence[str] | str | None = None,
    exclude_alias_regex: str | None = None,
) -> torch.Tensor:
    """v344 (2026-09-05): terminate if the object ORIENTATION deviates from its reference by more than
    ``threshold`` radians (quat_error_magnitude). Companion of object_pos_deviation with the same valid /
    role / time gates. Motivation: overfit runs reached 0.7 survival with 0.5-1.0 rad object orientation
    error -- nothing constrained orientation (the object reward kernel exp(-20*err) is ~0 there)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    if track_active_entity_base:
        ref_quat = command.active_entity_root_quat
        _, current_quat, _, _, valid = utils.get_active_entity_base_current_state(env, command)
    else:
        ref_quat = command.target_object_root_quat
        _, current_quat, _, _, valid = utils.get_target_object_current_state(env, command)
    ori_error = quat_error_magnitude(
        ref_quat.reshape(-1, 4), current_quat.reshape(-1, 4)
    ).reshape(env.num_envs, -1)
    if ori_error.shape[1] > 1:
        ori_error = ori_error[:, 0]
    else:
        ori_error = ori_error.reshape(env.num_envs)
    result = valid & _target_object_role_eligible(
        env, command, allowed_roles
    ) & _object_deviation_time_gate(
        env, command, start_after_first_contact, grace_frames
    ) & (ori_error > float(threshold))
    if exclude_alias_regex:
        # v347: rotationally symmetric objects (balls, bottles, buckets, mops) have no meaningful orientation
        # target -> exempt envs whose current motion alias matches. Mask over motion ids cached per regex.
        cache = getattr(env, "_object_ori_exclude_cache", None)
        if cache is None:
            cache = {}
            setattr(env, "_object_ori_exclude_cache", cache)
        keys = command.motion_lib.curr_motion_keys
        entry = cache.get(exclude_alias_regex)
        if entry is None or entry[0] != len(keys):
            pattern = re.compile(exclude_alias_regex)
            mask = torch.tensor(
                [bool(pattern.search(str(k))) for k in keys], dtype=torch.bool, device=env.device
            )
            cache[exclude_alias_regex] = (len(keys), mask)
            print(  # noqa: T201
                f"[object_ori_deviation] exclude_alias_regex={exclude_alias_regex!r}: "
                f"{int(mask.sum())}/{len(keys)} motions exempt from orientation termination"
            )
            entry = cache[exclude_alias_regex]
        excluded = entry[1][command.motion_ids.long()]
        result = result & ~excluded
    return result


def object_z_pos_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.2,
    start_after_first_contact: bool = False,
    grace_frames: int = 0,
    track_active_entity_base: bool = False,
    allowed_roles: Sequence[str] | str | None = None,
) -> torch.Tensor:
    """Terminate if object Z position deviates from reference by more than threshold.

    Z-only variant of :func:`object_pos_deviation` — implements condition (i)
    from Interaction Early Termination (IET) projected onto the vertical axis.
    Used by termination bundles that tolerate XY drift but not dropping /
    lifting the object above threshold.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        threshold: Maximum allowed Z deviation in meters (default 0.2).

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    if track_active_entity_base:
        ref_object_z_pos = command.active_entity_root_pos[:, 2]
        current_object_pos, _, _, _, valid = (
            utils.get_active_entity_base_current_state(env, command)
        )
    else:
        ref_object_z_pos = command.target_object_root_pos[:, 2]
        current_object_pos, _, _, _, valid = utils.get_target_object_current_state(
            env, command
        )
    current_object_z_pos = current_object_pos[:, 2]

    z_pos_error = torch.abs(ref_object_z_pos - current_object_z_pos)
    return valid & _target_object_role_eligible(
        env, command, allowed_roles
    ) & _object_deviation_time_gate(
        env, command, start_after_first_contact, grace_frames
    ) & (z_pos_error > threshold)


def _object_deviation_time_gate(
    env: ManagerBasedRLEnv,
    command: TrackingCommand,
    start_after_first_contact: bool,
    grace_frames: int = 0,
) -> torch.Tensor:
    """Optional curriculum gate for object-deviation terminations.

    Default termination behavior is unchanged. When enabled, object deviation is
    ignored until the reference first-contact frame plus ``grace_frames`` so a
    contact-window curriculum can be tested without changing reference data.
    """
    if not start_after_first_contact:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    first_contact = getattr(command, "_per_env_first_contact", None)
    if first_contact is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    current_time = command.motion_start_time_steps + command.time_steps
    return current_time >= (first_contact + int(grace_frames or 0))


def active_articulation_joint_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold_normalized: float = 0.75,
    start_after_first_contact: bool = False,
    grace_frames: int = 0,
) -> torch.Tensor:
    """Terminate when any active articulation joint falls too far behind reference.

    Position errors are divided by each joint's half-range, so one threshold can
    cover revolute and prismatic mechanisms. This term is only enabled by the
    opt-in articulated mirror config.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos, _, reference_pos, _, joint_valid = (
        utils.get_active_articulation_joint_tracking_state(env, command)
    )
    if joint_pos.shape[1] == 0:
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if hasattr(command, "metrics"):
            command.metrics["active_articulation/joint_termination_armed"] = (
                terminated.float()
            )
        return terminated
    articulation = utils.get_active_articulation(env)
    limits = articulation.data.soft_joint_pos_limits
    half_range = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1e-3)
    normalized_error = torch.abs(reference_pos - joint_pos) / half_range
    masked_error = torch.where(
        joint_valid,
        normalized_error,
        torch.zeros_like(normalized_error),
    )
    max_error = masked_error.max(dim=-1).values
    has_valid_joint = joint_valid.any(dim=-1)
    if hasattr(command, "metrics"):
        command.metrics["active_articulation/joint_deviation_normalized"] = max_error
        command.metrics["active_articulation/joint_termination_armed"] = (
            has_valid_joint.float()
        )
    return _object_deviation_time_gate(
        env, command, start_after_first_contact, grace_frames
    ) & has_valid_joint & (max_error > float(threshold_normalized))


def _active_object_mask(command: TrackingCommand, valid: torch.Tensor, mask_name: str) -> torch.Tensor:
    if mask_name == "motion":
        mask = command.object_motion_mask
    else:
        mask = command.object_interaction_mask
    if mask is None:
        return valid
    mask = mask.to(valid.device)
    if mask.ndim == 1:
        mask = mask.view(-1, 1)
    else:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.shape[1] < valid.shape[1]:
        padded = torch.zeros(valid.shape, dtype=mask.dtype, device=valid.device)
        padded[:, : mask.shape[1]] = mask
        mask = padded
    return (mask[:, : valid.shape[1]] > 0.5) & valid


def object_pos_deviation_multi(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.2,
    mask_name: str = "interaction",
    start_after_first_contact: bool = False,
    grace_frames: int = 0,
) -> torch.Tensor:
    """Terminate when any active full-scene object exceeds the position threshold."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)

    current_pos, _, _, _, valid = utils.get_all_object_current_state(
        env, command, max_objects=max_objects
    )
    ref_pos = command.object_root_pos[:, :max_objects]
    active = _active_object_mask(command, valid, mask_name)
    pos_error = torch.norm(ref_pos - current_pos, dim=-1)
    time_gate = _object_deviation_time_gate(
        env, command, start_after_first_contact, grace_frames
    ).view(-1, 1)
    return ((pos_error > threshold) & active & time_gate).any(dim=-1)


def object_z_pos_deviation_multi(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.2,
    mask_name: str = "interaction",
    start_after_first_contact: bool = False,
    grace_frames: int = 0,
) -> torch.Tensor:
    """Terminate when any active full-scene object exceeds the Z threshold."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)

    current_pos, _, _, _, valid = utils.get_all_object_current_state(
        env, command, max_objects=max_objects
    )
    ref_z = command.object_root_pos[:, :max_objects, 2]
    active = _active_object_mask(command, valid, mask_name)
    z_error = torch.abs(ref_z - current_pos[..., 2])
    time_gate = _object_deviation_time_gate(
        env, command, start_after_first_contact, grace_frames
    ).view(-1, 1)
    return ((z_error > threshold) & active & time_gate).any(dim=-1)


def rp2_support_instability(
    env: ManagerBasedRLEnv,
    command_name: str,
    role_names: list[str] | str = "moving_support",
    pos_threshold: float = 100.0,
    ori_threshold: float = 100.0,
    height_threshold: float = 100.0,
    support_body_pos_threshold: float = 100.0,
    support_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Optional RP2 support/chair instability termination.

    Defaults are intentionally inert. Validation and curriculum jobs can lower
    thresholds through Hydra overrides once support instability should become a
    hard reset condition.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if max_objects <= 1 or not hasattr(command, "object_root_pos"):
        return terminated

    current_pos, current_quat, _, _, valid = utils.get_all_object_current_state(
        env, command, max_objects=max_objects
    )
    role_mask = utils.get_object_role_mask(env, command, role_names, max_objects, valid=valid)
    if role_mask.any():
        target_pos = command.object_root_pos[:, :max_objects]
        target_quat = command.object_root_quat[:, :max_objects]
        pos_error = torch.norm(target_pos - current_pos, dim=-1)
        ori_error = quat_error_magnitude(
            target_quat.reshape(-1, 4), current_quat.reshape(-1, 4)
        ).view(env.num_envs, max_objects)
        height_error = torch.abs(target_pos[..., 2] - current_pos[..., 2])

        pos_metric = utils.reduce_masked_object_metric(pos_error, role_mask)
        ori_metric = utils.reduce_masked_object_metric(ori_error, role_mask)
        height_metric = utils.reduce_masked_object_metric(height_error, role_mask)
        terminated |= pos_metric > pos_threshold
        terminated |= ori_metric > ori_threshold
        terminated |= height_metric > height_threshold

    if hasattr(command, "body_pos_relative_w"):
        if support_body_names is None:
            support_body_names = getattr(command.cfg, "rp2_support_body_metric_names", [])
        support_indices = _get_body_indexes(command, list(support_body_names))
        if support_indices:
            body_error = torch.norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1)
            support_body_error = body_error[:, support_indices].max(dim=-1).values
            terminated |= support_body_error > support_body_pos_threshold

    return terminated


def hand_table_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if right hand contacts the table.

    Uses force_matrix_w (filtered contact forces between table and hand links only),
    NOT net_forces_w (which includes all forces on the table such as gravity and
    the object resting on it).

    Args:
        env: The environment.
        sensor_cfg: Contact sensor config (e.g. table_to_hand_contact_sensor).
        force_threshold: Minimum total contact force (N) to trigger termination.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    contact_force = sensor.data.force_matrix_w
    force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))
    return force_magnitude > force_threshold


class GraspFailureAfterContact(ManagerTermBase):
    """Terminate if robot fails to achieve grasp after reference motion's contact phase begins.

    Uses the reference motion's ``in_contact`` label (from motion lib) to detect when
    the contact phase starts, then checks if a proper grasp (``min_contacts`` fingers
    with force > ``contact_threshold``) is ever achieved within ``grace_frames``.
    Falls back to the sim contact sensor for first-contact detection when in_contact
    labels are unavailable.

    Does NOT terminate if a grasp was achieved at any point (``has_ever_grasped``),
    even if the object is later released. State resets on env reset.
    """

    def __init__(self, cfg: TerminationTermCfg, env):
        """Initialize grasp-failure tracking state.

        Args:
            cfg: Termination term config. Optional ``params`` keys:
                ``sensor_name``, ``contact_threshold``, ``min_contacts``,
                ``grace_frames``, ``command_name``, ``hand``,
                ``require_reference_contact_label``, and
                ``reset_countdown_on_reference_release``.
            env: The manager-based RL environment.
        """
        super().__init__(cfg=cfg, env=env)
        self._first_contact_step = torch.full(
            (env.num_envs,), fill_value=-1, dtype=torch.long, device=env.device
        )
        self._has_ever_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.sensor_name = cfg.params.get("sensor_name", "object_to_hand_contact_sensor")
        self.contact_threshold = cfg.params.get("contact_threshold", 0.1)
        self.min_contacts = cfg.params.get("min_contacts", 3)
        self.grace_frames = cfg.params.get("grace_frames", 50)
        self.command_name = cfg.params.get("command_name", "motion")
        self.hand = cfg.params.get("hand", "right_hand")
        self.require_reference_contact_label = cfg.params.get(
            "require_reference_contact_label", False
        )
        # The first-contact step is latched, so by default the countdown keeps
        # running even after the reference releases the object — an episode can
        # be terminated for "failing to grasp" a contact phase that already
        # ended. When enabled, frames where the reference is NOT in contact
        # un-latch the countdown, so each reference contact phase gets a fresh
        # grace window (requires reference labels; the sim-sensor fallback
        # keeps the latched behavior).
        self.reset_countdown_on_reference_release = cfg.params.get(
            "reset_countdown_on_reference_release", False
        )
        # Episodes that begin inside a reference contact phase reset the robot
        # with its hands already posed at the object; demanding a full
        # multi-finger grasp within the grace there kills episodes whose
        # object tracking is otherwise perfect. When set, envs whose countdown
        # armed at episode step 0 only need this many finger contacts.
        self.min_contacts_when_start_in_contact = cfg.params.get(
            "min_contacts_when_start_in_contact", None
        )
        self._armed_at_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.allowed_roles = cfg.params.get("allowed_roles", None)

    def __call__(
        self,
        env,
        sensor_name: str = None,
        contact_threshold: float = None,
        min_contacts: int = None,
        grace_frames: int = None,
        command_name: str = None,
        hand: str = None,
        require_reference_contact_label: bool = None,
        reset_countdown_on_reference_release: bool = None,
        min_contacts_when_start_in_contact: int = None,
        allowed_roles: Sequence[str] | str | None = None,
    ) -> torch.Tensor:
        """Check whether a grasp was achieved within the grace period after first contact.

        Uses reference motion ``in_contact`` labels when available; falls back to
        the sim contact sensor otherwise.

        Returns:
            Boolean tensor of shape ``(num_envs,)``.
        """
        current_step = env.episode_length_buf

        command: TrackingCommand = env.command_manager.get_term(self.command_name)
        eligible = _target_object_role_eligible(
            env,
            command,
            self.allowed_roles if allowed_roles is None else allowed_roles,
        )
        ineligible = ~eligible
        self._first_contact_step[ineligible] = -1
        self._has_ever_grasped[ineligible] = False
        self._armed_at_reset[ineligible] = False
        in_contact = command.get_in_contact(self.hand)

        if in_contact is None and self.require_reference_contact_label:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        contact_force = utils.get_target_object_contact_forces(
            env,
            command,
            hand=self.hand,
            fallback_sensor_name=self.sensor_name,
        )
        if contact_force is None:
            sensor: ContactSensor = env.scene[self.sensor_name]
            contact_force = sensor.data.force_matrix_w
        force_magnitude = torch.norm(contact_force, dim=-1)
        if force_magnitude.ndim == 3:
            force_magnitude = force_magnitude.sum(dim=1)
        fingers_in_contact = (force_magnitude > self.contact_threshold).float()
        num_fingers = fingers_in_contact.sum(dim=-1)
        if self.min_contacts_when_start_in_contact is None:
            is_grasping = num_fingers >= self.min_contacts
        else:
            required = torch.where(
                self._armed_at_reset,
                torch.full_like(num_fingers, float(self.min_contacts_when_start_in_contact)),
                torch.full_like(num_fingers, float(self.min_contacts)),
            )
            is_grasping = num_fingers >= required

        if in_contact is not None:
            ref_in_contact = (in_contact > 0.5) & eligible
            if self.reset_countdown_on_reference_release:
                # A released frame un-latches the countdown, so each reference
                # contact phase re-arms with a fresh grace window instead of
                # inheriting the elapsed time of a phase that already ended.
                self._first_contact_step[~ref_in_contact] = -1
            newly_in_contact = ref_in_contact & (self._first_contact_step < 0)
            # While the command's in-contact-reset hold drives the object
            # kinematically, the grasp test is not yet meaningful — arm only
            # once the object has been released to free physics.
            hold_frames_left = getattr(command, "object_hold_frames_left", None)
            if hold_frames_left is not None:
                newly_in_contact = newly_in_contact & ~(hold_frames_left > 0)
            self._first_contact_step[newly_in_contact] = current_step[newly_in_contact]
            self._armed_at_reset[newly_in_contact] = current_step[newly_in_contact] == 0
        else:
            contact_force = utils.get_target_object_contact_forces(
                env,
                command,
                hand=self.hand,
                fallback_sensor_name=self.sensor_name,
            )
            if contact_force is None:
                sensor: ContactSensor = env.scene[self.sensor_name]
                contact_force = sensor.data.force_matrix_w
            force_magnitude = torch.norm(contact_force, dim=-1)
            if force_magnitude.ndim == 3:
                force_magnitude = force_magnitude.sum(dim=(-1, -2))
            elif force_magnitude.ndim == 2:
                force_magnitude = force_magnitude.sum(dim=-1)
            has_any_contact = force_magnitude > self.contact_threshold
            newly_in_contact = has_any_contact & eligible & (self._first_contact_step < 0)
            self._first_contact_step[newly_in_contact] = current_step[newly_in_contact]

        self._has_ever_grasped = self._has_ever_grasped | is_grasping

        has_contact = self._first_contact_step >= 0
        steps_since_contact = current_step - self._first_contact_step
        past_grace = steps_since_contact > self.grace_frames

        terminate = eligible & has_contact & past_grace & ~self._has_ever_grasped

        return terminate

    def reset(self, env_ids):
        """Reset first-contact and grasp state for the given environments."""
        self._first_contact_step[env_ids] = -1
        self._has_ever_grasped[env_ids] = False
        self._armed_at_reset[env_ids] = False
