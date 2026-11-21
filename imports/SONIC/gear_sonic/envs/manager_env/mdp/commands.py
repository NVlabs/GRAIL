"""Motion tracking command terms for humanoid whole-body control RL environments."""

from __future__ import annotations

import copy
import dataclasses
import glob
import json
import os
import re
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import easydict
import isaaclab.sim as sim_utils
import numpy as np
import torch
from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.envs.manager_env.mdp import utils as mdp_utils
from gear_sonic.isaac_utils import rotations as isaac_utils_rotations
from gear_sonic.rp2.joint_source_roster import joint_source_roster_from_view
from gear_sonic.rp2.discrete_contact import (
    CONTACT_BODY_GROUPS,
    CONTACT_ENTITY_CLASSES,
    duration_seconds_to_frames,
)
from gear_sonic.rp2.joint_roster import build_asset_compatible_motion_groups
from gear_sonic.trl.utils import common, order_converter, torch_transform
from gear_sonic.utils.generative_command import advance_motion_time_steps
from gear_sonic.utils.motion_lib import motion_lib_robot
from gear_sonic.utils.rp2_reset_starts import load_reset_start_cache
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG, FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_yaw,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Constants for multi-object mode: inactive objects placed far away
# Objects spread vertically (Z) since envs only vary in X,Y - much simpler!
INACTIVE_OBJECT_BASE_OFFSET = torch.tensor([1000.0, 0.0, -50.0])  # 1km away in X, 50m underground
INACTIVE_OBJECT_Z_SPACING = 10.0  # 10m vertical spacing between objects (must be > chair height)


def _first_required_canonical_contact_by_motion(
    reference_contact_state: torch.Tensor,
    length_starts,
    motion_keys: Sequence[str],
) -> dict[str, int]:
    """Return first active-entity required-contact frame for every motion."""
    state = torch.as_tensor(reference_contact_state)
    expected_tail = (len(CONTACT_ENTITY_CLASSES), len(CONTACT_BODY_GROUPS))
    if state.ndim != 3 or tuple(state.shape[1:]) != expected_tail:
        raise ValueError(
            "Canonical reference contact tensor must have shape "
            f"[T, {expected_tail[0]}, {expected_tail[1]}], got "
            f"{tuple(state.shape)}"
        )
    starts = torch.as_tensor(length_starts, dtype=torch.long).cpu().tolist()
    keys = [str(key) for key in motion_keys]
    if len(starts) != len(keys):
        raise ValueError(
            f"Canonical contact starts/keys mismatch: {len(starts)} vs "
            f"{len(keys)}"
        )
    active_axis = CONTACT_ENTITY_CLASSES.index("active_entity")
    result = {}
    for index, key in enumerate(keys):
        start = int(starts[index])
        end = (
            int(starts[index + 1])
            if index + 1 < len(starts)
            else int(state.shape[0])
        )
        if not 0 <= start < end <= int(state.shape[0]):
            raise ValueError(
                f"{key}: invalid canonical contact frame span [{start}, {end})"
            )
        required = (state[start:end, active_axis] > 0).any(dim=-1)
        frames = torch.where(required)[0]
        result[key] = int(frames[0].item()) if len(frames) else end - start
    return result


def _mechanism_resistance_tensors(
    slots: Sequence[dict],
    *,
    num_envs: int,
    num_joints: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Expand per-slot mechanism resistance into simulator joint tensors."""
    if len(slots) != int(num_envs):
        raise ValueError(
            "Mechanism-resistance slot count differs from environment count: "
            f"{len(slots)} vs {num_envs}"
        )
    fields = (
        "static_friction",
        "dynamic_friction",
        "viscous_friction",
        "damping",
    )
    rows = []
    for env_id, slot in enumerate(slots):
        contract = slot.get("mechanism_resistance")
        if not isinstance(contract, dict):
            raise ValueError(
                f"Joint-roster slot {env_id} is missing mechanism_resistance"
            )
        missing = [field for field in fields if field not in contract]
        if missing:
            raise ValueError(
                f"Joint-roster slot {env_id} mechanism resistance is missing "
                f"{missing}"
            )
        rows.append([float(contract[field]) for field in fields])
    values = torch.tensor(rows, dtype=dtype, device=device)
    if not torch.all(torch.isfinite(values)) or torch.any(values < 0.0):
        raise ValueError(
            "Joint-roster mechanism resistance must be finite and non-negative"
        )
    return {
        field: values[:, field_id : field_id + 1].expand(
            int(num_envs), int(num_joints)
        )
        for field_id, field in enumerate(fields)
    }


def _scene_object_key_from_stem(stem: str) -> str:
    return f"object_{stem.replace('-', '_')}"


def motion_order_object_scene_keys(
    env,
    max_objects: int | None = None,
    motion_key: str | None = None,
) -> list[str]:
    """Return multi-object scene keys in the same order as object USD stems."""
    rigid_objects = getattr(env.scene, "rigid_objects", {})
    stems_by_motion = getattr(env.cfg, "object_usd_stems_by_motion", None) or {}
    slot_count = int(getattr(env.cfg, "object_usd_slot_count", 0) or 0)
    if slot_count > 0:
        if motion_key is not None and motion_key in stems_by_motion:
            num_motion_objects = len(stems_by_motion[motion_key])
        else:
            num_motion_objects = slot_count
        limit = min(slot_count, num_motion_objects)
        if max_objects is not None:
            limit = min(limit, max_objects)
        return [f"object_{idx}" for idx in range(limit) if f"object_{idx}" in rigid_objects]
    if motion_key is not None and motion_key in stems_by_motion:
        stems = stems_by_motion[motion_key]
    else:
        stems = getattr(env.cfg, "object_usd_stems", None) or []
    keys = []
    for stem in stems:
        key = _scene_object_key_from_stem(stem)
        if key in rigid_objects:
            keys.append(key)
    if not keys:
        keys = sorted(k for k in rigid_objects if k.startswith("object_"))
    if max_objects is not None:
        keys = keys[:max_objects]
    return keys


def _init_variable_frames(
    enabled: bool,
    min_frames: int,
    num_future_frames: int,
    step: int,
    num_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Initialize variable frame support tensors.

    Returns (per_env_num_frames, frame_choices) or (None, None) if disabled.
    """
    if not enabled:
        return None, None
    assert step > 0, f"variable_frames_step must be positive, got {step}"
    per_env_num_frames = torch.full((num_envs,), num_future_frames, device=device, dtype=torch.long)
    frame_choices = torch.arange(min_frames, num_future_frames + 1, step, device=device)
    assert len(frame_choices) > 0, (
        f"No valid frame choices: variable_frames_min={min_frames} "
        f"> num_future_frames={num_future_frames}"
    )
    return per_env_num_frames, frame_choices


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = None
    force = None


class TrackingCommand(CommandTerm):
    """Provide reference motion trajectories for motion-tracking RL.

    This is the primary command term for SONIC-style humanoid control. It manages
    a motion library of pre-recorded motion clips (robot joint trajectories, SMPL
    poses, and optionally object trajectories) and serves reference frames to the
    policy and reward system at each simulation step.

    Key responsibilities:
        - Load and index motion clips from the motion library (MotionLibRobot).
        - Sample motion IDs and start times for each environment at episode reset.
        - Advance the time cursor each step and provide current + multi-future
          reference frames (positions, orientations, velocities, joint states).
        - Transform reference quantities into robot-local, egocentric, or
          heading-canonicalized coordinate frames for observation terms.
        - Handle DOF mismatch between motion data and robot (e.g., extra finger
          joints not present in the motion library).
        - Support adaptive sampling, contact-based initialization, variable future
          frame counts, encoder mode sampling (G1/SMPL/teleop), and multi-object
          scene management.

    The command exposes ~60 property-based accessors that observation and reward
    terms query each step. Properties are named by the pattern::

        {quantity}[_multi_future][_dif][_{frame}]

    where ``quantity`` is body_pos, joint_pos, smpl_pose, etc.; ``multi_future``
    means values for all ``num_future_frames`` reference frames stacked;
    ``dif`` is the difference relative to the robot; and ``frame`` is ``w``
    (world), ``l`` (robot-local / de-headed), or ``b`` (body).
    """

    cfg: TrackingCommandCfg

    def __init__(
        self,
        cfg: TrackingCommandCfg,
        env: ManagerBasedRLEnv,
        max_num_load_motions: int = None,  # noqa: RUF013
    ):
        """Initialize the tracking command with motion library and body mappings.

        Loads motion data, builds DOF/body index mappings between the motion
        library (MuJoCo ordering) and the IsaacLab robot, pre-allocates per-env
        tensors for motion IDs, time steps, and future frame offsets, and
        optionally sets up height-map raycasting.

        Args:
            cfg: Configuration dataclass specifying motion files, body names,
                future frame settings, encoder sampling probabilities, etc.
            env: The manager-based RL environment that owns this command.
            max_num_load_motions: Cap on how many unique motion clips to load
                into GPU memory. Defaults to ``min(num_envs, 1024)`` or the
                full library when ``use_paired_motions`` is enabled.
        """
        super().__init__(cfg, env)

        self.is_evaluating = False
        self.cmd_body_names = self.cfg.body_names
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.vr_3point_body_indices = [
            self.robot.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_indices_motion = [
            self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_offsets = (
            torch.tensor(self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device)
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )

        self.reward_point_body_indices = [
            self.robot.body_names.index(name) for name in self.cfg.reward_point_body
        ]
        self.reward_point_body_offsets = (
            torch.tensor(self.cfg.reward_point_body_offset, dtype=torch.float32, device=self.device)
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )
        self.reward_point_body_indices_motion = [
            self.cfg.body_names.index(name) for name in self.cfg.reward_point_body
        ]

        self.down_dir = (
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        isaac_lab_joints = env.cfg.isaaclab_to_mujoco_mapping["isaaclab_joints"]

        self.isaaclab_to_mujoco_dof = env.cfg.isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_dof"]
        self.mujoco_to_isaaclab_dof = env.cfg.isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_dof"]
        self.lower_joint_indices_mujoco = list(range(12))
        self.lower_joint_isaaclab_indices = [
            self.isaaclab_to_mujoco_dof[i] for i in self.lower_joint_indices_mujoco
        ]
        self.isaaclab_to_mujoco_body = env.cfg.isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_body"]
        self.mujoco_to_isaaclab_body = env.cfg.isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_body"]
        self.running_ref_root_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.body_indexes_data = [isaac_lab_joints.index(name) for name in self.cfg.body_names]
        if self.cfg.motion_lib_cfg is not None:
            motion_lib_cfg = easydict.EasyDict(self.cfg.motion_lib_cfg)
        else:
            motion_lib_cfg = easydict.EasyDict(
                {
                    "motion_file": self.cfg.motion_file,
                    "smpl_motion_file": getattr(self.cfg, "smpl_motion_file", None),
                    "asset": {
                        "assetRoot": "gear_sonic/data/assets/bm/unitree_description/mjcf/",
                        "assetFileName": "g1_29dof_rev_1_0.xml",
                        "urdfFileName": "",
                    },
                    "extend_config": [],
                    "target_fps": 50,
                    "multi_thread": True,
                    "filter_motion_keys": self.cfg.filter_motion_keys,
                }
            )
        self.use_asset_compatible_motions = bool(
            self.cfg.use_asset_compatible_motions
        )
        self.asset_compatible_roster = getattr(
            env.cfg, "joint_roster_view", None
        )
        if self.use_asset_compatible_motions:
            if self.cfg.use_paired_motions:
                raise ValueError(
                    "use_asset_compatible_motions and use_paired_motions are "
                    "mutually exclusive"
                )
            if not isinstance(self.asset_compatible_roster, dict):
                raise ValueError(
                    "Asset-compatible motion sampling requires a resolved "
                    "rank-local joint roster"
                )
            motion_lib_cfg["motion_file"] = self.asset_compatible_roster[
                "motion_dir"
            ]
            motion_lib_cfg["object_motion_file"] = (
                self.asset_compatible_roster["object_motion_dir"]
            )
            motion_lib_cfg["bps_dir"] = self.asset_compatible_roster[
                "bps_dir"
            ]
            motion_lib_cfg["filter_motion_keys"] = list(
                self.asset_compatible_roster["motion_aliases"]
            )
            # The no-VOF cohorts use this manifest only for immutable USD slot
            # compatibility. They do not use the canonical-contact or
            # checkpoint-hashed joint-source sampler.
            # Source::motion identity tracking reads the same manifest records
            # (source, source_motion_key, source_sample_fps_hz, world_size),
            # which every unified cohort carries whether or not it was
            # materialized with --track-source-stats. An explicit
            # motion_lib_cfg.joint_source_tracking (true/false) therefore
            # decides at runtime without re-materializing; when unset the
            # cohort's own track_source_stats annotation applies as before.
            tracking_override = motion_lib_cfg.get("joint_source_tracking")
            if tracking_override is None:
                skip_tracking = bool(
                    self.asset_compatible_roster.get("skip_joint_source_tracking", False)
                )
            else:
                skip_tracking = not bool(tracking_override)
            # Identity source when tracking is effective: a manifest FILE (unified
            # cohorts) is opened by the motion library as before; a direct-directory
            # roster labels manifest_path with the pack ROOT DIRECTORY (never opened)
            # and carries the identities in memory (identity_records,
            # direct_roster_identity_v1) -> motion_lib_cfg.joint_source_roster. The
            # roster's skip flag only says "no identity file"; it never discards
            # in-memory identities when tracking was asked for.
            roster_manifest_label = self.asset_compatible_roster["manifest_path"]
            motion_lib_cfg["joint_source_manifest"] = None
            motion_lib_cfg["joint_source_roster"] = None
            if not skip_tracking:
                if os.path.isfile(str(roster_manifest_label)):
                    motion_lib_cfg["joint_source_manifest"] = roster_manifest_label
                else:
                    motion_lib_cfg["joint_source_roster"] = joint_source_roster_from_view(
                        self.asset_compatible_roster
                    )
            # Unified rank-local rosters mix sources per rank. Carry the roster
            # manifest under its own key (joint_source_manifest above stays None
            # unless the cohort tracks source stats) so the motion library
            # transports the assembled per-hand legacy contact placeholders on
            # pure-source ranks exactly as mixed ranks already do.
            motion_lib_cfg["joint_roster_manifest"] = self.asset_compatible_roster[
                "manifest_path"
            ]
            motion_lib_cfg["motion_shard_world_size"] = 1
            motion_lib_cfg["motion_shard_rank"] = 0
        # Only override filter_motion_keys if explicitly set at command level
        # (don't overwrite the value from motion_lib_cfg if it exists there)
        filter_keys = self.cfg.filter_motion_keys
        if filter_keys is None:
            filter_keys = motion_lib_cfg.get("filter_motion_keys", None)

        # Per-rank motion key filtering: if scene config wrote a keys file,
        # use it to limit motion loading to only this rank's needed motions.
        per_rank_keys_file = getattr(env.cfg, "config", {}).get("per_rank_motion_keys_file", None)
        if per_rank_keys_file and os.path.isfile(per_rank_keys_file):
            with open(per_rank_keys_file) as _kf:
                per_rank_keys = [line.strip() for line in _kf if line.strip()]
            if per_rank_keys:
                # Combine with existing filter if any
                if filter_keys is not None:
                    filter_keys = list(filter_keys) + per_rank_keys
                else:
                    filter_keys = per_rank_keys
                # Caller already did rank-aware filtering (one keys file per rank).
                # Disable motion_lib's generic shard-slicing to avoid double-partitioning
                # the already-sliced set — e.g. use_paired_motions pairing check at
                # commands.py:329 sees usd_count vs motion_count/world_size mismatch.
                motion_lib_cfg["motion_shard_world_size"] = 1
                print(
                    f"[PerRankMotion] Loaded {len(per_rank_keys)} motion keys from "
                    f"{per_rank_keys_file}; forced motion_shard_world_size=1"
                )

        motion_lib_cfg.update(
            {
                "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
                "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
                "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
                "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
                "body_indexes": self.body_indexes,
                "body_indexes_data": self.body_indexes_data,
                "filter_motion_keys": filter_keys,
                "lower_joint_indices_mujoco": self.lower_joint_indices_mujoco,
                "cat_upper_body_poses": self.cfg.cat_upper_body_poses,
                "cat_upper_body_poses_prob": self.cfg.cat_upper_body_poses_prob,
                "randomize_heading": self.cfg.randomize_heading,
                "freeze_frame_aug": self.cfg.freeze_frame_aug,
                "freeze_frame_aug_prob": self.cfg.freeze_frame_aug_prob,
                "randomize_wrist_poses": self.cfg.randomize_wrist_poses,
                "randomize_wrist_prob": self.cfg.randomize_wrist_prob,
                "randomize_wrist_std": self.cfg.randomize_wrist_std,
                # Reset-only retarget channels use one canonical body width.
                # Source hand encodings remain in their dedicated channels.
                "retarget_initialization_dof_count": int(
                    self.cfg.motion_lib_num_dof or self.robot.num_joints
                ),
            }
        )

        # If flat_motion_dir is set, combine terrain + flat dirs for motion loading
        _flat_motion_dir = getattr(env.cfg, "config", {}).get("flat_motion_dir", None)
        if _flat_motion_dir and os.path.isdir(str(motion_lib_cfg.get("motion_file", ""))):
            _terrain_dir = motion_lib_cfg["motion_file"]
            motion_lib_cfg["motion_file"] = [_terrain_dir, _flat_motion_dir]
            print(
                f"[TerrainMotionLib] Combined motion_file: "
                f"terrain={_terrain_dir}, flat={_flat_motion_dir}"
            )

        self.motion_lib = motion_lib_robot.MotionLibRobot(
            motion_lib_cfg, self.num_envs, self.device
        )
        if max_num_load_motions is None:
            if self.cfg.use_paired_motions or self.use_asset_compatible_motions:
                self.max_num_load_motions = self.motion_lib._num_unique_motions  # noqa: SLF001
            else:
                self.max_num_load_motions = min(self.num_envs, 1024)
        else:
            self.max_num_load_motions = max_num_load_motions

        _terrain_aware_cfg = getattr(self.cfg, "terrain_aware_sampling", False)
        _n_terrain_cfg = getattr(env.cfg, "config", {}).get("n_terrain_motions", 0)

        if _terrain_aware_cfg and _n_terrain_cfg > 0:
            # Terrain-aware load: lock terrain motions at [0, n_terrain), random flat after
            _n_flat_to_load = max(self.max_num_load_motions - _n_terrain_cfg, 0)
            self.motion_lib.load_motions_terrain_aware(
                n_terrain=_n_terrain_cfg,
                n_flat_to_load=_n_flat_to_load,
            )
            self.max_num_load_motions = self.motion_lib._num_motions
        else:
            self.motion_lib.load_motions_for_training(max_num_seqs=self.max_num_load_motions)
        self.use_adaptive_sampling = self.motion_lib.use_adaptive_sampling
        self._asset_compatible_motion_groups = ()
        self._asset_compatible_env_group_ids = None
        self._asset_compatible_representative_motion_ids = None
        if self.use_asset_compatible_motions:
            grouping = build_asset_compatible_motion_groups(
                self.asset_compatible_roster,
                self.motion_lib.curr_motion_keys,
            )
            self._asset_compatible_motion_groups = tuple(
                torch.tensor(
                    group["motion_ids"],
                    dtype=torch.long,
                    device=self.device,
                )
                for group in grouping["groups"]
            )
            self._asset_compatible_env_group_ids = torch.tensor(
                grouping["env_group_ids"],
                dtype=torch.long,
                device=self.device,
            )
            self._asset_compatible_representative_motion_ids = torch.tensor(
                grouping["representative_motion_ids"],
                dtype=torch.long,
                device=self.device,
            )
            self._apply_joint_roster_mechanism_resistance()
            self._apply_joint_roster_physics_overrides()

        # Load contact data for contact-based initialization
        self._load_contact_data()
        self._load_reset_start_cache()

        # Setup DOF mapping for handling mismatch between motion library and robot

        self.robot_num_dof = self.robot.num_joints
        self.motion_lib_num_dof = self.cfg.motion_lib_num_dof
        if self.motion_lib_num_dof is None:
            self.motion_lib_num_dof = self.robot_num_dof

        self.extra_num_dof = self.robot_num_dof - self.motion_lib_num_dof
        self.has_dof_mismatch = self.extra_num_dof > 0

        if self.has_dof_mismatch:
            self.body_joint_indices = joint_utils.get_body_joint_indices(self.robot)
            self.extra_joint_indices = joint_utils.get_hand_joint_indices(self.robot)
            self.extra_default_positions = torch.tensor(
                self.cfg.hand_default_positions or [0.0] * self.extra_num_dof,
                dtype=torch.float32,
                device=self.device,
            )
            self.extra_default_velocities = torch.tensor(
                self.cfg.hand_default_velocities or [0.0] * self.extra_num_dof,
                dtype=torch.float32,
                device=self.device,
            )

            # Reference-hand-DOF reset: map reference dof[29:43] to robot joints BY NAME.
            # The data (converter output) hand block is in HAND_ORDER_USD (breadth-first,
            # == IsaacLab USD articulation order); the IsaacGym source URDF order was
            # thumb-first per finger and already remapped by convert_dex31_to_motion_lib.py.
            self.reference_hand_dof_indices = None
            if bool(getattr(self.cfg, "reset_hand_from_reference", False)):
                hand_order_usd = [
                    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
                    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
                    "left_hand_thumb_2_joint",
                    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
                    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
                    "right_hand_thumb_2_joint",
                ]
                robot_names = list(self.robot.joint_names)
                missing = [n for n in hand_order_usd if n not in robot_names]
                if missing:
                    print(f"[HandDOF] WARNING: hand joints not on robot, disabling ref-hand init: {missing}", flush=True)
                else:
                    self.reference_hand_dof_indices = torch.tensor(
                        [robot_names.index(n) for n in hand_order_usd],
                        dtype=torch.long, device=self.device,
                    )
                print("[HandDOF] IsaacLab robot.joint_names (43):", robot_names, flush=True)
                print("[HandDOF] IsaacLab robot.joint_names[-14:]:", robot_names[-14:], flush=True)
                print("[HandDOF] data hand order (HAND_ORDER_USD):", hand_order_usd, flush=True)
                if self.reference_hand_dof_indices is not None:
                    print("[HandDOF] reference dof[29:43] -> robot joint indices:",
                          self.reference_hand_dof_indices.tolist(), flush=True)

        # Step 1: Select which motions to use
        if self.use_asset_compatible_motions:
            self.motion_ids = self._sample_asset_compatible_motion_ids(
                torch.arange(self.num_envs, device=self.device),
                evaluation=False,
            )
        elif self.cfg.use_paired_motions:
            # Assign motion IDs sequentially (wraps around if more envs than motions)
            self.motion_ids = (
                torch.arange(self.num_envs, device=self.device)
                % self.motion_lib._num_motions  # noqa: SLF001
            )
            # ── Verify motion-USD pairing ──
            # MultiUsdFileCfg(random_choice=False) assigns USD[k % N_usd] to env k.
            # use_paired_motions assigns motion[k % N_mot] to env k.
            # For correct pairing we need: N_usd == N_mot AND stems identical in order.
            motion_keys = self.motion_lib.curr_motion_keys
            usd_stems = getattr(self._env.cfg, "object_usd_stems", None)
            if usd_stems is None:
                usd_stems = getattr(
                    self._env.cfg, "active_articulation_usd_stems", None
                )
            if usd_stems is not None:
                num_usds = len(usd_stems)
                num_motions = len(motion_keys)
                full_scene_per_env = getattr(self.cfg, "full_scene_per_env", False)
                env_multi_object = getattr(env.cfg, "config", {}).get("multi_object_per_env", False)
                if full_scene_per_env and env_multi_object:
                    stems_by_motion = getattr(self._env.cfg, "object_usd_stems_by_motion", None) or {}
                    missing_lists = [key for key in motion_keys if key not in stems_by_motion]
                    if stems_by_motion and missing_lists:
                        raise ValueError(
                            "[FATAL] Full-scene per-motion USD lists missing for "
                            f"{len(missing_lists)} motion(s), first={missing_lists[:5]}"
                        )
                    print(  # noqa: T201
                        f"[OK] Full-scene multi-object mode: {num_usds} object USDs for "
                        f"{num_motions} motion(s); using "
                        f"{len(stems_by_motion) or 1} per-motion object order list(s)."
                    )
                else:
                    # Check 1: counts must match
                    if num_usds != num_motions:
                        raise ValueError(
                            f"[FATAL] Motion-USD count mismatch: {num_usds} USDs vs {num_motions} motions. "
                            f"Cannot guarantee per-env pairing. Fix the data or use motion-driven USD resolution."
                        )
                    # Check 2: every stem must match in order
                    mismatches = [
                        (i, usd_stems[i], motion_keys[i])
                        for i in range(num_motions)
                        if usd_stems[i] != motion_keys[i]
                    ]
                    if mismatches:
                        detail = "\n".join(
                            f"  idx {i}: USD={u}  MOTION={m}" for i, u, m in mismatches[:10]
                        )
                        raise ValueError(
                            f"[FATAL] Motion-USD stem mismatch at {len(mismatches)}/{num_motions} indices!\n{detail}"
                        )
                    # All checks passed — log sample pairings
                    print(  # noqa: T201
                        f"[OK] Motion-USD pairing verified: {num_motions} motions ↔ {num_usds} USDs, all stems match."
                    )
                    for k in [0, num_motions // 2, num_motions - 1]:
                        print(f"  env {k}: {motion_keys[k]}")  # noqa: T201

                    # ── Terrain-mode extra checks ──
                    # n_terrain_motions is actually n_terrain_ENVS (scene cfg sets
                    # it from the env count, not motion count). With num_envs >
                    # unique terrain motions, env indices wrap — so clamp to the
                    # actual loaded motion count before checking stem alignment.
                    _n_terrain_envs = getattr(env.cfg, "config", {}).get("n_terrain_motions", 0)
                    _n_check = min(_n_terrain_envs, num_motions)
                    if _n_check > 0:
                        terrain_stems = list(motion_keys[:_n_check])
                        terrain_usd_stems = list(usd_stems[:_n_check])
                        t_mismatches = [
                            (i, terrain_usd_stems[i], terrain_stems[i])
                            for i in range(_n_check)
                            if terrain_usd_stems[i] != terrain_stems[i]
                        ]
                        if t_mismatches:
                            detail = "\n".join(
                                f"  idx {i}: USD={u}  MOTION={m}" for i, u, m in t_mismatches[:10]
                            )
                            raise ValueError(
                                f"[FATAL] Terrain motion-USD mismatch at "
                                f"{len(t_mismatches)}/{_n_check} terrain indices!\n{detail}"
                            )
                        print(  # noqa: T201
                            f"[OK] Terrain-mode pairing verified: {_n_check} unique terrain motions ↔ USDs "
                            f"(n_terrain_envs={_n_terrain_envs} with wrap)."
                        )
        elif getattr(self.cfg, "sample_unique_motions", False):
            # Sample without replacement - each env gets a unique motion
            num_available = len(self.motion_lib._curr_motion_ids)  # noqa: SLF001
            if self.num_envs > num_available:
                raise ValueError(
                    f"sample_unique_motions=True requires num_envs ({self.num_envs}) <= "
                    f"num_available_motions ({num_available})"
                )
            perm = torch.randperm(num_available, device=self.device)[: self.num_envs]
            self.motion_ids = perm
            print(  # noqa: T201
                f"[TrackingCommand] Sampled {self.num_envs} unique motions (no duplicates)"
            )  # noqa: T201
        else:
            # Random sampling (can have duplicates)
            self.motion_ids = self.motion_lib.sample_motions(self.num_envs)

        # Step 2: Sample start time steps for selected motions
        initial_env_ids = torch.arange(self.num_envs, device=self.device)
        self.motion_start_time_steps = self._sample_start_time_steps_for_current_motions(
            initial_env_ids
        )
        self.motion_start_time_steps = self._sample_from_reset_start_cache(
            initial_env_ids, self.motion_start_time_steps
        )
        if (
            self.cfg.sample_before_contact
            and self._first_contact_frame is not None
            and not self.is_evaluating
        ):
            self.motion_start_time_steps = self._sample_before_contact(
                initial_env_ids, self.motion_start_time_steps
            )
        self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
        self._apply_ownership_phase_end(torch.arange(self.num_envs, device=self.device))

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Completed-episode attribution for adaptive sampling. Isaac Lab steps
        # termination -> _reset_idx (CommandTerm.reset -> _resample_command)
        # -> command_manager.compute (_update_command), so when the
        # update_on_episode_end_only update runs, motion_ids /
        # motion_start_time_steps / time_steps of a finished env already
        # describe its NEXT episode. _resample_command therefore snapshots the
        # finished episode into this pending buffer and _update_command
        # consumes it (see _snapshot_completed_episodes).
        self._adaptive_terminal_only = bool(
            self.use_adaptive_sampling
            and self.motion_lib.adaptive_sampling_cfg.get(
                "update_on_episode_end_only", False
            )
        )
        self._pending_episode_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pending_episode_failed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Private numeric snapshot buffers are int64 by construction: the live motion ids /
        # start steps may be native int32 (a restored motion state replaces the live tensors
        # with the restored dtypes, see set_motion_state), and an indexed assignment never
        # casts, so the snapshot casts explicitly into these buffers instead.
        self._pending_episode_motion_ids = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._pending_episode_start_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._pending_episode_end_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

        # Remaining hold frames for the in-contact-reset hold-and-release.
        # Always allocated (zeros = inactive) so terminations can read it
        # without config coupling.
        self.object_hold_frames_left = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.object_hold_total_frames = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.object_hold_exact_frames = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.object_hold_release_frames = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Terrain-aware sampling: split envs into flat (resample freely) and terrain (fixed motion)
        self._terrain_aware = getattr(self.cfg, "terrain_aware_sampling", False)
        self._n_terrain_motions = 0
        if self._terrain_aware:
            R = getattr(env.cfg, "config", {}).get(
                "flat_to_terrain_ratio", getattr(self.cfg, "flat_to_terrain_ratio", 4)
            )
            block_size = R + 1
            env_ids_t = torch.arange(self.num_envs, device=self.device)
            self._env_is_terrain = (env_ids_t % block_size) == R
            self._env_is_flat = ~self._env_is_terrain
            n_flat = self._env_is_flat.sum().item()
            n_terrain = self._env_is_terrain.sum().item()

            self._n_terrain_motions = getattr(env.cfg, "config", {}).get("n_terrain_motions", 0)
            if self._n_terrain_motions > 0:
                terrain_env_ids = torch.where(self._env_is_terrain)[0]
                flat_env_ids = torch.where(self._env_is_flat)[0]
                self.motion_ids[terrain_env_ids] = torch.arange(
                    len(terrain_env_ids), device=self.device
                )
                n_flat_motions = self.motion_lib._num_motions - self._n_terrain_motions
                if n_flat_motions > 0:
                    self.motion_ids[flat_env_ids] = torch.randint(
                        self._n_terrain_motions,
                        self.motion_lib._num_motions,
                        (len(flat_env_ids),),
                        device=self.device,
                    )
                self.motion_start_time_steps = self.motion_lib.sample_time_steps(
                    self.motion_ids, truncate_time=self.cfg.sample_start_truncate_time_s
                )
                self.motion_start_time_steps = self._sample_from_reset_start_cache(
                    env_ids_t, self.motion_start_time_steps
                )
                if (
                    self.cfg.sample_before_contact
                    and self._first_contact_frame is not None
                    and not self.is_evaluating
                ):
                    self.motion_start_time_steps = self._sample_before_contact(
                        env_ids_t, self.motion_start_time_steps
                    )
                self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
                self._apply_ownership_phase_end(env_ids_t)
                print(
                    f"[TerrainAwareSampling] Bound {n_terrain} terrain envs to motion_ids [0,{self._n_terrain_motions}), "
                    f"flat envs sample from [{self._n_terrain_motions},{self.motion_lib._num_motions})"
                )

            print(
                f"[TerrainAwareSampling] ratio={R}:1, block_size={block_size}, "
                f"flat_envs={n_flat}, terrain_envs={n_terrain}"
            )

        # Object position randomization offset (per-env, resampled at reset)
        self._object_position_offset = torch.zeros(self.num_envs, 3, device=self.device)

        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.num_future_frames = self.cfg.num_future_frames
        # Motion lib is at target_fps; ref frames are spaced by dt_future_ref_frames (seconds).
        # frame_skips = number of motion-lib steps between consecutive ref frames (integer).
        # Effective spacing equals dt_future_ref_frames only when (dt_future_ref_frames * target_fps) is an integer;  # noqa: E501
        # otherwise integer division truncates and the velocity calculation will be incorrect.
        self.frame_skips = self.cfg.dt_future_ref_frames // (1.0 / motion_lib_cfg.target_fps)
        steps_exact = self.cfg.dt_future_ref_frames * motion_lib_cfg.target_fps
        if abs(steps_exact - round(steps_exact)) > 1e-9:
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"dt_future_ref_frames={self.cfg.dt_future_ref_frames} * target_fps={motion_lib_cfg.target_fps} "
                f"= {steps_exact} is not an integer; "
                f"using frame_skips={self.frame_skips} so effective ref-frame spacing is "
                f"{self.frame_skips / motion_lib_cfg.target_fps:.4f}s (not {self.cfg.dt_future_ref_frames}s). "
                "Velocity-based losses may use incorrect dt.",
                stacklevel=2,
            )

        self.future_time_steps_init = (
            (
                torch.arange(self.num_future_frames, device=self.device, dtype=torch.long)
                * self.frame_skips
            )
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )

        if self.cfg.smpl_num_future_frames is None:
            self.smpl_num_future_frames = self.num_future_frames
        else:
            self.smpl_num_future_frames = self.cfg.smpl_num_future_frames
        if self.cfg.smpl_dt_future_ref_frames is None:
            self.smpl_dt_future_ref_frames = self.cfg.dt_future_ref_frames
        else:
            self.smpl_dt_future_ref_frames = self.cfg.smpl_dt_future_ref_frames

        self.smpl_frame_skips = self.smpl_dt_future_ref_frames // (1.0 / motion_lib_cfg.target_fps)
        self.smpl_future_time_steps_init = (
            (
                torch.arange(self.smpl_num_future_frames, device=self.device, dtype=torch.long)
                * self.smpl_frame_skips
            )
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )

        self.future_motion_ids = self.motion_ids.repeat_interleave(self.num_future_frames)
        self.smpl_future_motion_ids = self.motion_ids.repeat_interleave(self.smpl_num_future_frames)

        # Variable frame support
        self.variable_frames_enabled = getattr(self.cfg, "variable_frames_enabled", False)
        self.per_env_num_frames, self._frame_choices = _init_variable_frames(
            self.variable_frames_enabled,
            getattr(self.cfg, "variable_frames_min", 16),
            self.num_future_frames,
            getattr(self.cfg, "variable_frames_step", 4),
            self.num_envs,
            self.device,
        )

        self.encoder_sample_probs_dict = self.cfg.encoder_sample_probs
        self.optimize_encoders_ratio_for_CHIP = getattr(
            self.cfg, "optimize_encoders_ratio_for_CHIP", False
        )
        self.encoder_sample_probs = None
        if self.encoder_sample_probs_dict is not None:
            self.encoder_sample_probs = torch.tensor(
                list(self.encoder_sample_probs_dict.values()), device=self.device
            )
            self.encoder_sample_probs = self.encoder_sample_probs / self.encoder_sample_probs.sum()
            self.encoder_sample_probs_no_smpl_dict = copy.deepcopy(self.encoder_sample_probs_dict)

            if "smpl" in self.encoder_sample_probs_no_smpl_dict:
                self.encoder_sample_probs_no_smpl_dict["smpl"] = 0.0
            self.encoder_sample_probs_no_smpl = torch.tensor(
                list(self.encoder_sample_probs_no_smpl_dict.values()), device=self.device
            )
            self.encoder_sample_probs_no_smpl = (
                self.encoder_sample_probs_no_smpl / self.encoder_sample_probs_no_smpl.sum()
            )
            self.g1_encoder_index = list(self.encoder_sample_probs_dict.keys()).index("g1")
            if "smpl" in self.encoder_sample_probs_dict:
                self.smpl_encoder_index = list(self.encoder_sample_probs_dict.keys()).index("smpl")
            else:
                self.smpl_encoder_index = None

            if "teleop" in self.encoder_sample_probs_dict:
                self.teleop_encoder_index = list(self.encoder_sample_probs_dict.keys()).index(
                    "teleop"
                )
            else:
                self.teleop_encoder_index = None

            if "soma" in self.encoder_sample_probs_dict:
                self.soma_encoder_index = list(self.encoder_sample_probs_dict.keys()).index("soma")
                encoder_sample_probs_no_soma_dict = copy.deepcopy(self.encoder_sample_probs_dict)
                encoder_sample_probs_no_soma_dict["soma"] = 0.0
                self.encoder_sample_probs_no_soma = torch.tensor(
                    list(encoder_sample_probs_no_soma_dict.values()), device=self.device
                )
                no_soma_sum = self.encoder_sample_probs_no_soma.sum()
                if no_soma_sum > 0:
                    self.encoder_sample_probs_no_soma = (
                        self.encoder_sample_probs_no_soma / no_soma_sum
                    )
                else:
                    # All probs zero (e.g. use_encoder=soma forced) — fall back to G1
                    self.encoder_sample_probs_no_soma[self.g1_encoder_index] = 1.0
                encoder_sample_probs_no_smpl_no_soma_dict = copy.deepcopy(
                    self.encoder_sample_probs_no_smpl_dict
                )
                if "soma" in encoder_sample_probs_no_smpl_no_soma_dict:
                    encoder_sample_probs_no_smpl_no_soma_dict["soma"] = 0.0
                self.encoder_sample_probs_no_smpl_no_soma = torch.tensor(
                    list(encoder_sample_probs_no_smpl_no_soma_dict.values()), device=self.device
                )
                no_smpl_no_soma_sum = self.encoder_sample_probs_no_smpl_no_soma.sum()
                if no_smpl_no_soma_sum > 0:
                    self.encoder_sample_probs_no_smpl_no_soma = (
                        self.encoder_sample_probs_no_smpl_no_soma / no_smpl_no_soma_sum
                    )
                else:
                    # All probs zero — fall back to G1
                    self.encoder_sample_probs_no_smpl_no_soma[self.g1_encoder_index] = 1.0
            else:
                self.soma_encoder_index = None
                self.encoder_sample_probs_no_soma = None
                self.encoder_sample_probs_no_smpl_no_soma = None

            self.teleop_sample_prob_when_smpl = self.cfg.teleop_sample_prob_when_smpl

            self.encoder_index = torch.zeros(
                (self.num_envs, self.encoder_sample_probs.shape[0]),
                dtype=torch.long,
                device=self.device,
            )

            if "smpl" in self.encoder_sample_probs_dict:
                assert (
                    self.smpl_encoder_index > self.g1_encoder_index
                ), f"SMPL encoder index {self.smpl_encoder_index} must be greater than G1 encoder index {self.g1_encoder_index} to when both exist!"  # noqa: E501

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

        self.use_ref_motion_root_quat_w_as_anchor = False
        self.ref_motion_root_rotation_noise = None
        self.num_bodies = len(self.cfg.body_names)

        # Multi-object mode detection: check for multiple object_* entries in scene
        # (dynamically detected to avoid storing metadata on scene config)
        self._multi_object_mode = False
        self._object_names = []  # List of (safe_name, original_name) tuples
        self._active_object_name = None

        # Check if we have multiple objects (object_*) vs single object ("object")
        if hasattr(self._env, "scene") and hasattr(self._env.scene, "rigid_objects"):
            rigid_obj_keys = list(self._env.scene.rigid_objects.keys())
            multi_obj_keys = [k for k in rigid_obj_keys if k.startswith("object_")]
            if len(multi_obj_keys) > 0:
                self._multi_object_mode = True
                # Build object names list: (safe_name, original_name)
                # safe_name has underscores, original_name has hyphens (for motion lookup)
                for key in multi_obj_keys:
                    safe_name = key[7:]  # Remove "object_" prefix
                    original_name = safe_name.replace(  # noqa: F841
                        "_", "-"
                    )  # Convert back for motion lookup  # noqa: F841
                    # But we need to be careful - only convert the ones that were originally hyphens
                    # Actually we don't know which underscores were hyphens, so let's store just safe_name
                    # and try both when looking up motions
                    self._object_names.append(safe_name)
                print(  # noqa: T201
                    f"[Multi-Object Mode] Detected {len(self._object_names)} objects in scene"
                )  # noqa: T201

        # Cache for table metadata loaded from pkl files (loaded once per motion key)
        self._table_meta_cache = {}  # motion_key -> {'table_pos': tensor, 'table_quat': tensor}

        self._object_roles_path = getattr(self.cfg, "object_roles_path", None)
        self._object_roles = self._load_object_roles(self._object_roles_path)
        self._refresh_object_metadata_cache()
        self._object_audit_path = getattr(self.cfg, "audit_path", None)
        self._object_audit_max_resets = int(getattr(self.cfg, "audit_max_resets", 0) or 0)
        self._object_audit_reset_calls = 0
        self._object_audit_records = []
        self._object_audit_created_at = time.time()
        if self._object_audit_path:
            os.makedirs(os.path.dirname(os.path.abspath(self._object_audit_path)), exist_ok=True)
            self._write_object_audit()

        if self.cfg.use_height_map:
            import omni  # noqa: PLC0415
            from gear_sonic.utils.raycaster_vendored import MultiMeshRaycaster  # noqa: PLC0415

            prim_path_patterns = [
                # r"/World/ground",
                r"/World/envs/env_\d+/Object",
            ]
            stage = omni.usd.get_context().get_stage()

            self.height_map = MultiMeshRaycaster.from_prim_paths(
                paths=prim_path_patterns,
                stage=stage,
                device=str(self.device),
            )
            mesh_filters = [[f"/World/envs/env_{i}/Object"] for i in range(self.num_envs)]
            self.num_mesh_per_cam, self.mesh_ids, self.cam_ids = self.height_map.get_mesh_ids(
                mesh_filters, device=self.device
            )

            lin = torch.linspace(
                -0.5 * self.cfg.height_map_size,
                0.5 * self.cfg.height_map_size,
                int(self.cfg.height_map_size / self.cfg.height_map_resolution) + 1,
                device=self.device,
            )
            grid_x, grid_y = torch.meshgrid(lin, lin)
            zeros = torch.zeros_like(grid_x)
            scan_offsets = torch.stack(
                [grid_x, grid_y, zeros],
                dim=-1,
            ).view(-1, 3)
            scan_offsets[:, 2] = -1.0
            ray_dirs_local = torch.nn.functional.normalize(scan_offsets, dim=-1)
            self.ray_dirs_local = ray_dirs_local.expand(self.num_envs, -1, -1)
            self.num_rays = self.ray_dirs_local.shape[1]
            self.num_rays_x, self.num_rays_y = grid_x.shape
            self.scan_dot_pos_w = torch.zeros(
                self.num_envs, self.num_rays_x, self.num_rays_y, 3, device=self.device
            )

    def _apply_joint_roster_mechanism_resistance(self) -> None:
        """Apply and verify source-independent resistance for each asset slot."""
        articulations = getattr(self._env.scene, "articulations", {})
        if "active_articulation" not in articulations:
            raise ValueError(
                "Joint-roster mode requires the normalized active articulation"
            )
        articulation = articulations["active_articulation"]
        if articulation.num_joints <= 0:
            raise ValueError(
                "Normalized active articulation must expose its mechanism joint"
            )
        slots = list((self.asset_compatible_roster or {}).get("slots") or [])
        resistance = _mechanism_resistance_tensors(
            slots,
            num_envs=self.num_envs,
            num_joints=articulation.num_joints,
            device=self.device,
            dtype=articulation.data.joint_damping.dtype,
        )
        env_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        articulation.write_joint_friction_coefficient_to_sim(
            resistance["static_friction"],
            joint_dynamic_friction_coeff=resistance["dynamic_friction"],
            joint_viscous_friction_coeff=resistance["viscous_friction"],
            env_ids=env_ids,
        )
        articulation.write_joint_damping_to_sim(
            resistance["damping"],
            env_ids=env_ids,
        )
        observed = {
            "static_friction": articulation.data.joint_friction_coeff,
            "dynamic_friction": articulation.data.joint_dynamic_friction_coeff,
            "viscous_friction": articulation.data.joint_viscous_friction_coeff,
            "damping": articulation.data.joint_damping,
        }
        mismatches = [
            name
            for name, expected in resistance.items()
            if not isinstance(observed.get(name), torch.Tensor)
            or observed[name].shape != expected.shape
            or not torch.allclose(
                observed[name],
                expected,
                rtol=0.0,
                atol=1.0e-7,
            )
        ]
        if mismatches:
            raise RuntimeError(
                "Simulator mechanism-resistance readback differs from the "
                f"joint roster for {mismatches}"
            )
        self._joint_roster_mechanism_resistance = resistance
        unique_rows = torch.unique(
            torch.stack(
                [
                    resistance["static_friction"][:, 0],
                    resistance["dynamic_friction"][:, 0],
                    resistance["viscous_friction"][:, 0],
                    resistance["damping"][:, 0],
                ],
                dim=-1,
            ),
            dim=0,
        )
        print(  # noqa: T201
            "[JointRoster] Applied per-asset mechanism resistance "
            f"to {self.num_envs} environments: {unique_rows.cpu().tolist()}"
        )

    def _apply_joint_roster_physics_overrides(self) -> None:
        """Apply per-slot object physics overrides to the live active articulation.

        Joint-roster scenes spawn the manipuland as ``ActiveArticulation`` (frame
        helper + tracked panel link), which the legacy rigid-object
        ``_apply_target_object_mass_overrides`` skips. Each slot's override dict
        comes from ``manager_env.config.object_physics_overrides_path`` matched
        on the slot's object key (resolved in the scene config). The torch
        helper applies mass (inertia scaled by the same ratio) and optional
        tracked-link material fields, then verifies the PhysX readback. No
        overrides path -> every slot is {} -> no-op.
        """
        per_env = getattr(self._env.cfg, "joint_roster_physics_overrides", None)
        if not per_env or not any(per_env):
            return
        from gear_sonic.rp2.joint_roster_physics import apply_joint_roster_physics_overrides

        articulations = getattr(self._env.scene, "articulations", {})
        if "active_articulation" not in articulations:
            raise ValueError("Joint-roster physics overrides require the active articulation")
        articulation = articulations["active_articulation"]
        tracked = str(
            getattr(self._env.cfg, "config", {}).get(
                "active_articulation_tracked_body_name", "panel"
            )
            or "panel"
        )
        report = apply_joint_roster_physics_overrides(
            articulation.root_physx_view,
            body_names=list(articulation.body_names),
            tracked_body=tracked,
            per_env=list(per_env),
            num_envs=self.num_envs,
        )
        if report is None:
            return
        runtime_mass = report.pop("runtime_tracked_mass", None)
        if runtime_mass is not None:
            # Keep the readback separately from the metric tensor: CommandTerm.reset
            # zeroes every metric for the reset envs, so the metric is refilled
            # from this copy at each resample (see _resample_command).
            self._object_runtime_mass_kg = runtime_mass.to(self.device, torch.float32)
            self.metrics["rp2_object/runtime_mass_kg"] = self._object_runtime_mass_kg.clone()
        self._joint_roster_physics_overrides_report = report
        print(  # noqa: T201
            f"[JointRoster] Applied per-slot physics overrides: {report}"
        )

    def _refresh_joint_roster_runtime_mass_metric(self, env_ids: torch.Tensor) -> None:
        """Refill the roster-mode runtime-mass metric after the per-reset metric zeroing."""
        if getattr(self, "_joint_roster_physics_overrides_report", None) is None:
            return
        metric = self.metrics.get("rp2_object/runtime_mass_kg")
        if metric is None or env_ids.numel() == 0:
            return
        metric[env_ids] = self._object_runtime_mass_kg[env_ids]

    def _load_object_roles(self, path: str | None) -> dict:
        """Load optional object-role metadata for audit and future reward routing."""
        default_roles = {
            "default": {
                "role": "context",
                "dynamic": False,
                "targetable": False,
                "reward_target": False,
            },
            "rules": [],
        }
        if not path:
            return default_roles
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            print(f"[ObjectAudit] Role metadata missing: {path}")  # noqa: T201
            return default_roles
        with open(path) as f:
            data = json.load(f)
        data.setdefault("default", default_roles["default"])
        data.setdefault("rules", [])
        print(f"[ObjectAudit] Loaded object roles from {path}")  # noqa: T201
        return data

    def _object_role_for_stem(self, stem: str) -> dict:
        role = dict(self._object_roles.get("default", {}))
        object_roles = self._object_roles.get("objects", {})
        exact = object_roles.get(stem)
        if exact:
            role.update(exact)
        # Per-motion paired USD stems use ``clip_key__object_name``. The
        # clip-level entry owns target-specific semantics such as body support
        # versus hand manipulation, while the object-level entry supplies the
        # reusable asset defaults.
        if "__" in stem:
            clip_key = stem.rsplit("__", 1)[0]
            clip_role = object_roles.get(clip_key)
            if clip_role:
                role.update(clip_role)
        for rule in self._object_roles.get("rules", []):
            contains = rule.get("contains")
            pattern = rule.get("match")
            matched = bool(contains and contains in stem)
            if pattern:
                matched = matched or re.search(pattern, stem) is not None
            if not matched:
                continue
            for key, value in rule.items():
                if key not in {"contains", "match"}:
                    role[key] = value
        return role

    def _object_stems_for_motion_key(
        self, motion_key: str, max_objects: int | None = None
    ) -> list[str]:
        stems_by_motion = getattr(self._env.cfg, "object_usd_stems_by_motion", None) or {}
        stems = (
            stems_by_motion.get(motion_key)
            or getattr(self._env.cfg, "object_usd_stems", None)
            or []
        )
        if not stems:
            # Normalized rigid and articulated entities are spawned through
            # ActiveArticulation rather than the legacy Object branch. Keep
            # their per-env USD stem paired with the motion so role/contact
            # gates resolve exactly as they do for rigid-object scenes.
            active_stems = list(
                getattr(self._env.cfg, "active_articulation_usd_stems", None)
                or []
            )
            motion_keys = list(getattr(self.motion_lib, "curr_motion_keys", []) or [])
            if len(active_stems) == len(motion_keys):
                stem_by_motion = getattr(
                    self, "_active_articulation_stem_by_motion", None
                )
                if stem_by_motion is None:
                    stem_by_motion = dict(zip(motion_keys, active_stems, strict=True))
                    self._active_articulation_stem_by_motion = stem_by_motion
                active_stem = stem_by_motion.get(motion_key)
                stems = [active_stem] if active_stem is not None else []
        stems = list(stems)
        if max_objects is not None:
            stems = stems[:max_objects]
        return stems

    def _target_object_stems_for_env_ids(self, env_ids: torch.Tensor) -> list[str | None]:
        """Resolve each environment's selected object stem in motion order."""
        if env_ids.numel() == 0:
            return []
        max_objects = int(getattr(self.motion_lib, "max_num_objects", 1) or 1)
        motion_ids = self.motion_ids[env_ids].detach().cpu().tolist()
        target_indices = self.target_object_indices()[env_ids].detach().cpu().tolist()
        stems_out: list[str | None] = []
        for motion_id, target_index in zip(motion_ids, target_indices, strict=False):
            motion_key = self.motion_lib.curr_motion_keys[int(motion_id)]
            stems = self._object_stems_for_motion_key(
                motion_key, max_objects=max_objects
            )
            index = int(target_index)
            stems_out.append(stems[index] if 0 <= index < len(stems) else None)
        return stems_out

    def _target_object_role_mask(self, role_names: Sequence[str] | str | None) -> torch.Tensor:
        """Return the per-env selected-target mask for the requested roles."""
        if not role_names:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if isinstance(role_names, str):
            role_names = [role_names]
        max_objects = int(getattr(self.motion_lib, "max_num_objects", 1) or 1)
        role_mask = torch.zeros(
            self.num_envs, max_objects, dtype=torch.bool, device=self.device
        )
        cache = getattr(self, "_object_role_mask_cache", {}) or {}
        for role_name in role_names:
            cached = cache.get(str(role_name))
            if cached is not None and cached.shape == role_mask.shape:
                role_mask |= cached.to(device=self.device, dtype=torch.bool)
        indices = self.target_object_indices().clamp(0, max_objects - 1)
        return role_mask.gather(1, indices[:, None]).squeeze(1)

    def _apply_target_object_mass_overrides(self, env_ids: torch.Tensor):
        """Apply per-motion rigid-object masses after motion assignment."""
        if not self.cfg.object_mass_from_physics_overrides:
            return
        if self._multi_object_mode or "object" not in self._env.scene.rigid_objects:
            return
        overrides = getattr(
            self._env.cfg, "object_physics_overrides_by_stem", {}
        ) or {}
        stems = self._target_object_stems_for_env_ids(env_ids)
        requested = []
        for stem in stems:
            physics = overrides.get(stem, {}) if stem is not None else {}
            mass = physics.get("mass")
            requested.append(float(mass) if mass is not None else None)
        if not any(mass is not None for mass in requested):
            return

        obj = self._env.scene["object"]
        masses = obj.root_physx_view.get_masses().clone()
        inertias = obj.root_physx_view.get_inertias().clone()
        env_ids_cpu = env_ids.detach().cpu().to(dtype=torch.int32)
        for env_id, target_mass in zip(
            env_ids_cpu.tolist(), requested, strict=False
        ):
            if target_mass is None:
                continue
            old_mass = masses[env_id].sum().clamp_min(1.0e-8)
            ratio = float(target_mass) / float(old_mass)
            masses[env_id] *= ratio
            inertias[env_id] *= ratio
        obj.root_physx_view.set_masses(masses, env_ids_cpu)
        obj.root_physx_view.set_inertias(inertias, env_ids_cpu)

        if not hasattr(self, "_object_runtime_mass_kg"):
            self._object_runtime_mass_kg = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        actual = masses[env_ids_cpu.long()].sum(dim=-1).to(self.device)
        self._object_runtime_mass_kg[env_ids] = actual
        self.metrics["rp2_object/runtime_mass_kg"] = self._object_runtime_mass_kg

    def _refresh_object_metadata_cache(self, env_ids: Sequence[int] | None = None):
        """Cache per-env object validity, role priority, and BPS tensors on device.

        RP2 full-scene metadata is motion-specific but only changes when an env is
        assigned a new motion. Keep the small Python/JSON work on reset instead of
        repeating it in observations, rewards, and contact routing every step.
        """
        max_objects = int(getattr(self.motion_lib, "max_num_objects", 1) or 1)
        bps_dim = int(getattr(self.motion_lib, "_bps_dim", 0) or 0)
        needs_alloc = (
            not hasattr(self, "_object_valid_mask_cache")
            or self._object_valid_mask_cache.shape != (self.num_envs, max_objects)
            or self._object_valid_mask_cache.device != torch.device(self.device)
        )
        if needs_alloc:
            self._object_valid_mask_cache = torch.ones(
                self.num_envs, max_objects, dtype=torch.bool, device=self.device
            )
            self._object_role_priority_cache = torch.full(
                (self.num_envs, max_objects), 3, dtype=torch.long, device=self.device
            )
            if bps_dim > 0:
                self._object_bps_all_cache = torch.zeros(
                    self.num_envs, max_objects, bps_dim, dtype=torch.float32, device=self.device
                )
            else:
                self._object_bps_all_cache = None
            self._object_role_mask_cache = {}

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        self._object_valid_mask_cache[env_ids_t] = False
        self._object_role_priority_cache[env_ids_t] = 4
        if getattr(self, "_object_bps_all_cache", None) is not None:
            self._object_bps_all_cache[env_ids_t] = 0.0
        role_mask_cache = getattr(self, "_object_role_mask_cache", {})
        for role_mask in role_mask_cache.values():
            role_mask[env_ids_t] = False

        bps_lookup = getattr(self.motion_lib, "_bps_lookup", None) or {}
        motion_ids_cpu = self.motion_ids[env_ids_t].detach().cpu().tolist()
        env_ids_cpu = env_ids_t.detach().cpu().tolist()
        common_roles = {
            "manipuland",
            "moving_support",
            "articulated_part",
            "support",
            "articulated_context",
            "context",
            "contact_semantics:hand_manipulation",
            "contact_semantics:body_support",
        }
        for role_name in common_roles:
            role_mask_cache.setdefault(
                role_name,
                torch.zeros(
                    self.num_envs, max_objects, dtype=torch.bool, device=self.device
                ),
            )
        for env_id, motion_id in zip(env_ids_cpu, motion_ids_cpu, strict=False):
            motion_key = self.motion_lib.curr_motion_keys[int(motion_id)]
            stems = self._object_stems_for_motion_key(motion_key, max_objects=max_objects)
            count = min(max_objects, len(stems) if stems else max_objects)
            if count <= 0:
                count = max_objects
            self._object_valid_mask_cache[env_id, :count] = True
            for obj_idx, stem in enumerate(stems[:max_objects]):
                role = self._object_role_for_stem(stem)
                if bool(role.get("reward_target", False)):
                    priority = 0
                elif role.get("role") == "manipuland":
                    priority = 1
                elif bool(role.get("targetable", False)):
                    priority = 2
                else:
                    priority = 3
                self._object_role_priority_cache[env_id, obj_idx] = priority
                role_name = str(role.get("role", "unknown"))
                if role_name and role_name != "unknown":
                    role_mask_cache.setdefault(
                        role_name,
                        torch.zeros(
                            self.num_envs, max_objects, dtype=torch.bool, device=self.device
                        ),
                    )[env_id, obj_idx] = True
                contact_semantics = str(role.get("contact_semantics", ""))
                semantic_key = f"contact_semantics:{contact_semantics}"
                if contact_semantics:
                    role_mask_cache.setdefault(
                        semantic_key,
                        torch.zeros(
                            self.num_envs,
                            max_objects,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    )[env_id, obj_idx] = True
                if getattr(self, "_object_bps_all_cache", None) is not None:
                    vec = bps_lookup.get(stem)
                    if vec is not None:
                        self._object_bps_all_cache[env_id, obj_idx] = torch.as_tensor(
                            vec, dtype=torch.float32, device=self.device
                        )
        self._object_role_mask_cache = role_mask_cache

    def _target_object_index_for_motion_key(
        self,
        motion_key: str,
        max_objects: int | None = None,
        candidate_indices: Sequence[int] | None = None,
    ) -> int:
        stems = self._object_stems_for_motion_key(motion_key, max_objects=max_objects)
        if not stems and not candidate_indices:
            return 0

        if candidate_indices:
            candidates = [
                int(idx)
                for idx in candidate_indices
                if int(idx) >= 0 and (max_objects is None or int(idx) < max_objects)
            ]
        else:
            candidates = list(range(len(stems)))
        if not candidates:
            return 0

        priority_checks = (
            lambda role: bool(role.get("reward_target", False)),
            lambda role: role.get("role") == "manipuland",
            lambda role: bool(role.get("targetable", False)),
        )
        roles = [self._object_role_for_stem(stem) for stem in stems]
        for check in priority_checks:
            for idx in candidates:
                if idx < len(roles) and check(roles[idx]):
                    return idx
        return candidates[0]

    def target_object_indices(self) -> torch.Tensor:
        """Return the per-env reference object index used by HOI obs/rewards.

        Legacy datasets fall back to index 0. Full-scene RP2 datasets may provide
        dynamic per-frame object masks. Those masks pick the active object set
        first; role metadata only breaks ties for legacy single-target terms.
        """
        max_objects = int(getattr(self.motion_lib, "max_num_objects", 1) or 1)
        if max_objects <= 1:
            cached = getattr(self, "_single_target_object_indices", None)
            if (
                cached is None
                or cached.shape[0] != self.num_envs
                or cached.device != self.device
            ):
                cached = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
                self._single_target_object_indices = cached
            return cached
        valid = getattr(self, "_object_valid_mask_cache", None)
        priority = getattr(self, "_object_role_priority_cache", None)
        if (
            valid is not None
            and priority is not None
            and valid.shape == (self.num_envs, max_objects)
            and priority.shape == (self.num_envs, max_objects)
        ):
            candidate = valid
            if bool(getattr(self.cfg, "use_dynamic_target_object_mask", True)):
                active_mask = self.object_interaction_mask
                if active_mask is not None:
                    active_mask = active_mask.to(self.device)
                    if active_mask.ndim == 1:
                        active_mask = active_mask.view(-1, 1)
                    else:
                        active_mask = active_mask.reshape(active_mask.shape[0], -1)
                    if active_mask.shape[1] < max_objects:
                        padded = torch.zeros(
                            self.num_envs,
                            max_objects,
                            dtype=active_mask.dtype,
                            device=self.device,
                        )
                        padded[:, : active_mask.shape[1]] = active_mask
                        active_mask = padded
                    candidate = (active_mask[:, :max_objects] > 0.5) & valid
                    no_active = ~candidate.any(dim=-1)
                    candidate = torch.where(no_active[:, None], valid, candidate)
            scores = priority.masked_fill(~candidate, 1000)
            indices = scores.argmin(dim=-1)
            has_candidate = candidate.any(dim=-1)
            return torch.where(
                has_candidate,
                indices,
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
            )
        motion_ids = self.motion_ids.detach().cpu().tolist()
        active_mask = None
        if bool(getattr(self.cfg, "use_dynamic_target_object_mask", True)):
            active_mask = self.object_interaction_mask
            if active_mask is not None:
                active_mask = active_mask.detach().cpu()
                if active_mask.ndim == 1:
                    active_mask = active_mask.view(-1, 1)
                else:
                    active_mask = active_mask.reshape(active_mask.shape[0], -1)
                if active_mask.shape[1] < max_objects:
                    padded = torch.zeros(
                        active_mask.shape[0], max_objects, dtype=active_mask.dtype
                    )
                    padded[:, : active_mask.shape[1]] = active_mask
                    active_mask = padded
        indices = []
        for env_idx, motion_id in enumerate(motion_ids):
            motion_key = self.motion_lib.curr_motion_keys[int(motion_id)]
            candidates = None
            if active_mask is not None and env_idx < active_mask.shape[0]:
                candidates = torch.nonzero(active_mask[env_idx] > 0.5, as_tuple=False).flatten()
                candidates = candidates.tolist()
                if not candidates:
                    candidates = None
            idx = self._target_object_index_for_motion_key(
                motion_key, max_objects=max_objects, candidate_indices=candidates
            )
            indices.append(max(0, min(idx, max_objects - 1)))
        return torch.tensor(indices, device=self.device, dtype=torch.long)

    def _get_object_mask(self, mask_name: str) -> torch.Tensor | None:
        getter = getattr(self.motion_lib, f"get_{mask_name}", None)
        if getter is None:
            return None
        return getter(self.motion_ids, self.motion_start_time_steps + self.time_steps)

    @property
    def object_interaction_mask(self) -> torch.Tensor | None:
        return self._get_object_mask("object_interaction_mask")

    @property
    def object_motion_mask(self) -> torch.Tensor | None:
        return self._get_object_mask("object_motion_mask")

    def _object_role_names_by_env(self, max_objects: int) -> list[list[str]]:
        motion_ids_cpu = self.motion_ids.detach().cpu().tolist()
        roles_by_env = []
        for motion_id in motion_ids_cpu:
            motion_key = self.motion_lib.curr_motion_keys[int(motion_id)]
            stems = self._object_stems_for_motion_key(motion_key, max_objects=max_objects)
            role_names = [
                str(self._object_role_for_stem(stem).get("role", "unknown"))
                for stem in stems[:max_objects]
            ]
            if len(role_names) < max_objects:
                role_names.extend(["padding"] * (max_objects - len(role_names)))
            roles_by_env.append(role_names)
        return roles_by_env

    def _multi_object_contact_force_by_sensor_key(
        self, sensor_key: str, max_objects: int
    ) -> torch.Tensor | None:
        contact_map = getattr(self._env.cfg, "multi_object_contact_sensor_map", {}) or {}
        if not contact_map:
            return None
        scene = getattr(self._env, "scene", None)
        sensors = getattr(scene, "sensors", {}) if scene is not None else {}
        if not isinstance(sensors, dict):
            return None

        stems_by_motion = getattr(self._env.cfg, "object_usd_stems_by_motion", None) or {}
        global_stems = getattr(self._env.cfg, "object_usd_stems", None) or []
        slot_count = int(getattr(self._env.cfg, "object_usd_slot_count", 0) or 0)
        motion_ids_cpu = self.motion_ids.detach().cpu().tolist()
        force_by_object = torch.zeros(
            self.num_envs, max_objects, dtype=torch.float32, device=self.device
        )
        found_any = False

        for env_idx, motion_id in enumerate(motion_ids_cpu):
            motion_key = self.motion_lib.curr_motion_keys[int(motion_id)]
            stems = stems_by_motion.get(motion_key) or global_stems
            for obj_idx, stem in enumerate(stems[:max_objects]):
                if slot_count > 0:
                    entry = contact_map.get(str(obj_idx))
                else:
                    entry = contact_map.get(str(stem).replace("-", "_"))
                if not entry:
                    continue
                sensor_name = entry.get(sensor_key)
                if not sensor_name or sensor_name not in sensors:
                    continue
                data = getattr(sensors[sensor_name], "data", None)
                if data is None or not hasattr(data, "force_matrix_w"):
                    continue
                force = data.force_matrix_w
                if force.shape[0] <= env_idx:
                    continue
                force_env = force[env_idx]
                force_by_object[env_idx, obj_idx] = torch.norm(force_env, dim=-1).sum()
                found_any = True
        return force_by_object if found_any else None

    def _update_rp2_body_support_metrics(self):
        body_error = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1)
        self.metrics["rp2_object_body_pos_error/mean"] = body_error.mean(dim=-1)
        self.metrics["rp2_object_body_pos_error/max"] = body_error.max(dim=-1).values

        ee_body_names = getattr(
            self.cfg,
            "rp2_ee_body_height_metric_names",
            [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        )
        ee_indices = _get_body_indexes(self, list(ee_body_names))
        if ee_indices:
            height_error = (
                self.body_pos_relative_w[:, ee_indices, 2]
                - self.robot_body_pos_w[:, ee_indices, 2]
            ).abs()
            threshold = float(getattr(self.cfg, "rp2_ee_body_height_metric_threshold", 0.25))
            self.metrics["rp2_object_ee_body_height_error/mean"] = height_error.mean(dim=-1)
            self.metrics["rp2_object_ee_body_height_error/max"] = height_error.max(dim=-1).values
            self.metrics["rp2_object_ee_body_height_deviation_count/all"] = (
                height_error > threshold
            ).float().sum(dim=-1)

        support_body_names = getattr(
            self.cfg,
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
        support_indices = _get_body_indexes(self, list(support_body_names))
        if support_indices:
            support_error = body_error[:, support_indices]
            self.metrics["rp2_object_support_body_pos_error/mean"] = support_error.mean(dim=-1)
            self.metrics["rp2_object_support_body_pos_error/max"] = (
                support_error.max(dim=-1).values
            )

    def _update_rp2_object_role_metrics(self):
        """Log per-role active counts and object position error without affecting reward."""
        if not bool(getattr(self.cfg, "enable_rp2_object_role_metrics", True)):
            return
        if not hasattr(self.motion_lib, "max_num_objects"):
            return
        max_objects = int(getattr(self.motion_lib, "max_num_objects", 1) or 1)
        if max_objects <= 1 or not hasattr(self, "object_root_pos"):
            return

        current_pos, current_quat, _, _, valid = mdp_utils.get_all_object_current_state(
            self._env, self, max_objects=max_objects
        )
        target_pos = self.object_root_pos[:, :max_objects]
        target_quat = self.object_root_quat[:, :max_objects]
        pos_error = torch.norm(target_pos - current_pos, dim=-1)
        ori_error = quat_error_magnitude(
            target_quat.reshape(-1, 4), current_quat.reshape(-1, 4)
        ).reshape(self.num_envs, max_objects)
        height_error = torch.abs(target_pos[..., 2] - current_pos[..., 2])
        up_axis = torch.zeros(self.num_envs, max_objects, 3, device=self.device)
        up_axis[..., 2] = 1.0
        current_up = quat_apply(current_quat.reshape(-1, 4), up_axis.reshape(-1, 3)).reshape(
            self.num_envs, max_objects, 3
        )
        tilt_deg = torch.rad2deg(torch.acos(torch.clamp(current_up[..., 2], -1.0, 1.0)))
        support_contact_force = self._multi_object_contact_force_by_sensor_key(
            "support_body_sensor", max_objects
        )

        mask = self.object_interaction_mask
        if mask is None:
            active = valid
        else:
            active = (mask[:, :max_objects] > 0.5) & valid

        active_f = active.float()
        active_denom = active_f.sum(dim=-1).clamp_min(1.0)
        self.metrics["rp2_object_active_count/all"] = active_f.sum(dim=-1)
        self.metrics["rp2_object_pos_error/active_all"] = (
            pos_error * active_f
        ).sum(dim=-1) / active_denom
        active_pos_error = torch.where(active, pos_error, torch.zeros_like(pos_error))
        self.metrics["rp2_object_pos_error_max/active_all"] = active_pos_error.max(dim=-1).values
        self.metrics["rp2_object_valid_count/all"] = valid.float().sum(dim=-1)
        pos_reward_coeff = float(getattr(self.cfg, "rp2_role_metric_pos_exp_coeff", -100.0))
        pos_reward = torch.exp(pos_reward_coeff * pos_error)
        self.metrics["rp2_object_pos_reward/active_all"] = (
            pos_reward * active_f
        ).sum(dim=-1) / active_denom
        deviation_threshold = float(
            getattr(self.cfg, "rp2_role_metric_pos_deviation_threshold", 0.1)
        )
        deviated = (pos_error > deviation_threshold) & active
        self.metrics["rp2_object_deviation_count/all"] = deviated.float().sum(dim=-1)
        active_height_error = torch.where(active, height_error, torch.zeros_like(height_error))
        active_ori_error = torch.where(active, ori_error, torch.zeros_like(ori_error))
        active_tilt_deg = torch.where(active, tilt_deg, torch.zeros_like(tilt_deg))
        self.metrics["rp2_object_height_error/active_all"] = (
            height_error * active_f
        ).sum(dim=-1) / active_denom
        self.metrics["rp2_object_ori_error/active_all"] = (
            ori_error * active_f
        ).sum(dim=-1) / active_denom
        self.metrics["rp2_object_tilt_deg_max/active_all"] = active_tilt_deg.max(dim=-1).values
        self.metrics["rp2_object_height_error_max/active_all"] = (
            active_height_error.max(dim=-1).values
        )
        self.metrics["rp2_object_ori_error_max/active_all"] = (
            active_ori_error.max(dim=-1).values
        )
        if support_contact_force is not None:
            self.metrics["rp2_object_support_contact_force/active_all"] = (
                support_contact_force * active_f
            ).sum(dim=-1) / active_denom

        roles_by_env = self._object_role_names_by_env(max_objects)
        role_names = {
            role
            for roles in roles_by_env
            for role in roles
            if role not in {"padding", "unknown"}
        }
        role_names.update(
            {
                "manipuland",
                "moving_support",
                "articulated_part",
                "support",
                "articulated_context",
            }
        )

        for role_name in sorted(role_names):
            role_mask = torch.zeros(
                self.num_envs, max_objects, dtype=torch.bool, device=self.device
            )
            for env_idx, roles in enumerate(roles_by_env):
                for obj_idx, role in enumerate(roles[:max_objects]):
                    if role == role_name:
                        role_mask[env_idx, obj_idx] = True
            role_active = active & role_mask
            role_valid = valid & role_mask
            role_active_f = role_active.float()
            role_valid_f = role_valid.float()
            denom = role_active_f.sum(dim=-1).clamp_min(1.0)
            valid_denom = role_valid_f.sum(dim=-1).clamp_min(1.0)
            self.metrics[f"rp2_object_active_count/{role_name}"] = role_active_f.sum(dim=-1)
            self.metrics[f"rp2_object_pos_error/{role_name}"] = (
                pos_error * role_active_f
            ).sum(dim=-1) / denom
            role_pos_error = torch.where(role_active, pos_error, torch.zeros_like(pos_error))
            self.metrics[f"rp2_object_pos_error_max/{role_name}"] = (
                role_pos_error.max(dim=-1).values
            )
            self.metrics[f"rp2_object_pos_reward/{role_name}"] = (
                pos_reward * role_active_f
            ).sum(dim=-1) / denom
            self.metrics[f"rp2_object_deviation_count/{role_name}"] = (
                (deviated & role_mask).float().sum(dim=-1)
            )
            self.metrics[f"rp2_object_height_error/{role_name}"] = (
                height_error * role_valid_f
            ).sum(dim=-1) / valid_denom
            self.metrics[f"rp2_object_height_error_max/{role_name}"] = (
                torch.where(role_valid, height_error, torch.zeros_like(height_error))
                .max(dim=-1)
                .values
            )
            self.metrics[f"rp2_object_ori_error/{role_name}"] = (
                ori_error * role_valid_f
            ).sum(dim=-1) / valid_denom
            self.metrics[f"rp2_object_ori_error_max/{role_name}"] = (
                torch.where(role_valid, ori_error, torch.zeros_like(ori_error))
                .max(dim=-1)
                .values
            )
            self.metrics[f"rp2_object_tilt_deg_max/{role_name}"] = (
                torch.where(role_valid, tilt_deg, torch.zeros_like(tilt_deg))
                .max(dim=-1)
                .values
            )
            if support_contact_force is not None:
                self.metrics[f"rp2_object_support_contact_force/{role_name}"] = (
                    support_contact_force * role_valid_f
                ).sum(dim=-1) / valid_denom
                self.metrics[f"rp2_object_support_contact_force_active/{role_name}"] = (
                    support_contact_force * role_active_f
                ).sum(dim=-1) / denom
        self._update_rp2_body_support_metrics()

    def object_contact_mask(self, hand: str = "right_hand") -> torch.Tensor | None:
        return self.motion_lib.get_object_contact_mask(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )

    @staticmethod
    def _gather_object_by_index(object_tensor: torch.Tensor, object_indices: torch.Tensor):
        if object_tensor.ndim == 3:
            env_ids = torch.arange(object_tensor.shape[0], device=object_tensor.device)
            return object_tensor[env_ids, object_indices]
        if object_tensor.ndim == 4:
            expand_shape = (
                object_tensor.shape[0],
                object_tensor.shape[1],
                1,
                object_tensor.shape[-1],
            )
            gather_idx = object_indices.view(-1, 1, 1, 1).expand(expand_shape)
            return torch.gather(object_tensor, dim=2, index=gather_idx).squeeze(2)
        raise ValueError(f"Expected object tensor with 3 or 4 dims, got {object_tensor.shape}")

    @staticmethod
    def _object_mask_for_audit(
        mask: torch.Tensor | None,
        env_ids: torch.Tensor,
        max_objects: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        selected = mask[env_ids].detach().cpu()
        if selected.ndim == 1:
            selected = selected.view(-1, 1)
        else:
            selected = selected.reshape(selected.shape[0], -1)
        if selected.shape[1] >= max_objects:
            return selected[:, :max_objects]
        padded = torch.zeros(selected.shape[0], max_objects, dtype=selected.dtype)
        padded[:, : selected.shape[1]] = selected
        return padded

    @property
    def target_object_root_pos(self) -> torch.Tensor:
        if self.object_root_pos.ndim == 3 and self.object_root_pos.shape[1] == 1:
            return self.object_root_pos[:, 0]
        return self._gather_object_by_index(self.object_root_pos, self.target_object_indices())

    @property
    def target_object_root_quat(self) -> torch.Tensor:
        if self.object_root_quat.ndim == 3 and self.object_root_quat.shape[1] == 1:
            return self.object_root_quat[:, 0]
        return self._gather_object_by_index(self.object_root_quat, self.target_object_indices())

    @property
    def target_object_root_pos_multi_future(self) -> torch.Tensor:
        if (
            self.object_root_pos_multi_future.ndim == 4
            and self.object_root_pos_multi_future.shape[2] == 1
        ):
            return self.object_root_pos_multi_future[:, :, 0]
        return self._gather_object_by_index(
            self.object_root_pos_multi_future, self.target_object_indices()
        )

    @property
    def target_object_root_quat_multi_future(self) -> torch.Tensor:
        if (
            self.object_root_quat_multi_future.ndim == 4
            and self.object_root_quat_multi_future.shape[2] == 1
        ):
            return self.object_root_quat_multi_future[:, :, 0]
        return self._gather_object_by_index(
            self.object_root_quat_multi_future, self.target_object_indices()
        )

    @property
    def target_object_lin_vel(self) -> torch.Tensor:
        object_lin_vel = self.motion_lib.get_object_lin_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        if object_lin_vel.ndim == 3 and object_lin_vel.shape[1] == 1:
            return object_lin_vel[:, 0]
        return self._gather_object_by_index(object_lin_vel, self.target_object_indices())

    @property
    def target_object_ang_vel(self) -> torch.Tensor:
        object_ang_vel = self.motion_lib.get_object_ang_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        if object_ang_vel.ndim == 3 and object_ang_vel.shape[1] == 1:
            return object_ang_vel[:, 0]
        return self._gather_object_by_index(object_ang_vel, self.target_object_indices())

    @property
    def target_object_bps(self) -> torch.Tensor:
        max_objects = int(getattr(self.motion_lib, "max_num_objects", 1) or 1)
        if max_objects <= 1:
            return self.object_bps

        bps_lookup = getattr(self.motion_lib, "_bps_lookup", None)
        bps_dim = int(getattr(self.motion_lib, "_bps_dim", 0) or 0)
        if not bps_lookup or bps_dim <= 0:
            return self.motion_lib.get_object_bps(self.motion_ids)

        rows = []
        motion_ids = self.motion_ids.detach().cpu().tolist()
        indices = self.target_object_indices().detach().cpu().tolist()
        for motion_id, object_idx in zip(motion_ids, indices, strict=False):
            motion_key = self.motion_lib.curr_motion_keys[int(motion_id)]
            stems = self._object_stems_for_motion_key(
                motion_key, max_objects=getattr(self.motion_lib, "max_num_objects", None)
            )
            stem = stems[object_idx] if object_idx < len(stems) else os.path.basename(motion_key)
            vec = bps_lookup.get(stem)
            if vec is None:
                vec = np.zeros(bps_dim, dtype=np.float32)
            rows.append(torch.as_tensor(vec, device=self.device, dtype=torch.float32))
        return torch.stack(rows, dim=0)

    def _next_object_audit_reset_index(self) -> int | None:
        if not self._object_audit_path:
            return None
        if (
            self._object_audit_max_resets > 0
            and self._object_audit_reset_calls >= self._object_audit_max_resets
        ):
            return None
        self._object_audit_reset_calls += 1
        return self._object_audit_reset_calls

    def _append_full_scene_reset_audit(
        self,
        reset_index: int,
        motion_id: int,
        motion_key: str,
        env_ids: torch.Tensor,
        object_scene_keys: list[str],
        all_object_keys: list[str],
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        reference_time_steps: torch.Tensor | None = None,
    ):
        stems_by_motion = getattr(self._env.cfg, "object_usd_stems_by_motion", None) or {}
        stems = stems_by_motion.get(motion_key)
        if not stems:
            all_stems = list(getattr(self._env.cfg, "object_usd_stems", []) or [])
            if (
                self.cfg.use_paired_motions
                and len(all_stems) == len(self.motion_lib.curr_motion_keys)
                and 0 <= int(motion_id) < len(all_stems)
            ):
                stems = [all_stems[int(motion_id)]]
            else:
                stems = all_stems
        stems = list(stems)[: len(object_scene_keys)]
        bps_dir = None
        if isinstance(self.cfg.motion_lib_cfg, dict):
            bps_dir = self.cfg.motion_lib_cfg.get("bps_dir")

        env_ids_cpu = env_ids.detach().cpu().tolist()
        start_steps = self.motion_start_time_steps[env_ids].detach().cpu().tolist()
        if reference_time_steps is None:
            reference_steps = start_steps
        else:
            reference_steps = reference_time_steps.detach().cpu().tolist()
        motion_steps = self.motion_num_steps[env_ids].detach().cpu().tolist()
        object_pos_cpu = object_pos.detach().cpu()
        object_quat_cpu = object_quat.detach().cpu()
        target_indices = self.target_object_indices()[env_ids].detach().cpu().tolist()
        interaction_mask_cpu = self._object_mask_for_audit(
            self.object_interaction_mask, env_ids, len(object_scene_keys)
        )
        motion_mask_cpu = self._object_mask_for_audit(
            self.object_motion_mask, env_ids, len(object_scene_keys)
        )
        inactive_keys = sorted(set(all_object_keys) - set(object_scene_keys))
        object_dynamics_by_stem = getattr(self._env.cfg, "object_dynamics_by_stem", {}) or {}
        object_physics_overrides_by_stem = (
            getattr(self._env.cfg, "object_physics_overrides_by_stem", {}) or {}
        )
        audit_motion_ids = self.motion_ids[env_ids]
        audit_steps = (
            self.motion_start_time_steps[env_ids]
            if reference_time_steps is None
            else reference_time_steps
        )

        def source_audit_channel(getter_name: str):
            getter = getattr(self.motion_lib, getter_name, None)
            if getter is None:
                return None
            value = getter(audit_motion_ids, audit_steps)
            return value.detach().cpu() if isinstance(value, torch.Tensor) else None

        source_relation = source_audit_channel("get_rp2_source_relation_vector")
        source_contact = source_audit_channel("get_rp2_source_contact_probability")
        source_root = source_audit_channel("get_rp2_source_root_pos")
        source_object_lin_vel = source_audit_channel("get_object_lin_vel")
        source_object_ang_vel = source_audit_channel("get_object_ang_vel")

        for local_idx, env_id in enumerate(env_ids_cpu):
            objects = []
            for obj_idx, scene_key in enumerate(object_scene_keys):
                stem = stems[obj_idx] if obj_idx < len(stems) else scene_key.removeprefix("object_")
                bps_path = os.path.join(bps_dir, f"{stem}.npy") if bps_dir else None
                objects.append(
                    {
                        "index": obj_idx,
                        "stem": stem,
                        "scene_key": scene_key,
                        "role": self._object_role_for_stem(stem),
                        "scene_dynamic": object_dynamics_by_stem.get(stem),
                        "physics_override": object_physics_overrides_by_stem.get(stem, {}),
                        "bps_path": bps_path,
                        "bps_exists": bool(bps_path and os.path.isfile(bps_path)),
                        "interaction_mask": (
                            float(interaction_mask_cpu[local_idx, obj_idx])
                            if interaction_mask_cpu is not None
                            else None
                        ),
                        "motion_mask": (
                            float(motion_mask_cpu[local_idx, obj_idx])
                            if motion_mask_cpu is not None
                            else None
                        ),
                        "selected_target": obj_idx == target_indices[local_idx],
                        "reference_position_w": object_pos_cpu[local_idx, obj_idx].tolist(),
                        "reference_quat_wxyz": object_quat_cpu[local_idx, obj_idx].tolist(),
                    }
                )

            source_reference = None
            if source_relation is not None:
                source_reference = {
                    "relation_vector": source_relation[local_idx].tolist(),
                    "relation_mean_norm_m": float(
                        torch.linalg.vector_norm(source_relation[local_idx], dim=-1)
                        .mean()
                        .item()
                    ),
                    "contact_probability": (
                        source_contact[local_idx].tolist()
                        if source_contact is not None
                        else None
                    ),
                    "source_root_position": (
                        source_root[local_idx].tolist() if source_root is not None else None
                    ),
                    "object_linear_velocity": (
                        source_object_lin_vel[local_idx].tolist()
                        if source_object_lin_vel is not None
                        else None
                    ),
                    "object_angular_velocity": (
                        source_object_ang_vel[local_idx].tolist()
                        if source_object_ang_vel is not None
                        else None
                    ),
                }

            self._object_audit_records.append(
                {
                    "reset_index": reset_index,
                    "env_id": int(env_id),
                    "motion_id": int(motion_id),
                    "motion_key": motion_key,
                    "motion_start_time_step": int(start_steps[local_idx]),
                    "reference_time_step": int(reference_steps[local_idx]),
                    "motion_num_steps": int(motion_steps[local_idx]),
                    "object_tensor_shape": list(object_pos.shape),
                    "target_object_index": int(target_indices[local_idx]),
                    "active_object_count": len(object_scene_keys),
                    "inactive_object_count": len(inactive_keys),
                    "inactive_scene_keys": inactive_keys,
                    "objects": objects,
                    "source_reference": source_reference,
                }
            )

    def _write_object_audit(self):
        if not self._object_audit_path:
            return
        rp2_reward_overrides = {
            "soma_object_relation_weight": os.environ.get("RP2_SOMA_OBJECT_RELATION_WEIGHT"),
            "soma_object_relation_mask_name": os.environ.get("RP2_SOMA_OBJECT_RELATION_MASK_NAME"),
            "soma_object_relation_role_names": os.environ.get(
                "RP2_SOMA_OBJECT_RELATION_ROLE_NAMES"
            ),
            "soma_object_relation_dist_exp_coeff": os.environ.get(
                "RP2_SOMA_OBJECT_RELATION_DIST_EXP_COEFF"
            ),
        }
        payload = {
            "schema_version": 1,
            "created_unix": self._object_audit_created_at,
            "updated_unix": time.time(),
            "num_envs": self.num_envs,
            "full_scene_per_env": bool(getattr(self.cfg, "full_scene_per_env", False)),
            "multi_object_mode": bool(self._multi_object_mode),
            "object_roles_path": self._object_roles_path,
            "object_usd_stems": getattr(self._env.cfg, "object_usd_stems", None),
            "object_usd_stems_by_motion": getattr(
                self._env.cfg, "object_usd_stems_by_motion", {}
            ),
            "object_dynamics_by_stem": getattr(self._env.cfg, "object_dynamics_by_stem", {}),
            "object_physics_overrides_by_stem": getattr(
                self._env.cfg, "object_physics_overrides_by_stem", {}
            ),
            "articulated_context_usd_paths": getattr(
                self._env.cfg, "articulated_context_usd_paths", []
            ),
            "rp2_reward_overrides": rp2_reward_overrides,
            "motion_keys": list(getattr(self.motion_lib, "curr_motion_keys", [])),
            "object_mask_summary": self._object_mask_audit_summary(),
            "contact_sensor_summary": self._contact_sensor_audit_summary(),
            "records": self._object_audit_records,
        }
        tmp_path = f"{self._object_audit_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self._object_audit_path)

    def _object_mask_audit_summary(self) -> dict:
        summary = {}
        mask_specs = {
            "interaction": getattr(self.motion_lib, "_motion_object_interaction_mask", None),
            "motion": getattr(self.motion_lib, "_motion_object_motion_mask", None),
            "contact_left": getattr(
                self.motion_lib, "_motion_object_in_contact_left", None
            ),
            "contact_right": getattr(
                self.motion_lib, "_motion_object_in_contact_right", None
            ),
        }
        length_starts = getattr(self.motion_lib, "length_starts", None)
        motion_frames = getattr(self.motion_lib, "_motion_num_frames", None)
        motion_keys = list(getattr(self.motion_lib, "curr_motion_keys", []))
        if length_starts is None or motion_frames is None:
            return summary

        for name, mask_tensor in mask_specs.items():
            if mask_tensor is None:
                summary[name] = {"available": False}
                continue
            mask_tensor_cpu = mask_tensor.detach().cpu()
            per_motion = {}
            for motion_idx, motion_key in enumerate(motion_keys):
                start = int(length_starts[motion_idx].detach().cpu().item())
                count = int(motion_frames[motion_idx].detach().cpu().item())
                mask = mask_tensor_cpu[start : start + count]
                if mask.ndim == 1:
                    mask = mask[:, None]
                if mask.ndim != 2:
                    continue
                per_motion[motion_key] = {
                    "shape": list(mask.shape),
                    "frames_with_any_active": int((mask > 0.5).any(dim=1).sum().item()),
                    "active_frames_per_object": [
                        int(v) for v in (mask > 0.5).sum(dim=0).tolist()
                    ],
                }
            summary[name] = {
                "available": True,
                "total_active_entries": int((mask_tensor_cpu > 0.5).sum().item()),
                "per_motion": per_motion,
            }
        return summary

    def _contact_sensor_audit_summary(self) -> dict:
        summary = {
            "legacy_sensors": {},
            "multi_object_sensor_count": 0,
            "multi_object_sensors": {},
            "missing_multi_object_sensors": [],
            "support_body_sensor_count": 0,
        }
        scene = getattr(self._env, "scene", None)
        sensors = getattr(scene, "sensors", {}) if scene is not None else {}
        sensor_names = set(sensors.keys()) if isinstance(sensors, dict) else set()

        def _sensor_record(sensor_name: str) -> dict:
            if sensor_name not in sensor_names:
                return {"available": False}
            sensor = sensors[sensor_name]
            cfg = getattr(sensor, "cfg", None)
            record = {
                "available": True,
                "prim_path": getattr(cfg, "prim_path", None),
                "filter_prim_paths_expr": list(getattr(cfg, "filter_prim_paths_expr", []) or []),
            }
            data = getattr(sensor, "data", None)
            if data is not None and hasattr(data, "force_matrix_w"):
                force = data.force_matrix_w
                record["force_matrix_shape"] = list(force.shape)
                force_mag = torch.norm(force, dim=-1)
                record["nonzero_force_entries"] = int((force_mag > 1e-6).sum().detach().cpu().item())
                record["force_magnitude_sum"] = float(force_mag.sum().detach().cpu().item())
            if data is not None and hasattr(data, "net_forces_w"):
                net = data.net_forces_w
                record["net_forces_shape"] = list(net.shape)
            return record

        for legacy_name in (
            "object_to_hand_contact_sensor",
            "object_to_left_hand_contact_sensor",
            "table_to_hand_contact_sensor",
            "object_to_table_contact_sensor",
        ):
            summary["legacy_sensors"][legacy_name] = _sensor_record(legacy_name)

        contact_map = getattr(self._env.cfg, "multi_object_contact_sensor_map", {}) or {}
        for stem, entry in contact_map.items():
            right_name = entry.get("right_hand_sensor")
            left_name = entry.get("left_hand_sensor")
            support_name = entry.get("support_body_sensor")
            sensor_record = {
                "object_index": entry.get("object_index"),
                "object_stem": entry.get("object_stem", stem),
                "prim_path": entry.get("prim_path"),
                "right_hand_sensor": right_name,
                "left_hand_sensor": left_name,
                "support_body_sensor": support_name,
                "right": _sensor_record(right_name) if right_name else {"available": False},
                "left": _sensor_record(left_name) if left_name else {"available": False},
                "support": _sensor_record(support_name) if support_name else {"available": False},
            }
            if support_name:
                summary["support_body_sensor_count"] += 1
            for sensor_name in (right_name, left_name, support_name):
                if sensor_name and sensor_name not in sensor_names:
                    summary["missing_multi_object_sensors"].append(sensor_name)
            summary["multi_object_sensors"][stem] = sensor_record
        summary["multi_object_sensor_count"] = len(contact_map)
        return summary

    # =========================================================================
    # Offline (kinematic-only) factory
    # =========================================================================

    @classmethod
    def create_offline(
        cls,
        motion_lib_cfg,
        device,
    ):
        """Create a MotionCommand without env/sim for offline kinematic use.

        All @property methods that only use motion_lib data work unchanged.
        Properties that require robot state (self.robot) will use the reference
        motion root as the anchor (equivalent to use_ref_motion_root_quat_w_as_anchor=True).

        Expected keys in motion_lib_cfg:
            num_future_frames: Number of future reference frames.
            dt_future_ref_frames: Time delta between future reference frames (seconds).
        """
        inst = object.__new__(cls)
        inst._offline = True  # noqa: SLF001
        inst._debug_vis_handle = None  # noqa: SLF001
        num_envs = 1

        # device / num_envs are read-only properties on ManagerTermBase that
        # delegate to self._env, so we provide a lightweight stand-in.
        class _MinimalEnv:
            pass

        _env = _MinimalEnv()
        _env.device = device
        _env.num_envs = num_envs
        inst._env = _env  # noqa: SLF001

        # Motion lib setup
        motion_lib_cfg = (
            easydict.EasyDict(motion_lib_cfg)
            if not isinstance(motion_lib_cfg, easydict.EasyDict)
            else motion_lib_cfg
        )

        num_future_frames = motion_lib_cfg.get("num_future_frames", 8)
        dt_future_ref_frames = motion_lib_cfg.get("dt_future_ref_frames", 0.1)

        # Inject body/DOF mapping into motion_lib_cfg so motion_lib handles
        # body reordering and xyzw→wxyz quaternion conversion at load time.

        isaaclab_to_mujoco_mapping = order_converter.G1Converter().get_isaaclab_to_mujoco_mapping()
        motion_lib_cfg.update(
            {
                "mujoco_to_isaaclab_body": isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_body"],
                "mujoco_to_isaaclab_dof": isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_dof"],
                "isaaclab_to_mujoco_body": isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_body"],
                "isaaclab_to_mujoco_dof": isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_dof"],
            }
        )
        # body_indexes_data: same logic as __init__ — use body_names to select
        # a subset when provided, otherwise default to all bodies.
        isaaclab_joints = isaaclab_to_mujoco_mapping["isaaclab_joints"]
        body_names = motion_lib_cfg.get("body_names", None)
        if "body_indexes_data" not in motion_lib_cfg:
            if body_names is not None:
                motion_lib_cfg.body_indexes_data = [
                    isaaclab_joints.index(name) for name in body_names
                ]
            else:
                num_bodies_full = len(isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_body"])
                motion_lib_cfg.body_indexes_data = list(range(num_bodies_full))

        inst.motion_lib = motion_lib_robot.MotionLibRobot(motion_lib_cfg, num_envs, device)
        max_num_motions = motion_lib_cfg.get("max_num_motions", None)
        inst.motion_lib.load_motions_for_training(max_num_seqs=max_num_motions)

        # Timing
        target_fps = motion_lib_cfg.get("target_fps", 50)
        inst.num_future_frames = num_future_frames
        inst.frame_skips = dt_future_ref_frames // (1.0 / target_fps)
        steps_exact = dt_future_ref_frames * target_fps
        if abs(steps_exact - round(steps_exact)) > 1e-9:
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"dt_future_ref_frames={dt_future_ref_frames} * target_fps={target_fps} "
                f"= {steps_exact} is not an integer; using frame_skips={inst.frame_skips}",
                stacklevel=2,
            )
        inst.future_time_steps_init = (
            (torch.arange(num_future_frames, device=device, dtype=torch.long) * inst.frame_skips)
            .view(1, -1)
            .repeat(num_envs, 1)
        )
        # Anchor body index (from the body_names list, matching __init__)
        anchor_body = motion_lib_cfg.get("anchor_body", None)
        if body_names is not None and anchor_body is not None:
            inst.motion_anchor_body_index = body_names.index(anchor_body)
        else:
            inst.motion_anchor_body_index = 0  # Default to root
        inst.num_bodies = len(motion_lib_cfg.body_indexes_data)

        # DOF mapping
        if isaaclab_to_mujoco_mapping is not None:
            inst.isaaclab_to_mujoco_dof = isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_dof"]
            inst.mujoco_to_isaaclab_dof = isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_dof"]
        inst.lower_joint_indices_mujoco = list(range(12))
        if hasattr(inst, "isaaclab_to_mujoco_dof"):
            inst.lower_joint_isaaclab_indices = [
                inst.isaaclab_to_mujoco_dof[i] for i in inst.lower_joint_indices_mujoco
            ]

        # Variable frame support (offline)
        inst.variable_frames_enabled = motion_lib_cfg.get("variable_frames_enabled", False)
        inst.per_env_num_frames, inst._frame_choices = _init_variable_frames(  # noqa: SLF001
            inst.variable_frames_enabled,
            motion_lib_cfg.get("variable_frames_min", 16),
            num_future_frames,
            motion_lib_cfg.get("variable_frames_step", 4),
            num_envs,
            device,
        )

        # Motion state (updated per-sample via set_motion_state)
        inst.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.motion_start_time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.future_motion_ids = inst.motion_ids.repeat_interleave(num_future_frames)
        inst.motion_num_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Offline mode: use ref motion root as robot anchor
        inst.use_ref_motion_root_quat_w_as_anchor = True
        inst.ref_motion_root_rotation_noise = None

        # SMPL future frames (may differ from regular future frames)
        smpl_num_future_frames = motion_lib_cfg.get("smpl_num_future_frames", None)
        smpl_dt_future_ref_frames = motion_lib_cfg.get("smpl_dt_future_ref_frames", None)
        inst.smpl_num_future_frames = (
            smpl_num_future_frames if smpl_num_future_frames is not None else num_future_frames
        )
        inst.smpl_dt_future_ref_frames = (
            smpl_dt_future_ref_frames
            if smpl_dt_future_ref_frames is not None
            else dt_future_ref_frames
        )
        inst.smpl_frame_skips = inst.smpl_dt_future_ref_frames // (1.0 / target_fps)
        smpl_steps_exact = inst.smpl_dt_future_ref_frames * target_fps
        if abs(smpl_steps_exact - round(smpl_steps_exact)) > 1e-9:
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"smpl_dt_future_ref_frames={inst.smpl_dt_future_ref_frames} * target_fps={target_fps} "
                f"= {smpl_steps_exact} is not an integer; using smpl_frame_skips={inst.smpl_frame_skips}",
                stacklevel=2,
            )
        inst.smpl_future_time_steps_init = (
            (
                torch.arange(inst.smpl_num_future_frames, device=device, dtype=torch.long)
                * inst.smpl_frame_skips
            )
            .view(1, -1)
            .repeat(num_envs, 1)
        )
        inst.smpl_future_motion_ids = inst.motion_ids.repeat_interleave(inst.smpl_num_future_frames)

        # Encoder sampling (not used offline, but set for compat)
        inst.encoder_sample_probs_dict = None
        inst.encoder_sample_probs = None
        inst.is_evaluating = False

        # Metrics dict (normally set by CommandTerm.__init__)
        inst.metrics = {}

        return inst

    def set_is_evaluating(self, is_evaluating: bool = True):
        """Toggle evaluation mode, which disables reset randomizations."""
        self.is_evaluating = is_evaluating

    def forward_motion_samples(self, env_ids: Sequence[int]):
        """Assign sequential motion IDs and reset time steps for given envs.

        Used during paired/evaluation mode to deterministically cycle through
        motions. Updates motion IDs, start times, motion lengths, and caches
        the initial body pose for relative-frame computation.

        Args:
            env_ids: Environment indices to reassign motions for.
        """
        self.motion_ids[env_ids] = (
            torch.arange(self.num_envs).to(self.device)
            % self.motion_lib._num_motions  # noqa: SLF001
        )[env_ids]
        sampled_times = self.motion_lib.sample_time_steps(
            self.motion_ids[env_ids], truncate_time=self.cfg.sample_start_truncate_time_s
        )
        if self.cfg.sample_from_n_initial_frames is not None:
            # Sample uniformly from first N frames
            n_frames = self.cfg.sample_from_n_initial_frames
            sampled_times = torch.randint(
                0, n_frames, (len(env_ids),), dtype=sampled_times.dtype, device=self.device
            )
        elif self.cfg.start_from_first_frame:
            sampled_times.zero_()
        sampled_times = self._enforce_valid_start_window(env_ids, sampled_times)
        sampled_times = self._constrain_to_ownership_phase(env_ids, sampled_times)
        sampled_times = self._sample_from_reset_start_cache(env_ids, sampled_times)
        self.motion_start_time_steps[env_ids] = sampled_times
        self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
            self.motion_ids[env_ids]
        )
        self._apply_ownership_phase_end(env_ids)
        self.time_steps[env_ids] = 0
        self.body_pos_relative_w[env_ids] = self.motion_lib.get_body_pos_w(
            self.motion_ids[env_ids], self.motion_start_time_steps[env_ids]
        )
        self.body_quat_relative_w[env_ids] = self.motion_lib.get_body_quat_w(
            self.motion_ids[env_ids], self.motion_start_time_steps[env_ids]
        )

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation  # noqa: TD002, TD003, TD004
        """Return current-frame joint positions and velocities concatenated.

        Returns:
            Tensor of shape ``(num_envs, 2 * num_dof)``.
        """
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def command_z(self) -> torch.Tensor:
        """Return reference root height (z) for the current frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        return self.root_z

    @property
    def command_z_multi_future(self) -> torch.Tensor:
        """Return reference root height (z) for the first future frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        return self.root_z_multi_future

    @property
    def command_vel(self) -> torch.Tensor:  # TODO Consider again if this is the best observation  # noqa: TD002, TD003, TD004
        """Return reference root velocity (2D linear + 1D angular) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        return torch.cat([self.root_lin_vel_b_2d, self.root_ang_vel_b_1d], dim=1)

    @property
    def command_max(self) -> torch.Tensor:
        """Return all body state (pos, quat, lin_vel, ang_vel) flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w], dim=-1
        ).view(self.num_envs, -1)

    @property
    def command_max(self) -> torch.Tensor:  # noqa: F811
        """Return all body state (pos, quat, lin_vel, ang_vel) flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w], dim=-1
        ).view(self.num_envs, -1)

    @property
    def command_max_diff_l(self) -> torch.Tensor:
        """Return body state differences in robot-local frame, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [
                self.body_pos_dif_l.view(self.num_envs, -1),
                self.body_quat_dif_l.view(self.num_envs, -1),
                self.body_lin_vel_l,
                self.body_ang_vel_l,
            ],
            dim=-1,
        ).view(self.num_envs, -1)

    @property
    def command_max_diff_l_multi_future(self) -> torch.Tensor:
        """Return multi-future body state diffs and local poses concatenated.

        Returns:
            Tensor of shape ``(num_envs, <variable>)`` depending on body count and future frames.
        """
        # TODO: this is not done.  # noqa: TD002, TD003
        return torch.cat(
            [
                self.body_pos_dif_l_multi_future.view(self.num_envs, -1),
                self.body_quat_dif_l_multi_future.view(self.num_envs, -1),
                self.body_lin_vel_dif_l_multi_future,
                self.body_ang_vel_dif_l_multi_future,
                self.bod_pos_local_multi_future,
                self.body_quat_local_multi_future,
                self.body_lin_vel_l_multi_future,
                self.body_ang_vel_l_multi_future,
            ],
            dim=-1,
        ).view(self.num_envs, -1)

    @property
    def command_max_multi_future(self) -> torch.Tensor:
        """Return all body state for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_bodies * 13)``.
        """
        return torch.cat(
            [
                self.body_pos_w_multi_future,
                self.body_quat_w_multi_future,
                self.body_lin_vel_w_multi_future,
                self.body_ang_vel_w_multi_future,
            ],
            dim=1,
        )

    @property
    def command_multi_future(self) -> torch.Tensor:
        """Return joint positions and velocities for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, 2 * num_future_frames * num_dof)``.
        """
        return torch.cat([self.joint_pos_multi_future, self.joint_vel_multi_future], dim=1)

    @property
    def command_multi_future_joint_pos(self) -> torch.Tensor:
        """Return joint positions for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.joint_pos_multi_future

    @property
    def command_multi_future_joint_body_pos(self) -> torch.Tensor:
        """Return joint positions and anchor-relative body positions for all future frames.

        Body positions are expressed relative to each frame's anchor position,
        rotated into the anchor's yaw-inverse frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        body_pos_multi_frame = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, self.num_bodies, 3)
        anchor_pos_multi_frame = (
            self.motion_lib.get_body_pos_w(self.future_motion_ids, self.future_time_steps)[
                :, self.motion_anchor_body_index
            ]
            .view(self.num_envs, self.num_future_frames, 1, 3)
            .expand(self.num_envs, self.num_future_frames, self.num_bodies, 3)
        )
        body_pos_relative_to_anchor_multi_frame = body_pos_multi_frame - anchor_pos_multi_frame
        anchor_quat_w_repeat = self.anchor_quat_w_multi_future.view(
            self.num_envs, self.num_future_frames, 1, 4
        ).expand(self.num_envs, self.num_future_frames, self.num_bodies, 4)
        body_pos_multi_frame = quat_apply_yaw(
            quat_inv(anchor_quat_w_repeat), body_pos_relative_to_anchor_multi_frame
        ).reshape(self.num_envs, -1)
        return torch.cat([self.joint_pos_multi_future, body_pos_multi_frame], dim=1)

    @property
    def command_multi_future_joint_body_abs_pos(self) -> torch.Tensor:
        """Return joint positions and anchor-relative body positions (without rotation).

        Unlike ``command_multi_future_joint_body_pos``, the body position deltas
        are not rotated into the anchor's local frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        body_pos_multi_frame = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).reshape(self.num_envs, -1)
        anchor_pos_multi_frame = (
            self.motion_lib.get_body_pos_w(self.future_motion_ids, self.future_time_steps)[
                :, self.motion_anchor_body_index
            ]
            .view(self.num_envs, self.num_future_frames, 1, 3)
            .expand(self.num_envs, self.num_future_frames, self.num_bodies, 3)
            .reshape(self.num_envs, -1)
        )
        body_pos_relative_to_anchor_multi_frame = body_pos_multi_frame - anchor_pos_multi_frame
        return torch.cat(
            [self.joint_pos_multi_future, body_pos_relative_to_anchor_multi_frame], dim=1
        )

    # @property
    # def command_multi_future_joint_body_diff_pos(self) -> torch.Tensor:
    #     body_pos_w = self.motion_lib.get_body_pos_w(self.future_motion_ids, self.future_time_steps).view(self.num_envs, self.num_future_frames, -1, 3)  # noqa: E501
    #     body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, None, :]
    #     body_pos_dif = body_pos_w_env - self.robot_body_pos_w[:, None, :, :]
    #     body_pos_relative_to_robot_anchor_multi_frame = quat_apply_yaw(quat_inv(self.robot_anchor_quat_w.repeat(1, self.num_future_frames, self.num_bodies, 1)), body_pos_dif).reshape(self.num_envs, -1)  # noqa: E501
    #     return torch.cat([self.joint_pos_multi_future, body_pos_relative_to_robot_anchor_multi_frame], dim=1)

    @property
    def command_multi_future_joint_body_diff_pos(self) -> torch.Tensor:
        """Return joint positions and robot-relative body position differences.

        Computes body positions in robot-anchor-relative frame: positions are
        translated so the XY comes from the robot root and Z from the reference
        anchor, then rotated by the heading difference. The result is the
        difference from the current robot body positions, expressed in the
        robot's local frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        N, F, B = self.num_envs, self.num_future_frames, self.num_bodies
        anchor_pos_w_repeat = self.anchor_pos_w_multi_future.view(N, F, 1, 3).expand(N, F, B, 3)
        anchor_quat_w_repeat = self.anchor_quat_w_multi_future.view(N, F, 1, 4).expand(N, F, B, 4)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, None, :].expand(N, F, B, 3)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, None, :].expand(N, F, B, 4)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()  # Root position of the robot
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = torch_transform.get_heading_q(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )
        body_pos_relative_w_multi_frame = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w_multi_future.view(N, F, B, 3) - anchor_pos_w_repeat
        )
        body_pos_dif = body_pos_relative_w_multi_frame - self.robot_body_pos_w.view(
            N, 1, B, 3
        ).expand(N, F, B, 3)
        body_pos_relative_to_robot_anchor_multi_frame = quat_apply_yaw(
            quat_inv(robot_anchor_quat_w_repeat), body_pos_dif
        ).reshape(N, -1)
        return torch.cat(
            [self.joint_pos_multi_future, body_pos_relative_to_robot_anchor_multi_frame], dim=1
        )

    @property
    def command_multi_future_lower_body(self) -> torch.Tensor:
        return torch.cat(
            [self.joint_pos_lower_body_multi_future, self.joint_vel_lower_body_multi_future], dim=1
        )

    @property
    def command_multi_future_lower_body_joint_pos(self) -> torch.Tensor:
        return self.joint_pos_lower_body_multi_future

    # =========================================================================
    # Egocentric joint transforms (positions + rotations) for reference frames
    # =========================================================================

    @property
    def num_bodies_full(self) -> int:
        """Get the full number of bodies (all bodies, not just selected body_indexes)."""
        return self.motion_lib.num_bodies_full

    @property
    def egocentric_joint_positions_multi_future(self) -> torch.Tensor:
        """Get body positions (including pelvis) in egocentric frame for all future frames.

        Egocentric frame = projected root frame (heading/yaw only rotation, z=0 projected).
        For each future frame, positions are relative to that frame's projected root.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 3] joint positions
                          in egocentric (projected root) frame
        """
        N, F, B = self.num_envs, self.num_future_frames, self.num_bodies_full

        # Get full body positions in world frame for all future frames
        body_pos_w = self.motion_lib.get_body_pos_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 3)

        # Get anchor (pelvis) positions for each future frame
        anchor_body_idx_full = self.motion_lib.m_cfg.get("anchor_body_idx_full", 0)
        anchor_pos_w = body_pos_w[:, :, anchor_body_idx_full, :]  # [N, F, 3]

        # Project anchor to ground plane (z=0) for egocentric frame
        anchor_pos_projected = anchor_pos_w.clone()
        anchor_pos_projected[..., 2] = 0  # Set z to 0
        anchor_pos_projected_expanded = anchor_pos_projected.view(N, F, 1, 3).expand(N, F, B, 3)

        # Get anchor quaternions for each future frame
        body_quat_w = self.motion_lib.get_body_quat_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 4)
        anchor_quat_w = body_quat_w[:, :, anchor_body_idx_full, :]  # [N, F, 4]

        # Extract heading quaternion using get_heading_q (canonicalize)
        anchor_heading_quat = torch_transform.get_heading_q(anchor_quat_w.reshape(-1, 4)).reshape(
            N, F, 4
        )
        anchor_heading_quat_expanded = anchor_heading_quat.view(N, F, 1, 4).expand(N, F, B, 4)

        # Compute egocentric positions:
        # 1. Translate to projected anchor origin (z=0)
        body_pos_relative = body_pos_w - anchor_pos_projected_expanded

        # 2. Rotate by inverse of heading quaternion
        body_pos_egocentric = quat_apply(
            quat_inv(anchor_heading_quat_expanded.reshape(-1, 4)), body_pos_relative.reshape(-1, 3)
        ).reshape(N, F, B, 3)

        return body_pos_egocentric

    @property
    def egocentric_joint_rotations_multi_future(self) -> torch.Tensor:
        """Get body rotations (including pelvis) in egocentric frame for all future frames.

        Returns 6D rotation representation (first two columns of rotation matrix).
        For each future frame, rotations are relative to that frame's projected root.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 6] joint rotations
                          in 6D representation (first two columns of rotation matrix)
        """
        N, F, B = self.num_envs, self.num_future_frames, self.num_bodies_full

        # Get full body quaternions in world frame for all future frames
        body_quat_w = self.motion_lib.get_body_quat_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 4)

        # Get anchor quaternions for each future frame
        anchor_body_idx_full = self.motion_lib.m_cfg.get(
            "anchor_body_idx_full", 0
        )  # pelvis is usually index 0
        anchor_quat_w = body_quat_w[:, :, anchor_body_idx_full, :]  # [N, F, 4]

        # Extract heading quaternion using get_heading_q (canonicalize)
        anchor_heading_quat = torch_transform.get_heading_q(anchor_quat_w.reshape(-1, 4)).reshape(
            N, F, 4
        )
        anchor_heading_quat_expanded = anchor_heading_quat.view(N, F, 1, 4).expand(N, F, B, 4)

        # Compute relative rotation: q_relative = q_heading_inv * q_body
        body_quat_egocentric = quat_mul(
            quat_inv(anchor_heading_quat_expanded.reshape(-1, 4)), body_quat_w.reshape(-1, 4)
        ).reshape(N, F, B, 4)

        # Convert to 6D representation (first two columns of rotation matrix)
        mat = matrix_from_quat(body_quat_egocentric)  # [N, F, B, 3, 3]
        body_rot_6d = isaac_utils_rotations.mat_to_rot6d_first_two_cols(mat).reshape(
            N, F, B, 6
        )  # [N, F, B, 6]

        return body_rot_6d

    @property
    def egocentric_joint_transforms_multi_future(self) -> torch.Tensor:
        """Combined egocentric joint transforms (positions + rotations) for all future frames.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 9]
                          (3 for position + 6 for 6D rotation)
        """
        positions = self.egocentric_joint_positions_multi_future  # [N, F, B, 3]
        rotations_ = self.egocentric_joint_rotations_multi_future  # [N, F, B, 6]
        return torch.cat([positions, rotations_], dim=-1)

    @property
    def root_transforms_relative_to_first_frame(self) -> torch.Tensor:
        """Get root transforms (position + rotation) relative to the first reference frame's
        projected root (z=0, heading only).

        Position: Delta from first frame's projected root position, rotated to first frame's heading frame.
        Rotation: Relative rotation from first frame's heading quaternion (6D representation).

        Returns:
            torch.Tensor: [num_envs, num_future_frames, 9]
                          (3 for relative position + 6 for 6D relative rotation)
        """  # noqa: D205
        N, F = self.num_envs, self.num_future_frames

        # Get root positions for all future frames
        root_pos_w = self.motion_lib.get_root_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, 3)

        # Get root quaternions for all future frames
        root_quat_w = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, 4)

        # First frame as reference - use projected root (z=0)
        first_frame_pos = root_pos_w[:, 0:1, :]  # [N, 1, 3]
        first_frame_pos_projected = first_frame_pos.clone()
        first_frame_pos_projected[..., 2] = 0  # Project to ground plane

        # Extract heading quaternion using get_heading_q (canonicalize)
        first_frame_quat = root_quat_w[:, 0:1, :]  # [N, 1, 4]
        first_frame_heading = torch_transform.get_heading_q(
            first_frame_quat.reshape(-1, 4)
        ).reshape(N, 1, 4)

        # Relative position: delta from projected first frame, rotated to heading frame
        delta_pos_w = root_pos_w - first_frame_pos_projected.expand(N, F, 3)

        # Rotate delta position by inverse of first frame's heading
        delta_pos_local = quat_apply(
            quat_inv(first_frame_heading.expand(N, F, 4).reshape(-1, 4)), delta_pos_w.reshape(-1, 3)
        ).reshape(N, F, 3)

        # Relative rotation: q_relative = q_heading_inv * q_current
        relative_quat = quat_mul(
            quat_inv(first_frame_heading.expand(N, F, 4).reshape(-1, 4)), root_quat_w.reshape(-1, 4)
        ).reshape(N, F, 4)

        # Convert to 6D representation (first two columns of rotation matrix)
        mat = matrix_from_quat(relative_quat)  # [N, F, 3, 3]
        relative_rot_6d = isaac_utils_rotations.mat_to_rot6d_first_two_cols(mat).reshape(
            N, F, 6
        )  # [N, F, 6]

        return torch.cat([delta_pos_local, relative_rot_6d], dim=-1)

    # =========================================================================
    # Motionbricks observations (global and first-frame-heading variants)
    # =========================================================================

    @property
    def global_joint_transforms_with_pelvis_rotation_and_height_multi_future(self) -> torch.Tensor:
        """
        Global joint transforms with per-frame root XY subtracted.
        Pelvis (body 0) has XY=(0,0), Z=height, rotation=raw global.
        No heading canonicalization — rotations are raw global quaternions in 6D.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 9]
                          (3 for position + 6 for 6D rotation)
        """
        N, F, B = self.num_envs, self.num_future_frames, self.num_bodies_full
        anchor_body_idx_full = self.motion_lib.m_cfg.get("anchor_body_idx_full", 0)

        # Get full body positions in world frame for all future frames
        body_pos_w = self.motion_lib.get_body_pos_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 3)

        # Per-frame root xy (keep z absolute)
        per_frame_root_xy = body_pos_w[:, :, anchor_body_idx_full, :].clone()  # [N, F, 3]
        per_frame_root_xy[..., 2] = 0  # Only subtract xy, z stays absolute
        body_pos_relative = body_pos_w - per_frame_root_xy.unsqueeze(2)

        # Get full body quaternions — raw global, no canonicalization
        body_quat_w = self.motion_lib.get_body_quat_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 4)

        # Convert to 6D representation
        mat = matrix_from_quat(body_quat_w)  # [N, F, B, 3, 3]
        body_rot_6d = isaac_utils_rotations.mat_to_rot6d_first_two_cols(mat).reshape(N, F, B, 6)

        return torch.cat([body_pos_relative, body_rot_6d], dim=-1)

    @property
    def global_root_xy_delta_wrt_first_frame_multi_future(self) -> torch.Tensor:
        """
        Root xy in global frame minus first frame root xy: root_xy[t] - root_xy[0].

        Returns:
            torch.Tensor: [num_envs, num_future_frames, 2]
        """
        N, F = self.num_envs, self.num_future_frames

        root_pos_w = self.motion_lib.get_root_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, 3)

        return root_pos_w[:, :, :2] - root_pos_w[:, 0:1, :2]

    @property
    def global_root_xy_delta_wrt_previous_frame_multi_future(self) -> torch.Tensor:
        """
        Per-frame root xy delta in global frame: root_xy[t] - root_xy[t-1].
        First frame delta is zero.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, 2]
        """
        N, F = self.num_envs, self.num_future_frames

        root_pos_w = self.motion_lib.get_root_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, 3)

        xy = root_pos_w[:, :, :2]
        deltas = xy[:, 1:, :] - xy[:, :-1, :]  # [N, F-1, 2]
        return torch.cat([torch.zeros_like(deltas[:, :1, :]), deltas], dim=1)

    @property
    def firstframe_heading_canonicalized_global_joint_transforms_multi_future(self) -> torch.Tensor:
        """
        Joint transforms in first-frame heading frame with per-frame root XY subtracted.
        Pelvis (body 0) has XY=(0,0), Z=height in heading frame, rotation=heading-relative.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 9]
                          (3 for position + 6 for 6D rotation)
        """
        N, F, B = self.num_envs, self.num_future_frames, self.num_bodies_full
        anchor_body_idx_full = self.motion_lib.m_cfg.get("anchor_body_idx_full", 0)

        # Get full body positions and quaternions in world frame
        body_pos_w = self.motion_lib.get_body_pos_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 3)

        body_quat_w = self.motion_lib.get_body_quat_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 4)

        # First frame heading quaternion
        first_root_quat = body_quat_w[:, 0:1, anchor_body_idx_full, :]  # [N, 1, 4]
        first_heading = torch_transform.get_heading_q(first_root_quat.reshape(-1, 4)).reshape(
            N, 1, 4
        )
        first_heading_inv = quat_inv(first_heading)  # [N, 1, 4]

        # Rotate all positions into heading frame (subtract first-frame root XY first for z=0 proj)
        first_root_pos = body_pos_w[:, 0:1, anchor_body_idx_full, :]  # [N, 1, 3]
        first_root_pos_projected = first_root_pos.clone()
        first_root_pos_projected[..., 2] = 0
        body_pos_relative = body_pos_w - first_root_pos_projected.unsqueeze(2)

        first_heading_inv_expanded = first_heading_inv.view(N, 1, 1, 4).expand(N, F, B, 4)
        body_pos_in_heading = quat_apply(
            first_heading_inv_expanded.reshape(-1, 4), body_pos_relative.reshape(-1, 3)
        ).reshape(N, F, B, 3)

        # Now subtract per-frame root XY (in heading frame)
        per_frame_root_xy = body_pos_in_heading[:, :, anchor_body_idx_full, :].clone()  # [N, F, 3]
        per_frame_root_xy[..., 2] = 0  # Only subtract XY
        body_pos_in_heading = body_pos_in_heading - per_frame_root_xy.unsqueeze(2)

        # Rotations: inv(heading_0) * body_quat_t
        body_quat_in_heading = quat_mul(
            first_heading_inv_expanded.reshape(-1, 4), body_quat_w.reshape(-1, 4)
        ).reshape(N, F, B, 4)

        # Convert to 6D representation
        mat = matrix_from_quat(body_quat_in_heading)  # [N, F, B, 3, 3]
        body_rot_6d = isaac_utils_rotations.mat_to_rot6d_first_two_cols(mat).reshape(N, F, B, 6)

        return torch.cat([body_pos_in_heading, body_rot_6d], dim=-1)

    @property
    def firstframe_heading_canonicalized_global_root_xy_delta_multi_future(self) -> torch.Tensor:
        """
        Root XY delta in first frame's heading frame.
        Reuses root_transforms_relative_to_first_frame (which is already in first heading frame).

        Returns:
            torch.Tensor: [num_envs, num_future_frames, 2]
        """
        return self.root_transforms_relative_to_first_frame[:, :, :2]

    @property
    def smpl_joints(self) -> torch.Tensor:
        """Return SMPL joint positions for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_smpl_joints, 3)``.
        """
        return self.motion_lib.get_smpl_joints(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def smpl_joints_multi_future(self) -> torch.Tensor:
        """Return SMPL joint positions for all SMPL future frames.

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames, num_smpl_joints, 3)``.
        """
        smpl_joints_mf = self.motion_lib.get_smpl_joints(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return smpl_joints_mf.view(
            self.num_envs, self.smpl_num_future_frames, *smpl_joints_mf.shape[1:]
        )

    @property
    def smpl_transl_multi_future(self) -> torch.Tensor:
        smpl_transl_mf = self.motion_lib.get_smpl_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return smpl_transl_mf.view(
            self.num_envs, self.smpl_num_future_frames, *smpl_transl_mf.shape[1:]
        )

    @property
    def smpl_transl_z_multi_future(self) -> torch.Tensor:
        smpl_transl_mf = self.motion_lib.get_smpl_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        # Only return y dimension (height) --- y is the z for smpl
        return smpl_transl_mf[..., 1:2].view(self.num_envs, self.smpl_num_future_frames, 1)

    @property
    def smpl_pose(self) -> torch.Tensor:
        """Return full SMPL pose (root + body joints) in axis-angle for current frame.

        Returns:
            Tensor of shape ``(num_envs, 72)`` (24 joints * 3 axis-angle).
        """
        return self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def smpl_body_pose(self) -> torch.Tensor:
        """Return SMPL body pose (excluding root) in axis-angle for current frame.

        Returns:
            Tensor of shape ``(num_envs, 69)`` (23 body joints * 3 axis-angle).
        """
        return self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[..., 3:]

    @property
    def smpl_body_pose_6d(self) -> torch.Tensor:
        """Return SMPL body pose in 6D rotation representation for current frame.

        Returns:
            Tensor of shape ``(num_envs, 23 * 6)``.
        """
        smpl_body_pose = self.smpl_body_pose.reshape(-1, 23, 3)
        quat = torch_transform.angle_axis_to_quaternion(smpl_body_pose)
        mat = matrix_from_quat(quat)
        smpl_body_pose_6d = mat[..., :2].reshape(smpl_body_pose.shape[0], -1)
        return smpl_body_pose_6d

    def smpl_root_ytoz_up(self, root_quat_y_up: torch.Tensor) -> torch.Tensor:
        """Convert SMPL root quaternion from Y-up to Z-up convention.

        Args:
            root_quat_y_up: Root quaternions in Y-up frame, ``(N, 4)``.

        Returns:
            Root quaternions rotated to Z-up frame, ``(N, 4)``.
        """
        base_rot = torch_transform.angle_axis_to_quaternion(
            torch.tensor([[np.pi / 2, 0.0, 0.0]]).to(root_quat_y_up)
        )
        root_quat_z_up = isaac_utils_rotations.quat_mul(
            base_rot.repeat(root_quat_y_up.shape[0], 1), root_quat_y_up, w_last=False
        )
        return root_quat_z_up

    @property
    def smpl_pose_noheading(self) -> torch.Tensor:
        """Return SMPL pose with root heading removed (preserving pitch/roll).

        Converts SMPL root from Y-up if needed, removes the base rotation offset,
        extracts and removes heading, then re-encodes as axis-angle.

        Returns:
            Tensor of shape ``(num_envs, 72)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = isaac_utils_rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_heading_inv = isaac_utils_rotations.calc_heading_quat_inv(root_quat, w_last=False)
        root_quat_noheading = isaac_utils_rotations.quat_mul(
            root_heading_inv, root_quat, w_last=False
        )
        smpl_pose[..., :3] = torch_transform.quaternion_to_angle_axis(root_quat_noheading).view(
            smpl_pose.shape[:-1] + (3,)
        )
        return smpl_pose

    @property
    def smpl_root_quat_w(self) -> torch.Tensor:
        """Return SMPL root quaternion in Z-up world frame for current frame.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = isaac_utils_rotations.remove_smpl_base_rot(root_quat, w_last=False)
        return root_quat

    @property
    def smpl_root_quat_w_multi_future(self) -> torch.Tensor:
        """Return SMPL root quaternions in Z-up world frame for all SMPL future frames.

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames, 4)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = isaac_utils_rotations.remove_smpl_base_rot(root_quat, w_last=False).view(
            self.num_envs, self.smpl_num_future_frames, 4
        )
        return root_quat

    @property
    def smpl_root_quat_w_dif_l_multi_future(self) -> torch.Tensor:
        """Return SMPL root orientation relative to robot orientation for all SMPL future frames.

        Computes ``quat_inv(robot_anchor) * smpl_root`` and returns 6D rotation
        matrix representation (first 2 columns).

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames * 6)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = isaac_utils_rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.smpl_num_future_frames, 1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def smpl_root_quat_w_dif_refheading_multi_future(self) -> torch.Tensor:
        """SMPL root orientation canonicalized by the first SMPL future frame's heading.

        Same extraction as smpl_root_quat_w_dif_l_multi_future but uses the heading
        of the first SMPL future frame instead of the robot's orientation.

        Returns:
            torch.Tensor: 6D rotation matrix representation,
                shape (num_envs, smpl_num_future_frames * 6)
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        smpl_root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            smpl_root_quat = self.smpl_root_ytoz_up(smpl_root_quat)
        smpl_root_quat = isaac_utils_rotations.remove_smpl_base_rot(smpl_root_quat, w_last=False)
        smpl_root_quat = smpl_root_quat.view(self.num_envs, self.smpl_num_future_frames, 4)
        ref_first_heading = torch_transform.get_heading_q(smpl_root_quat[:, 0, :])
        root_rot_dif = quat_mul(
            quat_inv(
                ref_first_heading.view(self.num_envs, 1, 4).expand(
                    -1, self.smpl_num_future_frames, -1
                )
            ),
            smpl_root_quat,
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def smpl_root_quat_w_dif_heading_multi_future(self) -> torch.Tensor:
        """SMPL root orientation canonicalized by robot heading (yaw) only.

        Same extraction as smpl_root_quat_w_dif_l_multi_future but uses the
        robot's heading (yaw) instead of full orientation for canonicalization.

        Returns:
            torch.Tensor: 6D rotation matrix representation,
                shape (num_envs, smpl_num_future_frames * 6)
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = isaac_utils_rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.anchor_heading_quat.view(self.num_envs, 1, 4).expand(
                    -1, self.smpl_num_future_frames, -1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def smpl_pose_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        ).reshape(self.num_envs, self.smpl_num_future_frames, -1)

    @property
    def smpl_body_pose_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )[..., 3:].reshape(self.num_envs, -1)

    @property
    def smpl_body_pose_multi_future_6d(self) -> torch.Tensor:
        smpl_body_pose = self.smpl_body_pose_multi_future.reshape(
            -1, self.smpl_num_future_frames, 23, 3
        )
        quat = torch_transform.angle_axis_to_quaternion(smpl_body_pose)
        mat = matrix_from_quat(quat)
        smpl_body_pose_6d = mat[..., :2].reshape(smpl_body_pose.shape[0], -1)
        return smpl_body_pose_6d

    # --- SOMA skeleton properties ---

    @property
    def soma_joints(self) -> torch.Tensor:
        """SOMA joints in Z-up body-local frame. Stored Z-up in PKL (same as SMPL)."""
        return self.motion_lib.get_soma_joints(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def soma_joints_multi_future(self) -> torch.Tensor:
        """SOMA joints multi-future in Z-up body-local frame."""
        soma_joints_mf = self.motion_lib.get_soma_joints(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return soma_joints_mf.view(
            self.num_envs, self.smpl_num_future_frames, *soma_joints_mf.shape[1:]
        )

    @property
    def soma_root_quat_w(self) -> torch.Tensor:
        """SOMA root quaternion in Z-up world frame (wxyz)."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        return root_quat

    @property
    def soma_root_quat_w_multi_future(self) -> torch.Tensor:
        """SOMA root quaternion multi-future in Z-up world frame (wxyz)."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        return root_quat.view(self.num_envs, self.smpl_num_future_frames, 4)

    @property
    def soma_root_quat_w_dif_l_multi_future(self) -> torch.Tensor:
        """SOMA root orientation relative to robot anchor, as 6D rotation matrix."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.smpl_num_future_frames, 1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def soma_transl_multi_future(self) -> torch.Tensor:
        """SOMA hips translation multi-future. Stored Y-up (same as SMPL transl)."""
        soma_transl_mf = self.motion_lib.get_soma_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return soma_transl_mf.view(
            self.num_envs, self.smpl_num_future_frames, *soma_transl_mf.shape[1:]
        )

    @property
    def object_root_pos(self) -> torch.Tensor:
        """Return object root position in world frame with env origin and z-offset.

        Returns:
            Tensor of shape ``(num_envs, num_objects, 3)``.
        """
        object_root_pos = self.motion_lib.get_object_root_pos(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        # Apply z-offset if configured (e.g., to lower chair into ground)
        z_offset = getattr(self.cfg, "object_z_offset", 0.0)
        if z_offset != 0.0:
            object_root_pos = object_root_pos.clone()
            object_root_pos[..., 2] += z_offset
        return object_root_pos + self._env.scene.env_origins[:, None, :]

    @property
    def object_root_quat(self) -> torch.Tensor:
        return self.motion_lib.get_object_root_quat(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def object_root_lin_vel(self) -> torch.Tensor:
        return self.motion_lib.get_object_lin_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def object_root_ang_vel(self) -> torch.Tensor:
        return self.motion_lib.get_object_ang_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def active_articulation_root_pos(self) -> torch.Tensor:
        root_pos = self.motion_lib.get_active_articulation_root_pos(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        return root_pos + self._env.scene.env_origins

    @property
    def active_articulation_root_quat(self) -> torch.Tensor:
        return self.motion_lib.get_active_articulation_root_quat(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def active_articulation_root_lin_vel(self) -> torch.Tensor:
        return self.motion_lib.get_active_articulation_root_lin_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def active_articulation_root_ang_vel(self) -> torch.Tensor:
        return self.motion_lib.get_active_articulation_root_ang_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def active_articulation_joint_pos(self) -> torch.Tensor:
        return self.motion_lib.get_active_articulation_joint_pos(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def active_articulation_joint_vel(self) -> torch.Tensor:
        return self.motion_lib.get_active_articulation_joint_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def active_articulation_joint_valid_mask(self) -> torch.Tensor:
        return self.motion_lib.get_active_articulation_joint_valid_mask(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def active_articulation_root_dynamic_mask(self) -> torch.Tensor:
        values = self.motion_lib.get_active_articulation_root_dynamic_mask(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        return values.reshape(self.num_envs, -1)[:, 0] > 0.5

    @property
    def active_articulation_joint_pos_multi_future(self) -> torch.Tensor:
        values = self.motion_lib.get_active_articulation_joint_pos(
            self.future_motion_ids, self.future_time_steps
        )
        return values.view(self.num_envs, self.num_future_frames, -1)

    @property
    def active_articulation_joint_valid_mask_multi_future(self) -> torch.Tensor:
        values = self.motion_lib.get_active_articulation_joint_valid_mask(
            self.future_motion_ids, self.future_time_steps
        )
        return values.view(self.num_envs, self.num_future_frames, -1)

    @property
    def active_entity_root_pos(self) -> torch.Tensor:
        """Reference base position for either a rigid object or articulation."""
        if self._active_entity_policy_pose_mode() == "panel":
            return self.target_object_root_pos
        if hasattr(self.motion_lib, "_motion_active_articulation_root_pos"):
            return self.active_articulation_root_pos
        return self.target_object_root_pos

    @property
    def active_entity_root_quat(self) -> torch.Tensor:
        if self._active_entity_policy_pose_mode() == "panel":
            return self.target_object_root_quat
        if hasattr(self.motion_lib, "_motion_active_articulation_root_quat"):
            return self.active_articulation_root_quat
        return self.target_object_root_quat

    @property
    def active_entity_root_lin_vel(self) -> torch.Tensor:
        if self._active_entity_policy_pose_mode() == "panel":
            return self.target_object_lin_vel
        if hasattr(self.motion_lib, "_motion_active_articulation_root_lin_vel"):
            return self.active_articulation_root_lin_vel
        return self.target_object_lin_vel

    @property
    def active_entity_root_ang_vel(self) -> torch.Tensor:
        if self._active_entity_policy_pose_mode() == "panel":
            return self.target_object_ang_vel
        if hasattr(self.motion_lib, "_motion_active_articulation_root_ang_vel"):
            return self.active_articulation_root_ang_vel
        return self.target_object_ang_vel

    @property
    def active_entity_root_pos_multi_future(self) -> torch.Tensor:
        if self._active_entity_policy_pose_mode() == "panel":
            return self.target_object_root_pos_multi_future
        if hasattr(self.motion_lib, "_motion_active_articulation_root_pos"):
            root_pos = self.motion_lib.get_active_articulation_root_pos(
                self.future_motion_ids, self.future_time_steps
            ).view(self.num_envs, self.num_future_frames, 3)
            return root_pos + self._env.scene.env_origins[:, None, :]
        return self.target_object_root_pos_multi_future

    @property
    def active_entity_root_quat_multi_future(self) -> torch.Tensor:
        if self._active_entity_policy_pose_mode() == "panel":
            return self.target_object_root_quat_multi_future
        if hasattr(self.motion_lib, "_motion_active_articulation_root_quat"):
            return self.motion_lib.get_active_articulation_root_quat(
                self.future_motion_ids, self.future_time_steps
            ).view(self.num_envs, self.num_future_frames, 4)
        return self.target_object_root_quat_multi_future

    def _active_entity_policy_pose_mode(self) -> str:
        config = getattr(self._env.cfg, "config", {}) or {}
        mode = str(config.get("active_entity_policy_pose_mode", "frame"))
        if mode not in {"frame", "panel"}:
            raise ValueError(
                "active_entity_policy_pose_mode must be 'frame' or 'panel', "
                f"got {mode!r}"
            )
        return mode

    @property
    def object_bps(self) -> torch.Tensor:
        """BPS shape encoding for the current object per env.

        Static within an episode (does not depend on frame).

        Returns:
            Tensor of shape ``(num_envs, d)`` where d = num_basis points.
        """
        return self.motion_lib.get_object_bps(self.motion_ids)

    def _get_object_pos_with_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object position with randomization offset applied.

        This method also resamples the offset for the given env_ids.
        Called at environment reset to place object with random offset.

        Args:
            env_ids: Environment indices being reset

        Returns:
            Object positions with offset applied, shape [len(env_ids), 3]
        """
        obj_pos = self.object_root_pos[env_ids, 0].clone()

        if self.cfg.object_position_randomize:
            # Resample offset for these environments
            rand_cfg = self.cfg.object_position_randomization or {}
            x_range = rand_cfg.get("x", 0.0)
            y_range = rand_cfg.get("y", 0.0)
            z_range = rand_cfg.get("z", 0.0)

            # Generate new random offsets
            self._object_position_offset[env_ids, 0] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * x_range
            self._object_position_offset[env_ids, 1] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * y_range
            self._object_position_offset[env_ids, 2] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * z_range

            # Apply offset
            obj_pos = obj_pos + self._object_position_offset[env_ids]

        # Apply initial z offset (e.g., +0.05 to raise taller objects above table)
        init_z_offset = getattr(self.cfg, "object_init_z_offset", 0.0)
        if init_z_offset != 0.0:
            obj_pos[:, 2] += init_z_offset

        return obj_pos

    @property
    def object_root_pos_multi_future(self) -> torch.Tensor:
        """Return object root positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames, num_objects, 3)``.
        """
        object_root_pos = self.motion_lib.get_object_root_pos(
            self.future_motion_ids, self.future_time_steps
        )
        object_root_pos_view = object_root_pos.view(self.num_envs, self.num_future_frames, -1, 3)
        # Apply z-offset if configured (e.g., to lower chair into ground)
        z_offset = getattr(self.cfg, "object_z_offset", 0.0)
        if z_offset != 0.0:
            object_root_pos_view = object_root_pos_view.clone()
            object_root_pos_view[..., 2] += z_offset
        return object_root_pos_view + self._env.scene.env_origins[:, None, None, :]

    @property
    def object_root_quat_multi_future(self) -> torch.Tensor:
        object_root_quat = self.motion_lib.get_object_root_quat(
            self.future_motion_ids, self.future_time_steps
        )
        return object_root_quat.view(self.num_envs, self.num_future_frames, -1, 4)

    def _get_contact_center_world(self, hand: str) -> torch.Tensor | None:
        """Get object contact center in world frame for the given hand.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs, 3) in world frame, or None if not available.
        """
        contact_center = self.motion_lib.get_object_contact_center(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )
        if contact_center is None:
            return None

        # Transform from object-local to world frame
        obj_pos = self.target_object_root_pos
        obj_quat = self.target_object_root_quat

        from gear_sonic.isaac_utils import rotations as isaac_utils_rotations  # noqa: PLC0415

        rotated_center = isaac_utils_rotations.quat_rotate(obj_quat, contact_center, w_last=False)
        world_center = rotated_center + obj_pos
        return world_center

    @property
    def object_contact_center_left(self) -> torch.Tensor | None:
        """Get left hand object contact center in world frame. Shape: (num_envs, 3)."""
        return self._get_contact_center_world("left_hand")

    @property
    def object_contact_center_right(self) -> torch.Tensor | None:
        """Get right hand object contact center in world frame. Shape: (num_envs, 3)."""
        return self._get_contact_center_world("right_hand")

    def get_in_contact(self, hand: str = "right_hand") -> torch.Tensor | None:
        """Get binary in_contact label for the given hand at current timestep.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs,) with 1.0 if in contact, 0.0 otherwise,
            or None if not available.
        """
        return self.motion_lib.get_object_in_contact(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )

    def get_hand_action(self, hand: str = "right_hand") -> torch.Tensor | None:
        """Get discrete hand action (open/closed) for the given hand at current timestep.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs,) with -1.0 = open, +1.0 = closed,
            or None if not available.
        """
        return self.motion_lib.get_hand_action(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )

    @property
    def joint_pos(self) -> torch.Tensor:
        """Return reference joint positions for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_dof)``.
        """
        return self.motion_lib.get_dof_pos(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def joint_pos_multi_future(self) -> torch.Tensor:
        """Return reference joint positions for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.motion_lib.get_dof_pos(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def joint_pos_multi_future_for_smpl(self) -> torch.Tensor:
        return self.motion_lib.get_dof_pos(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        ).view(self.num_envs, -1)

    @property
    def joint_pos_lower_body_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_dof_pos(self.future_motion_ids, self.future_time_steps)[
            ..., self.lower_joint_isaaclab_indices
        ].view(self.num_envs, -1)

    @property
    def joint_vel(self) -> torch.Tensor:
        """Return reference joint velocities for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_dof)``.
        """
        return self.motion_lib.get_dof_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def joint_vel_multi_future(self) -> torch.Tensor:
        """Return reference joint velocities for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.motion_lib.get_dof_vel(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def root_pos_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def root_quat_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_quat_w(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def root_z(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, 2:3].view(self.num_envs, -1)

    @property
    def root_z_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(self.future_motion_ids, self.future_time_steps)[
            :, 2:3
        ].view(self.num_envs, -1)

    @property
    def joint_vel_lower_body_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_dof_vel(self.future_motion_ids, self.future_time_steps)[
            ..., self.lower_joint_isaaclab_indices
        ].view(self.num_envs, -1)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Return reference body positions in world frame for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        return (
            self.motion_lib.get_body_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference body positions in world frame for all future frames.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_bodies * 3)``.
        """
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 3)
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, None, :]
        return body_pos_w_env.reshape(self.num_envs, -1)

    @property
    def body_pos_dif_w(self) -> torch.Tensor:
        """Return position difference (reference - robot) in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, :]
        body_pos_dif = body_pos_w_env - self.robot_body_pos_w
        return body_pos_dif

    @property
    def body_pos_dif_w_multi_future(self) -> torch.Tensor:
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 3)
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, None, :]
        body_pos_dif = body_pos_w_env - self.robot_body_pos_w[:, None, :, :]
        return body_pos_dif.reshape(self.num_envs, -1)

    @property
    def body_pos_dif_l(self) -> torch.Tensor:
        """Return body position difference de-headed into robot-local frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        body_pos_dif_w = self.body_pos_dif_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)
        deheaded_dif_l = quat_apply_yaw(quat_inv(root_quat), body_pos_dif_w)
        return deheaded_dif_l

    @property
    def body_pos_dif_l_multi_future(self) -> torch.Tensor:
        body_pos_dif_w = self.body_pos_dif_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_dif_l = quat_apply_yaw(quat_inv(root_quat), body_pos_dif_w)
        return deheaded_dif_l

    @property
    def root_lin_vel_b_2d(self) -> torch.Tensor:
        """Return reference root linear velocity (XY only) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 2)``.
        """
        root_lin_vel_w = self.motion_lib.get_root_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = self.anchor_quat_w.view(self.num_envs, 1, 4)
        root_lin_vel_l = quat_apply_yaw(quat_inv(root_quat), root_lin_vel_w)[:, :2]
        return root_lin_vel_l

    @property
    def root_ang_vel_b_1d(self) -> torch.Tensor:
        """Return reference root angular velocity (yaw only) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        root_ang_vel_w = self.motion_lib.get_root_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = self.anchor_quat_w.view(self.num_envs, 1, 4)
        root_ang_vel_l = quat_apply_yaw(quat_inv(root_quat), root_ang_vel_w)[:, 2:3]
        return root_ang_vel_l

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Return reference body quaternions in world frame for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 4)``.
        """
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_quat_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def body_quat_dif_w(self) -> torch.Tensor:
        body_quat_w = self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        body_quat_dif = quat_mul(quat_inv(body_quat_w), self.robot_body_quat_w)
        return body_quat_dif

    @property
    def body_quat_dif_w_multi_future(self) -> torch.Tensor:
        ref_body_quat_w = self.motion_lib.get_body_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 4)
        robot_body_quat_w = self.robot_body_quat_w.view(
            self.num_envs, 1, self.num_bodies, 4
        ).repeat(1, self.num_future_frames, 1, 1)
        body_quat_dif = quat_mul(quat_inv(ref_body_quat_w), robot_body_quat_w)
        return body_quat_dif

    @property
    def anchor_heading_quat(self) -> torch.Tensor:
        """Return robot anchor heading quaternion (yaw-only, pitch/roll removed).

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        return torch_transform.get_heading_q(self.robot_anchor_quat_w)

    # @property
    # def body_quat_dif_l(self) -> torch.Tensor:
    #     body_quat_dif_w = self.body_quat_dif_w
    #     root_heading = self.anchor_heading_quat.view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)
    #     root_heading_inv = quat_inv(self.anchor_heading_quat).view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)  # noqa: E501
    #     deheaded_dif_l = quat_mul(quat_mul(root_heading_inv, body_quat_dif_w), root_heading)
    #     mat = matrix_from_quat(deheaded_dif_l)
    #     deheaded_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
    #     return deheaded_dif_l_mat

    # @property
    # def body_quat_dif_l_multi_future(self) -> torch.Tensor:
    #     body_quat_dif_w = self.body_quat_dif_w_multi_future
    #     root_heading = self.anchor_heading_quat.view(self.num_envs, 1, 1, 4).repeat(1, self.num_future_frames, self.num_bodies, 1)  # noqa: E501
    #     root_heading_inv = quat_inv(self.anchor_heading_quat).view(self.num_envs, 1, 1, 4).repeat(1, self.num_future_frames, self.num_bodies, 1)  # noqa: E501
    #     deheaded_dif_l = quat_mul(quat_mul(root_heading_inv, body_quat_dif_w), root_heading)
    #     mat = matrix_from_quat(deheaded_dif_l)
    #     deheaded_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
    #     return deheaded_dif_l_mat

    @property
    def root_rot_dif_l(self) -> torch.Tensor:
        """Return reference root orientation relative to robot orientation in 6D repr.

        Returns:
            Tensor of shape ``(num_envs, 6)``.
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_rot_dif_w = quat_mul(quat_inv(self.robot_anchor_quat_w), ref_root_quat)
        # root_heading = self.anchor_heading_quat.view(self.num_envs, 1, 4).repeat(1, 1, 1)
        # root_heading_inv = quat_inv(self.anchor_heading_quat).view(self.num_envs, 1, 4).repeat(1, 1, 1)
        # deheaded_rot_dif_l = quat_mul(quat_mul(root_heading_inv, root_rot_dif_w), root_heading)
        mat = matrix_from_quat(root_rot_dif_w)
        deheaded_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return deheaded_rot_dif_l_mat

    @property
    def root_rot_dif_l_multi_future(self) -> torch.Tensor:
        """Return reference root orientation relative to robot for all future frames.

        Uses the full robot orientation for canonicalization (preserves heading diff).

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 6)``.
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.num_future_frames, 1
                )
            ),
            ref_root_quat.view(self.num_envs, self.num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def root_rot_dif_heading_multi_future(self) -> torch.Tensor:
        """Reference root orientation canonicalized by robot heading (yaw) only.

        Unlike root_rot_dif_l_multi_future which uses the full robot orientation,
        this version only uses the heading (yaw) for canonicalization. This preserves
        the reference motion's pitch/roll relative to gravity while removing the
        heading difference.

        Returns:
            torch.Tensor: 6D rotation matrix representation (first 2 columns),
                shape (num_envs, num_future_frames * 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        # Use only the heading (yaw) of the robot orientation for canonicalization
        root_rot_dif = quat_mul(
            quat_inv(
                self.anchor_heading_quat.view(self.num_envs, 1, 4).expand(
                    -1, self.num_future_frames, -1
                )
            ),
            ref_root_quat.view(self.num_envs, self.num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_heading_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_heading_mat

    @property
    def root_rot_dif_refheading_multi_future(self) -> torch.Tensor:
        """Reference root orientation canonicalized by the first future frame's heading.

        Instead of using the robot's current heading for canonicalization, this uses
        the heading of the first (immediate) target frame from the reference motion.
        This makes the trajectory representation independent of the robot's heading.

        Returns:
            torch.Tensor: 6D rotation matrix representation (first 2 columns),
                shape (num_envs, num_future_frames * 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, 4)
        # Use the heading of the first future frame as the canonical frame
        ref_first_heading = torch_transform.get_heading_q(ref_root_quat[:, 0, :])
        root_rot_dif = quat_mul(
            quat_inv(
                ref_first_heading.view(self.num_envs, 1, 4).expand(-1, self.num_future_frames, -1)
            ),
            ref_root_quat,
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def heading_diff_robot_ref(self) -> torch.Tensor:
        """Relative heading rotation from robot heading to reference first frame heading.

        Computes quat_mul(quat_inv(robot_heading), ref_first_heading), expressing
        the reference motion's heading direction in the robot's heading frame.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, 4)
        ref_first_heading = torch_transform.get_heading_q(ref_root_quat[:, 0, :])
        heading_diff = quat_mul(quat_inv(self.anchor_heading_quat), ref_first_heading)
        mat = matrix_from_quat(heading_diff)
        return mat[..., :2].reshape(self.num_envs, -1)  # (num_envs, 6)

    @property
    def raw_root_quat_w_multi_future(self) -> torch.Tensor:
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        return ref_root_quat.reshape(self.num_envs, self.num_future_frames, 4)

    @property
    def root_rot_w_multi_future(self) -> torch.Tensor:
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        mat = matrix_from_quat(ref_root_quat)
        root_rot_w_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_w_mat

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_lin_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def body_lin_vel_l(self) -> torch.Tensor:
        body_lin_vel_w = self.body_lin_vel_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_lin_vel_w)
        return deheaded_vel_l

    @property
    def body_lin_vel_l_multi_future(self) -> torch.Tensor:
        body_lin_vel_w = self.body_lin_vel_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_lin_vel_w)
        return deheaded_vel_l

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_ang_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def body_ang_vel_l(self) -> torch.Tensor:
        body_ang_vel_w = self.body_ang_vel_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_ang_vel_w)
        return deheaded_vel_l

    @property
    def body_ang_vel_l_multi_future(self) -> torch.Tensor:
        body_ang_vel_w = self.body_ang_vel_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_ang_vel_w)
        return deheaded_vel_l

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Return reference anchor body position in world frame.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        return (
            self.motion_lib.get_body_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )[:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference anchor positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 3)``.
        """
        anchor_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.motion_anchor_body_index].view(self.num_envs, self.num_future_frames, -1)
        anchor_pos_w_env = anchor_pos_w + self._env.scene.env_origins[:, None, :]
        return anchor_pos_w_env.reshape(self.num_envs, -1)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Return reference anchor body quaternion in world frame.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w_multi_future(self) -> torch.Tensor:
        """Return reference anchor quaternions for all future frames.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 4)``.
        """
        return self.motion_lib.get_body_quat_w(self.future_motion_ids, self.future_time_steps)[
            :, self.motion_anchor_body_index
        ].reshape(self.num_envs, -1)

    @property
    def anchor_ori_refheading(self) -> torch.Tensor:
        """Current anchor orientation canonicalized by its own heading.

        Uses get_heading_q(anchor_quat_w) as the canonical frame, removing the
        heading component and preserving pitch/roll relative to gravity.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        ref_heading = torch_transform.get_heading_q(self.anchor_quat_w)
        ori = quat_mul(quat_inv(ref_heading), self.anchor_quat_w)
        mat = matrix_from_quat(ori)
        return mat[..., :2].reshape(self.num_envs, -1)

    @property
    def anchor_ori_heading(self) -> torch.Tensor:
        """Current anchor orientation canonicalized by robot heading (yaw).

        Uses get_heading_q(robot_anchor_quat_w) as the canonical frame, preserving
        the reference motion's pitch/roll relative to gravity while removing the
        robot's heading.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        robot_heading = self.anchor_heading_quat
        ori = quat_mul(quat_inv(robot_heading), self.anchor_quat_w)
        mat = matrix_from_quat(ori)
        return mat[..., :2].reshape(self.num_envs, -1)

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(self.future_motion_ids, self.future_time_steps)[
            :, self.motion_anchor_body_index
        ].view(self.num_envs, -1)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(self.future_motion_ids, self.future_time_steps)[
            :, self.motion_anchor_body_index
        ].view(self.num_envs, -1)

    @property
    def vr_3point_body_quat_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.vr_3point_body_indices_motion]

    @property
    def reward_point_body_quat_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.reward_point_body_indices_motion]

    @property
    def vr_3point_body_quat_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(self.future_motion_ids, self.future_time_steps)[
            :, self.vr_3point_body_indices_motion
        ].view(self.num_envs, self.num_future_frames, len(self.cfg.vr_3point_body), -1)

    @property
    def head_orn_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(self.future_motion_ids, self.future_time_steps)[
            :, self.vr_3point_body_indices_motion[2]
        ].view(self.num_envs, self.num_future_frames, -1)

    @property
    def reward_point_body_pos_w(self) -> torch.Tensor:
        reward_point_original = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.reward_point_body_indices_motion]
        return (
            reward_point_original
            + quat_apply(self.reward_point_body_quat_w, self.reward_point_body_offsets)
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def vr_3point_body_pos_w(self) -> torch.Tensor:
        """Return reference VR 3-point body positions (with offsets) in world frame.

        The 3 points are typically left wrist, right wrist, and head. Offsets
        allow tracking a point displaced from the body origin (e.g., palm center).

        Returns:
            Tensor of shape ``(num_envs, 3, 3)``.
        """
        vr_3point_original = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.vr_3point_body_indices_motion]
        return (
            vr_3point_original
            + quat_apply(self.vr_3point_body_quat_w, self.vr_3point_body_offsets)
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def vr_3point_body_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference VR 3-point positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames, 3, 3)``.
        """
        vr_3point_original = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.vr_3point_body_indices_motion].view(
            self.num_envs, self.num_future_frames, len(self.cfg.vr_3point_body), -1
        )
        vr_3point_offset_extend = self.vr_3point_body_offsets.unsqueeze(1).repeat(
            1, self.num_future_frames, 1, 1
        )
        vr_3point_pos_w = (
            vr_3point_original
            + quat_apply(self.vr_3point_body_quat_w_multi_future, vr_3point_offset_extend)
            + self._env.scene.env_origins[:, None, None, :]
        )
        return vr_3point_pos_w

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Return the robot's current anchor body position in world frame.

        In offline mode, falls back to the reference motion root position.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        if getattr(self, "_offline", False):
            return self.motion_lib.get_root_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Return the robot's current anchor body quaternion in world frame.

        When ``use_ref_motion_root_quat_w_as_anchor`` is True or in offline mode,
        returns the reference motion root orientation (optionally with added noise)
        instead of the simulated robot state.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        if self.use_ref_motion_root_quat_w_as_anchor or getattr(self, "_offline", False):
            ref_root_quat = self.motion_lib.get_root_quat_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
            if self.ref_motion_root_rotation_noise is not None:
                ref_root_quat = quat_mul(ref_root_quat, self.ref_motion_root_rotation_noise)
            return ref_root_quat
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        assert not getattr(
            self, "_offline", False
        ), "robot_anchor_lin_vel_w is not available in offline mode"
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        assert not getattr(
            self, "_offline", False
        ), "robot_anchor_ang_vel_w is not available in offline mode"
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_vr_3point_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.vr_3point_body_indices]

    @property
    def robot_reward_point_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.reward_point_body_indices] + quat_apply(
            self.robot.data.body_quat_w[:, self.reward_point_body_indices],
            self.reward_point_body_offsets,
        )

    @property
    def robot_vr_3point_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.vr_3point_body_indices] + quat_apply(
            self.robot.data.body_quat_w[:, self.vr_3point_body_indices], self.vr_3point_body_offsets
        )

    @property
    def feet_l(self) -> torch.Tensor:
        return self.motion_lib.get_feet_l(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def feet_r(self) -> torch.Tensor:
        return self.motion_lib.get_feet_r(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def episode_encoder_index(self) -> torch.Tensor:
        return self.encoder_index

    def _update_metrics(self):
        """Compute tracking error metrics between reference motion and robot state.

        Populates ``self.metrics`` with per-env errors for anchor position/rotation,
        body position/rotation, and joint position/velocity. When there is a DOF
        mismatch (extra finger joints), only body joints are compared.
        """
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        # Compare only body joints when there's a DOF mismatch (motion lib has fewer DOFs than robot)
        if self.has_dof_mismatch:
            robot_body_joint_pos = self.robot_joint_pos[:, self.body_joint_indices]
            robot_body_joint_vel = self.robot_joint_vel[:, self.body_joint_indices]
            self.metrics["error_joint_pos"] = torch.abs(self.joint_pos - robot_body_joint_pos).mean(
                dim=-1
            )
            self.metrics["error_joint_vel"] = torch.abs(self.joint_vel - robot_body_joint_vel).mean(
                dim=-1
            )
        else:
            self.metrics["error_joint_pos"] = torch.abs(self.joint_pos - self.robot_joint_pos).mean(
                dim=-1
            )
            self.metrics["error_joint_vel"] = torch.abs(self.joint_vel - self.robot_joint_vel).mean(
                dim=-1
            )
        self._update_rp2_object_role_metrics()

    def resample_all_commands(self):
        """Resample motion clips and reset state for all environments at once."""
        self._resample_command(torch.arange(self.num_envs))

    def _load_contact_data(self):
        """Load contact data from file or directory and validate frame counts against motion library.

        Supports both:
        - Single file: loads one pkl file
        - Directory: loads all pkl files and merges them

        Sets:
            self._contact_data: Raw contact data dict
            self._first_contact_frame: Dict mapping motion_name -> first contact frame index
        """
        import joblib  # noqa: PLC0415

        self._first_contact_frame = None
        self._contact_data = None
        self._first_contact_lookup = None
        self._per_env_first_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._motion_contact_flags = None

        first_contact_source = str(
            getattr(self.cfg, "first_contact_reference_source", "legacy")
        )
        if first_contact_source == "canonical_active_entity_required":
            if self.cfg.contact_file is not None:
                raise ValueError(
                    "canonical first-contact reset timing cannot be combined "
                    "with a legacy contact_file"
                )
            if getattr(self.cfg, "reference_contact_boundary_file", None):
                raise ValueError(
                    "canonical first-contact reset timing cannot be combined "
                    "with a legacy reference_contact_boundary_file"
                )
            self._derive_first_contact_from_motion_lib()
            self._build_first_contact_lookup()
            return
        if first_contact_source != "legacy":
            raise ValueError(
                "Unsupported first_contact_reference_source "
                f"{first_contact_source!r}"
            )

        if self.cfg.contact_file is None or not os.path.exists(self.cfg.contact_file):
            # Derive first contact frames from motion_lib's in_contact labels
            # These are computed from contact_points_right_hand in the object motions pkl.
            # Always attempt this so _first_contact_lookup is available for rewards
            # (e.g. wrist_smoothness_at_contact) even without sample_before_contact.
            self._derive_first_contact_from_motion_lib()
            self._merge_reference_contact_boundaries()
            self._build_first_contact_lookup()
            return

        # Support both file and directory modes
        if os.path.isfile(self.cfg.contact_file):
            contact_data = joblib.load(self.cfg.contact_file)
        elif os.path.isdir(self.cfg.contact_file):
            # Directory mode: load all pkl files
            # Only load data if internal key EXACTLY matches filename (without .pkl)
            contact_data = {}
            pkl_files = glob.glob(os.path.join(self.cfg.contact_file, "*.pkl"))
            for pkl_file in sorted(pkl_files):
                expected_key = os.path.splitext(os.path.basename(pkl_file))[0]
                try:
                    data = joblib.load(pkl_file)
                    if expected_key in data:
                        contact_data[expected_key] = data[expected_key]
                except Exception as e:  # noqa: BLE001
                    print(f"  Warning: Failed to load {pkl_file}: {e}")  # noqa: T201
            print(  # noqa: T201
                f"[TrackingCommand] Loaded {len(contact_data)} contact sequences from {self.cfg.contact_file}"
            )
        else:
            print(  # noqa: T201
                f"[TrackingCommand] Warning: contact_file path invalid: {self.cfg.contact_file}"
            )  # noqa: T201
            return

        # Filter contact data to only include motions that exist in motion_lib
        # This respects filter_motion_keys used by motion_lib
        motion_keys_set = set(self.motion_lib.curr_motion_keys)
        filtered_contact_data = {k: v for k, v in contact_data.items() if k in motion_keys_set}
        if len(filtered_contact_data) < len(contact_data):
            print(  # noqa: T201
                f"[TrackingCommand] Filtered contact data: {len(filtered_contact_data)}/{len(contact_data)} "
                f"motions match loaded motion keys"
            )
        self._contact_data = filtered_contact_data

        # Find first frame with contact for each motion
        # Contact file structure: {motion_name: {body: (N, 10475), object: (N, obj_verts), ...}}
        first_contact_frames = {}
        for motion_name, motion_data in filtered_contact_data.items():
            object_contact = motion_data.get("object", None)
            if object_contact is not None:
                # Find first frame where any body vertex is in contact
                contact_per_frame = (object_contact != 0).sum(axis=1)
                contact_frames = np.where(contact_per_frame > 0)[0]
                if len(contact_frames) > 0:
                    first_contact_frames[motion_name] = int(contact_frames[0])
                else:
                    # No contact, use last frame
                    first_contact_frames[motion_name] = object_contact.shape[0]
            else:
                first_contact_frames[motion_name] = 0

        self._first_contact_frame = first_contact_frames
        print(f"[TrackingCommand] Loaded contact data from {self.cfg.contact_file}")  # noqa: T201
        for motion_name, first_frame in first_contact_frames.items():
            print(f"  {motion_name}: first contact at frame {first_frame}")  # noqa: T201

        # Validate and align contact data frame counts to match motion data
        self._validate_and_align_contact_frame_counts(filtered_contact_data)

        self._merge_reference_contact_boundaries()

        # Build motion_key -> first_contact_frame lookup tensor for efficient per-env updates
        self._build_first_contact_lookup()

        # Preprocess contact flags for each motion: motion_id -> (num_frames,) bool tensor
        self._build_motion_contact_flags()

    def _merge_reference_contact_boundaries(self):
        """Merge opt-in hand-or-body reset boundaries into first-contact timing.

        This affects reset sampling and temporary reset stabilization only. It
        does not replace per-hand contact labels used by grasp rewards.
        """
        path = getattr(self.cfg, "reference_contact_boundary_file", None)
        if path is None:
            return
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Reference contact boundary file not found: {path}")
        with open(path) as boundary_file:
            payload = json.load(boundary_file)
        target_fps = float(payload.get("target_fps", self.motion_lib._sim_fps))  # noqa: SLF001
        runtime_fps = float(self.motion_lib._sim_fps)  # noqa: SLF001
        if abs(target_fps - runtime_fps) > 1.0e-6:
            raise ValueError(
                "Reference contact boundary target_fps does not match motion lib: "
                f"{target_fps} vs {runtime_fps}"
            )
        boundaries = payload.get("motions", {})
        if not isinstance(boundaries, dict):
            raise TypeError("Reference contact boundary payload must contain a motions dict")
        if self._first_contact_frame is None:
            self._first_contact_frame = {}
        merged = 0
        body_advanced = 0
        for motion_key in self.motion_lib.curr_motion_keys:
            value = boundaries.get(motion_key)
            if value is None:
                continue
            boundary = value.get("first_any_contact_frame") if isinstance(value, dict) else value
            if boundary is None:
                continue
            boundary = int(boundary)
            if boundary < 0:
                raise ValueError(f"Negative contact boundary for {motion_key}: {boundary}")
            previous = self._first_contact_frame.get(motion_key)
            if previous is None or boundary < int(previous):
                self._first_contact_frame[motion_key] = boundary
                body_advanced += int(previous is not None)
            merged += 1
        print(  # noqa: T201
            f"[TrackingCommand] Merged any-body reset contact boundaries for "
            f"{merged}/{len(self.motion_lib.curr_motion_keys)} motions "
            f"({body_advanced} earlier than hand-only contact)"
        )

    def _build_first_contact_lookup(self):
        """Build a tensor for efficient per-env first contact frame lookup."""
        if self._first_contact_frame is None:
            self._first_contact_lookup = None
            return

        # Create a tensor indexed by motion_id for O(1) lookup
        num_motions = len(self.motion_lib.curr_motion_keys)
        self._first_contact_lookup = torch.zeros(num_motions, device=self.device, dtype=torch.long)

        for motion_idx, motion_key in enumerate(self.motion_lib.curr_motion_keys):
            if motion_key in self._first_contact_frame:
                self._first_contact_lookup[motion_idx] = self._first_contact_frame[motion_key]
            else:
                if (
                    getattr(
                        self.cfg,
                        "first_contact_reference_source",
                        "legacy",
                    )
                    == "canonical_active_entity_required"
                ):
                    raise KeyError(
                        f"Canonical first-contact timing is missing exact motion "
                        f"key {motion_key!r}"
                    )
                # Legacy fallback retained for historical configurations.
                self._first_contact_lookup[motion_idx] = 0

    def _build_motion_contact_flags(self):
        """Preprocess contact flags for each motion: motion_id -> (num_frames,) bool tensor."""
        if self._contact_data is None or len(self._contact_data) == 0:
            self._motion_contact_flags = None
            return

        self._motion_contact_flags = {}

        for motion_idx, motion_key in enumerate(self.motion_lib.curr_motion_keys):
            if motion_key not in self._contact_data:
                continue

            motion_contact_data = self._contact_data[motion_key]
            object_contact = motion_contact_data.get("object", None)

            if object_contact is not None:
                # Convert to torch tensor if needed
                if not isinstance(object_contact, torch.Tensor):
                    object_contact_tensor = torch.from_numpy(object_contact).to(self.device)
                else:
                    object_contact_tensor = object_contact.to(self.device)

                # Compute contact flags: (num_frames,) - True if any vertex has contact
                contact_per_frame = (object_contact_tensor != 0).sum(dim=1)  # (num_frames,)
                contact_flags = contact_per_frame > 0  # (num_frames,) bool

                self._motion_contact_flags[motion_idx] = contact_flags

    def _update_per_env_first_contact(self, env_ids):
        """Update _per_env_first_contact for given env_ids based on their assigned motion."""
        if self._first_contact_lookup is None:
            return

        # Vectorized lookup using motion_ids as indices
        motion_ids = self.motion_ids[env_ids]
        self._per_env_first_contact[env_ids] = self._first_contact_lookup[motion_ids]

    def _validate_and_align_contact_frame_counts(self, contact_data: dict):
        """Validate contact data frame counts match the loaded motion frame counts.

        Allows a tolerance of ±3 frames due to slight duration differences between
        GRAB (120Hz) and robot motion (30Hz) sources.

        Args:
            contact_data: Dict mapping motion_name -> contact arrays

        Raises:
            AssertionError: If frame count difference exceeds 3 frames
        """
        FRAME_TOLERANCE = 3

        for motion_name, motion_contact in contact_data.items():
            # Get contact frame count
            contact_frames = None
            if "object" in motion_contact and motion_contact["object"] is not None:
                contact_frames = motion_contact["object"].shape[0]
            elif "body" in motion_contact and motion_contact["body"] is not None:
                contact_frames = motion_contact["body"].shape[0]

            if contact_frames is None:
                continue

            # Find matching motion in motion_lib and get actual frame count
            motion_idx = None
            for idx, key in enumerate(self.motion_lib.curr_motion_keys):
                if key == motion_name:
                    motion_idx = idx
                    break

            if motion_idx is None:
                print(  # noqa: T201
                    f"[TrackingCommand] Warning: Contact motion '{motion_name}' "
                    f"not found in loaded motions"
                )
                continue

            # Get actual frame count from loaded motion
            motion_frames = int(
                self.motion_lib._motion_num_frames[motion_idx].item()  # noqa: SLF001
            )  # noqa: SLF001
            frame_diff = abs(contact_frames - motion_frames)

            if frame_diff == 0:
                print(f"  {motion_name}: {contact_frames} frames (exact match)")  # noqa: T201
            elif frame_diff <= FRAME_TOLERANCE:
                print(  # noqa: T201
                    f"  {motion_name}: {contact_frames} frames "
                    f"(motion lib: {motion_frames}, diff: {contact_frames - motion_frames})"
                )
            else:
                raise AssertionError(
                    f"[TrackingCommand] Frame count mismatch too large for motion '{motion_name}':\n"
                    f"  Contact data: {contact_frames} frames\n"
                    f"  Motion lib: {motion_frames} frames\n"
                    f"  Difference: {frame_diff} frames (max allowed: ±{FRAME_TOLERANCE})"
                )

    def _load_table_meta(self, motion_key: str):
        """Load and cache table meta info (table_pos, table_quat, table_size) from meta/.

        Meta pkl is the single source of truth. Requires `table_pos` and `table_quat`;
        `table_size` defaults to [1.0, 0.6, 0.04] (GRAB uses USD scene_scale instead).
        Raises FileNotFoundError / KeyError if meta is missing or incomplete.
        """
        import joblib  # noqa: PLC0415

        motion_file = (
            self.cfg.motion_lib_cfg.get("motion_file", "") if self.cfg.motion_lib_cfg else ""
        )
        if motion_file and os.path.isdir(motion_file):
            meta_dir = motion_file.replace("/robot", "/meta")
        elif motion_file and "/robot" in motion_file:
            meta_dir = os.path.dirname(motion_file).replace("/robot", "/meta")
        else:
            meta_dir = "data/motion_lib_grab/meta"
        meta_file = os.path.join(meta_dir, f"{motion_key}.pkl")

        if not os.path.exists(meta_file):
            raise FileNotFoundError(
                f"Meta pkl missing for motion '{motion_key}' at {meta_file}. "
                "table_pos/table_quat must live in meta/; no fallback is applied. "
                "If your data pipeline writes table_pos into robot pkl, migrate it to meta/."
            )

        meta = joblib.load(meta_file)
        if "table_pos" not in meta or "table_quat" not in meta:
            raise KeyError(
                f"Meta pkl {meta_file} missing required keys 'table_pos' and/or 'table_quat'. "
                f"Found keys: {list(meta.keys())}"
            )

        self._table_meta_cache[motion_key] = {
            "table_pos": torch.tensor(meta["table_pos"], device=self.device).float(),
            "table_quat": torch.tensor(meta["table_quat"], device=self.device).float(),
            "table_size": torch.tensor(
                meta.get("table_size", [1.0, 0.6, 0.04]), device=self.device
            ).float(),
        }

    def _derive_first_contact_from_motion_lib(self):
        """Derive first contact frames from motion_lib's in_contact labels.

        Uses the object motions pkl's contact_points data (loaded into motion_lib
        as _motion_object_in_contact_{left,right}) to find the earliest frame where
        either hand makes contact. When both hands have data, takes min(left, right)
        per motion.

        Called from _load_contact_data() when no separate contact_file is provided.
        """
        first_contact_source = str(
            getattr(self.cfg, "first_contact_reference_source", "legacy")
        )
        if first_contact_source == "canonical_active_entity_required":
            reference = getattr(
                self.motion_lib,
                "_motion_reference_contact_state",
                None,
            )
            if reference is None:
                raise RuntimeError(
                    "Canonical first-contact reset timing requires "
                    "reference_contact_state for every motion"
                )
            self._first_contact_frame = (
                _first_required_canonical_contact_by_motion(
                    reference,
                    self.motion_lib.length_starts,
                    self.motion_lib.curr_motion_keys,
                )
            )
            print(  # noqa: T201
                "[TrackingCommand] Derived first contact from canonical "
                "active-entity required labels "
                f"({len(self._first_contact_frame)} motions)"
            )
            return
        if first_contact_source != "legacy":
            raise ValueError(
                "Unsupported first_contact_reference_source "
                f"{first_contact_source!r}"
            )

        contact_left = getattr(self.motion_lib, "_motion_object_in_contact_left", None)
        contact_right = getattr(self.motion_lib, "_motion_object_in_contact_right", None)

        if contact_left is None and contact_right is None:
            if self.cfg.sample_before_contact and not getattr(
                self.cfg, "reference_contact_boundary_file", None
            ):
                raise RuntimeError(
                    "sample_before_contact=True but no contact data available. "
                    "The object motions pkl must contain contact_points_left_hand "
                    "or contact_points_right_hand, or provide a contact_file or "
                    "reference_contact_boundary_file path."
                )
            self._first_contact_frame = {}
            return

        hands_used = []
        if contact_left is not None:
            hands_used.append("left")
        if contact_right is not None:
            hands_used.append("right")

        num_motions = len(self.motion_lib.curr_motion_keys)

        def _first_contact_for_tensor(contact_tensor):
            """Return dict of motion_key -> first contact frame for a single hand."""
            last_start = int(self.motion_lib.length_starts[-1]) if num_motions > 0 else 0
            if len(contact_tensor) < last_start:
                raise RuntimeError(
                    f"contact tensor length ({len(contact_tensor)}) is shorter "
                    f"than expected from length_starts ({last_start}). Some motions "
                    f"may be missing contact data in the object motions pkl."
                )
            result = {}
            for motion_idx, motion_key in enumerate(self.motion_lib.curr_motion_keys):
                start = int(self.motion_lib.length_starts[motion_idx])
                if motion_idx + 1 < num_motions:
                    end = int(self.motion_lib.length_starts[motion_idx + 1])
                else:
                    end = len(contact_tensor)
                num_frames = end - start
                contact_frames = torch.where(contact_tensor[start:end] > 0.5)[0]
                if len(contact_frames) > 0:
                    result[motion_key] = contact_frames[0].item()
                else:
                    result[motion_key] = num_frames
            return result

        # Compute per-hand first contact, then take min across available hands
        per_hand = {}
        if contact_left is not None:
            per_hand["left"] = _first_contact_for_tensor(contact_left)
        if contact_right is not None:
            per_hand["right"] = _first_contact_for_tensor(contact_right)

        first_contact_frames = {}
        for motion_key in self.motion_lib.curr_motion_keys:
            first_contact_frames[motion_key] = min(
                h[motion_key] for h in per_hand.values()
            )

        self._first_contact_frame = first_contact_frames
        hands_str = "+".join(f"{h}_hand" for h in hands_used)
        print(  # noqa: T201
            f"[TrackingCommand] Derived first contact frames from motion_lib "
            f"({hands_str}, {len(first_contact_frames)} motions)"
        )
        # Diagnostic: print per-motion details
        for motion_key, fc_frame in first_contact_frames.items():
            motion_idx = self.motion_lib.curr_motion_keys.index(motion_key)
            start = int(self.motion_lib.length_starts[motion_idx])
            if motion_idx + 1 < num_motions:
                end = int(self.motion_lib.length_starts[motion_idx + 1])
            else:
                contact_tensor = contact_left if contact_left is not None else contact_right
                end = len(contact_tensor)
            num_frames = end - start
            print(f"  {motion_key[:60]}: first_contact={fc_frame}/{num_frames} frames")  # noqa: T201

        # Print summary stats
        frames = list(first_contact_frames.values())
        if frames:
            import numpy as np
            arr = np.array(frames)
            print(  # noqa: T201
                f"  First contact: min={arr.min()}, max={arr.max()}, "
                f"mean={arr.mean():.0f}, median={np.median(arr):.0f}"
            )

        # Build lookup tensor and contact flags (same as _load_contact_data)
        self._build_first_contact_lookup()

    def _load_reset_start_cache(self) -> None:
        """Load optional per-motion reset ranges in runtime-frame coordinates."""
        self._reset_start_cache = None
        self._reset_start_range_table = None
        path = getattr(self.cfg, "reset_start_cache_file", None)
        if not path:
            return

        target_fps = float(self.motion_lib._sim_fps)  # noqa: SLF001
        required_level = str(
            getattr(self.cfg, "reset_start_cache_required_validation", "reference")
        )
        require_all = bool(getattr(self.cfg, "reset_start_cache_require_all", False))
        cache = load_reset_start_cache(
            path,
            motion_keys=self.motion_lib.curr_motion_keys,
            target_fps=target_fps,
            required_validation_level=required_level,
            require_all=require_all,
        )

        truncate_time = self.cfg.sample_start_truncate_time_s
        runtime_window = (
            None
            if truncate_time is None
            else int(round(float(truncate_time) * target_fps))
        )
        if cache.episode_window_frames != runtime_window:
            raise ValueError(
                "Reset-start cache episode window does not match runtime: "
                f"{cache.episode_window_frames} vs {runtime_window} frames"
            )

        num_motions = len(self.motion_lib.curr_motion_keys)
        motion_ids = torch.arange(num_motions, device=self.device, dtype=torch.long)
        runtime_steps = self.motion_lib.get_motion_num_steps(motion_ids)
        if runtime_window is None:
            valid_upper = runtime_steps
        else:
            valid_seconds = torch.clamp(
                self.motion_lib.get_motion_length(motion_ids) - float(truncate_time),
                min=0.0,
            )
            valid_upper = torch.ceil(valid_seconds * target_fps).long().clamp_min(1)

        max_ranges = max(
            (len(record.start_ranges) for record in cache.records if record is not None),
            default=0,
        )
        table = torch.full(
            (num_motions, max(max_ranges, 1), 2),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        loaded = 0
        for motion_idx, record in enumerate(cache.records):
            if record is None:
                continue
            current_steps = int(runtime_steps[motion_idx].item())
            if record.runtime_num_steps != current_steps:
                raise ValueError(
                    f"Reset-start cache is stale for {record.motion_key}: "
                    f"runtime_num_steps={record.runtime_num_steps}, current={current_steps}"
                )
            upper = int(valid_upper[motion_idx].item())
            if record.start_ranges[-1][1] > upper:
                raise ValueError(
                    f"Reset-start cache range for {record.motion_key} ends at "
                    f"{record.start_ranges[-1][1]}, beyond valid upper bound {upper}"
                )
            for range_idx, (start, end) in enumerate(record.start_ranges):
                table[motion_idx, range_idx] = torch.tensor(
                    (start, end), dtype=torch.long, device=self.device
                )
            loaded += 1

        self._reset_start_cache = cache
        self._reset_start_range_table = table
        print(  # noqa: T201
            f"[TrackingCommand] Reset-start cache loaded for {loaded}/{num_motions} "
            f"motions ({required_level} validation, {cache.path})"
        )

    def _sample_from_reset_start_cache(
        self, env_ids: Sequence[int], sampled_times: torch.Tensor
    ) -> torch.Tensor:
        """Replace fallback samples with a uniform draw over cached valid frames."""
        table = self._reset_start_range_table
        if table is None or len(env_ids) == 0:
            return sampled_times
        if self.is_evaluating and not bool(
            getattr(self.cfg, "reset_start_cache_apply_in_evaluation", True)
        ):
            return sampled_times

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        ranges = table[self.motion_ids[env_ids_t]]
        lengths = (ranges[..., 1] - ranges[..., 0]).clamp_min(0)
        total = lengths.sum(dim=1)
        has_range = total > 0
        if bool(getattr(self.cfg, "reset_start_cache_require_all", False)) and not bool(
            has_range.all()
        ):
            missing_ids = self.motion_ids[env_ids_t[~has_range]].detach().cpu().tolist()
            missing_keys = [self.motion_lib.curr_motion_keys[idx] for idx in missing_ids[:8]]
            raise RuntimeError(
                f"Reset-start cache has no qualified range for {len(missing_ids)} "
                f"assigned environments; first={missing_keys}"
            )
        if not bool(has_range.any()):
            return sampled_times

        draw = torch.floor(torch.rand(len(env_ids_t), device=self.device) * total.clamp_min(1)).long()
        cumulative = lengths.cumsum(dim=1)
        range_index = (draw[:, None] >= cumulative).sum(dim=1).clamp_max(ranges.shape[1] - 1)
        previous = torch.where(
            range_index > 0,
            cumulative.gather(1, (range_index - 1).clamp_min(0)[:, None]).squeeze(1),
            torch.zeros_like(draw),
        )
        selected_start = ranges[..., 0].gather(1, range_index[:, None]).squeeze(1)
        cached_times = selected_start + draw - previous
        return torch.where(has_range, cached_times.to(sampled_times.dtype), sampled_times)

    def _valid_start_upper_bounds(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Return exclusive upper bounds for valid episode start steps."""
        motion_ids = self.motion_ids[env_ids]
        truncate_time = self.cfg.sample_start_truncate_time_s
        if truncate_time is None:
            upper = self.motion_lib.get_motion_num_steps(motion_ids)
        else:
            valid_seconds = torch.clamp(
                self.motion_lib.get_motion_length(motion_ids) - float(truncate_time),
                min=0.0,
            )
            upper = torch.ceil(
                valid_seconds * float(self.motion_lib._sim_fps)  # noqa: SLF001
            ).long()
        return upper.clamp_min(1)

    def _enforce_valid_start_window(
        self, env_ids: Sequence[int], sampled_times: torch.Tensor
    ) -> torch.Tensor:
        """Resample starts outside the configured full-episode window."""
        if self.cfg.sample_start_truncate_time_s is None:
            return sampled_times
        valid_upper = self._valid_start_upper_bounds(env_ids)
        invalid = (sampled_times < 0) | (sampled_times >= valid_upper)
        if torch.any(invalid):
            replacement = torch.floor(
                torch.rand(len(env_ids), device=self.device) * valid_upper.float()
            ).to(sampled_times.dtype)
            sampled_times = torch.where(invalid, replacement, sampled_times)
        return sampled_times

    def _load_ownership_phase_table(self) -> None:
        """Load per-motion ownership phases into a padded device tensor.

        Reads rp2_ownership_phases from <ownership_meta_dir>/<motion>.pkl for
        every loaded motion. Motions without phases get zero rows and sample
        unconstrained (full-motion episodes), preserving behavior for
        single-actor data.
        """
        import joblib  # noqa: PLC0415

        keys = list(self.motion_lib.curr_motion_keys)
        min_len = int(self.cfg.ownership_min_episode_frames)
        per_motion: list[list[tuple[int, int]]] = []
        found = 0
        for key in keys:
            phases: list[tuple[int, int]] = []
            meta_path = os.path.join(self.cfg.ownership_meta_dir, f"{key}.pkl")
            if os.path.isfile(meta_path):
                try:
                    payload = joblib.load(meta_path)
                    meta = payload.get(key, payload) if isinstance(payload, dict) else {}
                    raw = meta.get("rp2_ownership_phases") or []
                    phases = [
                        (int(s), int(e)) for s, e in raw if int(e) - int(s) >= min_len
                    ]
                except Exception as exc:  # noqa: BLE001
                    print(  # noqa: T201
                        f"[TrackingCommand] ownership meta unreadable for {key}: {exc}"
                    )
            if phases:
                found += 1
            per_motion.append(phases)
        max_phases = max((len(p) for p in per_motion), default=0)
        table = torch.full(
            (len(keys), max(max_phases, 1), 2), -1, dtype=torch.long, device=self.device
        )
        for m, phases in enumerate(per_motion):
            for j, (s, e) in enumerate(phases):
                table[m, j, 0] = s
                table[m, j, 1] = e
        self._ownership_phase_table = table
        print(  # noqa: T201
            f"[TrackingCommand] ownership phases loaded for {found}/{len(keys)} motions "
            f"(max {max_phases} phases; min episode {min_len} frames)"
        )

    def _constrain_to_ownership_phase(
        self, env_ids: Sequence[int], sampled_times: torch.Tensor
    ) -> torch.Tensor:
        """Snap sampled starts into one ownership phase and record its end.

        Envs whose motion has no phases keep their sample and a phase end of
        -1 (no clamp). Phase choice is uniform over valid phases; the start is
        uniform within [start, end - min_episode_frames].
        """
        if not getattr(self.cfg, "ownership_phase_episodes", False):
            return sampled_times
        if getattr(self, "_ownership_phase_table", None) is None:
            self._load_ownership_phase_table()
        if not hasattr(self, "_ownership_phase_end"):
            self._ownership_phase_end = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
        min_len = int(self.cfg.ownership_min_episode_frames)
        table = self._ownership_phase_table
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        motions = self.motion_ids[env_ids_t]
        phases = table[motions]  # (E, K, 2)
        valid = (phases[..., 0] >= 0) & ((phases[..., 1] - phases[..., 0]) >= min_len)
        has_phase = valid.any(dim=1)
        self._ownership_phase_end[env_ids_t] = -1
        if not bool(has_phase.any()):
            return sampled_times
        # Uniform choice among valid phases per env.
        weights = valid.float() + 1e-9
        choice = torch.multinomial(weights, 1).squeeze(1)
        sel = phases[torch.arange(len(env_ids_t), device=self.device), choice]
        span = (sel[:, 1] - sel[:, 0] - min_len).clamp(min=1)
        starts = sel[:, 0] + (
            torch.rand(len(env_ids_t), device=self.device) * span.float()
        ).long()
        sampled_times = torch.where(has_phase, starts.to(sampled_times.dtype), sampled_times)
        self._ownership_phase_end[env_ids_t] = torch.where(
            has_phase, sel[:, 1], torch.full_like(sel[:, 1], -1)
        )
        return sampled_times

    def _apply_ownership_phase_end(self, env_ids: Sequence[int]) -> None:
        """Clamp episode length so it ends at the chosen phase's end."""
        if not getattr(self.cfg, "ownership_phase_episodes", False):
            return
        if not hasattr(self, "_ownership_phase_end"):
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        num_steps = self.motion_num_steps[env_ids_t]
        phase_end = self._ownership_phase_end[env_ids_t].to(num_steps.dtype)
        clamped = torch.where(
            phase_end > 0,
            torch.minimum(num_steps, phase_end),
            num_steps,
        )
        self.motion_num_steps[env_ids_t] = clamped.to(self.motion_num_steps.dtype)

    def _sample_before_contact_legacy_mode(self) -> bool:
        """Legacy pre-contact window (default) vs the ContactFirst windowed mixture."""
        return (
            str(getattr(self.cfg, "sample_before_contact_window_mode", "legacy") or "legacy")
            == "legacy"
        )

    def _sample_before_contact(
        self, env_ids: Sequence[int], sampled_times: torch.Tensor
    ) -> torch.Tensor:
        """Bias a configurable fraction of valid reset starts toward pre-contact frames.

        Args:
            env_ids: Environment indices to resample
            sampled_times: Originally sampled time steps

        Returns:
            Valid full-window samples with a configurable pre-contact bias.
        """
        if self._first_contact_frame is None or len(self._first_contact_frame) == 0:
            return sampled_times
        if self._sample_before_contact_legacy_mode():
            # Legacy window (released task-general tracking): override every start
            # with a uniform sample from [0, first_contact - margin).
            curr_motion_keys = getattr(self.motion_lib, "curr_motion_keys", None)
            for i, env_idx in enumerate(env_ids):
                motion_id = self.motion_ids[env_idx].item()
                if curr_motion_keys is not None and motion_id < len(curr_motion_keys):
                    motion_key = curr_motion_keys[motion_id]
                else:
                    motion_key = list(self._first_contact_frame.keys())[0]  # noqa: RUF015
                first_contact = None
                if motion_key in self._first_contact_frame:
                    first_contact = self._first_contact_frame[motion_key]
                else:
                    for contact_key in self._first_contact_frame.keys():  # noqa: SIM118
                        if contact_key in motion_key or motion_key in contact_key:
                            first_contact = self._first_contact_frame[contact_key]
                            break
                    if first_contact is None:
                        first_contact = list(self._first_contact_frame.values())[0]  # noqa: RUF015
                margin = getattr(self.cfg, "sample_before_contact_margin", 10)
                if first_contact > margin:
                    sampled_times[i] = torch.randint(
                        0, first_contact - margin, (1,), device=self.device, dtype=sampled_times.dtype
                    )
                else:
                    sampled_times[i] = 0
            return sampled_times
        bias_probability = float(
            getattr(self.cfg, "sample_before_contact_bias_probability", 1.0)
        )
        bias_probability = min(max(bias_probability, 0.0), 1.0)
        if bias_probability <= 0.0:
            return sampled_times
        biased = torch.rand(len(env_ids), device=self.device) < bias_probability

        truncate_time = self.cfg.sample_start_truncate_time_s
        motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids[env_ids])
        valid_upper = self._valid_start_upper_bounds(env_ids)
        if truncate_time is None:
            episode_window_steps = None
        else:
            episode_window_steps = max(
                int(
                    round(
                        float(truncate_time)
                        * float(self.motion_lib._sim_fps)  # noqa: SLF001
                    )
                ),
                1,
            )
        # Get motion keys from motion library
        curr_motion_keys = getattr(self.motion_lib, "curr_motion_keys", None)

        for i, env_idx in enumerate(env_ids):
            if not bool(biased[i]):
                continue
            motion_id = self.motion_ids[env_idx].item()

            # Get motion key for this motion_id
            if curr_motion_keys is not None and motion_id < len(curr_motion_keys):
                motion_key = curr_motion_keys[motion_id]
            else:
                # Fallback: use first contact key
                motion_key = list(self._first_contact_frame.keys())[0]  # noqa: RUF015
            first_contact = self._first_contact_frame.get(motion_key)
            if first_contact is None:
                if (
                    getattr(
                        self.cfg,
                        "first_contact_reference_source",
                        "legacy",
                    )
                    == "canonical_active_entity_required"
                ):
                    raise KeyError(
                        f"Canonical first-contact timing is missing exact motion "
                        f"key {motion_key!r}"
                    )
                for contact_key in self._first_contact_frame.keys():  # noqa: SIM118
                    if contact_key in motion_key or motion_key in contact_key:
                        first_contact = self._first_contact_frame[contact_key]
                        break
                if first_contact is None:
                    first_contact = list(self._first_contact_frame.values())[0]  # noqa: RUF015

            # A sentinel at or beyond the motion end means that this motion has
            # no labeled contact. Keep its original uniform sample.
            if int(first_contact) >= int(motion_num_steps[i].item()):
                continue

            # Redraw only from starts for which contact occurs inside the
            # episode, with at least `margin` reference steps of lead-in. This
            # remains a mixture with the untouched uniform sample above, so the
            # complete valid interval retains nonzero probability.
            margin_seconds = getattr(
                self.cfg, "sample_before_contact_margin_seconds", None
            )
            if margin_seconds is None:
                margin = int(getattr(self.cfg, "sample_before_contact_margin", 10))
            else:
                margin = duration_seconds_to_frames(
                    float(margin_seconds),
                    float(self.motion_lib._sim_fps),  # noqa: SLF001
                )
            contact_step = int(first_contact)
            lower = 0
            if episode_window_steps is not None:
                lower = max(contact_step - episode_window_steps, 0)
            upper = min(
                contact_step - int(margin) + 1,
                int(valid_upper[i].item()),
            )
            if upper <= lower:
                continue

            # When a validated reset cache is active, sample only from the
            # intersection of its ranges and the pre-contact interval. The
            # cache is otherwise applied after contact bias and would silently
            # erase the configured mixture.
            if self._reset_start_range_table is not None:
                ranges = self._reset_start_range_table[motion_id]
                starts = torch.maximum(
                    ranges[:, 0],
                    torch.full_like(ranges[:, 0], lower),
                )
                ends = torch.minimum(
                    ranges[:, 1],
                    torch.full_like(ranges[:, 1], upper),
                )
                lengths = (ends - starts).clamp_min(0)
                total = int(lengths.sum().item())
                if total <= 0:
                    continue
                draw = torch.randint(0, total, (1,), device=self.device)
                cumulative = lengths.cumsum(dim=0)
                range_index = int((draw >= cumulative).sum().item())
                previous = (
                    int(cumulative[range_index - 1].item())
                    if range_index > 0
                    else 0
                )
                sampled_times[i] = starts[range_index] + draw[0] - previous
            else:
                sampled_times[i] = torch.randint(
                    lower,
                    upper,
                    (1,),
                    device=self.device,
                    dtype=sampled_times.dtype,
                )

        return sampled_times

    def _sample_start_time_steps_for_current_motions(
        self, env_ids: Sequence[int]
    ) -> torch.Tensor:
        """Sample reset start times for the already-assigned motion IDs."""
        adaptive_paired_starts = self.use_adaptive_sampling and bool(
            self.motion_lib.adaptive_sampling_cfg.get(
                "paired_motion_time_sampling", False
            )
        )
        if adaptive_paired_starts:
            sampled_times = self.motion_lib.sample_time_steps_for_motion_ids_adaptive(
                self.motion_ids[env_ids],
                truncate_time=self.cfg.sample_start_truncate_time_s,
            )
        else:
            sampled_times = self.motion_lib.sample_time_steps(
                self.motion_ids[env_ids],
                truncate_time=self.cfg.sample_start_truncate_time_s,
            )

        if self.cfg.sample_from_n_initial_frames is not None:
            n_frames = self.cfg.sample_from_n_initial_frames
            sampled_times = torch.randint(
                0, n_frames, (len(env_ids),), dtype=sampled_times.dtype, device=self.device
            )
        elif self.cfg.start_from_first_frame:
            sampled_times.zero_()

        sampled_times = self._enforce_valid_start_window(env_ids, sampled_times)
        sampled_times = self._constrain_to_ownership_phase(env_ids, sampled_times)

        return sampled_times

    def _sample_asset_compatible_motion_ids(
        self,
        env_ids: Sequence[int],
        *,
        evaluation: bool,
    ) -> torch.Tensor:
        """Sample motions that use each environment's immutable USD assets."""
        env_ids_t = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.tensor(env_ids, dtype=torch.long, device=self.device)
        )
        env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long)
        if evaluation:
            return self._asset_compatible_representative_motion_ids[env_ids_t]

        sampled = torch.empty(
            len(env_ids_t),
            dtype=torch.long,
            device=self.device,
        )
        group_ids = self._asset_compatible_env_group_ids[env_ids_t]
        for group_id in group_ids.unique():
            assignment = group_ids == group_id
            candidates = self._asset_compatible_motion_groups[
                int(group_id.item())
            ]
            sampled[assignment] = self.motion_lib.sample_motions_from_candidates(
                candidates,
                int(assignment.sum().item()),
            )
        return sampled

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample motion clips, reset robot state, and position objects for given envs.

        This is the main episode-reset handler. It performs the following in order:

        1. Sample new motion IDs and start times (respecting evaluation mode,
           paired motions, multi-object mode, and adaptive sampling).
        2. Resample encoder mode (G1/SMPL/teleop) per env.
        3. Apply pose and velocity randomization (skipped during evaluation).
        4. Handle DOF mismatch by mapping motion lib joints to robot joints.
        5. Write joint state and root state to the simulator.
        6. Position scene objects (single or multi-object mode) and tables.
        7. Cache body-relative poses for reward computation.

        Args:
            env_ids: Environment indices being reset.
        """
        if getattr(self, "_adaptive_terminal_only", False) and len(env_ids) > 0:
            # Pre-reset attribution: capture the finished episode before the
            # buffers below are overwritten with the next one.
            self._snapshot_completed_episodes(env_ids)
        self.time_steps[env_ids] = 0
        # Variable frames: resample per-env num_frames at episode reset
        if self.variable_frames_enabled and len(env_ids) > 0:
            idx = torch.randint(0, len(self._frame_choices), (len(env_ids),), device=self.device)
            self.per_env_num_frames[env_ids] = self._frame_choices[idx]
        if len(env_ids) > 0:
            if self.use_asset_compatible_motions:
                self.motion_ids[env_ids] = self._sample_asset_compatible_motion_ids(
                    env_ids,
                    evaluation=self.is_evaluating,
                )
                if self.is_evaluating:
                    self.motion_start_time_steps[env_ids] = 0
                else:
                    self.motion_start_time_steps[env_ids] = (
                        self._sample_start_time_steps_for_current_motions(env_ids)
                    )
            elif self.is_evaluating and self._terrain_aware:
                # Eval + terrain-aware: respect flat/terrain split
                env_ids_t = (
                    env_ids
                    if isinstance(env_ids, torch.Tensor)
                    else torch.tensor(env_ids, device=self.device)
                )
                flat_mask = self._env_is_flat[env_ids_t]
                flat_envs = env_ids_t[flat_mask]
                terrain_envs = env_ids_t[~flat_mask]
                # Flat envs: sequential from flat range
                if len(flat_envs) > 0:
                    flat_sequential = (
                        flat_envs % max(self.motion_lib._num_motions - self._n_terrain_motions, 1)
                    ) + self._n_terrain_motions
                    self.motion_ids[flat_envs] = flat_sequential
                # Terrain envs: keep fixed motion_id (bound at init)
                self.motion_start_time_steps[env_ids] = 0
            elif self.is_evaluating:
                self.motion_ids[env_ids] = (
                    torch.arange(self.num_envs).to(self.device)
                    % self.motion_lib._num_motions  # noqa: SLF001
                )[env_ids]
                self.motion_start_time_steps[env_ids] = 0
                if (
                    self.cfg.evaluation_start_before_contact_frames is not None
                    and self._first_contact_lookup is not None
                ):
                    first_contact = self._first_contact_lookup[self.motion_ids[env_ids]]
                    motion_steps = self.motion_lib.get_time_step_total(
                        self.motion_ids[env_ids]
                    )
                    has_contact = first_contact < motion_steps
                    contact_centered_start = (
                        first_contact
                        - int(self.cfg.evaluation_start_before_contact_frames)
                    ).clamp(min=0).to(self.motion_start_time_steps.dtype)
                    self.motion_start_time_steps[env_ids] = torch.where(
                        has_contact,
                        contact_centered_start,
                        self.motion_start_time_steps[env_ids],
                    )
            elif self.cfg.use_paired_motions:
                self.motion_ids[env_ids] = (
                    torch.arange(self.num_envs).to(self.device)
                    % self.motion_lib._num_motions  # noqa: SLF001
                )[env_ids]
                self.motion_start_time_steps[env_ids] = (
                    self._sample_start_time_steps_for_current_motions(env_ids)
                )

            elif self._multi_object_mode:
                # MULTI-OBJECT MODE: Resetting envs sample a new motion (and corresponding object)
                new_motion_id = self.motion_lib.sample_motions(1)[0]
                self.motion_ids[env_ids] = new_motion_id
                self.motion_start_time_steps[env_ids] = self.motion_lib.sample_time_steps(
                    self.motion_ids[env_ids],
                    truncate_time=self.cfg.sample_start_truncate_time_s,
                )
            elif self._terrain_aware:
                # TERRAIN-AWARE SAMPLING: flat envs resample motions, terrain envs keep fixed motion
                env_ids_t = (
                    env_ids
                    if isinstance(env_ids, torch.Tensor)
                    else torch.tensor(env_ids, device=self.device)
                )
                flat_mask = self._env_is_flat[env_ids_t]
                flat_envs = env_ids_t[flat_mask]
                terrain_envs = env_ids_t[~flat_mask]

                # Flat envs: resample motions from flat range only [n_terrain, num_motions)
                if len(flat_envs) > 0:
                    if (
                        self._n_terrain_motions > 0
                        and self._n_terrain_motions < self.motion_lib._num_motions
                    ):
                        flat_ids = torch.randint(
                            self._n_terrain_motions,
                            self.motion_lib._num_motions,
                            (len(flat_envs),),
                            device=self.device,
                        )
                        self.motion_ids[flat_envs] = flat_ids
                        self.motion_start_time_steps[flat_envs] = self.motion_lib.sample_time_steps(
                            self.motion_ids[flat_envs],
                            truncate_time=self.cfg.sample_start_truncate_time_s,
                        )
                    else:
                        if self.use_adaptive_sampling:
                            ids, times = self.motion_lib.sample_motion_ids_and_time_steps(
                                len(flat_envs)
                            )
                            self.motion_ids[flat_envs] = ids.to(self.motion_ids.dtype)
                            self.motion_start_time_steps[flat_envs] = times.to(
                                self.motion_start_time_steps.dtype
                            )
                        else:
                            self.motion_ids[flat_envs] = self.motion_lib.sample_motions(
                                len(flat_envs)
                            )
                            self.motion_start_time_steps[flat_envs] = (
                                self.motion_lib.sample_time_steps(
                                    self.motion_ids[flat_envs],
                                    truncate_time=self.cfg.sample_start_truncate_time_s,
                                )
                            )

                # Terrain envs: keep motion_id fixed, only resample start time
                if len(terrain_envs) > 0:
                    self.motion_start_time_steps[terrain_envs] = self.motion_lib.sample_time_steps(
                        self.motion_ids[terrain_envs],
                        truncate_time=self.cfg.sample_start_truncate_time_s,
                    )

                # Build combined sampled_times for downstream overrides
                sampled_times = self.motion_start_time_steps[env_ids_t].clone()
                sampled_times = self._enforce_valid_start_window(env_ids_t, sampled_times)
                self.motion_start_time_steps[env_ids_t] = sampled_times
            else:
                if self.use_adaptive_sampling:
                    sampled_ids, sampled_times = self.motion_lib.sample_motion_ids_and_time_steps(
                        len(env_ids)
                    )
                    self.motion_ids[env_ids] = sampled_ids.to(self.motion_ids.dtype)
                    sampled_times = sampled_times.to(self.motion_start_time_steps.dtype)
                else:
                    self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
                    sampled_times = self.motion_lib.sample_time_steps(
                        self.motion_ids[env_ids],
                        truncate_time=self.cfg.sample_start_truncate_time_s,
                    )

                # Override to sample from initial frames if configured
                if self.cfg.sample_from_n_initial_frames is not None:
                    # Sample uniformly from first N frames
                    n_frames = self.cfg.sample_from_n_initial_frames
                    sampled_times = torch.randint(
                        0, n_frames, (len(env_ids),), dtype=sampled_times.dtype, device=self.device
                    )
                elif self.cfg.start_from_first_frame:
                    sampled_times.zero_()

                sampled_times = self._enforce_valid_start_window(env_ids, sampled_times)
                self.motion_start_time_steps[env_ids] = sampled_times

            # Apply the same cache contract after every reset branch. This is
            # intentionally last: cached ranges supersede legacy uniform,
            # contact-biased, terrain, adaptive, and evaluation start choices.
            self.motion_start_time_steps[env_ids] = self._sample_from_reset_start_cache(
                env_ids, self.motion_start_time_steps[env_ids]
            )
            if (
                self.cfg.sample_before_contact
                and self._first_contact_frame is not None
                and not self.is_evaluating
            ):
                self.motion_start_time_steps[env_ids] = self._sample_before_contact(
                    env_ids, self.motion_start_time_steps[env_ids]
                )

            if self.encoder_sample_probs is not None:
                has_smpl = self.motion_lib.motion_has_smpl[self.motion_ids[env_ids]]
                if self.soma_encoder_index is not None and hasattr(
                    self.motion_lib, "motion_has_soma"
                ):
                    has_soma = self.motion_lib.motion_has_soma[self.motion_ids[env_ids]]
                    sampling_cases = [
                        (env_ids[has_smpl & has_soma], self.encoder_sample_probs),
                        (env_ids[has_smpl & ~has_soma], self.encoder_sample_probs_no_soma),
                        (env_ids[~has_smpl & has_soma], self.encoder_sample_probs_no_smpl),
                        (
                            env_ids[~has_smpl & ~has_soma],
                            self.encoder_sample_probs_no_smpl_no_soma,
                        ),
                    ]
                else:
                    sampling_cases = [
                        (env_ids[has_smpl], self.encoder_sample_probs),
                        (env_ids[~has_smpl], self.encoder_sample_probs_no_smpl),
                    ]
                for subset_ids, probs in sampling_cases:
                    if len(subset_ids) > 0:
                        encoder_index = torch.multinomial(
                            probs, len(subset_ids), replacement=True
                        ).to(self.device)
                        self.encoder_index[subset_ids] = 0
                        self.encoder_index[subset_ids, encoder_index] = 1

                # =============================================================
                # Legacy behavior: SMPL-native envs also activate G1 encoder
                # This causes G1 tokens to be computed (then overwritten by SMPL)
                # in the main encoding loop, enabling G1-SMPL latent alignment.
                #
                # When optimize_encoders_ratio_for_CHIP=True:
                #   - Skip this OR logic (cleaner native encoder selection)
                #   - G1 encoder only runs for G1-native envs in main loop
                #   - G1 latents for SMPL-native are computed separately in
                #     aux losses, ONLY when compliance=0 (stiff mode)
                # =============================================================
                if (
                    self.smpl_encoder_index is not None
                    and not self.optimize_encoders_ratio_for_CHIP
                ):
                    use_smpl = self.encoder_index[env_ids, self.smpl_encoder_index]
                    self.encoder_index[env_ids, self.g1_encoder_index] = (
                        self.encoder_index[env_ids, self.g1_encoder_index] | use_smpl
                    )
                    # Also sample teleop mode when smpl mode is active (for latent alignment)
                    if (
                        self.teleop_encoder_index is not None
                        and self.teleop_sample_prob_when_smpl > 0.0
                    ):
                        smpl_env_ids = env_ids[use_smpl.bool()]
                        if len(smpl_env_ids) > 0:
                            sample_teleop = (
                                torch.rand(len(smpl_env_ids), device=self.device)
                                < self.teleop_sample_prob_when_smpl
                            )
                            self.encoder_index[smpl_env_ids, self.teleop_encoder_index] = (
                                self.encoder_index[smpl_env_ids, self.teleop_encoder_index]
                                | sample_teleop.long()
                            )

                # When soma is sampled, also activate g1 (for g1-soma latent alignment)
                if (
                    self.soma_encoder_index is not None
                    and not self.optimize_encoders_ratio_for_CHIP
                ):
                    use_soma = self.encoder_index[env_ids, self.soma_encoder_index]
                    self.encoder_index[env_ids, self.g1_encoder_index] = (
                        self.encoder_index[env_ids, self.g1_encoder_index] | use_soma
                    )

            self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
                self.motion_ids[env_ids]
            )
            if self.num_future_frames > 1:
                self.future_motion_ids = self.motion_ids.repeat_interleave(self.num_future_frames)
                self.smpl_future_motion_ids = self.motion_ids.repeat_interleave(
                    self.smpl_num_future_frames
                )
                self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
                    self.motion_ids[env_ids]
                )

            self._refresh_object_metadata_cache(env_ids)
            self._apply_target_object_mass_overrides(env_ids)
            self._refresh_joint_roster_runtime_mass_metric(env_ids)

            # Update per-env first contact frame based on assigned motion
            if self._first_contact_frame is not None:
                self._update_per_env_first_contact(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        self.running_ref_root_height[env_ids] = self.anchor_pos_w[env_ids, 2]

        # Skip reset randomizations during evaluation — they cause visible stumbling
        # at the start of rendered episodes and are only needed for training robustness.
        if not self.is_evaluating:
            range_list = [
                self.cfg.pose_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
            )
            root_pos[env_ids] += rand_samples[:, 0:3]
            if self.cfg.init_z_offset != 0.0:
                root_pos[env_ids, 2] += self.cfg.init_z_offset
            orientations_delta = quat_from_euler_xyz(
                rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
            )
            root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
            range_list = [
                self.cfg.velocity_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
            )
            root_lin_vel[env_ids] += rand_samples[:, :3]
            root_ang_vel[env_ids] += rand_samples[:, 3:]

        # Handle DOF mismatch between motion library and robot
        motion_lib_joint_pos = self.joint_pos.clone()  # Shape: [num_envs, motion_lib_num_dof]
        motion_lib_joint_vel = self.joint_vel.clone()  # Shape: [num_envs, motion_lib_num_dof]

        if self.has_dof_mismatch:
            # Create full robot joint tensors and map using name-based indices.
            # Extra joints (typically hands) can use separate reset-noise ranges so
            # body randomization doesn't overwhelm short-stroke grippers like YAM.
            joint_pos = torch.zeros(
                self.num_envs, self.robot_num_dof, dtype=torch.float32, device=self.device
            )
            joint_vel = torch.zeros(
                self.num_envs, self.robot_num_dof, dtype=torch.float32, device=self.device
            )
            joint_pos[:, self.body_joint_indices] = motion_lib_joint_pos
            joint_vel[:, self.body_joint_indices] = motion_lib_joint_vel
            # Extra (hand) joints: exact reference dof (name-mapped) if configured+available,
            # else the zero/default hand pose.
            ref_hand_dof = None
            if getattr(self, "reference_hand_dof_indices", None) is not None:
                # all-env fetch (num_envs, 14) to match joint_pos[:, ...]; only env_ids rows
                # are applied downstream, same as the body block above.
                ref_hand_dof = self.motion_lib.get_hand_dof_pos(
                    self.motion_ids,
                    self.motion_start_time_steps + self.time_steps,
                )
            if ref_hand_dof is not None:
                joint_pos[:, self.reference_hand_dof_indices] = ref_hand_dof.to(joint_pos.dtype)
            else:
                joint_pos[:, self.extra_joint_indices] = self.extra_default_positions
            joint_vel[:, self.extra_joint_indices] = self.extra_default_velocities

            if not self.is_evaluating:
                joint_pos[:, self.body_joint_indices] += sample_uniform(
                    *self.cfg.joint_position_range,
                    joint_pos[:, self.body_joint_indices].shape,
                    joint_pos.device,
                )
                joint_vel[:, self.body_joint_indices] += sample_uniform(
                    *self.cfg.joint_velocity_range,
                    joint_vel[:, self.body_joint_indices].shape,
                    joint_vel.device,
                )

                hand_joint_position_range = self.cfg.hand_joint_position_range
                if hand_joint_position_range is None:
                    hand_joint_position_range = self.cfg.joint_position_range
                hand_joint_velocity_range = self.cfg.hand_joint_velocity_range
                if hand_joint_velocity_range is None:
                    hand_joint_velocity_range = self.cfg.joint_velocity_range

                joint_pos[:, self.extra_joint_indices] += sample_uniform(
                    *hand_joint_position_range,
                    joint_pos[:, self.extra_joint_indices].shape,
                    joint_pos.device,
                )
                joint_vel[:, self.extra_joint_indices] += sample_uniform(
                    *hand_joint_velocity_range,
                    joint_vel[:, self.extra_joint_indices].shape,
                    joint_vel.device,
                )
        else:
            joint_pos = motion_lib_joint_pos
            joint_vel = motion_lib_joint_vel

        if not self.is_evaluating:
            joint_pos += sample_uniform(
                *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
            )
            joint_vel += sample_uniform(
                *self.cfg.joint_velocity_range, joint_vel.shape, joint_vel.device
            )

        if self.cfg.source_only_reset:
            from gear_sonic.envs.manager_env.mdp.rp2_source_relation import (  # noqa: PLC0415
                build_source_only_reset_state,
            )

            source_state = build_source_only_reset_state(self, env_ids)
            root_pos[env_ids] = source_state["root_pos"]
            root_ori[env_ids] = source_state["root_quat"]
            root_lin_vel[env_ids] = source_state["root_lin_vel"]
            root_ang_vel[env_ids] = source_state["root_ang_vel"]
            joint_pos[env_ids] = source_state["joint_pos"]
            joint_vel[env_ids] = source_state["joint_vel"]

        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        ####### Resetting Humaonid States #######
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )
        # Handle object positioning. Active articulation is an opt-in, mutually
        # exclusive replacement for the legacy rigid active object.
        active_articulations = getattr(self._env.scene, "articulations", {})
        if "active_articulation" in active_articulations:
            required_channels = (
                "_motion_active_articulation_root_pos",
                "_motion_active_articulation_root_quat",
                "_motion_active_articulation_root_lin_vel",
                "_motion_active_articulation_root_ang_vel",
                "_motion_active_articulation_joint_pos",
                "_motion_active_articulation_joint_vel",
            )
            missing_channels = [
                name for name in required_channels if not hasattr(self.motion_lib, name)
            ]
            if missing_channels:
                raise ValueError(
                    "Active articulation scene requires reference channels in the object "
                    f"motion data; missing={missing_channels}"
                )

            articulation = active_articulations["active_articulation"]
            root_state = torch.cat(
                [
                    self.active_articulation_root_pos[env_ids],
                    self.active_articulation_root_quat[env_ids],
                    self.active_articulation_root_lin_vel[env_ids],
                    self.active_articulation_root_ang_vel[env_ids],
                ],
                dim=-1,
            )
            config = getattr(self._env.cfg, "config", {}) or {}
            passive_reset = bool(
                config.get("active_articulation_passive_static_reset", False)
            )
            static_mask = ~self.active_articulation_root_dynamic_mask[env_ids]
            if passive_reset and torch.any(static_mask):
                static_env_ids = env_ids[static_mask]
                static_root_state = root_state[static_mask].clone()
                static_root_state[:, 7:] = 0.0
                root_state[static_mask] = static_root_state
                if not hasattr(self, "_active_articulation_static_root_state"):
                    self._active_articulation_static_root_state = torch.zeros(
                        self.num_envs,
                        13,
                        dtype=root_state.dtype,
                        device=self.device,
                    )
                self._active_articulation_static_root_state[static_env_ids] = (
                    static_root_state
                )
            articulation.write_root_state_to_sim(root_state, env_ids=env_ids)

            joint_pos = self.active_articulation_joint_pos[env_ids]
            joint_vel = self.active_articulation_joint_vel[env_ids]
            reference_joint_names = list(
                config.get("active_articulation_reference_joint_names", [])
            )
            if reference_joint_names:
                missing_joints = [
                    name for name in articulation.joint_names if name not in reference_joint_names
                ]
                if missing_joints:
                    raise ValueError(
                        "Articulation joints are absent from reference channel ordering: "
                        f"missing={missing_joints}, reference={reference_joint_names}"
                    )
                source_order = [
                    reference_joint_names.index(name) for name in articulation.joint_names
                ]
                joint_pos = joint_pos[:, source_order]
                joint_vel = joint_vel[:, source_order]
            if joint_pos.shape[1] != articulation.num_joints:
                raise ValueError(
                    "Active articulation reference joint count does not match the USD: "
                    f"reference={joint_pos.shape[1]}, usd={articulation.num_joints}, "
                    f"names={list(articulation.joint_names)}"
                )
            soft_limits = articulation.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clamp(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
            if passive_reset and torch.any(static_mask):
                joint_vel = joint_vel.clone()
                joint_vel[static_mask] = 0.0
                passive_joint_friction = float(
                    config.get(
                        "active_articulation_passive_joint_static_friction", 0.0
                    )
                )
                passive_joint_dynamic_friction = float(
                    config.get(
                        "active_articulation_passive_joint_dynamic_friction", 0.0
                    )
                )
                passive_joint_viscous_friction = float(
                    config.get(
                        "active_articulation_passive_joint_viscous_friction", 0.0
                    )
                )
                passive_joint_damping = float(
                    config.get("active_articulation_passive_joint_damping", 0.0)
                )
                passive_joint_values = {
                    "static_friction": passive_joint_friction,
                    "dynamic_friction": passive_joint_dynamic_friction,
                    "viscous_friction": passive_joint_viscous_friction,
                    "damping": passive_joint_damping,
                }
                invalid_passive_values = {
                    name: value
                    for name, value in passive_joint_values.items()
                    if value < 0.0
                }
                if invalid_passive_values:
                    raise ValueError(
                        "Active-articulation passive joint parameters must be "
                        f"non-negative: {invalid_passive_values}"
                    )
                if any(
                    value > 0.0
                    for value in (
                        passive_joint_friction,
                        passive_joint_dynamic_friction,
                        passive_joint_viscous_friction,
                    )
                ):
                    articulation.write_joint_friction_coefficient_to_sim(
                        passive_joint_friction,
                        joint_dynamic_friction_coeff=(
                            passive_joint_dynamic_friction
                        ),
                        joint_viscous_friction_coeff=(
                            passive_joint_viscous_friction
                        ),
                        env_ids=env_ids[static_mask],
                    )
                if passive_joint_damping > 0.0:
                    articulation.write_joint_damping_to_sim(
                        passive_joint_damping,
                        env_ids=env_ids[static_mask],
                    )
            articulation.write_joint_state_to_sim(
                joint_pos,
                joint_vel,
                env_ids=env_ids,
            )
        elif self._multi_object_mode and len(self._object_names) > 0:
            if getattr(self.cfg, "full_scene_per_env", False):
                audit_reset_index = self._next_object_audit_reset_index()
                object_pos = self.object_root_pos[env_ids]
                object_quat = self.object_root_quat[env_ids]
                inactive_base = INACTIVE_OBJECT_BASE_OFFSET.to(self.device)
                inactive_quat = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]], device=self.device
                )
                all_object_keys = [
                    key for key in self._env.scene.rigid_objects if key.startswith("object_")
                ]
                motion_ids_for_envs = self.motion_ids[env_ids]
                for motion_id in torch.unique(motion_ids_for_envs):
                    group_mask = motion_ids_for_envs == motion_id
                    group_env_ids = env_ids[group_mask]
                    motion_id_int = int(motion_id.item())
                    motion_key = self.motion_lib.curr_motion_keys[int(motion_id.item())]
                    object_scene_keys = motion_order_object_scene_keys(
                        self._env,
                        max_objects=object_pos.shape[1],
                        motion_key=motion_key,
                    )
                    zero_vel = torch.zeros(len(group_env_ids), 6, device=self.device)
                    for obj_idx, obj_key in enumerate(object_scene_keys):
                        if obj_idx >= object_pos.shape[1]:
                            break
                        obj_state = torch.cat(
                            [
                                object_pos[group_mask, obj_idx],
                                object_quat[group_mask, obj_idx],
                                zero_vel,
                            ],
                            dim=-1,
                        )
                        self._env.scene[obj_key].write_root_state_to_sim(
                            obj_state,
                            env_ids=group_env_ids,
                        )

                    inactive_zero_vel = torch.zeros(len(group_env_ids), 6, device=self.device)
                    inactive_idx = 0
                    for obj_key in all_object_keys:
                        if obj_key in object_scene_keys:
                            continue
                        z_offset = torch.tensor(
                            [0.0, 0.0, -inactive_idx * INACTIVE_OBJECT_Z_SPACING],
                            device=self.device,
                        )
                        inactive_pos = (
                            self._env.scene.env_origins[group_env_ids]
                            + inactive_base
                            + z_offset
                        )
                        inactive_state = torch.cat(
                            [
                                inactive_pos,
                                inactive_quat.expand(len(group_env_ids), -1),
                                inactive_zero_vel,
                            ],
                            dim=-1,
                        )
                        self._env.scene[obj_key].write_root_state_to_sim(
                            inactive_state,
                            env_ids=group_env_ids,
                        )
                        inactive_idx += 1
                    if audit_reset_index is not None:
                        self._append_full_scene_reset_audit(
                            reset_index=audit_reset_index,
                            motion_id=motion_id_int,
                            motion_key=motion_key,
                            env_ids=group_env_ids,
                            object_scene_keys=object_scene_keys,
                            all_object_keys=all_object_keys,
                            object_pos=object_pos[group_mask],
                            object_quat=object_quat[group_mask],
                        )
                if audit_reset_index is not None:
                    self._write_object_audit()
            else:
                # MULTI-OBJECT MODE: Position active object, move others far away.
                # Get active object name from motion key (original name with hyphens).
                active_motion_id = self.motion_ids[env_ids[0]].item() if len(env_ids) > 0 else 0
                active_motion_key = self.motion_lib.curr_motion_keys[active_motion_id]
                active_obj_safe_name = active_motion_key.replace("-", "_")
                self._active_object_name = active_motion_key

                active_key = f"object_{active_obj_safe_name}"
                if active_key in self._env.scene.rigid_objects:
                    obj = self._env.scene[active_key]
                    obj_pos = self._get_object_pos_with_offset(env_ids)
                    zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
                    obj.write_root_state_to_sim(
                        torch.cat(
                            [obj_pos, self.object_root_quat[env_ids, 0], zero_vel],
                            dim=-1,
                        ),
                        env_ids=env_ids,
                    )
                else:
                    print(  # noqa: T201
                        f"[Warning] Active object '{active_key}' not found in scene. Available: {list(self._env.scene.rigid_objects.keys())[:5]}..."  # noqa: E501
                    )

                inactive_base = INACTIVE_OBJECT_BASE_OFFSET.to(self.device)
                inactive_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(
                    len(env_ids), -1
                )

                inactive_idx = 0
                for safe_name in self._object_names:
                    obj_key = f"object_{safe_name}"
                    if obj_key != active_key and obj_key in self._env.scene.rigid_objects:
                        z_offset = torch.tensor(
                            [0.0, 0.0, -inactive_idx * INACTIVE_OBJECT_Z_SPACING],
                            device=self.device,
                        )
                        inactive_pos = (
                            self._env.scene.env_origins[env_ids] + inactive_base + z_offset
                        )
                        inactive_zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
                        inactive_state = torch.cat(
                            [inactive_pos, inactive_quat, inactive_zero_vel], dim=-1
                        )
                        self._env.scene[obj_key].write_root_state_to_sim(
                            inactive_state, env_ids=env_ids
                        )
                        inactive_idx += 1
        elif "object" in self._env.scene.rigid_objects:
            # SINGLE OBJECT MODE: Existing behavior
            obj = self._env.scene["object"]
            obj_pos = self._get_object_pos_with_offset(env_ids)
            if self.cfg.object_reset_use_reference_velocity:
                object_vel = torch.cat(
                    (
                        self.object_root_lin_vel[env_ids, 0],
                        self.object_root_ang_vel[env_ids, 0],
                    ),
                    dim=-1,
                )
            else:
                # Preserve original GRAIL-SONIC reset behavior by default.
                object_vel = torch.zeros(len(env_ids), 6, device=self.device)
            obj.write_root_state_to_sim(
                torch.cat([obj_pos, self.object_root_quat[env_ids, 0], object_vel], dim=-1),
                env_ids=env_ids,
            )
            self._cache_reference_locked_object_state(
                env_ids, obj_pos, self.object_root_quat[env_ids, 0]
            )
            audit_reset_index = self._next_object_audit_reset_index()
            if audit_reset_index is not None:
                motion_ids_for_envs = self.motion_ids[env_ids]
                reference_steps = self.motion_start_time_steps[env_ids] + self.time_steps[env_ids]
                for motion_id in torch.unique(motion_ids_for_envs):
                    group_mask = motion_ids_for_envs == motion_id
                    group_env_ids = env_ids[group_mask]
                    motion_id_int = int(motion_id.item())
                    self._append_full_scene_reset_audit(
                        reset_index=audit_reset_index,
                        motion_id=motion_id_int,
                        motion_key=self.motion_lib.curr_motion_keys[motion_id_int],
                        env_ids=group_env_ids,
                        object_scene_keys=["object"],
                        all_object_keys=["object"],
                        object_pos=obj_pos[group_mask, None, :],
                        object_quat=self.object_root_quat[group_env_ids, 0, None, :],
                        reference_time_steps=reference_steps[group_mask],
                    )
                self._write_object_audit()
        if "table" in self._env.scene.rigid_objects:
            table = self._env.scene["table"]

            # Per-env table positions based on each env's assigned motion
            table_pos_list = []
            table_quat_list = []

            # Prefetch motion_ids to CPU to avoid per-iteration GPU sync
            motion_ids_cpu = self.motion_ids.cpu()

            for i, env_idx in enumerate(env_ids):  # noqa: B007
                motion_idx = motion_ids_cpu[env_idx].item()
                motion_key = self.motion_lib.curr_motion_keys[motion_idx]

                # Load and cache meta if not already cached
                if motion_key not in self._table_meta_cache:
                    self._load_table_meta(motion_key)

                cached = self._table_meta_cache[motion_key]
                pos = cached["table_pos"].clone()
                quat = cached["table_quat"].clone()
                pos = pos + self._env.scene.env_origins[env_idx]
                table_pos_list.append(pos)
                table_quat_list.append(quat)

            if len(table_pos_list) > 0:
                table_pos = torch.stack(table_pos_list, dim=0)
                table_quat = torch.stack(table_quat_list, dim=0)

                # Apply table_offset if configured
                if self.cfg.table_offset is not None:
                    table_offset = torch.tensor(
                        self.cfg.table_offset, device=self._env.device, dtype=table_pos.dtype
                    )
                    table_pos = table_pos + table_offset

                table_root_pose = torch.cat([table_pos, table_quat], dim=-1)
                table.write_root_pose_to_sim(table_root_pose, env_ids=env_ids)

        # Clear any hold state inherited from the previous episode before
        # arming the optional single-object reference hold. It can cover either an
        # in-contact reset or the pre-contact approach plus a short release grace.
        self.object_hold_frames_left[env_ids] = 0
        self.object_hold_total_frames[env_ids] = 0
        self.object_hold_exact_frames[env_ids] = 0
        self.object_hold_release_frames[env_ids] = 0
        if (
            (
                self.cfg.object_hold_after_in_contact_reset
                or self.cfg.object_hold_before_reference_contact
            )
            and not self._multi_object_mode
            and "object" in self._env.scene.rigid_objects
        ):
            start_steps = self.motion_start_time_steps[env_ids] + self.time_steps[env_ids]
            hold_frames = torch.zeros(
                len(env_ids), dtype=torch.long, device=self.device
            )
            exact_frames = torch.zeros_like(hold_frames)
            release_frames = torch.zeros_like(hold_frames)
            precontact_held = torch.zeros(
                len(env_ids), dtype=torch.bool, device=self.device
            )

            if (
                self.cfg.object_hold_before_reference_contact
                and self._first_contact_lookup is not None
            ):
                first_contact = self._per_env_first_contact[env_ids]
                motion_steps = self.motion_lib.get_time_step_total(
                    self.motion_ids[env_ids]
                )
                precontact_held = (first_contact < motion_steps) & (
                    start_steps < first_contact
                )
                precontact_frames = (
                    first_contact - start_steps
                ).clamp(min=0)
                precontact_release_frames = torch.full_like(
                    precontact_frames,
                    max(int(self.cfg.object_hold_release_grace_frames), 0),
                )
                hold_frames = torch.where(
                    precontact_held,
                    precontact_frames + precontact_release_frames,
                    hold_frames,
                )
                exact_frames = torch.where(
                    precontact_held, precontact_frames, exact_frames
                )
                release_frames = torch.where(
                    precontact_held, precontact_release_frames, release_frames
                )

            held = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            if self.cfg.object_hold_after_in_contact_reset:
                for hand in ("left_hand", "right_hand"):
                    in_contact = self.motion_lib.get_object_in_contact(
                        self.motion_ids[env_ids], start_steps, hand=hand
                    )
                    if in_contact is not None:
                        held |= in_contact > 0.5
                configured_exact = max(
                    int(getattr(self.cfg, "object_hold_replay_frames", 0)), 0
                )
                configured_release = max(
                    int(getattr(self.cfg, "object_hold_release_frames", 0)), 0
                )
                if configured_exact > 0 or configured_release > 0:
                    in_contact_exact = configured_exact
                    in_contact_release = configured_release
                else:
                    # Legacy behavior: object_hold_frames is one cosine fade.
                    in_contact_exact = 0
                    in_contact_release = max(int(self.cfg.object_hold_frames), 0)
                in_contact_total = in_contact_exact + in_contact_release
                use_in_contact = held & (in_contact_total > hold_frames)
                hold_frames = torch.where(
                    use_in_contact,
                    torch.full_like(hold_frames, in_contact_total),
                    hold_frames,
                )
                exact_frames = torch.where(
                    use_in_contact,
                    torch.full_like(exact_frames, in_contact_exact),
                    exact_frames,
                )
                release_frames = torch.where(
                    use_in_contact,
                    torch.full_like(release_frames, in_contact_release),
                    release_frames,
                )

            self.object_hold_frames_left[env_ids] = hold_frames
            self.object_hold_total_frames[env_ids] = hold_frames
            self.object_hold_exact_frames[env_ids] = exact_frames
            self.object_hold_release_frames[env_ids] = release_frames
            if not getattr(self, "_object_hold_logged", False):
                active = hold_frames > 0
                active_frames = hold_frames[active]
                frame_range = (
                    f"{int(active_frames.min())}-{int(active_frames.max())}"
                    if len(active_frames) > 0
                    else "none"
                )
                print(  # noqa: T201
                    f"[TrackingCommand] Object hold-and-release active: "
                    f"{int(active.sum())}/{len(active)} reset envs held "
                    f"(pre-contact={int(precontact_held.sum())}, "
                    f"in-contact={int(held.sum())}, frames={frame_range}, "
                    f"exact={int(exact_frames.max()) if len(exact_frames) else 0}, "
                    f"release={int(release_frames.max()) if len(release_frames) else 0})"
                )
                self._object_hold_logged = True

        self._refresh_relative_body_targets()

    def _refresh_relative_body_targets(self):
        """Refresh current-frame body targets in the robot anchor frame.

        The reference frame advances every control step. Keeping these tensors
        from the reset frame makes body rewards and end-effector terminations
        compare the robot against a stale pose. Recompute the same heading- and
        XY-invariant targets without changing their established semantics.
        """
        num_bodies = len(self.cfg.body_names)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, num_bodies, 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, num_bodies, 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = torch_transform.get_heading_q(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

    def _apply_object_hold_after_in_contact_reset(self):
        """Blend temporarily held objects from reference replay into free physics.

        ``_resample_command`` may arm an in-contact reset hold or a hold through
        the reference pre-contact approach. A cosine blend starts at exact
        reference replay and reaches zero correction on the final held frame,
        avoiding a hard transition to free simulation.
        """
        held = self.object_hold_frames_left > 0
        self.metrics["object_hold/active"] = held.float()
        self.metrics["rp2_object_hold/active"] = held.float()
        if not torch.any(held):
            zeros = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.metrics["object_hold/reference_blend"] = zeros
            self.metrics["rp2_object_hold/reference_blend"] = zeros
            self.metrics["rp2_object_hold/exact_replay"] = zeros
            self.metrics["rp2_object_hold/release"] = zeros
            return
        if self._multi_object_mode or "object" not in self._env.scene.rigid_objects:
            self.object_hold_frames_left.zero_()
            self.object_hold_total_frames.zero_()
            self.object_hold_exact_frames.zero_()
            self.object_hold_release_frames.zero_()
            return
        env_ids = torch.where(held)[0]
        obj = self._env.scene["object"]
        reference_pos = self.object_root_pos[env_ids, 0].clone()
        if self.cfg.object_position_randomize and hasattr(self, "_object_position_offset"):
            reference_pos = reference_pos + self._object_position_offset[env_ids]
        init_z_offset = getattr(self.cfg, "object_init_z_offset", 0.0)
        if init_z_offset != 0.0:
            reference_pos[:, 2] += init_z_offset
        reference_quat = self.object_root_quat[env_ids, 0]
        reference_vel = torch.cat(
            (
                self.object_root_lin_vel[env_ids, 0],
                self.object_root_ang_vel[env_ids, 0],
            ),
            dim=-1,
        )

        remaining = self.object_hold_frames_left[env_ids]
        total = self.object_hold_total_frames[env_ids].clamp(min=1)
        exact = self.object_hold_exact_frames[env_ids].clamp(min=0)
        release = self.object_hold_release_frames[env_ids].clamp(min=0)
        elapsed = total - remaining
        exact_phase = elapsed < exact
        release_elapsed = (elapsed - exact).clamp(min=0)
        release_denominator = (release - 1).clamp(min=1)
        release_progress = release_elapsed.float() / release_denominator.float()
        release_blend = 0.5 * (
            1.0 + torch.cos(torch.pi * release_progress.clamp(max=1.0))
        )
        release_blend = torch.where(
            release > 1, release_blend, torch.ones_like(release_blend)
        )
        reference_blend = torch.where(
            exact_phase, torch.ones_like(release_blend), release_blend
        )
        blend = reference_blend[:, None]

        object_pos = torch.lerp(obj.data.root_pos_w[env_ids, :3], reference_pos, blend)
        object_quat = isaac_utils_rotations.slerp(
            obj.data.root_quat_w[env_ids], reference_quat, blend
        )
        current_vel = torch.cat(
            (obj.data.root_lin_vel_w[env_ids, :3], obj.data.root_ang_vel_w[env_ids, :3]),
            dim=-1,
        )
        object_vel = torch.lerp(current_vel, reference_vel, blend)
        obj.write_root_state_to_sim(
            torch.cat([object_pos, object_quat, object_vel], dim=-1),
            env_ids=env_ids,
        )
        blend_metric = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        blend_metric[env_ids] = reference_blend
        self.metrics["object_hold/reference_blend"] = blend_metric
        self.metrics["rp2_object_hold/reference_blend"] = blend_metric
        exact_metric = torch.zeros_like(blend_metric)
        exact_metric[env_ids] = exact_phase.float()
        self.metrics["rp2_object_hold/exact_replay"] = exact_metric
        release_metric = torch.zeros_like(blend_metric)
        release_metric[env_ids] = (~exact_phase).float()
        self.metrics["rp2_object_hold/release"] = release_metric
        self.object_hold_frames_left[held] -= 1

    def _cache_reference_locked_object_state(
        self, env_ids: torch.Tensor, object_pos: torch.Tensor, object_quat: torch.Tensor
    ):
        """Cache the reset pose for targets configured as stationary supports."""
        if not self.cfg.object_reference_lock_roles:
            return
        if not hasattr(self, "_object_reference_locked_state"):
            self._object_reference_locked_state = torch.zeros(
                self.num_envs, 13, dtype=object_pos.dtype, device=self.device
            )
        locked = self._target_object_role_mask(self.cfg.object_reference_lock_roles)[
            env_ids
        ]
        if not torch.any(locked):
            return
        locked_env_ids = env_ids[locked]
        zero_velocity = torch.zeros(
            len(locked_env_ids), 6, dtype=object_pos.dtype, device=self.device
        )
        self._object_reference_locked_state[locked_env_ids] = torch.cat(
            [object_pos[locked], object_quat[locked], zero_velocity], dim=-1
        )

    def _apply_object_reference_lock(self):
        """Hold stationary support targets at their sampled reset pose."""
        if not self.cfg.object_reference_lock_roles:
            return
        if self._multi_object_mode or "object" not in self._env.scene.rigid_objects:
            return
        if not hasattr(self, "_object_reference_locked_state"):
            return
        locked = self._target_object_role_mask(self.cfg.object_reference_lock_roles)
        self.metrics["rp2_object/reference_locked"] = locked.float()
        if not torch.any(locked):
            return
        env_ids = torch.where(locked)[0]
        self._env.scene["object"].write_root_state_to_sim(
            self._object_reference_locked_state[env_ids], env_ids=env_ids
        )

    def _clamp_static_active_articulation_roots(self):
        """Hold stationary mechanism roots according to the configured clamp mode."""
        config = getattr(self._env.cfg, "config", {}) or {}
        if not bool(config.get("active_entity_runtime_clamp_static_roots", False)):
            return
        articulation = getattr(self._env.scene, "articulations", {}).get(
            "active_articulation"
        )
        if articulation is None or not hasattr(
            self.motion_lib, "_motion_active_articulation_root_dynamic_mask"
        ):
            return
        root_dynamic = self.active_articulation_root_dynamic_mask
        static_env_ids = torch.where(~root_dynamic)[0]
        self.metrics["active_articulation/root_dynamic"] = root_dynamic.float()
        if static_env_ids.numel() == 0:
            return
        passive_reset = bool(
            config.get("active_articulation_passive_static_reset", False)
        )
        if passive_reset:
            if not hasattr(self, "_active_articulation_static_root_state"):
                raise RuntimeError(
                    "Passive active-articulation root clamp was enabled before reset "
                    "state was cached."
                )
            root_state = self._active_articulation_static_root_state[static_env_ids]
        else:
            root_state = torch.cat(
                [
                    self.active_articulation_root_pos[static_env_ids],
                    self.active_articulation_root_quat[static_env_ids],
                    self.active_articulation_root_lin_vel[static_env_ids],
                    self.active_articulation_root_ang_vel[static_env_ids],
                ],
                dim=-1,
            )
        articulation.write_root_state_to_sim(root_state, env_ids=static_env_ids)

    def _snapshot_completed_episodes(self, env_ids: Sequence[int]):
        """Record the episode ``env_ids`` just finished, before the reset overwrites it.

        Used only with ``adaptive_sampling.update_on_episode_end_only``. The
        snapshot keeps the finished episode's motion id, sampled start step and
        terminal step, and whether it ended by termination
        (``env.reset_terminated``; time-outs and the soft motion-end wrap are
        not failures). Envs that have not stepped since their previous reset
        (the initial reset, duplicate resets; ``episode_length_buf == 0``, which
        Isaac Lab zeroes only after the command reset) are not counted, so an
        unfinished slot never adds an observation. ``_update_command`` consumes
        the pending buffer exactly once.
        """
        env_ids_t = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        if env_ids_t.dtype == torch.bool:
            env_ids_t = torch.where(env_ids_t)[0]
        episode_length = getattr(self._env, "episode_length_buf", None)
        stepped = (
            episode_length[env_ids_t] > 0
            if episode_length is not None
            else self.time_steps[env_ids_t] > 0
        )
        if not bool(stepped.any()):
            return
        env_ids_t = env_ids_t[stepped]
        terminated = getattr(self._env, "reset_terminated", None)
        failed = (
            terminated[env_ids_t].bool()
            if terminated is not None
            else torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)
        )
        self._pending_episode_mask[env_ids_t] = True
        self._pending_episode_failed[env_ids_t] = failed
        # Explicit casts into the int64 snapshot buffers: the live fields may be native int32
        # (a restored motion state keeps the restored dtypes); indexed assignment does not cast,
        # and start + elapsed must be added in int64 so two int32 operands cannot overflow.
        self._pending_episode_motion_ids[env_ids_t] = self.motion_ids[env_ids_t].to(
            self._pending_episode_motion_ids.dtype
        )
        start_steps = self.motion_start_time_steps[env_ids_t].to(
            self._pending_episode_start_steps.dtype
        )
        elapsed_steps = self.time_steps[env_ids_t].to(self._pending_episode_end_steps.dtype)
        self._pending_episode_start_steps[env_ids_t] = start_steps
        self._pending_episode_end_steps[env_ids_t] = start_steps + elapsed_steps

    def _consume_completed_episodes(self) -> int:
        """Feed the pending completed episodes to the motion library and clear them.

        Returns the number of episodes credited (to their OLD motion and bins).
        """
        completed = torch.where(self._pending_episode_mask)[0]
        if completed.numel() == 0:
            return 0
        self.motion_lib.update_adaptive_sampling(
            self._pending_episode_failed[completed],
            self._pending_episode_motion_ids[completed],
            self._pending_episode_end_steps[completed],
            sampled_start_time_steps=self._pending_episode_start_steps[completed],
        )
        self._pending_episode_mask[completed] = False
        return int(completed.numel())

    def _update_command(self):
        """Advance the motion time cursor by one step and handle episode wrap-around.

        Called once per simulation step. Updates adaptive sampling statistics,
        increments the time cursor, resamples any environments that have reached
        the end of their motion clip, updates the running reference root height
        EMA, and optionally performs height-map raycasting for object-aware
        observations.
        """
        if self.use_adaptive_sampling:
            with common.Timer("update_adaptive_sampling"):
                cur_time_steps = self.motion_start_time_steps + self.time_steps
                terminal_only = bool(
                    self.motion_lib.adaptive_sampling_cfg.get(
                        "update_on_episode_end_only", False
                    )
                )
                if terminal_only:
                    # Completed episodes were snapshotted by _resample_command
                    # BEFORE their reset; the live buffers already hold the
                    # next episode of those envs.
                    self._consume_completed_episodes()
                else:
                    self.motion_lib.update_adaptive_sampling(
                        self._env.reset_terminated,
                        self.motion_ids,
                        cur_time_steps,
                        sampled_start_time_steps=self.motion_start_time_steps,
                    )
        self.time_steps, env_ids = advance_motion_time_steps(
            self.time_steps,
            self.motion_start_time_steps,
            self.motion_lib.get_time_step_total(self.motion_ids),
            resample_on_motion_end=getattr(
                self.cfg, "resample_on_motion_end", True
            ),
        )
        if len(env_ids) > 0 and not getattr(self, "_warned_soft_wrap", False):
            # This wrap-around teleports envs to a fresh start without an env
            # reset, so stateful termination terms (e.g. grasp-failure
            # trackers) keep values from the previous clip segment. With the
            # tracking time_out termination enabled it fires one step earlier
            # and this path stays dead; reaching it means terminations are
            # configured such that mid-episode wraps corrupt that state.
            print(  # noqa: T201
                f"[TrackingCommand] WARNING: soft wrap-around resample fired for "
                f"{len(env_ids)} env(s); termination-term state is NOT reset on "
                "this path. Check that the tracking time_out termination is enabled."
            )
            self._warned_soft_wrap = True
        self._resample_command(env_ids)
        self._refresh_relative_body_targets()
        self._clamp_static_active_articulation_roots()
        self._apply_object_reference_lock()
        if (
            self.cfg.object_hold_after_in_contact_reset
            or self.cfg.object_hold_before_reference_contact
        ):
            self._apply_object_hold_after_in_contact_reset()

        # Exponential moving average update for running_ref_root_height.
        # ZL this should be moved to the recorders???
        ema_alpha = 0.1  # Smoothing factor, adjust as needed
        self.running_ref_root_height = (
            ema_alpha * self.anchor_pos_w[:, 2] + (1 - ema_alpha) * self.running_ref_root_height
        )

        if self.cfg.use_height_map:
            root_pos_w = self.robot.data.root_pos_w
            root_quat_w = self.robot.data.root_quat_w

            ray_starts_w = root_pos_w.unsqueeze(1).expand(-1, self.num_rays, -1)
            root_quat_expanded = (
                torch_transform.get_heading_q(root_quat_w)
                .unsqueeze(1)
                .expand(-1, self.num_rays, -1)
            )
            ray_dirs_w = quat_apply(
                root_quat_expanded,
                self.ray_dirs_local,
            )

            scan_dot_pos_w, _ = self.height_map.raycast_fused(
                self.object_root_pos,
                self.object_root_quat,
                ray_starts_w,
                ray_dirs_w,
                n_mesh_per_cam=self.num_mesh_per_cam,
                mesh_ids_flattened=self.mesh_ids,
                cam_ids_flattened=self.cam_ids,
                min_dist=0.0,
                max_dist=self.cfg.height_map_max_dist,
            )
            # Adjust hits that fall below the ground plane so all zs are non-negative.
            denom = ray_starts_w[:, :, 2] - scan_dot_pos_w[:, :, 2]
            denom = torch.clamp_min(denom, 1e-8)
            scale = torch.clamp(ray_starts_w[:, :, 2] / denom, max=1.0)
            self.scan_dot_pos_w[:] = (
                ray_starts_w + (scan_dot_pos_w - ray_starts_w) * scale.unsqueeze(-1)
            ).view(self.num_envs, self.num_rays_x, self.num_rays_y, 3)

    @property
    def future_time_steps(self) -> torch.Tensor:
        """Compute absolute time-step indices for all future reference frames.

        Clamps to the last valid frame of each motion to avoid out-of-bounds access.

        Returns:
            Flattened tensor of shape ``(num_envs * num_future_frames,)``.
        """
        return (
            torch.clip(
                self.future_time_steps_init
                + self.time_steps[:, None]
                + self.motion_start_time_steps[:, None],
                max=self.motion_num_steps[:, None] - 1,
            )
            .flatten()
            .long()
        )

    @property
    def smpl_future_time_steps(self) -> torch.Tensor:
        """Compute absolute time-step indices for SMPL future reference frames.

        SMPL future frames may use different count and spacing than robot frames.

        Returns:
            Flattened tensor of shape ``(num_envs * smpl_num_future_frames,)``.
        """
        return (
            torch.clip(
                self.smpl_future_time_steps_init
                + self.time_steps[:, None]
                + self.motion_start_time_steps[:, None],
                max=self.motion_num_steps[:, None] - 1,
            )
            .flatten()
            .long()
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle visibility of debug visualization markers.

        Lazily initializes per-body frame markers, anchor markers, feet contact
        markers, height-map dot markers, contact center sphere markers, and
        optional current/reference object markers on first enable. Subsequent
        calls toggle visibility without re-creating prims.

        Args:
            debug_vis: Whether to enable or disable debug visualization.
        """
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/anchor"
                    )
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.goal_pos_visualizer = VisualizationMarkers(
                    self.cfg.body_pos_visualizer_cfg.replace(
                        prim_path="/Visuals/goal_marker_sphere"
                    )
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                self.feet_contact_goal_visualizers = []

                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/" + name
                            )
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )
                for name in self.cfg.feet_body_names:
                    self.feet_contact_goal_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.feet_contact_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )

            self.current_anchor_visualizer.set_visibility(self.cfg.debug_vis_body)
            self.goal_anchor_visualizer.set_visibility(self.cfg.debug_vis_body)
            self.goal_pos_visualizer.set_visibility(self.cfg.debug_vis_body)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(self.cfg.debug_vis_body)
                self.goal_body_visualizers[i].set_visibility(self.cfg.debug_vis_body)
            for i in range(len(self.cfg.feet_body_names)):
                self.feet_contact_goal_visualizers[i].set_visibility(self.cfg.debug_vis_body)

            if self.cfg.use_height_map:
                if not hasattr(self, "height_map_visualizer"):
                    height_map_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/height_map",
                        markers={
                            "scan_dots": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(1.0, 1.0, 0.0)
                                ),
                            ),
                        },
                    )
                    self.height_map_visualizer = VisualizationMarkers(height_map_cfg)
                self.height_map_visualizer.set_visibility(True)

            # Contact center visualizers: deferred to _debug_vis_callback (lazy init)
            # because motion_lib is not yet available during super().__init__()
            if not hasattr(self, "contact_center_visualizers"):
                self.contact_center_visualizers = None

            if self.cfg.debug_vis_object:
                if not hasattr(self, "current_object_visualizer"):
                    current_object_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/current/active_object",
                        markers={
                            "object": sim_utils.SphereCfg(
                                radius=self.cfg.debug_vis_object_radius,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(0.0, 1.0, 0.0),
                                ),
                            ),
                        },
                    )
                    reference_object_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/goal/active_object",
                        markers={
                            "object": sim_utils.SphereCfg(
                                radius=self.cfg.debug_vis_object_radius * 0.65,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(1.0, 0.0, 0.0),
                                ),
                            ),
                        },
                    )
                    self.current_object_visualizer = VisualizationMarkers(current_object_cfg)
                    self.reference_object_visualizer = VisualizationMarkers(reference_object_cfg)
                self.current_object_visualizer.set_visibility(True)
                self.reference_object_visualizer.set_visibility(True)
            elif hasattr(self, "current_object_visualizer"):
                self.current_object_visualizer.set_visibility(False)
                self.reference_object_visualizer.set_visibility(False)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "feet_contact_goal_visualizers"):
                for vis in self.feet_contact_goal_visualizers:
                    vis.set_visibility(False)
            if hasattr(self, "height_map_visualizer"):
                self.height_map_visualizer.set_visibility(False)
            if hasattr(self, "contact_center_visualizers") and self.contact_center_visualizers:
                for vis in self.contact_center_visualizers.values():
                    vis.set_visibility(False)
            if hasattr(self, "current_object_visualizer"):
                self.current_object_visualizer.set_visibility(False)
                self.reference_object_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):  # noqa: ARG002
        """Update debug visualization marker positions each render frame.

        Draws current robot body frames, reference (goal) body frames, feet
        contact indicators, height-map hit points, and object contact centers.

        Args:
            event: Render event from the simulation (unused).
        """
        if not self.robot.is_initialized:
            return

        if self.cfg.debug_vis_body:
            self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
            self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

            self.goal_pos_visualizer.visualize(self.body_pos_w.view(-1, 3))

            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].visualize(
                    self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i]
                )

                if self.cfg.body_names[i] == "left_ankle_roll_link":
                    self.feet_contact_goal_visualizers[0].visualize(
                        translations=self.body_pos_relative_w[:, i],
                        marker_indices=self.feet_l.int().reshape(-1),
                    )
                if self.cfg.body_names[i] == "right_ankle_roll_link":
                    self.feet_contact_goal_visualizers[1].visualize(
                        translations=self.body_pos_relative_w[:, i],
                        marker_indices=self.feet_r.int().reshape(-1),
                    )

                self.goal_body_visualizers[i].visualize(
                    self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i]
                )

            if self.cfg.use_height_map:
                self.height_map_visualizer.visualize(
                    translations=self.scan_dot_pos_w.view(-1, 3),
                )

        if self.cfg.debug_vis_object and hasattr(self, "current_object_visualizer"):
            current_pos, _, _, _, current_valid = mdp_utils.get_target_object_current_state(
                self._env, self
            )
            reference_pos = self.active_entity_root_pos
            marker_offset = torch.zeros_like(reference_pos)
            marker_offset[:, 2] = self.cfg.debug_vis_object_z_offset
            hidden_pos = torch.zeros_like(current_pos)
            hidden_pos[:, 2] = -1000.0
            current_marker_pos = current_pos + marker_offset
            current_marker_pos = torch.where(
                current_valid[:, None], current_marker_pos, hidden_pos
            )
            self.current_object_visualizer.visualize(translations=current_marker_pos)
            self.reference_object_visualizer.visualize(
                translations=reference_pos + marker_offset
            )

        # Contact center visualization (lazy init on first callback)
        if (
            hasattr(self, "contact_center_visualizers") and self.contact_center_visualizers is None
        ):  # noqa: SIM102
            if hasattr(self, "motion_lib"):
                self.contact_center_visualizers = {}
                if hasattr(self.motion_lib, "_motion_object_contact_center_left"):
                    left_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/contact_center_left",
                        markers={
                            "contact": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(0.0, 0.0, 1.0),
                                ),
                            ),
                        },
                    )
                    self.contact_center_visualizers["left_hand"] = VisualizationMarkers(left_cfg)
                if hasattr(self.motion_lib, "_motion_object_contact_center_right"):
                    right_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/contact_center_right",
                        markers={
                            "contact": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(0.0, 1.0, 1.0),
                                ),
                            ),
                        },
                    )
                    self.contact_center_visualizers["right_hand"] = VisualizationMarkers(right_cfg)
                for vis in self.contact_center_visualizers.values():
                    vis.set_visibility(True)

        if self.contact_center_visualizers:
            hidden_pos = torch.tensor([[0.0, 0.0, -1000.0]], device=self._env.device)
            for hand, visualizer in self.contact_center_visualizers.items():
                contact_center = self.motion_lib.get_object_contact_center(
                    self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
                )
                if contact_center is None:
                    visualizer.visualize(translations=hidden_pos)
                    continue
                valid_mask = torch.norm(contact_center, dim=-1) > 1e-6
                world_center = self._get_contact_center_world(hand)
                world_center[~valid_mask] = hidden_pos
                visualizer.visualize(translations=world_center)

    def set_motion_state(self, motion_ids, time_steps, motion_start_time_steps=None):
        """Update which motion clip/frame this command serves (for offline use)."""
        self.motion_ids = motion_ids
        self.time_steps = time_steps
        self.motion_start_time_steps = (
            motion_start_time_steps
            if motion_start_time_steps is not None
            else torch.zeros_like(motion_ids)
        )
        self.future_motion_ids = motion_ids.repeat_interleave(self.num_future_frames)
        self.smpl_future_motion_ids = motion_ids.repeat_interleave(self.smpl_num_future_frames)
        self.motion_num_steps = self.motion_lib.get_time_step_total(motion_ids)


class ForceTrackingCommand(CommandTerm):
    """Apply external perturbation forces and manage compliance state for training robustness.

    This command term works alongside ``TrackingCommand`` to add domain-randomized
    external forces on specified robot bodies (typically wrists and torso). It also
    tracks per-env compliance levels that control how stiffly the end-effectors
    track their targets, enabling the policy to learn compliant manipulation
    behaviors.

    Key responsibilities:
        - Maintain per-body force direction and magnitude buffers that event terms
          sample into periodically (every ``force_update_frequency`` steps).
        - Track per-env compliance levels for left wrist, right wrist, and head
          via ``eef_stiffness_buf``.
        - Compute Jacobian-based end-effector analysis for compliance calculations.
        - Report detailed force and compliance metrics per encoder type
          (G1/teleop/SMPL) for W&B logging.

    NOTE: Force application itself is handled by event terms that read and write
    this command's buffers. This command only manages the state and metrics.
    """

    cfg: ForceTrackingCommandCfg

    def __init__(self, cfg: ForceTrackingCommandCfg, env: ManagerBasedRLEnv):
        """Initialize force tracking state, Jacobian indices, and compliance buffers.

        Args:
            cfg: Configuration specifying force bodies, max force, update frequency,
                joint dependencies for Jacobian computation, and debug settings.
            env: The manager-based RL environment that owns this command.
        """
        super().__init__(cfg, env)

        self.is_evaluating = False
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)

        self.left_eef_deps_ids = np.array(
            [self.robot.joint_names.index(joint) for joint in self.cfg.left_eef_deps]
        )
        self.right_eef_deps_ids = np.array(
            [self.robot.joint_names.index(joint) for joint in self.cfg.right_eef_deps]
        )
        self.waist_deps_ids = np.array(
            [self.robot.joint_names.index(joint) for joint in self.cfg.waist_joints]
        )
        self.upper_body_deps_ids = np.concatenate(
            [self.waist_deps_ids, self.left_eef_deps_ids, self.right_eef_deps_ids]
        )

        self.kp = self.robot.data.joint_stiffness.clone()
        self.kd = self.robot.data.joint_damping.clone()

        self.force_update_frequency = self.cfg.force_update_frequency
        self.max_force = self.cfg.max_force
        self.vr_3point_body_indices = [
            self.robot.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_indices_motion = [
            self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_offsets = (
            torch.tensor(self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device)
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        ### Force related
        self.num_bodies = len(self.cfg.body_names)
        self.body_force_dir_buf = torch.randn(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.body_force_dir_buf /= torch.norm(
            self.body_force_dir_buf, dim=-1, keepdim=True
        )  # normalize

        # NOTE: We initialize force_push_ids first so we can use len(force_push_ids) for buffer shape
        self.force_push_ids = self.robot.find_bodies(self.cfg.force_push_body, preserve_order=True)[
            0
        ]
        self.num_force_push_bodies = len(self.force_push_ids)

        # Per-body force magnitude buffer: [num_envs, num_force_push_bodies]
        # Each body can have different force magnitudes, enabling differentiated
        # force application (e.g., stronger forces on wrists than torso)
        self.body_force_magnitude_buf = torch.rand(
            self.num_envs,
            self.num_force_push_bodies,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # [0, 1] per body

        self.force_push_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.force_duration_per_env = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.force_config_init = False
        self.non_force_push_ids_rel = []
        self.force_push_ids_rel = []
        for i, idx in enumerate(self.body_indexes.tolist()):
            if idx not in self.force_push_ids:
                self.non_force_push_ids_rel.append(i)
            else:
                self.force_push_ids_rel.append(i)
        # self.non_force_push_ids = [i for i in self.body_indexes.tolist() if i not in self.force_push_ids]
        self.force_push_body_offsets = (
            torch.tensor(self.cfg.force_push_body_offset, dtype=torch.float32, device=self.device)
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )
        self.last_force_applied = torch.zeros(
            self.num_envs,
            len(self.force_push_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # compliance related counters
        self.compliance_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.compliance_duration_per_env = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.eef_stiffness_buf = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.compliance_config_init = False

        # Compliance monitoring - track cumulative stats for sanity checks
        self._force_update_count = 0  # Tracks how many times force was applied (non-zero)
        self._compliance_update_count = 0  # Tracks how many times compliance was updated
        self._total_steps = 0
        self._warned_no_force = False  # Prevent spamming warnings
        # Debug print frequency (0 = disabled, nonzero = print every N steps)
        # Can be set via config: manager_env.commands.force.debug_print_every_n_steps=10
        self._debug_print_every_n_steps = self.cfg.debug_print_every_n_steps
        # Note: Metrics are now created dynamically in _update_metrics()
        # The old "force applied" metric (with space) has been removed to avoid
        # confusion with "force_applied" (with underscore)

    def set_is_evaluating(self, is_evaluating: bool):
        """Toggle evaluation mode."""
        self.is_evaluating = is_evaluating

    @property
    def jacobian(self) -> torch.Tensor:
        """Return the full articulation Jacobian from PhysX.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 6, num_dof + 6)``.
        """
        return self.robot.root_physx_view.get_jacobians()

    @property
    def eef_jacobian(self) -> torch.Tensor:
        """Return the translational Jacobian for left and right end-effectors.

        Returns:
            Tensor of shape ``(num_envs, 2, 3, num_dof + 6)``.
        """
        return self.jacobian[:, self.vr_3point_body_indices[:2], :3, :]

    @property
    def matrix_M(self) -> torch.Tensor:
        """Compute the combined upper-body Jacobian mapping for compliance control.

        Assembles a ``(6, 17)`` matrix per env that maps the 17 upper-body joint
        velocities (3 waist + 7 left arm + 7 right arm) to 6D end-effector
        velocities (3 left + 3 right translational).

        Returns:
            Tensor of shape ``(num_envs, 6, 17)``.
        """
        eef_jacobian = self.eef_jacobian
        left_eef_jac = eef_jacobian[:, 0, :, self.left_eef_deps_ids + 6]
        right_eef_jac = eef_jacobian[:, 1, :, self.right_eef_deps_ids + 6]
        waist_jac_left = eef_jacobian[:, 0, :, self.waist_deps_ids + 6]
        waist_jac_right = eef_jacobian[:, 1, :, self.waist_deps_ids + 6]

        M = torch.zeros(self.num_envs, 6, 17).to(self.device)
        M[:, :3, :3] = waist_jac_left
        M[:, 3:, :3] = waist_jac_right
        M[:, :3, 3:10] = left_eef_jac
        M[:, 3:, 10:] = right_eef_jac
        return M

    def _resample_command(self, env_ids: Sequence[int]):
        """Reset force-related buffers when environments are reset.

        This prevents stale force values from affecting compliance calculations
        immediately after environment reset.

        NOTE: We intentionally do NOT reset force_push_counter here.
        The counter needs to reach force_update_frequency (default: 100) before
        forces are applied. If we reset it on every env reset, and episodes are
        shorter than 100 steps, forces would never be applied!
        """
        if len(env_ids) > 0:
            self.last_force_applied[env_ids] = 0.0

    def _update_command(self):
        """No-op; force state is updated by event terms, not per-step."""
        pass

    def _update_compliance_force_push_related_metrics(self):
        """Compute and log per-body force magnitudes, compliance levels, and encoder ratios.

        Populates ``self.metrics`` with force norms per body, compliance levels
        per encoder type (G1/teleop/SMPL), stiff vs. compliant environment ratios,
        and cross-referenced force-by-compliance-status metrics. Optionally prints
        debug summaries every ``_debug_print_every_n_steps`` steps.
        """
        self._total_steps += 1

        # =====================================================================
        # Get encoder masks from TrackingCommand (for per-encoder metrics)
        # encoder_index: [num_envs, num_encoders] one-hot encoding
        # =====================================================================
        motion_command = None
        encoder_g1_mask = None
        encoder_teleop_mask = None
        encoder_smpl_mask = None
        has_encoder_info = False

        try:
            motion_command = self._env.command_manager.get_term("motion")
            if hasattr(motion_command, "encoder_index") and hasattr(
                motion_command, "encoder_sample_probs_dict"
            ):
                encoder_names = list(motion_command.encoder_sample_probs_dict.keys())
                encoder_index = motion_command.encoder_index  # [num_envs, num_encoders]

                # Get masks for each encoder type
                if "g1" in encoder_names:
                    g1_idx = encoder_names.index("g1")
                    encoder_g1_mask = encoder_index[:, g1_idx].bool()
                if "teleop" in encoder_names:
                    teleop_idx = encoder_names.index("teleop")
                    encoder_teleop_mask = encoder_index[:, teleop_idx].bool()
                if "smpl" in encoder_names:
                    smpl_idx = encoder_names.index("smpl")
                    encoder_smpl_mask = encoder_index[:, smpl_idx].bool()

                has_encoder_info = True
        except Exception:  # noqa: BLE001, S110
            pass  # Motion command not available, skip encoder-specific metrics

        # =====================================================================
        # Per-body force metrics
        # last_force_applied: [num_envs, num_force_push_bodies, 3]
        # force_push_body order: ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
        # =====================================================================
        force_norm = torch.norm(self.last_force_applied, dim=-1)  # [num_envs, num_bodies]
        num_force_bodies = self.last_force_applied.shape[1]

        # Per-body force magnitudes (for wrists and torso)
        if num_force_bodies >= 1:
            self.metrics["force_left_wrist"] = force_norm[:, 0]
        if num_force_bodies >= 2:
            self.metrics["force_right_wrist"] = force_norm[:, 1]
        if num_force_bodies >= 3:
            self.metrics["force_torso"] = force_norm[:, 2]

        # Combined wrist force
        if num_force_bodies >= 2:
            self.metrics["force_wrists_sum"] = force_norm[:, 0] + force_norm[:, 1]
            self.metrics["force_wrists_max"] = torch.max(force_norm[:, 0], force_norm[:, 1])

        # Overall force stats
        self.metrics["force_applied"] = force_norm.mean(dim=-1)
        self.metrics["force_applied_max"] = force_norm.max(dim=-1).values

        # Track if any force was actually applied
        force_nonzero = (force_norm.sum() > 0.01).float()
        if force_nonzero > 0:
            self._force_update_count += 1
        self.metrics["force_nonzero_ratio"] = torch.tensor(
            self._force_update_count / max(1, self._total_steps), device=self.device
        ).expand(self.num_envs)

        # =====================================================================
        # Compliance distribution masks
        # eef_stiffness_buf: [num_envs, 3] -> [left_wrist, right_wrist, head]
        # =====================================================================
        compliance_threshold = 0.001  # Threshold to consider compliance as "active"
        is_compliant = self.eef_stiffness_buf[:, :2].abs().sum(dim=-1) > compliance_threshold
        is_stiff = ~is_compliant

        # =====================================================================
        # ENCODER RATIO METRICS
        # ratio_of_total_envs/encoder_*: Fraction of envs using each encoder
        # =====================================================================
        if has_encoder_info:
            if encoder_g1_mask is not None:
                n_g1 = encoder_g1_mask.sum().float()
                self.metrics["ratio_of_total_envs/encoder_g1"] = (n_g1 / self.num_envs).expand(
                    self.num_envs
                )
            if encoder_teleop_mask is not None:
                n_teleop = encoder_teleop_mask.sum().float()
                self.metrics["ratio_of_total_envs/encoder_teleop"] = (
                    n_teleop / self.num_envs
                ).expand(self.num_envs)
            if encoder_smpl_mask is not None:
                n_smpl = encoder_smpl_mask.sum().float()
                self.metrics["ratio_of_total_envs/encoder_smpl"] = (n_smpl / self.num_envs).expand(
                    self.num_envs
                )

        # =====================================================================
        # COMPLIANCE LEVEL METRICS PER ENCODER
        # compliance_level_LH/encoder_*: Mean compliance for each encoder type
        # =====================================================================
        if has_encoder_info:
            # G1 encoder (should always be stiff = 0)
            if encoder_g1_mask is not None and encoder_g1_mask.sum() > 0:
                g1_compliance = self.eef_stiffness_buf[encoder_g1_mask]
                self.metrics["compliance_level_LH/encoder_g1"] = (
                    g1_compliance[:, 0].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_RH/encoder_g1"] = (
                    g1_compliance[:, 1].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_Head/encoder_g1"] = (
                    g1_compliance[:, 2].mean().expand(self.num_envs)
                )
            else:
                self.metrics["compliance_level_LH/encoder_g1"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_RH/encoder_g1"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_Head/encoder_g1"] = torch.zeros(
                    self.num_envs, device=self.device
                )

            # Teleop encoder
            if encoder_teleop_mask is not None and encoder_teleop_mask.sum() > 0:
                teleop_compliance = self.eef_stiffness_buf[encoder_teleop_mask]
                self.metrics["compliance_level_LH/encoder_teleop"] = (
                    teleop_compliance[:, 0].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_RH/encoder_teleop"] = (
                    teleop_compliance[:, 1].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_Head/encoder_teleop"] = (
                    teleop_compliance[:, 2].mean().expand(self.num_envs)
                )
            else:
                self.metrics["compliance_level_LH/encoder_teleop"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_RH/encoder_teleop"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_Head/encoder_teleop"] = torch.zeros(
                    self.num_envs, device=self.device
                )

            # SMPL encoder
            if encoder_smpl_mask is not None and encoder_smpl_mask.sum() > 0:
                smpl_compliance = self.eef_stiffness_buf[encoder_smpl_mask]
                self.metrics["compliance_level_LH/encoder_smpl"] = (
                    smpl_compliance[:, 0].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_RH/encoder_smpl"] = (
                    smpl_compliance[:, 1].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_Head/encoder_smpl"] = (
                    smpl_compliance[:, 2].mean().expand(self.num_envs)
                )
            else:
                self.metrics["compliance_level_LH/encoder_smpl"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_RH/encoder_smpl"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_Head/encoder_smpl"] = torch.zeros(
                    self.num_envs, device=self.device
                )

        # =====================================================================
        # FORCE METRICS BY COMPLIANCE STATUS
        # compliance_related_force_on_LH/nonzero_compliance_envs: Force on envs with compliance > threshold
        # compliance_related_force_on_LH/stiff_envs: Force on stiff envs (G1 + stiff teleop/smpl)
        # =====================================================================
        if num_force_bodies >= 1:
            # Forces on envs with nonzero compliance (teleop/SMPL with active compliance)
            if is_compliant.sum() > 0:
                force_compliant = force_norm[is_compliant]
                self.metrics["compliance_related_force_on_LH/nonzero_compliance_envs"] = (
                    force_compliant[:, 0].mean().expand(self.num_envs)
                )
                if num_force_bodies >= 2:
                    self.metrics["compliance_related_force_on_RH/nonzero_compliance_envs"] = (
                        force_compliant[:, 1].mean().expand(self.num_envs)
                    )
                if num_force_bodies >= 3:
                    self.metrics["compliance_related_force_on_Head/nonzero_compliance_envs"] = (
                        force_compliant[:, 2].mean().expand(self.num_envs)
                    )
            else:
                self.metrics["compliance_related_force_on_LH/nonzero_compliance_envs"] = (
                    torch.zeros(self.num_envs, device=self.device)
                )
                if num_force_bodies >= 2:
                    self.metrics["compliance_related_force_on_RH/nonzero_compliance_envs"] = (
                        torch.zeros(self.num_envs, device=self.device)
                    )
                if num_force_bodies >= 3:
                    self.metrics["compliance_related_force_on_Head/nonzero_compliance_envs"] = (
                        torch.zeros(self.num_envs, device=self.device)
                    )

            # Forces on stiff envs (G1 envs + teleop/smpl envs with stiff mode)
            if is_stiff.sum() > 0:
                force_stiff = force_norm[is_stiff]
                self.metrics["compliance_related_force_on_LH/stiff_envs"] = (
                    force_stiff[:, 0].mean().expand(self.num_envs)
                )
                if num_force_bodies >= 2:
                    self.metrics["compliance_related_force_on_RH/stiff_envs"] = (
                        force_stiff[:, 1].mean().expand(self.num_envs)
                    )
                if num_force_bodies >= 3:
                    self.metrics["compliance_related_force_on_Head/stiff_envs"] = (
                        force_stiff[:, 2].mean().expand(self.num_envs)
                    )
            else:
                self.metrics["compliance_related_force_on_LH/stiff_envs"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                if num_force_bodies >= 2:
                    self.metrics["compliance_related_force_on_RH/stiff_envs"] = torch.zeros(
                        self.num_envs, device=self.device
                    )
                if num_force_bodies >= 3:
                    self.metrics["compliance_related_force_on_Head/stiff_envs"] = torch.zeros(
                        self.num_envs, device=self.device
                    )

        # =====================================================================
        # Debug print every N steps
        # =====================================================================
        if (
            self._debug_print_every_n_steps > 0
            and self._total_steps % self._debug_print_every_n_steps == 0
        ):
            force_mean = force_norm.mean(dim=0)
            force_vals = [f"{force_mean[i].item():.2f}" for i in range(min(3, num_force_bodies))]
            force_str = ", ".join(force_vals) if force_vals else "N/A"
            body_labels = ["L_wrist", "R_wrist", "Torso"][:num_force_bodies]

            n_compliant = is_compliant.sum().item()
            n_stiff = is_stiff.sum().item()

            # Per-encoder counts
            encoder_str = ""
            if has_encoder_info:
                n_g1 = encoder_g1_mask.sum().item() if encoder_g1_mask is not None else 0
                n_teleop = (
                    encoder_teleop_mask.sum().item() if encoder_teleop_mask is not None else 0
                )
                n_smpl = encoder_smpl_mask.sum().item() if encoder_smpl_mask is not None else 0
                encoder_str = f" | Encoders(G1/Teleop/SMPL): {n_g1}/{n_teleop}/{n_smpl}"

            # Compliance stats for compliant envs only
            if n_compliant > 0:
                compliance_compliant_mean = self.eef_stiffness_buf[is_compliant].mean(dim=0)
                comp_str = f"[{compliance_compliant_mean[0].item():.3f}, {compliance_compliant_mean[1].item():.3f}, {compliance_compliant_mean[2].item():.3f}]"  # noqa: E501
            else:
                comp_str = "[N/A - all stiff]"

            # Debug info about event calls
            event_calls = getattr(self, "_event_call_count", 0)
            counter_stats = f"counter_min={self.force_push_counter.min().item()} max={self.force_push_counter.max().item()}"  # noqa: E501

            print(  # noqa: T201
                f"[ForceDebug @ step {self._total_steps}] "
                f"Force({'/'.join(body_labels)}): [{force_str}] | "
                f"Compliant: {n_compliant} Stiff: {n_stiff}{encoder_str} | "
                f"Compliance(L/R/H): {comp_str} | "
                f"Force ratio: {self._force_update_count}/{self._total_steps} | "
                f"Event calls: {event_calls} | {counter_stats}"
            )

        # =====================================================================
        # Legacy compliance metrics (kept for backward compatibility)
        # =====================================================================
        self.metrics["compliance_left_wrist"] = self.eef_stiffness_buf[:, 0]
        self.metrics["compliance_right_wrist"] = self.eef_stiffness_buf[:, 1]
        self.metrics["compliance_head"] = self.eef_stiffness_buf[:, 2]
        self.metrics["compliance_mean"] = self.eef_stiffness_buf.mean(dim=-1)

        # =====================================================================
        # Compliance ratio metrics FOR NON-G1 ENCODER ENVS ONLY
        # G1 envs are always forced to stiff, so including them is misleading.
        # These metrics show the stiff/compliant distribution among TELEOP/SMPL envs.
        # =====================================================================
        if has_encoder_info and encoder_g1_mask is not None:
            non_g1_mask = ~encoder_g1_mask
            n_non_g1 = non_g1_mask.sum()
            if n_non_g1 > 0:
                stiff_ratio_non_g1 = is_stiff[non_g1_mask].float().mean()
                self.metrics["stiff_ratio_for_non_G1_encoder_envs"] = stiff_ratio_non_g1.expand(
                    self.num_envs
                )
                self.metrics["compliant_ratio_for_non_G1_encoder_envs"] = (
                    1.0 - stiff_ratio_non_g1
                ).expand(self.num_envs)
            else:
                # All envs are G1, ratio is undefined (report 0)
                self.metrics["stiff_ratio_for_non_G1_encoder_envs"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliant_ratio_for_non_G1_encoder_envs"] = torch.zeros(
                    self.num_envs, device=self.device
                )
        else:
            # No encoder info available, fall back to global ratio
            stiff_ratio = is_stiff.float().mean()
            self.metrics["stiff_ratio_for_non_G1_encoder_envs"] = stiff_ratio.expand(self.num_envs)
            self.metrics["compliant_ratio_for_non_G1_encoder_envs"] = (1.0 - stiff_ratio).expand(
                self.num_envs
            )

        # Sanity check warning: If compliance is being varied but no forces are applied
        compliance_is_active = self.compliance_config_init and (
            self.eef_stiffness_buf.abs().sum() > 0
        )
        if compliance_is_active and self._total_steps > 1000:  # noqa: SIM102
            if self._force_update_count == 0 and not self._warned_no_force:
                import warnings  # noqa: PLC0415

                warnings.warn(  # noqa: B028
                    "\n" + "=" * 80 + "\n"
                    "[COMPLIANCE SANITY CHECK FAILED]\n"
                    "Compliance values are being changed, but NO external forces have been applied!\n"
                    "This means vr_3point_local_target_compliant == vr_3point_local_target always.\n"
                    "The policy will NOT learn to use the compliance signal.\n\n"
                    "FIX: Add 'compliance_force_push' event to your events config:\n"
                    "  defaults:\n"
                    "    - terms/compliance_force_push@_here_\n"
                    "=" * 80
                )
                self._warned_no_force = True

    def _update_metrics(self):
        """Refresh max_force from config and compute all force/compliance metrics."""
        self.max_force = self.cfg.max_force
        self._update_compliance_force_push_related_metrics()

    @property
    def command(self):
        """Return None; ForceTrackingCommand has no direct observation tensor."""
        return None


@configclass
class TrackingCommandCfg(CommandTermCfg):
    """Configuration for TrackingCommand motion tracking.

    Controls motion library loading, future reference frame layout, encoder
    sampling probabilities (G1/SMPL/teleop), episode initialization strategy,
    object handling, and debug visualization marker styles.
    """

    class_type: type = TrackingCommand

    asset_name: str = dataclasses.MISSING

    motion_lib_cfg: dict = None
    motion_file: str = None
    smpl_motion_file: str = None
    filter_motion_keys: list[str] = None
    use_paired_motions: bool = False
    # Opt-in joint-roster mode: each environment keeps one immutable asset
    # signature while reset sampling may choose any compatible motion.
    use_asset_compatible_motions: bool = False
    # Ownership-phase episodes (multi-human handoff handling): when enabled,
    # reset starts are constrained to one contiguous actor-ownership phase
    # (meta rp2_ownership_phases, written by rp2_patch_pack_ownership_channels
    # or the actor-owned compose) and the episode ends at that phase's end, so
    # no episode spans a handoff into partner-driven object motion.
    ownership_phase_episodes: bool = False
    # Directory of per-motion meta pkls carrying rp2_ownership_phases
    # (typically <pack>/meta). Required when ownership_phase_episodes is True.
    ownership_meta_dir: str = None
    # Minimum episode frames remaining inside the chosen phase after the
    # sampled start; also the minimum phase length considered.
    ownership_min_episode_frames: int = 30
    # Contact-based initialization: sample timestamps before first contact frame
    # Path to contact pickle file (e.g., data/motion_lib_grab/contact/s2_apple_lift.pkl)
    contact_file: str = None
    # Optional JSON mapping motion keys to first hand-or-body contact in the
    # motion-lib target-FPS frame domain. This changes reset timing only.
    reference_contact_boundary_file: str = None
    # If True, optionally bias reset starts toward frames before first contact.
    sample_before_contact: bool = False
    # "legacy" (default) overrides every start with uniform [0, first_contact - margin);
    # "windowed" is the ContactFirst mixture that keeps the full valid window reachable.
    sample_before_contact_window_mode: str = "legacy"
    # Source of the first-contact boundary used by reset bias. The joint
    # contract uses the same canonical active-entity +1 labels as reward and
    # termination; legacy keeps historical per-hand/contact-file behavior.
    first_contact_reference_source: str = "legacy"
    # Probability of replacing a valid uniform start with a pre-contact start.
    # A value below one preserves full support over the valid start interval.
    sample_before_contact_bias_probability: float = 1.0
    # Minimum lead-in (in target-FPS reference steps) before first contact.
    sample_before_contact_margin: int = 10
    # Preferred source-independent lead-in in physical seconds. When set, it
    # overrides sample_before_contact_margin after conversion to runtime steps.
    sample_before_contact_margin_seconds: float = None
    # Which hand's in_contact label to use for deriving first contact frame
    sample_before_contact_hand: str = "right_hand"
    # Optional versioned reset-start cache. It is opt-in so existing GRAIL and
    # GRAIL-SONIC experiments retain their historical reset behavior.
    reset_start_cache_file: str = None
    # Minimum accepted cache validation: candidate, reference, or isaaclab.
    reset_start_cache_required_validation: str = "reference"
    # Fail during initialization if any loaded motion lacks an accepted range.
    reset_start_cache_require_all: bool = False
    # An explicitly configured cache applies consistently to training and evaluation.
    # Set false only for a deliberate out-of-cache evaluation.
    reset_start_cache_apply_in_evaluation: bool = True
    anchor_body: str = dataclasses.MISSING
    body_names: list[str] = dataclasses.MISSING

    vr_3point_body: list[str] = []  # noqa: RUF012
    vr_3point_body_offset: list[list[float]] = []  # noqa: RUF012

    reward_point_body: list[str] = []  # noqa: RUF012
    reward_point_body_offset: list[list[float]] = []  # noqa: RUF012

    # For backward compatibility (to remove)
    force_push_body: list[str] = []  # noqa: RUF012
    force_push_body_offset: list[list[float]] = []  # noqa: RUF012

    num_future_frames: int = 1
    dt_future_ref_frames: float = 0.1
    randomize_heading: bool = False

    # Variable frame support: when enabled, num_future_frames serves as max_frames
    # and each environment/sample gets a random num_frames from
    # [variable_frames_min, num_future_frames] with step variable_frames_step.
    # Step must be a power of 2 compatible with down_t (step=4 works for down_t=2).
    variable_frames_enabled: bool = False
    variable_frames_min: int = 16
    variable_frames_step: int = 4

    freeze_frame_aug: bool = False
    freeze_frame_aug_prob: float = 0.1

    smpl_num_future_frames: int = None
    smpl_dt_future_ref_frames: float = None

    smpl_num_future_frames: int = None  # noqa: PIE794
    smpl_dt_future_ref_frames: float = None  # noqa: PIE794

    encoder_sample_probs: dict[str, float] = None

    # ==========================================================================
    # CHIP Compliance Training Optimization Flag
    # ==========================================================================
    # When True, enables cleaner encoder logic for compliance-aware training:
    #   1. SMPL-native envs do NOT automatically activate G1 encoder in main loop
    #   2. G1 encoder only runs for G1-native envs (policy tokens)
    #   3. G1 latents for SMPL-native envs are computed ONLY in aux losses
    #      and ONLY when compliance=0 (stiff mode)
    #
    # This eliminates wasted computation where G1 tokens were computed for
    # SMPL-native envs but immediately overwritten by SMPL tokens.
    #
    # Set to False (default) for backward compatibility with non-compliance runs.
    # ==========================================================================
    optimize_encoders_ratio_for_CHIP: bool = False

    # Probability to also sample teleop mode when smpl mode is active
    # This enables teleop-smpl and g1-teleop latent alignment losses
    teleop_sample_prob_when_smpl: float = 0.0

    # Always start from the first frame of the motion file during resampling
    # Useful for debugging and replaying specific motions from the beginning
    start_from_first_frame: bool = False

    # Preserve historical tracking behavior by default. Reference-free
    # generative controllers disable this: _resample_command writes the source
    # motion's robot/object state into the simulator at clip exhaustion.
    resample_on_motion_end: bool = True

    # Evaluation-only deterministic start relative to first reference contact.
    # None preserves the original frame-zero evaluation behavior.
    evaluation_start_before_contact_frames: int = None

    # Sample each motion at most once (no duplicates across environments)
    # Requires num_envs <= num_available_motions, otherwise will error
    # Useful for replay/evaluation to ensure coverage of all unique motions
    sample_unique_motions: bool = False

    # Sample from the first N frames of the motion (random uniform in [0, N-1])
    # If set, this takes precedence over start_from_first_frame
    # Useful for adding slight variation while still starting near the beginning
    sample_from_n_initial_frames: int = None

    # Reserve this many seconds at the end of a motion when sampling reset starts.
    # None keeps the legacy full-window sampling used by the released
    # task-general tracking configs; the ContactFirst manipulation chain sets
    # 10.0 to match its ten-second environment episode.
    sample_start_truncate_time_s: float = None

    # Source-only RP2 mode. The default remains the original motion-reference
    # reset. When enabled, reset uses neutral G1 joints plus scaled SOMA root
    # XY/yaw channels and never writes retargeted G1 joint/root state.
    source_only_reset: bool = False
    source_reset_use_source_z: bool = True
    source_reset_min_root_height: float = 0.72
    # Optional initialization-only curriculum. Retargeted G1 state may seed a
    # reset, but is never exposed to observations, rewards, or terminations.
    source_reset_retarget_probability_start: float = 0.0
    source_reset_retarget_probability_end: float = 0.0
    source_reset_curriculum_steps: int = 5000
    source_reset_force_retarget_nonstanding: bool = False
    source_reset_nonstanding_joint_rms: float = 0.35
    source_reset_retarget_min_root_height: float = 0.30
    source_reset_use_reference_hand_action: bool = False
    # Reset the extra (hand) joints to the EXACT reference dof[29:43] (mapped BY JOINT NAME in
    # HAND_ORDER_USD == IsaacLab articulation order) instead of hand_default_positions. Requires
    # motion_lib_cfg.hand_dof_count>0 so hand_dof_pos is loaded. False preserves the zero-default.
    reset_hand_from_reference: bool = False
    # Initialize a sampled moving object with its reference linear/angular
    # velocity. False preserves the original zero-velocity reset.
    object_reset_use_reference_velocity: bool = False
    # Hold-and-release for episodes that reset into a reference frame that is
    # already in contact: the object is driven along its reference trajectory
    # for the first `object_hold_frames` control steps after such a reset. The
    # reference correction follows a cosine fade to free physics. This gives
    # the policy time to establish the grasp instead of racing a free-falling
    # object it was never holding. Single-object scenes only and disabled by
    # default to preserve original GRAIL-SONIC behavior.
    object_hold_after_in_contact_reset: bool = False
    object_hold_frames: int = 10
    # Optional two-phase in-contact reset replay. Positive values replace the
    # legacy single cosine fade with exact reference replay followed by a
    # separate cosine release. Both zero preserves object_hold_frames exactly.
    object_hold_replay_frames: int = 0
    object_hold_release_frames: int = 0
    # Keep a dynamic single target on its reference trajectory while the
    # episode is still before the first left-or-right reference hand contact.
    # A short grace period supports contact establishment before free physics.
    # Disabled by default to preserve the original GRAIL-SONIC behavior.
    object_hold_before_reference_contact: bool = False
    object_hold_release_grace_frames: int = 10
    # Optional RP2 one-object corrections. Per-motion mass metadata comes from
    # manager_env.config.object_physics_overrides_path. Reference locking holds
    # stationary support targets at the pose sampled during reset. Both remain
    # disabled by default to preserve legacy GRAIL-SONIC behavior.
    object_mass_from_physics_overrides: bool = False
    object_reference_lock_roles: list[str] = None

    # Object position randomization (adds random offset to object position at reset)
    object_position_randomize: bool = False
    # In multi_object_per_env mode, keep all object_i assets in the scene and
    # reset them from root_pos/root_quat[:, i] instead of activating one object.
    full_scene_per_env: bool = False
    # Optional full-scene reset audit JSON path. When set, reset records include
    # motion/object pairing, per-object first-frame reference poses, BPS presence,
    # and role metadata. This is intentionally opt-in for validation runs.
    audit_path: str = None
    audit_max_resets: int = 0
    object_roles_path: str = None
    # When object motion data includes object_interaction_mask, legacy single-object
    # terms track the currently active object set instead of a static object index.
    use_dynamic_target_object_mask: bool = True
    # Log RP2 per-role object active counts and position errors as command metrics.
    enable_rp2_object_role_metrics: bool = True
    rp2_role_metric_pos_exp_coeff: float = -100.0
    rp2_role_metric_pos_deviation_threshold: float = 0.1
    rp2_support_body_metric_names: list[str] = [  # noqa: RUF012
        "pelvis",
        "torso_link",
        "left_hip_roll_link",
        "left_knee_link",
        "right_hip_roll_link",
        "right_knee_link",
    ]
    rp2_ee_body_height_metric_names: list[str] = [  # noqa: RUF012
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    rp2_ee_body_height_metric_threshold: float = 0.25
    enable_rp2_support_diagnostic_labels: bool = True
    rp2_diag_moving_support_pos_error_threshold: float = 0.5
    rp2_diag_moving_support_ori_error_threshold: float = 1.5
    rp2_diag_moving_support_height_error_threshold: float = 0.3
    rp2_diag_support_contact_force_threshold: float = 500.0
    rp2_diag_support_contact_force_active_threshold: float = 250.0
    rp2_diag_support_body_pos_error_threshold: float = 0.2
    rp2_diag_ee_body_height_error_threshold: float = 0.25
    # Randomization range for each axis: {"x": 0.05, "y": 0.05, "z": 0.0}
    # Values are half-range, so 0.05 means uniform random in [-0.05, 0.05]
    object_position_randomization: dict[str, float] = None

    # Table position offset: [x, y, z] added to table position from meta file
    # Useful for adjusting table position relative to the robot
    table_offset: list[float] = None

    # ==========================================================================
    # Terrain-Aware Sampling
    # ==========================================================================
    # When True, envs are split into flat and terrain groups based on their
    # fixed USD assignment. Flat envs resample motions freely (adaptive sampling).
    # Terrain envs keep their motion fixed (only resample start time).
    # The split is determined by flat_to_terrain_ratio: every (R+1)th env is terrain.
    terrain_aware_sampling: bool = False
    flat_to_terrain_ratio: int = 4  # R flat envs per 1 terrain env
    init_z_offset: float = 0.0  # z offset added to root_pos at reset (compensate collision mesh)

    pose_range: dict[str, tuple[float, float]] = {}  # noqa: RUF012
    velocity_range: dict[str, tuple[float, float]] = {}  # noqa: RUF012

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    joint_velocity_range: tuple[float, float] = (-0, 0)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    body_pos_visualizer_cfg: VisualizationMarkersCfg = DEFORMABLE_TARGET_MARKER_CFG.replace(
        prim_path="/Visuals/goal_marker_sphere"
    )
    body_pos_visualizer_cfg.markers["target"].radius = 0.05
    body_pos_visualizer_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 1.0, 0.0)
    )

    feet_contact_visualizer_cfg: VisualizationMarkersCfg = DEFORMABLE_TARGET_MARKER_CFG.replace(
        prim_path="/Visuals/goal_marker_sphere"
    )
    feet_contact_visualizer_cfg.markers = {  # noqa: RUF012
        "target_small": sim_utils.SphereCfg(
            radius=0.0001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "target_big": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }

    waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]  # noqa: RUF012

    left_eef_deps = [  # noqa: RUF012
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
    ]

    right_eef_deps = [  # noqa: RUF012
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    ]

    force_push_body: list[str] = []  # default only two wrists  # noqa: PIE794
    force_push_body_offset: list[list[float]] = []  # noqa: PIE794

    feet_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]  # noqa: RUF012
    cat_upper_body_poses: bool = False
    cat_upper_body_poses_prob: float = 0.5
    randomize_wrist_poses: bool = False
    randomize_wrist_prob: float = 0.3
    randomize_wrist_std: float = 0.1  # radians (~5.7 degrees)

    use_height_map: bool = False
    height_map_resolution: float = 0.15
    height_map_size: float = 1.5
    height_map_max_dist: float = 5.0
    motion_lib_num_dof: int | None = None  # If None, assumes motion lib DOF matches robot DOF
    hand_default_positions: list[float] | None = (
        None  # Default positions for extra joints (e.g., hands)
    )
    hand_default_velocities: list[float] | None = (
        None  # Default velocities for extra joints (defaults to 0)
    )
    hand_joint_position_range: tuple[float, float] | None = (
        None  # Optional reset noise range for extra joints; defaults to joint_position_range
    )
    hand_joint_velocity_range: tuple[float, float] | None = (
        None  # Optional reset velocity range for extra joints; defaults to joint_velocity_range
    )
    # Object z-offset (e.g., -0.05 to lower chair 5cm into ground)
    object_z_offset: float = 0.0
    # Object initial z-offset — only applied at reset, not during motion tracking
    object_init_z_offset: float = 0.0

    # When False, suppress body/skeleton reference markers in debug vis
    # (contact center spheres still render). Useful for cleaner video captures.
    debug_vis_body: bool = True
    # Evaluation diagnostic only: green=current active object, red=reference target.
    debug_vis_object: bool = False
    debug_vis_object_radius: float = 0.06
    debug_vis_object_z_offset: float = 0.12


@configclass
class ForceTrackingCommandCfg(CommandTermCfg):
    """Configuration for ForceTrackingCommand external perturbation and compliance."""

    class_type: type = ForceTrackingCommand

    asset_name: str = dataclasses.MISSING

    anchor_body: str = dataclasses.MISSING
    body_names: list[str] = dataclasses.MISSING

    force_update_frequency: int = 100
    max_force: float = 20.0

    vr_3point_body: list[str] = []  # noqa: RUF012
    vr_3point_body_offset: list[list[float]] = []  # noqa: RUF012

    force_push_body: list[str] = []  # default only two wrists  # noqa: PIE794
    force_push_body_offset: list[list[float]] = []  # noqa: PIE794

    # Debug print frequency (0 = disabled, nonzero = print every N steps)
    # Usage: manager_env.commands.force.debug_print_every_n_steps=10
    debug_print_every_n_steps: int = 0

    waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]  # noqa: RUF012

    left_eef_deps = [  # noqa: RUF012
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
    ]

    right_eef_deps = [  # noqa: RUF012
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    ]


def _get_body_indexes(command: TrackingCommand, body_names: list[str] | None) -> list[int]:
    """Return indices into command.cfg.body_names for the requested subset.

    Args:
        command: TrackingCommand instance whose cfg.body_names defines the full body list.
        body_names: Subset of body names to select. If None, return all indices.

    Returns:
        List of integer indices into ``command.cfg.body_names``.
    """
    return [
        idx
        for idx, body_name in enumerate(command.cfg.body_names)
        if body_names is None or body_name in body_names
    ]


# Backward compat — remove after all checkpoints migrated
MotionCommand = TrackingCommand
MotionCommandCfg = TrackingCommandCfg
ForceCommand = ForceTrackingCommand
ForceCommandCfg = ForceTrackingCommandCfg
