from typing import TYPE_CHECKING  # noqa: I001
from functools import wraps

import numpy as np
from omegaconf import OmegaConf
import omni
from pxr import Gf, UsdGeom
import torch
from loguru import logger
from gear_sonic.trl.utils.common import custom_instantiate

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv

# Import joint index functions (single source of truth)
from gear_sonic.envs.env_utils.joint_utils import get_body_joint_indices, get_hand_joint_indices
from gear_sonic.envs.manager_env.mdp.commands import (
    INACTIVE_OBJECT_BASE_OFFSET,
    INACTIVE_OBJECT_Z_SPACING,
    motion_order_object_scene_keys,
)
from gear_sonic.rp2.discrete_contact import canonical_hand_contact_indices
from gear_sonic.utils.fsq_tokens import (
    DIRECT_FSQ_ACTION_MODE,
    decode_post_fsq_codes,
    require_policy_atm_proprioception,
    resolve_gpc_action_mode,
)
from gear_sonic.utils.contact_sensor_parity import ContactSensorParityProbe
from gear_sonic.utils.frozen_module import freeze_module
from gear_sonic.utils.runtime_sync_cleanup import (
    RuntimeSyncCleanupCounters,
    RuntimeSyncCleanupOptions,
)
from gear_sonic.utils.rendering import (
    camera_pose_from_object_axis,
    camera_pose_radial_from_root,
    camera_pose_with_target_offset,
    load_camera_source_orientation_trace,
    resolve_camera_focus_body_indices,
    resolve_camera_visibility_gprim_policy,
    write_camera_runtime_trace,
)
from gear_sonic.utils.runtime_debug import (
    log_invalid_stats,
    tensor_batch_invalid_stats,
)
from gear_sonic.utils.safe_nan_utils import (
    collect_bad_env_ids_from_obs,
    merge_reset_obs,
    sanitize_obs_for_env_ids,
    tensor_bad_env_ids,
)

# Import visualization markers for contact point visualization
try:
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def _restore_static_context_after_replay(replay):
    """Restore context poses on every exit from the existing replay method."""
    @wraps(replay)
    def run(self, *args, **kwargs):
        try:
            return replay(self, *args, **kwargs)
        finally:
            self._restore_replay_static_context()

    return run


def _replay_required_contact_metrics(metrics):
    """Return the exact required-contact union and complement for one frame."""
    legacy_by_side = {
        side: float(
            metrics.get(f"rp2_grasp_reference_contact/{side}", 0.0)
        )
        > 0.5
        for side in ("left", "right")
    }
    canonical_by_side = {
        side: float(
            metrics.get(f"canonical_hand_reference_contact/{side}", 0.0)
        )
        > 0.5
        for side in ("left", "right")
    }
    legacy_either_hand = any(legacy_by_side.values())
    canonical_either_hand = any(canonical_by_side.values())
    either_hand = legacy_either_hand or canonical_either_hand
    required = either_hand or (
        float(metrics.get("canonical_required_contact/count", 0.0)) > 0.5
    )
    return {
        "rp2_grasp_reference_contact/either": float(legacy_either_hand),
        "canonical_hand_reference_contact/either": float(
            canonical_either_hand
        ),
        "reference_hand_contact/left": float(
            legacy_by_side["left"] or canonical_by_side["left"]
        ),
        "reference_hand_contact/right": float(
            legacy_by_side["right"] or canonical_by_side["right"]
        ),
        "reference_hand_contact/either": float(either_hand),
        "canonical_required_contact/union": float(required),
        "canonical_required_contact/none": float(not required),
    }


def _replay_required_contact_summary(records):
    """Summarize exact frame-level required-contact partition evidence."""
    either_hand_samples = []
    required_samples = []
    none_samples = []
    per_motion = {}
    for record in records:
        metrics = record.get("metrics", {})
        derived = _replay_required_contact_metrics(metrics)
        for name, expected in derived.items():
            actual = float(metrics.get(name, expected))
            if actual != expected:
                raise ValueError(
                    f"Replay required-contact metric {name!r} disagrees with "
                    "its frame-level source metrics"
                )
        either_hand = bool(derived["reference_hand_contact/either"])
        required = bool(derived["canonical_required_contact/union"])
        none = bool(derived["canonical_required_contact/none"])

        sample = {
            "motion_key": str(record["motion_key"]),
            "env_id": int(record["env_id"]),
            "frame": int(record["frame"]),
        }
        if either_hand:
            either_hand_samples.append(sample)
        destination = required_samples if required else none_samples
        destination.append(sample)

        motion = per_motion.setdefault(
            sample["motion_key"],
            {
                "sample_count": 0,
                "either_hand_contact_count": 0,
                "required_contact_count": 0,
                "no_required_contact_count": 0,
                "either_hand_contact_frame_indices": [],
                "required_contact_frame_indices": [],
                "no_required_contact_frame_indices": [],
                "either_hand_contact_samples": [],
                "required_contact_samples": [],
                "no_required_contact_samples": [],
            },
        )
        motion["sample_count"] += 1
        if either_hand:
            motion["either_hand_contact_count"] += 1
            motion["either_hand_contact_frame_indices"].append(sample["frame"])
            motion["either_hand_contact_samples"].append(
                {"env_id": sample["env_id"], "frame": sample["frame"]}
            )
        if required:
            motion["required_contact_count"] += 1
            motion["required_contact_frame_indices"].append(sample["frame"])
            motion["required_contact_samples"].append(
                {"env_id": sample["env_id"], "frame": sample["frame"]}
            )
        else:
            motion["no_required_contact_count"] += 1
            motion["no_required_contact_frame_indices"].append(sample["frame"])
            motion["no_required_contact_samples"].append(
                {"env_id": sample["env_id"], "frame": sample["frame"]}
            )

    either_hand_samples.sort(
        key=lambda item: (item["motion_key"], item["env_id"], item["frame"])
    )
    required_samples.sort(
        key=lambda item: (item["motion_key"], item["env_id"], item["frame"])
    )
    none_samples.sort(
        key=lambda item: (item["motion_key"], item["env_id"], item["frame"])
    )
    for motion in per_motion.values():
        motion["either_hand_contact_frame_indices"].sort()
        motion["required_contact_frame_indices"].sort()
        motion["no_required_contact_frame_indices"].sort()
        motion["either_hand_contact_samples"].sort(
            key=lambda item: (item["env_id"], item["frame"])
        )
        motion["required_contact_samples"].sort(
            key=lambda item: (item["env_id"], item["frame"])
        )
        motion["no_required_contact_samples"].sort(
            key=lambda item: (item["env_id"], item["frame"])
        )

    sample_count = len(records)
    return {
        "metric_names": {
            "legacy_hand_union": "rp2_grasp_reference_contact/either",
            "canonical_hand_union": "canonical_hand_reference_contact/either",
            "hand_union": "reference_hand_contact/either",
            "union": "canonical_required_contact/union",
            "complement": "canonical_required_contact/none",
        },
        "sample_count": sample_count,
        "either_hand_contact_count": len(either_hand_samples),
        "required_contact_count": len(required_samples),
        "no_required_contact_count": len(none_samples),
        "partition_is_exact": (
            len(required_samples) + len(none_samples) == sample_count
        ),
        "either_hand_contact_samples": either_hand_samples,
        "required_contact_samples": required_samples,
        "no_required_contact_samples": none_samples,
        "per_motion": dict(sorted(per_motion.items())),
    }


def _replay_hand_contact_alignment_summary(records):
    """Return exact reference-versus-measured hand-contact evidence."""

    output = {}
    for side in ("left", "right"):
        measured_name = f"rp2_grasp_sim_contact_count/{side}"
        canonical_reference_name = f"canonical_hand_reference_contact/{side}"
        canonical_eligible_name = f"canonical_hand_reference_eligible/{side}"
        legacy_reference_name = f"rp2_grasp_reference_contact/{side}"
        legacy_eligible_name = f"rp2_grasp_role_eligible/{side}"
        eligible_records = []
        for record in records:
            metrics = record.get("metrics", {})
            if float(metrics.get(legacy_eligible_name, 0.0)) > 0.5:
                eligible_records.append(
                    (record, legacy_reference_name, "rp2_role")
                )
            elif float(metrics.get(canonical_eligible_name, 0.0)) > 0.5:
                eligible_records.append(
                    (record, canonical_reference_name, "canonical_dex31")
                )
        samples = {
            "true_positive": [],
            "false_negative": [],
            "false_positive": [],
            "true_negative": [],
        }
        authority_counts = {"canonical_dex31": 0, "rp2_role": 0}
        for record, reference_name, authority in eligible_records:
            metrics = record.get("metrics", {})
            reference = float(metrics.get(reference_name, 0.0)) > 0.5
            measured = float(metrics.get(measured_name, 0.0)) > 0.5
            authority_counts[authority] += 1
            if reference and measured:
                outcome = "true_positive"
            elif reference:
                outcome = "false_negative"
            elif measured:
                outcome = "false_positive"
            else:
                outcome = "true_negative"
            samples[outcome].append(
                {
                    "motion_key": str(record["motion_key"]),
                    "env_id": int(record["env_id"]),
                    "frame": int(record["frame"]),
                }
            )
        counts = {name: len(values) for name, values in samples.items()}
        reference_positive = (
            counts["true_positive"] + counts["false_negative"]
        )
        measured_positive = (
            counts["true_positive"] + counts["false_positive"]
        )
        output[side] = {
            # Keep the legacy field names for existing RP2 audit consumers.
            # ``*_priority`` and ``reference_authority_counts`` describe the
            # actual per-frame fallback used by this schema version.
            "reference_metric": legacy_reference_name,
            "eligibility_metric": legacy_eligible_name,
            "reference_metric_priority": [
                legacy_reference_name,
                canonical_reference_name,
            ],
            "measured_metric": measured_name,
            "eligibility_metric_priority": [
                legacy_eligible_name,
                canonical_eligible_name,
            ],
            "reference_authority_counts": authority_counts,
            "eligible_sample_count": len(eligible_records),
            "reference_positive_count": reference_positive,
            "measured_positive_count": measured_positive,
            "true_positive_count": counts["true_positive"],
            "false_negative_count": counts["false_negative"],
            "false_positive_count": counts["false_positive"],
            "true_negative_count": counts["true_negative"],
            "recall": (
                float(counts["true_positive"] / reference_positive)
                if reference_positive
                else None
            ),
            "precision": (
                float(counts["true_positive"] / measured_positive)
                if measured_positive
                else None
            ),
            "samples": samples,
        }
    return output


def _replay_termination_frame_summary(records, termination_name, motion_info=None):
    """Return exact frame evidence for one replay termination term."""
    terminated_samples = [
        {
            "motion_key": str(record["motion_key"]),
            "env_id": int(record["env_id"]),
            "frame": int(record["frame"]),
        }
        for record in records
        if bool(record.get("terminations", {}).get(termination_name, False))
    ]
    terminated_samples.sort(
        key=lambda item: (item["motion_key"], item["env_id"], item["frame"])
    )
    info_by_motion = {
        str(info["motion_key"]): info
        for info in (motion_info or [])
        if "motion_key" in info and "num_steps" in info
    }

    per_motion = {}
    for sample in terminated_samples:
        motion = per_motion.setdefault(
            sample["motion_key"],
            {
                "count": 0,
                "frame_indices": [],
                "samples": [],
            },
        )
        motion["count"] += 1
        motion["frame_indices"].append(sample["frame"])
        motion["samples"].append(
            {"env_id": sample["env_id"], "frame": sample["frame"]}
        )

    for motion_key, motion in per_motion.items():
        motion["frame_indices"].sort()
        motion["samples"].sort(key=lambda item: (item["env_id"], item["frame"]))
        motion["first_frame"] = motion["frame_indices"][0]
        motion["last_frame"] = motion["frame_indices"][-1]
        info = info_by_motion.get(motion_key, {})
        final_frame = (
            int(info["num_steps"]) - 1 if "num_steps" in info else None
        )
        motion["reference_final_frame"] = final_frame
        motion["only_at_reference_final_frame"] = (
            all(frame == final_frame for frame in motion["frame_indices"])
            if final_frame is not None
            else None
        )
        if termination_name == "time_out":
            horizon_steps = int(info.get("episode_horizon_steps", 0) or 0)
            expected_first = (
                min(horizon_steps - 1, final_frame)
                if horizon_steps > 0 and final_frame is not None
                else None
            )
            motion["episode_horizon_steps"] = (
                horizon_steps if horizon_steps > 0 else None
            )
            motion["expected_first_frame"] = expected_first
            frames_by_env = {}
            for sample in motion["samples"]:
                frames_by_env.setdefault(sample["env_id"], []).append(
                    sample["frame"]
                )
            expected_range = (
                list(range(expected_first, final_frame + 1))
                if expected_first is not None
                and final_frame is not None
                and expected_first <= final_frame
                else None
            )
            motion["matches_expected_active_range"] = (
                expected_range is not None
                and bool(frames_by_env)
                and all(
                    sorted(frames) == expected_range
                    for frames in frames_by_env.values()
                )
            )

    frame_indices = sorted(sample["frame"] for sample in terminated_samples)
    return {
        "frame_indices": frame_indices,
        "first_frame": frame_indices[0] if frame_indices else None,
        "last_frame": frame_indices[-1] if frame_indices else None,
        "samples": terminated_samples,
        "per_motion": dict(sorted(per_motion.items())),
    }


class ManagerEnvWrapper:
    def __init__(self, env: "ManagerBasedEnv", config):
        env.wrapper = self
        self.env = env
        self.config = OmegaConf.create(config)
        self.device = env.device
        self.viewer_focused = True
        self.is_manager_env = True
        if hasattr(self.env, "num_envs"):
            self.num_envs = self.env.num_envs
        else:
            self.num_envs = self.env.env.unwrapped.num_envs
        try:
            self.motion_command = env.command_manager.get_term("motion")
            self._motion_lib = self.motion_command.motion_lib
        except:  # noqa: E722
            logger.info("No motion lib found")
            self.motion_command = None
            self._motion_lib = None
        try:
            self.force_command = env.command_manager.get_term("force")
        except:  # noqa: E722
            self.force_command = None
        self.is_evaluating = False
        self.start_idx = 0
        self._last_predicted_object_pos = (
            None  # For debug visualization of predicted object position
        )
        if not self.config.headless:
            self.setup_keyboard()

        # Action visualization toggle and state
        self.turn_on_visualization = bool(self.config.get("turn_on_visualization", False)) and (
            not self.config.get("headless", False)
        )
        self._viz_every_n_steps = int(self.config.get("viz_every_n_steps", 1))
        self._plot_action_dim = int(self.config.get("action_plot_dim", 29))
        clip_default = self.config.get("action_clip_value", 1.0)  # noqa: F841
        self._action_ylim = float(self.config.get("action_plot_ylim", 10.0))
        self._plot_window = int(self.config.get("action_plot_window", 200))
        self._step_counter = 0
        self._action_fig = None
        self._action_lines = None
        self._action_hist = None
        self._hist_idx = 0

        self._blit_background = None
        self._blit_supported = True
        self._blit_refresh_interval = int(self.config.get("plot_blit_refresh_interval", 120))

        # Initialize action transform module from config
        self.action_transform_module = None
        self._needs_policy_atm = False  # Default: use policy obs directly
        self._policy_atm_indices = None

        # Initialize joint index mappings (for DOF mismatch handling in replay and step)
        self._body_joint_indices = None
        self._hand_joint_indices = None
        self._setup_replay_joint_indices()  # Setup for replay mode
        # Inferred boundary support (gear_sonic/rp2/inferred_support.py): the synthetic slabs are kinematic colliders
        # the OBJECT rests on; the robot does not collide with them (reference semantics), so a hand reaching the
        # object is never blocked by the board. Authored as PhysX filtered pairs on the slab prims of each env
        # (the same mechanism the pack builders use for frame/panel pairs). No-op when no slab prim exists.
        self._inferred_support_filtered_pairs = 0
        if not bool(self.config.get("inferred_support_robot_collision", False)):
            self._inferred_support_filtered_pairs = self._exclude_robot_from_inferred_support()

        # safe_nan: detect and reset envs with NaN/Inf state/obs/reward after each step.
        # Critical for HOI sweeps where unstable grasp contacts can drive invalid values;
        # tnch/tnf (locomotion) rarely hit this path so the flag has minimal effect there.
        self._invalid_obs_abs_max = float(self.config.get("invalid_obs_abs_max", 1.0e6))
        self._invalid_state_abs_max = float(self.config.get("invalid_state_abs_max", 1.0e6))
        self.safe_nan = bool(self.config.get("safe_nan", False))

        # [D perf migration] Opt-in runtime synchronization cleanup ported from the
        # audited HiPHI wrapper (manager_env.config.runtime_sync_cleanup.*). The
        # absent/default configuration is a no-op and keeps the legacy step tail.
        self._runtime_sync_cleanup = RuntimeSyncCleanupOptions.from_config(
            self.config.get("runtime_sync_cleanup", None)
        )
        self._runtime_sync_cleanup_counters = RuntimeSyncCleanupCounters()
        if self._runtime_sync_cleanup.enabled:
            logger.info(
                "[RUNTIME SYNC CLEANUP] enabled "
                f"omit_action_cpu_copy="
                f"{self._runtime_sync_cleanup.omit_unused_env_actions_cpu_copy} "
                f"batch_source_metrics="
                f"{self._runtime_sync_cleanup.batch_joint_source_metrics} "
                f"sync_free_contact_cache_invalidation="
                f"{self._runtime_sync_cleanup.sync_free_contact_cache_invalidation} "
                f"benchmark_interval_steps="
                f"{self._runtime_sync_cleanup.benchmark_log_interval_steps}"
            )
        # Opt-in contact-sensor cadence parity probe (needs contact_sensor_parity_shadow).
        self._contact_sensor_parity_probe = ContactSensorParityProbe.from_config(
            self.config, logger=logger
        )
        if self._contact_sensor_parity_probe is not None:
            logger.info(
                "[CONTACT SENSOR PARITY] probe enabled "
                f"log_every_steps={self._contact_sensor_parity_probe.log_every_steps} "
                f"output_path={self._contact_sensor_parity_probe.output_path}"
            )

        # Initialize finger primitive support
        self._use_finger_primitive = self.config.get("use_finger_primitive", False)
        self._finger_primitive_map = None
        if self._use_finger_primitive:
            self._setup_finger_primitives()

        # Latent residual mode: policy outputs residual added to token latent space
        self._use_latent_residual = self.config.get("use_latent_residual", False)

        # Latent residual options (only used when use_latent_residual=True)
        self._latent_residual_mode = self.config.get("latent_residual_mode", "post_quantization")
        self._latent_residual_scale = self.config.get("latent_residual_scale", 1.0)

        # Optional PPO-owned residual applied after ATM decoding.  The default
        # dimension is zero, preserving the original latent + hand action path.
        self._tokenizer_action_dim = int(self.config.get("tokenizer_action_dim", 64))
        self._raw_body_residual_dim = int(self.config.get("raw_body_residual_dim", 0) or 0)
        # SONIC decoder LoRA arm: the policy appends sampled body-joint targets whose
        # mean is the (LoRA-adapted) decoder output; see trl/modules/sonic_decoder_lora.py.
        self._stochastic_body_action_dim = int(
            self.config.get("stochastic_decoder_body_action_dim", 0) or 0
        )
        self._stochastic_body_mean_max_err = 0.0
        hand_action_dim = self.config.get("hand_action_dim", None)
        self._hand_action_dim = int(hand_action_dim) if hand_action_dim is not None else None
        self._raw_body_residual_units = self.config.get(
            "raw_body_residual_units", "joint_position_radians"
        )
        raw_clip = self.config.get("raw_body_residual_clip", None)
        self._raw_body_residual_clip = float(raw_clip) if raw_clip is not None else None
        self._raw_body_action_scale = None
        self.env._tokenizer_action_dim = self._tokenizer_action_dim  # noqa: SLF001
        self.env._raw_body_residual_dim = self._raw_body_residual_dim  # noqa: SLF001
        if self._hand_action_dim is not None:
            self.env._hand_action_dim = self._hand_action_dim  # noqa: SLF001
        if self._raw_body_residual_dim not in (0, 29):
            raise ValueError(
                "raw_body_residual_dim must be 0 or 29 for the G1 ATM, got "
                f"{self._raw_body_residual_dim}"
            )
        if self._raw_body_residual_units not in ("joint_position_radians", "normalized"):
            raise ValueError(
                "raw_body_residual_units must be 'joint_position_radians' or 'normalized', got "
                f"{self._raw_body_residual_units!r}"
            )

        # Student direct latent mode: policy outputs FULL latent (not residual)
        # This is used for vision student policies that learned to output full latent
        # directly, bypassing the ATM encoder entirely at inference time.
        # The 64-dim output goes directly to ATM decoder (no encoding step).
        self._use_student_direct_latent = self.config.get("use_student_direct_latent", False)
        self._gpc_action_mode = self.config.get("gpc_action_mode", None)
        if self._gpc_action_mode not in (None, DIRECT_FSQ_ACTION_MODE):
            raise ValueError(
                f"unsupported gpc_action_mode={self._gpc_action_mode!r}; "
                f"expected {DIRECT_FSQ_ACTION_MODE!r}"
            )

        if self._use_latent_residual:
            logger.info(
                f"Latent residual enabled: mode={self._latent_residual_mode}, "
                f"scale={self._latent_residual_scale}"
            )

        if self._raw_body_residual_dim:
            logger.info(
                "Raw body residual enabled: "
                f"dim={self._raw_body_residual_dim}, "
                f"units={self._raw_body_residual_units}, "
                f"clip={self._raw_body_residual_clip}"
            )

        if self._use_student_direct_latent:
            logger.info(
                "Student direct latent mode enabled: policy output goes directly to ATM decoder "
                "(no encoding step)"
            )

        # Camera extrinsics randomization state
        self._camera_extrinsics_randomized = False
        self._camera_base_transforms = {}  # Store original transforms per env

        action_transform_module_cfg = self.config.get("action_transform_module_cfg", None)
        freeze_action_transform_module = bool(
            self.config.get("freeze_action_transform_module", False)
        )
        require_frozen_action_transform_module = bool(
            self.config.get("require_frozen_action_transform_module", False)
        )
        if require_frozen_action_transform_module and not freeze_action_transform_module:
            raise ValueError(
                "require_frozen_action_transform_module=true requires "
                "freeze_action_transform_module=true"
            )

        if action_transform_module_cfg is not None:
            # Load configs from exported YAML file. Rewrite legacy gr00t-internal
            # module paths to their gear_sonic equivalents so SONIC can
            # instantiate checkpoints trained before the rename (e.g. the
            # GRAIL-release sonic_sm_3pt_heading_hh-20260203_131215 bundle).
            with open(action_transform_module_cfg) as f:
                raw = f.read()
            raw = raw.replace("groot.rl.trl.", "gear_sonic.trl.")
            raw = raw.replace("groot.rl.envs.", "gear_sonic.envs.")
            raw = raw.replace("groot.rl.utils.", "gear_sonic.utils.")
            raw = raw.replace("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module.")
            raw = raw.replace("groot.rl.agents.", "gear_sonic.trl.")
            exported_config = OmegaConf.create(raw)

            env_config = exported_config.get("env_config", {})
            algo_config = exported_config.get("algo_config", {})

            self.action_transform_module = custom_instantiate(
                algo_config.actor, env_config=env_config, algo_config=algo_config, _resolve=False
            ).to(self.device)
            logger.info(f"Initialized action_transform_module from config: {action_transform_module_cfg}")

            # Load checkpoint if provided
            action_transform_module_checkpoint = self.config.get(
                "action_transform_module_checkpoint", None
            )
            if action_transform_module_checkpoint is not None:
                # Compatibility shim: checkpoints saved with TRL < 0.28.0 reference
                # trl.trainer.utils.OnlineTrainerState, which was moved in 0.28.0
                try:
                    from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
                    import trl.trainer.utils

                    trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
                    trl.trainer.utils.exact_div = exact_div
                except ImportError:
                    pass
                checkpoint = torch.load(
                    action_transform_module_checkpoint, map_location=self.device, weights_only=False
                )
                self.action_transform_module.load_state_dict(checkpoint["policy_state_dict"])
                logger.info(
                    f"Loaded action_transform_module checkpoint: {action_transform_module_checkpoint}"
                )

            if freeze_action_transform_module:
                frozen = freeze_module(
                    self.action_transform_module,
                    module_name="SONIC action transform module",
                )
                logger.info(
                    "Froze complete SONIC action transform: encoders, FSQ quantizer/codebook, "
                    f"and decoders ({frozen.parameter_count} parameters across "
                    f"{frozen.submodule_count} modules)"
                )

            # Precompute tokenizer observation indices for meta_action target
            self._tokenizer_obs_indices = self._compute_tokenizer_obs_indices()

            # Setup policy_atm observations for action_transform_module if DOF mismatch exists
            self._setup_policy_atm(env_config, algo_config)
        elif require_frozen_action_transform_module:
            raise ValueError(
                "require_frozen_action_transform_module=true requires "
                "action_transform_module_cfg"
            )

        try:
            self.viewer_focused = True
            self.env.viewport_camera_controller.update_view_to_world()
            self.env.viewport_camera_controller.update_view_to_asset_root("robot")
        except:  # noqa: E722
            self.viewer_focused = False

    def _setup_replay_joint_indices(self):
        """Setup joint indices for replay mode if robot has more DOFs than motion lib (29)."""
        if self.env.scene["robot"].num_joints > 29:
            self._setup_action_joint_indices()

    def _compute_tokenizer_obs_indices(self):
        """Compute start and end indices for each tokenizer observation in the flattened tensor."""
        if self.action_transform_module is None:
            return {}

        tokenizer_obs_names = self.action_transform_module.actor_module.tokenizer_obs_names
        tokenizer_obs_dims = self.action_transform_module.actor_module.tokenizer_obs_dims

        indices = {}
        current_index = 0
        for name in tokenizer_obs_names:
            all_dim = int(np.prod(tokenizer_obs_dims[name]))
            indices[name] = (current_index, current_index + all_dim)
            current_index += all_dim

        return indices

    def _setup_policy_atm(self, env_config, algo_config):  # noqa: ARG002
        """Setup policy_atm for action_transform_module when robot has more DOFs than ATM expects."""
        atm_num_joints = env_config.get("robot", {}).get("actions_dim") or env_config.get(
            "robot", {}
        ).get("num_joints", 29)
        self._needs_policy_atm = self.config.get("needs_policy_atm", True)
        self._atm_num_joints = atm_num_joints
        self._current_num_joints = self.env.scene["robot"].num_joints
        if self._current_num_joints > atm_num_joints:
            assert (
                self._needs_policy_atm
            ), "Robot has more DOFs than ATM expects, but needs_policy_atm is False"

        has_policy_atm = (
            hasattr(self.env, "observation_manager")
            and "policy_atm" in self.env.observation_manager._group_obs_term_names  # noqa: SLF001
        )
        self._use_policy_atm_group = self._needs_policy_atm and has_policy_atm

        if self._use_policy_atm_group:
            self._setup_action_joint_indices()

    def _setup_action_joint_indices(self):
        """Compute joint indices for mapping body (29 DOF) and hand (14 DOF) actions."""
        if self._body_joint_indices is not None:
            return

        robot = self.env.scene["robot"]
        self._body_joint_indices = get_body_joint_indices(robot)
        self._hand_joint_indices = get_hand_joint_indices(robot)
        # Name-map the reference hand dof (dof[29:43], HAND_ORDER_USD == per-hand grouped) to the
        # robot's actual joints. The IsaacLab hand articulation order is INTERLEAVED L/R by depth,
        # so positional [:, -14:] scrambles fingers (index_1<->thumb_0 etc.); map by name instead.
        _hand_order_usd = [
            "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
            "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
            "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
        ]
        _rn = list(robot.joint_names)
        self._reference_hand_dof_indices = (
            torch.tensor([_rn.index(n) for n in _hand_order_usd], dtype=torch.long, device=self.device)
            if all(n in _rn for n in _hand_order_usd) else None
        )

    def _get_body_action_scale(self, body_actions: torch.Tensor) -> torch.Tensor:
        """Return IsaacLab action scales in the ATM's 29-joint body order."""
        if self._raw_body_action_scale is not None:
            return self._raw_body_action_scale.to(
                device=body_actions.device, dtype=body_actions.dtype
            )

        self._setup_action_joint_indices()
        robot = self.env.scene["robot"]
        body_joint_names = [robot.joint_names[int(index)] for index in self._body_joint_indices]
        action_term = self.env.action_manager.get_term("joint_pos")
        action_joint_names = list(action_term._joint_names)  # noqa: SLF001
        action_indices = []
        for joint_name in body_joint_names:
            if joint_name not in action_joint_names:
                raise ValueError(
                    f"Body joint {joint_name!r} is missing from the joint_pos action term"
                )
            action_indices.append(action_joint_names.index(joint_name))

        scale = torch.as_tensor(
            action_term._scale,  # noqa: SLF001
            device=body_actions.device,
            dtype=body_actions.dtype,
        )
        if scale.ndim == 0:
            scale = scale.repeat(self._raw_body_residual_dim)
        else:
            scale = scale[..., action_indices]
        if scale.shape[-1] != self._raw_body_residual_dim:
            raise ValueError(
                f"Expected {self._raw_body_residual_dim} body action scales, got {scale.shape}"
            )
        if torch.any(torch.abs(scale) < 1.0e-8):
            raise ValueError("Raw body residual cannot use a zero joint_pos action scale")
        self._raw_body_action_scale = scale
        return scale

    def _apply_raw_body_residual(
        self, body_actions: torch.Tensor, raw_body_residual: torch.Tensor | None
    ) -> torch.Tensor:
        """Add a PPO-owned 29-DOF residual to decoded ATM body actions."""
        if self._raw_body_residual_dim == 0:
            return body_actions
        if raw_body_residual is None:
            raise ValueError("Raw body residual is enabled but missing from the policy action")
        if body_actions.shape[-1] != self._raw_body_residual_dim:
            raise ValueError(
                f"ATM returned {body_actions.shape[-1]} body actions; "
                f"expected {self._raw_body_residual_dim}"
            )
        if raw_body_residual.shape[-1] != self._raw_body_residual_dim:
            raise ValueError(
                f"Policy returned {raw_body_residual.shape[-1]} raw body residuals; "
                f"expected {self._raw_body_residual_dim}"
            )

        residual = raw_body_residual
        if self._raw_body_residual_clip is not None:
            residual = torch.clamp(
                residual,
                min=-self._raw_body_residual_clip,
                max=self._raw_body_residual_clip,
            )
        self.env._last_raw_body_residual = residual  # noqa: SLF001
        if self._raw_body_residual_units == "joint_position_radians":
            residual = residual / self._get_body_action_scale(body_actions)
        return body_actions + residual

    def _setup_finger_primitives(self):
        """Setup finger primitive action mapping from config.

        Finger primitives allow the policy to output 2 actions (left/right gripper)
        instead of 14 individual finger joint actions. Each primitive action
        interpolates between open (pos_0) and closed (pos_1) positions.
        """
        primitive_cfg = self.config.get("finger_primitive", {})
        primitive_action_map = primitive_cfg.get("primitive_action_map", {})

        if not primitive_action_map:
            logger.info("Warning: use_finger_primitive=True but no primitive_action_map defined")
            self._use_finger_primitive = False
            return

        robot = self.env.scene["robot"]
        joint_names = robot.joint_names

        self._finger_primitive_map = []
        for action_name in sorted(primitive_action_map.keys()):
            prim_cfg = primitive_action_map[action_name]
            dof_names = list(prim_cfg.get("dof_names", []))
            pos_0 = torch.tensor(prim_cfg.get("pos_0", []), device=self.device, dtype=torch.float32)
            pos_1 = torch.tensor(prim_cfg.get("pos_1", []), device=self.device, dtype=torch.float32)

            # Find indices in the hand joints
            dof_idx = []
            for dof_name in dof_names:
                if dof_name in joint_names:
                    dof_idx.append(joint_names.index(dof_name))
                else:
                    logger.info(f"Warning: DOF {dof_name} not found in robot joints")
            # Support both "mode" (new) and "discrete" (legacy) config keys
            mode = prim_cfg.get("mode", None)
            if mode is None:
                # Legacy support: discrete=True → mode="discrete", discrete=False → mode="linear"
                mode = "discrete" if prim_cfg.get("discrete", False) else "linear"

            self._finger_primitive_map.append(
                {
                    "action_name": action_name,
                    "dof_names": dof_names,
                    "dof_idx": dof_idx,
                    "pos_0": pos_0,
                    "pos_1": pos_1,
                    "mode": mode,
                }
            )
            logger.info(f"Finger primitive '{action_name}': {len(dof_names)} DOFs, mode={mode}")

        self._num_finger_primitives = len(self._finger_primitive_map)
        # Publish the resolved map (absolute joint indices, open pos_0 / closed pos_1 per hand) on the inner
        # env so reward terms can score ACTUAL finger joint openness from robot.data.joint_pos instead of the
        # raw primitive action sign (see mdp.rewards._hand_openness).
        self.env._finger_primitive_map = self._finger_primitive_map  # noqa: SLF001
        logger.info(f"Initialized {self._num_finger_primitives} finger primitives")

    def _convert_primitive_to_finger_actions(self, primitive_actions: torch.Tensor) -> torch.Tensor:
        """Convert primitive actions (num_envs, num_primitives) to finger joint targets (num_envs, 14).

        Follows groot_backup implementation:
        - "linear" mode: maps [-1, 1] → [0, 1] via (x + 1) / 2, then lerps between pos_0 and pos_1
        - "discrete" mode: action >= 0 → pos_1 (closed), action < 0 → pos_0 (open)

        Args:
            primitive_actions: Tensor of shape (num_envs, num_primitives), values in [-1, 1]

        Returns:
            Tensor of shape (num_envs, num_finger_joints) with joint position targets
        """
        num_envs = primitive_actions.shape[0]
        num_finger_joints = (
            len(self._hand_joint_indices) if self._hand_joint_indices is not None else 14
        )
        finger_targets = torch.zeros(
            num_envs, num_finger_joints, device=self.device, dtype=primitive_actions.dtype
        )

        for i, prim_cfg in enumerate(self._finger_primitive_map):
            action = primitive_actions[:, i]  # (num_envs,), values in [-1, 1]
            p0 = prim_cfg["pos_0"]
            p1 = prim_cfg["pos_1"]
            dof_idx = prim_cfg["dof_idx"]
            mode = prim_cfg.get("mode", "linear")
            if mode == "linear":
                # Clamp to [-1, 1] then map to [0, 1]
                action_clamped = action.clamp(min=-1.0, max=1.0)
                t = (action_clamped + 1.0) / 2.0  # (num_envs,)
                joint_targets = torch.lerp(p0.unsqueeze(0), p1.unsqueeze(0), t.unsqueeze(1))
            elif mode == "discrete":
                # action >= 0 → closed (pos_1), action < 0 → open (pos_0)
                joint_targets = torch.where(
                    action.unsqueeze(1) >= 0, p1.unsqueeze(0), p0.unsqueeze(0)
                )
            else:
                raise ValueError(f"Invalid finger primitive mode: {mode}")

            # Map to the correct indices in finger_targets
            # dof_idx are absolute joint indices, need to convert to relative hand indices
            for j, abs_idx in enumerate(dof_idx):
                if self._hand_joint_indices is not None:
                    # Convert tensor to list if needed for .index() lookup
                    hand_indices_list = (
                        self._hand_joint_indices.tolist()
                        if isinstance(self._hand_joint_indices, torch.Tensor)
                        else self._hand_joint_indices
                    )
                    rel_idx = hand_indices_list.index(abs_idx)
                    finger_targets[:, rel_idx] = joint_targets[:, j]
                else:
                    # Fallback: assume hand joints are at the end
                    finger_targets[:, abs_idx - 29] = joint_targets[:, j]

        return finger_targets

    def _prepare_obs_for_action_transform_module(self, obs_dict):
        """Use policy_atm observations for ATM if DOF mismatch exists, else use policy."""
        if not self._use_policy_atm_group or "policy_atm" not in obs_dict:
            atm_obs_dict = obs_dict.copy()
        else:
            atm_obs_dict = obs_dict.copy()
            atm_obs_dict["actor_obs"] = atm_obs_dict["policy_atm"]

        # Ensure all observations have a sequence dimension [num_envs, seq_len, dim]
        # The action_transform_module expects 3D tensors
        for k, v in atm_obs_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2:
                atm_obs_dict[k] = v.unsqueeze(1)  # Add seq_len=1 dimension

        return atm_obs_dict

    def reset_all(self, global_rank=0):  # noqa: ARG002
        return self.reset()

    @staticmethod
    def _tensor_bad_env_ids(tensor, abs_limit=None):
        return tensor_bad_env_ids(tensor, abs_limit=abs_limit)

    @staticmethod
    def _collect_bad_env_ids_from_obs(obs, abs_limit=None):
        return collect_bad_env_ids_from_obs(obs, abs_limit=abs_limit)

    def _collect_bad_env_ids_from_state(self, abs_limit=None):
        bad_env_ids = set()

        if "robot" in self.env.scene.keys():
            robot = self.env.scene["robot"]
            for attr_name in [
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_w",
                "root_ang_vel_w",
                "joint_pos",
                "joint_vel",
            ]:
                tensor = getattr(robot.data, attr_name, None)
                if tensor is None:
                    continue
                bad_env_ids.update(self._tensor_bad_env_ids(tensor, abs_limit=abs_limit))

        if "object" in self.env.scene.keys():
            obj = self.env.scene["object"]
            for attr_name in [
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_w",
                "root_ang_vel_w",
            ]:
                tensor = getattr(obj.data, attr_name, None)
                if tensor is None:
                    continue
                bad_env_ids.update(self._tensor_bad_env_ids(tensor, abs_limit=abs_limit))

        return sorted(bad_env_ids)

    @staticmethod
    def _merge_reset_obs(dst_obs, src_obs, env_ids):
        merge_reset_obs(dst_obs, src_obs, env_ids)

    def _sanitize_obs_for_env_ids(self, obs, env_ids):
        sanitize_obs_for_env_ids(obs, env_ids, abs_max=self._invalid_obs_abs_max)

    def _zero_wrapper_buffers(self, env_ids):
        if hasattr(self.env, "_prev_meta_action"):
            self.env._prev_meta_action[env_ids] = 0.0
            self.env._last_meta_action[env_ids] = 0.0
            self.env._prev_full_latent[env_ids] = 0.0
            self.env._full_latent[env_ids] = 0.0
        finger_actions = getattr(self.env, "_finger_primitive_actions_raw", None)
        if isinstance(finger_actions, torch.Tensor):
            finger_actions[env_ids] = 0.0
        elif finger_actions is not None:
            # The production wrapper stores a tensor here. Clear an unexpected
            # representation rather than carrying stale pre-reset hand state.
            self.env._finger_primitive_actions_raw = None  # noqa: SLF001
        if hasattr(self.env, "success_lift"):
            self.env.success_lift[env_ids] = False

    def _reset_invalid_envs(self, env_ids):
        reset_result = None
        partial_reset_used = False
        reset_error = None

        try:
            # IsaacLab's first positional reset argument is ``seed``. Always
            # use the explicit environment-id keyword for the pinned runtime.
            reset_result = self.env.reset(env_ids=env_ids)
            partial_reset_used = reset_result is not None
        except TypeError as exc:
            reset_error = exc
            # Only an explicit signature rejection authorizes the legacy
            # positional fallback. A TypeError raised *inside* reset is a real
            # partial-reset failure and must take the fail-closed path below.
            signature_rejected_env_ids = (
                "unexpected keyword argument 'env_ids'" in str(exc)
            )
            if signature_rejected_env_ids:
                try:
                    reset_result = self.env.reset(env_ids)
                    partial_reset_used = reset_result is not None
                except Exception as fallback_exc:  # noqa: BLE001
                    reset_error = fallback_exc
        except Exception as exc:  # noqa: BLE001
            reset_error = exc

        if reset_result is None:
            partial_reset_used = False
            logger.info(
                "[OBS RECOVERY] partial reset failed, falling back to full reset "
                f"error={type(reset_error).__name__ if reset_error is not None else 'unknown'}"
            )
            reset_result = self.env.reset()

        if isinstance(reset_result, tuple):
            reset_obs = reset_result[0]
        else:
            reset_obs = reset_result

        return reset_obs, partial_reset_used

    def _recover_invalid_envs_after_step(
        self, obs_dict, rew, terminated, truncated, extras, env_actions
    ):
        """Detect envs with NaN/Inf state/obs/reward after step() and reset them.

        Gated by ``self.safe_nan``; no-op when the flag is off. Ported from
        gr00t-internal's ManagerEnvWrapper — without this, HOI sweeps (bp2/bp3)
        accumulate corrupted gradients from unstable grasp contacts and fail
        to learn despite having ``safe_nan=true`` in the sweep config.
        """
        if not self.safe_nan:
            return obs_dict, rew, terminated, truncated, extras

        invalid_env_ids = set(
            self._collect_bad_env_ids_from_state(abs_limit=self._invalid_state_abs_max)
        )
        invalid_env_ids.update(
            self._tensor_bad_env_ids(
                env_actions, abs_limit=self._invalid_state_abs_max
            )
        )
        strict_reference_free = bool(
            self.config.get("gpc_strict_reference_free_recovery", False)
        )
        if not strict_reference_free:
            invalid_env_ids.update(
                self._collect_bad_env_ids_from_obs(
                    obs_dict, abs_limit=self._invalid_obs_abs_max
                )
            )
            invalid_env_ids.update(
                self._tensor_bad_env_ids(
                    rew, abs_limit=self._invalid_obs_abs_max
                )
            )

        if not invalid_env_ids:
            return obs_dict, rew, terminated, truncated, extras

        invalid_env_ids = sorted(invalid_env_ids)
        env_ids = torch.tensor(invalid_env_ids, device=self.device, dtype=torch.long)

        logger.info(
            "[OBS RECOVERY] resetting invalid envs "
            f"count={len(invalid_env_ids)} "
            f"env_ids={invalid_env_ids[:8]}"
        )
        action_stats = tensor_batch_invalid_stats(env_actions[env_ids])
        if action_stats is None:
            action_stats = {
                "shape": tuple(env_actions[env_ids].shape),
                "nan": 0,
                "posinf": 0,
                "neginf": 0,
                "bad_env_ids": [],
                "min": float(env_actions[env_ids].min().item()),
                "max": float(env_actions[env_ids].max().item()),
                "mean": float(env_actions[env_ids].mean().item()),
            }
        log_invalid_stats("[OBS RECOVERY] env_actions_for_invalid_envs", action_stats)

        reset_obs, partial_reset_used = self._reset_invalid_envs(env_ids)
        if self.config.get("gpc_strict_partial_reset", False) and not partial_reset_used:
            raise RuntimeError(
                "GPC safe-NaN recovery fell back to a full reset; unaffected "
                "environment trajectories can no longer be trusted"
            )
        self._merge_reset_obs(obs_dict, reset_obs, env_ids)
        self._sanitize_obs_for_env_ids(obs_dict, env_ids)
        self._zero_wrapper_buffers(env_ids)

        rew = rew.clone()
        rew[env_ids] = 0.0
        terminated = terminated.clone()
        truncated = truncated.clone()
        terminated[env_ids] = True
        truncated[env_ids] = False

        if "log" not in extras:
            extras["log"] = {}
        extras["log"]["invalid_env_resets"] = float(len(invalid_env_ids))
        extras["log"]["invalid_env_resets_partial"] = float(partial_reset_used)
        invalid_mask = extras.get("invalid_reset_mask")
        if invalid_mask is None:
            invalid_mask = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.device
            )
            extras["invalid_reset_mask"] = invalid_mask
        invalid_mask[env_ids] = True

        return obs_dict, rew, terminated, truncated, extras

    def process_raw_obs(self, obs, flatten_dict_obs):
        new_obs = {
            "actor_obs": obs["policy"],
            "critic_obs": obs["critic"],
        }
        for k, v in obs.items():
            if k not in ["policy", "critic"]:
                if isinstance(v, dict) and flatten_dict_obs:
                    if k == "height_map":
                        # Special case: do not flatten height map
                        new_obs[k] = v["height_map"]
                        continue
                    if k == "camera_rgb":
                        # Special case: do not flatten camera RGB image
                        # Keep original shape [B, H, W, C] for vision encoder
                        new_obs[k] = v["camera_rgb"]
                        continue
                    obs_names = self.env.observation_manager._group_obs_term_names[k]  # noqa: SLF001
                    new_obs[k] = torch.cat(
                        [v[obs_name].reshape(v[obs_name].shape[0], -1) for obs_name in obs_names],
                        dim=-1,
                    )
                else:
                    new_obs[k] = v
        contact_phase = self._compute_rp2_contact_phase_obs()
        if contact_phase is not None:
            new_obs["contact_phase"] = contact_phase.unsqueeze(-1)
        return new_obs

    def _clamp_motion_steps(self, motion_steps: torch.Tensor) -> torch.Tensor:
        motion_steps = torch.clamp(motion_steps, min=0)
        if self.motion_command is None:
            return motion_steps
        motion_num_steps = getattr(self.motion_command, "motion_num_steps", None)
        if motion_num_steps is None:
            return motion_steps
        if torch.is_tensor(motion_num_steps):
            max_steps = torch.clamp(motion_num_steps.to(motion_steps.device) - 1, min=0)
        else:
            max_steps = torch.full_like(motion_steps, max(int(motion_num_steps) - 1, 0))
        return torch.minimum(motion_steps, max_steps)

    def _reference_contact_label_at(
        self, motion_steps: torch.Tensor, hand: str = "either"
    ) -> torch.Tensor | None:
        if self.motion_command is None or self._motion_lib is None:
            return None
        motion_steps = self._clamp_motion_steps(motion_steps)
        if hand == "either":
            labels = []
            for side in ("left_hand", "right_hand"):
                label = self._motion_lib.get_object_in_contact(
                    self.motion_command.motion_ids, motion_steps, hand=side
                )
                if label is not None:
                    labels.append(label.float())
            if not labels:
                return None
            out = labels[0]
            for label in labels[1:]:
                out = torch.maximum(out, label)
            return out
        label = self._motion_lib.get_object_in_contact(
            self.motion_command.motion_ids, motion_steps, hand=hand
        )
        return None if label is None else label.float()

    def _compute_rp2_contact_phase_obs(self) -> torch.Tensor | None:
        cfg = self.config.get("rp2_contact_phase_obs", {}) or {}
        if not bool(cfg.get("enabled", False)):
            return None
        if self.motion_command is None or self._motion_lib is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        hand = cfg.get("hand", "either")
        pre_frames = max(int(cfg.get("pre_contact_frames", 15)), 0)
        post_frames = max(int(cfg.get("post_contact_frames", 15)), 0)
        mode = str(cfg.get("mode", "gaussian"))
        sigma = max(float(cfg.get("sigma_frames", 5.0)), 1e-6)

        current_steps = self.motion_command.motion_start_time_steps + self.motion_command.time_steps
        phase = torch.zeros(self.num_envs, dtype=torch.float32, device=current_steps.device)
        for offset in range(-post_frames, pre_frames + 1):
            label = self._reference_contact_label_at(current_steps + offset, hand=hand)
            if label is None:
                continue
            dist = abs(offset)
            if mode == "binary":
                weight = 1.0
            elif mode == "ramp":
                window = post_frames if offset < 0 else pre_frames
                weight = 1.0 if dist == 0 or window <= 0 else max(0.0, 1.0 - dist / float(window))
            else:
                weight = float(np.exp(-0.5 * (dist / sigma) ** 2))
            phase = torch.maximum(phase, label.float() * weight)
        return torch.clamp(phase, min=0.0, max=1.0)

    def reset(self, flatten_dict_obs=True):
        obs, info = self.env.reset()
        new_obs = self.process_raw_obs(obs, flatten_dict_obs)
        # Initialize success_lift to False for all envs after reset (used unconditionally in step())
        self.env.success_lift = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        if self.action_transform_module is not None:
            # Store obs for action_transform_module when obs_dict is not provided in step()
            self._last_obs_dict = new_obs
            # Initialize last meta action buffer (policy output: latent + primitives)
            # meta_action_dim = tokenizer_action_dim + hand_action_dim (e.g., 64 + 2 = 66)
            meta_action_dim = self.config.get("meta_action_dim", 66)
            self.env._last_meta_action = torch.zeros(  # noqa: SLF001
                self.env.num_envs, meta_action_dim, dtype=torch.float32, device=self.env.device
            )
            # Previous meta action buffer for meta_action_rate_l2 reward (token smoothness)
            self.env._prev_meta_action = torch.zeros(  # noqa: SLF001
                self.env.num_envs, meta_action_dim, dtype=torch.float32, device=self.env.device
            )
            # Full latent buffers for full_latent_rate_l2 reward (decoder input smoothness)
            # tokenizer_action_dim = latent_dim (e.g., 64 = num_tokens * token_dim)
            tokenizer_action_dim = self.config.get("tokenizer_action_dim", 64)
            self.env._full_latent = torch.zeros(  # noqa: SLF001
                self.env.num_envs, tokenizer_action_dim, dtype=torch.float32, device=self.env.device
            )
            self.env._prev_full_latent = torch.zeros(  # noqa: SLF001
                self.env.num_envs, tokenizer_action_dim, dtype=torch.float32, device=self.env.device
            )
            if self._raw_body_residual_dim:
                self.env._last_raw_body_residual = torch.zeros(  # noqa: SLF001
                    self.env.num_envs,
                    self._raw_body_residual_dim,
                    dtype=torch.float32,
                    device=self.env.device,
                )

            # Apply camera extrinsics randomization on every reset
            self.apply_random_camera_extrinsics()

        return new_obs

    def apply_random_camera_extrinsics(self):
        """Apply per-environment random camera extrinsics (position and rotation offsets).

        Reads randomization ranges from config:
        - cameras.camera_extrinsics_randomization: Enable/disable switch (default: False)
        - cameras.camera_pos_rand_range: ±meters for x, y, z position
        - cameras.camera_roll_rand_range: ±radians for roll
        - cameras.camera_pitch_rand_range: ±radians for pitch
        - cameras.camera_yaw_rand_range: ±radians for yaw
        """
        cameras_config = self.config.get("cameras", {})

        # Check if randomization is enabled via switch
        if not cameras_config.get("camera_extrinsics_randomization", False):
            return

        # Get randomization ranges
        pos_range = cameras_config.get("camera_pos_rand_range", 0.0)
        roll_range = cameras_config.get("camera_roll_rand_range", 0.0)
        pitch_range = cameras_config.get("camera_pitch_rand_range", 0.0)
        yaw_range = cameras_config.get("camera_yaw_rand_range", 0.0)

        # Check if any randomization values are non-zero
        if pos_range == 0 and roll_range == 0 and pitch_range == 0 and yaw_range == 0:
            return

        # Get camera attached link from config
        camera_attached_link = cameras_config.get("camera_attached_link", None)
        if camera_attached_link is None:
            logger.info("Skipping camera extrinsics randomization: no camera_attached_link configured")
            return

        # Only print details on first call
        if not self._camera_extrinsics_randomized:
            logger.info("Applying random camera extrinsics per environment:")
            logger.info(f"  Position range: ±{pos_range*100:.1f}cm")
            logger.info(f"  Roll range: ±{roll_range:.3f} rad ({np.degrees(roll_range):.1f}°)")
            logger.info(f"  Pitch range: ±{pitch_range:.3f} rad ({np.degrees(pitch_range):.1f}°)")
            logger.info(f"  Yaw range: ±{yaw_range:.3f} rad ({np.degrees(yaw_range):.1f}°)")

        stage = omni.usd.get_context().get_stage()

        for env_id in range(self.env.scene.num_envs):
            camera_prim_path = f"/World/envs/env_{env_id}/Robot/{camera_attached_link}/ego_camera"
            camera_prim = stage.GetPrimAtPath(camera_prim_path)

            if not camera_prim.IsValid():
                if env_id == 0:
                    logger.info(f"  Warning: Camera prim not found at {camera_prim_path}")
                continue

            # Get current camera transform
            xformable = UsdGeom.Xformable(camera_prim)

            # On first call, store the original/base transforms
            if env_id not in self._camera_base_transforms:
                base_translate = None
                base_orient = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        base_translate = Gf.Vec3d(op.Get())  # Make a copy
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        base_orient = Gf.Quatd(op.Get())  # Make a copy
                self._camera_base_transforms[env_id] = {
                    "translate": base_translate,
                    "orient": base_orient,
                }

            # Get base transforms
            base_translate = self._camera_base_transforms[env_id]["translate"]
            base_orient = self._camera_base_transforms[env_id]["orient"]

            # Sample random deltas
            if pos_range > 0:  # noqa: SIM108
                pos_delta = np.random.uniform(-pos_range, pos_range, 3)  # noqa: NPY002
            else:
                pos_delta = np.zeros(3)

            # Sample rotation deltas (roll, pitch, yaw)
            roll_delta = np.random.uniform(-roll_range, roll_range) if roll_range > 0 else 0.0  # noqa: NPY002
            pitch_delta = np.random.uniform(-pitch_range, pitch_range) if pitch_range > 0 else 0.0  # noqa: NPY002
            yaw_delta = np.random.uniform(-yaw_range, yaw_range) if yaw_range > 0 else 0.0  # noqa: NPY002

            # Apply position delta relative to BASE (not current)
            if base_translate is not None:
                new_pos = Gf.Vec3d(
                    base_translate[0] + pos_delta[0],
                    base_translate[1] + pos_delta[1],
                    base_translate[2] + pos_delta[2],
                )
                # Find and update the translate op
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(new_pos)
                        break

            # Apply rotation delta relative to BASE (not current)
            if base_orient is not None and (roll_delta != 0 or pitch_delta != 0 or yaw_delta != 0):
                # Convert euler deltas to quaternion
                # Order: roll (X), pitch (Y), yaw (Z)
                cr, sr = np.cos(roll_delta / 2), np.sin(roll_delta / 2)
                cp, sp = np.cos(pitch_delta / 2), np.sin(pitch_delta / 2)
                cy, sy = np.cos(yaw_delta / 2), np.sin(yaw_delta / 2)

                # Quaternion from euler (ZYX convention)
                delta_quat = Gf.Quatd(
                    cr * cp * cy + sr * sp * sy,  # w
                    sr * cp * cy - cr * sp * sy,  # x
                    cr * sp * cy + sr * cp * sy,  # y
                    cr * cp * sy - sr * sp * cy,  # z
                )

                # Compose: new = delta * BASE (not current!)
                new_orient = delta_quat * base_orient

                # Find and update the orient op
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        op.Set(new_orient)
                        break

            # Store the random deltas for each environment (for projection in observations)
            if not hasattr(self, "_camera_random_deltas"):
                self._camera_random_deltas = {}
            self._camera_random_deltas[env_id] = {
                "pos_delta": pos_delta.tolist() if isinstance(pos_delta, np.ndarray) else [0, 0, 0],
                "roll_delta": float(roll_delta),
                "pitch_delta": float(pitch_delta),
                "yaw_delta": float(yaw_delta),
            }

            if env_id == 0 and not self._camera_extrinsics_randomized:
                logger.info(
                    f"  Env 0: pos_delta={pos_delta}, rot_delta=[{roll_delta:.3f}, {pitch_delta:.3f}, {yaw_delta:.3f}]"  # noqa: E501
                )

        # Mark as initialized (for print suppression)
        self._camera_extrinsics_randomized = True

    def _decode_direct_latent(self, full_latent, atm_obs_dict):
        """Decode full latent directly using ATM decoder (skip encoder).
        Used for student rollout where policy outputs full latent.

        Args:
            full_latent: (batch, latent_dim) full latent from student policy
            atm_obs_dict: observation dict for ATM

        Returns:
            body_actions: (batch, seq_len, action_dim) decoded actions
        """  # noqa: D205
        # Get proprioception for decoder
        if "policy_atm" in atm_obs_dict:  # noqa: SIM108
            proprioception = atm_obs_dict["policy_atm"]
        else:
            proprioception = atm_obs_dict["actor_obs"]

        # Ensure sequence dimension
        if proprioception.dim() == 2:
            proprioception = proprioception.unsqueeze(1)

        atm = self.action_transform_module.actor_module
        return self._decode_direct_latent_batch(full_latent, proprioception, atm)

    def _decode_direct_fsq_codes(self, full_codes, atm_obs_dict):
        """Decode exact post-FSQ codes without invoking an ATM encoder or FSQ.

        GPC predicts the complete quantized code captured from the teacher, not
        a continuous pre-quantization latent.  Reapplying FSQ would silently
        remap many otherwise-valid grid values because FSQ includes a tanh
        bounding transform.
        """
        proprioception = require_policy_atm_proprioception(atm_obs_dict)
        atm = self.action_transform_module.actor_module
        return decode_post_fsq_codes(atm, full_codes, proprioception)

    def _decode_direct_latent_batch(self, full_latent, proprioception, atm):
        """Decode a batch of full latents directly using ATM decoder.

        The student policy outputs pre-quantization values (latent + residual).
        We apply quantization here before decoding to match the teacher's flow:
        - Teacher: latent + residual -> quantize -> decode
        - Student: direct_latent -> quantize -> decode

        Args:
            full_latent: (batch, latent_dim) full latent (pre-quantization)
            proprioception: (batch, seq, proprio_dim) proprioception input
            atm: ATM actor module

        Returns:
            body_actions: (batch, seq_len, action_dim) decoded actions
        """
        batch_size = full_latent.shape[0]

        # Reshape latent for quantization: (batch, latent_dim) -> (batch, num_tokens, token_dim)
        latent_reshaped = full_latent.view(batch_size, atm.max_num_tokens, atm.token_dim)

        # Apply quantization (same as teacher's flow)
        # This ensures student inference matches teacher: latent -> quantize -> decode
        if atm.quantizer is not None:
            quantized_codes, _ = atm.quantizer(latent_reshaped)
            tokens_for_decode = quantized_codes.contiguous()
        else:
            tokens_for_decode = latent_reshaped

        # Reshape for decoder: (batch, num_tokens, token_dim) -> (batch, 1, num_tokens, token_dim)
        tokens_reshaped = tokens_for_decode.unsqueeze(1)
        tokens_flattened = tokens_for_decode.view(batch_size, -1).unsqueeze(1)

        # Prepare decode input
        decode_input_dict = {
            "token": tokens_reshaped,
            "token_flattened": tokens_flattened,  # (batch, 1, latent_dim)
            "proprioception": proprioception,
        }

        # Decode directly (skip encoding entirely)
        decoded_output = atm.decode("g1_dyn", decode_input_dict)

        # Get body actions from decoder output
        # Note: "meta_action" is for special hierarchical policies, "action" is standard
        body_actions = decoded_output.get("meta_action", decoded_output.get("action"))
        if body_actions is None:
            raise KeyError(
                f"Decoder output missing 'action' or 'meta_action'. Keys: {decoded_output.keys()}"
            )

        return body_actions

    def _ensure_meta_action_history(self, meta_actions):
        """Initialize ATM history buffers for reset-bypassing replay paths."""

        for name in ("_last_meta_action", "_prev_meta_action"):
            value = getattr(self.env, name, None)
            if (
                value is None
                or value.shape != meta_actions.shape
                or value.device != meta_actions.device
                or value.dtype != meta_actions.dtype
            ):
                setattr(self.env, name, torch.zeros_like(meta_actions))
        latent_template = meta_actions.new_zeros(
            meta_actions.shape[0], self._tokenizer_action_dim
        )
        for name in ("_full_latent", "_prev_full_latent"):
            value = getattr(self.env, name, None)
            if (
                value is None
                or value.shape != latent_template.shape
                or value.device != latent_template.device
                or value.dtype != latent_template.dtype
            ):
                setattr(self.env, name, latent_template.clone())
        if self._raw_body_residual_dim:
            residual_template = meta_actions.new_zeros(
                meta_actions.shape[0], self._raw_body_residual_dim
            )
            value = getattr(self.env, "_last_raw_body_residual", None)
            if (
                value is None
                or value.shape != residual_template.shape
                or value.device != residual_template.device
                or value.dtype != residual_template.dtype
            ):
                self.env._last_raw_body_residual = residual_template  # noqa: SLF001

    def step(self, actions):
        if self.action_transform_module is not None:
            # Use provided obs_dict or fall back to stored obs from last reset/step
            if "obs_dict" in actions:
                obs_dict = actions["obs_dict"].copy()
            else:
                # Fallback for callbacks that don't provide obs_dict (e.g., im_eval)
                obs_dict = getattr(self, "_last_obs_dict", None)
                if obs_dict is None:
                    raise ValueError(
                        "action_transform_module requires obs_dict but none was provided or stored"
                    )
                obs_dict = obs_dict.copy() if isinstance(obs_dict, dict) else obs_dict
            meta_actions = actions["actions"]
            body_dim = self._stochastic_body_action_dim
            sampled_body_actions = None
            if body_dim:
                sampled_body_actions = meta_actions[:, -body_dim:]
                meta_actions = meta_actions[:, :-body_dim]
            # Determine action mode: exact FSQ codes, continuous direct latent,
            # teacher residual, or a teacher/student mixed rollout.
            # Priority: 1) explicit action_mode in actions dict, 2) config flag
            # During training: trainer sets action_mode explicitly
            # During eval: fallback to config flags
            action_mode = resolve_gpc_action_mode(
                actions.get("action_mode", None), self._gpc_action_mode
            )
            if action_mode is None:
                # Fallback for eval scripts that don't set action_mode
                if self._use_student_direct_latent:
                    action_mode = "direct_latent"
                elif self._use_latent_residual:
                    action_mode = "residual"
                else:
                    action_mode = "residual"  # Default to residual if nothing specified

            # Some replay/evidence paths bypass wrapper.reset(). Initialize from
            # the real policy shape before shifting action-history buffers.
            self._ensure_meta_action_history(meta_actions)
            # Shift meta action buffers for meta_action_rate_l2 reward (token smoothness)
            self.env._prev_meta_action = self.env._last_meta_action.clone()  # noqa: SLF001
            # Store meta action for observation (last policy output)
            self.env._last_meta_action = meta_actions.clone()  # noqa: SLF001

            atm_obs_dict = self._prepare_obs_for_action_transform_module(obs_dict)

            # Optional action layout: latent, raw body residual, hand primitives.
            tokenizer_end = self._tokenizer_action_dim
            raw_body_end = tokenizer_end + self._raw_body_residual_dim
            hand_action_dim = self._hand_action_dim
            if hand_action_dim is None:
                hand_action_dim = meta_actions.shape[-1] - raw_body_end
            expected_action_dim = raw_body_end + hand_action_dim
            if meta_actions.shape[-1] != expected_action_dim:
                raise ValueError(
                    f"Policy action has {meta_actions.shape[-1]} dims, expected "
                    f"{expected_action_dim} = {self._tokenizer_action_dim} latent + "
                    f"{self._raw_body_residual_dim} raw body + "
                    f"{hand_action_dim} hand"
                )
            tokenizer_meta_actions = meta_actions[:, :tokenizer_end]
            raw_body_residual = (
                meta_actions[:, tokenizer_end:raw_body_end]
                if self._raw_body_residual_dim
                else None
            )
            hand_actions_raw = meta_actions[:, raw_body_end:expected_action_dim]

            # Override hand actions with motion data if configured
            if self.config.get("use_motion_hand_actions", False):
                motion_cmd = self.env.command_manager.get_term("motion")
                left_action = motion_cmd.get_hand_action("left_hand")
                right_action = motion_cmd.get_hand_action("right_hand")

                if left_action is None or right_action is None:
                    raise ValueError(
                        "use_motion_hand_actions=True but hand_action_left/right not found in motion data. "
                        "Ensure processed_robot_motions.pkl contains 'hand_action_left' and 'hand_action_right' arrays."  # noqa: E501
                    )

                # Use motion data directly: -1.0 = open, +1.0 = closed
                # Threshold at 0 in _convert_primitive_to_finger_actions
                hand_actions_raw = torch.stack([left_action, right_action], dim=-1)

            # Convert primitive actions to finger joint targets if enabled
            if self._use_finger_primitive and self._finger_primitive_map:
                # Store raw primitive actions on env for reward computation (before clamping)
                self.env._finger_primitive_actions_raw = hand_actions_raw  # noqa: SLF001
                hand_actions = self._convert_primitive_to_finger_actions(hand_actions_raw)
            else:
                self.env._finger_primitive_actions_raw = None  # noqa: SLF001
                hand_actions = hand_actions_raw

            if action_mode == DIRECT_FSQ_ACTION_MODE:
                # GPC mode: policy emits the complete post-FSQ code.  Validate
                # the exact grid contract and bypass both encoder and quantizer.
                body_actions = self._decode_direct_fsq_codes(
                    tokenizer_meta_actions, atm_obs_dict
                )

            elif action_mode == "direct_latent":
                # Student direct latent mode: policy outputs FULL latent, not residual
                # Skip ATM encoder entirely - go directly to decoder
                body_actions = self._decode_direct_latent(tokenizer_meta_actions, atm_obs_dict)

            elif action_mode == "residual":
                # Teacher/residual mode: policy outputs residual that's added to ATM encoded tokens
                # Apply scaling to residual before passing to ATM
                scaled_residual = tokenizer_meta_actions * self._latent_residual_scale
                # Add residual in latent/token space (after encoding, before decoding)
                body_actions = self.action_transform_module(
                    atm_obs_dict,
                    latent_residual=scaled_residual,
                    latent_residual_mode=self._latent_residual_mode,
                )

            elif action_mode == "mixed":
                # Mixed rollout: some envs use teacher (residual), some use student (direct_latent)
                # is_teacher_env is a boolean mask: True = teacher/residual, False = student/direct_latent
                is_teacher_env = actions.get("is_teacher_env")
                if is_teacher_env is None:
                    raise ValueError(
                        "action_mode='mixed' requires 'is_teacher_env' mask in actions dict"
                    )

                num_envs = tokenizer_meta_actions.shape[0]
                atm = self.action_transform_module.actor_module

                # Get proprioception for decoder (needed for student/direct_latent mode)
                if "policy_atm" in atm_obs_dict:
                    proprioception = atm_obs_dict["policy_atm"]
                else:
                    proprioception = atm_obs_dict["actor_obs"]
                if proprioception.dim() == 2:
                    proprioception = proprioception.unsqueeze(1)

                teacher_mask = is_teacher_env
                student_mask = ~is_teacher_env
                num_teacher = teacher_mask.sum().item()
                num_student = student_mask.sum().item()

                # Initialize placeholders
                teacher_indices = None
                student_indices = None
                teacher_body_actions = None
                student_body_actions = None

                # Process teacher envs (residual mode) if any
                if num_teacher > 0:
                    teacher_indices = teacher_mask.nonzero(as_tuple=True)[0]
                    teacher_latent = tokenizer_meta_actions[teacher_indices]

                    # Prepare teacher obs dict (subset of envs)
                    # Only include keys that ATM actually needs: tokenizer and actor_obs
                    teacher_atm_obs = {}
                    atm_keys = ["tokenizer", "actor_obs"]
                    for obs_key in atm_keys:
                        if obs_key in atm_obs_dict:
                            teacher_atm_obs[obs_key] = atm_obs_dict[obs_key][teacher_indices]

                    scaled_residual = teacher_latent * self._latent_residual_scale
                    teacher_body_actions = self.action_transform_module(
                        teacher_atm_obs,
                        latent_residual=scaled_residual,
                        latent_residual_mode=self._latent_residual_mode,
                    )

                # Process student envs (direct_latent mode) if any
                if num_student > 0:
                    student_indices = student_mask.nonzero(as_tuple=True)[0]
                    student_latent = tokenizer_meta_actions[student_indices]
                    student_proprio = proprioception[student_indices]

                    student_body_actions = self._decode_direct_latent_batch(
                        student_latent, student_proprio, atm
                    )

                # Merge results - determine output shape from whichever mode ran
                if teacher_body_actions is not None:
                    out_seq_len = teacher_body_actions.shape[1]
                    out_action_dim = teacher_body_actions.shape[2]
                    dtype = teacher_body_actions.dtype
                elif student_body_actions is not None:
                    out_seq_len = student_body_actions.shape[1]
                    out_action_dim = student_body_actions.shape[2]
                    dtype = student_body_actions.dtype
                else:
                    raise RuntimeError("Mixed mode: no envs to process (both masks empty)")

                body_actions = torch.zeros(
                    num_envs, out_seq_len, out_action_dim, device=self.device, dtype=dtype
                )

                if teacher_body_actions is not None and teacher_indices is not None:
                    body_actions[teacher_indices] = teacher_body_actions
                if student_body_actions is not None and student_indices is not None:
                    body_actions[student_indices] = student_body_actions

            else:
                raise ValueError(
                    f"Unknown action_mode: {action_mode}. "
                    "Valid modes are 'direct_fsq_codes', 'direct_latent', "
                    "'residual', or 'mixed'."
                )

            # Store full latent (decoder input) for full_latent_rate_l2 reward
            # Only needed for residual mode (teacher RL training); student modes
            # (direct_latent, mixed) use L2 distillation loss, not RL rewards.
            if action_mode == "residual":
                self.env._prev_full_latent = self.env._full_latent.clone()  # noqa: SLF001
                atm_module = self.action_transform_module.actor_module
                if (
                    hasattr(atm_module, "_last_full_latent_flat")
                    and atm_module._last_full_latent_flat is not None  # noqa: SLF001
                ):
                    fl = atm_module._last_full_latent_flat  # noqa: SLF001
                    if fl.dim() == 3:
                        fl = fl[:, -1, :]  # (batch, latent_dim)
                    self.env._full_latent = fl.to(self.env.device)  # noqa: SLF001

            body_actions = body_actions[:, -1]  # Take last timestep
            body_actions = self._apply_raw_body_residual(body_actions, raw_body_residual)
            if sampled_body_actions is not None:
                if action_mode != "residual":
                    raise ValueError("stochastic decoder body actions require residual mode")
                policy_mean = actions.get("action_mean", None)
                if policy_mean is not None and policy_mean.shape[-1] == actions["actions"].shape[-1]:
                    # The policy's body mean must equal the decoder output computed here.
                    err = float((body_actions - policy_mean[:, -body_dim:]).abs().max())
                    self._stochastic_body_mean_max_err = max(self._stochastic_body_mean_max_err, err)
                body_actions = sampled_body_actions

            if (
                self._body_joint_indices is not None
                and self._hand_joint_indices is not None
                and len(self._body_joint_indices) > 0
                and len(self._hand_joint_indices) > 0
            ):
                num_envs = body_actions.shape[0]
                env_actions = torch.zeros(
                    num_envs, self._current_num_joints, device=self.device, dtype=body_actions.dtype
                )
                env_actions[:, self._body_joint_indices] = body_actions
                env_actions[:, self._hand_joint_indices] = hand_actions
            else:
                env_actions = torch.cat([body_actions, hand_actions], dim=-1)

        else:
            env_actions = actions["actions"]

        action_clip_value = self.config.get("action_clip_value", None)

        if action_clip_value is not None and action_clip_value > 0:
            env_actions = torch.clip(env_actions, -action_clip_value, action_clip_value)

        # Lightweight action plot update (env 0, first N joints)
        if self.turn_on_visualization:
            try:
                if self._action_fig is None:
                    self._init_action_plot(env_actions.shape[-1])
                if (self._step_counter % self._viz_every_n_steps) == 0:
                    self._update_action_plot(env_actions)
                self._step_counter += 1
            except Exception:  # noqa: S110, BLE001
                pass

        obs_dict, rew, terminated, truncated, extras = self.env.step(env_actions)
        if self._contact_sensor_parity_probe is not None:
            # Read-only parity accounting; no physics step occurs between the
            # reward-phase sensor reads inside env.step and this point. The
            # grouped canonical family force is compared through the same
            # aggregation the required-contact reward consumes.
            from gear_sonic.envs.manager_env.mdp import rp2_source_relation as _rp2sr

            self._contact_sensor_parity_probe.observe(
                getattr(self.env.scene, "sensors", {}),
                int(getattr(self.env, "common_step_counter", 0)),
                env=self.env,
                family_force_fn=lambda env, family: _rp2sr._canonical_sensor_force(  # noqa: SLF001
                    env, family, allow_missing=True
                ),
            )
        extras["invalid_reset_mask"] = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.device
        )

        # Recover envs with NaN/Inf state/obs/reward (gated by safe_nan flag).
        # Critical for HOI sweeps where unstable grasp contacts can diverge.
        if self.safe_nan:
            obs_dict, rew, terminated, truncated, extras = self._recover_invalid_envs_after_step(
                obs_dict, rew, terminated, truncated, extras, env_actions
            )

        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)

        # Zero out action/latent rate buffers for envs that just reset
        # This prevents a false large rate penalty on the first step of a new episode
        # Only applies when action_transform_module is used (buffers created in reset())
        reset_mask = dones.bool()
        if reset_mask.any() and hasattr(self.env, "_prev_meta_action"):
            self.env._prev_meta_action[reset_mask] = 0.0  # noqa: SLF001
            self.env._last_meta_action[reset_mask] = 0.0  # noqa: SLF001
            self.env._prev_full_latent[reset_mask] = 0.0  # noqa: SLF001
            self.env._full_latent[reset_mask] = 0.0  # noqa: SLF001
            if hasattr(self.env, "_last_raw_body_residual"):
                self.env._last_raw_body_residual[reset_mask] = 0.0  # noqa: SLF001

        strict_reference_free = bool(
            self.config.get("gpc_strict_reference_free_recovery", False)
        )
        # Compute success_lift metric: check if object has no contact with table (lifted)
        # Skip frames before first contact (from contact label data)
        if strict_reference_free:
            # This legacy metric gates on reference first-contact timing. GPC
            # computes physical lift/success externally from live state and
            # must not expose the reference-derived latch.
            self.env.success_lift = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )
        elif (
            hasattr(self.env, "scene")
            and "object_to_table_contact_sensor" in self.env.scene.sensors
        ):
            from isaaclab.sensors import ContactSensor

            sensor: ContactSensor = self.env.scene["object_to_table_contact_sensor"]
            contact_force = sensor.data.force_matrix_w  # [num_envs, 1, 1, 3]
            force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))  # [num_envs]

            contact_force_threshold = self.config.get("lift_contact_force_threshold", 0.1)

            if hasattr(self.env, "success_lift"):
                self.env.success_lift = self.env.success_lift & (~dones.bool())

            # Get first contact frame from motion command (contact label data)
            is_before_contact = torch.ones(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )
            if self.motion_command is not None:
                per_env_first_contact = getattr(self.motion_command, "_per_env_first_contact", None)
                if per_env_first_contact is not None:
                    current_time = (
                        self.motion_command.motion_start_time_steps + self.motion_command.time_steps
                    )
                    is_before_contact = current_time < per_env_first_contact

            # Object is currently lifted if there's no contact force from table
            is_currently_lifted = force_magnitude <= contact_force_threshold

            # Only update success_lift after first contact (cumulative OR)
            # Before first contact, success_lift stays False regardless of contact
            self.env.success_lift = torch.where(
                is_before_contact,
                self.env.success_lift,  # Keep current value (False after reset)
                is_currently_lifted | self.env.success_lift,
            )
        else:
            # If no contact sensor, set to False for all envs
            self.env.success_lift = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )

        if strict_reference_free:
            # Environment tracking reward is reference-derived and unused by
            # GPC evaluation/RLFT. Return a neutral tensor so it cannot become
            # an accidental controller signal or recovery trigger.
            rew = torch.zeros_like(rew)
        else:
            rew = self._compute_rp2_reward_channels(rew)

        extras["time_outs"] = truncated
        extras["episode"] = {}
        extras["to_log"] = {}
        for k, v in extras["log"].items():
            if isinstance(v, torch.Tensor):
                extras["to_log"][k] = v
            else:
                extras["to_log"][k] = torch.tensor(v, dtype=torch.float)
        if (
            self.motion_command is not None
            and bool(self.config.get("log_rp2_object_role_metrics", True))
        ):
            for k, v in self.motion_command.metrics.items():
                if not k.startswith("rp2_object_"):
                    continue
                if isinstance(v, torch.Tensor):
                    extras["to_log"][f"Metrics/motion/{k}"] = v
                else:
                    extras["to_log"][f"Metrics/motion/{k}"] = torch.tensor(
                        v, dtype=torch.float, device=self.env.device
                    )
        # (per-step adaptive-sampling summary reductions over every bin - min/max/mean episode, failure and
        #  probability statistics, coverage thresholds, effective bins, the eps_mean host/GPU comparison - were
        #  log-only work on the hot path and are removed: user decision 2026-10-14, no runtime work PPO does not need)
        self._append_joint_source_metrics(extras)
        new_obs = self.process_raw_obs(obs_dict, flatten_dict_obs=True)
        # Store obs for action_transform_module when obs_dict is not provided in next step()
        self._last_obs_dict = new_obs
        self.extras = extras
        self._finalize_runtime_sync_cleanup(env_actions, extras)
        return new_obs, rew, dones, extras

    def _append_joint_source_metrics(self, extras):
        """Append identical joint-source metrics through the legacy or on-device path."""
        if self._motion_lib is None or not getattr(
            self._motion_lib, "use_joint_source_tracking", False
        ):
            return

        if self._runtime_sync_cleanup.batch_joint_source_metrics:
            # One newly allocated (num_sources, 3) snapshot replaces four scalar
            # .item() host syncs per source. Values are cast to float32 exactly
            # like the legacy torch.tensor(float(x), dtype=float32) construction.
            source_names, source_values = (
                self._motion_lib.get_joint_source_statistics_tensor()
            )
            if source_values is None or source_values.shape != (len(source_names), 3):
                raise RuntimeError(
                    "Batched joint-source metrics must have shape "
                    f"({len(source_names)}, 3), got "
                    f"{None if source_values is None else tuple(source_values.shape)}"
                )
            source_values = source_values.to(dtype=torch.float32)
            for source_index, source_name in enumerate(source_names):
                prefix = f"joint_source/{source_name}"
                extras["to_log"][f"{prefix}/episodes"] = source_values[source_index, 0]
                extras["to_log"][f"{prefix}/failures"] = source_values[source_index, 1]
                extras["to_log"][f"{prefix}/success_rate"] = source_values[source_index, 2]
            self._runtime_sync_cleanup_counters.record_batched_source_metrics(
                len(source_names)
            )
            return

        source_stats = self._motion_lib.get_joint_source_statistics()
        for source_name, values in source_stats.items():
            prefix = f"joint_source/{source_name}"
            extras["to_log"][f"{prefix}/episodes"] = torch.tensor(
                values["episode_count"],
                dtype=torch.float32,
                device=self.env.device,
            )
            extras["to_log"][f"{prefix}/failures"] = torch.tensor(
                values["failure_count"],
                dtype=torch.float32,
                device=self.env.device,
            )
            extras["to_log"][f"{prefix}/success_rate"] = torch.tensor(
                values["success_rate"],
                dtype=torch.float32,
                device=self.env.device,
            )

    def _finalize_runtime_sync_cleanup(self, env_actions, extras):
        """Preserve the legacy extras unless the audited cleanup is explicitly on."""
        if not self._runtime_sync_cleanup.enabled:
            # Store env_actions for callbacks (e.g., MultiLatentSaveCallback)
            extras["env_actions"] = env_actions.detach().cpu()
            return

        counters = self._runtime_sync_cleanup_counters
        counters.record_step()
        if self._runtime_sync_cleanup.omit_unused_env_actions_cpu_copy:
            # No in-repository consumer reads this payload (consumer map
            # runs/d_perf_migration_20261011/consumer_map/wrapper_relation_map.md).
            counters.record_omitted_action_copy(env_actions)
        else:
            extras["env_actions"] = env_actions.detach().cpu()

        interval = self._runtime_sync_cleanup.benchmark_log_interval_steps
        if interval > 0 and counters.steps % interval == 0:
            values = counters.as_dict()
            logger.info(
                "[RUNTIME SYNC CLEANUP] counters "
                + " ".join(f"{key}={value}" for key, value in values.items())
            )

    def get_runtime_sync_cleanup_statistics(self):
        """Return synchronization-free cumulative cleanup instrumentation."""
        return self._runtime_sync_cleanup_counters.as_dict()

    def _compute_rp2_reward_channels(self, rew: torch.Tensor) -> torch.Tensor:
        cfg = self.config.get("rp2_reward_channels", {}) or {}
        if not bool(cfg.get("enabled", False)):
            return rew
        reward_manager = getattr(self.env, "reward_manager", None)
        term_names = list(getattr(reward_manager, "_term_names", []) or [])
        step_reward = getattr(reward_manager, "_step_reward", None)
        if step_reward is None or not term_names:
            return torch.stack([rew, torch.zeros_like(rew)], dim=-1)

        dt = float(getattr(self.env, "step_dt", 1.0))
        tracking_terms = set(
            cfg.get(
                "tracking_terms",
                [
                    "tracking_anchor_pos",
                    "tracking_anchor_ori",
                    "tracking_relative_body_pos",
                    "tracking_relative_body_ori_weighted",
                    "tracking_body_linvel",
                    "tracking_body_angvel",
                    "undesired_contacts_no_ankle_hand",
                    "hand_table_contact_penalty",
                    "finger_primitive_limit",
                ],
            )
        )
        interaction_terms = set(
            cfg.get(
                "interaction_terms",
                [
                    "object_tracking_reward",
                    "grasp_contact_center_right",
                    "grasp_contact_center_left",
                    "grasp_finger_direction",
                    "grasp_finger_direction_left",
                    "rp2_soma_object_relation_reward",
                    "rp2_support_stability_penalty",
                    "rp2_contact_phase_meta_action_rate_l2",
                ],
            )
        )
        unmatched_channel = str(cfg.get("unmatched_channel", "tracking"))

        tracking = torch.zeros_like(rew)
        interaction = torch.zeros_like(rew)
        for term_idx, term_name in enumerate(term_names):
            value = step_reward[:, term_idx] * dt
            if term_name in interaction_terms:
                interaction = interaction + value
            elif term_name in tracking_terms or unmatched_channel == "tracking":
                tracking = tracking + value
            elif unmatched_channel == "interaction":
                interaction = interaction + value

        return torch.stack([tracking, interaction], dim=-1)

    def get_env_data(self, key):
        if key == "ref_body_pos_extend":
            return self.motion_command.robot_body_pos_w
        elif key == "rigid_body_pos_extend":
            return self.motion_command.body_pos_w
        else:
            return self.env.get_env_data(key)

    def render_results(self):
        pass

    def end_render_results(self):
        if "render_envs" in self.env.recorder_manager._terms:  # noqa: SLF001
            self.env.recorder_manager._terms["render_envs"].close_writers()  # noqa: SLF001
        if "trajectory" in self.env.recorder_manager._terms:  # noqa: SLF001
            self.env.recorder_manager._terms["trajectory"].close_writers()  # noqa: SLF001
        if self._action_fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._action_fig)
            except Exception:  # noqa: S110, BLE001
                pass
            self._action_fig = None
            self._action_lines = None
            self._action_hist = None
            self._hist_idx = 0
            self._blit_background = None

    def set_is_evaluating(self, is_evaluating: bool = True, global_rank=0, **_kwargs):
        self.is_evaluating = is_evaluating
        if self.motion_command is not None:
            self.motion_command.set_is_evaluating(is_evaluating)
        if self.force_command is not None:
            self.force_command.set_is_evaluating(is_evaluating)
        if is_evaluating:
            self.begin_seq_motion_samples(global_rank)

    def begin_seq_motion_samples(self, global_rank=0):
        logger.info("Loading motions for evaluation")
        if bool(
            getattr(
                self.motion_command,
                "use_asset_compatible_motions",
                False,
            )
        ):
            # The joint roster has already loaded every rank-local compatible
            # motion. Sequential reloading would break the immutable USD/motion
            # compatibility contract, so periodic evaluation uses each slot's
            # deterministic representative motion.
            self.start_idx = 0
            self.reset_all(global_rank=global_rank)
            return
        self.start_idx = global_rank * self.num_envs
        self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
        self.reset_all(global_rank=global_rank)

    def forward_motion_samples(self, global_rank=0, world_size=1):
        if bool(
            getattr(
                self.motion_command,
                "use_asset_compatible_motions",
                False,
            )
        ):
            raise RuntimeError(
                "Joint-roster periodic evaluation is one fixed representative "
                "pass; use the roster-aware exhaustive evaluator for all motions."
            )
        old_start_idx = self.start_idx
        self.start_idx += world_size * self.num_envs
        logger.info(
            f"Forward motions for evaluation from {old_start_idx} to {self.start_idx} - rank: {global_rank} - world size: {world_size}"  # noqa: E501
        )
        self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
        self.reset_all(global_rank=global_rank)

    def focusing_viewer(self):
        if not self.viewer_focused:
            # Focus on robot asset (tracking mode)
            self.env.viewport_camera_controller.update_view_to_world()
            self.env.viewport_camera_controller.update_view_to_asset_root("robot")
            self.viewer_focused = True
            logger.info("Focused on robot")
        else:
            # Switch to free camera mode centered on robot
            self.env.viewport_camera_controller.viewer_origin = torch.zeros_like(
                self.env.scene["robot"].data.root_pos_w[0]
            )
            self.env.viewport_camera_controller.cfg.origin_type = "world"
            cam_eye = self.env.scene["robot"].data.root_pos_w[0].cpu().numpy() + np.array(
                [2.0, 2.0, 1.5]
            )
            cam_target = self.env.scene["robot"].data.root_pos_w[0].cpu().numpy()
            self.env.viewport_camera_controller._env.sim.set_camera_view(  # noqa: SLF001
                eye=cam_eye, target=cam_target
            )
            self.viewer_focused = False
            logger.info("Switched to free camera mode")  # noqa: RUF100, T201

    def set_is_training(self, **_kwargs):
        self.is_evaluating = False
        self.resample_motion()

    def _exclude_robot_from_inferred_support(self) -> int:
        """Apply UsdPhysics.FilteredPairsAPI(slab -> this env's robot) to every inferred-support slab prim
        (custom attribute ``hoi:inferredSupport``). Returns the number of slab prims filtered; never raises."""
        try:
            import omni.usd
            from pxr import Sdf, Usd, UsdPhysics
            from gear_sonic.rp2.inferred_support import INFERRED_ATTR

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return 0
            envs_root = stage.GetPrimAtPath("/World/envs")
            if not envs_root or not envs_root.IsValid():
                return 0
            robot_name = str(self.config.get("inferred_support_robot_prim_name", "Robot"))
            count = 0
            for env_prim in envs_root.GetChildren():
                context = env_prim.GetChild("StaticContext")
                if not context or not context.IsValid():
                    continue
                robot_path = env_prim.GetPath().AppendChild(robot_name)
                if not stage.GetPrimAtPath(robot_path).IsValid():
                    continue
                for prim in Usd.PrimRange(context):
                    attr = prim.GetAttribute(INFERRED_ATTR)
                    if not attr.IsValid() or not attr.Get() or prim.GetTypeName() != "Cube":
                        continue
                    UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(Sdf.Path(robot_path))
                    count += 1
            if count:
                logger.info(f"[inferred_support] robot excluded from {count} inferred support slab prims (object still collides)")
            return count
        except Exception as exc:  # noqa: BLE001 - diagnostics only; training never depends on this
            logger.info(f"[inferred_support] robot/slab collision filter not applied: {type(exc).__name__}: {exc}")
            return 0

    def resample_motion(self, *, reset=True):
        res = self._motion_lib.load_motions_for_training(
            max_num_seqs=min(self.num_envs, self.motion_command.max_num_load_motions)
        )
        if not reset:
            # the caller performs the full reset_all of every env right afterwards (PPO trainer resume path): a second
            # back-to-back full reset would be redundant work (2026-10-15 restore-path attribution, Generalist 36613735)
            logger.info("Motions restored; the caller resets all envs next (no reset here)")  # noqa: RUF100, T201
            return
        if res:
            self.reset_all()
        elif bool(
            getattr(
                self.motion_command,
                "use_asset_compatible_motions",
                False,
            )
        ):
            # All compatible motions stay resident. A reset is still required
            # after evaluation or sampler-state restore to draw training starts.
            self.reset_all()
        else:
            logger.info("No new motions loaded, skipping reset")  # noqa: RUF100, T201

    def sync_and_compute_adaptive_sampling(self, accelerator, sync_across_gpus=False):
        if self._motion_lib is not None:
            self._motion_lib.sync_and_compute_adaptive_sampling(
                accelerator, sync_across_gpus=sync_across_gpus
            )

    def load_env_state_dict(self, state_dict, *, reset_after_restore=True):
        if "motion_lib" in state_dict:
            self._motion_lib.load_state_dict(state_dict["motion_lib"])
            if self._motion_lib.use_adaptive_sampling:
                self.resample_motion(reset=reset_after_restore)

    def get_env_state_dict(self):
        state_dict = {
            "motion_lib": self._motion_lib.get_state_dict(),
        }
        return state_dict

    def reinit_dr(self, **_kwargs):
        pass

    def setup_keyboard(self):
        try:
            from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg

            cfg = Se2KeyboardCfg()
            self.keyboard_interface = Se2Keyboard(cfg)
            self.keyboard_interface.add_callback("R", self.reset_all)
            self.keyboard_interface.add_callback("T", self.forward_motion_samples)
            self.keyboard_interface.add_callback("F", self.focusing_viewer)
            self.keyboard_interface.add_callback("V", self.toggle_debug_vis)
        except Exception as e:  # noqa: BLE001
            logger.info(f"Error setting up keyboard: {e}")  # noqa: RUF100, T201

    def toggle_debug_vis(self):
        if self.motion_command is not None and hasattr(self.motion_command, "_set_debug_vis_impl"):
            self._debug_vis_enabled = not getattr(self, "_debug_vis_enabled", True)
            self.motion_command._set_debug_vis_impl(self._debug_vis_enabled)  # noqa: SLF001
            logger.info(f"Debug visualization: {'ON' if self._debug_vis_enabled else 'OFF'}")  # noqa: RUF100, T201

    # --- Action plotting helpers ---
    def _init_action_plot(self, action_dim: int):
        plot_dim = min(int(self._plot_action_dim), int(action_dim))
        if plot_dim <= 0:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001, F841
            return
        plt.ion()
        import math

        cols = math.ceil(math.sqrt(plot_dim))
        rows = math.ceil(plot_dim / cols)
        fig, axes = plt.subplots(
            rows, cols, sharex=True, sharey=True, figsize=(cols * 3.0, rows * 2.2)
        )
        axes = np.array(axes).reshape(-1)
        self._plot_window = max(10, int(self._plot_window))
        x_vals = np.arange(self._plot_window)
        self._action_hist = np.zeros((self._plot_window, plot_dim), dtype=np.float32)
        lines = []
        y_min, y_max = -float(self._action_ylim), float(self._action_ylim)
        for i in range(rows * cols):
            ax_i = axes[i]
            if i < plot_dim:
                (ln,) = ax_i.plot(x_vals, self._action_hist[:, i], linewidth=1.0)
                ln.set_animated(True)
                lines.append(ln)
                ax_i.set_ylim((y_min, y_max))
                ax_i.set_xlim((0, self._plot_window - 1))
                ax_i.set_title(f"J{i}", fontsize=8)
                if i // cols == rows - 1:
                    ax_i.set_xlabel("step", fontsize=8)
                if i % cols == 0:
                    ax_i.set_ylabel("act", fontsize=8)
            else:
                ax_i.axis("off")
        fig.suptitle(f"Action time series (env 0) - first {plot_dim} joints", fontsize=10)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        self._action_fig = fig
        self._action_lines = lines
        self._hist_idx = -1
        try:
            fig.canvas.draw()
            self._blit_background = fig.canvas.copy_from_bbox(fig.bbox)
        except Exception:  # noqa: BLE001
            self._blit_background = None
            self._blit_supported = False
        fig.canvas.flush_events()

    def _update_action_plot(self, env_actions: torch.Tensor):
        if self._action_lines is None or self._action_hist is None:
            return
        plot_dim = len(self._action_lines)
        with torch.no_grad():
            vals = env_actions[0, :plot_dim].detach().to("cpu").numpy().astype(np.float32)
        vals = np.clip(vals, -self._action_ylim, self._action_ylim)
        # advance circular buffer
        self._hist_idx = (self._hist_idx + 1) % self._plot_window
        self._action_hist[self._hist_idx, :plot_dim] = vals
        # display in chronological order using a view with roll
        disp = np.roll(self._action_hist, shift=-(self._hist_idx + 1), axis=0)
        for i, ln in enumerate(self._action_lines):
            ln.set_ydata(disp[:, i])
        if self._action_fig is not None:
            canvas = self._action_fig.canvas
            # Periodically refresh background to handle resizes or overdraw
            if self._blit_supported and (
                self._blit_background is None
                or (self._step_counter % max(1, self._blit_refresh_interval) == 0)
            ):
                try:
                    self._action_fig.canvas.draw()
                    self._blit_background = canvas.copy_from_bbox(self._action_fig.bbox)
                except Exception:  # noqa: BLE001
                    self._blit_background = None
                    self._blit_supported = False
            if self._blit_supported and self._blit_background is not None:
                try:
                    canvas.restore_region(self._blit_background)
                    for ln in self._action_lines:
                        ln.axes.draw_artist(ln)
                    canvas.blit(self._action_fig.bbox)
                    canvas.flush_events()
                    return
                except Exception:  # noqa: BLE001
                    self._blit_supported = False
            # Fallback full redraw
            canvas.draw_idle()
            canvas.flush_events()

    @property
    def motion_ids(self):
        return self.motion_command.motion_ids

    def _restore_replay_static_context(self):
        """Restore the context's world pose before applying a replay grid."""
        if hasattr(self, "_replay_static_context_view"):
            self._replay_static_context_view.set_transforms(
                self._replay_static_context_reference_pose,
                self._replay_static_context_indices,
            )

    def setup_replay_grid(self, spacing=2.0, rows=None, cols=None):
        """Setup a custom grid layout for environment origins during replay.

        Args:
            spacing: Distance between environments in meters (default: 2.0)
            rows: Number of rows in the grid (default: auto-calculated)
            cols: Number of columns in the grid (default: auto-calculated)

        Returns:
            Tensor of shape (num_envs, 3) with custom origins
        """
        import math

        num_envs = self.num_envs

        # Auto-calculate grid dimensions if not provided
        if rows is None and cols is None:
            # Try to make a square-ish grid
            cols = int(math.ceil(math.sqrt(num_envs)))  # noqa: RUF046
            rows = int(math.ceil(num_envs / cols))  # noqa: RUF046
        elif rows is None:
            rows = int(math.ceil(num_envs / cols))  # noqa: RUF046
        elif cols is None:
            cols = int(math.ceil(num_envs / rows))  # noqa: RUF046

        logger.info(f"Setting up replay grid: {rows} rows x {cols} cols with {spacing}m spacing")

        # Create grid origins
        custom_origins = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)

        for i in range(num_envs):
            row = i // cols
            col = i % cols

            # Center the grid around origin
            x_offset = (col - (cols - 1) / 2.0) * spacing
            y_offset = (row - (rows - 1) / 2.0) * spacing

            custom_origins[i, 0] = x_offset
            custom_origins[i, 1] = y_offset
            custom_origins[i, 2] = 0.0  # Keep z at ground level

        # Store custom origins
        self._replay_custom_origins = custom_origins

        # AssetBase static contexts are not updated by the robot/object replay
        # writers. Move their existing kinematic roots by the same origin delta;
        # the table/support poses authored below each root remain unchanged.
        if "static_context" in self.env.scene.extras:
            if not hasattr(self, "_replay_static_context_view"):
                context_view = (
                    self.env.sim.physics_sim_view.create_rigid_body_view(
                        f"{self.env.scene.env_regex_ns}/StaticContext".replace(".*", "*")
                    )
                )
                self._replay_static_context_reference_pose = (
                    context_view.get_transforms().clone()
                )
                self._replay_static_context_indices = torch.arange(
                    num_envs, device=self.device, dtype=torch.int32
                )
                self._replay_static_context_view = context_view
            # Always start from the original world pose, including scene origin,
            # so repeated grid setup never adds the replay translation twice.
            context_pose = self._replay_static_context_reference_pose.clone()
            context_pose[:, :3] += custom_origins - self.env.scene.env_origins
            self._replay_static_context_view.set_transforms(
                context_pose, self._replay_static_context_indices
            )

        logger.info(
            f"Grid bounds: X=[{custom_origins[:, 0].min():.1f}, {custom_origins[:, 0].max():.1f}], "
            f"Y=[{custom_origins[:, 1].min():.1f}, {custom_origins[:, 1].max():.1f}]"
        )

        return custom_origins

    @_restore_static_context_after_replay
    def run_replay(
        self,
        motion_id=None,
        start_time_step=0,
        speed=1.0,
        loop=True,
        enable_vis=False,
        target_fps=50,
        grid_spacing=2.0,
        grid_rows=None,
        grid_cols=None,
        save_video_path=None,
        hold_completed=False,
        per_env_video_dir=None,
        env_titles=None,
    ):
        """Run a complete motion replay with automatic rendering loop for all environments.
        This is the high-level convenience function that handles everything.

        Args:
            motion_id: The motion ID(s) to replay. Can be:
                       - None: uses current motion_ids for all envs
                       - int: single motion ID for all envs
                       - list/tensor: motion ID per environment
            start_time_step: Starting time step in the motion (default: 0)
            speed: Playback speed multiplier (default: 1.0)
            loop: Whether to loop the motion (default: True)
            enable_vis: Whether to enable debug visualization (default: False)
            target_fps: Target frames per second for rendering (default: 50)
            grid_spacing: Distance between environments in meters (default: 2.0)
            grid_rows: Number of rows in grid layout (default: auto)
            grid_cols: Number of columns in grid layout (default: auto)
            save_video_path: Path to save video file (default: None, no video saved)
                             Requires render_results=True in config to enable eval_camera.
                             The file holds the env-0 eval camera only.
            hold_completed: When not looping, keep each finished environment at its last
                            motion frame and end the replay only when EVERY environment has
                            finished (default False: the replay ends at the first completion).
            per_env_video_dir: Directory for one mp4 per replayed environment (its own eval
                               camera); each clip stops at that environment's last motion frame.
            env_titles: Optional {motion key or env index: short title} burned into the
                        per-env clips and used in their file names.

        Returns:
            List of dictionaries with motion metadata for each environment
        """  # noqa: D205
        import os
        import time

        # Setup custom grid origins
        self.setup_replay_grid(spacing=grid_spacing, rows=grid_rows, cols=grid_cols)

        # Initialize replay
        info = self.setup_replay_motion(
            motion_id=motion_id,
            start_time_step=start_time_step,
            speed=speed,
            loop=loop,
            enable_vis=enable_vis,
        )
        self._replay_hold_completed = bool(hold_completed) and not loop

        if info is None:
            return None

        # Initialize video writer if save_video_path is specified
        video_writer = None
        if save_video_path is not None:
            if "eval_camera" not in self.env.scene.sensors:
                logger.info("WARNING: save_video_path specified but eval_camera not available.")
                logger.info("         Set ++manager_env.config.render_results=True to enable it.")
            else:
                import imageio

                os.makedirs(os.path.dirname(os.path.abspath(save_video_path)), exist_ok=True)
                video_writer = imageio.get_writer(
                    save_video_path,
                    fps=target_fps,
                    codec="libx264",
                    quality=5,
                    pixelformat="yuv420p",
                )
                logger.info(f"[Video] Recording to: {save_video_path}")
        per_env_writers = None
        per_env_frames = None
        per_env_done = None
        per_env_files = None
        per_env_draw = None
        titles = {}
        if per_env_video_dir is not None and "eval_camera" in self.env.scene.sensors:
            import re as _re

            import imageio

            os.makedirs(per_env_video_dir, exist_ok=True)
            # Titles are keyed by the motion key/alias the environment actually plays
            # (robust to slot order) or, for digit keys, by environment index.
            motion_keys_lib = list(getattr(self._motion_lib, "curr_motion_keys", []) or [])
            env_motion_key = {}
            for slot, env_idx in enumerate(self._replay_env_ids.tolist()):
                motion_idx = int(self._replay_motion_ids[slot].item())
                env_motion_key[int(env_idx)] = (
                    str(motion_keys_lib[motion_idx])
                    if 0 <= motion_idx < len(motion_keys_lib)
                    else ""
                )
            titles = {}
            for key, value in (env_titles or {}).items():
                key = str(key)
                if key.isdigit():
                    titles[int(key)] = str(value)
                else:
                    for env_idx, motion_key in env_motion_key.items():
                        if motion_key == key:
                            titles[env_idx] = str(value)
            per_env_writers, per_env_frames, per_env_files = {}, {}, {}
            per_env_done = {}
            for env_idx in self._replay_env_ids.tolist():
                env_idx = int(env_idx)
                title = titles.get(env_idx, "")
                slug = _re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
                file_name = f"env{env_idx:02d}" + (f"_{slug}" if slug else "") + ".mp4"
                per_env_files[env_idx] = os.path.join(per_env_video_dir, file_name)
                per_env_writers[env_idx] = imageio.get_writer(
                    per_env_files[env_idx],
                    fps=target_fps,
                    codec="libx264",
                    quality=5,
                    pixelformat="yuv420p",
                    macro_block_size=1,
                )
                per_env_frames[env_idx] = 0
                per_env_done[env_idx] = False
            try:
                from PIL import Image, ImageDraw, ImageFont

                def per_env_draw(frame_np, text):
                    if not text:
                        return frame_np
                    image = Image.fromarray(frame_np)
                    draw = ImageDraw.Draw(image)
                    try:
                        font = ImageFont.load_default(size=max(14, frame_np.shape[0] // 24))
                    except TypeError:
                        font = ImageFont.load_default()
                    box = draw.textbbox((0, 0), text, font=font)
                    draw.rectangle(
                        (6, 6, 6 + box[2] - box[0] + 12, 6 + box[3] - box[1] + 12),
                        fill=(0, 0, 0),
                    )
                    draw.text((12, 12), text, fill=(255, 255, 255), font=font)
                    return np.asarray(image)

            except Exception as exc:  # noqa: BLE001
                logger.info(f"[Video] per-env titles not drawn (PIL unavailable: {exc})")
                per_env_draw = None
            logger.info(
                f"[Video] Recording {len(per_env_writers)} per-env clips to: {per_env_video_dir}"
            )
        recording = "eval_camera" in self.env.scene.sensors and (
            video_writer is not None or per_env_writers is not None
        )

        # Run the rendering loop
        frame_time = 1.0 / target_fps
        last_time = time.time()
        frame_count = 0
        fixed_camera_eye = None
        fixed_camera_target = None
        fixed_camera_rotation_opengl_world = None
        camera_focus_entity = self.config.get("eval_camera_focus_entity", None)
        camera_focus_body_name = self.config.get(
            "eval_camera_focus_body_name", None
        )
        camera_focus_body_names = self.config.get(
            "eval_camera_focus_body_names", None
        )
        camera_motion_object_slot = self.config.get(
            "eval_camera_motion_object_slot", None
        )
        if camera_motion_object_slot is not None:
            camera_motion_object_slot = int(camera_motion_object_slot)
            if camera_motion_object_slot < 0:
                raise RuntimeError("eval camera motion-object slot must be nonnegative")
        focus_selector_count = sum(
            selector is not None
            for selector in (
                camera_focus_entity,
                camera_focus_body_name,
                camera_focus_body_names,
                camera_motion_object_slot,
            )
        )
        if focus_selector_count > 1:
            raise RuntimeError(
                "eval camera focus entity, body name/roster, and motion-object slot are "
                "mutually exclusive"
            )
        camera_focus_body_indices = None
        requested_body_names = None
        if camera_focus_body_names is not None:
            requested_body_names = list(camera_focus_body_names)
        elif camera_focus_body_name is not None:
            requested_body_names = [str(camera_focus_body_name)]
        if requested_body_names is not None:
            try:
                camera_focus_body_indices = resolve_camera_focus_body_indices(
                    self.motion_command.robot.body_names,
                    requested_body_names,
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
        camera_radial_from_root = bool(
            self.config.get("eval_camera_radial_from_root", False)
        )
        camera_axis_entity = self.config.get("eval_camera_axis_entity", None)
        camera_axis_trace_path = self.config.get(
            "eval_camera_source_orientation_trace_path", None
        )
        camera_axis_selector_count = sum(
            selector is not None
            for selector in (
                camera_axis_entity,
                camera_axis_trace_path,
                camera_motion_object_slot,
            )
        )
        if camera_axis_selector_count > 1:
            raise RuntimeError(
                "live-entity and frozen-source eval camera axes are mutually exclusive"
            )
        if camera_radial_from_root and camera_axis_selector_count:
            raise RuntimeError(
                "radial and object-axis eval camera modes are mutually exclusive"
            )
        if (
            camera_radial_from_root or camera_axis_selector_count
        ) and focus_selector_count != 1:
            raise RuntimeError(
                "radial/object-axis eval camera requires exactly one focus selector"
            )
        camera_radial_distance_m = float(
            self.config.get("eval_camera_radial_distance_m", 0.32)
        )
        camera_radial_height_m = float(
            self.config.get("eval_camera_radial_height_m", 0.10)
        )
        camera_tangent_offset_m = float(
            self.config.get("eval_camera_tangent_offset_m", 0.0)
        )
        camera_axis_local_index = int(
            self.config.get("eval_camera_axis_local_index", 2)
        )
        camera_axis_local_sign = float(
            self.config.get("eval_camera_axis_local_sign", 1.0)
        )
        camera_axis_local_up_index = self.config.get(
            "eval_camera_axis_local_up_index", None
        )
        if camera_axis_local_up_index is not None:
            camera_axis_local_up_index = int(camera_axis_local_up_index)
        camera_axis_local_up_sign = float(
            self.config.get("eval_camera_axis_local_up_sign", 1.0)
        )
        camera_axis_distance_m = float(
            self.config.get("eval_camera_axis_distance_m", 0.4)
        )
        camera_debug_trace = bool(
            self.config.get("eval_camera_debug_trace", False)
        )
        camera_debug_trace_stride = int(
            self.config.get("eval_camera_debug_trace_stride", 21)
        )
        if camera_debug_trace_stride <= 0:
            raise RuntimeError("eval camera debug trace stride must be positive")
        camera_runtime_trace_path = self.config.get(
            "eval_camera_runtime_trace_path", None
        )
        camera_runtime_trace_view_id = self.config.get(
            "eval_camera_runtime_trace_view_id", None
        )
        camera_runtime_trace_angle = self.config.get(
            "eval_camera_runtime_trace_angle", None
        )
        source_camera_quat_wxyz = None
        source_camera_trace_binding = None
        source_camera_motion_key = None
        source_camera_expected_frames = None
        camera_runtime_trace_records = None
        if camera_axis_trace_path is not None:
            if self.num_envs != 1:
                raise RuntimeError(
                    "frozen-source eval camera currently requires exactly one environment"
                )
            source_camera_motion_id = int(self._replay_motion_ids[0].item())
            source_camera_motion_key = str(
                self._motion_lib.curr_motion_keys[source_camera_motion_id]
            )
            source_camera_expected_frames = int(
                self._replay_num_steps_per_env[0].item()
            )
            source_camera_quat_wxyz, source_camera_trace_binding = (
                load_camera_source_orientation_trace(
                    camera_axis_trace_path,
                    expected_motion_key=source_camera_motion_key,
                    expected_frame_count=source_camera_expected_frames,
                    expected_replay_fps=float(target_fps),
                    device=self.device,
                    dtype=self.motion_command.robot.data.body_pos_w.dtype,
                )
            )
            if camera_axis_local_up_index is None:
                raise RuntimeError(
                    "frozen-source eval camera requires an explicit object-local up axis"
                )
            # The dense camera-command trace is an optional diagnostic output,
            # written only when a path is configured; it is never required.
            camera_runtime_trace_records = (
                [] if camera_runtime_trace_path else None
            )
        elif camera_runtime_trace_path is not None:
            raise RuntimeError(
                "dense camera-command trace is only valid with a frozen source trace"
            )
        if camera_focus_body_indices is not None:
            body_names = tuple(self.motion_command.robot.body_names)
            body_count = int(
                self.motion_command.robot.data.body_pos_w.shape[1]
            )
            if len(body_names) != body_count:
                raise RuntimeError(
                    "eval camera body-name/data ordering length mismatch: "
                    f"names={len(body_names)}, body_pos_w={body_count}"
                )
            resolved_body_names = tuple(
                body_names[index] for index in camera_focus_body_indices
            )
            if resolved_body_names != tuple(requested_body_names):
                raise RuntimeError(
                    "eval camera resolved body-name/index correspondence changed: "
                    f"requested={tuple(requested_body_names)}, "
                    f"resolved={resolved_body_names}, "
                    f"indices={tuple(camera_focus_body_indices)}"
                )
            if camera_debug_trace:
                logger.info(
                    "[CameraAudit] focus roster exact mapping: "
                    f"{tuple(zip(resolved_body_names, camera_focus_body_indices))}"
                )
        if camera_axis_selector_count and camera_debug_trace:
            logger.info(
                "[CameraAudit] object-axis mode: "
                f"entity={camera_axis_entity!r}, "
                f"source_trace={camera_axis_trace_path!r}, "
                f"motion_object_slot={camera_motion_object_slot!r}, "
                f"local_axis={camera_axis_local_index}, "
                f"sign={camera_axis_local_sign:+.0f}, "
                f"local_up_axis={camera_axis_local_up_index}, "
                f"local_up_sign={camera_axis_local_up_sign:+.0f}, "
                f"distance_m={camera_axis_distance_m}, "
                "object_quaternion_convention=wxyz, camera_convention=opengl"
            )

        camera_visibility_mode = self.config.get(
            "eval_camera_visibility_mode", "full_scene"
        )
        camera_visibility_policy = {"mode": "full_scene"}
        if camera_visibility_mode != "full_scene":
            if camera_visibility_mode != "isolate_robot_keep_exact_bodies":
                raise RuntimeError(
                    f"unsupported eval camera visibility mode: {camera_visibility_mode!r}"
                )
            if camera_axis_trace_path is None:
                raise RuntimeError(
                    "robot visibility isolation is only supported for frozen-source "
                    "close cameras"
                )
            requested_keep_body_names = self.config.get(
                "eval_camera_visibility_keep_body_names", None
            )
            if requested_keep_body_names is None:
                raise RuntimeError(
                    "camera visibility isolation requires an exact keep-body roster"
                )
            robot_root_path = str(
                self.config.get(
                    "eval_camera_visibility_robot_root_path",
                    "/World/envs/env_0/Robot",
                )
            )
            stage = omni.usd.get_context().get_stage()
            robot_root_prim = stage.GetPrimAtPath(robot_root_path)
            if not robot_root_prim.IsValid():
                raise RuntimeError(
                    f"camera visibility robot root is unavailable: {robot_root_path}"
                )
            robot_gprims = [
                prim
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(robot_root_path.rstrip("/") + "/")
                and prim.IsA(UsdGeom.Gprim)
            ]
            try:
                gprim_initial_visibilities = [
                    str(UsdGeom.Imageable(prim).ComputeVisibility())
                    for prim in robot_gprims
                ]
                gprim_effective_purposes = [
                    str(UsdGeom.Imageable(prim).ComputePurpose())
                    for prim in robot_gprims
                ]
                camera_visibility_policy = resolve_camera_visibility_gprim_policy(
                    tuple(self.motion_command.robot.body_names),
                    list(requested_keep_body_names),
                    [str(prim.GetPath()) for prim in robot_gprims],
                    robot_root_path=robot_root_path,
                    gprim_initial_visibilities=gprim_initial_visibilities,
                    gprim_effective_purposes=gprim_effective_purposes,
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            hidden_paths = set(camera_visibility_policy["hidden_gprim_paths"])
            kept_paths = set(camera_visibility_policy["kept_gprim_paths"])
            initially_invisible_paths = set(
                camera_visibility_policy["initially_invisible_gprim_paths"]
            )
            nonvisual_purpose_paths = set(
                camera_visibility_policy["nonvisual_purpose_gprim_paths"]
            )
            initial_state_by_path = {
                record["path"]: record
                for record in camera_visibility_policy["gprim_initial_states"]
            }
            for prim in robot_gprims:
                path = str(prim.GetPath())
                imageable = UsdGeom.Imageable(prim)
                if path in hidden_paths:
                    imageable.MakeInvisible()
                    if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
                        raise RuntimeError(
                            f"camera visibility failed to hide robot Gprim: {path}"
                        )
                elif path in kept_paths:
                    if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                        raise RuntimeError(
                            f"camera visibility protected visual Gprim became hidden: {path}"
                        )
                elif path in initially_invisible_paths:
                    if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
                        raise RuntimeError(
                            "camera visibility initially-invisible Gprim changed: "
                            f"{path}"
                        )
                elif path in nonvisual_purpose_paths:
                    initial_state = initial_state_by_path[path]
                    if (
                        str(imageable.ComputeVisibility())
                        != initial_state["computed_visibility"]
                        or str(imageable.ComputePurpose())
                        != initial_state["effective_purpose"]
                    ):
                        raise RuntimeError(
                            "camera visibility nonvisual-purpose Gprim changed: "
                            f"{path}"
                        )
                else:
                    raise RuntimeError(
                        f"camera visibility policy omitted robot Gprim: {path}"
                    )
            logger.info(
                "[CameraAudit] isolated robot geometry: "
                f"kept={len(kept_paths)}, hidden={len(hidden_paths)}, "
                f"initially_invisible={len(initially_invisible_paths)}, "
                f"nonvisual_purpose={len(nonvisual_purpose_paths)}, "
                f"bodies={camera_visibility_policy['resolved_keep_body_names']}"
            )
        elif camera_axis_trace_path is not None:
            raise RuntimeError(
                "frozen-source close camera requires fail-closed robot visibility isolation"
            )

        try:
            while self.step_replay():
                # Position camera BEFORE rendering (if recording)
                if recording:
                    root_pos = self.motion_command.robot.data.body_pos_w[:, 0]
                    camera_offset = self.config.get("eval_camera_offset", [-2, -2, 1])
                    camera_target_offset = self.config.get(
                        "eval_camera_target_offset", [0, 0, 0]
                    )
                    focus_pos = None
                    focus_body_positions = None
                    if camera_focus_body_indices is not None:
                        focus_body_positions = self.motion_command.robot.data.body_pos_w[
                            :, camera_focus_body_indices
                        ]
                        focus_pos = focus_body_positions.mean(dim=1)
                    elif camera_focus_entity is not None:
                        try:
                            focus_pos = self.env.scene[
                                str(camera_focus_entity)
                            ].data.root_pos_w
                        except (KeyError, AttributeError) as exc:
                            raise RuntimeError(
                                "eval camera focus entity is unavailable: "
                                f"{camera_focus_entity!r}"
                            ) from exc
                    elif camera_motion_object_slot is not None:
                        object_positions = self._motion_lib.get_object_root_pos(
                            self._replay_motion_ids, self._replay_time_steps
                        )
                        if camera_motion_object_slot >= object_positions.shape[1]:
                            raise RuntimeError(
                                "eval camera motion-object slot is out of bounds: "
                                f"slot={camera_motion_object_slot}, "
                                f"objects={object_positions.shape[1]}"
                            )
                        focus_pos = object_positions[:, camera_motion_object_slot]
                        if hasattr(self, "_replay_custom_origins"):
                            focus_pos = focus_pos + self._replay_custom_origins[
                                self._replay_env_ids
                            ]
                        else:
                            focus_pos = focus_pos + self.env.scene.env_origins[
                                self._replay_env_ids
                            ]
                    fix_camera = self.config.get("fix_camera_after_first_frame", False)
                    camera_rotation_opengl_world = None
                    object_quat_wxyz = None
                    if fix_camera and fixed_camera_eye is not None:
                        eye = fixed_camera_eye
                        target = fixed_camera_target
                        camera_rotation_opengl_world = (
                            fixed_camera_rotation_opengl_world
                        )
                    elif camera_axis_trace_path is not None:
                        replay_frame_index = int(self._replay_time_steps[0].item())
                        if (
                            replay_frame_index < 0
                            or replay_frame_index >= source_camera_expected_frames
                        ):
                            raise RuntimeError(
                                "frozen-source camera replay frame is out of bounds: "
                                f"{replay_frame_index}"
                            )
                        object_quat_wxyz = source_camera_quat_wxyz[
                            replay_frame_index : replay_frame_index + 1
                        ]
                        eye, target, camera_rotation_opengl_world = (
                            camera_pose_from_object_axis(
                                focus_pos,
                                object_quat_wxyz,
                                local_axis_index=camera_axis_local_index,
                                local_axis_sign=camera_axis_local_sign,
                                local_up_axis_index=camera_axis_local_up_index,
                                local_up_axis_sign=camera_axis_local_up_sign,
                                distance_m=camera_axis_distance_m,
                                target_offset=torch.tensor(
                                    camera_target_offset, device=self.device
                                ),
                                quaternion_convention="wxyz",
                            )
                        )
                    elif camera_motion_object_slot is not None:
                        object_quaternions = self._motion_lib.get_object_root_quat(
                            self._replay_motion_ids, self._replay_time_steps
                        )
                        if camera_motion_object_slot >= object_quaternions.shape[1]:
                            raise RuntimeError(
                                "eval camera motion-object quaternion slot is out of bounds"
                            )
                        object_quat_wxyz = object_quaternions[
                            :, camera_motion_object_slot
                        ]
                        eye, target, camera_rotation_opengl_world = (
                            camera_pose_from_object_axis(
                                focus_pos,
                                object_quat_wxyz,
                                local_axis_index=camera_axis_local_index,
                                local_axis_sign=camera_axis_local_sign,
                                local_up_axis_index=camera_axis_local_up_index,
                                local_up_axis_sign=camera_axis_local_up_sign,
                                distance_m=camera_axis_distance_m,
                                target_offset=torch.tensor(
                                    camera_target_offset, device=self.device
                                ),
                                quaternion_convention="wxyz",
                            )
                        )
                    elif camera_axis_entity is not None:
                        try:
                            object_quat_wxyz = self.env.scene[
                                str(camera_axis_entity)
                            ].data.root_quat_w
                        except (KeyError, AttributeError) as exc:
                            raise RuntimeError(
                                "eval camera axis entity is unavailable: "
                                f"{camera_axis_entity!r}"
                            ) from exc
                        eye, target, camera_rotation_opengl_world = (
                            camera_pose_from_object_axis(
                                focus_pos,
                                object_quat_wxyz,
                                local_axis_index=camera_axis_local_index,
                                local_axis_sign=camera_axis_local_sign,
                                local_up_axis_index=camera_axis_local_up_index,
                                local_up_axis_sign=camera_axis_local_up_sign,
                                distance_m=camera_axis_distance_m,
                                target_offset=torch.tensor(
                                    camera_target_offset, device=self.device
                                ),
                                quaternion_convention="wxyz",
                            )
                        )
                    elif camera_radial_from_root:
                        eye, target = camera_pose_radial_from_root(
                            root_pos,
                            focus_pos,
                            radial_distance_m=camera_radial_distance_m,
                            height_m=camera_radial_height_m,
                            tangent_offset_m=camera_tangent_offset_m,
                            target_offset=torch.tensor(
                                camera_target_offset, device=self.device
                            ),
                            fallback_xy=torch.tensor(
                                camera_offset[:2], device=self.device
                            ),
                        )
                    else:
                        eye, target = camera_pose_with_target_offset(
                            root_pos,
                            torch.tensor(camera_offset, device=self.device),
                            target_offset=torch.tensor(
                                camera_target_offset, device=self.device
                            ),
                            focus_pos=focus_pos,
                        )
                        if fix_camera:
                            fixed_camera_eye = eye.clone()
                            fixed_camera_target = target.clone()
                    if fix_camera and fixed_camera_eye is None:
                        fixed_camera_eye = eye.clone()
                        fixed_camera_target = target.clone()
                        if camera_rotation_opengl_world is not None:
                            fixed_camera_rotation_opengl_world = (
                                camera_rotation_opengl_world.clone()
                            )
                    if (
                        camera_debug_trace
                        and frame_count % camera_debug_trace_stride == 0
                    ):
                        camera_audit = {
                            "frame": int(frame_count),
                            "env_id": 0,
                            "eye_world": eye[0].detach().cpu().tolist(),
                            "target_world": target[0].detach().cpu().tolist(),
                        }
                        if focus_body_positions is not None:
                            camera_audit["focus_roster_world"] = {
                                name: position
                                for name, position in zip(
                                    requested_body_names,
                                    focus_body_positions[0]
                                    .detach()
                                    .cpu()
                                    .tolist(),
                                )
                            }
                            camera_audit["focus_mean_world"] = (
                                focus_pos[0].detach().cpu().tolist()
                            )
                        if object_quat_wxyz is not None:
                            camera_audit["object_quat_wxyz"] = (
                                object_quat_wxyz[0].detach().cpu().tolist()
                            )
                        if camera_rotation_opengl_world is not None:
                            target_camera = torch.einsum(
                                "nij,nj->ni",
                                camera_rotation_opengl_world.transpose(-1, -2),
                                target - eye,
                            )
                            camera_audit["target_camera_opengl_xyz"] = (
                                target_camera[0].detach().cpu().tolist()
                            )
                        logger.info(f"[CameraAudit] frame: {camera_audit}")
                    cam = self.env.scene["eval_camera"]
                    # Keep replay capture consistent with RenderEnvsRecorderTerm:
                    # Fabric pose writes alone can leave the tiled camera's RGB
                    # buffer stale/grey in headless replay video captures.
                    cam._view._sync_usd_on_fabric_write = True  # noqa: SLF001
                    if camera_rotation_opengl_world is not None:
                        from isaaclab.utils.math import quat_from_matrix

                        camera_quat_wxyz = quat_from_matrix(
                            camera_rotation_opengl_world
                        )
                        cam.set_world_poses(
                            eye,
                            camera_quat_wxyz,
                            convention="opengl",
                        )
                    else:
                        cam.set_world_poses_from_view(eye, target)

                # Render the scene (triggers camera render in headless mode)
                if hasattr(self.env, "sim"):
                    try:
                        self.env.sim.render()
                        if recording:
                            self.env.sim.render()
                    except Exception as e:  # noqa: BLE001
                        logger.info(f"Render error: {e}")

                # Capture video frame if recording
                if recording:
                    # Update camera to refresh its data after render
                    cam = self.env.scene["eval_camera"]
                    cam._is_outdated[:] = True  # noqa: SLF001
                    cam.update(dt=0.0, force_recompute=True)

                    # Grab frame from camera
                    rgb_frame = cam.data.output["rgb"]
                    if video_writer is not None:
                        frame = rgb_frame[0].cpu().numpy().astype(np.uint8)
                        video_writer.append_data(frame)
                    if per_env_writers is not None:
                        rgb_all = rgb_frame.cpu().numpy().astype(np.uint8)
                        steps_now = self._replay_time_steps.tolist()
                        last_steps = (self._replay_num_steps_per_env - 1).tolist()
                        for slot, env_idx in enumerate(self._replay_env_ids.tolist()):
                            env_idx = int(env_idx)
                            if per_env_done[env_idx]:
                                continue
                            frame_i = rgb_all[env_idx]
                            if per_env_draw is not None:
                                frame_i = per_env_draw(frame_i, titles.get(env_idx, ""))
                            per_env_writers[env_idx].append_data(frame_i)
                            per_env_frames[env_idx] += 1
                            if steps_now[slot] >= last_steps[slot]:
                                per_env_done[env_idx] = True
                    if camera_runtime_trace_records is not None:
                        # Cache the command tensors associated with this captured
                        # frame. This is not realized camera-pose readback.
                        camera_runtime_trace_records.append(
                            {
                                "frame_ordinal": int(frame_count),
                                "replay_frame_index": int(
                                    self._replay_time_steps[0].item()
                                ),
                                "source_object_quat_wxyz": (
                                    object_quat_wxyz[0].detach().cpu().tolist()
                                ),
                                "eye_world": eye[0].detach().cpu().tolist(),
                                "target_world": target[0].detach().cpu().tolist(),
                                "rotation_opengl_world": (
                                    camera_rotation_opengl_world[0]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ),
                            }
                        )
                    frame_count += 1

                # Frame rate limiting (only when not headless or not saving video)
                if not self.config.get("headless", False) or not recording:
                    current_time = time.time()
                    elapsed = current_time - last_time
                    sleep_time = max(0, (frame_time / speed) - elapsed)

                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    last_time = time.time()

        except KeyboardInterrupt:
            logger.info("\nReplay interrupted by user")

        # Close video writer
        if video_writer is not None:
            video_writer.close()
            logger.info(f"[Video] Saved {frame_count} frames to: {save_video_path}")
        if per_env_writers is not None:
            import json as _json

            for writer in per_env_writers.values():
                writer.close()
            clips = {
                str(env_idx): {
                    "file": os.path.basename(per_env_files[env_idx]),
                    "frames": int(per_env_frames[env_idx]),
                    "title": titles.get(env_idx, ""),
                    "motion_key": env_motion_key.get(env_idx, ""),
                    "motion_steps": int(self._replay_num_steps_per_env[slot].item()),
                    "complete": bool(per_env_done[env_idx]),
                }
                for slot, env_idx in enumerate(int(e) for e in self._replay_env_ids.tolist())
            }
            with open(os.path.join(per_env_video_dir, "clips.json"), "w") as handle:
                _json.dump({"fps": target_fps, "clips": clips}, handle, indent=1)
            logger.info(
                f"[Video] Saved {len(clips)} per-env clips to: {per_env_video_dir} "
                f"(frames per clip: {[c['frames'] for c in clips.values()]})"
            )

        if camera_runtime_trace_records is not None:
            if len(camera_runtime_trace_records) != source_camera_expected_frames:
                raise RuntimeError(
                    "dense camera-command trace is incomplete: "
                    f"expected={source_camera_expected_frames}, "
                    f"recorded={len(camera_runtime_trace_records)}"
                )
            runtime_trace = write_camera_runtime_trace(
                camera_runtime_trace_path,
                view_id=str(camera_runtime_trace_view_id or ""),
                angle=str(camera_runtime_trace_angle or ""),
                motion_key=source_camera_motion_key,
                source_orientation_trace=source_camera_trace_binding,
                visibility_policy=camera_visibility_policy,
                frames=camera_runtime_trace_records,
            )
            logger.info(
                "[CameraAudit] dense camera-command trace (not pose readback) "
                f"written: path={camera_runtime_trace_path}, "
                f"frames={runtime_trace['frame_count']}"
            )

        self._finalize_matched_contact_overlay()
        self._finalize_replay_reward_audit()
        logger.info("\nReplay complete!")
        return info

    def setup_replay_motion(
        self, motion_id=None, start_time_step=0, speed=1.0, loop=True, enable_vis=False
    ):
        """Initialize motion replay for all environments (low-level function for manual control).
        Use run_replay() instead if you want automatic rendering loop.

        After calling this, you need to manually call step_replay() in a loop
        and handle rendering yourself. For automatic rendering, use run_replay().

        Args:
            motion_id: The motion ID(s) to replay. Can be:
                       - None: uses current motion_ids for all envs
                       - int: single motion ID for all envs
                       - list/tensor: motion ID per environment
            start_time_step: Starting time step in the motion (default: 0)
            speed: Playback speed multiplier (default: 1.0)
            loop: Whether to loop the motion (default: True)
            enable_vis: Whether to enable debug visualization (default: False)

        Returns:
            List of dictionaries with motion metadata for each environment
        """  # noqa: D205
        if self._motion_lib is None:
            logger.info("No motion library available for replay")
            return None

        # Replay all environments
        num_replay_envs = self.num_envs
        replay_env_ids = torch.arange(num_replay_envs, device=self.device, dtype=torch.long)

        # Get motion IDs
        if motion_id is None:
            motion_ids = self.motion_command.motion_ids
        elif isinstance(motion_id, int):
            motion_ids = torch.full(
                (num_replay_envs,), motion_id, device=self.device, dtype=torch.long
            )
        elif isinstance(motion_id, list | tuple):
            if len(motion_id) != num_replay_envs:
                logger.info(
                    f"Error: motion_id list length ({len(motion_id)}) doesn't match number of environments ({num_replay_envs})"  # noqa: E501
                )
                return None
            motion_ids = torch.tensor(motion_id, device=self.device, dtype=torch.long)
        elif isinstance(motion_id, torch.Tensor):
            if motion_id.shape[0] != num_replay_envs:
                logger.info(
                    f"Error: motion_id tensor size ({motion_id.shape[0]}) doesn't match number of environments ({num_replay_envs})"  # noqa: E501
                )
                return None
            motion_ids = motion_id.to(self.device)
        else:
            logger.info(f"Error: Invalid motion_id type: {type(motion_id)}")
            return None

        # Get motion info for display
        num_steps_per_env = self._motion_lib.get_motion_num_steps(motion_ids)
        max_num_steps = num_steps_per_env.max().item()

        unique_motions = torch.unique(motion_ids)
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Replaying {num_replay_envs} Environments")
        logger.info(f"Unique motion IDs: {unique_motions.tolist()}")
        logger.info(f"Max frames: {max_num_steps}")
        logger.info(f"Max duration: {max_num_steps / 50.0:.2f}s (@ 50 FPS)")
        logger.info(f"Speed: {speed}x")
        logger.info(f"{'='*60}\n")

        # Enable visualization if requested
        if enable_vis and hasattr(self.motion_command, "_set_debug_vis_impl"):
            self.motion_command._set_debug_vis_impl(True)  # noqa: SLF001

        # Initialize contact point visualization if available and enabled
        self._replay_contact_visualizer = None
        self._replay_vis_enabled = enable_vis
        if enable_vis and VISUALIZATION_AVAILABLE:
            self._setup_contact_center_visualizer()

        # Replay state
        self._replay_active = True
        self._replay_paused = False
        self._replay_reverse = False
        self._replay_time_steps = torch.full(
            (num_replay_envs,), start_time_step, device=self.device, dtype=torch.long
        )
        self._replay_motion_ids = motion_ids
        self._replay_env_ids = replay_env_ids
        self._replay_speed = speed
        self._replay_loop = loop
        self._replay_num_steps_per_env = num_steps_per_env
        self._replay_max_num_steps = max_num_steps
        self._replay_object_audit_written = False

        # The matched HiPHI overlay is an explicit replay-only diagnostic.  It
        # is independent of the broad ``enable_vis`` debug switch so normal
        # contact-center markers remain off in clean videos.
        self._replay_matched_contact_overlay = None
        overlay_path = str(
            self.config.get("replay_matched_contact_overlay_path", "") or ""
        )
        if overlay_path:
            self._setup_matched_contact_overlay(overlay_path)

        # Pre-load table metadata from pkl files for ALL motions (per-env)
        # This supports multi-motion replay where each env can have different table positions
        self._replay_table_pos = None  # Shape: (num_envs, 3)
        self._replay_table_quat = None  # Shape: (num_envs, 4)
        if hasattr(self.env, "scene") and "table" in self.env.scene.rigid_objects:
            try:
                import os

                import joblib

                # Derive meta directory from motion_file path in config
                motion_lib_cfg = getattr(self.motion_command.cfg, "motion_lib_cfg", None)
                motion_file = motion_lib_cfg.get("motion_file", "") if motion_lib_cfg else ""

                if motion_file and "/robot" in motion_file:
                    if os.path.isdir(motion_file):
                        meta_dir = motion_file.replace("/robot", "/meta")
                    else:
                        meta_dir = os.path.dirname(motion_file).replace("/robot", "/meta")
                else:
                    meta_dir = "data/motion_lib_grab/meta"

                # Pre-load motion file data if it's a single file (for table fallback)
                motion_file_data = None
                if motion_file and os.path.isfile(motion_file) and not os.path.isdir(meta_dir):
                    motion_file_data = joblib.load(motion_file)

                # Load table metadata for each environment's assigned motion
                table_pos_list = []
                table_quat_list = []
                table_scale_list = []  # Also load scales for offset calculation
                loaded_count = 0
                loaded_from_motion_file = 0

                for env_idx in range(num_replay_envs):
                    motion_idx = motion_ids[env_idx].item()
                    motion_key = self._motion_lib.curr_motion_keys[motion_idx]
                    meta_file = os.path.join(meta_dir, f"{motion_key}.pkl")

                    if os.path.exists(meta_file):
                        meta = joblib.load(meta_file)
                        pos = torch.tensor(
                            meta.get("table_pos", [0.0, 0.0, 0.8]), device=self.device
                        ).float()
                        quat = torch.tensor(
                            meta.get("table_quat", [1.0, 0.0, 0.0, 0.0]), device=self.device
                        ).float()
                        scale = torch.tensor(
                            meta.get("table_scale", [1.0, 1.0, 1.0]), device=self.device
                        ).float()
                        table_pos_list.append(pos)
                        table_quat_list.append(quat)
                        table_scale_list.append(scale)
                        loaded_count += 1
                    elif motion_file_data is not None and motion_key in motion_file_data:
                        # Fallback: try to get table data from motion file directly
                        motion_data = motion_file_data[motion_key]
                        if "table_pos" in motion_data and "table_quat" in motion_data:
                            pos = torch.tensor(motion_data["table_pos"], device=self.device).float()
                            quat = torch.tensor(
                                motion_data["table_quat"], device=self.device
                            ).float()
                            scale = torch.tensor(
                                motion_data.get("table_scale", [1.0, 1.0, 1.0]), device=self.device
                            ).float()
                            table_pos_list.append(pos)
                            table_quat_list.append(quat)
                            table_scale_list.append(scale)
                            loaded_from_motion_file += 1
                        else:
                            # No table data in motion file, use default
                            table_pos_list.append(
                                torch.tensor([0.0, 0.0, 0.8], device=self.device).float()
                            )
                            table_quat_list.append(
                                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).float()
                            )
                            table_scale_list.append(
                                torch.tensor([1.0, 1.0, 1.0], device=self.device).float()
                            )
                    else:
                        # Fallback: use default table position
                        table_pos_list.append(
                            torch.tensor([0.0, 0.0, 0.8], device=self.device).float()
                        )
                        table_quat_list.append(
                            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).float()
                        )
                        table_scale_list.append(
                            torch.tensor([1.0, 1.0, 1.0], device=self.device).float()
                        )

                self._replay_table_pos = torch.stack(table_pos_list, dim=0)  # (num_envs, 3)
                self._replay_table_quat = torch.stack(table_quat_list, dim=0)  # (num_envs, 4)
                self._replay_table_scale = torch.stack(table_scale_list, dim=0)  # (num_envs, 3)

                # Apply fixed table_offset if configured
                table_offset = self.config.get("table_offset", None)
                if table_offset is not None:
                    offset_tensor = torch.tensor(
                        table_offset, device=self.device, dtype=self._replay_table_pos.dtype
                    )
                    self._replay_table_pos = self._replay_table_pos + offset_tensor

                # Apply maximal X offset based on object starting position
                # This computes the maximum valid offset range such that object stays on table
                x_offset_maximal = self.config.get("replay_table_x_offset_maximal", False)
                x_offset_margin = self.config.get(
                    "replay_table_x_offset_margin", 0.05
                )  # 5cm safety margin

                if x_offset_maximal:
                    # Get base table width from config (required for maximal offset mode)
                    table_size_cfg = self.config.get("table_size")
                    if table_size_cfg is None:
                        raise ValueError(
                            "replay_table_x_offset_maximal requires table_size to be set in config"
                        )
                    base_table_width = table_size_cfg[0]

                    # Get object starting position (step 0) for each env
                    object_start_pos = self._motion_lib.get_object_root_pos(
                        motion_ids,
                        torch.zeros(num_replay_envs, device=self.device, dtype=torch.long),
                    )[
                        :, 0
                    ]  # Shape: (num_envs, 3) -> take first object if multiple
                    object_start_x = object_start_pos[:, 0]  # (num_envs,)

                    # Get table position and scaled width per env
                    table_x = self._replay_table_pos[:, 0]  # (num_envs,)
                    scale_x = self._replay_table_scale[:, 0]  # (num_envs,)
                    actual_width = base_table_width * scale_x  # (num_envs,)
                    half_width = actual_width / 2.0

                    # Compute valid offset range per env:
                    # Object must stay on table after offset dx
                    # dx_min = object_x - table_x - half_width + margin
                    # dx_max = object_x - table_x + half_width - margin
                    relative_obj_x = object_start_x - table_x
                    dx_min = relative_obj_x - half_width + x_offset_margin
                    dx_max = relative_obj_x + half_width - x_offset_margin

                    # Sample X offset uniformly within valid range per env
                    rand_vals = torch.rand(num_replay_envs, device=self.device)
                    x_offsets = dx_min + rand_vals * (dx_max - dx_min)

                    self._replay_table_pos[:, 0] += x_offsets

                    logger.info("Replay table X offset (maximal mode):")
                    logger.info(
                        f"  Base table width: {base_table_width:.3f}m, margin: {x_offset_margin:.3f}m"
                    )
                    logger.info(f"  dx_min range: [{dx_min.min():.3f}, {dx_min.max():.3f}]")
                    logger.info(f"  dx_max range: [{dx_max.min():.3f}, {dx_max.max():.3f}]")
                    logger.info(f"  Applied offsets: [{x_offsets.min():.3f}, {x_offsets.max():.3f}]")

                # Apply fixed XY offset range (legacy mode, used if maximal mode is disabled)
                # Format: [x_min, x_max, y_min, y_max] - per-env randomized
                xy_offset_range = self.config.get("replay_table_xy_offset_range", None)
                if xy_offset_range is not None and not x_offset_maximal:
                    x_min, x_max, y_min, y_max = xy_offset_range

                    # Sample X offset per environment
                    x_offsets = torch.empty(num_replay_envs, device=self.device).uniform_(
                        x_min, x_max
                    )
                    self._replay_table_pos[:, 0] += x_offsets

                    # Sample Y offset per environment (typically y_min = y_max for fixed offset)
                    y_offsets = torch.empty(num_replay_envs, device=self.device).uniform_(
                        y_min, y_max
                    )
                    self._replay_table_pos[:, 1] += y_offsets

                    logger.info(
                        f"Replay table XY offset: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]"
                    )

                if loaded_count > 0:
                    logger.info(
                        f"Loaded table metadata for {loaded_count}/{num_replay_envs} environments from {meta_dir}"
                    )
                elif loaded_from_motion_file > 0:
                    logger.info(
                        f"Loaded table metadata for {loaded_from_motion_file}/{num_replay_envs} environments from motion file"  # noqa: E501
                    )

            except Exception as e:  # noqa: BLE001
                logger.info(f"Warning: Could not load table metadata: {e}")

        # Setup keyboard controls for replay
        if not self.config.headless:
            self._setup_replay_keyboard()

        logger.info("Replay Controls:")
        logger.info("  G: Pause/Resume")
        logger.info("  B: Toggle Reverse Play")
        logger.info("  LEFT/RIGHT: Step backward/forward (when paused)")
        logger.info("  R: Restart from beginning")
        logger.info("  ESC: Exit replay")
        logger.info("  +/-: Increase/Decrease speed")
        logger.info("")

        # Return metadata
        results = []
        for i, (mid, nsteps) in enumerate(
            zip(motion_ids.tolist(), num_steps_per_env.tolist(), strict=False)
        ):
            # Get motion key from motion library (curr_motion_keys is the active list)
            if hasattr(self._motion_lib, "curr_motion_keys") and mid < len(
                self._motion_lib.curr_motion_keys
            ):
                motion_key = self._motion_lib.curr_motion_keys[mid]
            elif hasattr(self._motion_lib, "motion_keys") and mid < len(
                self._motion_lib.motion_keys
            ):
                motion_key = self._motion_lib.motion_keys[mid]
            else:
                motion_key = f"motion_{mid}"
            results.append(
                {
                    "env_id": i,
                    "motion_id": mid,
                    "motion_key": motion_key,
                    "num_steps": nsteps,
                    "duration": nsteps / 50.0,
                    "episode_horizon_steps": int(
                        getattr(self.env, "max_episode_length", 0) or 0
                    ),
                }
            )
        self._setup_replay_reward_audit(results)
        return results

    def _setup_replay_reward_audit(self, motion_info):
        """Initialize the opt-in oracle reward/termination replay audit."""
        self._replay_reward_audit_path = str(
            self.config.get("replay_reward_audit_path", "") or ""
        )
        self._replay_reward_audit_records = []
        self._replay_reward_audit_motion_info = motion_info
        if not self._replay_reward_audit_path:
            return
        if getattr(self.env, "reward_manager", None) is None:
            raise RuntimeError("Replay reward audit requires an environment reward manager")
        self.env.reward_manager.reset()
        termination_manager = getattr(self.env, "termination_manager", None)
        if termination_manager is not None:
            termination_manager.reset()
        logger.info(f"Replay reward audit enabled: {self._replay_reward_audit_path}")

    @staticmethod
    def _replay_audit_stats(values):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {
                "count": int(values.size),
                "finite_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "nonzero_count": 0,
                "mean": None,
                "p05": None,
                "median": None,
                "p95": None,
                "min": None,
                "max": None,
            }
        return {
            "count": int(values.size),
            "finite_count": int(finite.size),
            "positive_count": int(np.sum(finite > 0.5)),
            "negative_count": int(np.sum(finite < -0.5)),
            "nonzero_count": int(np.sum(np.abs(finite) > 1.0e-12)),
            "mean": float(np.mean(finite)),
            "p05": float(np.quantile(finite, 0.05)),
            "median": float(np.median(finite)),
            "p95": float(np.quantile(finite, 0.95)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }

    def _record_replay_reward_audit(self):
        """Evaluate configured rewards on the exact replayed reference state."""
        if not getattr(self, "_replay_reward_audit_path", ""):
            return

        simulate_contacts = bool(
            self.config.get("replay_reward_audit_simulate_contacts", True)
        )
        physics_dt = float(getattr(self.env, "physics_dt", 0.005))
        if simulate_contacts:
            self.env.sim.step(render=False)
            self.env.scene.update(dt=physics_dt)
        else:
            self.env.scene.update(dt=0.0)

        command = self.motion_command
        command.motion_ids.copy_(self._replay_motion_ids)
        command.motion_start_time_steps.zero_()
        command.time_steps.copy_(self._replay_time_steps)
        if hasattr(command, "_refresh_relative_body_targets"):
            command._refresh_relative_body_targets()  # noqa: SLF001
        if hasattr(command, "_update_per_env_first_contact"):
            command._update_per_env_first_contact(self._replay_env_ids)  # noqa: SLF001
        if hasattr(self.env, "episode_length_buf"):
            self.env.episode_length_buf.copy_(self._replay_time_steps)

        reward_manager = self.env.reward_manager
        dt = float(getattr(self.env, "step_dt", physics_dt))
        reward_manager.compute(dt=dt)
        reward_names = list(getattr(reward_manager, "_term_names", []))
        reward_values = reward_manager._step_reward.detach().cpu().numpy()  # noqa: SLF001
        reward_weights = [
            float(cfg.weight) for cfg in getattr(reward_manager, "_term_cfgs", [])
        ]

        termination_manager = getattr(self.env, "termination_manager", None)
        termination_names = []
        termination_values = None
        if termination_manager is not None:
            termination_manager.compute()
            termination_names = list(getattr(termination_manager, "_term_names", []))
            termination_values = (
                termination_manager._term_dones.detach().cpu().numpy()  # noqa: SLF001
            )

        metric_values = {}
        canonical_getter = getattr(
            command.motion_lib, "get_reference_contact_state", None
        )
        canonical_reference = (
            canonical_getter(
                command.motion_ids,
                command.motion_start_time_steps + command.time_steps,
            )
            if callable(canonical_getter)
            else None
        )
        if canonical_reference is not None:
            for side, hand in (("left", "left_hand"), ("right", "right_hand")):
                entity_index, body_indices = canonical_hand_contact_indices(hand)
                selected_reference = canonical_reference[
                    :, entity_index, list(body_indices)
                ]
                label = (selected_reference > 0.5).any(dim=-1).float()
                eligible = (
                    torch.abs(selected_reference) > 0.5
                ).any(dim=-1).float()
                command.metrics[
                    f"canonical_hand_reference_contact/{side}"
                ] = label
                command.metrics[
                    f"canonical_hand_reference_eligible/{side}"
                ] = eligible
        for name, value in getattr(command, "metrics", {}).items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    metric_values[name] = np.full(
                        self.num_envs, float(value.detach().cpu().item())
                    )
                elif value.shape[0] == self.num_envs:
                    array = value.detach().reshape(self.num_envs, -1).cpu().numpy()
                    if array.shape[1] == 1:
                        metric_values[name] = array[:, 0]
            elif isinstance(value, int | float):
                metric_values[name] = np.full(self.num_envs, float(value))

        for env_idx in range(self.num_envs):
            motion_id = int(self._replay_motion_ids[env_idx].item())
            motion_key = self._motion_lib.curr_motion_keys[motion_id]
            record_metrics = {
                name: float(value[env_idx]) for name, value in metric_values.items()
            }
            record_metrics.update(_replay_required_contact_metrics(record_metrics))
            record = {
                "env_id": env_idx,
                "motion_key": motion_key,
                "frame": int(self._replay_time_steps[env_idx].item()),
                "reward_terms": {
                    name: float(reward_values[env_idx, term_idx])
                    for term_idx, name in enumerate(reward_names)
                },
                "reward_weights": {
                    name: reward_weights[term_idx]
                    for term_idx, name in enumerate(reward_names)
                },
                "metrics": record_metrics,
                "terminations": (
                    {
                        name: bool(termination_values[env_idx, term_idx])
                        for term_idx, name in enumerate(termination_names)
                    }
                    if termination_values is not None
                    else {}
                ),
            }
            self._replay_reward_audit_records.append(record)

    def _finalize_replay_reward_audit(self):
        """Aggregate and write oracle replay reward diagnostics."""
        path = getattr(self, "_replay_reward_audit_path", "")
        if not path:
            return
        import json
        from pathlib import Path

        records = self._replay_reward_audit_records
        reward_names = sorted(
            {name for record in records for name in record["reward_terms"]}
        )
        metric_names = sorted({name for record in records for name in record["metrics"]})
        termination_names = sorted(
            {name for record in records for name in record["terminations"]}
        )
        contact_metric_names = {
            "left": (
                "canonical_hand_reference_contact/left",
                "rp2_grasp_reference_contact/left",
            ),
            "right": (
                "canonical_hand_reference_contact/right",
                "rp2_grasp_reference_contact/right",
            ),
        }
        reward_validity_metrics = {
            "object_tracking_reward": (
                "rp2_object_tracking_reward_active_count",
                0.5,
            ),
            "grasp_reward": (
                "reference_hand_contact/right",
                0.5,
            ),
            "grasp_reward_left": (
                "reference_hand_contact/left",
                0.5,
            ),
            "rp2_source_relation_tracking": (
                "rp2_source_relation/active_links",
                0.5,
            ),
            "active_articulation_joint_tracking": (
                "active_articulation/valid_joint_count",
                0.5,
            ),
            "canonical_required_nonhand_contact": (
                "canonical_required_contact/count",
                0.5,
            ),
            "canonical_forbidden_full_body_contact": (
                "canonical_forbidden_contact/count",
                0.5,
            ),
        }

        def reference_contact_active(record, side):
            if side == "either":
                return any(
                    float(record["metrics"].get(metric_name, 0.0)) > 0.5
                    for metric_names in contact_metric_names.values()
                    for metric_name in metric_names
                )
            return any(
                float(record["metrics"].get(metric_name, 0.0)) > 0.5
                for metric_name in contact_metric_names[side]
            )

        reward_summary = {}
        for name in reward_names:
            values = [record["reward_terms"][name] for record in records]
            weight = next(
                record["reward_weights"][name]
                for record in records
                if name in record["reward_weights"]
            )
            summary = self._replay_audit_stats(values)
            summary["weight"] = float(weight)
            if weight != 0.0:
                summary["unweighted"] = self._replay_audit_stats(
                    [value / weight for value in values]
                )
            phase_summary = {}
            for side in ("left", "right", "either"):
                phase_values = [
                    record["reward_terms"][name]
                    for record in records
                    if reference_contact_active(record, side)
                ]
                phase = self._replay_audit_stats(phase_values)
                if weight != 0.0:
                    phase["unweighted"] = self._replay_audit_stats(
                        [value / weight for value in phase_values]
                    )
                phase_summary[side] = phase
            summary["reference_contact_active"] = phase_summary
            validity = reward_validity_metrics.get(name)
            if validity is not None:
                metric_name, threshold = validity
                valid_values = [
                    record["reward_terms"][name]
                    for record in records
                    if float(record["metrics"].get(metric_name, 0.0))
                    > threshold
                ]
                valid_summary = self._replay_audit_stats(valid_values)
                if weight != 0.0:
                    valid_summary["unweighted"] = self._replay_audit_stats(
                        [value / weight for value in valid_values]
                    )
                summary["validity_condition"] = {
                    "metric": metric_name,
                    "threshold": threshold,
                    "samples": len(valid_values),
                    "summary": valid_summary,
                }
            reward_summary[name] = summary

        metric_summary = {
            name: self._replay_audit_stats(
                [record["metrics"][name] for record in records if name in record["metrics"]]
            )
            for name in metric_names
        }
        termination_summary = {
            name: {
                "count": int(
                    sum(bool(record["terminations"].get(name, False)) for record in records)
                ),
                "fraction": float(
                    np.mean(
                        [
                            bool(record["terminations"].get(name, False))
                            for record in records
                        ]
                    )
                ),
                "reference_contact_active": {
                    side: {
                        "samples": int(
                            sum(
                                reference_contact_active(record, side)
                                for record in records
                            )
                        ),
                        "count": int(
                            sum(
                                bool(record["terminations"].get(name, False))
                                and reference_contact_active(record, side)
                                for record in records
                            )
                        ),
                    }
                    for side in ("left", "right", "either")
                },
                **_replay_termination_frame_summary(
                    records,
                    name,
                    self._replay_reward_audit_motion_info,
                ),
            }
            for name in termination_names
        }
        for summary in termination_summary.values():
            for phase in summary["reference_contact_active"].values():
                phase["fraction"] = (
                    float(phase["count"] / phase["samples"])
                    if phase["samples"]
                    else None
                )
        nonfinite_rewards = sum(
            not np.isfinite(value)
            for record in records
            for value in record["reward_terms"].values()
        )
        payload = {
            "schema_version": 2,
            "mode": "oracle_reference_state_replay",
            "simulate_contacts": bool(
                self.config.get("replay_reward_audit_simulate_contacts", True)
            ),
            "motion_info": self._replay_reward_audit_motion_info,
            "num_frame_env_samples": len(records),
            "reward_terms": reward_summary,
            "metrics": metric_summary,
            "required_contact_evidence": _replay_required_contact_summary(records),
            "hand_contact_alignment": (
                _replay_hand_contact_alignment_summary(records)
            ),
            "terminations": termination_summary,
            "nonfinite_reward_values": int(nonfinite_rewards),
            "passed_structural_checks": bool(records) and nonfinite_rewards == 0,
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info(f"Replay reward audit written: {output}")

    def _setup_replay_keyboard(self):
        """Setup keyboard controls for replay mode"""  # noqa: D415
        try:
            if not hasattr(self, "keyboard_interface"):
                from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg

                cfg = Se2KeyboardCfg()
                self.keyboard_interface = Se2Keyboard(cfg)

            # Add replay-specific callbacks
            self.keyboard_interface.add_callback("G", self._toggle_replay_pause)
            self.keyboard_interface.add_callback("LEFT", self._replay_step_backward)
            self.keyboard_interface.add_callback("RIGHT", self._replay_step_forward)
            self.keyboard_interface.add_callback("R", self._restart_replay)
            self.keyboard_interface.add_callback("ESCAPE", self._exit_replay)
            self.keyboard_interface.add_callback("EQUAL", self._increase_replay_speed)
            self.keyboard_interface.add_callback("MINUS", self._decrease_replay_speed)
            self.keyboard_interface.add_callback("B", self._toggle_reverse_play)
        except Exception as e:  # noqa: BLE001
            logger.info(f"Could not setup replay keyboard controls: {e}")

    def _setup_contact_center_visualizer(self):
        """Setup visualization markers for per-hand contact centers during replay."""
        if not VISUALIZATION_AVAILABLE:
            return

        has_left = hasattr(self._motion_lib, "_motion_object_contact_center_left")
        has_right = hasattr(self._motion_lib, "_motion_object_contact_center_right")
        if not has_left and not has_right:
            logger.info("No contact center available in motion library for visualization")
            return

        try:
            self._replay_contact_visualizers = {}

            if has_left:
                left_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Replay/contact_center_left",
                    markers={
                        "contact": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.0, 1.0)  # Blue for left hand
                            ),
                        ),
                    },
                )
                self._replay_contact_visualizers["left_hand"] = VisualizationMarkers(left_cfg)
                self._replay_contact_visualizers["left_hand"].set_visibility(True)

            if has_right:
                right_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Replay/contact_center_right",
                    markers={
                        "contact": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 1.0, 1.0)  # Green for right hand
                            ),
                        ),
                    },
                )
                self._replay_contact_visualizers["right_hand"] = VisualizationMarkers(right_cfg)
                self._replay_contact_visualizers["right_hand"].set_visibility(True)

            hands = list(self._replay_contact_visualizers.keys())
            logger.info(f"Contact center visualizer initialized for: {', '.join(hands)}")

            # Keep _replay_contact_visualizer as a truthy check for the update loop
            self._replay_contact_visualizer = True

        except Exception as e:  # noqa: BLE001
            logger.info(f"Could not setup contact center visualizer: {e}")
            self._replay_contact_visualizer = None

    def _setup_matched_contact_overlay(self, overlay_path):
        """Load and initialize the opt-in compact matched-contact overlay."""
        from gear_sonic.utils.hiphi_matched_contact_overlay import (
            MatchedContactOverlayError,
            load_matched_contact_overlay,
        )
        if not VISUALIZATION_AVAILABLE:
            raise RuntimeError(
                "matched contact overlay requires Isaac Lab visualization markers"
            )
        if self.num_envs != 1:
            raise RuntimeError("matched contact overlay requires exactly one replay env")
        trace_path = str(
            self.config.get("replay_matched_contact_overlay_trace_path", "") or ""
        )
        if not trace_path:
            raise RuntimeError("matched contact overlay requires a live trace output path")
        motion_id = int(self._replay_motion_ids[0].item())
        motion_key = str(self._motion_lib.curr_motion_keys[motion_id])
        expected_frames = int(self._replay_num_steps_per_env[0].item())
        try:
            spec = load_matched_contact_overlay(
                overlay_path,
                expected_motion_key=motion_key,
                expected_frame_count=expected_frames,
                expected_replay_fps=50.0,
            )
        except MatchedContactOverlayError as exc:
            raise RuntimeError(str(exc)) from exc

        body_names = tuple(self.motion_command.robot.body_names)
        wrist_indices = {}
        for side in ("left", "right"):
            body_name = f"{side}_wrist_yaw_link"
            if body_names.count(body_name) != 1:
                raise RuntimeError(
                    f"matched overlay expected exactly one robot body {body_name!r}"
                )
            wrist_indices[side] = body_names.index(body_name)

        def sphere(path, color, radius):
            cfg = VisualizationMarkersCfg(
                prim_path=path,
                markers={
                    "point": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=color,
                            emissive_color=color,
                        ),
                    )
                },
            )
            marker = VisualizationMarkers(cfg)
            marker.set_visibility(True)
            return marker

        def line(path, color):
            cfg = VisualizationMarkersCfg(
                prim_path=path,
                markers={
                    "line": sim_utils.CylinderCfg(
                        radius=0.004,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=color,
                            emissive_color=color,
                        ),
                    )
                },
            )
            marker = VisualizationMarkers(cfg)
            marker.set_visibility(True)
            return marker

        colors = {
            "left": (0.0, 1.0, 1.0),
            "right": (1.0, 0.0, 1.0),
        }
        visualizers = {
            "target": sphere(
                "/Visuals/Replay/HiPHIContactOverlay/source_target",
                (1.0, 0.82, 0.0),
                0.025,
            )
        }
        for side in ("left", "right"):
            visualizers[f"actual_{side}"] = sphere(
                f"/Visuals/Replay/HiPHIContactOverlay/actual_{side}",
                colors[side],
                0.020,
            )
            visualizers[f"line_{side}"] = line(
                f"/Visuals/Replay/HiPHIContactOverlay/error_{side}",
                colors[side],
            )
        self._replay_matched_contact_overlay = {
            "spec": spec,
            "trace_path": trace_path,
            "trace_records": [],
            "wrist_indices": wrist_indices,
            "visualizers": visualizers,
        }
        logger.info(
            "[HiPHIOverlay] initialized shared source target/live rubber overlay: "
            f"motion={motion_key}, frames={spec.frame_count}, path={spec.path}"
        )

    def _update_matched_contact_overlay(self):
        """Place compact glyphs and record exact per-frame live errors."""
        state = getattr(self, "_replay_matched_contact_overlay", None)
        if not state:
            return
        from gear_sonic.utils.hiphi_matched_contact_overlay import (
            inverse_rotate_wxyz,
            segment_marker_pose,
        )
        from gear_sonic.isaac_utils.rotations import quat_rotate

        spec = state["spec"]
        frame_index = int(self._replay_time_steps[0].item())
        values = spec.frame(frame_index)
        if hasattr(self, "_replay_custom_origins"):
            env_origin = self._replay_custom_origins[self._replay_env_ids][0]
        else:
            env_origin = self.env.scene.env_origins[self._replay_env_ids][0]
        dtype = self.motion_command.robot.data.body_pos_w.dtype
        hidden = torch.tensor([0.0, 0.0, -1000.0], device=self.device, dtype=dtype)
        targets = hidden.repeat(2, 1)
        actuals = {side: hidden.clone() for side in ("left", "right")}
        trace_sides = {}

        for side_index, side in enumerate(("left", "right")):
            target_valid = bool(values["target_valid"][side_index])
            actual_valid = bool(values["actual_valid"][side_index])
            if not target_valid:
                trace_sides[side] = {
                    "valid": False,
                    "target_valid": False,
                    "actual_valid": False,
                }
                continue
            target = torch.as_tensor(
                values["target_world_m"][side_index], device=self.device, dtype=dtype
            ) + env_origin
            targets[side_index] = target
            target_np = target.detach().cpu().numpy().astype(np.float64)
            if not actual_valid:
                trace_sides[side] = {
                    "valid": False,
                    "target_valid": True,
                    "actual_valid": False,
                    "object_slot": int(values["object_slot"][side_index]),
                    "segment_id": int(values["segment_id"][side_index]),
                    "target_world_m": target_np.tolist(),
                    "status": "quarantined_no_validated_rubber_patch",
                }
                continue
            patch = torch.as_tensor(
                values["patch_body_m"][side_index], device=self.device, dtype=dtype
            ).reshape(1, 3)
            wrist_index = state["wrist_indices"][side]
            wrist_position = self.motion_command.robot.data.body_pos_w[0, wrist_index]
            wrist_quaternion = self.motion_command.robot.data.body_quat_w[0, wrist_index]
            actual = wrist_position + quat_rotate(
                wrist_quaternion.reshape(1, 4), patch, w_last=False
            )[0]
            actuals[side] = actual

            actual_np = actual.detach().cpu().numpy().astype(np.float64)
            error_world = actual_np - target_np
            error_local = inverse_rotate_wxyz(
                values["object_quat_wxyz"][side_index], error_world
            )
            trace_sides[side] = {
                "valid": True,
                "target_valid": True,
                "actual_valid": True,
                "object_slot": int(values["object_slot"][side_index]),
                "segment_id": int(values["segment_id"][side_index]),
                "target_world_m": target_np.tolist(),
                "actual_rubber_world_m": actual_np.tolist(),
                "error_world_m": error_world.tolist(),
                "error_object_local_m": np.asarray(error_local).tolist(),
                "error_mm": float(np.linalg.norm(error_world) * 1000.0),
            }

        state["visualizers"]["target"].visualize(translations=targets)
        for side in ("left", "right"):
            state["visualizers"][f"actual_{side}"].visualize(
                translations=actuals[side].reshape(1, 3)
            )
            if trace_sides[side]["actual_valid"]:
                target_np = np.asarray(trace_sides[side]["target_world_m"])
                actual_np = np.asarray(trace_sides[side]["actual_rubber_world_m"])
                midpoint, orientation, scale = segment_marker_pose(actual_np, target_np)
            else:
                midpoint = hidden.detach().cpu().numpy()
                orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
                scale = np.asarray([1.0, 1.0, 1.0e-6])
            state["visualizers"][f"line_{side}"].visualize(
                translations=torch.as_tensor(
                    midpoint, device=self.device, dtype=dtype
                ).reshape(1, 3),
                orientations=torch.as_tensor(
                    orientation, device=self.device, dtype=dtype
                ).reshape(1, 4),
                scales=torch.as_tensor(scale, device=self.device, dtype=dtype).reshape(1, 3),
            )

        records = state["trace_records"]
        if records and int(records[-1]["replay_frame_index"]) >= frame_index:
            raise RuntimeError("matched contact overlay replay frames are not strictly ordered")
        records.append(
            {
                "video_frame_ordinal": len(records),
                "replay_frame_index": frame_index,
                "sides": trace_sides,
            }
        )

    def _finalize_matched_contact_overlay(self):
        """Publish the complete live error trace for CPU HUD composition."""
        state = getattr(self, "_replay_matched_contact_overlay", None)
        if not state:
            return
        import json
        import os
        from pathlib import Path

        spec = state["spec"]
        records = state["trace_records"]
        if len(records) != spec.frame_count:
            raise RuntimeError(
                "matched contact overlay trace is incomplete: "
                f"expected={spec.frame_count}, actual={len(records)}"
            )
        if [int(record["replay_frame_index"]) for record in records] != list(
            range(spec.frame_count)
        ):
            raise RuntimeError("matched contact overlay trace does not cover every replay frame")
        output = Path(state["trace_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract": "hiphi_matched_contact_overlay_live_trace_v1",
            "status": "ok",
            "motion_key": spec.motion_key,
            "overlay_spec": str(spec.path),
            "replay_fps": spec.replay_fps,
            "frame_count": spec.frame_count,
            "side_order": ["left", "right"],
            "max_pairs_per_frame": 2,
            "target_authority": "shared_source_overlay_spec",
            "actual_authority": "live_isaaclab_wrist_yaw_pose",
            "identity_text": spec.identity_text,
            "status_text": spec.status_text,
            "quarantined": spec.quarantined,
            "dust_debug_inset": spec.dust_debug_inset,
            "records": records,
        }
        with output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        logger.info(f"[HiPHIOverlay] live trace written: {output}")

    def _update_contact_center_visualization(self):
        """Update per-hand contact center visualization during replay."""
        if not getattr(self, "_replay_contact_visualizers", None):
            return

        from gear_sonic.isaac_utils.rotations import quat_rotate

        # Get object pose (shared by both hands)
        object_root_pos = self._motion_lib.get_object_root_pos(
            self._replay_motion_ids, self._replay_time_steps
        )[
            :, 0, :
        ]  # (num_envs, 3)
        object_root_quat = self._motion_lib.get_object_root_quat(
            self._replay_motion_ids, self._replay_time_steps
        )[
            :, 0, :
        ]  # (num_envs, 4)

        # Environment origin offsets
        if hasattr(self, "_replay_custom_origins"):
            env_origins = self._replay_custom_origins[self._replay_env_ids]
        else:
            env_origins = self.env.scene.env_origins[self._replay_env_ids]

        # Hidden position for markers when no contact
        hidden_pos = torch.tensor([[0.0, 0.0, -1000.0]], device=self.device)

        for hand, visualizer in self._replay_contact_visualizers.items():
            contact_center = self._motion_lib.get_object_contact_center(
                self._replay_motion_ids, self._replay_time_steps, hand=hand
            )
            if contact_center is None:
                visualizer.visualize(translations=hidden_pos)
                continue

            # Move markers far away for envs with no contact (zero contact center)
            valid_mask = torch.norm(contact_center, dim=-1) > 1e-6
            rotated = quat_rotate(object_root_quat, contact_center, w_last=False)
            world_center = rotated + object_root_pos + env_origins
            world_center[~valid_mask] = hidden_pos
            visualizer.visualize(translations=world_center)

    def _toggle_replay_pause(self):
        """Toggle pause state for replay"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_paused = not self._replay_paused
            status = "PAUSED" if self._replay_paused else "PLAYING"
            max_frame = self._replay_time_steps.max().item()
            logger.info(f"Replay {status} (max frame {max_frame}/{self._replay_max_num_steps})")

    def _replay_step_backward(self):
        """Step backward one frame when paused"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active and self._replay_paused:
            self._replay_time_steps = torch.clamp(self._replay_time_steps - 1, min=0)
            self._update_replay_frame()
            max_frame = self._replay_time_steps.max().item()
            logger.info(f"Frame {max_frame}/{self._replay_max_num_steps}")
            self.env.sim.render()

    def _replay_step_forward(self):
        """Step forward one frame when paused"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active and self._replay_paused:
            self._replay_time_steps = torch.minimum(
                self._replay_time_steps + 1, self._replay_num_steps_per_env - 1
            )
            self._update_replay_frame()
            max_frame = self._replay_time_steps.max().item()
            logger.info(f"Frame {max_frame}/{self._replay_max_num_steps}")
            self.env.sim.render()

    def _restart_replay(self):
        """Restart replay from beginning"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_time_steps.fill_(0)
            self._replay_paused = False
            self._update_replay_frame()
            logger.info("Replay restarted from beginning")
            self.env.sim.render()

    def _exit_replay(self):
        """Exit replay mode"""  # noqa: D415
        if hasattr(self, "_replay_active"):
            self._replay_active = False
            self._restore_replay_static_context()
            # Hide contact center visualizers
            for vis in getattr(self, "_replay_contact_visualizers", {}).values():
                vis.set_visibility(False)
            overlay = getattr(self, "_replay_matched_contact_overlay", None)
            if overlay:
                for vis in overlay["visualizers"].values():
                    vis.set_visibility(False)
            logger.info("Exiting replay mode")
            self.env.sim.render()

    def _increase_replay_speed(self):
        """Increase replay speed"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_speed = min(5.0, self._replay_speed * 1.25)
            logger.info(f"Replay speed: {self._replay_speed:.2f}x")

    def _decrease_replay_speed(self):
        """Decrease replay speed"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_speed = max(0.1, self._replay_speed * 0.8)
            logger.info(f"Replay speed: {self._replay_speed:.2f}x")

    def _toggle_reverse_play(self):
        """Toggle reverse playback mode"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_reverse = not self._replay_reverse
            direction = "REVERSE" if self._replay_reverse else "FORWARD"
            logger.info(f"Replay direction: {direction}")

    def _update_replay_frame(self):
        """Update all robot states to match the current replay frames"""  # noqa: D415
        if not hasattr(self, "_replay_active") or not self._replay_active:
            return

        # Get motion data at current time steps for all environments
        root_pos = self._motion_lib.get_root_pos_w(self._replay_motion_ids, self._replay_time_steps)
        root_quat = self._motion_lib.get_root_quat_w(
            self._replay_motion_ids, self._replay_time_steps
        )
        root_lin_vel = self._motion_lib.get_root_lin_vel_w(
            self._replay_motion_ids, self._replay_time_steps
        )
        root_ang_vel = self._motion_lib.get_root_ang_vel_w(
            self._replay_motion_ids, self._replay_time_steps
        )
        motion_lib_joint_pos = self._motion_lib.get_dof_pos(
            self._replay_motion_ids, self._replay_time_steps
        )
        motion_lib_joint_vel = self._motion_lib.get_dof_vel(
            self._replay_motion_ids, self._replay_time_steps
        )

        # Handle DOF mismatch between motion library (e.g., 29 DOF) and robot (e.g., 43 DOF)
        robot_num_joints = self.motion_command.robot.num_joints
        motion_lib_num_dof = motion_lib_joint_pos.shape[-1]

        if robot_num_joints > motion_lib_num_dof and self._body_joint_indices is not None:
            # Robot has more DOFs than motion lib (e.g., 43 DOF robot with 29 DOF motion data)
            # Use body joint indices for proper mapping
            num_envs = motion_lib_joint_pos.shape[0]

            # Create full joint tensors with zeros for all DOFs
            joint_pos = torch.zeros(
                num_envs, robot_num_joints, device=self.device, dtype=motion_lib_joint_pos.dtype
            )
            joint_vel = torch.zeros(
                num_envs, robot_num_joints, device=self.device, dtype=motion_lib_joint_vel.dtype
            )

            # Map motion lib data to body joint indices (using G1_ISAACLab_ORDER mapping)
            joint_pos[:, self._body_joint_indices] = motion_lib_joint_pos
            joint_vel[:, self._body_joint_indices] = motion_lib_joint_vel

            # Use hand DOFs from motion lib if available, otherwise default to zero
            hand_dof_pos = self._motion_lib.get_hand_dof_pos(
                self._replay_motion_ids, self._replay_time_steps
            )
            if hand_dof_pos is not None:
                # Name-map reference hand dof to robot joints (IsaacLab hand order is interleaved
                # L/R, so positional [:, -14:] scrambles fingers). Fallback to positional if the
                # name map is unavailable.
                if getattr(self, "_reference_hand_dof_indices", None) is not None:
                    joint_pos[:, self._reference_hand_dof_indices] = hand_dof_pos
                else:
                    joint_pos[:, -hand_dof_pos.shape[-1]:] = hand_dof_pos
            elif bool(self.config.get("replay_reference_contact_hand_primitives", False)):
                primitive_actions = torch.full(
                    (num_envs, len(self._finger_primitive_map)),
                    -1.0,
                    device=self.device,
                    dtype=motion_lib_joint_pos.dtype,
                )
                for primitive_idx, primitive in enumerate(self._finger_primitive_map):
                    hand = f"{primitive['action_name']}_hand"
                    in_contact = self._motion_lib.get_object_in_contact(
                        self._replay_motion_ids,
                        self._replay_time_steps,
                        hand=hand,
                    )
                    if in_contact is not None:
                        primitive_actions[:, primitive_idx] = torch.where(
                            in_contact > 0.5,
                            torch.ones_like(in_contact),
                            -torch.ones_like(in_contact),
                        )
                finger_targets = self._convert_primitive_to_finger_actions(
                    primitive_actions
                )
                joint_pos[:, self._hand_joint_indices] = finger_targets
                joint_vel[:, self._hand_joint_indices] = 0.0
        else:
            joint_pos = motion_lib_joint_pos
            joint_vel = motion_lib_joint_vel

        # Add environment origin offsets (use custom grid origins if set)
        if hasattr(self, "_replay_custom_origins"):
            root_pos = root_pos + self._replay_custom_origins[self._replay_env_ids]
        else:
            root_pos = root_pos + self.env.scene.env_origins[self._replay_env_ids]

        # Write state to simulation for all environments
        self.motion_command.robot.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=self._replay_env_ids
        )
        self.motion_command.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=self._replay_env_ids,
        )

        # Active articulation replay is opt-in and mutually exclusive with the
        # legacy rigid-object replay below. Write both root and reduced joint
        # state so the reference video exercises the authored mechanism.
        active_articulations = getattr(self.env.scene, "articulations", {})
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
                name for name in required_channels if not hasattr(self._motion_lib, name)
            ]
            if missing_channels:
                raise ValueError(
                    "Active articulation replay requires articulation reference channels; "
                    f"missing={missing_channels}"
                )

            articulation = active_articulations["active_articulation"]
            articulation_root_pos = self._motion_lib.get_active_articulation_root_pos(
                self._replay_motion_ids, self._replay_time_steps
            )
            if hasattr(self, "_replay_custom_origins"):
                articulation_root_pos = (
                    articulation_root_pos + self._replay_custom_origins[self._replay_env_ids]
                )
            else:
                articulation_root_pos = (
                    articulation_root_pos
                    + self.env.scene.env_origins[self._replay_env_ids]
                )
            articulation_root_state = torch.cat(
                [
                    articulation_root_pos,
                    self._motion_lib.get_active_articulation_root_quat(
                        self._replay_motion_ids, self._replay_time_steps
                    ),
                    self._motion_lib.get_active_articulation_root_lin_vel(
                        self._replay_motion_ids, self._replay_time_steps
                    ),
                    self._motion_lib.get_active_articulation_root_ang_vel(
                        self._replay_motion_ids, self._replay_time_steps
                    ),
                ],
                dim=-1,
            )
            articulation.write_root_state_to_sim(
                articulation_root_state, env_ids=self._replay_env_ids
            )

            articulation_joint_pos = self._motion_lib.get_active_articulation_joint_pos(
                self._replay_motion_ids, self._replay_time_steps
            )
            articulation_joint_vel = self._motion_lib.get_active_articulation_joint_vel(
                self._replay_motion_ids, self._replay_time_steps
            )
            reference_joint_names = list(
                self.config.get("active_articulation_reference_joint_names", [])
            )
            if reference_joint_names:
                missing_joints = [
                    name
                    for name in articulation.joint_names
                    if name not in reference_joint_names
                ]
                if missing_joints:
                    raise ValueError(
                        "Articulation replay joints are absent from reference ordering: "
                        f"missing={missing_joints}, reference={reference_joint_names}"
                    )
                source_order = [
                    reference_joint_names.index(name) for name in articulation.joint_names
                ]
                articulation_joint_pos = articulation_joint_pos[:, source_order]
                articulation_joint_vel = articulation_joint_vel[:, source_order]
            if articulation_joint_pos.shape[1] != articulation.num_joints:
                raise ValueError(
                    "Active articulation replay joint count does not match the USD: "
                    f"reference={articulation_joint_pos.shape[1]}, "
                    f"usd={articulation.num_joints}, names={list(articulation.joint_names)}"
                )
            soft_limits = articulation.data.soft_joint_pos_limits[self._replay_env_ids]
            articulation_joint_pos = torch.clamp(
                articulation_joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1]
            )
            articulation.write_joint_state_to_sim(
                articulation_joint_pos,
                articulation_joint_vel,
                env_ids=self._replay_env_ids,
            )

        # Get legacy rigid-object motion data from the motion library.
        elif hasattr(self._motion_lib, "_motion_object_root_pos") and hasattr(
            self._motion_lib, "_motion_object_root_quat"
        ):
            object_root_pos = self._motion_lib.get_object_root_pos(
                self._replay_motion_ids, self._replay_time_steps
            )
            object_root_quat = self._motion_lib.get_object_root_quat(
                self._replay_motion_ids, self._replay_time_steps
            )

            # Add environment origin offsets to object position
            if hasattr(self, "_replay_custom_origins"):
                object_root_pos = object_root_pos + self._replay_custom_origins[
                    self._replay_env_ids
                ].unsqueeze(1)
            else:
                object_root_pos = object_root_pos + self.env.scene.env_origins[
                    self._replay_env_ids
                ].unsqueeze(1)

            # Write object state to simulation. In full-scene mode, each object
            # trajectory index maps to the corresponding object_<usd_stem> asset.
            # Shape: object_root_pos is (num_envs, max_num_objects, 3)
            # Shape: object_root_quat is (num_envs, max_num_objects, 4)
            all_object_keys = [
                key for key in self.env.scene.rigid_objects if key.startswith("object_")
            ]
            stems_by_motion = getattr(self.env.cfg, "object_usd_stems_by_motion", None) or {}
            has_ordered_object_stems = bool(stems_by_motion) or bool(
                getattr(self.env.cfg, "object_usd_stems", None)
            )
            if all_object_keys and has_ordered_object_stems:
                audit_reset_index = None
                if not getattr(self, "_replay_object_audit_written", False):
                    audit_reset_index = self.motion_command._next_object_audit_reset_index()  # noqa: SLF001
                    self._replay_object_audit_written = True
                inactive_base = INACTIVE_OBJECT_BASE_OFFSET.to(self.device)
                inactive_quat = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]], device=self.device
                )
                for motion_id in torch.unique(self._replay_motion_ids):
                    group_mask = self._replay_motion_ids == motion_id
                    group_env_ids = self._replay_env_ids[group_mask]
                    motion_id_int = int(motion_id.item())
                    motion_key = self._motion_lib.curr_motion_keys[motion_id_int]
                    object_scene_keys = motion_order_object_scene_keys(
                        self.env,
                        max_objects=object_root_pos.shape[1],
                        motion_key=motion_key,
                    )
                    for obj_idx, obj_key in enumerate(object_scene_keys):
                        if obj_idx >= object_root_pos.shape[1]:
                            break
                        obj_pos = object_root_pos[group_mask, obj_idx, :]
                        obj_quat = object_root_quat[group_mask, obj_idx, :]
                        object_root_pose = torch.cat([obj_pos, obj_quat], dim=-1)
                        self.env.scene[obj_key].write_root_pose_to_sim(
                            object_root_pose,
                            env_ids=group_env_ids,
                        )

                    inactive_idx = 0
                    for obj_key in all_object_keys:
                        if obj_key in object_scene_keys:
                            continue
                        z_offset = torch.tensor(
                            [0.0, 0.0, -inactive_idx * INACTIVE_OBJECT_Z_SPACING],
                            device=self.device,
                        )
                        if hasattr(self, "_replay_custom_origins"):
                            inactive_origins = self._replay_custom_origins[group_env_ids]
                        else:
                            inactive_origins = self.env.scene.env_origins[group_env_ids]
                        inactive_pos = inactive_origins + inactive_base + z_offset
                        inactive_pose = torch.cat(
                            [
                                inactive_pos,
                                inactive_quat.expand(len(group_env_ids), -1),
                            ],
                            dim=-1,
                        )
                        self.env.scene[obj_key].write_root_pose_to_sim(
                            inactive_pose,
                            env_ids=group_env_ids,
                        )
                        inactive_idx += 1
                    if audit_reset_index is not None:
                        self.motion_command._append_full_scene_reset_audit(  # noqa: SLF001
                            reset_index=audit_reset_index,
                            motion_id=motion_id_int,
                            motion_key=motion_key,
                            env_ids=group_env_ids,
                            object_scene_keys=object_scene_keys,
                            all_object_keys=all_object_keys,
                            object_pos=object_root_pos[group_mask],
                            object_quat=object_root_quat[group_mask],
                            reference_time_steps=self._replay_time_steps[group_mask],
                        )
                if audit_reset_index is not None:
                    self.motion_command._write_object_audit()  # noqa: SLF001
            else:
                multi_object_keys = motion_order_object_scene_keys(
                    self.env, max_objects=object_root_pos.shape[1]
                )
                if multi_object_keys:
                    object_scene_keys = multi_object_keys
                elif "object" in self.env.scene.rigid_objects:
                    object_scene_keys = ["object"]
                else:
                    object_scene_keys = []

                for obj_idx, obj_key in enumerate(object_scene_keys):
                    if obj_idx >= object_root_pos.shape[1]:
                        break
                    obj_pos = object_root_pos[:, obj_idx, :]
                    obj_quat = object_root_quat[:, obj_idx, :]
                    object_root_pose = torch.cat([obj_pos, obj_quat], dim=-1)
                    self.env.scene[obj_key].write_root_pose_to_sim(
                        object_root_pose, env_ids=self._replay_env_ids
                    )

        if hasattr(self.env, "scene") and "table" in self.env.scene.rigid_objects:
            # Use pre-loaded per-motion table metadata (loaded during setup_replay_motion)
            # _replay_table_pos and _replay_table_quat have shape (num_envs, 3) and (num_envs, 4)
            if self._replay_table_pos is not None:
                # Use per-env table positions directly (already the right shape)
                table_pos = self._replay_table_pos.clone()
                table_quat = self._replay_table_quat.clone()
            else:
                # Fallback: derive from object position with hardcoded offset
                table_pos = self._motion_lib.get_object_root_pos(
                    self._replay_motion_ids, torch.zeros_like(self._replay_time_steps)
                )[:, 0].clone()
                table_pos[:, 2] = 0.76  # Table height
                table_pos[:, 1] -= 0.15
                table_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(
                    len(self._replay_env_ids), 1
                )

                # Apply table_offset if configured (for fallback path)
                table_offset = self.config.get("table_offset", None)
                if table_offset is not None:
                    offset_tensor = torch.tensor(
                        table_offset, device=self.device, dtype=table_pos.dtype
                    )
                    table_pos = table_pos + offset_tensor

            # Add environment origin offsets (same as robot and object)
            if hasattr(self, "_replay_custom_origins"):
                table_pos = table_pos + self._replay_custom_origins[self._replay_env_ids]
            else:
                table_pos = table_pos + self.env.scene.env_origins[self._replay_env_ids]

            table_root_pose = torch.cat([table_pos, table_quat], dim=-1)
            self.env.scene["table"].write_root_pose_to_sim(
                table_root_pose, env_ids=self._replay_env_ids
            )

        # Visualize contact points if enabled
        if (
            hasattr(self, "_replay_contact_visualizer")
            and self._replay_contact_visualizer is not None
            and hasattr(self, "_replay_vis_enabled")
            and self._replay_vis_enabled
        ):
            self._update_contact_center_visualization()

        self.env.sim.forward()
        self._update_matched_contact_overlay()
        self._record_replay_reward_audit()

    def step_replay(self):
        """Step the replay forward for all environments. Call this in a loop to animate the motions.
        Returns False when replay is complete.
        """  # noqa: D205
        if not hasattr(self, "_replay_active") or not self._replay_active:
            return False

        # Update frames based on speed and direction
        if not self._replay_paused:
            if self._replay_reverse:
                # Playing in reverse
                self._replay_time_steps -= int(self._replay_speed)

                # Handle beginning of motions per environment
                at_start = self._replay_time_steps < 0
                if at_start.any():
                    if self._replay_loop:
                        # Loop to end when reaching start
                        self._replay_time_steps[at_start] = (
                            self._replay_num_steps_per_env[at_start] - 1
                        ).to(torch.long)
                    else:
                        # If any motion reached start and not looping, end replay
                        logger.info("Replay complete (reversed to start)!")
                        self._replay_active = False
                        return False
            else:
                # Playing forward (normal)
                self._replay_time_steps += int(self._replay_speed)

                # Handle end of motions per environment
                completed = self._replay_time_steps >= self._replay_num_steps_per_env
                if completed.any():
                    if self._replay_loop:
                        # Loop completed motions
                        self._replay_time_steps[completed] = 0
                        # Decrement loop counter if it's an integer (countdown mode)
                        if not isinstance(self._replay_loop, bool):
                            self._replay_loop -= 1
                            logger.info(f"Loop completed, {self._replay_loop} loops remaining")
                            if self._replay_loop <= 0:
                                logger.info("All loops complete!")
                                self._replay_active = False
                                return False
                    elif getattr(self, "_replay_hold_completed", False):
                        # Hold finished environments at their last frame; end only
                        # when every environment has finished.
                        self._replay_time_steps = torch.minimum(
                            self._replay_time_steps, self._replay_num_steps_per_env - 1
                        )
                        if bool(completed.all()):
                            logger.info("Replay complete (all motions finished)!")
                            self._replay_active = False
                            return False
                    else:
                        # If any motion is complete and not looping, end replay
                        logger.info("Replay complete!")
                        self._replay_active = False
                        return False

        # Update the robot states
        self._update_replay_frame()

        # Print progress every 50 frames (based on max time step)
        max_time_step = self._replay_time_steps.max().item()
        if max_time_step % 50 == 0 and max_time_step > 0:
            progress = (max_time_step / self._replay_max_num_steps) * 100
            logger.info(
                f"Progress: {progress:.1f}% (max frame {max_time_step}/{self._replay_max_num_steps})"
            )

        return True
