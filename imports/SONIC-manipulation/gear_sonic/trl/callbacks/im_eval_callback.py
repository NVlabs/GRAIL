import gc
import json
import os
from pathlib import Path
import time
from datetime import datetime

from loguru import logger
import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import TrainerCallback

from gear_sonic.trl.utils.eval_action_prefix_override import (
    apply_zero_latent_prefix,
    validate_zero_latent_prefix_config,
)
from gear_sonic.trl.utils.eval_action_sampling import (
    DETERMINISTIC_MEAN,
    MEAN_VS_SEEDED_SAMPLE,
    make_per_env_generators,
    mean_vs_seeded_sample,
    per_env_sample_seeds,
    validate_eval_action_sampling_config,
)


def create_html_table(metrics_dict):
    """
    Create a sortable HTML table for metrics logging using DataTables.

    Args:
        metrics_dict: Dictionary containing metrics data with keys like 'mpjpe_g', 'mpjpe_l', 'mpjpe_pa',
                     'terminated', 'motion_keys', etc.

    Returns:
        str: HTML string containing the sortable table
    """
    if not metrics_dict or len(metrics_dict) == 0:
        return wandb.Html("<p>No metrics data available</p>")

    # Get the motion keys and number of motions
    motion_keys = metrics_dict.get("motion_keys", [])
    if len(motion_keys) == 0:
        return wandb.Html("<p>No motion data available</p>")

    num_motions = len(motion_keys)

    # Get metric names (excluding special keys)
    special_keys = {"terminated", "motion_keys"}
    metric_names = [key for key in metrics_dict.keys() if key not in special_keys]

    # Create table header
    html = """
<!-- HTML Table -->
<table id="my-table" class="display">
  <thead>
    <tr>
      <th>Motion Key</th>
      <th>Terminated</th>
"""

    # Add metric column headers
    for metric_name in metric_names:
        html += f"      <th>{metric_name}</th>\n"

    html += """    </tr>
  </thead>
  <tbody>
"""

    # Create table rows
    for i in range(num_motions):
        motion_key = motion_keys[i] if i < len(motion_keys) else f"Motion_{i}"
        terminated = "Yes" if metrics_dict.get("terminated", [True] * num_motions)[i] else "No"

        html += f"    <tr><td>{motion_key}</td><td>{terminated}</td>"

        # Add metric values
        for metric_name in metric_names:
            metric_values = metrics_dict[metric_name]
            if i < len(metric_values):
                value = metric_values[i]
                # Format the value appropriately
                if isinstance(value, int | float):
                    if abs(value) < 0.001:
                        formatted_value = f"{value:.6f}"
                    elif abs(value) < 1:
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                html += f"<td>{formatted_value}</td>"
            else:
                html += "<td>N/A</td>"

        html += "</tr>\n"

    html += """  </tbody>
</table>

<!-- DataTables CSS and JS dependencies -->
<link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
<script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>

<!-- DataTables Initialization -->
<script>
  $(document).ready(function () {
    $('#my-table').DataTable({
      pageLength: 100,
      order: [[0, 'asc']],
    });
  });
</script>
"""
    return wandb.Html(html)


class ImEvalCallback(TrainerCallback):
    """Callback to evaluate motion imtiation during training. Supports multigpu ."""

    def __init__(
        self,
        eval_frequency,
        empty_cache_freq=20,
        eval_only=False,
        output_dir=None,
        log_keys=None,
        prime_policy_before_loop: bool = True,
        action_mode: str = DETERMINISTIC_MEAN,
        action_sample_seed_base: int = 20260714,
        action_deterministic_env_count: int = 1,
        action_log_steps: int = 0,
        zero_latent_prefix_motion_key: str = "",
        zero_latent_prefix_steps: int = 0,
        zero_latent_prefix_dim: int = 64,
        zero_latent_prefix_expected_hand_dim: int = 2,
    ):
        super().__init__()
        self.eval_frequency = eval_frequency
        self.empty_cache_freq = empty_cache_freq
        self.output_dir = output_dir
        self.eval_only = eval_only
        self.in_eval_mode = False
        self.render_only = False
        self.log_keys = log_keys
        self.prime_policy_before_loop = prime_policy_before_loop
        self.action_mode = str(action_mode)
        self.action_sample_seed_base = int(action_sample_seed_base)
        self.action_deterministic_env_count = int(action_deterministic_env_count)
        self.action_log_steps = int(action_log_steps)
        self.zero_latent_prefix_motion_key = str(zero_latent_prefix_motion_key or "")
        self.zero_latent_prefix_steps = int(zero_latent_prefix_steps)
        self.zero_latent_prefix_dim = int(zero_latent_prefix_dim)
        self.zero_latent_prefix_expected_hand_dim = int(
            zero_latent_prefix_expected_hand_dim
        )
        self._zero_latent_prefix_enabled = False
        self._action_sample_seeds = None
        self._action_sample_generators = None
        self._has_object = False

    def _save_debug_state_log(self):
        if self._debug_state_records is None or not self._debug_state_log_path:
            return
        save_path = Path(self._debug_state_log_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_path), "w") as _f:
            json.dump(self._debug_state_records, _f)
        print(f"[DEBUG] State log: {save_path} ({len(self._debug_state_records)} steps)")

    def _motion_command_cfg(self):
        try:
            return self.env.env.cfg.commands.motion
        except Exception:
            return None

    def _rp2_support_diagnostic_checks(self):
        cfg = self._motion_command_cfg()
        if cfg is not None and not bool(
            getattr(cfg, "enable_rp2_support_diagnostic_labels", True)
        ):
            return []

        def _threshold(name: str, default: float) -> float:
            return float(getattr(cfg, name, default)) if cfg is not None else default

        return [
            (
                "rp2_object_pos_error_max/moving_support",
                "moving_support_pos",
                _threshold("rp2_diag_moving_support_pos_error_threshold", 0.5),
            ),
            (
                "rp2_object_ori_error_max/moving_support",
                "moving_support_ori",
                _threshold("rp2_diag_moving_support_ori_error_threshold", 1.5),
            ),
            (
                "rp2_object_height_error_max/moving_support",
                "moving_support_height",
                _threshold("rp2_diag_moving_support_height_error_threshold", 0.3),
            ),
            (
                "rp2_object_support_contact_force/moving_support",
                "support_force",
                _threshold("rp2_diag_support_contact_force_threshold", 500.0),
            ),
            (
                "rp2_object_support_contact_force_active/moving_support",
                "active_support_force",
                _threshold("rp2_diag_support_contact_force_active_threshold", 250.0),
            ),
            (
                "rp2_object_support_body_pos_error/max",
                "support_body_pos",
                _threshold("rp2_diag_support_body_pos_error_threshold", 0.2),
            ),
            (
                "rp2_object_ee_body_height_error/max",
                "ee_body_height",
                _threshold("rp2_diag_ee_body_height_error_threshold", 0.25),
            ),
        ]

    def _rp2_support_diagnostic_labels(self, motion_cmd):
        checks = self._rp2_support_diagnostic_checks()
        if not checks:
            return []
        labels = [set() for _ in range(self.env.num_envs)]
        for metric_name, label, threshold in checks:
            value = motion_cmd.metrics.get(metric_name)
            if not isinstance(value, torch.Tensor):
                continue
            metric_values = value.detach()
            if metric_values.ndim == 0:
                metric_values = metric_values.repeat(self.env.num_envs)
            metric_values = metric_values.reshape(-1)
            count = min(self.env.num_envs, metric_values.shape[0])
            for env_id in range(count):
                if float(metric_values[env_id].item()) > threshold:
                    labels[env_id].add(label)
        return ["|".join(sorted(label_set)) for label_set in labels]

    def on_step_end(self, args, state, control, **kwargs):

        self.env = kwargs.get("env")
        self.model = kwargs.get("model")
        self.accelerator = kwargs.get("accelerator")

        # Init debug state log from env config
        if not hasattr(self, "_debug_state_records"):
            debug_log = self.env.config.get("debug_state_log", None) if self.env else None
            if debug_log is None:
                # Also check wrapper's full config
                try:
                    debug_log = self.env.env.cfg.debug_state_log
                except Exception:
                    pass
            self._debug_state_records = [] if debug_log else None
            self._debug_state_log_path = debug_log
        self.device = self.accelerator.device
        self.args = args
        self.model.eval()

        if (state.global_step + 1) % self.eval_frequency == 0:
            metrics_eval = self.evaluate_policy()

    def save_metrics_eval(self, metrics_eval):
        if not self.accelerator.is_main_process:
            return

        metrics_json = {}
        for k, v in metrics_eval.items():
            if k in ["eval/all_metrics_dict", "eval/failed_metrics_dict"]:
                metrics_json[k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        metrics_json[k][kk] = vv.tolist()
                    else:
                        metrics_json[k][kk] = vv
            elif isinstance(v, np.ndarray):
                metrics_json[k] = v.tolist()
            else:
                metrics_json[k] = v

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "metrics_eval.json"), "w") as f:
            print(f"Saving metrics_eval to {os.path.join(self.output_dir, 'metrics_eval.json')}")
            if self.log_keys is not None:
                metrics_json["log_keys"] = self.log_keys
            json.dump(metrics_json, f, indent=4)

    @torch.no_grad()
    def evaluate_policy(self):

        self.accelerator.wait_for_everyone()
        with torch.no_grad():
            self._eval_mode()
            print(
                "============================================Evaluating policy============================================"
            )

            self._pre_evaluate_policy()
            actor_state = {"done_indices": [], "stop": False}
            step = 0
            self.eval_policy = self._get_inference_policy()
            obs_dict = self.env.reset_all(global_rank=self.args.global_rank)
            self.model.policy.init_rollout()

            validate_eval_action_sampling_config(
                mode=self.action_mode,
                num_envs=self.env.num_envs,
                deterministic_env_count=self.action_deterministic_env_count,
                log_steps=self.action_log_steps,
            )
            self._zero_latent_prefix_enabled = validate_zero_latent_prefix_config(
                action_mode=self.action_mode,
                prime_policy_before_loop=self.prime_policy_before_loop,
                motion_key=self.zero_latent_prefix_motion_key,
                prefix_steps=self.zero_latent_prefix_steps,
                latent_dim=self.zero_latent_prefix_dim,
                expected_hand_dim=self.zero_latent_prefix_expected_hand_dim,
            )
            if self._zero_latent_prefix_enabled:
                print(
                    "[EVAL_ZERO_LATENT_PREFIX] "
                    f"motion_key={self.zero_latent_prefix_motion_key} "
                    f"absolute_reference_steps=[0,{self.zero_latent_prefix_steps}) "
                    f"latent_dim={self.zero_latent_prefix_dim} "
                    f"untouched_hand_dim={self.zero_latent_prefix_expected_hand_dim} "
                    "target=zero action_mode=deterministic_mean",
                    flush=True,
                )
            if self.action_mode == MEAN_VS_SEEDED_SAMPLE:
                if self.prime_policy_before_loop:
                    raise ValueError(
                        "mean_vs_seeded_sample requires prime_policy_before_loop=false "
                        "so the first seeded sample is applied rather than discarded"
                    )
                self._action_sample_seeds = per_env_sample_seeds(
                    base_seed=self.action_sample_seed_base,
                    num_envs=self.env.num_envs,
                    deterministic_env_count=self.action_deterministic_env_count,
                )
                self._action_sample_generators = make_per_env_generators(
                    self._action_sample_seeds, device=self.env.device
                )
                print(
                    "[EVAL_ACTION_DIAGNOSTIC] "
                    f"mode={self.action_mode} num_envs={self.env.num_envs} "
                    f"deterministic_env_count={self.action_deterministic_env_count} "
                    f"sample_seed_base={self.action_sample_seed_base} "
                    f"log_steps={self.action_log_steps}",
                    flush=True,
                )

            init_actions = torch.zeros(
                self.env.num_envs, self.env.config.robot.actions_dim, device=self.env.device
            )
            actor_state.update({"obs": obs_dict, "actions": init_actions})
            # Legacy evaluators prime policy state before entering the rollout loop.
            # The opt-out removes only that unused inference; the loop still computes
            # the first action from the same reset observation immediately below.
            if self.prime_policy_before_loop:
                actor_state = self._pre_eval_env_step(actor_state)

            while not actor_state.get("end_eval", False):
                self.env.render_results()
                actor_state["step"] = step
                actor_state = self._pre_eval_env_step(actor_state)
                actor_state = self.env_step(actor_state)
                actor_state = self._post_eval_env_step(actor_state)
                step += 1

                if step % self.empty_cache_freq == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            # Always close recorder writers (trajectory, video) after eval loop.
            # In render_only mode this was already called inside _post_eval_env_step,
            # but close_writers() is idempotent (_closed flag prevents double-save).
            self.env.end_render_results()

            if self.render_only:
                self._save_debug_state_log()
                return {}

            metrics_eval = self._post_evaluate_policy(actor_state)

            if self.eval_only:
                if self.output_dir is not None:
                    self.save_metrics_eval(metrics_eval)
            else:
                metrics_eval["eval/all_metrics_dict"] = create_html_table(
                    metrics_eval["eval/all_metrics_dict"]
                )
                metrics_eval["eval/failed_metrics_dict"] = create_html_table(
                    metrics_eval["eval/failed_metrics_dict"]
                )

        self._train_mode()
        self.model.policy.clear_rollout()
        if not self.eval_only:
            gc.collect()
            torch.cuda.empty_cache()
        if self.eval_frequency == 1:  # Exit if eval frequency is 1.
            self._save_debug_state_log()
            os._exit(0)
        return metrics_eval

    def _post_evaluate_policy(self, eval_res):
        metrics_success = eval_res["metrics_success"]
        metrics_all = eval_res["metrics_all"]
        metrics_eval = {}
        for k, v in metrics_success.items():
            metrics_eval[f"eval/success/{k}"] = v
        for k, v in metrics_all.items():
            metrics_eval[f"eval/all/{k}"] = v

        # Add failed_keys to metrics_eval for wandb logging
        metrics_eval["eval/all_metrics_dict"] = eval_res["all_metrics_dict"]
        metrics_eval["eval/failed_metrics_dict"] = eval_res["failed_metrics_dict"]
        if self.eval_only:
            metrics_eval["failed_keys"] = eval_res["failed_keys"]
            metrics_eval["failed_idxes"] = eval_res["failed_idxes"]

        return metrics_eval

    def _get_inference_policy(self, device=None):
        self.model.policy.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.model.policy.to(device)
        return self.model.policy.act_inference

    def _eval_mode(self):
        if self.eval_only and self.in_eval_mode:
            return
        self.in_eval_mode = True
        self.model.eval()
        if hasattr(self.model.policy, "eval_mode"):
            self.model.policy.eval_mode()  # For VAE, eval mode means that we are no longer sampling from the VAE but using the mean latent value.
        self.env.set_is_evaluating(True, global_rank=self.args.global_rank)

    def _train_mode(self):
        if self.eval_only and self.in_eval_mode:
            return
        self.in_eval_mode = False
        self.model.train()
        if hasattr(self.model.policy, "train_mode"):
            self.model.policy.train_mode()
        self.env.set_is_evaluating(False)
        self.env.set_is_training()

    def _pre_evaluate_policy(self, reset_env=True):
        if reset_env:
            _ = self.env.reset_all()

        asset_compatible_eval = bool(
            getattr(
                getattr(self.env, "motion_command", None),
                "use_asset_compatible_motions",
                False,
            )
        )
        if asset_compatible_eval:
            self.num_total_env_eval_loops = 1
        else:
            self.num_total_env_eval_loops = int(
                np.ceil(
                    self.env._motion_lib._num_unique_motions
                    / (self.env.num_envs * self.args.world_size)
                )
            )
        if "max_render_envs" in self.env.config:
            self.num_total_env_eval_loops = 1
            self.render_only = True

        # Multi-batch eval with per-object dynamic USDs is BROKEN: MultiUsdFileCfg assigns
        # collision meshes statically at scene creation. Batches 1+ have mismatched meshes,
        # causing ~80% object_pos_deviation failures. Use motion sharding instead.
        if self.num_total_env_eval_loops > 1:
            has_per_env_usd = (
                hasattr(self.env, "env")
                and hasattr(self.env.env, "cfg")
                and hasattr(self.env.env.cfg, "scene")
                and not getattr(self.env.env.cfg.scene, "replicate_physics", True)
            )
            is_dynamic = self.env.config.get("object_is_dynamic", False)
            if has_per_env_usd and is_dynamic:
                num_motions = self.env._motion_lib._num_unique_motions
                num_shards = self.num_total_env_eval_loops
                logger.error(
                    f"MULTI-BATCH EVAL WITH DYNAMIC PER-OBJECT USDs IS UNRELIABLE! "
                    f"{num_motions} motions > {self.env.num_envs} envs → {num_shards} batches, "
                    f"but USD collision meshes are STATIC (assigned at scene creation). "
                    f"Batches 1+ will have wrong collision geometry. "
                    f"FIX: Run {num_shards} separate eval jobs with motion_shard_world_size={num_shards} "
                    f"and motion_shard_rank=0..{num_shards - 1}."
                )

        self.env_eval_loop_idx = 0
        self.pbar = tqdm(range(self.num_total_env_eval_loops), desc="Total evaluation progress")
        self.steps_pbar = None
        self.success_rate = 0
        self.curr_steps = 0
        # self.env.start_compute_metrics(global_rank=self.args.global_rank)
        self.terminate_state = torch.zeros(self.env.num_envs, device=self.env.device)
        self.progress_state = torch.zeros(self.env.num_envs, device=self.env.device)
        self.terminate_reason = [""] * self.env.num_envs  # first termination term name per env
        self.terminate_memory = []
        self.progress_memory = []
        self.terminate_reason_memory = []
        self.rp2_diagnostic_label_state = [set() for _ in range(self.env.num_envs)]
        self.rp2_diagnostic_reason_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.gt_rot, self.gt_rot_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.pred_rot, self.pred_rot_all = [], []
        self.sampled_motion_idx = []
        self.time_eval_start = time.time()

        # Object tracking metrics
        _scene_has_object = (
            hasattr(self.env, "env")
            and hasattr(self.env.env, "scene")
            and (
                "object" in self.env.env.scene.rigid_objects
                or "object_0" in self.env.env.scene.rigid_objects
            )
        )
        self._has_object = (
            _scene_has_object
            and hasattr(self.env, "motion_command")
            and self.env.motion_command is not None
        )
        self.obj_pos_error, self.obj_pos_error_all = [], []
        self.obj_ori_error, self.obj_ori_error_all = [], []

    def _save_batch_trajectories(self, env_motion_ids):
        """Save trajectory data for current batch if trajectory recorder is active."""
        try:
            recorder_mgr = self.env.env.recorder_manager
            traj_term = None
            for term in recorder_mgr._terms.values():
                if hasattr(term, "save_batch_and_reset"):
                    traj_term = term
                    break
            if traj_term is None:
                return
            # Cap indices to valid range (last batch may have overflow envs)
            global_idxes = env_motion_ids.cpu().numpy()
            num_keys = len(self.env._motion_lib._motion_data_keys)
            motion_keys = []
            for idx in global_idxes:
                if idx < num_keys:
                    motion_keys.append(str(self.env._motion_lib._motion_data_keys[idx]))
                else:
                    motion_keys.append(None)  # Overflow env — skip
            traj_term.save_batch_and_reset(motion_keys)
        except (AttributeError, KeyError) as e:
            logger.debug(f"Trajectory batch save skipped: {e}")

    def _collect_object_tracking_errors(self):
        """Collect per-step object position and orientation errors (ref vs simulated)."""
        try:
            from gear_sonic.envs.manager_env.mdp.utils import get_target_object_current_state

            motion_cmd = self.env.motion_command
            current_obj_pos, current_obj_quat, _, _, valid = (
                get_target_object_current_state(self.env.env, motion_cmd)
            )
            target_obj_pos = motion_cmd.target_object_root_pos
            target_obj_quat = motion_cmd.target_object_root_quat

            pos_error = torch.norm(target_obj_pos - current_obj_pos, dim=-1)  # (num_envs,)

            # Quaternion error: angle between two quaternions
            from isaaclab.utils.math import quat_error_magnitude

            ori_error = quat_error_magnitude(target_obj_quat, current_obj_quat)  # (num_envs,)

            self.obj_pos_error.append(pos_error[valid].cpu())
            self.obj_ori_error.append(ori_error[valid].cpu())
        except (KeyError, AttributeError, IndexError):
            # Gracefully handle missing object, motion data, or shape mismatches
            pass

    def _collect_rp2_object_debug_state(self, motion_cmd):
        """Collect diagnostic-only per-object state for RP2 eval debug logs."""
        try:
            from gear_sonic.envs.manager_env.mdp import utils as mdp_utils
            from isaaclab.utils.math import quat_error_magnitude

            motion_lib = motion_cmd.motion_lib
            max_objects = int(getattr(motion_lib, "max_num_objects", 1) or 1)
            if max_objects <= 0 or not hasattr(motion_cmd, "object_root_pos"):
                return None

            env = self.env.env
            device = motion_cmd.device
            (
                current_pos,
                current_quat,
                current_lin_vel,
                current_ang_vel,
                valid,
            ) = mdp_utils.get_all_object_current_state(env, motion_cmd, max_objects=max_objects)
            target_pos = motion_cmd.object_root_pos[:, :max_objects]
            target_quat = motion_cmd.object_root_quat[:, :max_objects]
            current_steps = motion_cmd.motion_start_time_steps + motion_cmd.time_steps

            try:
                target_lin_vel = motion_lib.get_object_lin_vel(
                    motion_cmd.motion_ids, current_steps
                )[:, :max_objects]
            except Exception:
                target_lin_vel = torch.zeros_like(current_lin_vel)
            try:
                target_ang_vel = motion_lib.get_object_ang_vel(
                    motion_cmd.motion_ids, current_steps
                )[:, :max_objects]
            except Exception:
                target_ang_vel = torch.zeros_like(current_ang_vel)

            pos_error = torch.norm(target_pos - current_pos, dim=-1)
            height_error = torch.abs(target_pos[..., 2] - current_pos[..., 2])
            ori_error = quat_error_magnitude(
                target_quat.reshape(-1, 4), current_quat.reshape(-1, 4)
            ).reshape(motion_cmd.num_envs, max_objects)
            lin_vel_error = torch.norm(target_lin_vel - current_lin_vel, dim=-1)
            ang_vel_error = torch.norm(target_ang_vel - current_ang_vel, dim=-1)

            def _mask_or_none(mask):
                if mask is None:
                    return None
                mask = mask.to(device)
                if mask.ndim == 1:
                    mask = mask.view(-1, 1)
                else:
                    mask = mask.reshape(mask.shape[0], -1)
                if mask.shape[1] < max_objects:
                    padded = torch.zeros(
                        motion_cmd.num_envs, max_objects, dtype=mask.dtype, device=device
                    )
                    padded[:, : mask.shape[1]] = mask
                    mask = padded
                return mask[:, :max_objects]

            interaction_mask = _mask_or_none(motion_cmd.object_interaction_mask)
            motion_mask = _mask_or_none(motion_cmd.object_motion_mask)
            roles_by_env = motion_cmd._object_role_names_by_env(max_objects)
            target_indices = motion_cmd.target_object_indices().detach().cpu().tolist()
            motion_ids = motion_cmd.motion_ids.detach().cpu().tolist()
            dynamics_by_stem = getattr(env.cfg, "object_dynamics_by_stem", {}) or {}
            right_force = motion_cmd._multi_object_contact_force_by_sensor_key(
                "right_hand_sensor", max_objects
            )
            left_force = motion_cmd._multi_object_contact_force_by_sensor_key(
                "left_hand_sensor", max_objects
            )
            support_force = motion_cmd._multi_object_contact_force_by_sensor_key(
                "support_body_sensor", max_objects
            )

            env_records = []
            for env_idx, motion_id in enumerate(motion_ids):
                motion_key = str(motion_lib.curr_motion_keys[int(motion_id)])
                stems = motion_cmd._object_stems_for_motion_key(
                    motion_key, max_objects=max_objects
                )
                objects = []
                for obj_idx in range(max_objects):
                    stem = str(stems[obj_idx]) if obj_idx < len(stems) else ""
                    role_meta = motion_cmd._object_role_for_stem(stem) if stem else {}
                    interaction_value = (
                        float(interaction_mask[env_idx, obj_idx].detach().cpu())
                        if interaction_mask is not None
                        else None
                    )
                    motion_value = (
                        float(motion_mask[env_idx, obj_idx].detach().cpu())
                        if motion_mask is not None
                        else None
                    )
                    is_valid = bool(valid[env_idx, obj_idx].detach().cpu())
                    active_interaction = bool(
                        is_valid
                        and (interaction_value is None or interaction_value > 0.5)
                    )
                    active_motion = bool(is_valid and (motion_value is None or motion_value > 0.5))

                    def _force_value(force_tensor):
                        if force_tensor is None:
                            return None
                        return float(force_tensor[env_idx, obj_idx].detach().cpu())

                    objects.append(
                        {
                            "index": int(obj_idx),
                            "stem": stem,
                            "role": roles_by_env[env_idx][obj_idx]
                            if obj_idx < len(roles_by_env[env_idx])
                            else "padding",
                            "role_meta": role_meta,
                            "dynamic": bool(
                                dynamics_by_stem.get(stem, role_meta.get("dynamic", False))
                            ),
                            "valid": is_valid,
                            "interaction_mask": interaction_value,
                            "motion_mask": motion_value,
                            "active_interaction": active_interaction,
                            "active_motion": active_motion,
                            "selected_target": int(obj_idx) == int(target_indices[env_idx]),
                            "ref_pos": target_pos[env_idx, obj_idx].detach().cpu().tolist(),
                            "sim_pos": current_pos[env_idx, obj_idx].detach().cpu().tolist(),
                            "ref_quat": target_quat[env_idx, obj_idx].detach().cpu().tolist(),
                            "sim_quat": current_quat[env_idx, obj_idx].detach().cpu().tolist(),
                            "ref_lin_vel": target_lin_vel[env_idx, obj_idx]
                            .detach()
                            .cpu()
                            .tolist(),
                            "sim_lin_vel": current_lin_vel[env_idx, obj_idx]
                            .detach()
                            .cpu()
                            .tolist(),
                            "ref_ang_vel": target_ang_vel[env_idx, obj_idx]
                            .detach()
                            .cpu()
                            .tolist(),
                            "sim_ang_vel": current_ang_vel[env_idx, obj_idx]
                            .detach()
                            .cpu()
                            .tolist(),
                            "pos_error": float(pos_error[env_idx, obj_idx].detach().cpu()),
                            "height_error": float(height_error[env_idx, obj_idx].detach().cpu()),
                            "ori_error": float(ori_error[env_idx, obj_idx].detach().cpu()),
                            "lin_vel_error": float(
                                lin_vel_error[env_idx, obj_idx].detach().cpu()
                            ),
                            "ang_vel_error": float(
                                ang_vel_error[env_idx, obj_idx].detach().cpu()
                            ),
                            "right_contact_force": _force_value(right_force),
                            "left_contact_force": _force_value(left_force),
                            "support_contact_force": _force_value(support_force),
                        }
                    )
                env_records.append(
                    {
                        "env_id": int(env_idx),
                        "motion_id": int(motion_id),
                        "motion_key": motion_key,
                        "target_object_index": int(target_indices[env_idx]),
                        "objects": objects,
                    }
                )
            return {"max_objects": int(max_objects), "envs": env_records}
        except Exception as exc:
            logger.debug(f"RP2 object debug-state skipped: {exc}")
            return None

    def _collect_active_articulation_debug_state(self, motion_cmd):
        """Collect parent, manipulated-link, and joint state for articulated evals."""
        try:
            from gear_sonic.envs.manager_env.mdp import utils as mdp_utils
            from isaaclab.utils.math import quat_error_magnitude

            env = self.env.env
            articulation = mdp_utils.get_active_articulation(env)
            if articulation is None:
                return None

            (
                sim_joint_pos,
                sim_joint_vel,
                ref_joint_pos,
                ref_joint_vel,
                joint_valid,
            ) = mdp_utils.get_active_articulation_joint_tracking_state(env, motion_cmd)
            tracked_body_index = mdp_utils.get_active_articulation_tracked_body_index(
                env, articulation
            )

            sim_root_pos = articulation.data.root_pos_w
            sim_root_quat = articulation.data.root_quat_w
            sim_root_lin_vel = articulation.data.root_lin_vel_w
            sim_root_ang_vel = articulation.data.root_ang_vel_w
            ref_root_pos = motion_cmd.active_articulation_root_pos
            ref_root_quat = motion_cmd.active_articulation_root_quat
            ref_root_lin_vel = motion_cmd.active_articulation_root_lin_vel
            ref_root_ang_vel = motion_cmd.active_articulation_root_ang_vel

            sim_link_pos = articulation.data.body_pos_w[:, tracked_body_index]
            sim_link_quat = articulation.data.body_quat_w[:, tracked_body_index]
            sim_link_lin_vel = articulation.data.body_lin_vel_w[:, tracked_body_index]
            sim_link_ang_vel = articulation.data.body_ang_vel_w[:, tracked_body_index]
            # Active-articulation packs retain the manipulated source part as the
            # sole object channel, so these are the exact tracked-link references.
            ref_link_pos = motion_cmd.target_object_root_pos
            ref_link_quat = motion_cmd.target_object_root_quat
            ref_link_lin_vel = motion_cmd.target_object_lin_vel
            ref_link_ang_vel = motion_cmd.target_object_ang_vel

            root_pos_error = torch.norm(ref_root_pos - sim_root_pos, dim=-1)
            root_ori_error = quat_error_magnitude(ref_root_quat, sim_root_quat)
            root_lin_vel_error = torch.norm(ref_root_lin_vel - sim_root_lin_vel, dim=-1)
            root_ang_vel_error = torch.norm(ref_root_ang_vel - sim_root_ang_vel, dim=-1)
            link_pos_error = torch.norm(ref_link_pos - sim_link_pos, dim=-1)
            link_ori_error = quat_error_magnitude(ref_link_quat, sim_link_quat)
            link_lin_vel_error = torch.norm(ref_link_lin_vel - sim_link_lin_vel, dim=-1)
            link_ang_vel_error = torch.norm(ref_link_ang_vel - sim_link_ang_vel, dim=-1)

            joint_error = torch.where(
                joint_valid,
                sim_joint_pos - ref_joint_pos,
                torch.zeros_like(sim_joint_pos),
            )
            joint_velocity_error = torch.where(
                joint_valid,
                sim_joint_vel - ref_joint_vel,
                torch.zeros_like(sim_joint_vel),
            )
            joint_physics_tensors = {}
            for field in (
                "joint_stiffness",
                "joint_damping",
                "joint_friction_coeff",
                "joint_dynamic_friction_coeff",
                "joint_viscous_friction_coeff",
                "applied_torque",
                "computed_torque",
            ):
                value = getattr(articulation.data, field, None)
                if isinstance(value, torch.Tensor):
                    joint_physics_tensors[field] = value
            panel_net_contact_force = None
            panel_filtered_hand_force = None
            sensors = getattr(env.scene, "sensors", {})
            panel_sensor = sensors.get("object_to_hand_contact_sensor")
            panel_sensor_data = getattr(panel_sensor, "data", None)
            if panel_sensor_data is not None:
                net_force = getattr(panel_sensor_data, "net_forces_w", None)
                if isinstance(net_force, torch.Tensor):
                    panel_net_contact_force = net_force.reshape(
                        net_force.shape[0], -1, 3
                    ).sum(dim=1)
                filtered_force = getattr(panel_sensor_data, "force_matrix_w", None)
                if isinstance(filtered_force, torch.Tensor):
                    panel_filtered_hand_force = filtered_force.reshape(
                        filtered_force.shape[0], -1, 3
                    ).sum(dim=1)
            root_dynamic = motion_cmd.active_articulation_root_dynamic_mask
            config = getattr(env.cfg, "config", {}) or {}
            motion_ids = motion_cmd.motion_ids.detach().cpu().tolist()

            env_records = []
            for env_idx, motion_id in enumerate(motion_ids):
                motion_key = str(motion_cmd.motion_lib.curr_motion_keys[int(motion_id)])
                record = {
                        "env_id": int(env_idx),
                        "motion_id": int(motion_id),
                        "motion_key": motion_key,
                        "root_dynamic": bool(root_dynamic[env_idx].detach().cpu()),
                        "sim_root_pos": sim_root_pos[env_idx].detach().cpu().tolist(),
                        "ref_root_pos": ref_root_pos[env_idx].detach().cpu().tolist(),
                        "sim_root_quat": sim_root_quat[env_idx].detach().cpu().tolist(),
                        "ref_root_quat": ref_root_quat[env_idx].detach().cpu().tolist(),
                        "sim_root_lin_vel": sim_root_lin_vel[env_idx].detach().cpu().tolist(),
                        "ref_root_lin_vel": ref_root_lin_vel[env_idx].detach().cpu().tolist(),
                        "sim_root_ang_vel": sim_root_ang_vel[env_idx].detach().cpu().tolist(),
                        "ref_root_ang_vel": ref_root_ang_vel[env_idx].detach().cpu().tolist(),
                        "root_pos_error": float(root_pos_error[env_idx].detach().cpu()),
                        "root_ori_error": float(root_ori_error[env_idx].detach().cpu()),
                        "root_lin_vel_error": float(
                            root_lin_vel_error[env_idx].detach().cpu()
                        ),
                        "root_ang_vel_error": float(
                            root_ang_vel_error[env_idx].detach().cpu()
                        ),
                        "sim_tracked_link_pos": sim_link_pos[env_idx].detach().cpu().tolist(),
                        "ref_tracked_link_pos": ref_link_pos[env_idx].detach().cpu().tolist(),
                        "sim_tracked_link_quat": sim_link_quat[env_idx].detach().cpu().tolist(),
                        "ref_tracked_link_quat": ref_link_quat[env_idx].detach().cpu().tolist(),
                        "sim_tracked_link_lin_vel": sim_link_lin_vel[env_idx]
                        .detach()
                        .cpu()
                        .tolist(),
                        "ref_tracked_link_lin_vel": ref_link_lin_vel[env_idx]
                        .detach()
                        .cpu()
                        .tolist(),
                        "sim_tracked_link_ang_vel": sim_link_ang_vel[env_idx]
                        .detach()
                        .cpu()
                        .tolist(),
                        "ref_tracked_link_ang_vel": ref_link_ang_vel[env_idx]
                        .detach()
                        .cpu()
                        .tolist(),
                        "tracked_link_pos_error": float(
                            link_pos_error[env_idx].detach().cpu()
                        ),
                        "tracked_link_ori_error": float(
                            link_ori_error[env_idx].detach().cpu()
                        ),
                        "tracked_link_lin_vel_error": float(
                            link_lin_vel_error[env_idx].detach().cpu()
                        ),
                        "tracked_link_ang_vel_error": float(
                            link_ang_vel_error[env_idx].detach().cpu()
                        ),
                        "sim_joint_pos": sim_joint_pos[env_idx].detach().cpu().tolist(),
                        "ref_joint_pos": ref_joint_pos[env_idx].detach().cpu().tolist(),
                        "sim_joint_vel": sim_joint_vel[env_idx].detach().cpu().tolist(),
                        "ref_joint_vel": ref_joint_vel[env_idx].detach().cpu().tolist(),
                        "joint_valid": joint_valid[env_idx].detach().cpu().tolist(),
                        "joint_position_error": joint_error[env_idx].detach().cpu().tolist(),
                        "joint_velocity_error": joint_velocity_error[env_idx]
                        .detach()
                        .cpu()
                        .tolist(),
                    }
                for field, value in joint_physics_tensors.items():
                    record[field] = value[env_idx].detach().cpu().tolist()
                if panel_net_contact_force is not None:
                    record["panel_net_contact_force"] = (
                        panel_net_contact_force[env_idx].detach().cpu().tolist()
                    )
                if panel_filtered_hand_force is not None:
                    record["panel_filtered_right_hand_force"] = (
                        panel_filtered_hand_force[env_idx].detach().cpu().tolist()
                    )
                env_records.append(record)

            return {
                "tracked_body_name": str(
                    config.get("active_articulation_tracked_body_name", "panel")
                ),
                "tracked_body_index": int(tracked_body_index),
                "body_names": list(articulation.body_names),
                "joint_names": list(articulation.joint_names),
                "envs": env_records,
            }
        except Exception as exc:
            logger.warning(f"Active-articulation debug-state skipped: {exc}")
            return None

    def _collect_termination_debug_state(self, dones):
        """Return per-term masks and reasons for resets produced by this step."""
        try:
            manager = self.env.env.termination_manager
            term_names = list(getattr(manager, "_term_names", []))
            last_dones = getattr(manager, "_last_episode_dones", None)
            if not isinstance(last_dones, torch.Tensor) or last_dones.ndim != 2:
                return None

            done_mask = dones.detach().bool().reshape(-1)
            term_masks = {}
            reasons = [[] for _ in range(done_mask.shape[0])]
            for term_idx, term_name in enumerate(term_names):
                if term_idx >= last_dones.shape[1]:
                    break
                values = last_dones[:, term_idx].detach().bool() & done_mask
                term_masks[str(term_name)] = values.cpu().tolist()
                for env_idx in values.nonzero(as_tuple=False).reshape(-1).tolist():
                    reasons[env_idx].append(str(term_name))
            return {
                "term_masks": term_masks,
                "reasons_by_env": reasons,
            }
        except (AttributeError, IndexError, RuntimeError):
            return None

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})

        motion_cmd = None
        rp2_diagnostic_labels = []
        try:
            motion_cmd = self.env.env.command_manager.get_term("motion")
            rp2_diagnostic_labels = self._rp2_support_diagnostic_labels(motion_cmd)
            if rp2_diagnostic_labels and hasattr(self, "rp2_diagnostic_label_state"):
                for env_id, labels in enumerate(rp2_diagnostic_labels):
                    if labels:
                        self.rp2_diagnostic_label_state[env_id].update(labels.split("|"))
        except Exception:
            motion_cmd = None
            rp2_diagnostic_labels = []

        # Debug state logging
        if self._debug_state_records is not None and motion_cmd is not None:
            rp2_metrics = {}
            active_articulation_metrics = {}
            for key, value in motion_cmd.metrics.items():
                output = None
                if key.startswith("rp2_"):
                    output = rp2_metrics
                elif key.startswith("active_articulation/"):
                    output = active_articulation_metrics
                if output is None:
                    continue
                if isinstance(value, torch.Tensor):
                    output[key] = value.detach().cpu().tolist()
                else:
                    output[key] = value

            # Evaluation contact metrics must not depend on whether a grasp
            # reward term is enabled in the ablation under test.
            try:
                from gear_sonic.envs.manager_env.mdp import utils as mdp_utils

                for side, hand, sensor_name in (
                    ("left", "left_hand", "object_to_left_hand_contact_sensor"),
                    ("right", "right_hand", "object_to_hand_contact_sensor"),
                ):
                    force = mdp_utils.get_target_object_contact_forces(
                        self.env.env,
                        motion_cmd,
                        hand=hand,
                        fallback_sensor_name=sensor_name,
                    )
                    if force is None:
                        continue
                    force_magnitude = torch.norm(force, dim=-1)
                    rp2_metrics[f"rp2_eval_sim_contact_count/{side}"] = (
                        (force_magnitude > 0.1)
                        .float()
                        .sum(dim=-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                broad_sensor = self.env.env.scene.sensors.get(
                    "active_articulation_to_broad_hand_contact_sensor"
                )
                broad_force = getattr(
                    getattr(broad_sensor, "data", None), "force_matrix_w", None
                )
                if isinstance(broad_force, torch.Tensor):
                    broad_magnitude = torch.norm(broad_force, dim=-1).reshape(
                        broad_force.shape[0], -1
                    )
                    rp2_metrics["rp2_eval_sim_contact_count/broad_hand"] = (
                        (broad_magnitude > 0.1)
                        .float()
                        .sum(dim=-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
            except (AttributeError, KeyError, IndexError, RuntimeError) as exc:
                logger.debug(f"Independent RP2 eval contact logging skipped: {exc}")
            motion_frames = motion_cmd.motion_start_time_steps + motion_cmd.time_steps

            def _reference_contact(hand):
                try:
                    labels = motion_cmd.motion_lib.get_object_in_contact(
                        motion_cmd.motion_ids, motion_frames, hand=hand
                    )
                except (AttributeError, IndexError):
                    return None
                if labels is None:
                    return None
                return labels.detach().cpu().tolist()

            motion_keys = []
            try:
                motion_keys = [
                    str(motion_cmd.motion_lib.curr_motion_keys[int(motion_id)])
                    for motion_id in motion_cmd.motion_ids.detach().cpu().tolist()
                ]
            except Exception:
                motion_keys = []
            reward_components = rewards[0].detach().reshape(-1).cpu()
            ee_termination_debug = getattr(
                motion_cmd, "_rp2_last_ee_body_termination_debug", None
            )
            if isinstance(ee_termination_debug, dict):
                ee_termination_debug = {
                    key: (
                        value.detach().cpu().tolist()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in ee_termination_debug.items()
                }
            rec = {
                "step": actor_state.get("step", len(self._debug_state_records)),
                "motion_keys": motion_keys,
                "motion_frames": motion_frames.detach().cpu().tolist(),
                "reference_contact_left": _reference_contact("left_hand"),
                "reference_contact_right": _reference_contact("right_hand"),
                "root_pos": motion_cmd.robot_body_pos_w[0, 0].cpu().tolist(),
                "root_quat": motion_cmd.robot_body_quat_w[0, 0].cpu().tolist(),
                "ref_root_pos": motion_cmd.body_pos_w[0, 0].cpu().tolist(),
                "ref_root_quat": motion_cmd.body_quat_w[0, 0].cpu().tolist(),
                "body_pos": motion_cmd.robot_body_pos_w[0].detach().cpu().tolist(),
                "ref_body_pos": motion_cmd.body_pos_w[0].detach().cpu().tolist(),
                "body_quat": motion_cmd.robot_body_quat_w[0].detach().cpu().tolist(),
                "ref_body_quat": motion_cmd.body_quat_w[0].detach().cpu().tolist(),
                "joint_pos": motion_cmd.robot_joint_pos[0, :29].cpu().tolist(),
                "ref_joint_pos": motion_cmd.joint_pos[0].cpu().tolist(),
                "actions": actor_state["actions"][0].cpu().tolist(),
                # Keep the legacy env-0 fields above for existing analysis tools,
                # while recording every environment for stratified failure audits.
                "root_pos_by_env": motion_cmd.robot_body_pos_w[:, 0]
                .detach()
                .cpu()
                .tolist(),
                "root_quat_by_env": motion_cmd.robot_body_quat_w[:, 0]
                .detach()
                .cpu()
                .tolist(),
                "ref_root_pos_by_env": motion_cmd.body_pos_w[:, 0]
                .detach()
                .cpu()
                .tolist(),
                "ref_root_quat_by_env": motion_cmd.body_quat_w[:, 0]
                .detach()
                .cpu()
                .tolist(),
                "body_pos_by_env": motion_cmd.robot_body_pos_w.detach().cpu().tolist(),
                "ref_body_pos_by_env": motion_cmd.body_pos_w.detach().cpu().tolist(),
                "joint_pos_by_env": motion_cmd.robot_joint_pos[:, :29]
                .detach()
                .cpu()
                .tolist(),
                "ref_joint_pos_by_env": motion_cmd.joint_pos.detach().cpu().tolist(),
                "actions_by_env": actor_state["actions"].detach().cpu().tolist(),
                "reward": float(reward_components.mean().item()),
                "reward_components": reward_components.tolist(),
                "done": dones.detach().cpu().tolist() if dones.numel() > 0 else [],
                "termination_debug": self._collect_termination_debug_state(dones),
                "time_outs": (
                    extras.get("time_outs").detach().cpu().tolist()
                    if isinstance(extras.get("time_outs"), torch.Tensor)
                    else extras.get("time_outs")
                ),
                "ee_body_termination_debug": ee_termination_debug,
                "rp2_metrics": rp2_metrics,
                "active_articulation_metrics": active_articulation_metrics,
                "rp2_object_debug": self._collect_rp2_object_debug_state(motion_cmd),
                "active_articulation_debug": self._collect_active_articulation_debug_state(
                    motion_cmd
                ),
                "rp2_diagnostic_labels": rp2_diagnostic_labels,
            }
            policy_action_diagnostic = actor_state.get("policy_action_diagnostic")
            if policy_action_diagnostic is not None:
                rec["policy_action_diagnostic"] = policy_action_diagnostic
            self._debug_state_records.append(rec)
        return actor_state

    def _pre_eval_env_step(self, actor_state: dict):
        dones = actor_state.get("dones", torch.zeros(self.env.num_envs, device=self.env.device))
        actor_state.pop("policy_action_diagnostic", None)
        if self.action_mode == DETERMINISTIC_MEAN:
            actions = self.eval_policy(
                obs_dict=actor_state["obs"], cur_dones=dones, skip_episode_attnmask=True
            )
        elif self.action_mode == MEAN_VS_SEEDED_SAMPLE:
            if self._action_sample_generators is None or self._action_sample_seeds is None:
                raise RuntimeError("seeded evaluator action generators were not initialized")
            policy = self.model.policy
            if not hasattr(policy, "_update_obs_buffer") or not hasattr(
                policy, "update_distribution"
            ):
                raise RuntimeError(
                    "mean_vs_seeded_sample requires a rollout policy exposing "
                    "_update_obs_buffer and update_distribution"
                )
            episode_attnmask = policy._update_obs_buffer(
                actor_state["obs"], None, dones
            )
            policy.update_distribution(
                obs_dict=policy.obs_dict_buffer,
                episode_attnmask=episode_attnmask,
                last_step_only=True,
                skip_episode_attnmask=True,
            )
            policy.steps += 1
            action_mean = policy.action_mean.detach()
            action_std = policy.action_std.detach()
            actions = mean_vs_seeded_sample(
                action_mean,
                action_std,
                deterministic_env_count=self.action_deterministic_env_count,
                generators=self._action_sample_generators,
            )
            action_step = int(actor_state.get("step", 0))
            if action_step < self.action_log_steps:
                expanded_std = torch.broadcast_to(action_std, action_mean.shape)
                actor_state["policy_action_diagnostic"] = {
                    "mode": self.action_mode,
                    "action_step": action_step,
                    "deterministic_env_count": self.action_deterministic_env_count,
                    "sample_seed_base": self.action_sample_seed_base,
                    "envs": [
                        {
                            "env_id": env_id,
                            "action_mode": (
                                DETERMINISTIC_MEAN
                                if env_id < self.action_deterministic_env_count
                                else "fixed_seed_sample"
                            ),
                            "sample_seed": self._action_sample_seeds[env_id],
                            "sample_index": (
                                None
                                if env_id < self.action_deterministic_env_count
                                else action_step
                            ),
                            "action_mean": action_mean[env_id].cpu().tolist(),
                            "action_std": expanded_std[env_id].cpu().tolist(),
                            "action": actions[env_id].detach().cpu().tolist(),
                            "sample_delta_l2": float(
                                torch.linalg.vector_norm(
                                    actions[env_id] - action_mean[env_id]
                                ).item()
                            ),
                        }
                        for env_id in range(self.env.num_envs)
                    ],
                }
        else:
            raise RuntimeError(f"unhandled eval action mode {self.action_mode!r}")

        if self._zero_latent_prefix_enabled:
            motion_cmd = self.env.motion_command
            if motion_cmd is None or motion_cmd.motion_lib is None:
                raise RuntimeError(
                    "zero-latent prefix requires a motion command and motion library"
                )
            reference_steps = motion_cmd.motion_start_time_steps + motion_cmd.time_steps
            original_actions = actions.detach().clone()
            actions, intervention_mask, target_motion_id = apply_zero_latent_prefix(
                actions,
                motion_ids=motion_cmd.motion_ids,
                reference_steps=reference_steps,
                motion_keys=list(motion_cmd.motion_lib.curr_motion_keys),
                target_motion_key=self.zero_latent_prefix_motion_key,
                prefix_steps=self.zero_latent_prefix_steps,
                latent_dim=self.zero_latent_prefix_dim,
                expected_hand_dim=self.zero_latent_prefix_expected_hand_dim,
            )
            applied_env_ids = intervention_mask.nonzero(as_tuple=False).reshape(-1).tolist()
            if applied_env_ids:
                latent_dim = self.zero_latent_prefix_dim
                actor_state["policy_action_diagnostic"] = {
                    "mode": "deterministic_mean_zero_latent_prefix",
                    "action_step": int(actor_state.get("step", 0)),
                    "target_motion_key": self.zero_latent_prefix_motion_key,
                    "target_motion_id": int(target_motion_id),
                    "prefix_absolute_reference_steps": [
                        0,
                        self.zero_latent_prefix_steps,
                    ],
                    "latent_dim": latent_dim,
                    "untouched_hand_dim": self.zero_latent_prefix_expected_hand_dim,
                    "target": "zero_residual_latent",
                    "object_or_contact_reference_modified": False,
                    "envs": [
                        {
                            "env_id": int(env_id),
                            "absolute_reference_step": int(reference_steps[env_id].item()),
                            "original_action_mean": original_actions[env_id].cpu().tolist(),
                            "applied_action": actions[env_id].detach().cpu().tolist(),
                            "original_latent_l2": float(
                                torch.linalg.vector_norm(
                                    original_actions[env_id, :latent_dim]
                                ).item()
                            ),
                            "applied_latent_l2": float(
                                torch.linalg.vector_norm(actions[env_id, :latent_dim]).item()
                            ),
                            "hand_outputs_unchanged": bool(
                                torch.equal(
                                    original_actions[env_id, latent_dim:],
                                    actions[env_id, latent_dim:],
                                )
                            ),
                        }
                        for env_id in applied_env_ids
                    ],
                }
        actor_state.update({"actions": actions})
        return actor_state

    def _post_eval_env_step(self, actor_state):
        step = actor_state["step"]
        actor_state["end_eval"] = False

        if "ref_body_pos_extend" in self.env.extras:
            self.gt_pos.append(self.env.extras["ref_body_pos_extend"].cpu().numpy())
            self.pred_pos.append(self.env.extras["rigid_body_pos_extend"].cpu().numpy())
            self.mpjpe.append(self.env.dif_global_body_pos.norm(dim=-1).cpu() * 1000)
        else:
            gt_pos = self.env.get_env_data("ref_body_pos_extend")
            pred_pos = self.env.get_env_data("rigid_body_pos_extend")
            mpjpe = (gt_pos - pred_pos).norm(dim=-1) * 1000
            self.gt_pos.append(gt_pos.cpu().numpy())
            self.pred_pos.append(pred_pos.cpu().numpy())
            self.mpjpe.append(mpjpe.cpu())

        # Collect object tracking errors if object exists in scene
        if self._has_object:
            self._collect_object_tracking_errors()

        # self.gt_rot.append(self.env.extras['ref_body_rot_extend'].cpu().numpy())
        # self.pred_rot.append(self.env._rigid_body_rot_extend.cpu().numpy())

        died = actor_state["dones"]
        died[actor_state["extras"]["time_outs"]] = False

        termination_state = torch.logical_and(
            self.curr_steps <= self.env._motion_lib.get_motion_num_steps(self.env.motion_ids) - 1,
            died,
        )  # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation.

        # Record which termination term caused the first termination for each env
        # Use _last_episode_dones which persists after env reset (shape: num_envs x num_terms)
        newly_terminated = termination_state & (~self.terminate_state.bool())
        if newly_terminated.any():
            new_env_ids = newly_terminated.nonzero(as_tuple=False).view(-1).tolist()
            try:
                tm = self.env.env.termination_manager
                last_dones = tm._last_episode_dones  # (num_envs, num_terms) - survives reset
                for env_id in new_env_ids:
                    for term_idx, term_name in enumerate(tm._term_names):
                        if term_name == "time_out":
                            continue
                        if last_dones[env_id, term_idx] and self.terminate_reason[env_id] == "":
                            self.terminate_reason[env_id] = term_name
            except Exception:
                pass
            # Fallback: label any still-unlabeled as "unknown"
            for env_id in new_env_ids:
                if self.terminate_reason[env_id] == "":
                    self.terminate_reason[env_id] = "unknown"

        self.terminate_state = torch.logical_or(termination_state, self.terminate_state)

        self.progress_state[~self.terminate_state] += 1

        if (~self.terminate_state).sum() > 0:
            max_possible_id = self.env._motion_lib._num_unique_motions - 1
            curr_ids = self.env._motion_lib._curr_motion_ids
            if (max_possible_id == curr_ids).sum() > 0:  # When you are running out of motions.
                bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                if (~self.terminate_state[:bound]).sum() > 0:
                    curr_max = (
                        self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)[:bound][
                            ~self.terminate_state[:bound]
                        ]
                        .max()
                        .item()
                    )
                else:
                    curr_max = self.curr_steps - 1  # the ones that should be counted have teimrated
            else:
                curr_max = (
                    self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)[
                        ~self.terminate_state
                    ]
                    .max()
                    .item()
                )

            if self.curr_steps >= curr_max:
                curr_max = self.curr_steps + 1  # For matching up the current steps and max steps.
        else:
            curr_max = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids).max().item()

        if self.steps_pbar is None and (~self.terminate_state).sum() > 0:
            self.steps_pbar = tqdm(total=int(curr_max), desc="Sequence progress", leave=False)

        if self.steps_pbar is not None:
            self.steps_pbar.update(1)
            if self.steps_pbar.total != int(curr_max):
                self.steps_pbar.total = int(curr_max)
                self.steps_pbar.refresh()

        self.curr_steps += 1
        max_render_steps = 0
        if self.render_only:
            try:
                max_render_steps = int(self.env.config.get("max_render_steps", 0))
            except Exception:
                max_render_steps = 0
        reached_render_cap = max_render_steps > 0 and self.curr_steps >= max_render_steps
        if reached_render_cap:
            logger.info(
                f"Reached render-only max_render_steps={max_render_steps}; closing partial eval videos."
            )

        if (
            self.curr_steps >= curr_max
            or self.terminate_state.sum() == self.env.num_envs
            or reached_render_cap
        ):
            if self.steps_pbar is not None:
                self.steps_pbar.close()
                self.steps_pbar = None

            self.terminate_memory.append(self.terminate_state.cpu().numpy())
            self.terminate_reason_memory.extend(self.terminate_reason)
            diag_reasons = [
                "|".join(sorted(label_set)) for label_set in self.rp2_diagnostic_label_state
            ]
            self.rp2_diagnostic_reason_memory.extend(diag_reasons)
            self.rp2_diagnostic_label_state = [set() for _ in range(self.env.num_envs)]
            self.progress_memory.append(
                (
                    self.progress_state
                    / self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
                )
                .cpu()
                .numpy()
            )

            self.success_rate = (
                1
                - np.concatenate(self.terminate_memory)[
                    : self.env._motion_lib._num_unique_motions
                ].mean()
            )
            self.progress_rate = np.concatenate(self.progress_memory)[
                : self.env._motion_lib._num_unique_motions
            ].mean()

            # MPJPE
            all_mpjpe = torch.stack(self.mpjpe)
            try:
                assert (
                    all_mpjpe.shape[0] == curr_max
                    or self.terminate_state.sum() == self.env.num_envs
                )  # Max should be the same as the number of frames in the motion.
            except AssertionError:
                print(
                    f"Warning: MPJPE shape mismatch: {all_mpjpe.shape[0]} vs curr_max={curr_max}, terminated={self.terminate_state.sum()}/{self.env.num_envs}"
                )

            all_body_pos_pred = np.stack(self.pred_pos)
            all_body_pos_gt = np.stack(self.gt_pos)
            # all_body_rot_pred = np.stack(self.pred_rot)
            # all_body_rot_gt = np.stack(self.gt_rot)

            all_mpjpe = [
                all_mpjpe[: (i - 1), idx].mean()
                for idx, i in enumerate(
                    self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
                )
            ]  # -1 since we do not count the first frame.
            all_body_pos_pred = [
                all_body_pos_pred[: (i - 1), idx]
                for idx, i in enumerate(
                    self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
                )
            ]
            all_body_pos_gt = [
                all_body_pos_gt[: (i - 1), idx]
                for idx, i in enumerate(
                    self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
                )
            ]
            # all_body_rot_pred = [all_body_rot_pred[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]
            # all_body_rot_gt = [all_body_rot_gt[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]

            self.mpjpe_all.append(all_mpjpe)
            self.pred_pos_all += all_body_pos_pred
            self.gt_pos_all += all_body_pos_gt
            # self.pred_rot_all += all_body_rot_pred
            # self.gt_rot_all += all_body_rot_gt

            # Aggregate object tracking errors for this batch
            if self._has_object and len(self.obj_pos_error) > 0:
                all_obj_pos_err = torch.stack(self.obj_pos_error)  # (T, num_envs)
                all_obj_ori_err = torch.stack(self.obj_ori_error)  # (T, num_envs)
                motion_num_steps = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
                per_env_obj_pos_err = [
                    all_obj_pos_err[: (i - 1), idx].mean().item()
                    for idx, i in enumerate(motion_num_steps)
                ]
                per_env_obj_ori_err = [
                    all_obj_ori_err[: (i - 1), idx].mean().item()
                    for idx, i in enumerate(motion_num_steps)
                ]
                self.obj_pos_error_all.append(per_env_obj_pos_err)
                self.obj_ori_error_all.append(per_env_obj_ori_err)

            env_motion_ids = self.env.start_idx + self.env.motion_ids
            self.sampled_motion_idx.append(env_motion_ids)

            # Save per-motion trajectory data if trajectory recorder is active
            self._save_batch_trajectories(env_motion_ids)

            self.env_eval_loop_idx += 1

            if self.env_eval_loop_idx >= self.num_total_env_eval_loops:
                if self.render_only:
                    # Print per-motion termination summary even in render-only mode
                    terminate_hist = np.concatenate(self.terminate_memory)
                    progress_hist = np.concatenate(self.progress_memory)
                    num_unique = self.env._motion_lib._num_unique_motions
                    motion_keys = self.env._motion_lib._motion_data_keys
                    mpjpe_flat = [
                        m.item() if hasattr(m, "item") else m
                        for batch in self.mpjpe_all
                        for m in batch
                    ]
                    reasons = self.terminate_reason_memory
                    diag_reasons = self.rp2_diagnostic_reason_memory

                    print("Rendering only. Reached the end of the evaluation loop.", flush=True)
                    success_rate = 1 - terminate_hist[:num_unique].mean()
                    progress_rate = progress_hist[:num_unique].mean()
                    print(f"Success Rate: {success_rate:.10f}", flush=True)
                    print(f"Progress Rate: {progress_rate:.10f}", flush=True)
                    for idx in range(min(num_unique, len(terminate_hist))):
                        key = motion_keys[idx] if idx < len(motion_keys) else f"motion_{idx}"
                        t = bool(terminate_hist[idx])
                        p = float(progress_hist[idx])
                        mpjpe_str = f"{mpjpe_flat[idx]:.1f}" if idx < len(mpjpe_flat) else "N/A"
                        reason = reasons[idx] if idx < len(reasons) else ""
                        diag = diag_reasons[idx] if idx < len(diag_reasons) else ""
                        status = "FAIL" if t else "OK"
                        print(
                            f"MOTION_RESULT: {key} status={status} progress={p:.3f} mpjpe={mpjpe_str} reason={reason} diag={diag}",
                            flush=True,
                        )

                    self.env.end_render_results()
                    actor_state["end_eval"] = True
                    return actor_state

                terminate_hist = np.concatenate(self.terminate_memory)
                progress_hist = np.concatenate(self.progress_memory)
                succ_idxes = np.nonzero(
                    ~terminate_hist[: self.env._motion_lib._num_unique_motions]
                )[0].tolist()
                self.accelerator.wait_for_everyone()
                # metrics_all = compute_metrics_lite(self.pred_pos_all, self.gt_pos_all, self.pred_rot_all, self.gt_rot_all, concatenate = False) # OOM

                print(
                    f"!!!!!!! {len(self.pred_pos_all)} {len(self.gt_pos_all)} {self.env.start_idx} {self.args.global_rank} Time: {datetime.now().strftime('%H:%M:%S')}"
                )

                if hasattr(self.env, "motion_command"):
                    body_names = self.env.motion_command.cmd_body_names
                else:
                    print("No self.env.motion_command.cmd_body_names found!!!!")
                    exit()

                """
                # gear_sonic/config/manager_env/commands/terms/motion.yaml
                body_names: [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ]
                """

                # Define subsets
                # 6 + 3 + 5 = 14
                legs_subset_names = [
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                ]
                # NOTE use torso_link instead of head for vr_3points_subset_names
                vr_3points_subset_names = [
                    "torso_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ]
                other_upper_bodies_subset_names = [
                    "pelvis",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                ]

                foot_subset_names = ["left_ankle_roll_link", "right_ankle_roll_link"]

                # Get indices for subsets
                legs_indices = [body_names.index(name) for name in legs_subset_names]
                vr_3points_indices = [body_names.index(name) for name in vr_3points_subset_names]
                other_upper_bodies_indices = [
                    body_names.index(name) for name in other_upper_bodies_subset_names
                ]
                foot_indices = [body_names.index(name) for name in foot_subset_names]
                # Extract subset data
                pred_pos_legs = [p[:, legs_indices, :] for p in self.pred_pos_all]
                gt_pos_legs = [g[:, legs_indices, :] for g in self.gt_pos_all]

                pred_pos_foot = [p[:, foot_indices, :] for p in self.pred_pos_all]
                gt_pos_foot = [g[:, foot_indices, :] for g in self.gt_pos_all]

                pred_pos_vr_3points = [p[:, vr_3points_indices, :] for p in self.pred_pos_all]
                gt_pos_vr_3points = [g[:, vr_3points_indices, :] for g in self.gt_pos_all]

                pred_pos_other_upper_bodies = [
                    p[:, other_upper_bodies_indices, :] for p in self.pred_pos_all
                ]
                gt_pos_other_upper_bodies = [
                    g[:, other_upper_bodies_indices, :] for g in self.gt_pos_all
                ]

                # Lazy import to avoid cffi version conflict with IsaacSim
                from smpl_sim.smpllib.smpl_eval import compute_metrics_lite

                metrics_all = compute_metrics_lite(
                    self.pred_pos_all, self.gt_pos_all, concatenate=False
                )  # list of length N_env
                metrics_legs = compute_metrics_lite(pred_pos_legs, gt_pos_legs, concatenate=False)
                metrics_vr_3points = compute_metrics_lite(
                    pred_pos_vr_3points, gt_pos_vr_3points, concatenate=False
                )
                metrics_other_upper_bodies = compute_metrics_lite(
                    pred_pos_other_upper_bodies, gt_pos_other_upper_bodies, concatenate=False
                )
                metrics_foot = compute_metrics_lite(pred_pos_foot, gt_pos_foot, concatenate=False)

                # Rename keys for subset metrics
                metrics_legs = {f"{k}_legs": v for k, v in metrics_legs.items()}
                metrics_vr_3points = {f"{k}_vr_3points": v for k, v in metrics_vr_3points.items()}
                metrics_other_upper_bodies = {
                    f"{k}_other_upper_bodies": v for k, v in metrics_other_upper_bodies.items()
                }
                metrics_foot = {f"{k}_foot": v for k, v in metrics_foot.items()}

                metrics_all.update(metrics_legs)
                metrics_all.update(metrics_vr_3points)
                metrics_all.update(metrics_other_upper_bodies)
                metrics_all.update(metrics_foot)

                metrics_all_sum = {
                    k: torch.tensor(
                        [np.sum(i) / i.shape[1] if "mpjpe" in k else np.sum(i) for i in v]
                    ).to(self.env.device)
                    for k, v in metrics_all.items()
                }  # of length N_env -- mean over joint but sum over length
                length_all = torch.tensor([len(i) for i in self.pred_pos_all]).to(self.env.device)

                metrics_all_contactnate = torch.stack(
                    [v for k, v in metrics_all_sum.items()] + [length_all], dim=-1
                )
                terminate_hist_concatenate = torch.tensor(terminate_hist).to(self.env.device)
                progress_hist_concatenate = torch.tensor(progress_hist).to(self.env.device)
                all_motion_idxes = torch.cat(self.sampled_motion_idx).to(self.env.device)

                # Prepare object tracking metrics for gathering
                has_obj_metrics = self._has_object and len(self.obj_pos_error_all) > 0
                if has_obj_metrics:
                    obj_pos_err_flat = torch.tensor(
                        [v for batch in self.obj_pos_error_all for v in batch]
                    ).to(self.env.device)
                    obj_ori_err_flat = torch.tensor(
                        [v for batch in self.obj_ori_error_all for v in batch]
                    ).to(self.env.device)

                # Tensor layout: [metrics_all_sum..., length, terminate, progress, (obj_pos_err, obj_ori_err,) motion_idx]
                tail_tensors = [
                    terminate_hist_concatenate[:, None],
                    progress_hist_concatenate[:, None],
                ]
                if has_obj_metrics:
                    tail_tensors.append(obj_pos_err_flat[:, None])
                    tail_tensors.append(obj_ori_err_flat[:, None])
                tail_tensors.append(all_motion_idxes[:, None])

                all_tensors = torch.cat(
                    [metrics_all_contactnate] + tail_tensors,
                    dim=-1,
                )
                print("Gathering eval tensors", all_tensors.shape, self.accelerator.process_index)

                chunk_size = 1024  # Chunk gathering since it's 4096 is too large.
                chunks = all_tensors.split(chunk_size)
                gathered_chunks = [
                    self.accelerator.gather(chunk).reshape(-1, *chunk.shape) for chunk in chunks
                ]  # each with shape (num_processes x 1024 (chunked_num_env), D_metrics)
                all_metrics = torch.cat(gathered_chunks, dim=1)

                metric_size = all_metrics.shape[-1]
                gathered_metrics_stack = (
                    all_metrics.reshape(
                        self.accelerator.num_processes, -1, self.env.num_envs, metric_size
                    )
                    .transpose(0, 1)
                    .reshape(-1, metric_size)[: self.env._motion_lib._num_unique_motions]
                )  # make sure that we are selecting the correct ones.

                # Extract tail columns: terminate, progress, (obj_pos_err, obj_ori_err,) motion_idx
                num_tail = 3 + (
                    2 if has_obj_metrics else 0
                )  # terminate + progress + (obj*2) + motion_idx
                num_body_metrics = metric_size - num_tail  # metrics_all_sum columns + length

                gathered_terminate_hist_stack = gathered_metrics_stack[:, num_body_metrics].bool()
                gathered_progress_hist_stack = gathered_metrics_stack[:, num_body_metrics + 1]
                if has_obj_metrics:
                    gathered_obj_pos_err = gathered_metrics_stack[:, num_body_metrics + 2]
                    gathered_obj_ori_err = gathered_metrics_stack[:, num_body_metrics + 3]
                    gathered_motion_idxes = gathered_metrics_stack[:, num_body_metrics + 4].long()
                else:
                    gathered_motion_idxes = gathered_metrics_stack[:, num_body_metrics + 2].long()
                gathered_progress_hist_stack[~gathered_terminate_hist_stack] = 1

                assert (gathered_motion_idxes.diff(dim=0) == 1).all()

                # Micro-average: sum all frame-level sums, divide by total frames
                # (each timestep weighted equally, longer motions contribute more)
                metric_sums = gathered_metrics_stack[:, : num_body_metrics - 1]
                frame_counts = gathered_metrics_stack[:, num_body_metrics - 1 : num_body_metrics]

                success_mask = ~gathered_terminate_hist_stack
                success_metrics_mean = metric_sums[success_mask].sum(dim=0) / frame_counts[
                    success_mask
                ].sum(dim=0)
                all_metrics_mean = metric_sums.sum(dim=0) / frame_counts.sum(dim=0)

                # Also keep per-motion metrics for downstream use
                all_metrics = metric_sums / frame_counts
                metrics_all_print = {
                    k: all_metrics_mean[idx].cpu().numpy()
                    for idx, (k, v) in enumerate(metrics_all_sum.items())
                }
                metrics_succ_print = {
                    k: success_metrics_mean[idx].cpu().numpy()
                    for idx, (k, v) in enumerate(metrics_all_sum.items())
                }

                # Add object tracking metrics to printed summaries
                if has_obj_metrics:
                    obj_pos_err_mean = gathered_obj_pos_err.mean().cpu().numpy()
                    obj_ori_err_mean = gathered_obj_ori_err.mean().cpu().numpy()
                    obj_pos_err_succ = (
                        gathered_obj_pos_err[~gathered_terminate_hist_stack].mean().cpu().numpy()
                        if (~gathered_terminate_hist_stack).any()
                        else 0.0
                    )
                    obj_ori_err_succ = (
                        gathered_obj_ori_err[~gathered_terminate_hist_stack].mean().cpu().numpy()
                        if (~gathered_terminate_hist_stack).any()
                        else 0.0
                    )
                    metrics_all_print["obj_pos_error"] = obj_pos_err_mean
                    metrics_all_print["obj_ori_error"] = obj_ori_err_mean
                    metrics_succ_print["obj_pos_error"] = obj_pos_err_succ
                    metrics_succ_print["obj_ori_error"] = obj_ori_err_succ

                failed_keys = self.env._motion_lib._motion_data_keys[
                    gathered_terminate_hist_stack.cpu().numpy()
                ]
                success_keys = self.env._motion_lib._motion_data_keys[
                    ~gathered_terminate_hist_stack.cpu().numpy()
                ]
                success_rate = 1 - gathered_terminate_hist_stack.cpu().numpy().mean()
                progress_rate = gathered_progress_hist_stack.cpu().numpy().mean()

                all_metrics_dict = {
                    k: all_metrics[:, idx].cpu().numpy()
                    for idx, (k, v) in enumerate(metrics_all_sum.items())
                }
                all_metrics_dict["terminated"] = gathered_terminate_hist_stack.cpu().numpy()
                all_metrics_dict["terminate_reason"] = self.terminate_reason_memory[
                    : len(gathered_terminate_hist_stack)
                ]
                all_metrics_dict["rp2_diagnostic_reason"] = self.rp2_diagnostic_reason_memory[
                    : len(gathered_terminate_hist_stack)
                ]
                all_metrics_dict["progress"] = gathered_progress_hist_stack.cpu().numpy()
                all_metrics_dict["motion_keys"] = self.env._motion_lib._motion_data_keys[
                    gathered_motion_idxes.cpu().numpy()
                ]
                all_metrics_dict["sampling_prob"] = (
                    self.env._motion_lib._sampling_prob[gathered_motion_idxes.cpu().numpy()]
                    .cpu()
                    .numpy()
                )
                # Add per-motion object tracking metrics
                if has_obj_metrics:
                    all_metrics_dict["obj_pos_error"] = gathered_obj_pos_err.cpu().numpy()
                    all_metrics_dict["obj_ori_error"] = gathered_obj_ori_err.cpu().numpy()
                    # Save per-env obj_pos_error for threshold-based success rate analysis
                    if self.eval_only and len(self.obj_pos_error_all) > 0:
                        all_metrics_dict["per_env_obj_pos_error"] = [
                            v for batch in self.obj_pos_error_all for v in batch
                        ]
                        all_metrics_dict["per_env_obj_ori_error"] = [
                            v for batch in self.obj_ori_error_all for v in batch
                        ]

                failed_metrics_dict = {
                    k: all_metrics[gathered_terminate_hist_stack, idx].cpu().numpy()
                    for idx, (k, v) in enumerate(metrics_all_sum.items())
                }
                failed_metrics_dict["motion_keys"] = failed_keys
                failed_metrics_dict["sampling_prob"] = (
                    self.env._motion_lib._sampling_prob[gathered_terminate_hist_stack.cpu().numpy()]
                    .cpu()
                    .numpy()
                )
                if has_obj_metrics:
                    failed_metrics_dict["obj_pos_error"] = (
                        gathered_obj_pos_err[gathered_terminate_hist_stack].cpu().numpy()
                    )
                    failed_metrics_dict["obj_ori_error"] = (
                        gathered_obj_ori_err[gathered_terminate_hist_stack].cpu().numpy()
                    )

                if self.accelerator.is_main_process:
                    print(f"Success Rate: {success_rate:.10f}", flush=True)
                    print(f"Progress Rate: {progress_rate:.10f}", flush=True)
                    if has_obj_metrics:
                        print(
                            f"Object Pos Error (all): {obj_pos_err_mean:.4f}m | "
                            f"Object Ori Error (all): {obj_ori_err_mean:.4f}rad"
                        )
                    print(
                        "All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()])
                    )
                    print(
                        "Succ: ",
                        " \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]),
                    )
                    # Per-motion result lines for shell script parsing
                    _mk = all_metrics_dict.get("motion_keys", [])
                    _mt = all_metrics_dict.get("terminated", [])
                    _mp = all_metrics_dict.get("progress", [])
                    _mr = all_metrics_dict.get("terminate_reason", [])
                    _md = all_metrics_dict.get("rp2_diagnostic_reason", [])
                    _mm = all_metrics_dict.get("mpjpe_g", [])
                    for _i in range(len(_mk)):
                        _status = "FAIL" if (_i < len(_mt) and _mt[_i]) else "OK"
                        _prog = f"{_mp[_i]:.3f}" if _i < len(_mp) else "N/A"
                        _mpjpe = f"{_mm[_i]:.1f}" if _i < len(_mm) else "N/A"
                        _reason = _mr[_i] if _i < len(_mr) else ""
                        _diag = _md[_i] if _i < len(_md) else ""
                        print(
                            f"MOTION_RESULT: {_mk[_i]} status={_status} progress={_prog} mpjpe={_mpjpe} reason={_reason} diag={_diag}",
                            flush=True,
                        )

                metrics_succ_print["success_rate"] = success_rate
                metrics_succ_print["progress_rate"] = progress_rate
                actor_state["metrics_all"] = metrics_all_print
                actor_state["metrics_success"] = metrics_succ_print
                actor_state["failed_keys"] = failed_keys
                actor_state["success_keys"] = success_keys
                actor_state["all_metrics_dict"] = all_metrics_dict
                actor_state["failed_metrics_dict"] = failed_metrics_dict
                actor_state["failed_idxes"] = (
                    gathered_terminate_hist_stack.cpu().numpy().nonzero()[0]
                )

                if not self.eval_only:
                    del (
                        self.mpjpe,
                        self.mpjpe_all,
                        self.gt_pos,
                        self.gt_pos_all,
                        self.gt_rot,
                        self.gt_rot_all,
                        self.pred_pos,
                        self.pred_pos_all,
                        self.pred_rot,
                        self.pred_rot_all,
                        self.sampled_motion_idx,
                        self.obj_pos_error,
                        self.obj_pos_error_all,
                        self.obj_ori_error,
                        self.obj_ori_error_all,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()

                actor_state["end_eval"] = True
                self.pbar.update(1)
                self.pbar.refresh()
                return actor_state

            fresh_obs = self.env.forward_motion_samples(self.args.global_rank, self.args.world_size)
            actor_state["obs"] = fresh_obs
            self.model.policy.init_rollout()
            self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)
            self.progress_state = torch.zeros(self.env.num_envs, device=self.env.device)
            self.terminate_reason = [""] * self.env.num_envs

            self.success_rate = 0
            self.curr_steps = 0

            self.pbar.update(1)
            self.pbar.refresh()
            (
                self.mpjpe,
                self.gt_pos,
                self.pred_pos,
                self.obj_pos_error,
                self.obj_ori_error,
            ) = (
                [],
                [],
                [],
                [],
                [],
            )

        eval_time = (time.time() - self.time_eval_start) / 60  # in minutes
        obj_str = ""
        if self._has_object and len(self.obj_pos_error_all) > 0:
            mean_obj_pos_err = np.mean([v for batch in self.obj_pos_error_all for v in batch])
            obj_str = f" | ObjPosErr: {mean_obj_pos_err:.4f}m"
        update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_steps} | env_loop: {self.env_eval_loop_idx} | eval_time: {eval_time:.1f}m | Start: {self.env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all):.3f}{obj_str}"
        self.pbar.set_description(update_str)

        return actor_state
