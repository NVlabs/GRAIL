"""One continuous, physically simulated episode; stop before IsaacLab resets."""
from __future__ import annotations

import json
import math
from pathlib import Path


class EpisodeComplete(Exception):
    """A controlled exit from record_post_step, before the environment reset."""


def classify_end(reference_step, reference_total, terminated, truncated, terms):
    if reference_total < 1 or not 0 <= reference_step < reference_total:
        raise ValueError("Invalid reference timeline")
    reference_completed = reference_step + 1 >= reference_total
    if terminated:
        return "failure:" + ",".join(terms or ["unspecified_terminal"])
    if reference_completed:
        return "reference_complete"
    if truncated:
        return "early_truncation:" + ",".join(terms or ["unspecified_timeout"])
    return None


def summarize_errors(records):
    physics = [r for r in records if r["physics_step"] > 0]
    if not physics:
        raise RuntimeError("No physics transitions were recorded")
    result = {"valid_object_samples": len(physics), "missing_object_samples": 0}
    for prefix, field, unit in (
        ("object_position_error", "position_error_m", "m"),
        ("object_orientation_error", "orientation_error_rad", "rad"),
    ):
        values = [float(r[field]) for r in physics]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Nonfinite or negative object pose error")
        result[f"{prefix}_mean_{unit}"] = sum(values) / len(values)
        result[f"{prefix}_rms_{unit}"] = math.sqrt(sum(v * v for v in values) / len(values))
        result[f"{prefix}_max_{unit}"] = max(values)
    return result


def make_recorders(output_root, render=True):
    """Recorder for one deterministic episode; ``render=False`` records states and metrics without video."""
    # These imports must happen after AppLauncher starts Kit.
    from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from gear_sonic.envs.manager_env.mdp.recorders import (
        RecordersCfg, RenderEnvsRecorderCfg, RenderEnvsRecorderTerm,
    )
    from gear_sonic.envs.manager_env.mdp.utils import get_all_object_current_state
    from tracking_metrics import pose_errors

    base = RenderEnvsRecorderTerm if render else RecorderTerm

    class FullEpisodeRecorder(base):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.env = env
            self.active = False
            self.records = []
            self.physics_steps = 0
            self.terminal = None
            self.reset_count = 0

        def capture(self, initial=False):
            command = self.env.command_manager.get_term("motion")
            ref_step = int((command.motion_start_time_steps + command.time_steps)[0].item())
            ref_total = int(command.motion_lib.get_time_step_total(command.motion_ids)[0].item())
            if initial and ref_step != 0:
                raise RuntimeError(f"Evaluation must start at reference frame 0, got {ref_step}")
            if self.records and not initial and ref_step != self.physics_steps - 1:
                raise RuntimeError(f"Reference jump detected: {ref_step} at step {self.physics_steps}")
            actual_pos, actual_quat, _, _, valid = get_all_object_current_state(
                self.env, command, max_objects=1,
            )
            if valid.numel() != 1 or not bool(valid.all().item()):
                raise RuntimeError("Evaluation requires one valid physical object")
            position, orientation = pose_errors(
                actual_pos, actual_quat, command.object_root_pos[:, :1],
                command.object_root_quat[:, :1], valid,
            )
            position, orientation = float(position.item()), float(orientation.item())
            if not math.isfinite(position) or not math.isfinite(orientation):
                raise RuntimeError("Nonfinite object state during inference")
            manager = self.env.termination_manager
            names = list(manager.active_terms)
            masks = manager._term_dones[0].detach().cpu().tolist()
            terms = [str(name) for name, value in zip(names, masks) if value] if not initial else []
            terminated = bool(manager.terminated[0].item()) if not initial else False
            truncated = bool(manager.time_outs[0].item()) if not initial else False
            end = classify_end(ref_step, ref_total, terminated, truncated, terms) if not initial else None
            record = {
                "physics_step": self.physics_steps, "reference_step": ref_step,
                "reference_total_steps": ref_total,
                "simulation_time_s": self.physics_steps * float(self.env.step_dt),
                "position_error_m": position, "orientation_error_rad": orientation,
                "actual_object_position": actual_pos[0, 0].detach().cpu().tolist(),
                "actual_object_quaternion_wxyz": actual_quat[0, 0].detach().cpu().tolist(),
                "reference_object_position": command.object_root_pos[0, 0].detach().cpu().tolist(),
                "reference_object_quaternion_wxyz": command.object_root_quat[0, 0].detach().cpu().tolist(),
                "termination_terms": terms, "terminated": terminated,
                "truncated": truncated, "end_reason": end,
            }
            self.records.append(record)
            lines = [
                "Deterministic checkpoint inference | fixed camera | no reset",
                f"Reference {ref_step + 1}/{ref_total} | simulated {record['simulation_time_s']:.2f}s",
                f"Object error: {position * 100:.2f} cm, {math.degrees(orientation):.2f} deg",
            ]
            if end:
                lines.append("END: " + end)
            if render:
                self.env.wrapper.config.render_info = [lines]
                super().record_post_step()
            if end:
                self.terminal = record
                self.active = False
                if render:
                    self.close_writers()
                raise EpisodeComplete(end)

        def record_post_step(self):
            if self.active:
                self.physics_steps += 1
                self.capture()
            return None, None

        def begin(self):
            if self.active or self.records:
                raise RuntimeError("Only one episode is permitted")
            self.active = True
            self.capture(initial=True)

        def record_pre_reset(self, env_ids):
            if self.active:
                self.reset_count += len(env_ids)
                raise RuntimeError("Unexpected auto-reset before the recorder stopped")
            return None, None

        def finalize(self):
            if render:
                self.close_writers()
            if self.terminal is None or self.reset_count:
                raise RuntimeError("Continuous episode did not terminate without resets")
            Path(output_root, "frames.json").write_text(json.dumps(self.records, indent=2) + "\n")
            terminal = self.terminal
            return {
                "start_frame": 0, "deterministic": True,
                "action_mode": "policy.act_inference_mean", "reset_count": self.reset_count,
                "end_reason": terminal["end_reason"],
                "termination_terms": terminal["termination_terms"],
                "reference_total_steps": terminal["reference_total_steps"],
                "reached_reference_step": terminal["reference_step"],
                "reference_completed": terminal["reference_step"] + 1 >= terminal["reference_total_steps"],
                "tracking_success": terminal["end_reason"] == "reference_complete",
                "tracking_success_definition": "full reference reached without native failure termination; one deterministic in-distribution episode, not held-out success rate",
                "physics_steps": self.physics_steps,
                "control_dt_s": float(self.env.step_dt),
                "simulator_steps_per_control": int(self.env.cfg.decimation),
                "camera": "fixed after initial frame" if render else "none (metrics-only evaluation)",
                "video_frame_count": len(self.records) if render else 0,
            }, summarize_errors(self.records)

    if render:
        return RecordersCfg(
            render_envs=RenderEnvsRecorderCfg(
                class_type=FullEpisodeRecorder, video_save_path=str(Path(output_root) / "raw_video"),
            ),
            dataset_export_mode=DatasetExportMode.EXPORT_NONE,
            export_in_record_pre_reset=False, export_in_close=False,
        )
    cfg = RecorderManagerBaseCfg()
    cfg.render_envs = RecorderTermCfg(class_type=FullEpisodeRecorder)
    cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    cfg.export_in_record_pre_reset = False
    cfg.export_in_close = False
    return cfg
