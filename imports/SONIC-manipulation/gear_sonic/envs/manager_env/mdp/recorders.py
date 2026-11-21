"""Custom recorder terms for the manager environment MDP."""

from __future__ import annotations

import json
import os
import pickle
from typing import TYPE_CHECKING

import cv2
import imageio
from isaaclab.managers import manager_term_cfg, recorder_manager
from isaaclab.utils import configclass
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from gear_sonic.envs.manager_env.mdp import utils
from gear_sonic.utils.rendering import (
    camera_pose_with_focus,
    should_write_render_frame,
)

if TYPE_CHECKING:
    from isaaclab import envs


@configclass
class RecordersCfg(recorder_manager.RecorderManagerBaseCfg):
    """Recorders terms for the MDP."""

    render_envs = None
    running_ref_root_height = None
    trajectory = None


class RenderEnvsRecorderTerm(recorder_manager.RecorderTerm):
    """Recorder term for rendering environments with advanced features like text overlay and frame skipping."""

    cfg: RenderEnvsRecorderCfg

    def __init__(self, cfg: RenderEnvsRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        # Determine save directory (backward compatibility)
        self.save_dir = self.cfg.video_save_path
        logger.info(f"=== Start recording video to {self.save_dir} ===")

        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        self.video_writers = []
        self._writers_closed = False
        self.frame_id = 0
        self.first_render = True
        self._fixed_eye = None
        self._fixed_target = None

    def _initialize_writers(self):
        """Initialize video writers for each environment."""
        logger.info(f"Saving rendering to {self.save_dir}")
        # Get configuration parameters with defaults
        self.group_camera = self.env.wrapper.config.get("group_camera", False)
        self.max_render_envs = self.env.wrapper.config.get("max_render_envs", self.env.num_envs)
        requested_env_ids = self.env.wrapper.config.get("render_env_ids", None)
        self.sparse_render_selection = requested_env_ids is not None
        if self.group_camera:
            if requested_env_ids is not None:
                raise ValueError("group-camera rendering forbids render_env_ids")
            self.max_render_envs = 1  # single video from overview_camera
            self.render_env_ids = None
        elif requested_env_ids is None:
            self.render_env_ids = list(range(int(self.max_render_envs)))
        else:
            self.render_env_ids = [int(env_id) for env_id in requested_env_ids]
            if len(self.render_env_ids) != int(self.max_render_envs):
                raise ValueError(
                    "max_render_envs must equal the number of render_env_ids"
                )
            if self.render_env_ids != sorted(set(self.render_env_ids)):
                raise ValueError("render_env_ids must be unique and increasing")
            if (
                not self.render_env_ids
                or self.render_env_ids[0] < 0
                or self.render_env_ids[-1] >= self.env.num_envs
            ):
                raise ValueError("render_env_ids lie outside the environment batch")
        self.render_frame_skip = int(
            self.env.wrapper.config.get("render_frame_skip", 2)
        )
        if self.render_frame_skip <= 0:
            raise ValueError(
                f"render frame skip must be positive, got {self.render_frame_skip}"
            )
        self.render_skip_first_frame = self.env.wrapper.config.get(
            "render_skip_first_frame", True
        )
        self.start_idx = self.env.wrapper.start_idx

        writer_env_ids = [0] if self.group_camera else self.render_env_ids
        for env_id in writer_env_ids:
            file_name = f"{self.save_dir}/{self.start_idx+env_id:06d}.mp4"
            fps = 1 / (self.env.step_dt * self.render_frame_skip)
            writer = imageio.get_writer(
                file_name,
                fps=fps,
                codec="libx264",
                quality=self.cfg.video_quality,
                pixelformat="yuv420p",
            )
            self.video_writers.append(writer)

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record video frames after each step with frame skipping and text overlay support."""
        if len(self.video_writers) == 0:
            self._initialize_writers()

        # Check if we should render this frame. GPC audit renders use skip=1
        # and retain frame zero so a first-step terminal state cannot vanish.
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)

        # Set camera position based on robot root position
        root_pos = self.env.scene["robot"].data.root_pos_w
        if self.sparse_render_selection:
            root_pos = root_pos[self.render_env_ids]
        camera_offset = self.env.wrapper.config.get("eval_camera_offset", [2, 2, 1])
        focus_entities = self.env.wrapper.config.get(
            "eval_camera_focus_entities", None
        )
        if focus_entities is None:
            focus_entity = self.env.wrapper.config.get(
                "eval_camera_focus_entity", None
            )
            focus_entities = [focus_entity] if focus_entity is not None else []
        focus_pos = None
        if focus_entities:
            try:
                focus_pos = torch.stack(
                    [
                        (
                            self.env.scene[str(entity)].data.root_pos_w
                            if not self.sparse_render_selection
                            else self.env.scene[str(entity)].data.root_pos_w[
                                self.render_env_ids
                            ]
                        )
                        for entity in focus_entities
                    ],
                    dim=1,
                )
            except (KeyError, AttributeError) as exc:
                raise RuntimeError(
                    "eval camera focus entity is unavailable: "
                    f"{list(focus_entities)!r}"
                ) from exc
        runtime_focus_attr = self.env.wrapper.config.get(
            "eval_camera_runtime_focus_attr", None
        )
        if runtime_focus_attr is not None:
            runtime_focus = getattr(self.env, str(runtime_focus_attr), None)
            if runtime_focus is None:
                raise RuntimeError(
                    "eval camera runtime focus is unavailable: "
                    f"{runtime_focus_attr!r}"
                )
            if self.sparse_render_selection:
                runtime_focus = runtime_focus[self.render_env_ids]
            runtime_focus = runtime_focus.unsqueeze(1)
            focus_pos = (
                runtime_focus
                if focus_pos is None
                else torch.cat((focus_pos, runtime_focus), dim=1)
            )
        interaction_eye, interaction_target = camera_pose_with_focus(
            root_pos,
            torch.tensor(camera_offset, device=self.env.device),
            focus_pos=focus_pos,
            focus_margin_m=float(
                self.env.wrapper.config.get("eval_camera_focus_margin_m", 1.0)
            ),
        )
        fix_camera = self.env.wrapper.config.get("fix_camera_after_first_frame", False)
        cam = self.env.scene["eval_camera"]

        if fix_camera and self._fixed_eye is not None:
            # Reuse the camera position from the first frame
            eye, target = self._fixed_eye, self._fixed_target
        elif self.group_camera:
            center = root_pos.mean(dim=0, keepdim=True).expand_as(root_pos)
            eye = center + torch.tensor(camera_offset, device=self.env.device)
            target = center
            if fix_camera:
                self._fixed_eye = eye.clone()
                self._fixed_target = center.clone()
        else:
            eye = interaction_eye
            target = interaction_target
            if fix_camera:
                self._fixed_eye = eye.clone()
                self._fixed_target = target.clone()

        # Write world poses to Fabric AND sync to USD so both renderer paths see it
        cam._view._sync_usd_on_fabric_write = True  # noqa: SLF001
        cam.set_world_poses_from_view(eye, target)

        # Two render calls: 1st flushes pose to render pipeline, 2nd captures at new pose
        if hasattr(self.env, "sim"):
            self.env.sim.render()
            self.env.sim.render()

        # Mark sensor as outdated so update actually re-reads the annotator buffers
        cam._is_outdated[:] = True  # noqa: SLF001
        cam.update(dt=0.0, force_recompute=True)

        # Get RGB data
        rgb_viewer = cam.data.output["rgb"].clone()
        if (
            self.sparse_render_selection
            and int(rgb_viewer.shape[0]) != int(self.max_render_envs)
        ):
            raise RuntimeError(
                "camera tile count differs from selected render environments: "
                f"{int(rgb_viewer.shape[0])} != {int(self.max_render_envs)}"
            )

        # Get render info if available
        cur_render_info = None
        if self.env.wrapper.config.get("render_info", None) is not None:
            if self.group_camera:
                cur_render_info = self.env.wrapper.config.render_info[
                    self.start_idx : self.start_idx + 1
                ]
            else:
                cur_render_info = [
                    self.env.wrapper.config.render_info[self.start_idx + env_id]
                    for env_id in self.render_env_ids
                ]

        # Process each environment, loop over the video writers
        if should_write_render_frame(
            self.frame_id,
            frame_skip=self.render_frame_skip,
            skip_first_frame=bool(self.render_skip_first_frame),
        ):
            loop = (
                tqdm(range(self.max_render_envs))
                if self.first_render
                else range(self.max_render_envs)
            )
            show_terminal = bool(
                self.env.wrapper.config.get("render_terminal_overlay", False)
            )
            terminal_flags = (
                (
                    self.env.reset_buf[:1]
                    if self.group_camera
                    else self.env.reset_buf[self.render_env_ids]
                )
                .detach()
                .to(device="cpu", dtype=torch.bool)
                .tolist()
                if show_terminal
                else [False] * self.max_render_envs
            )
            text_color = tuple(
                int(value)
                for value in self.env.wrapper.config.get(
                    "render_text_color", [0, 0, 0]
                )
            )
            outline_value = self.env.wrapper.config.get(
                "render_text_outline_color", None
            )
            outline_color = (
                tuple(int(value) for value in outline_value)
                if outline_value is not None
                else None
            )
            for i in loop:
                frame = rgb_viewer[i].cpu().numpy()

                # Add text overlay if render info is provided
                if cur_render_info is not None and i < len(cur_render_info):
                    render_lines = list(cur_render_info[i])
                    if terminal_flags[i]:
                        render_lines.append("TERMINAL -> RESET")
                    for j, text in enumerate(render_lines):
                        position = (10, 30 + j * 25)
                        if outline_color is not None:
                            frame = cv2.putText(
                                frame,
                                str(text),
                                position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                outline_color,
                                3,
                            )
                        frame = cv2.putText(
                            frame,
                            str(text),
                            position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            text_color,
                            1,
                        )

                self.video_writers[i].append_data(frame)
            self.first_render = False

        self.frame_id += 1
        return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self):
        """Explicitly close all video writers."""
        if not self._writers_closed:
            for i, writer in enumerate(self.video_writers):
                try:
                    writer.close()
                    logger.info(f"Closed video writer {i}")
                except Exception as e:  # noqa: BLE001
                    logger.info(f"Error closing video writer {i}: {e}")
            self.video_writers.clear()
            self._writers_closed = True
            self.frame_id = 0
            self.first_render = True
            self._fixed_eye = None
            self._fixed_target = None
            logger.info("=== All video writers closed ===")

    def __del__(self):
        """Ensure writers are closed when object is destroyed."""
        self.close_writers()


@configclass
class RenderEnvsRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for environment rendering recorder with advanced features."""

    class_type = RenderEnvsRecorderTerm
    video_save_path: str = None
    video_quality: int = 5


class TrajectoryRecorderTerm(recorder_manager.RecorderTerm):
    """Recorder term that saves per-environment trajectory data (joint positions, root pose, object/table state).

    Saves .trajectory.pkl files alongside the video output, enabling kinematic replay
    in multi-scene composite renders.
    """

    cfg: TrajectoryRecorderCfg

    def __init__(self, cfg: TrajectoryRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        self.save_dir = self.cfg.save_path
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"=== TrajectoryRecorder: saving to {self.save_dir} ===")

        self._initialized = False
        self._closed = False
        self._frame_data: dict[int, dict] = {}  # env_idx -> {field: [frames]}
        self.frame_id = 0

    def _initialize(self):
        """Initialize per-env data buffers after environment is ready."""
        self.num_record_envs = self.env.num_envs
        self.start_idx = (
            getattr(self.env.wrapper, "start_idx", 0) if hasattr(self.env, "wrapper") else 0
        )

        # Match video recorder's frame skip to keep trajectory in sync with video
        if hasattr(self.env, "wrapper"):
            self.render_frame_skip = self.env.wrapper.config.get("render_frame_skip", 2)
        else:
            self.render_frame_skip = 2

        # Detect available scene entities
        self._has_object = (
            "object" in self.env.scene.rigid_objects
            or "active_articulation" in self.env.scene.articulations
        )
        self._has_table = "table" in self.env.scene.rigid_objects

        # Get motion command for root pose
        try:
            self._motion_cmd = self.env.command_manager.get_term("motion")
        except Exception:  # noqa: BLE001
            self._motion_cmd = None

        for i in range(self.num_record_envs):
            self._frame_data[i] = self._create_empty_data()

        self._initialized = True

    def _create_empty_data(self) -> dict:
        data = {
            "dof_pos": [],
            "root_pos_w": [],
            "root_quat_w": [],
        }
        if self._has_object:
            data["object_pos_w"] = []
            data["object_quat_w"] = []
        if self._has_table:
            data["table_pos_w"] = []
            data["table_quat_w"] = []
        return data

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record trajectory state after each step, synced with video frame skip."""
        if not self._initialized:
            self._initialize()

        # Skip frames to match video recorder cadence
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "trajectory_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

        robot = self.env.scene["robot"]
        env_origins = self.env.scene.env_origins
        object_pos_w = None
        object_quat_w = None
        if self._has_object and self._motion_cmd is not None:
            object_pos_w, object_quat_w, _, _, _ = utils.get_target_object_current_state(
                self.env, self._motion_cmd
            )

        for i in range(self.num_record_envs):
            # Joint positions
            joint_pos = robot.data.joint_pos[i].cpu().numpy().copy()
            self._frame_data[i]["dof_pos"].append(joint_pos)

            # Root position (relative to env origin)
            if self._motion_cmd is not None:
                root_pos = self._motion_cmd.robot_body_pos_w[i, 0].cpu().numpy().copy()
            else:
                root_pos = robot.data.root_pos_w[i].cpu().numpy().copy()
            root_pos_rel = root_pos - env_origins[i].cpu().numpy()
            self._frame_data[i]["root_pos_w"].append(root_pos_rel)

            # Root quaternion (wxyz)
            root_quat = robot.data.root_quat_w[i].cpu().numpy().copy()
            self._frame_data[i]["root_quat_w"].append(root_quat)

            # Object state
            if object_pos_w is not None:
                obj_pos = object_pos_w[i].cpu().numpy().copy()
                obj_pos_rel = obj_pos - env_origins[i].cpu().numpy()
                obj_quat = object_quat_w[i].cpu().numpy().copy()
                self._frame_data[i]["object_pos_w"].append(obj_pos_rel)
                self._frame_data[i]["object_quat_w"].append(obj_quat)

            # Table state
            if self._has_table:
                table = self.env.scene["table"]
                table_pos = table.data.root_pos_w[i].cpu().numpy().copy()
                table_pos_rel = table_pos - env_origins[i].cpu().numpy()
                table_quat = table.data.root_quat_w[i].cpu().numpy().copy()
                self._frame_data[i]["table_pos_w"].append(table_pos_rel)
                self._frame_data[i]["table_quat_w"].append(table_quat)

        self.frame_id += 1
        return "trajectory_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self):
        """Save all trajectory data to pkl files."""
        if self._closed or not self._initialized:
            return
        self._closed = True

        # FPS matches the video (after frame skip)
        effective_fps = 1.0 / (self.env.step_dt * self.render_frame_skip)

        scene_metadata = {}

        for i in range(self.num_record_envs):
            env_idx = self.start_idx + i
            data = self._frame_data[i]

            if not data["dof_pos"]:
                continue

            # Stack frame arrays
            trajectory = {
                "dof_pos": np.array(data["dof_pos"]),
                "root_pos_w": np.array(data["root_pos_w"]),
                "root_quat_w": np.array(data["root_quat_w"]),
                "quat_format": "wxyz",
                "fps": effective_fps,
                "num_joints": data["dof_pos"][0].shape[0],
                "total_frames": len(data["dof_pos"]),
            }

            if data.get("object_pos_w"):
                trajectory["object_pos_w"] = np.array(data["object_pos_w"])
                trajectory["object_quat_w"] = np.array(data["object_quat_w"])
            else:
                trajectory["object_pos_w"] = None
                trajectory["object_quat_w"] = None

            if data.get("table_pos_w"):
                trajectory["table_pos_w"] = np.array(data["table_pos_w"])
                trajectory["table_quat_w"] = np.array(data["table_quat_w"])
            else:
                trajectory["table_pos_w"] = None
                trajectory["table_quat_w"] = None

            # Save pkl
            pkl_path = os.path.join(self.save_dir, f"{env_idx:06d}.trajectory.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved trajectory: {pkl_path} ({trajectory['total_frames']} frames)")

            # Build metadata entry
            meta = {
                "trajectory_file": f"{env_idx:06d}.trajectory.pkl",
                "video_file": f"{env_idx:06d}.mp4",
                "num_frames": trajectory["total_frames"],
                "num_joints": trajectory["num_joints"],
                "fps": effective_fps,
                "has_object": trajectory["object_pos_w"] is not None,
                "has_table": trajectory["table_pos_w"] is not None,
            }

            # Add object USD path if available from config
            if hasattr(self.env, "wrapper"):
                obj_usd = self.env.wrapper.config.get("object_usd_path", None)
                if obj_usd:
                    meta["object_usd_path"] = obj_usd

            scene_metadata[str(env_idx)] = meta

        # Save scene metadata JSON
        meta_path = os.path.join(self.save_dir, "scene_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(scene_metadata, f, indent=2)
        logger.info(f"Saved scene metadata: {meta_path}")
        logger.info("=== TrajectoryRecorder: all data saved ===")

    def __del__(self):
        self.close_writers()


@configclass
class TrajectoryRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for trajectory recording alongside video."""

    class_type = TrajectoryRecorderTerm
    save_path: str = None
