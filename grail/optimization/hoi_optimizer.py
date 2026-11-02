import datetime
import json
import os
from glob import glob

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)

from grail.constants.image import FOCAL_LENGTH, HEIGHT, WIDTH
from grail.core.contact_label import detect_contact_joints_interval
from grail.core.depth_config import DEPTH_CACHE_SUBDIR
from grail.core.io import (
    load_character_data,
    load_human_motion_data,
    load_init_rendering_data,
    load_mesh,
    load_object_pose_data,
)
from grail.core.logging import create_logger
from grail.core.torch_utils import tensor_to_numpy
from grail.core.video import (
    extract_frames_from_video,
    get_video_fps_and_frame_count,
)
from grail.models.human_model import create_human_model
from grail.optimization.data_types import HOIData, HOIPrediction, OptParams
from grail.optimization.evaluator import pre_eval, truncate_data
from grail.optimization.interaction import (
    get_contact_labels_for_frame,
    identify_interaction_start_end,
    identify_interaction_start_end_with_mask,
)
from grail.optimization.loss_computer import LossComputer
from grail.optimization.visualizer import HOIVisualizer
from grail.pose_est.utils import smooth_axis_angle_sequence, smooth_pose_sequence
from grail.preprocessing.preprocess import load_depth_from_cache, load_masks_from_cache
from grail.rendering.camera import (
    cam_pose_blender_to_opencv,
    cam_pose_opencv_to_pytorch3d,
    get_camera,
    project_world_to_screen,
    transform_pose_c2w,
)
from grail.visualization.scenepic import ScenepicVisualizer


class HOIOptimizer:
    """
    Human-Object Interaction Optimizer
    Optimizes SMPL human motion and object trajectories
    """

    def __init__(self, exp_name, cfg, cache_dir, output_dir, device="cuda"):
        """
        Initialize the HOI optimizer

        Args:
            exp_name (str): Experiment name (dataset/category/video_id)
            cfg (dict): Optimizer configuration dictionary (optimization +
                        human_model sections merged from the unified YAML)
            cache_dir (str): Directory for cached data
            output_dir (str): Directory for optimization outputs and logs
            device (str): Device for computations
        """
        self.device = device
        self.cfg = cfg

        self.log_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.logger = create_logger(self.log_dir)
        self.cache_list = self.cfg.get("cache_list", [])
        self.cache_dir = cache_dir

        # TODO: double check this logic when changing the exp_name format
        dataset, category, video_id = exp_name.split("/")
        self.exp_name = exp_name
        self.obj_name = category
        self.character_name = "_".join(video_id.split("_")[:3])
        print(f"Experiment name: {self.exp_name}")
        print(f"Object name: {self.obj_name}")

        self.opt_stage_specs = self.cfg["opt_stage_specs"]

        self.vis_cfg = self.cfg["vis_cfg"]
        self.enable_vis = self.vis_cfg.get("enable", True)

        self.eval_cfg = self.cfg["eval"]

        # Pre-evaluation configuration
        self.pre_eval_cfg = self.cfg.get("pre_eval", {})
        self.min_frames_threshold = self.pre_eval_cfg.get("min_frames", None)

        # Body model — polymorphic: SOMA, SMPL-X, or G1-proportioned SMPL-X
        self.human_model = create_human_model(self.cfg["human_model"], device)
        self.logger.info(f"Using {self.human_model.model_type} body model")

    def init_data(self, video_file, hmr_file, obj_path, obj_pose_file, render_config_file):
        """Load all data and assemble HOIData for optimization."""
        video_id = self.exp_name

        # 1. Camera and rendering setup
        camera, cameras, opencv_cam_R, opencv_cam_t, obj_scale, static_objects = (
            self._load_camera_config(render_config_file)
        )

        # 2. Object mesh and poses
        obj_verts, obj_faces, obj_poses_incam, obj_poses = self._load_object(
            obj_path, obj_scale, obj_pose_file, opencv_cam_R, opencv_cam_t
        )

        # 3. Video frames and masks
        images_path, video_fps, video_frame_count = self._load_video_frames(video_file)
        human_masks, obj_masks = self._load_masks(video_id)

        # 4. Interaction detection
        inter_start_idx, inter_end_idx, is_static_obj = self._detect_interaction(
            obj_poses_incam, images_path, human_masks, obj_masks
        )

        # 5. Human motion data
        motion_data, motion_data_global_init, foot_contact_probs = self._load_motion(
            hmr_file, render_config_file, opencv_cam_R, opencv_cam_t, inter_start_idx
        )

        # Validate frame counts
        frame_num = motion_data["poses"].shape[0]
        if len(obj_poses) != frame_num:
            raise ValueError(f"Frame count mismatch - SMPL: {frame_num}, Object: {len(obj_poses)}")
        if video_frame_count != frame_num:
            raise ValueError(
                f"Frame count mismatch - Video: {video_frame_count}, SMPL: {frame_num}"
            )

        # 6. Contact labels
        contact_labels, contact_interval, contact_start_idx = self._detect_contact_labels(
            video_id, frame_num, inter_start_idx, is_static_obj, images_path
        )

        # 7. Depth maps
        depth_maps = self._load_depth(video_id)

        # 8. Prepare GT keypoints
        human_faces = self.human_model.generate_mesh(
            motion_data, output_joints=False, require_grad=False
        )[1]
        gt_body_kp, gt_hand_kp = self.human_model.extract_gt_keypoints(motion_data)

        # 9. Transform object vertices to per-frame positions
        obj_verts_seq = torch.bmm(
            obj_verts.unsqueeze(0).repeat(frame_num, 1, 1),
            obj_poses[:, :3, :3].transpose(1, 2),
        ) + obj_poses[:, :3, 3].reshape(frame_num, 1, 3)

        gt_obj_verts_tracking = project_world_to_screen(
            obj_verts_seq.reshape(-1, 3), cameras
        ).reshape(frame_num, -1, 3)[:, :, :2]

        # ── Set self.* state (all in one place) ──────────────────────────────
        self.cameras = cameras
        self.obj_path = obj_path
        self.obj_mesh = Meshes(verts=[obj_verts], faces=[obj_faces])
        self.video_fps = video_fps
        self.image_list = images_path
        self.depth_list = depth_maps
        self.contact_labels = contact_labels
        self.contact_interval = contact_interval
        self.contact_start_idx = contact_start_idx

        # ── Assemble HOIData ─────────────────────────────────────────────────
        return HOIData(
            frame_num=frame_num,
            inter_start_idx=inter_start_idx,
            inter_end_idx=inter_end_idx,
            human=HOIData.Human(
                faces=human_faces,
                masks=[human_masks[i] for i in range(frame_num)],
                motion_data=motion_data,
                motion_data_global_init=motion_data_global_init,
                body_keypoints_seq=gt_body_kp,
                hand_keypoints_seq=gt_hand_kp,
                foot_contact_probs=foot_contact_probs,
            ),
            obj=HOIData.Object(
                scale=obj_scale,
                verts=obj_verts,
                faces=obj_faces,
                masks=[obj_masks[i] for i in range(frame_num)],
                verts_seq=obj_verts_seq,
                poses=obj_poses,
                verts_tracking_seq=gt_obj_verts_tracking,
            ),
            camera=camera,
            images_path=images_path,
            depth_maps=depth_maps,
            is_static_obj=is_static_obj,
            static_objects=static_objects,
        )

    # ── init_data sub-methods (no self.* side effects) ───────────────────────

    def _load_camera_config(self, render_config_file):
        """Load rendering config and create camera."""
        _, _, obj_scale, blender_cam_R, blender_cam_t, render_config, additional_data = (
            load_init_rendering_data(
                render_config_file,
                to_tensor=True,
                with_human_data=True,
                with_scene_data=True,
                device=self.device,
            )
        )
        opencv_cam_R, opencv_cam_t = cam_pose_blender_to_opencv(blender_cam_R, blender_cam_t)
        if render_config is not None:
            frame_height, frame_width, focal_length = render_config
        else:
            frame_height, frame_width, focal_length = HEIGHT, WIDTH, FOCAL_LENGTH

        static_objects = additional_data.get("static_objects", None)

        cam_R, cam_t = cam_pose_opencv_to_pytorch3d(opencv_cam_R, opencv_cam_t)
        cameras = get_camera(
            cam_R, cam_t, focal_length, (frame_height, frame_width), device=self.device
        )

        cam_pose = torch.eye(4, device=self.device)
        cam_pose[:3, :3] = cam_R
        cam_pose[:3, 3] = cam_t.reshape((3,))

        camera = HOIData.Camera(
            pose=cam_pose,
            frame_height=frame_height,
            frame_width=frame_width,
            focal_length=focal_length,
        )
        return camera, cameras, opencv_cam_R, opencv_cam_t, obj_scale, static_objects

    def _load_object(self, obj_path, obj_scale, obj_pose_file, opencv_cam_R, opencv_cam_t):
        """Load object mesh and pose trajectory."""
        obj_verts, obj_faces, _ = load_mesh(
            obj_path, mesh_scale=obj_scale, target_num_verts=6000, device=self.device
        )
        obj_poses_incam = load_object_pose_data(obj_pose_file, to_tensor=True, device=self.device)
        obj_poses = transform_pose_c2w(obj_poses_incam, opencv_cam_R, opencv_cam_t)
        return obj_verts, obj_faces, obj_poses_incam, obj_poses

    def _load_video_frames(self, video_file):
        """Extract video frames and return image paths and fps."""
        video_basename = os.path.basename(video_file).split(".")[0]
        frame_cache_dir = os.path.join(os.path.dirname(video_file), "frames", video_basename)

        video_fps, video_frame_count = get_video_fps_and_frame_count(video_file)

        images_path = sorted(glob(os.path.join(frame_cache_dir, "*.jpg")))
        if not images_path:
            extracted = extract_frames_from_video(video_file, frame_cache_dir)
            if extracted != video_frame_count:
                raise RuntimeError(
                    f"Frame extraction mismatch - Video: {video_frame_count}, "
                    f"Extracted: {extracted}"
                )
            images_path = sorted(glob(os.path.join(frame_cache_dir, "*.jpg")))
        if len(images_path) != video_frame_count:
            raise RuntimeError(
                f"Incomplete frame cache - Video: {video_frame_count}, "
                f"Cached: {len(images_path)} at {frame_cache_dir}"
            )
        return images_path, video_fps, video_frame_count

    def _load_masks(self, video_id):
        """Load and split preprocessed masks into human and object masks."""
        masks_cache_file = os.path.join(self.cache_dir, "masks", f"{video_id}.npz")
        if not os.path.exists(masks_cache_file):
            raise FileNotFoundError(
                f"Masks cache not found: {masks_cache_file}. Run preprocessing (step1) first."
            )
        preprocess_masks = load_masks_from_cache(masks_cache_file)
        # Preprocess saves obj=0, human=1 — split into separate dicts
        human_masks = {fi: preprocess_masks[fi][1] for fi in preprocess_masks}
        obj_masks = {fi: preprocess_masks[fi][0] for fi in preprocess_masks}
        self.logger.info(f"Loaded masks from cache: {masks_cache_file}")
        return human_masks, obj_masks

    def _detect_interaction(self, obj_poses_incam, images_path, human_masks, obj_masks):
        """Detect interaction start/end frame."""
        has_interaction_end = self.cfg.get("has_interaction_end", False)
        if self.cfg.get("detect_interaction_with_mask", False):
            self.logger.info("Using mask-based interaction detection")
            masks = {fi: {0: human_masks[fi], 1: obj_masks[fi]} for fi in human_masks}
            result = identify_interaction_start_end_with_mask(
                masks,
                images_path,
                self.logger,
                has_interaction_end=has_interaction_end,
            )
        else:
            result = identify_interaction_start_end(
                obj_poses_incam,
                images_path,
                self.obj_name,
                self.logger,
                has_interaction_end=has_interaction_end,
            )
        inter_start_idx, inter_end_idx, is_static_obj = result
        if self.cfg.get("is_static_obj", False):
            # override the result if is_static_obj is set True in the config
            is_static_obj = True
        self.logger.info(f"Interaction start frame: {inter_start_idx}/{len(images_path)}")
        self.logger.info(f"Interaction end frame: {inter_end_idx}/{len(images_path)}")
        self.logger.info(f"Is static object: {is_static_obj}")
        return inter_start_idx, inter_end_idx, is_static_obj

    def _load_motion(
        self, hmr_file, render_config_file, opencv_cam_R, opencv_cam_t, inter_start_idx
    ):
        """Load human motion data and transform to camera frame.

        Handles shape/height scaling so that motion_data is fully scaled on return.
        Two cases:
            1. GT shape params available: replace predicted shape, transform_global_motion
               handles translation scaling internally.
            2. No GT shape, but GT height: keep predicted shape, adjust body scale so
               generate_mesh produces correctly sized geometry.
        """
        global_motion_data = load_human_motion_data(
            hmr_file, is_global=True, to_tensor=True, device=self.device
        )
        incam_motion_data = load_human_motion_data(
            hmr_file, is_global=False, to_tensor=True, device=self.device
        )
        foot_contact_probs = incam_motion_data["foot_contact_probs"]

        # HMR-predicted body height is computed upstream (human_pose.py) and
        # stored in the motion dict. Fail fast if it's missing (stale cache).
        if incam_motion_data.get("predicted_body_height", None) is None:
            raise RuntimeError(
                "predicted_body_height missing from motion data. "
                "Regenerate HMR output (step 1 of grail.pipelines.recon_4dhoi) to populate this field."
            )

        # Load GT character height from rendering metadata
        character_height = None
        for candidate in [
            render_config_file.replace("first_frame_pose.pickle", "character_data.pickle"),
            render_config_file.replace("foundation_pose_output", "foundation_pose").replace(
                "first_frame_pose.pickle", "character_data.pickle"
            ),
        ]:
            character_data = load_character_data(candidate)
            if character_data is not None:
                character_height = character_data.get("character_height", None)
                break

        # Load GT shape params (from pre-fitted files)
        gt_shape_params = self.human_model.load_shape_params(
            self.cfg, self.character_name, self.device
        )

        use_global = self.cfg.get("use_global", False)
        motion_data = self.human_model.transform_global_motion(
            global_motion_data,
            incam_motion_data,
            cam_R=opencv_cam_R,
            cam_t=opencv_cam_t,
            align_frame=inter_start_idx,
            use_global=use_global,
            gt_shape_params=gt_shape_params,
            gt_height=character_height,
            device=self.device,
        )
        motion_data_global_init = self.human_model.transform_global_motion(
            global_motion_data,
            incam_motion_data,
            cam_R=opencv_cam_R,
            cam_t=opencv_cam_t,
            align_frame=inter_start_idx,
            use_global=True,
            gt_shape_params=gt_shape_params,
            gt_height=character_height,
            device=self.device,
        )
        return motion_data, motion_data_global_init, foot_contact_probs

    def _detect_contact_labels(
        self, video_id, frame_num, inter_start_idx, is_static_obj, images_path
    ):
        """Detect or load cached per-interval contact labels.

        Returns:
            (contact_labels, contact_interval, contact_start_idx)
            contact_labels is a list of per-interval label lists, e.g. [["R_Hand"], None, ["L_Hand"]]
        """
        interval = self.cfg.get("contact_interval_length", 8)
        cache_file = os.path.join(self.cache_dir, "contact_labels", f"{video_id}.json")

        # skip VLM contact detection for static objects (e.g. stairs)
        # since the contact loss is zeroed out for them anyway.
        if is_static_obj:
            contact_labels = []
            interval = []
            start_idx = inter_start_idx
            self.logger.info("Skipping contact label detection (is_static_obj=True)")
        elif "contact_labels" in self.cache_list and os.path.exists(cache_file):
            self.logger.info(f"Loading contact labels from cache: {cache_file}")
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            # Support both old and new cache key names
            contact_labels = cache_data.get(
                "contact_labels", cache_data.get("contact_labels_per_interval", [])
            )
            interval = cache_data.get(
                "contact_interval", cache_data.get("contact_interval_length", interval)
            )
            start_idx = cache_data.get(
                "contact_start_idx", cache_data.get("contact_interval_start_idx", inter_start_idx)
            )
        elif self.cfg.get("contact_labels", None) is not None:
            per_interval = self.cfg["contact_labels"]
            n_intervals = max(1, (frame_num - inter_start_idx + interval - 1) // interval)
            contact_labels = [per_interval for _ in range(n_intervals)]
            start_idx = inter_start_idx
            self.logger.info(f"Using contact labels from config: {per_interval}")
        else:
            contact_labels = detect_contact_joints_interval(
                images_path,
                self.obj_name,
                interval_length=interval,
                start_idx=inter_start_idx,
                end_idx=frame_num,
            )
            start_idx = inter_start_idx

            # Validate: ensure at least one interval has labels
            has_any = any(labels is not None and len(labels) > 0 for labels in contact_labels)
            if not has_any:
                contact_labels = [["R_Hand"]]
                self.logger.warning("No contact labels detected, using default: R_Hand")

            self.logger.info(f"Detected contact labels: {contact_labels}")

            # Save to cache
            if "contact_labels" in self.cache_list:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(
                        {
                            "contact_labels": contact_labels,
                            "contact_interval": interval,
                            "contact_start_idx": start_idx,
                        },
                        f,
                        indent=2,
                    )
                self.logger.info(f"Saved contact labels to cache: {cache_file}")

        # backward compatibility with old caches
        for i, entry in enumerate(contact_labels):
            if isinstance(entry, str):
                contact_labels[i] = [entry]

        self.logger.info(
            f"Contact labels ({len(contact_labels)} intervals, interval={interval}): {contact_labels}"
        )
        return contact_labels, interval, start_idx

    def _load_depth(self, video_id):
        """Load preprocessed depth maps from cache."""
        depth_cache_file = os.path.join(self.cache_dir, DEPTH_CACHE_SUBDIR, f"{video_id}.pt")
        if not os.path.exists(depth_cache_file):
            raise FileNotFoundError(
                f"Depth cache not found: {depth_cache_file}. Run preprocessing (step1) first."
            )
        depth_maps = load_depth_from_cache(depth_cache_file, device=self.device)
        self.logger.info(f"Loaded depth from cache: {depth_cache_file}")
        return depth_maps

    def init_params(self, data):
        frame_num = data.frame_num
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=self.device)

        self.num_body_joints = self.human_model.num_body_joints
        self.num_hand_joints = self.human_model.num_hand_joints

        self.params = OptParams(
            human_trans_global=torch.zeros(3, device=self.device, requires_grad=True),
            human_trans_res=torch.zeros(frame_num, 3, device=self.device, requires_grad=True),
            human_pose_res=torch.tensor(
                identity_6d.reshape(1, 1, 6).repeat(frame_num, self.num_body_joints, 1),
                device=self.device,
                requires_grad=True,
            ),
            hand_pose_res=torch.tensor(
                identity_6d.reshape(1, 1, 6).repeat(frame_num, self.num_hand_joints, 1),
                device=self.device,
                requires_grad=True,
            ),
            obj_R_res=torch.tensor(
                identity_6d.unsqueeze(0).repeat(frame_num, 1),
                device=self.device,
                requires_grad=True,
            ),
            obj_t_res=torch.zeros(frame_num, 3, device=self.device, requires_grad=True),
        )
        return self.params

    def get_opt_params(self, params, opt_vars, is_static_obj=False):
        opt_params = {}

        for opt_var in opt_vars:
            if is_static_obj and opt_var in ("obj_R_res", "obj_t_res"):
                self.logger.info(f"Skipping optimization of {opt_var} (static object)")
                continue

            if hasattr(params, opt_var):
                opt_params[opt_var] = getattr(params, opt_var)
            else:
                self.logger.warning(f"Optimization parameter {opt_var} not found")

        return opt_params

    def get_contact_labels_for_frame(self, frame_idx):
        return get_contact_labels_for_frame(
            frame_idx,
            self.contact_labels,
            self.contact_start_idx,
            self.contact_interval,
        )

    def init_opt(self, data, params, opt_config):
        is_static_obj = data.is_static_obj
        opt_params = self.get_opt_params(
            params, opt_config["opt_vars"].keys(), is_static_obj=is_static_obj
        )
        if len(opt_params.keys()) == 0:
            return None, opt_params

        opt_params_cfg = []
        for opt_var in opt_params.keys():
            var_config = opt_config["opt_vars"][opt_var]
            # Check if xy_only is set for human_trans_global
            if opt_var == "human_trans_global" and var_config.get("xy_only", False):
                # Register a gradient hook to zero out z-dimension gradients (1D tensor)
                def xy_only_hook_1d(grad):
                    grad[2] = 0  # Zero out z-dimension gradient
                    return grad

                opt_params[opt_var].register_hook(xy_only_hook_1d)
                opt_params_cfg.append(
                    {
                        "params": opt_params[opt_var],
                        "lr": var_config["lr"],
                    }
                )
                self.logger.info(f"Optimizing {opt_var} with xy_only=True (only x, y dimensions)")
            # Check if xy_only is set for human_trans_res
            elif opt_var == "human_trans_res" and var_config.get("xy_only", False):
                # Register a gradient hook to zero out z-dimension gradients (2D tensor)
                def xy_only_hook_2d(grad):
                    grad[:, 2] = 0  # Zero out z-dimension gradient
                    return grad

                opt_params[opt_var].register_hook(xy_only_hook_2d)
                opt_params_cfg.append(
                    {
                        "params": opt_params[opt_var],
                        "lr": var_config["lr"],
                    }
                )
                self.logger.info(f"Optimizing {opt_var} with xy_only=True (only x, y dimensions)")
            else:
                opt_params_cfg.append(
                    {
                        "params": opt_params[opt_var],
                        "lr": var_config["lr"],
                    }
                )

        optimizer = torch.optim.AdamW(opt_params_cfg)

        return optimizer, opt_params

    def forward(self, data, params):
        frame_num = data.frame_num

        # Extract per-frame parameters
        human_trans_res = params.human_trans_res
        human_pose_res = params.human_pose_res
        hand_pose_res = params.hand_pose_res
        obj_R_res = params.obj_R_res
        obj_t_res = params.obj_t_res

        # Predict human trajectory
        human_trans_global = params.human_trans_global
        human_trans_res = human_trans_res.reshape(frame_num, 1, 3)

        human_pose_res = human_pose_res.reshape(frame_num, self.num_body_joints, 6)
        hand_pose_res = hand_pose_res.reshape(frame_num, self.num_hand_joints, 6)

        motion_data = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in data.human.motion_data.items()
        }

        motion_data = self.human_model.apply_pose_residuals(
            motion_data,
            human_pose_res,
            hand_pose_res,
            frame_num,
        )

        pred_root_pose, pred_body_pose = self.human_model.extract_root_body_pose(motion_data)

        # Apply translation residuals
        trans = motion_data["trans"].reshape(frame_num, 3)
        motion_data["trans"] = (
            trans + human_trans_res.reshape(frame_num, 3) + human_trans_global.reshape(1, 3)
        )

        pred_human_verts_seq, _ = self.human_model.generate_mesh(
            motion_data,
            output_joints=False,
            require_grad=True,
        )

        pred_body_joints_seq = self.human_model.get_body_joints(motion_data, require_grad=True)
        pred_hand_joints_seq = self.human_model.get_hand_joints(motion_data, require_grad=True)

        pred_human_root_trans = motion_data["trans"]

        # Project body and hand joints to 2D screen coordinates
        pred_body_keypoints_seq = project_world_to_screen(
            pred_body_joints_seq.reshape(-1, 3), self.cameras
        ).reshape(frame_num, -1, 3)[:, :, :2]

        pred_hand_keypoints_seq = project_world_to_screen(
            pred_hand_joints_seq.reshape(-1, 3), self.cameras
        ).reshape(frame_num, -1, 3)[:, :, :2]

        pred_human = HOIPrediction.Human(
            trans=pred_human_root_trans,
            root_pose=pred_root_pose,
            pose=pred_body_pose,
            verts_seq=pred_human_verts_seq,
            body_joints_seq=pred_body_joints_seq,
            body_keypoints_seq=pred_body_keypoints_seq,
            hand_joints_seq=pred_hand_joints_seq,
            hand_keypoints_seq=pred_hand_keypoints_seq,
            pose_res=human_pose_res,
            trans_res=human_trans_res,
            motion_data=motion_data,
        )

        # Predict object trajectory
        obj_verts = data.obj.verts
        obj_R_res_mat = rotation_6d_to_matrix(obj_R_res)

        obj_poses = data.obj.poses.clone()

        pred_obj_R = torch.bmm(obj_R_res_mat, obj_poses[:, :3, :3])
        pred_obj_t = obj_poses[:, :3, 3].reshape(frame_num, 3) + obj_t_res.reshape(frame_num, 3)

        pred_obj_verts_seq = obj_verts.unsqueeze(0).repeat(frame_num, 1, 1)
        pred_obj_verts_seq = torch.bmm(
            pred_obj_verts_seq, pred_obj_R.transpose(1, 2)
        ) + pred_obj_t.reshape(frame_num, 1, 3)

        pred_obj = HOIPrediction.Object(
            trans=pred_obj_t,
            R=pred_obj_R,
            verts_seq=pred_obj_verts_seq,
        )

        return HOIPrediction(human=pred_human, obj=pred_obj)

    def optimize_main(self, data, opt_config):
        """Run optimization for a single stage."""
        optimizer, opt_params = self.init_opt(data, self.params, opt_config)
        if len(opt_params.keys()) == 0:
            self.logger.warning("No optimization parameters found. Skipping optimization...")
            return

        opt_niter = opt_config["niter"]
        loss_cfg = opt_config["loss_cfg"]

        for cur_iter in range(opt_niter):
            optimizer.zero_grad()
            pred = self.forward(data, self.params)
            loss, loss_dict = self.loss_computer.compute_loss(data, pred, loss_cfg)
            loss.backward()
            optimizer.step()

            self.write_logs(cur_iter, loss_dict, opt_config)

        self.logger.info(
            f"Human pelvis after optimization stage: {pred.human.body_joints_seq[0, 0, :]}"
        )

    def optimize(self, data):
        print("Starting HOI optimization...")
        # Validate that data has been initialized
        if not hasattr(self, "obj_mesh") or self.obj_mesh is None:
            raise ValueError("Data not initialized. Call init_data() first.")

        if self.enable_vis:
            self.visualizer = HOIVisualizer(
                device=self.device,
                human_model=self.human_model,
                cameras=self.cameras,
                image_list=self.image_list,
                video_fps=self.video_fps,
                log_dir=self.log_dir,
                obj_path=self.obj_path,
            )
            if self.vis_cfg.get("vis_html", False):
                self.visualizer.sp_visualizer = ScenepicVisualizer()
            self.visualizer.init_vis_meshes(data)

        pre_eval_pass, failed_frame = pre_eval(
            data,
            self.cameras,
            self.pre_eval_cfg,
            self.min_frames_threshold,
            self.device,
            self.logger,
        )
        if failed_frame is not None:
            truncate_data(data, failed_frame, self.logger)
            self.image_list = data.images_path
            self.depth_list = data.depth_maps

        # Initialize params after pre_eval (in case data was truncated)
        self.init_params(data)

        # Create loss computer after init_params (needs self.num_body_joints)
        self.loss_computer = LossComputer(
            cameras=self.cameras,
            human_model=self.human_model,
            device=self.device,
            get_contact_labels_for_frame_fn=self.get_contact_labels_for_frame,
            num_body_joints=self.num_body_joints,
            logger=self.logger,
        )

        if self.enable_vis:
            if self.vis_cfg.get("vis_init", False):
                pred = self.forward(data, self.params)
                hoi_data = self.get_optimized_data(data, pred, to_numpy=False)
                self.visualizer.visualize(data, pred, hoi_data, "init", self.vis_cfg)

        if pre_eval_pass:
            for stage, stage_specs in self.opt_stage_specs.items():
                stage_specs["stage"] = stage
                self.optimize_main(data, stage_specs)

                if self.enable_vis:
                    pred = self.forward(data, self.params)
                    hoi_data = self.get_optimized_data(data, pred, to_numpy=False)
                    self.visualizer.visualize(data, pred, hoi_data, stage, self.vis_cfg)
        else:
            print("Pre-evaluation failed. Skipping optimization...")

        pred = self.forward(data, self.params)
        optimized_data = self.get_optimized_data(data, pred)
        eval_data = self.eval(data, pred, self.eval_cfg)
        optimized_data["eval_data"] = eval_data

        return optimized_data

    @torch.no_grad()
    def eval(self, data, pred, eval_cfg):
        eval_data = {}
        _, eval_data = self.loss_computer.compute_loss(data, pred, eval_cfg)
        eval_data = tensor_to_numpy(eval_data)
        return eval_data

    def get_optimized_data(self, data, pred=None, to_numpy=True, smooth=True):
        if pred is None:
            pred = self.forward(data, self.params)

        def smooth_results(human_data, obj_R, obj_t):
            """Apply Savitzky-Golay smoothing to human motion and object trajectory."""
            self.logger.info("Applying smoothing to human motion...")
            frame_num = human_data["poses"].shape[0]

            human_data["poses"] = smooth_axis_angle_sequence(
                human_data["poses"].reshape(frame_num, -1, 3),
                window_length=11,
                polyorder=2,
            ).reshape(frame_num, -1)
            human_data["trans"] = smooth_pose_sequence(
                human_data["trans"], window_length=11, polyorder=2
            )
            for hand_key in ("left_hand_pose", "right_hand_pose"):
                if hand_key in human_data:
                    human_data[hand_key] = smooth_axis_angle_sequence(
                        human_data[hand_key].reshape(frame_num, -1, 3),
                        window_length=11,
                        polyorder=2,
                    ).reshape(frame_num, -1)

            self.logger.info("Applying smoothing to object trajectory...")
            obj_R = axis_angle_to_matrix(
                smooth_axis_angle_sequence(
                    matrix_to_axis_angle(obj_R).unsqueeze(1), window_length=11, polyorder=2
                ).squeeze(1)
            )
            obj_t = smooth_pose_sequence(obj_t, window_length=11, polyorder=2)
            return human_data, obj_R, obj_t

        human_data = pred.human.motion_data
        obj_R = pred.obj.R
        obj_t = pred.obj.trans

        if smooth:
            human_data, obj_R, obj_t = smooth_results(human_data, obj_R, obj_t)

        optimized_data = {
            "human_data": tensor_to_numpy(human_data) if to_numpy else human_data,
            "obj_data": (
                tensor_to_numpy(
                    {
                        "obj_R": obj_R,
                        "obj_t": obj_t,
                        "obj_scale": data.obj.scale,
                    }
                )
                if to_numpy
                else {"obj_R": obj_R, "obj_t": obj_t, "obj_scale": data.obj.scale}
            ),
            "meta": {
                "inter_start_idx": data.inter_start_idx,
                "inter_end_idx": data.inter_end_idx,
            },
        }

        if data.static_objects is not None:
            optimized_data["scene_data"] = tensor_to_numpy(data.static_objects)

        return optimized_data

    def write_logs(self, cur_iter, loss_dict, opt_config):
        opt_niters = opt_config["niter"]
        loss_str = " | ".join([f"{x}: {y:7.3f}" for x, y in loss_dict.items()])
        head_str = f'{self.cfg["exp_name"]} - {opt_config["stage"]}'
        info_str = f"{head_str} | {cur_iter:4d}/{opt_niters} | {loss_str}"

        self.logger.info(info_str)
