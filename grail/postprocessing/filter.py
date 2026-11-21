import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from grail.core.io import load_init_rendering_data, load_mesh, load_object_pose_data
from grail.models.human_model import create_human_model

# Add parent directories to path
from grail.rendering.camera import *
from grail.rendering.camera import (
    cam_pose_blender_to_opencv,
    cam_pose_opencv_to_pytorch3d,
    get_camera,
)
from grail.rendering.renderer import RendererType, create_renderer, render_frame
from grail.rendering.textures import create_colored_meshes
from grail.visualization.utils.vis_utils import prep_visualizer_input


def _log(msg, logger=None):
    """Helper function to log messages to logger if available, otherwise print."""
    if logger:
        logger.info(msg)
    else:
        print(msg)


def filter_hoi_result(result_camera, hoi_data, cfg, device="cuda", logger=None):
    """
    Filter HOI result based on camera translation, object mask alignment, and evaluation metrics.

    Args:
        result_camera: Camera parameters array
        hoi_data: Dictionary containing human_data, obj_data, meta, eval_data
        cfg: Either a dict with 'eval_cfg' and 'filtering_cfg' keys, or a file
             path string (legacy) to load the config from.
        device: PyTorch device
        logger: Optional logger

    Returns:
        tuple: (valid, hoi_data) - (bool indicating validity, potentially truncated hoi_data)
    """
    human_model_cfg = cfg["human_model"]
    eval_cfg = cfg["eval"]
    filtering_cfg = cfg.get("filtering", {})
    camera_trans_thr = filtering_cfg.get("camera_trans_thr", 0.1)
    object_mask_tol = filtering_cfg.get("object_mask_tol", 0.5)
    total_mask_tol = filtering_cfg.get("total_mask_tol", 0.3)
    human_static_thr = filtering_cfg.get("human_static_thr", 0.01)
    min_frames = filtering_cfg.get("min_frames", None)

    valid = True

    # check if camera has large translation
    if result_camera is None:
        valid = True
    else:
        valid = valid and check_camera_translation(result_camera, camera_trans_thr, logger=logger)
    if not valid:
        return False, hoi_data

    data, cameras, mask_renderer = prepare_data_for_validation(
        hoi_data, human_model_cfg, device=device
    )

    # check initial human position close to rendered config
    if data["human_data"]["first_frame_pose"] is not None:
        valid = check_initial_human_position(data, logger=logger)
        if not valid:
            return False, hoi_data

    # check predicted object mask with ground truth object mask
    valid, failed_frame_idx = check_object_mask(
        data, cameras, mask_renderer, device, object_mask_tol, total_mask_tol, logger=logger
    )
    if not valid:
        # If we have min_frames and enough valid frames, truncate and continue
        if (
            min_frames is not None
            and failed_frame_idx is not None
            and failed_frame_idx >= min_frames
        ):
            _log(f"Truncating data to {failed_frame_idx} frames (min_frames={min_frames})", logger)
            hoi_data = truncate_hoi_data(hoi_data, failed_frame_idx)
        else:
            return False, hoi_data

    # check eval data, e.g. loss terms including keypoint tracking, contact, etc.
    valid = check_eval_data(hoi_data["eval_data"], eval_cfg, logger=logger)
    if not valid:
        return False, hoi_data

    # check initial penetration
    valid = check_init_penetration(data, device=device, logger=logger)
    if not valid:
        return False, hoi_data

    # check static human
    valid = check_static_human(data, device=device, threshold=human_static_thr, logger=logger)
    if not valid:
        return False, hoi_data

    # check object initially on the table
    if filtering_cfg.get("check_object_initially_on_table", True):
        valid = check_object_initially_on_table(data, device=device, logger=logger)
        if not valid:
            return False, hoi_data

    # check object motion type (static vs dynamic)
    filter_object_motion = filtering_cfg.get("filter_object_motion", "all")
    if filter_object_motion != "all":
        object_static_thr = filtering_cfg.get("object_static_thr", 0.02)
        is_static, motion_range = check_object_is_static(hoi_data, threshold=object_static_thr)
        motion_label = "static" if is_static else "dynamic"
        _log(
            f"Object motion: {motion_label} (range={motion_range:.4f}m, thr={object_static_thr})",
            logger,
        )
        if filter_object_motion == "static_only" and not is_static:
            _log("Rejected: object is dynamic but static_only filter is active", logger)
            return False, hoi_data
        elif filter_object_motion == "dynamic_only" and is_static:
            _log("Rejected: object is static but dynamic_only filter is active", logger)
            return False, hoi_data

    _log("Valid HOI result!", logger)
    return valid, hoi_data


def check_object_initially_on_table(data, device="cuda", logger=None):
    """
    Check if the object's center (XY) at frame 0 falls within the XY bounding box
    of the table mesh in world space.
    """
    obj_verts_seq = data["obj_data"]["verts_seq"]  # (num_frames, num_verts, 3)
    obj_center_xy = obj_verts_seq[0].mean(dim=0)[:2]  # object center XY at frame 0

    scene_data = data.get("scene_data", None)
    if scene_data is None or "table1" not in scene_data:
        _log("No static objects (table) data available, skipping on-table check", logger)
        return True

    # Compute table XY bounding box from the static object mesh (same transform as hoi_optimizer.py)
    obj_data = scene_data["table1"]
    obj_path = obj_data.get("path", "data/Scene/long_table.obj")
    if os.path.exists(obj_path):
        try:
            static_verts, _, _ = load_mesh(obj_path, mesh_scale=obj_data["scale"], device=device)
            static_rot = torch.from_numpy(obj_data["rot"]).float().to(device)
            static_pos = torch.from_numpy(obj_data["pos"]).float().to(device)
            static_verts = torch.matmul(static_verts, static_rot.T) + static_pos

            table_min_xy = static_verts[:, :2].min(dim=0).values
            table_max_xy = static_verts[:, :2].max(dim=0).values

            within_x = (obj_center_xy[0] >= table_min_xy[0]) and (
                obj_center_xy[0] <= table_max_xy[0]
            )
            within_y = (obj_center_xy[1] >= table_min_xy[1]) and (
                obj_center_xy[1] <= table_max_xy[1]
            )
            valid = within_x and within_y

            _log(
                f"Object center XY: ({obj_center_xy[0].item():.4f}, {obj_center_xy[1].item():.4f}), "
                f"Table XY range: X[{table_min_xy[0].item():.4f}, {table_max_xy[0].item():.4f}], "
                f"Y[{table_min_xy[1].item():.4f}, {table_max_xy[1].item():.4f}]",
                logger,
            )
            if not valid:
                _log("Not valid: object center is outside the table XY bounds", logger)
            else:
                _log("Valid: object center is within the table XY bounds", logger)
            return valid
        except Exception as e:
            _log(f"Failed to load table mesh: {e}", logger)

    _log("Could not compute table XY bounds, skipping on-table check", logger)
    return True


def truncate_hoi_data(hoi_data, num_frames):
    """
    Truncate all sequence data in hoi_data to num_frames.

    Args:
        hoi_data: Dictionary containing human_data, obj_data, meta, eval_data
        num_frames: Number of frames to keep

    Returns:
        Dictionary with truncated data
    """
    truncated_data = {}

    # Truncate human_data
    human_data = hoi_data["human_data"]
    truncated_human_data = {}
    for key, value in human_data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] >= num_frames:
            truncated_human_data[key] = value[:num_frames]
        else:
            truncated_human_data[key] = value
    truncated_data["human_data"] = truncated_human_data

    # Truncate obj_data
    obj_data = hoi_data["obj_data"]
    truncated_obj_data = {}
    for key, value in obj_data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] >= num_frames:
            truncated_obj_data[key] = value[:num_frames]
        else:
            truncated_obj_data[key] = value
    truncated_data["obj_data"] = truncated_obj_data

    # Update meta - adjust inter_end_idx if needed
    meta = hoi_data.get("meta", {}).copy()
    if "inter_end_idx" in meta and meta["inter_end_idx"] > num_frames:
        meta["inter_end_idx"] = num_frames
    truncated_data["meta"] = meta

    # Keep eval_data as-is (aggregated metrics)
    truncated_data["eval_data"] = hoi_data.get("eval_data", {})

    return truncated_data


def check_camera_translation(result_camera, camera_trans_thr=0.1, logger=None):
    """
    Check if the camera has large translation
    """
    camera_start_t = result_camera[0, :3]
    camera_end_t = result_camera[-1, :3]
    camera_shift = np.linalg.norm(camera_end_t - camera_start_t)
    valid = camera_shift < camera_trans_thr
    if not valid:
        _log(
            f"Not valid: camera translation is too large: {camera_shift} > {camera_trans_thr}",
            logger,
        )
    else:
        _log(
            f"Valid: camera translation is within threshold: {camera_shift} < {camera_trans_thr}",
            logger,
        )
    return valid


def prepare_data_for_validation(hoi_data, human_model_cfg, device="cuda"):
    """
    Prepare data needed for validation by loading cameras, masks, generating predictions,
    and other required components. This is a lightweight version of init_data focused only
    on what's needed for validation. Assumes masks are already cached from the optimization process.

    Args:
        hoi_data: Dictionary with hoi_data
        device: PyTorch device (cuda or cpu)

    Returns:
        data: Dictionary containing masks, obj_data
        cameras: Camera setup for rendering
        mask_renderer: Renderer for mask generation
    """
    # Load render config
    _, _, obj_scale, blender_cam_R, blender_cam_t, render_config, additional_data = (
        load_init_rendering_data(
            hoi_data["meta"]["render_config_file"],
            to_tensor=True,
            with_human_data=True,
            device=device,
        )
    )
    frame_height, frame_width, focal_length = render_config
    opencv_cam_R, opencv_cam_t = cam_pose_blender_to_opencv(blender_cam_R, blender_cam_t)
    human_R = additional_data.get("human_R", None)
    human_t = additional_data.get("human_t", None)

    # Setup cameras
    cam_R, cam_t = cam_pose_opencv_to_pytorch3d(opencv_cam_R, opencv_cam_t)
    cameras = get_camera(cam_R, cam_t, focal_length, (frame_height, frame_width), device=device)

    # Load cached masks (assumes they exist from optimization)
    masks_cache_file = hoi_data["meta"]["masks_cache_file"]
    if not os.path.exists(masks_cache_file):
        raise FileNotFoundError(
            f"Masks cache not found: {masks_cache_file}. Masks must be generated during optimization first."
        )
    masks = np.load(masks_cache_file, allow_pickle=True)["masks"].item()

    # Create mask renderer for validation (halved size for speed)
    mask_renderer = create_renderer(
        cameras,
        (frame_height, frame_width),
        renderer_type=RendererType.HARD_PHONG,
        neutral_light=True,
        background_color=[0, 0, 0],
    )

    # Generate human and object vertices sequence
    hoi_data["object_path"] = hoi_data["meta"]["obj_path"]
    human_model = create_human_model(human_model_cfg, device=device)
    motion_seq = prep_visualizer_input(
        hoi_data,
        human_model=human_model,
        normalize_trans=False,
        to_numpy=False,
        simplify_mesh=False,
        device=device,
    )
    human_verts_seq = motion_seq["human_seq"]["vertices"]
    human_joints = motion_seq["human_seq"]["joints_pos"]
    human_faces = motion_seq["human_seq"]["triangles"]
    obj_verts_seq = motion_seq["obj_seq"]["vertices_transformed"]
    obj_faces = motion_seq["obj_seq"]["faces"]

    # Prepare data dictionary
    data = {
        "masks": masks,
        "obj_data": {
            "verts_seq": obj_verts_seq,
            "obj_faces": obj_faces,
        },
        "human_data": {
            "verts_seq": human_verts_seq,
            "joints_seq": human_joints,
            "human_faces": human_faces,
        },
        "scene_data": hoi_data.get("scene_data", None),
    }

    # save human_init_data into hoi_data for later use
    if human_R is not None and human_t is not None:
        data["human_data"]["first_frame_pose"] = {
            "R": human_R,
            "t": human_t,
        }
    else:
        data["human_data"]["first_frame_pose"] = None

    return data, cameras, mask_renderer


def check_initial_human_position(data, logger=None):
    """
    Check if the human is at the initial position
    """
    first_frame_trans_render = data["human_data"]["first_frame_pose"]["t"]
    first_frame_root_recon = data["human_data"]["joints_seq"][0, 0, :]

    init_dist_error = torch.norm(first_frame_trans_render[:2] - first_frame_root_recon[:2])
    valid = init_dist_error < 0.5
    if not valid:
        _log(
            f"Not valid: initial human position error is too large: {init_dist_error} > 0.5m",
            logger,
        )
    else:
        _log(
            f"Valid: initial human position error is within threshold: {init_dist_error} < 0.5m",
            logger,
        )

    return valid


def check_object_mask(data, cameras, mask_renderer, device, tol=0.5, total_tol=0.2, logger=None):
    """
    Check if the optimization was successful by comparing predicted and ground truth object masks.

    Args:
        data: Dictionary containing ground truth data, predictions, masks, and object data
        cameras: Camera setup for rendering
        mask_renderer: Renderer for generating masks
        device: PyTorch device (cuda or cpu)
        tol: Tolerance threshold for mask difference for each frame
        total_tol: Tolerance threshold for total mask difference
        logger: Optional logger for output

    Returns:
        tuple: (bool, int or None) - (True if optimization is successful, failed frame index or None)
    """
    valid = True
    failed_frame_idx = None
    # Check the difference between the gt and pred object mask
    obj_verts_seq = data["obj_data"]["verts_seq"]
    obj_faces = data["obj_data"]["obj_faces"]
    obj_colors = torch.tensor([0.0, 0.0, 1.0], device=device)

    total_err = 0
    for i in range(len(obj_verts_seq)):
        obj_mesh = create_colored_meshes(obj_verts_seq[i], obj_faces, obj_colors)
        _, pred_obj_mask = render_frame(obj_mesh, cameras, mask_renderer, require_grad=False)
        pred_obj_mask = (pred_obj_mask > 0.1).float()

        gt_obj_mask = torch.from_numpy(data["masks"][i][0]).to(device).squeeze(0).float()
        if gt_obj_mask.shape != pred_obj_mask.shape:
            gt_obj_mask = gt_obj_mask.unsqueeze(0).unsqueeze(0)
            gt_obj_mask = F.interpolate(
                gt_obj_mask, size=pred_obj_mask.shape, mode="bilinear", align_corners=False
            )
            gt_obj_mask = gt_obj_mask.squeeze(0).squeeze(0)

        err = ((1 - pred_obj_mask) * gt_obj_mask).sum() / (pred_obj_mask.sum() + 100)
        total_err += err
        if err > tol:
            valid = False
            failed_frame_idx = i
            # import cv2
            # # save the pred and gt object mask
            # pred_obj_mask_path = os.path.join("debug", f"pred_obj_mask_{i}.png")
            # gt_obj_mask_path = os.path.join("debug", f"gt_obj_mask_{i}.png")
            # diff_path = os.path.join("debug", f"diff_{i}.png")
            # cv2.imwrite(pred_obj_mask_path, (pred_obj_mask.cpu().numpy() * 255).astype(np.uint8))
            # cv2.imwrite(gt_obj_mask_path, (gt_obj_mask.cpu().numpy() * 255).astype(np.uint8))
            # diff = (1-pred_obj_mask) * gt_obj_mask
            # cv2.imwrite(diff_path, (diff.cpu().numpy() * 255).astype(np.uint8))
            _log(
                f"Not valid: object mask not aligned at frame {i} / {len(obj_verts_seq)}: {err} > {tol}",
                logger,
            )
            # print("Image saved to {} and {} and {}".format(pred_obj_mask_path, gt_obj_mask_path, diff_path))

            return False, failed_frame_idx

    total_err /= len(obj_verts_seq)
    if total_err > total_tol:
        valid = False
        _log(
            f"Not valid: total object mask difference is too large: {total_err} > {total_tol}",
            logger,
        )
        return valid, failed_frame_idx
    else:
        _log(
            f"Valid: total object mask difference is within threshold: {total_err} < {total_tol}",
            logger,
        )

    _log("Valid: object mask aligned at all frames!", logger)

    return valid, failed_frame_idx


def check_init_penetration(data, device="cuda", threshold=0.02, logger=None):
    """
    Check if human and object are penetrating at the initial frame.

    Args:
        data: Dictionary containing human_data and obj_data with verts_seq
        device: PyTorch device
        threshold: Minimum allowed distance between human and object vertices.
                   If the closest distance is below this, they are considered penetrating.
        logger: Optional logger for output

    Returns:
        bool: True if no significant penetration, False otherwise
    """
    # Get vertices at the initial frame (frame 0)
    human_verts = data["human_data"]["verts_seq"][0]  # (V_human, 3)
    obj_verts = data["obj_data"]["verts_seq"][0]  # (V_obj, 3)

    # Ensure tensors are on the correct device
    if not isinstance(human_verts, torch.Tensor):
        human_verts = torch.from_numpy(human_verts).float().to(device)
    if not isinstance(obj_verts, torch.Tensor):
        obj_verts = torch.from_numpy(obj_verts).float().to(device)

    # Compute pairwise distances between human and object vertices
    # Using broadcasting: (V_human, 1, 3) - (1, V_obj, 3) -> (V_human, V_obj, 3)
    # Then compute norm -> (V_human, V_obj)
    diff = human_verts.unsqueeze(1) - obj_verts.unsqueeze(0)  # (V_human, V_obj, 3)
    distances = torch.norm(diff, dim=-1)  # (V_human, V_obj)

    # Find minimum distance
    min_distance = distances.min().item()

    valid = min_distance >= threshold
    if not valid:
        _log(
            f"Not valid: human-object penetration detected at initial frame. "
            f"Min distance: {min_distance:.4f} < threshold: {threshold}",
            logger,
        )
    else:
        _log(
            f"Valid: no penetration at initial frame. Min distance: {min_distance:.4f} >= threshold: {threshold}",
            logger,
        )

    return valid


def check_static_human(data, device="cuda", threshold=0.01, logger=None):
    """
    Check if the human is too static across the video (indicating a failed reconstruction).

    Args:
        data: Dictionary containing human_data with verts_seq
        device: PyTorch device
        threshold: Minimum required average displacement per frame.
                   If displacement is below this, the human is considered too static.
        logger: Optional logger for output

    Returns:
        bool: True if human has sufficient motion, False if too static
    """
    human_verts_seq = data["human_data"]["verts_seq"]  # (L, V_human, 3)

    # Ensure tensor is on the correct device
    if not isinstance(human_verts_seq, torch.Tensor):
        human_verts_seq = torch.from_numpy(human_verts_seq).float().to(device)

    # Compute per-frame velocity for all vertices
    # velocity[i] = verts[i+1] - verts[i], shape: (L-1, V_human, 3)
    velocities = human_verts_seq[1:] - human_verts_seq[:-1]

    # Compute velocity magnitude for each vertex: (L-1, V_human)
    velocity_magnitudes = torch.norm(velocities, dim=-1)

    # For each frame, find the maximum velocity among all vertices: (L-1,)
    max_velocity_per_frame = velocity_magnitudes.max(dim=1).values

    # Average of maximum velocities across all frames
    avg_max_velocity = max_velocity_per_frame.mean().item()

    valid = avg_max_velocity >= threshold
    if not valid:
        _log(
            f"Not valid: human is too static. "
            f"Avg max velocity: {avg_max_velocity:.4f} < threshold: {threshold}",
            logger,
        )
    else:
        _log(
            f"Valid: human has sufficient motion. "
            f"Avg max velocity: {avg_max_velocity:.4f} >= threshold: {threshold}",
            logger,
        )

    return valid


def check_object_is_static(hoi_data, threshold=0.02):
    """Determine whether the object is approximately static across all frames.
    Returns (is_static, max_range) tuple.
    """
    obj_t = hoi_data["obj_data"]["obj_t"]
    if isinstance(obj_t, torch.Tensor):
        obj_t = obj_t.cpu().numpy()
    t_range = obj_t.max(axis=0) - obj_t.min(axis=0)
    max_range = float(np.linalg.norm(t_range))
    return max_range < threshold, max_range


def check_eval_data(eval_data, eval_cfg, logger=None):
    valid = True

    for eval_name, eval_result in eval_data.items():
        threshold = eval_cfg[eval_name]["threshold"]

        if eval_result > threshold:
            valid = False
            _log(
                f"Not valid: {eval_name} is not within threshold: {eval_result} > {threshold}",
                logger,
            )
            break
        else:
            _log(f"Valid: {eval_name} is within threshold: {eval_result} < {threshold}", logger)

    return valid
