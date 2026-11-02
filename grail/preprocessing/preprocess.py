#!/usr/bin/env python3
"""
Preprocessing Functions for 4D HOI Reconstruction

This module provides preprocessing functions for:
1. Segmentation mask tracking using SAM2
2. Depth estimation
3. Depth alignment with ground truth

These functions are called from grail.pipelines.recon_4dhoi.
"""

import os
import sys
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
from grail.adapters.depth import est_depth
from grail.adapters.sam import get_bbox_from_mask, track_masks, track_masks_from_bbox
from grail.core.depth_config import (
    depth_input_bindings,
    preserve_depth_cache,
    read_depth_cache_metadata,
    write_depth_cache_metadata,
)
from grail.core.video import extract_frames_from_video, save_images_to_video


def preprocess_masks(
    video_path,
    first_frame_obj_mask_path,
    first_frame_human_mask_path,
    cache_file,
    device="cuda",
    debug_dir=None,
):
    """
    Track segmentation masks through the video using SAM2.

    Args:
        video_path: Path to the video file
        first_frame_obj_mask_path: Path to first frame object mask from Blender
        first_frame_human_mask_path: Path to first frame human mask from Blender
        cache_file: Path to save the masks cache (.npz)
        device: Device to run inference on
        debug_dir: Directory to save debug visualizations (optional)

    Returns:
        dict: Video masks dictionary mapping frame_idx -> obj_id -> binary_mask
              obj_id 0 = object mask, obj_id 1 = human mask
    """
    # Load first frame masks
    if not os.path.exists(first_frame_obj_mask_path):
        raise FileNotFoundError(f"Object mask not found at {first_frame_obj_mask_path}")
    if not os.path.exists(first_frame_human_mask_path):
        raise FileNotFoundError(f"Human mask not found at {first_frame_human_mask_path}")

    first_frame_obj_mask = cv2.imread(first_frame_obj_mask_path, -1)
    first_frame_human_mask = cv2.imread(first_frame_human_mask_path, -1)

    # Convert to grayscale if needed
    if len(first_frame_obj_mask.shape) == 3:
        first_frame_obj_mask = cv2.cvtColor(first_frame_obj_mask, cv2.COLOR_RGB2GRAY)
    if len(first_frame_human_mask.shape) == 3:
        first_frame_human_mask = cv2.cvtColor(first_frame_human_mask, cv2.COLOR_RGB2GRAY)

    # Binarize masks
    first_frame_obj_mask = (first_frame_obj_mask > 0).astype(np.uint8)
    first_frame_human_mask = (first_frame_human_mask > 0).astype(np.uint8)

    # Extract RGB frames for SAM2
    basename = os.path.basename(video_path)
    temp_rgb_dir = os.path.join(os.path.dirname(cache_file), f"{basename}_frames_temp")
    os.makedirs(temp_rgb_dir, exist_ok=True)
    frame_count = extract_frames_from_video(video_path, temp_rgb_dir, image_format="jpg")

    print(f"Tracking masks through video using SAM2 ({frame_count} frames)...")

    # Track masks through the video
    # Object mask = obj_id 0, Human mask = obj_id 1
    video_masks = track_masks(
        [first_frame_obj_mask, first_frame_human_mask], temp_rgb_dir, device=device, frame_idx=0
    )

    # Save masks to cache (compressed to reduce file size)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, masks=video_masks)
    print(f"Saved masks to cache: {cache_file}")

    # Remove temporary RGB frames
    import shutil

    shutil.rmtree(temp_rgb_dir)

    # Save debug visualization if requested
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        obj_masks = [video_masks[i][0].squeeze() for i in range(frame_count)]
        human_masks = [video_masks[i][1].squeeze() for i in range(frame_count)]
        save_images_to_video(obj_masks, os.path.join(debug_dir, "obj_masks.mp4"))
        save_images_to_video(human_masks, os.path.join(debug_dir, "human_masks.mp4"))
        print(f"Saved mask debug videos to: {debug_dir}")

    return video_masks


def preprocess_depth(
    image_list,
    cache_file,
    gt_depth_path=None,
    video_masks=None,
    first_frame_obj_mask=None,
    first_frame_human_mask=None,
    intrinsics=None,
    device="cuda",
    source_video_path=None,
    input_paths=None,
):
    """
    Estimate per-frame metric depth with MoGe-3, optionally aligning to
    ground-truth depth from Blender rendering.

    Args:
        image_list: List of image paths (frames extracted from video)
        cache_file: Path to save the depth cache (.pt)
        gt_depth_path: Path to ground truth depth PNG from Blender (optional)
        video_masks: Video masks dict from SAM2 (frame_idx -> obj_id -> mask)
        first_frame_obj_mask: First frame object mask from Blender (numpy array)
        first_frame_human_mask: First frame human mask from Blender (numpy array)
        intrinsics: Camera intrinsics (3, 3) numpy array or path to cam_K.txt (optional)
        device: Device to run inference on
        source_video_path: Original video path recorded in cache provenance (optional)
        input_paths: Selected source/calibration paths to bind for safe cache reuse (optional)

    Returns:
        list: List of depth tensors for each frame
    """
    if not image_list:
        raise ValueError("Cannot estimate depth for an empty frame list")
    inputs = depth_input_bindings(input_paths) if input_paths is not None else None
    print(f"Estimating depth for {len(image_list)} frames using MoGe-3...")

    # Load ground truth depth for first frame if available
    gt_depth_first_frame = None
    if gt_depth_path is not None and os.path.exists(gt_depth_path):
        gt_depth_mm = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        if gt_depth_mm is not None:
            # Convert from mm to meters
            gt_depth_first_frame = gt_depth_mm.astype(np.float32) / 1000.0
            print(f"Loaded GT depth for first frame from: {gt_depth_path}")

    # Estimate depth (MoGe, per-frame, metric)
    depth_list, inference_metadata = est_depth(
        image_list,
        device=device,
        intrinsics=intrinsics,
        gt_depth_first_frame=gt_depth_first_frame,
        return_metadata=True,
    )

    # Align depth with ground truth (per-frame), preserving the existing method.
    alignment = {
        "status": "not_requested" if gt_depth_path is None else "gt_unavailable",
        "method": "per_frame_background_linear_depth_scale_shift",
        "gt_depth_path": os.fspath(gt_depth_path) if gt_depth_path is not None else None,
        "frames": [],
    }
    if gt_depth_first_frame is not None:
        print(f"Aligning depth with ground truth (per-frame): {gt_depth_path}")
        depth_list, alignment = align_depth_with_gt(
            depth_list,
            gt_depth_path,
            video_masks=video_masks,
            first_frame_obj_mask=first_frame_obj_mask,
            first_frame_human_mask=first_frame_human_mask,
            device=device,
            return_metadata=True,
        )

    if len(depth_list) != len(image_list):
        raise ValueError("Depth inference returned a different frame count than the source")
    if isinstance(intrinsics, (str, os.PathLike)):
        camera_matrix = np.loadtxt(intrinsics).reshape(3, 3).tolist()
        intrinsics_path = os.fspath(intrinsics)
    elif intrinsics is not None:
        camera_matrix = np.asarray(intrinsics).reshape(3, 3).tolist()
        intrinsics_path = None
    else:
        camera_matrix = None
        intrinsics_path = None

    # Publish the sidecar last: a crash cannot leave an apparently current cache.
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    if inputs is not None and depth_input_bindings(input_paths) != inputs:
        raise ValueError("Depth source/calibration inputs changed during inference")
    temporary_cache = f"{cache_file}.{os.getpid()}.tmp"
    try:
        torch.save(depth_list, temporary_cache)
        archive = preserve_depth_cache(cache_file)
        if archive is not None:
            print(f"  Preserved previous depth cache: {archive}")
        os.replace(temporary_cache, cache_file)
    finally:
        if os.path.exists(temporary_cache):
            os.remove(temporary_cache)
    write_depth_cache_metadata(
        cache_file,
        {
            "frame_count": len(image_list),
            "input_bindings": inputs,
            "source_video_path": os.fspath(source_video_path) if source_video_path else None,
            "source_frames": [os.fspath(path) for path in image_list],
            "intrinsics": camera_matrix,
            "intrinsics_path": intrinsics_path,
            "gt_calibration": alignment,
            "inference": inference_metadata,
        },
    )
    print(f"Saved depth to cache: {cache_file}")

    return depth_list


def align_depth_with_gt(
    depth_list,
    gt_depth_path,
    video_masks=None,
    first_frame_obj_mask=None,
    first_frame_human_mask=None,
    device="cuda",
    return_metadata=False,
):
    """
    Align estimated depth with ground truth depth using scale and shift alignment.
    Uses only the BACKGROUND area (excluding human and object) for alignment.
    Computes scale/shift per-frame since the camera is static and background is consistent.

    The background mask is computed as:
    1. Get combined human+object mask from GT frame (first_frame_obj_mask + first_frame_human_mask)
    2. For each frame i, get combined human+object mask from video_masks
    3. Create combined foreground mask (union of GT mask and frame i mask)
    4. Use the background (inverse of combined mask) for depth alignment

    Args:
        depth_list: List of estimated depth tensors
        gt_depth_path: Path to ground truth depth PNG (16-bit, millimeter scale)
        video_masks: Video masks dict from SAM2 (frame_idx -> obj_id -> mask)
        first_frame_obj_mask: First frame object mask from Blender (numpy array, binary)
        first_frame_human_mask: First frame human mask from Blender (numpy array, binary)
        device: Device for computation
        return_metadata: Also return per-frame calibration provenance when True

    Returns:
        list: Aligned depth tensors, or (list, metadata) if return_metadata=True
    """
    metadata = {
        "status": "gt_unavailable",
        "method": "per_frame_background_linear_depth_scale_shift",
        "gt_depth_path": os.fspath(gt_depth_path),
        "first_frame_obj_mask_supplied": first_frame_obj_mask is not None,
        "first_frame_human_mask_supplied": first_frame_human_mask is not None,
        "video_masks_supplied": video_masks is not None,
        "frames": [],
    }
    # Load ground truth depth (16-bit PNG, millimeter scale)
    gt_depth_mm = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
    if gt_depth_mm is None:
        print(f"Warning: Could not read ground truth depth from {gt_depth_path}")
        return (depth_list, metadata) if return_metadata else depth_list

    # Convert to meters
    gt_depth = gt_depth_mm.astype(np.float32) / 1000.0
    gt_depth = torch.from_numpy(gt_depth).to(device)

    # Get reference shape from first frame
    ref_depth = depth_list[0]

    # Resize ground truth to match estimated depth if needed
    if gt_depth.shape != ref_depth.shape:
        gt_depth = torch.nn.functional.interpolate(
            gt_depth.unsqueeze(0).unsqueeze(0),
            size=ref_depth.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    # Create GT foreground mask (combined human + object from Blender)
    gt_foreground_mask = None
    if first_frame_obj_mask is not None and first_frame_human_mask is not None:
        # Combine human and object masks
        gt_foreground = (first_frame_obj_mask > 0) | (first_frame_human_mask > 0)
        gt_foreground = gt_foreground.astype(np.float32)
        gt_foreground_mask = torch.from_numpy(gt_foreground).to(device)

        # Resize to match depth if needed
        if gt_foreground_mask.shape != ref_depth.shape:
            gt_foreground_mask = torch.nn.functional.interpolate(
                gt_foreground_mask.unsqueeze(0).unsqueeze(0), size=ref_depth.shape, mode="nearest"
            ).squeeze()

    # Align each frame independently
    aligned_depth_list = []
    print(f"Aligning depth for {len(depth_list)} frames (per-frame scale/shift)...")

    for frame_idx, est_depth in enumerate(tqdm(depth_list, desc="Aligning depth")):
        # Get foreground mask for this frame from video_masks (SAM2 tracked)
        frame_foreground_mask = None
        if video_masks is not None and frame_idx in video_masks:
            # obj_id 0 = object, obj_id 1 = human
            obj_mask = video_masks[frame_idx].get(0, np.zeros_like(ref_depth.cpu().numpy()))
            human_mask = video_masks[frame_idx].get(1, np.zeros_like(ref_depth.cpu().numpy()))

            # Handle different mask formats
            if isinstance(obj_mask, np.ndarray):
                obj_mask = obj_mask.squeeze()
            if isinstance(human_mask, np.ndarray):
                human_mask = human_mask.squeeze()

            frame_foreground = (obj_mask > 0) | (human_mask > 0)
            frame_foreground = frame_foreground.astype(np.float32)
            frame_foreground_mask = torch.from_numpy(frame_foreground).to(device)

            # Resize to match depth if needed
            if frame_foreground_mask.shape != ref_depth.shape:
                frame_foreground_mask = torch.nn.functional.interpolate(
                    frame_foreground_mask.unsqueeze(0).unsqueeze(0),
                    size=ref_depth.shape,
                    mode="nearest",
                ).squeeze()

        # Combine GT and frame foreground masks to get combined foreground
        combined_foreground_mask = None
        if gt_foreground_mask is not None and frame_foreground_mask is not None:
            combined_foreground_mask = (gt_foreground_mask > 0.5) | (frame_foreground_mask > 0.5)
        elif gt_foreground_mask is not None:
            combined_foreground_mask = gt_foreground_mask > 0.5
        elif frame_foreground_mask is not None:
            combined_foreground_mask = frame_foreground_mask > 0.5

        # Create valid mask for background only
        # Background = NOT foreground, valid depth values
        prediction_valid = torch.isfinite(est_depth) & (est_depth > 0)
        valid_mask = (gt_depth > 0.1) & (gt_depth < 65.0) & (est_depth > 0.1) & prediction_valid

        if combined_foreground_mask is not None:
            background_mask = ~combined_foreground_mask
            valid_mask = valid_mask & background_mask

        if valid_mask.sum() < 100:
            # Not enough valid pixels, use identity transform
            print(
                f"Warning: Frame {frame_idx} - not enough valid background pixels, skipping alignment"
            )
            aligned_depth_list.append(torch.where(prediction_valid, est_depth, 0.0))
            metadata["frames"].append(
                {
                    "frame_idx": frame_idx,
                    "status": "insufficient_valid_background",
                    "valid_pixels": int(valid_mask.sum().item()),
                    "scale": 1.0,
                    "shift": 0.0,
                    "foreground_excluded": combined_foreground_mask is not None,
                }
            )
            continue

        # Compute scale and shift using least squares on background only
        # depth_aligned = scale * depth_est + shift
        gt_valid = gt_depth[valid_mask]
        est_valid = est_depth[valid_mask]

        # Solve for scale and shift: minimize ||scale * est + shift - gt||^2
        A = torch.stack([est_valid, torch.ones_like(est_valid)], dim=1)
        b = gt_valid

        # Solve using least squares
        solution = torch.linalg.lstsq(A, b).solution
        scale = solution[0].item()
        shift = solution[1].item()

        # Apply alignment
        aligned_depth = scale * est_depth + shift
        aligned_depth = torch.clamp(aligned_depth, min=0.0)  # Ensure non-negative
        # An affine shift must not invent geometry at model-invalid pixels.
        aligned_depth = torch.where(prediction_valid, aligned_depth, 0.0)
        aligned_depth_list.append(aligned_depth)
        metadata["frames"].append(
            {
                "frame_idx": frame_idx,
                "status": "aligned",
                "valid_pixels": int(valid_mask.sum().item()),
                "scale": scale,
                "shift": shift,
                "foreground_excluded": combined_foreground_mask is not None,
            }
        )

        # Print stats for first and last frame
        if frame_idx == 0 or frame_idx == len(depth_list) - 1:
            print(
                f"Frame {frame_idx}: scale={scale:.4f}, shift={shift:.4f}, bg_pixels={valid_mask.sum().item()}"
            )

    aligned_count = sum(frame["status"] == "aligned" for frame in metadata["frames"])
    metadata["status"] = (
        "aligned" if aligned_count == len(depth_list) else "partial" if aligned_count else "skipped"
    )
    return (aligned_depth_list, metadata) if return_metadata else aligned_depth_list


def load_masks_from_cache(cache_file):
    """
    Load masks from cache file.

    Args:
        cache_file: Path to the masks cache (.npz)

    Returns:
        dict: Video masks dictionary
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Masks cache not found: {cache_file}")

    masks = np.load(cache_file, allow_pickle=True)["masks"].item()
    print(f"Loaded masks from cache: {cache_file}")
    return masks


def load_depth_from_cache(cache_file, device="cuda", *, metadata_binding=None):
    """
    Load depth from cache file.

    Args:
        cache_file: Path to the depth cache (.pt)
        device: Device to load tensors to
        metadata_binding: Optional inventory-bound {path, sha256, bytes} for
            relocated provenance; defaults to the adjacent .json sidecar.

    Returns:
        list: List of depth tensors
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Depth cache not found: {cache_file}")

    metadata = read_depth_cache_metadata(cache_file, metadata_binding=metadata_binding)
    depth_list = torch.load(cache_file, map_location=device, weights_only=True)
    if len(depth_list) != metadata["frame_count"]:
        raise ValueError(f"Depth cache frame count does not match provenance: {cache_file}")
    print(f"Loaded depth from cache: {cache_file}")
    return depth_list


def load_camera_intrinsics(cam_K_path):
    """
    Load camera intrinsic matrix from cam_K.txt file.

    Args:
        cam_K_path: Path to the cam_K.txt file

    Returns:
        np.ndarray: 3x3 camera intrinsic matrix K
    """
    if not os.path.exists(cam_K_path):
        raise FileNotFoundError(f"Camera intrinsics not found: {cam_K_path}")

    K = np.loadtxt(cam_K_path).reshape(3, 3)
    return K


def depth_to_point_cloud(depth, K, rgb=None, max_depth=65.0, stride=1):
    """
    Convert depth map to 3D point cloud.

    Args:
        depth: Depth map tensor or numpy array (H, W) in meters
        K: Camera intrinsic matrix (3, 3)
        rgb: Optional RGB image for coloring (H, W, 3) as uint8
        max_depth: Maximum depth threshold in meters
        stride: Sampling stride (use 1 pixel every `stride` pixels). Default 1 = no sampling.

    Returns:
        points: (N, 3) numpy array of 3D points
        colors: (N, 3) numpy array of RGB colors (0-255) if rgb provided, else None
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()

    H, W = depth.shape

    # Get intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel coordinate grids with stride
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    u, v = np.meshgrid(u, v)

    # Sample depth and rgb with stride
    depth_sampled = depth[::stride, ::stride]
    rgb_sampled = rgb[::stride, ::stride] if rgb is not None else None

    # Valid depth mask
    valid_mask = (depth_sampled > 0.1) & (depth_sampled < max_depth)

    # Back-project to 3D
    z = depth_sampled[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)

    # Get colors if RGB provided
    colors = None
    if rgb_sampled is not None:
        if len(rgb_sampled.shape) == 3 and rgb_sampled.shape[2] == 3:
            colors = rgb_sampled[valid_mask]

    return points, colors


def save_point_cloud_ply(points, colors, save_path):
    """
    Save point cloud to PLY file.

    Args:
        points: (N, 3) numpy array of 3D points
        colors: (N, 3) numpy array of RGB colors (0-255), or None
        save_path: Path to save the .ply file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    N = points.shape[0]
    has_colors = colors is not None and len(colors) == N

    with open(save_path, "w") as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write vertices
        for i in range(N):
            if has_colors:
                f.write(
                    f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n"
                )
            else:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")

    print(f"Saved point cloud to: {save_path} ({N} points)")


def visualize_depth_as_point_cloud(
    depth_list,
    image_list,
    cam_K_path,
    output_dir,
    frame_indices=None,
    max_depth=65.0,
    stride=8,
):
    """
    Visualize depth maps as colored point clouds (.ply files).

    Args:
        depth_list: List of depth tensors (each H, W)
        image_list: List of RGB image paths
        cam_K_path: Path to camera intrinsics (cam_K.txt)
        output_dir: Directory to save .ply files
        frame_indices: List of frame indices to visualize (default: [0] for first frame only)
        max_depth: Maximum depth threshold in meters
        stride: Sampling stride (use 1 pixel every `stride` pixels). Default 8 to reduce file size.
    """
    # Load camera intrinsics
    K = load_camera_intrinsics(cam_K_path)

    os.makedirs(output_dir, exist_ok=True)

    # Default to first frame only
    if frame_indices is None:
        frame_indices = [0]

    for idx in frame_indices:
        if idx >= len(depth_list) or idx >= len(image_list):
            print(f"Warning: Frame index {idx} out of range, skipping")
            continue

        depth = depth_list[idx]
        rgb_path = image_list[idx]

        # Load RGB image
        rgb = cv2.imread(rgb_path)
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Resize RGB to match depth if needed
            if isinstance(depth, torch.Tensor):
                depth_shape = depth.shape
            else:
                depth_shape = depth.shape

            if rgb.shape[:2] != depth_shape:
                rgb = cv2.resize(rgb, (depth_shape[1], depth_shape[0]))

        # Convert to point cloud with sampling
        points, colors = depth_to_point_cloud(depth, K, rgb, max_depth=max_depth, stride=stride)

        # Save as PLY
        ply_path = os.path.join(output_dir, f"depth_{idx:06d}.ply")
        save_point_cloud_ply(points, colors, ply_path)
