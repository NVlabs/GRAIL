#!/usr/bin/env python3
"""Object pose estimation orchestration.

Defaults to MoSE3 + FoundationPose and smoothing on the original timeline.
The explicit FoundationPose backend retains optional crop/interpolation controls.
"""

import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import torch

from grail.constants.image import HEIGHT, WIDTH
from grail.core.io import load_init_rendering_data, run_subprocess, save_object_pose_data
from grail.core.video import (
    extract_frames_from_video,
    get_video_fps_and_frame_count,
    save_images_to_video,
)
from grail.rendering.camera import world_to_camera_matrix

STATIC_OBJECT_MASK_MOTION_THRESHOLD_FRAC = 0.1


@dataclass(frozen=True)
class ObjectMaskMotionResult:
    """Result of comparing tracked object masks against the first frame."""

    is_static: bool
    frame_index: int | None
    new_fraction: float
    vanished_fraction: float
    threshold_fraction: float


class StaticObjectValidationError(RuntimeError):
    """Raised when a required-static source object cannot be validated as static."""


# ---------------------------------------------------------------------------
# Video interpolation & subsampling
# ---------------------------------------------------------------------------


def interpolate_video(input_video_path, output_video_path, interpolation_factor=2):
    """Interpolate video frames using FFmpeg minterpolate. Returns True on success."""
    input_fps, _ = get_video_fps_and_frame_count(input_video_path)
    target_fps = input_fps * interpolation_factor

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-filter:v",
        f"minterpolate='mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:fps={target_fps}'",
        "-pix_fmt",
        "yuv420p",
        "-q:v",
        "2",
        output_video_path,
    ]

    print(f"  Interpolating video {input_fps:.0f}fps → {target_fps:.0f}fps")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FFmpeg interpolation failed: {e.stderr.decode() if e.stderr else e}")
        return False


def _get_sample_indices(total, target):
    """Get evenly spaced indices to pick *target* items from *total*."""
    if target >= total:
        return list(range(total))
    return [round(i * (total - 1) / (target - 1)) for i in range(target)]


def _subsample_files(directory, pattern, target_count):
    """Subsample files matching *pattern* in *directory* to *target_count*."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        return 0

    indices = _get_sample_indices(len(files), target_count)
    ext = os.path.splitext(files[0])[1]

    tmp = os.path.join(directory, "_subsample_tmp")
    os.makedirs(tmp, exist_ok=True)
    for new_i, src_i in enumerate(indices):
        shutil.copy2(files[src_i], os.path.join(tmp, f"{new_i:06d}{ext}"))

    for f in files:
        os.remove(f)
    for f in glob.glob(os.path.join(tmp, f"*{ext}")):
        shutil.move(f, directory)
    shutil.rmtree(tmp)
    return len(indices)


def subsample_pose_output(input_dir, target_frame_count):
    """Subsample pose estimation output to *target_frame_count* frames."""
    pe_dir = os.path.join(input_dir, "pose_estimation_output")

    # Subsample ob_in_cam txt files
    ob_dir = os.path.join(pe_dir, "debug", "ob_in_cam")
    if os.path.isdir(ob_dir):
        _subsample_files(ob_dir, "*.txt", target_frame_count)

    # Subsample track_vis images and regenerate video
    vis_dir = os.path.join(pe_dir, "debug", "track_vis")
    if os.path.isdir(vis_dir):
        n = _subsample_files(vis_dir, "*.png", target_frame_count)
        if n > 0:
            video_out = os.path.join(pe_dir, "pose_estimation_tracking.mp4")
            if os.path.exists(video_out):
                os.remove(video_out)
            from grail.core.video import compile_images_to_video

            compile_images_to_video(vis_dir, video_out, fps=24, image_pattern="*.png")

    # Subsample poses_in_cam.pkl
    pkl_path = os.path.join(pe_dir, "poses_in_cam.pkl")
    if os.path.exists(pkl_path):
        import pickle

        with open(pkl_path, "rb") as f:
            poses = pickle.load(f)
        indices = _get_sample_indices(len(poses), target_frame_count)
        with open(pkl_path, "wb") as f:
            pickle.dump([poses[i] for i in indices], f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Crop utilities
# ---------------------------------------------------------------------------


def determine_crop_bbox_centered(masks, pixel_boundary=50):
    """Compute minimal crop bbox covering all object positions across frames."""
    if not masks:
        raise ValueError("masks list cannot be empty")

    H, W = masks[0].shape[:2]
    x_min, x_max, y_min, y_max = W, 0, H, 0

    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x_min = min(x_min, int(np.min(xs)))
            x_max = max(x_max, int(np.max(xs)))
            y_min = min(y_min, int(np.min(ys)))
            y_max = max(y_max, int(np.max(ys)))

    if x_min >= x_max or y_min >= y_max:
        return (0, 0, W, H)

    return (
        max(0, x_min - pixel_boundary),
        max(0, y_min - pixel_boundary),
        min(W, x_max + pixel_boundary),
        min(H, y_max + pixel_boundary),
    )


def crop_image(image, crop_bbox):
    """Crop image to *crop_bbox* = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = crop_bbox
    return image[y0:y1, x0:x1] if image.ndim == 2 else image[y0:y1, x0:x1, :]


def crop_frames_in_directory(frames_dir, crop_bbox):
    """Crop all image frames in a directory in-place."""
    import cv2

    files = sorted(
        glob.glob(os.path.join(frames_dir, "*.png")) + glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    for p in files:
        cv2.imwrite(p, crop_image(cv2.imread(p, -1), crop_bbox))


def adjust_camera_intrinsics_for_crop(input_dir, crop_bbox):
    """Adjust cam_K.txt principal point for cropping offset."""
    x_start, y_start, _, _ = crop_bbox
    cam_K_path = os.path.join(input_dir, "cam_K.txt")
    if not os.path.exists(cam_K_path):
        return
    K = np.loadtxt(cam_K_path).reshape(3, 3)
    K[0, 2] -= x_start
    K[1, 2] -= y_start
    with open(cam_K_path, "w") as f:
        for row in K:
            f.write(f"{row[0]:.18e} {row[1]:.18e} {row[2]:.18e}\n")


# ---------------------------------------------------------------------------
# Static-object validation
# ---------------------------------------------------------------------------


def check_object_masks_static(
    video_masks,
    threshold_fraction=STATIC_OBJECT_MASK_MOTION_THRESHOLD_FRAC,
):
    """Check whether tracked object masks remain at their first-frame location.

    Pixels that disappear behind the tracked human are ignored. This preserves
    the original FoundationPose fast-path heuristic while making it available
    to callers that require the source object to be static.
    """
    if not video_masks:
        raise ValueError("video_masks cannot be empty")
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be between 0 and 1")

    first_obj_mask = (video_masks[0][0].squeeze() > 0).astype(np.float32)
    object_pixel_count = float(first_obj_mask.sum())
    if object_pixel_count == 0:
        raise ValueError("first-frame object mask is empty")

    max_new_fraction = 0.0
    max_vanished_fraction = 0.0
    for frame_index in range(len(video_masks)):
        object_mask = (video_masks[frame_index][0].squeeze() > 0).astype(np.float32)
        human_mask = (video_masks[frame_index][1].squeeze() > 0).astype(np.float32)

        new_fraction = float((object_mask * (1 - first_obj_mask)).sum() / object_pixel_count)
        vanished_fraction = float(
            (first_obj_mask * (1 - object_mask) * (1 - human_mask)).sum() / object_pixel_count
        )
        max_new_fraction = max(max_new_fraction, new_fraction)
        max_vanished_fraction = max(max_vanished_fraction, vanished_fraction)

        if new_fraction > threshold_fraction or vanished_fraction > threshold_fraction:
            return ObjectMaskMotionResult(
                is_static=False,
                frame_index=frame_index,
                new_fraction=new_fraction,
                vanished_fraction=vanished_fraction,
                threshold_fraction=threshold_fraction,
            )

    return ObjectMaskMotionResult(
        is_static=True,
        frame_index=None,
        new_fraction=max_new_fraction,
        vanished_fraction=max_vanished_fraction,
        threshold_fraction=threshold_fraction,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_obj_pose_est(
    video_path,
    mesh_file,
    input_dir,
    video_masks,
    debug=2,
    device="cuda",
    crop_image=False,
    interpolation_factor=1,
    is_static=False,
    object_tracking_backend="mose3_fp_hybrid",
    object_tracking_config="configs/object_tracking/mose3_fp_hybrid.yaml",
):
    """Run dynamic tracking or the existing validated static-object fast path.

    Args:
        video_path: Path to the video file.
        mesh_file: Path to the object mesh (.obj).
        input_dir: Directory for outputs.
        video_masks: Dict mapping frame_idx → {obj_id → mask}.
        debug: FoundationPose debug level.
        device: Compute device.
        crop_image: Crop to minimal bbox covering all object positions.
        interpolation_factor: Frame interpolation factor (1 = none).
        is_static: Whether the source object is required to be static. Static
            masks use the fast path; dynamic masks run FoundationPose unless
            this requirement is set, in which case they are rejected.
        object_tracking_backend: Default hybrid, or explicit legacy foundationpose.
        object_tracking_config: Hybrid inference, selection and smoothing preset.
    """
    _, original_frame_count = get_video_fps_and_frame_count(video_path)

    # Save mask videos for debugging
    frame_count = len(video_masks)
    obj_masks = [video_masks[i][0].squeeze() for i in range(frame_count)]
    human_masks = [video_masks[i][1].squeeze() for i in range(frame_count)]

    debug_dir = os.path.join(input_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    save_images_to_video(obj_masks, os.path.join(debug_dir, "obj_masks.mp4"))
    save_images_to_video(human_masks, os.path.join(debug_dir, "human_masks.mp4"))

    requires_static_object = is_static
    try:
        mask_motion = check_object_masks_static(video_masks)
    except ValueError as exc:
        if requires_static_object:
            raise StaticObjectValidationError(
                f"Could not validate required-static object: {exc}"
            ) from exc
        raise

    if mask_motion.is_static:
        print(
            "Object-mask motion check: static "
            f"(max_new={mask_motion.new_fraction:.4f}, "
            f"max_vanished={mask_motion.vanished_fraction:.4f}, "
            f"thr={mask_motion.threshold_fraction:.4f})"
        )
    else:
        print(
            "Object-mask motion check: dynamic "
            f"(frame={mask_motion.frame_index}, new={mask_motion.new_fraction:.4f}, "
            f"vanished={mask_motion.vanished_fraction:.4f}, "
            f"thr={mask_motion.threshold_fraction:.4f})"
        )
        if requires_static_object:
            raise StaticObjectValidationError(
                "Required-static object moved in the source video: "
                f"frame={mask_motion.frame_index}, new={mask_motion.new_fraction:.4f}, "
                f"vanished={mask_motion.vanished_fraction:.4f}, "
                f"threshold={mask_motion.threshold_fraction:.4f}"
            )

    is_static = mask_motion.is_static

    if is_static:
        print("Object is static — skipping FoundationPose, generating static poses directly")

        first_frame_file = os.path.join(input_dir, "first_frame_pose.pickle")
        obj_R, obj_t, obj_scale, cam_R, cam_t, render_config = load_init_rendering_data(
            first_frame_file
        )

        world_to_camera_blender = world_to_camera_matrix(
            torch.from_numpy(cam_R).float(), torch.from_numpy(cam_t).float()
        ).numpy()
        blender_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
        world_to_camera_opencv = blender_to_opencv @ world_to_camera_blender

        object_matrix = np.eye(4)
        object_matrix[:3, :3] = obj_R
        object_matrix[:3, 3] = obj_t.reshape(-1)
        ob_in_cam = world_to_camera_opencv @ object_matrix

        pose_list = [ob_in_cam.copy() for _ in range(original_frame_count)]
        poses_output_file = os.path.join(input_dir, "pose_estimation_output", "poses_in_cam.pkl")
        save_object_pose_data(pose_list, poses_output_file)
        print(f"Saved {len(pose_list)} static poses to: {poses_output_file}")
        return True

    if object_tracking_backend == "mose3_fp_hybrid":
        if crop_image or interpolation_factor != 1:
            raise ValueError("Hybrid initialization requires uncropped original-rate source frames")
        masks_file = os.path.join(input_dir, "initial_object_masks.npz")
        if os.path.exists(masks_file):
            raise FileExistsError("Use a fresh scene directory for object initialization")
        np.savez_compressed(masks_file, object_mask0=obj_masks[0], human_mask0=human_masks[0])
        return run_subprocess(
            [
                sys.executable,
                "-m",
                "grail.pose_est.object_initialization",
                "--video",
                os.path.abspath(video_path),
                "--mesh-file",
                os.path.abspath(mesh_file),
                "--scene",
                os.path.abspath(input_dir),
                "--masks",
                masks_file,
                "--preset",
                object_tracking_config,
            ],
            "MoSE3 + FoundationPose initialization and smoothing",
        )
    if object_tracking_backend != "foundationpose":
        raise ValueError(f"Unknown object tracking backend: {object_tracking_backend}")

    # Frame interpolation
    actual_video = video_path
    interp_path = None
    if interpolation_factor > 1:
        interp_path = os.path.join(input_dir, f"interpolated_{interpolation_factor}x.mp4")
        if interpolate_video(video_path, interp_path, interpolation_factor):
            actual_video = interp_path
        else:
            interpolation_factor = 1

    # Crop
    crop_bbox = None
    if crop_image:
        crop_bbox = determine_crop_bbox_centered(obj_masks)
        cw = crop_bbox[2] - crop_bbox[0]
        ch = crop_bbox[3] - crop_bbox[1]
        print(f"  Crop {cw}x{ch} bbox={crop_bbox}")
        adjust_camera_intrinsics_for_crop(input_dir, crop_bbox)

    # Extract frames
    rgb_dir = os.path.join(input_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    extract_frames_from_video(actual_video, rgb_dir, image_format="png")

    if crop_bbox is not None:
        crop_frames_in_directory(rgb_dir, crop_bbox)

    # Run FoundationPose
    adapter_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "adapters", "foundation_pose.py"
    )
    cmd = [
        sys.executable,
        adapter_script,
        "--mesh_file",
        mesh_file,
        "--test_scene_dir",
        input_dir,
        "--debug",
        str(debug),
    ]
    if is_static:
        cmd.append("--is_static")

    success = run_subprocess(cmd, "FoundationPose tracking")

    # Subsample if interpolated
    if interpolation_factor > 1 and success:
        subsample_pose_output(input_dir, target_frame_count=original_frame_count)

    # Cleanup
    if interp_path and os.path.exists(interp_path):
        os.remove(interp_path)

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Object trajectory initialization")
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--test_scene_dir", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--masks-cache", required=True, help="Original SAM mask cache from step 2")
    parser.add_argument(
        "--object-tracking-backend",
        choices=["mose3_fp_hybrid", "foundationpose"],
        default="mose3_fp_hybrid",
    )
    parser.add_argument(
        "--object-tracking-config", default="configs/object_tracking/mose3_fp_hybrid.yaml"
    )
    parser.add_argument("--is-static", action="store_true")
    args = parser.parse_args()

    for p in (args.mesh_file, args.test_scene_dir, args.video):
        if not os.path.exists(p):
            print(f"Error: not found: {p}")
            sys.exit(1)

    from grail.preprocessing.preprocess import load_masks_from_cache

    success = run_obj_pose_est(
        args.video,
        args.mesh_file,
        args.test_scene_dir,
        load_masks_from_cache(args.masks_cache),
        debug=args.debug,
        is_static=args.is_static,
        object_tracking_backend=args.object_tracking_backend,
        object_tracking_config=args.object_tracking_config,
    )
    sys.exit(0 if success else 1)
