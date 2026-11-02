"""Centralized path building for GRAIL pipelines.

Instead of scattered f-string path construction across modules, use these
helpers to build artifact paths consistently.
"""

from __future__ import annotations

import os

from grail.core.depth_config import DEPTH_CACHE_SUBDIR

# ============================================================================
# 4D Reconstruction Paths
# ============================================================================


def video_path(results_dir: str, video_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, video_dir, f"{video_id}.mp4")


def hmr_output(results_dir: str, hmr_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, hmr_dir, f"{video_id}.npz")


def hmr_cache(results_dir: str, hmr_cache_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, hmr_cache_dir, video_id)


def masks_cache(results_dir: str, recon_cache_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, recon_cache_dir, "masks", f"{video_id}.npz")


def depth_cache(results_dir: str, recon_cache_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, recon_cache_dir, DEPTH_CACHE_SUBDIR, f"{video_id}.pt")


def foundation_pose_input(results_dir: str, fp_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, fp_dir, video_id)


def foundation_pose_output(results_dir: str, fp_output_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, fp_output_dir, video_id)


def foundation_pose_poses(results_dir: str, fp_output_dir: str, video_id: str) -> str:
    return os.path.join(
        results_dir, fp_output_dir, video_id, "pose_estimation_output", "poses_in_cam.pkl"
    )


def render_config(results_dir: str, fp_output_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, fp_output_dir, video_id, "first_frame_pose.pickle")


def first_frame_mask(results_dir: str, fp_dir: str, video_id: str, mask_type: str = "obj") -> str:
    subdir = "masks" if mask_type == "obj" else "human_masks"
    return os.path.join(results_dir, fp_dir, video_id, subdir, "000000.png")


def cam_intrinsics(results_dir: str, fp_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, fp_dir, video_id, "cam_K.txt")


def recon_output(results_dir: str, output_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, output_dir, video_id)


def hoi_data_file(results_dir: str, output_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, output_dir, video_id, "hoi_data.pkl")


def valid_output(results_dir: str, output_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, f"{output_dir}_valid", video_id)


def mesh_dir(results_dir: str, dataset: str, category: str) -> str:
    return os.path.join(results_dir, "generation", "mesh", dataset, category)


def depth_gt(results_dir: str, depth_gt_dir: str, video_id: str) -> str:
    return os.path.join(results_dir, depth_gt_dir, video_id, "000000.png")


def video_frames_dir(
    results_dir: str, video_dir: str, dataset: str, category: str, video_id: str
) -> str:
    video_name = os.path.basename(video_id)
    return os.path.join(results_dir, video_dir, dataset, category, "frames", video_name)
