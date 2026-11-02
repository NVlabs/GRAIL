#!/usr/bin/env python3
"""4D Human-Object Interaction Reconstruction Pipeline.

Steps:
    1. Native SOMA03 body/hand prediction (GEM-X, default)
    2. Preprocessing (SAM2 mask tracking + depth estimation)
    3. Object pose estimation (MoSE3/FoundationPose hybrid + smoothing)
    4. 4D HOI optimization with the original contact-label pipeline
    5. Filter & post-process results
    6. Visualize results (ScenePic)
"""

import argparse
import os
import pickle
import shutil
import sys
import time
import traceback
from functools import partial
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from grail.core.config import DEFAULT_RECON_CONFIG, load_recon_config
from grail.core.dataset import category2object
from grail.core.depth_config import DEPTH_CACHE_SUBDIR, depth_cache_is_current
from grail.core.io import load_hoi_data, save_hoi_data, save_human_motion_data, vis_keypoints_data
from grail.core.logging import create_logger
from grail.core.paths import depth_cache as depth_cache_path
from grail.core.torch_utils import tensor_to, tensor_to_numpy
from grail.core.video import (
    concat_videos,
    extract_frames_from_video,
    get_motion_fps,
    get_video_fps_and_frame_count,
)
from grail.optimization.hoi_optimizer import HOIOptimizer
from grail.pose_est.human_pose import run_human_pose_est
from grail.pose_est.object_pose import StaticObjectValidationError, run_obj_pose_est
from grail.postprocessing.filter import filter_hoi_result
from grail.postprocessing.postprocess import post_process_hoi_result
from grail.preprocessing.depth_frames import ensure_depth_frames
from grail.preprocessing.human_crop_masks import (
    ensure_human_crop_masks,
    masks_prepared_this_invocation,
    prepare_pipeline_human_crop,
)
from grail.preprocessing.preprocess import (
    depth_to_point_cloud,
    load_camera_intrinsics,
    load_masks_from_cache,
    preprocess_depth,
    preprocess_masks,
    save_point_cloud_ply,
    visualize_depth_as_point_cloud,
)
from grail.visualization.scenepic import ScenepicVisualizer
from grail.visualization.utils.vis_utils import prep_visualizer_input


def _strip_mp4(video_id):
    return video_id.replace(".mp4", "") if video_id.endswith(".mp4") else video_id


def _origin_id(video_id):
    """Strip -end suffix to get the original video ID."""
    return video_id[: video_id.find("-end")] if "end" in video_id else video_id


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step1_predict_human_motion(video_ids, args):
    """Predict the explicitly selected native model, preserving estimated shape."""
    if args.body_model not in ("soma03", "smplx", "g1_smplx"):
        raise ValueError("Human prediction requires soma03, smplx or g1_smplx")
    output_dir = f"{args.results_dir}/{args.hmr_dir}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/{args.hmr_cache_dir}", exist_ok=True)

    for video_id in tqdm(sorted(video_ids), desc="Step 1 — HMR"):
        try:
            video_id = _strip_mp4(video_id)
            video_path = f"{args.results_dir}/{args.video_dir}/{video_id}.mp4"
            output_npz = f"{output_dir}/{video_id}.npz"

            # The adapter verifies input/model/timing hashes even with skip_done.
            # Keep native caches; an existing NPZ alone is not a validity check.
            cache_dir = f"{args.results_dir}/{args.hmr_cache_dir}/{video_id}"
            model_cfg = dict(getattr(args, "cfg", {}).get("human_model", {}))
            model_cfg["body_model"] = args.body_model
            for key in (
                "human_pose_backend",
                "soma_runtime",
                "gem_x_source_root",
                "gem_x_config_path",
                "gem_x_python",
                "gem_x_device",
                "gem_x_postprocess",
                "gem_x_soma_assets",
                "gem_x_human_crop_repair",
                "gem_x_human_crop_masks",
                "gem_x_human_crop_receipt",
                "gem_x_human_crop_config",
                "static_cam",
                "soma_model_path",
                "soma_identity_params_path",
            ):
                if hasattr(args, key):
                    model_cfg[key] = getattr(args, key)

            model_cfg = prepare_pipeline_human_crop(
                video_path, model_cfg, args, video_id, _origin_id(video_id)
            )

            motion_data = run_human_pose_est(
                video_path,
                model=args.body_model,
                cache_dir=cache_dir,
                human_model_config=model_cfg,
            )
            global_motion = tensor_to_numpy(motion_data["motion_global"])
            incam_motion = tensor_to_numpy(motion_data["motion_incam"])
            save_human_motion_data(global_motion, incam_motion, output_npz)

            vis_keypoints_data(
                video_path,
                incam_motion["vitpose"],
                incam_motion["hand_keypoints_2d"],
                cache_dir,
            )

        except Exception as e:
            print(f"  Error (step1) {video_id}: {e}\n{traceback.format_exc()}")


def step2_preprocess_data(video_ids, args):
    """Step 2: SAM2 mask tracking + depth estimation."""
    os.makedirs(f"{args.results_dir}/{args.recon_cache_dir}", exist_ok=True)

    for video_id in tqdm(sorted(video_ids), desc="Step 2 — Preprocess"):
        try:
            video_id = _strip_mp4(video_id)
            video_id_origin = _origin_id(video_id)
            video_path = f"{args.results_dir}/{args.video_dir}/{video_id}.mp4"
            if not os.path.exists(video_path):
                print(f"  Video not found: {video_path}")
                continue

            masks_cache = f"{args.results_dir}/{args.recon_cache_dir}/masks/{video_id}.npz"
            depth_cache = depth_cache_path(args.results_dir, args.recon_cache_dir, video_id)
            masks_prepared = masks_prepared_this_invocation(args, video_id, masks_cache)
            fp_dir = f"{args.results_dir}/{args.foundation_pose_dir}/{video_id_origin}"
            # Step 1 already validated the masks used by the pose estimator.
            # Standalone Step 2 must enforce the same source/seed binding before
            # calibrating depth, even when an old mask archive already exists.
            if not masks_prepared:
                debug_dir = (
                    f"{args.results_dir}/{args.recon_cache_dir}/masks_debug/{video_id}"
                    if args.verbose
                    else None
                )
                ensure_human_crop_masks(
                    video_path,
                    masks_cache,
                    f"{fp_dir}/masks/000000.png",
                    f"{fp_dir}/human_masks/000000.png",
                    device=args.device,
                    force=not args.skip_done,
                    producer=partial(preprocess_masks, debug_dir=debug_dir),
                )
            gt_depth_path = None
            if args.depth_gt_dir:
                candidate = f"{args.results_dir}/{args.depth_gt_dir}/{video_id}/000000.png"
                if os.path.exists(candidate):
                    gt_depth_path = candidate
            cam_K_path = f"{fp_dir}/cam_K.txt"
            intrinsics = cam_K_path if os.path.exists(cam_K_path) else None
            input_paths = {
                "source_video": video_path,
                "masks": masks_cache,
                "intrinsics": intrinsics,
                "gt_depth": gt_depth_path,
                **{
                    name: path if os.path.exists(path) else None
                    for name, path in (
                        ("rendered_object_mask", f"{fp_dir}/masks/000000.png"),
                        ("rendered_human_mask", f"{fp_dir}/human_masks/000000.png"),
                    )
                },
            }
            depth_done = depth_cache_is_current(depth_cache, input_paths=input_paths)
            if args.skip_done and depth_done:
                continue

            # Estimate depth
            if not depth_done or not args.skip_done:
                video_bn = os.path.basename(video_path).replace(".mp4", "")
                # Keep the lossless MoGe-3 inputs separate from legacy JPEG frames.
                frame_dir = os.path.join(
                    os.path.dirname(video_path), "frames", video_bn, DEPTH_CACHE_SUBDIR
                )
                _, expected_frames = get_video_fps_and_frame_count(video_path)
                image_list = ensure_depth_frames(video_path, frame_dir, expected_frames)

                video_masks = None
                if os.path.exists(masks_cache):
                    video_masks = load_masks_from_cache(masks_cache)

                def _load_mask(path):
                    if not os.path.exists(path):
                        return None
                    m = cv2.imread(path, -1)
                    if len(m.shape) == 3:
                        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
                    return (m > 0).astype(np.uint8)

                first_obj_mask = _load_mask(f"{fp_dir}/masks/000000.png")
                first_human_mask = _load_mask(f"{fp_dir}/human_masks/000000.png")

                depth_list = preprocess_depth(
                    image_list=image_list,
                    cache_file=depth_cache,
                    gt_depth_path=gt_depth_path,
                    video_masks=video_masks,
                    first_frame_obj_mask=first_obj_mask,
                    first_frame_human_mask=first_human_mask,
                    intrinsics=intrinsics,
                    device=args.device,
                    source_video_path=video_path,
                    input_paths=input_paths,
                )

                if args.verbose and intrinsics:
                    depth_debug_dir = f"{args.results_dir}/{args.recon_cache_dir}/{DEPTH_CACHE_SUBDIR}_debug/{video_id}"
                    visualize_depth_as_point_cloud(
                        depth_list=depth_list,
                        image_list=image_list,
                        cam_K_path=cam_K_path,
                        output_dir=depth_debug_dir,
                        frame_indices=list(range(len(depth_list))),
                    )

                    if gt_depth_path and os.path.exists(gt_depth_path):
                        gt_mm = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
                        if gt_mm is not None:
                            gt_m = gt_mm.astype(np.float32) / 1000.0
                            rgb = cv2.imread(image_list[0])
                            if rgb is not None:
                                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                                if rgb.shape[:2] != gt_m.shape:
                                    rgb = cv2.resize(rgb, (gt_m.shape[1], gt_m.shape[0]))
                            K = load_camera_intrinsics(cam_K_path)
                            pts, cols = depth_to_point_cloud(gt_m, K, rgb, stride=8)
                            save_point_cloud_ply(
                                pts, cols, os.path.join(depth_debug_dir, "gt_depth_000000.ply")
                            )

        except Exception as e:
            print(f"  Error (step2) {video_id}: {e}\n{traceback.format_exc()}")


def step3_obj_pose_estimation(video_ids, args):
    """Step 3: default hybrid object initialization or validated static poses."""
    for video_id in tqdm(sorted(video_ids), desc="Step 3 — Object pose"):
        try:
            video_id = _strip_mp4(video_id)
            video_id_origin = _origin_id(video_id)

            out_dir = f"{args.results_dir}/{args.foundation_pose_output_dir}/{video_id}"
            failed_marker = f"{out_dir}/failed"
            recon_dir = Path(args.results_dir) / args.output_dir / video_id
            invalid_marker = recon_dir / "invalid"
            motion_rejection_marker = recon_dir / "unexpected_object_motion"
            valid_dir = Path(f"{args.results_dir}/{args.output_dir}_valid") / video_id
            had_motion_rejection = motion_rejection_marker.exists()
            if args.skip_done and (
                os.path.exists(f"{out_dir}/pose_estimation_output/poses_in_cam.pkl")
                or os.path.exists(failed_marker)
            ):
                continue

            video_file = f"{args.results_dir}/{args.video_dir}/{video_id}.mp4"
            dataset, category = video_id.split("/")[:2]
            parent_dir = os.path.dirname(f"{args.results_dir}/{args.foundation_pose_dir}")
            mesh_files = glob(f"{parent_dir}/mesh/{dataset}/{category}/*.obj")
            if not mesh_files:
                print(f"  No mesh: {parent_dir}/mesh/{dataset}/{category}/")
                return False
            mesh_file = mesh_files[0]

            masks_cache = f"{args.results_dir}/{args.recon_cache_dir}/masks/{video_id}.npz"
            if not os.path.exists(masks_cache):
                raise FileNotFoundError(f"Masks cache missing: {masks_cache}. Run step2 first.")

            video_masks = load_masks_from_cache(masks_cache)

            fp_input = f"{args.results_dir}/{args.foundation_pose_dir}/{video_id_origin}"
            if (
                os.path.exists(out_dir)
                and getattr(args, "object_tracking_backend", "mose3_fp_hybrid") == "mose3_fp_hybrid"
            ):
                # A requested rerun keeps every previous raw/processed result.
                import time

                previous = f"{out_dir}.previous-{time.time_ns()}"
                os.rename(out_dir, previous)
                print(f"  Preserved previous object initialization: {previous}")
            else:
                shutil.rmtree(out_dir, ignore_errors=True)
            shutil.copytree(fp_input, out_dir)

            try:
                success = run_obj_pose_est(
                    video_path=video_file,
                    mesh_file=mesh_file,
                    input_dir=out_dir,
                    video_masks=video_masks,
                    debug=args.foundation_pose_debug,
                    crop_image=args.crop_image,
                    interpolation_factor=args.interpolation_factor,
                    is_static=args.is_static_obj,
                    object_tracking_backend=getattr(
                        args, "object_tracking_backend", "mose3_fp_hybrid"
                    ),
                    object_tracking_config=getattr(
                        args,
                        "object_tracking_config",
                        "configs/object_tracking/mose3_fp_hybrid.yaml",
                    ),
                )
            except StaticObjectValidationError as exc:
                reason = f"unexpected_object_motion: {exc}"
                Path(failed_marker).write_text(f"{reason}\n")
                recon_dir.mkdir(parents=True, exist_ok=True)
                invalid_marker.write_text(f"{reason}\n")
                motion_rejection_marker.write_text(f"{reason}\n")
                if valid_dir.exists():
                    shutil.rmtree(valid_dir)
                print(f"  Rejected (step3) {video_id}: {exc}")
                continue

            if not success:
                open(failed_marker, "w").close()
            elif had_motion_rejection:
                motion_rejection_marker.unlink(missing_ok=True)
                invalid_marker.unlink(missing_ok=True)

        except Exception as e:
            print(f"  Error (step3) {video_id}: {e}\n{traceback.format_exc()}")


def step4_optimize_4dhoi(video_ids, args):
    """Step 4: 4D HOI optimization."""
    for video_id in tqdm(sorted(video_ids), desc="Step 4 — Optimize"):
        try:
            video_id = _strip_mp4(video_id)
            output_dir = os.path.join(args.results_dir, args.output_dir, video_id)
            failed_marker = (
                f"{args.results_dir}/{args.foundation_pose_output_dir}/{video_id}/failed"
            )

            if os.path.exists(failed_marker):
                print(f"  Skipping step4 {video_id}: object pose estimation failed")
                continue

            if args.skip_done and os.path.exists(f"{output_dir}/hoi_data.pkl"):
                continue

            hmr_file = f"{args.results_dir}/{args.hmr_dir}/{video_id}.npz"
            video_file = f"{args.results_dir}/{args.video_dir}/{video_id}.mp4"
            if not os.path.exists(hmr_file):
                raise FileNotFoundError(f"No HMR file: {hmr_file}")

            dataset, category = video_id.split("/")[:2]
            obj_path = sorted(
                glob(f"{args.results_dir}/generation/mesh/{dataset}/{category}/*.obj")
            )[0]
            obj_pose_file = f"{args.results_dir}/{args.foundation_pose_output_dir}/{video_id}/pose_estimation_output/poses_in_cam.pkl"
            render_cfg_file = f"{args.results_dir}/{args.foundation_pose_output_dir}/{video_id}/first_frame_pose.pickle"

            if "optimization" in args.cfg:
                opt_cfg = dict(args.cfg["optimization"])
                opt_cfg["human_model"] = args.cfg["human_model"]
            else:
                opt_cfg = dict(args.cfg)

            optimizer = HOIOptimizer(
                exp_name=video_id,
                cfg=opt_cfg,
                cache_dir=f"{args.results_dir}/{args.recon_cache_dir}",
                output_dir=output_dir,
                device=args.device,
            )

            data = optimizer.init_data(
                video_file, hmr_file, obj_path, obj_pose_file, render_cfg_file
            )
            hoi_data = optimizer.optimize(data=data)
            save_hoi_data(hoi_data, f"{output_dir}/hoi_data.pkl")

        except Exception as e:
            print(f"  Error (step4) {video_id}: {e}\n{traceback.format_exc()}")


def step5_filter_hoi_result(video_ids, args):
    """Step 5: Filter, post-process, and package valid results."""
    sp_vis = ScenepicVisualizer()
    os.makedirs(f"{args.results_dir}/{args.output_dir}_valid", exist_ok=True)

    num_total = num_valid = num_frames_total = num_frames_valid = 0
    filter_cfg = {
        "human_model": args.cfg["human_model"],
        "eval": args.cfg["optimization"].get("eval", {}),
        "filtering": args.cfg["filtering"],
    }
    post_cfg = {
        "human_model": args.cfg["human_model"],
        "post_processing": args.cfg["post_processing"],
    }

    for video_id in tqdm(sorted(video_ids), desc="Step 5 — Filter"):
        try:
            video_id = _strip_mp4(video_id)
            invalid_marker = f"{args.results_dir}/{args.output_dir}/{video_id}/invalid"
            valid_dir = f"{args.results_dir}/{args.output_dir}_valid/{video_id}"
            motion_rejection_marker = (
                f"{args.results_dir}/{args.output_dir}/{video_id}/unexpected_object_motion"
            )

            if os.path.exists(motion_rejection_marker):
                if os.path.exists(valid_dir):
                    shutil.rmtree(valid_dir)
                continue

            if args.skip_done and (
                os.path.exists(f"{valid_dir}/hoi_data/hoi_data.pkl")
                or os.path.exists(invalid_marker)
            ):
                continue

            hoi_file = f"{args.results_dir}/{args.output_dir}/{video_id}/hoi_data.pkl"
            if not os.path.exists(hoi_file):
                continue

            cam_file = (
                f"{args.results_dir}/{args.hmr_cache_dir}/{video_id}/preprocess/slam_results.pt"
            )
            result_camera = torch.load(cam_file) if os.path.exists(cam_file) else None

            hoi_data = load_hoi_data(hoi_file)
            source_video = f"{args.results_dir}/{args.video_dir}/{video_id}.mp4"
            if os.path.exists(source_video):
                source_fps, _ = get_video_fps_and_frame_count(source_video)
                hoi_data["human_data"]["mocap_frame_rate"] = source_fps
            num_total += 1
            num_frames_total += hoi_data["human_data"]["poses"].shape[0]
            if not hoi_data.get("success", True):
                continue

            dataset, category = video_id.split("/")[:2]
            obj_path = sorted(
                glob(f"{args.results_dir}/generation/mesh/{dataset}/{category}/*.obj")
            )[0]
            hoi_data["meta"].update(
                {
                    "obj_path": obj_path,
                    "obj_pose_file": f"{args.results_dir}/{args.foundation_pose_output_dir}/{video_id}/pose_estimation_output/poses_in_cam.pkl",
                    "render_config_file": f"{args.results_dir}/{args.foundation_pose_output_dir}/{video_id}/first_frame_pose.pickle",
                    "masks_cache_file": f"{args.results_dir}/{args.recon_cache_dir}/masks/{video_id}.npz",
                }
            )

            log_dir = f"{args.results_dir}/{args.output_dir}/{video_id}"
            os.makedirs(log_dir, exist_ok=True)
            logger = create_logger(log_dir)

            valid_hoi, hoi_data = filter_hoi_result(
                result_camera, hoi_data, filter_cfg, device=args.device, logger=logger
            )

            if valid_hoi:
                hoi_dir = os.path.join(valid_dir, "hoi_data")
                mesh_dir = os.path.join(valid_dir, "mesh_data")
                result_dir = os.path.join(valid_dir, "result_vis")
                for d in (hoi_dir, mesh_dir, result_dir):
                    os.makedirs(d, exist_ok=True)

                hoi_processed = post_process_hoi_result(hoi_data, mesh_dir, post_cfg)
                save_hoi_data(hoi_processed, f"{hoi_dir}/hoi_data.pkl")

                # Copy mesh data
                src_mesh = f"{args.results_dir}/generation/mesh/{dataset}/{category}"
                if args.dataset == "robocasa":
                    os.system(f"cp -r '{src_mesh}'/image*.png '{mesh_dir}'")
                    os.system(f"cp -r '{src_mesh}'/material.mtl '{mesh_dir}'")
                else:
                    os.system(f"cp -r '{src_mesh}'/* '{mesh_dir}'")

                # Copy videos
                src_video = f"{args.results_dir}/{args.video_dir}/{video_id}.mp4"
                input_copy = f"{result_dir}/input.mp4"
                if os.path.exists(src_video):
                    os.system(f"cp '{src_video}' '{input_copy}'")

                recon_base = f"{args.results_dir}/{args.output_dir}/{video_id}"
                vid_names = {
                    "result": "recon_result.mp4",
                    "result_top_view": "recon_result_top_view.mp4",
                    "result_front_view": "recon_result_front_view.mp4",
                    "result_side_view": "recon_result_side_view.mp4",
                }
                vid_copies = {}
                for name, fname in vid_names.items():
                    dest = f"{result_dir}/{fname}"
                    candidates = sorted(glob(f"{recon_base}/*/{name}.mp4"))
                    if candidates:
                        os.system(f"cp '{candidates[-1]}' '{dest}'")
                    vid_copies[name] = dest

                concat_videos(
                    input_copy,
                    vid_copies["result"],
                    vid_copies["result_top_view"],
                    f"{result_dir}/recon_comparison.mp4",
                )

                num_valid += 1
                num_frames_valid += hoi_data["human_data"]["poses"].shape[0]

                # ScenePic visualization
                hoi_processed["object_path"] = obj_path
                from grail.models.human_model import create_human_model

                human_model = create_human_model(args.cfg["human_model"], device=args.device)
                if human_model.model_type == "soma03":
                    human_model.load_shape_params(args.cfg["human_model"], None, args.device)
                vis_input = prep_visualizer_input(
                    hoi_processed,
                    human_model=human_model,
                    normalize_trans=False,
                    device=args.device,
                )
                sp_vis.vis_scene(
                    vis_input,
                    f"{result_dir}/recon_result.html",
                    window_size=(400, 400),
                    fps=get_motion_fps(hoi_processed),
                )
            else:
                open(invalid_marker, "w").close()
                if os.path.exists(valid_dir):
                    shutil.rmtree(valid_dir)

        except Exception as e:
            print(f"  Error (step5) {video_id}: {e}\n{traceback.format_exc()}")

    if num_total > 0:
        print(
            f"  Videos: {num_valid}/{num_total} valid | "
            f"Frames: {num_frames_valid}/{num_frames_total} valid"
        )


def step6_visualize_hoi_result(video_ids, args):
    """Step 6: ScenePic HTML visualization."""
    os.makedirs(f"{args.results_dir}/{args.vis_scenepic_dir}", exist_ok=True)
    sp_vis = ScenepicVisualizer()

    for video_id in sorted(video_ids):
        try:
            video_id = _strip_mp4(video_id)
            html_file = f"{args.results_dir}/{args.vis_scenepic_dir}/{video_id}.html"
            if args.skip_done and os.path.exists(html_file):
                continue
            if args.vis_valid_only and not os.path.exists(
                f"{args.results_dir}/{args.output_dir}_valid/{video_id}"
            ):
                continue

            hoi_data = pickle.load(
                open(f"{args.results_dir}/{args.output_dir}/{video_id}/hoi_data.pkl", "rb")
            )
            source_video = f"{args.results_dir}/{args.video_dir}/{video_id}.mp4"
            if os.path.exists(source_video):
                source_fps, _ = get_video_fps_and_frame_count(source_video)
                hoi_data["human_data"]["mocap_frame_rate"] = source_fps
            dataset, category = video_id.split("/")[:2]
            hoi_data["object_path"] = category2object(
                f"{args.results_dir}/generation/mesh/{dataset}", category
            )
            vis_input = prep_visualizer_input(hoi_data, normalize_trans=False)
            os.makedirs(os.path.dirname(html_file), exist_ok=True)
            sp_vis.vis_scene(
                vis_input,
                html_file,
                window_size=(400, 400),
                fps=get_motion_fps(hoi_data),
            )

        except Exception as e:
            print(f"  Error (step6) {video_id}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_STEPS = [
    (1, "skip_step1", step1_predict_human_motion),
    (2, "skip_step2", step2_preprocess_data),
    (3, "skip_step3", step3_obj_pose_estimation),
    (4, "skip_step4", step4_optimize_4dhoi),
    (5, "skip_step5", step5_filter_hoi_result),
    # (6, "skip_step6", step6_visualize_hoi_result),
]


def main():
    """Entry point for the 4D HOI reconstruction pipeline."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=DEFAULT_RECON_CONFIG)
    pre_args, _ = pre.parse_known_args()

    cfg, cfg_flat = load_recon_config(pre_args.config)

    parser = argparse.ArgumentParser(description="4D HOI Reconstruction Pipeline")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_job_chunks", type=int, default=1)
    parser.add_argument("--job_chunk_idx", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_step1", action="store_true")
    parser.add_argument("--skip_step2", action="store_true")
    parser.add_argument("--skip_step3", action="store_true")
    parser.add_argument("--skip_step4", action="store_true")
    parser.add_argument("--skip_step5", action="store_true")
    parser.add_argument("--skip_step6", action="store_true")
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--is_static_obj", action="store_true")
    parser.add_argument(
        "--object_tracking_backend",
        choices=["mose3_fp_hybrid", "foundationpose"],
        default="mose3_fp_hybrid",
    )
    parser.add_argument(
        "--object_tracking_config", default="configs/object_tracking/mose3_fp_hybrid.yaml"
    )

    parser.set_defaults(**cfg_flat)
    args = parser.parse_args()

    from grail.core.types import parse_recon_config

    cfg = parse_recon_config(cfg)
    args.cfg = cfg

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Discover videos
    if args.video_id is not None:
        video_ids = [args.video_id]
    else:
        if args.category is not None:
            char_pat = args.character or ""
            video_paths = glob(
                f"{args.results_dir}/{args.video_dir}/{args.dataset}/{args.category}/{char_pat}*.mp4"
            )
        elif args.dataset is not None:
            video_paths = glob(f"{args.results_dir}/{args.video_dir}/{args.dataset}/*/*.mp4")
        else:
            video_paths = glob(f"{args.results_dir}/{args.video_dir}/*/*/*.mp4")
        if not video_paths:
            print(f"No videos found in {args.results_dir}/{args.video_dir}")
            return
        video_ids = sorted(vp.split(f"{args.video_dir}/")[1] for vp in video_paths)[
            args.job_chunk_idx :: args.num_job_chunks
        ]

    print(
        f"Config: {pre_args.config} | Worker {args.job_chunk_idx}/{args.num_job_chunks} | "
        f"{len(video_ids)} videos"
    )

    t0 = time.time()
    success = True

    try:
        for step_num, skip_attr, step_fn in _STEPS:
            if not getattr(args, skip_attr):
                step_fn(video_ids, args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        success = False
    except Exception as e:
        print(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        success = False

    elapsed = time.time() - t0
    print(f"{'OK' if success else 'FAILED'} — {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
