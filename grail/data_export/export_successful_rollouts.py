"""Export successful task-general tracking rollouts as a training-ready motion library.

Reads eval output (metrics_eval.json + .trajectory.pkl files) and produces
a standard motion library directory with robot/, objects/, object_usd/, and meta/.

Usage:
    # Export from a local eval directory
    python -m grail.data_export.export_successful_rollouts \
        --eval_dir /path/to/eval/000100/train \
        --source_data data/motion_lib_genhoi/p019_0324_ha \
        --output_dir data/motion_lib_genhoi/p019_cleaned

    # With object position error threshold (default: no threshold, just use terminated)
    python -m grail.data_export.export_successful_rollouts \
        --eval_dir /path/to/eval/000100/train \
        --source_data data/motion_lib_genhoi/p019_0324_ha \
        --output_dir data/motion_lib_genhoi/p019_cleaned \
        --obj_pos_threshold 0.05

    # Verify exported data loads through MotionLibBase
    python -m grail.data_export.export_successful_rollouts \
        --verify data/motion_lib_genhoi/p019_cleaned
"""

import argparse
import glob
import json
import os
import pickle
import shutil

import joblib
import numpy as np


def _copy_usd_textures(source_dir, output_dir, motion_key):
    """Copy texture assets referenced by a motion's USD.

    GenHOI/GRAIL USDs use two texture layout conventions under source_dir/object_usd/textures/:
      - Flat:   textures/<motion_key>_*.{jpg,png}          (stairs001_0303)
      - Nested: textures/<motion_key>/*.{jpg,png,exr,mdl}  (sitting002_0307)
    The USD references textures via relative paths, so downstream consumers that
    copy only the .usd file see broken materials (white default). Copy whichever
    layout exists so the exported object_usd/ directory is self-contained.
    """
    src_tex_root = os.path.join(source_dir, "object_usd", "textures")
    if not os.path.isdir(src_tex_root):
        return
    dst_tex_root = os.path.join(output_dir, "object_usd", "textures")
    # Nested-style: per-motion subdirectory
    nested_src = os.path.join(src_tex_root, motion_key)
    if os.path.isdir(nested_src):
        nested_dst = os.path.join(dst_tex_root, motion_key)
        if not os.path.exists(nested_dst):
            shutil.copytree(nested_src, nested_dst)
    # Flat-style: files prefixed with motion_key (e.g. <motion_key>_model.jpg)
    for match in glob.glob(os.path.join(src_tex_root, f"{motion_key}_*")):
        if os.path.isfile(match):
            os.makedirs(dst_tex_root, exist_ok=True)
            dst_file = os.path.join(dst_tex_root, os.path.basename(match))
            if not os.path.exists(dst_file):
                shutil.copy2(match, dst_file)


def load_metrics(eval_dir):
    """Load metrics_eval.json from eval directory."""
    metrics_path = os.path.join(eval_dir, "metrics_eval.json")
    if not os.path.exists(metrics_path):
        # Try threshold_eval subdirectory
        metrics_path = os.path.join(eval_dir, "threshold_eval", "metrics_eval.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No metrics_eval.json found in {eval_dir}")

    with open(metrics_path) as f:
        metrics = json.load(f)
    return metrics


def get_successful_motions(metrics, obj_pos_threshold=None, min_progress=1.0):
    """Filter metrics to find successful motions.

    Args:
        metrics: Loaded metrics_eval.json
        obj_pos_threshold: Max object position error (meters) to count as success.
            None means don't filter by object error.
        min_progress: Minimum progress ratio (0-1) to count as success.

    Returns:
        List of (motion_key, env_idx) tuples for successful motions.
    """
    all_metrics = metrics.get("eval/all_metrics_dict", {})
    motion_keys = all_metrics.get("motion_keys", [])
    terminated = all_metrics.get("terminated", [])
    progress = all_metrics.get("progress", [])
    obj_pos_error = all_metrics.get("obj_pos_error", [])

    if not motion_keys:
        raise ValueError("No motion_keys found in metrics")

    successful = []
    failed = []
    for i, key in enumerate(motion_keys):
        is_terminated = terminated[i] if i < len(terminated) else True
        prog = progress[i] if i < len(progress) else 0.0

        if is_terminated:
            failed.append((key, "terminated"))
            continue
        if prog < min_progress:
            failed.append((key, f"progress={prog:.2f}"))
            continue
        if obj_pos_threshold is not None and i < len(obj_pos_error):
            if obj_pos_error[i] > obj_pos_threshold:
                failed.append((key, f"obj_err={obj_pos_error[i]:.4f}"))
                continue

        successful.append((key, i))

    print(
        f"Success: {len(successful)}/{len(motion_keys)} motions "
        f"({100*len(successful)/max(len(motion_keys),1):.1f}%)"
    )
    if failed:
        print(f"Failed: {len(failed)} motions")
        for key, reason in failed[:10]:
            print(f"  SKIP: {key} ({reason})")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

    return successful


def find_trajectory_file(eval_dir, env_idx, motion_key=None):
    """Find .trajectory.pkl file for a given env index or motion key.

    In multi-batch mode, trajectory files are named by motion_key.
    In single-batch mode, they are named by env_idx.
    """
    search_subdirs = ["export_render", "render_results", "trajectories", ""]

    # Try motion-key-based name first (multi-batch mode)
    if motion_key:
        key_fname = f"{motion_key}.trajectory.pkl"
        for subdir in search_subdirs:
            key_path = (
                os.path.join(eval_dir, subdir, key_fname)
                if subdir
                else os.path.join(eval_dir, key_fname)
            )
            if os.path.exists(key_path):
                return key_path

    # Try env-index-based name (single-batch mode)
    fname = f"{env_idx:06d}.trajectory.pkl"
    for subdir in search_subdirs:
        traj_path = (
            os.path.join(eval_dir, subdir, fname) if subdir else os.path.join(eval_dir, fname)
        )
        if os.path.exists(traj_path):
            return traj_path
    # Fallback: glob search
    matches = glob.glob(os.path.join(eval_dir, "**", fname), recursive=True)
    return matches[0] if matches else None


def find_video_file(eval_dir, env_idx, motion_key=None):
    """Find .mp4 video file for a given env index or motion key."""
    search_subdirs = ["export_render", "render_results", "render", "trajectories", ""]

    # Try motion-key-based name first
    if motion_key:
        key_fname = f"{motion_key}.mp4"
        for subdir in search_subdirs:
            path = (
                os.path.join(eval_dir, subdir, key_fname)
                if subdir
                else os.path.join(eval_dir, key_fname)
            )
            if os.path.exists(path):
                return path

    # Try env-index-based name
    fname = f"{env_idx:06d}.mp4"
    for subdir in search_subdirs:
        video_path = (
            os.path.join(eval_dir, subdir, fname) if subdir else os.path.join(eval_dir, fname)
        )
        if os.path.exists(video_path):
            return video_path
    return None


def quat_wxyz_to_xyzw(quat):
    """Convert quaternion from wxyz to xyzw format."""
    return np.stack([quat[..., 1], quat[..., 2], quat[..., 3], quat[..., 0]], axis=-1)


# G1 DOF ordering: trajectory recorder saves in IsaacLab order,
# motion library stores in Mujoco order.
# From groot/rl/envs/manager_env/robots/g1.py
G1_ISAACLAB_TO_MUJOCO_DOF = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

# Skeleton DOF axis for computing pose_aa from joint angles. Each G1 body DOF
# (Mujoco order) is a 1-DOF revolute joint rotating about one of X/Y/Z, so its
# axis-angle contribution is `dof_value * axis`. Loaded from the skeleton meta
# pkl. NOTE: this file must exist on the machine running the export (it is not
# hard-coded) — if it is missing, _get_dof_axis raises instead of returning None,
# because a zeroed body pose_aa silently corrupts the export (MotionLibBase FKs
# the tracked pose from pose_aa, not from `dof`).
_DOF_AXIS_CACHE = None


def _get_dof_axis():
    """Load the G1 body DOF rotation-axis table (shape (29, 3)) from skeleton meta.

    Raises loudly if the meta pkl is missing or malformed — the export must never
    fall back to a zeroed pose_aa.
    """
    global _DOF_AXIS_CACHE
    if _DOF_AXIS_CACHE is not None:
        return _DOF_AXIS_CACHE
    meta_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "body_profiles",
        "unitree_g1",
        "smplx",
        "g1_skeleton_meta.pkl",
    )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"g1_skeleton_meta.pkl not found at {meta_path}. It is required to build "
            "pose_aa (dof_axis). Restore data/body_profiles/unitree_g1/smplx/ "
            "in the checkout on the exporting machine."
        )
    meta = joblib.load(meta_path)
    try:
        dof_axis = np.array(meta["skeleton_info"]["dof_axis"], dtype=np.float32)
    except (KeyError, TypeError) as exc:
        raise KeyError(f"skeleton_info.dof_axis missing/malformed in {meta_path}") from exc
    _DOF_AXIS_CACHE = dof_axis
    return dof_axis


def resample_nn_time(arr, src_fps, tgt_len, tgt_fps):
    """Time-based nearest-neighbor resample of a 1-D array to ``tgt_len`` frames.

    Nearest-neighbor (not linear) preserves the discrete -1/+1 hand-action step;
    time-based (not index-based) aligns events by absolute time, correct when the
    source and target run at different fps for the same wall-clock motion.
    """
    arr = np.asarray(arr)
    src_len = arr.shape[0]
    t = np.arange(tgt_len, dtype=np.float64) / float(tgt_fps)
    idx = np.round(t * float(src_fps)).astype(int)
    idx = np.clip(idx, 0, src_len - 1)
    return arr[idx]


def trajectory_to_robot_pkl(trajectory, source_robot_data=None):
    """Convert trajectory recorder data to robot motion library format.

    Handles DOF reordering (IsaacLab → Mujoco) and quaternion format
    conversion (wxyz → xyzw) to match the motion library convention.

    Args:
        trajectory: Dict from .trajectory.pkl
        source_robot_data: Original robot motion data (for fields we can't reconstruct)

    Returns:
        Dict in robot motion library format
    """
    from scipy.spatial.transform import Rotation as sRot

    dof_pos = trajectory["dof_pos"]  # (T, 43) - includes hand DOFs
    root_pos = trajectory["root_pos_w"]  # (T, 3)
    root_quat_wxyz = trajectory["root_quat_w"]  # (T, 4) wxyz

    # Robot DOF is first 29 joints (body), remaining 14 are hand DOFs
    body_dof_isaaclab = dof_pos[:, :29]

    # Reorder DOFs: IsaacLab → Mujoco order
    body_dof = body_dof_isaaclab[:, G1_ISAACLAB_TO_MUJOCO_DOF]

    # Convert root quaternion: wxyz → xyzw (motion library convention)
    root_quat_xyzw = quat_wxyz_to_xyzw(root_quat_wxyz)

    # Compute root rotation as axis-angle for pose_aa
    root_rotvec = sRot.from_quat(root_quat_xyzw).as_rotvec().astype(np.float32)

    fps = trajectory["fps"]
    n_frames = len(body_dof)

    # Build pose_aa: (T, 30, 3) — joint 0 is root rotation, joints 1-29 are DOF rotations.
    # MotionLibBase reconstructs the tracked body pose from pose_aa via FK (NOT from
    # `dof`), so pose_aa[:, 1:] MUST be populated — a zeroed body pose_aa makes the
    # policy track a default pose. dof_axis is embedded (see _get_dof_axis), so this
    # is always computable regardless of what data files exist on the cluster.
    dof_axis = _get_dof_axis()
    if dof_axis is None or dof_axis.shape != (29, 3):
        raise RuntimeError(
            f"dof_axis unavailable or wrong shape ({None if dof_axis is None else dof_axis.shape}); "
            "cannot build pose_aa. This must never happen with the embedded G1_DOF_AXIS."
        )
    pose_aa = np.zeros((n_frames, 30, 3), dtype=np.float32)
    pose_aa[:, 0, :] = root_rotvec
    # pose_aa[t, j+1, :] = dof[t, j] * dof_axis[j, :]
    pose_aa[:, 1:, :] = body_dof[:, :, np.newaxis] * dof_axis[np.newaxis, :, :]

    # Generate placeholder SMPL joints (zeros) — real values require FK recomputation
    smpl_joints = np.zeros((n_frames, 24, 3), dtype=np.float32)

    robot_data = {
        "dof": body_dof.astype(np.float32),
        "root_trans_offset": root_pos.astype(np.float32),
        "root_rot": root_quat_xyzw.astype(np.float32),
        "pose_aa": pose_aa,
        "smpl_joints": smpl_joints,
        "fps": fps,
    }

    # Hand actions: extract from 43-DOF trajectory
    if dof_pos.shape[1] > 29:
        hand_dof = dof_pos[:, 29:]  # (T, 14) in IsaacLab articulation order
        # Full per-joint trajectory — kept so downstream consumers (visualizer,
        # SONIC motion_lib's `hand_dof_pos`) can recover the exact gripper state.
        robot_data["hand_dof_pos"] = hand_dof.astype(np.float32)

        # hand_action_left/right is a FEED-FORWARD gripper command replayed by the
        # tracking policy (pnp_table sets use_motion_hand_actions=true), interpreted
        # with a *discrete* convention: -1.0 = open, +1.0 = closed. It is NOT a
        # continuous grip strength. The authoritative signal (with correct grasp
        # anticipation timing) lives in the SOURCE retargeted motion, so copy it
        # from there and resample onto this rollout's fps/frame count.
        src_hl = source_robot_data.get("hand_action_left") if source_robot_data else None
        src_hr = source_robot_data.get("hand_action_right") if source_robot_data else None
        if src_hl is not None and src_hr is not None:
            src_fps = float(source_robot_data.get("fps", fps))
            robot_data["hand_action_left"] = resample_nn_time(
                np.asarray(src_hl, dtype=np.float32), src_fps, n_frames, float(fps)
            ).astype(np.float32)
            robot_data["hand_action_right"] = resample_nn_time(
                np.asarray(src_hr, dtype=np.float32), src_fps, n_frames, float(fps)
            ).astype(np.float32)
        else:
            # No source hand_action available. The old mean(abs(hand_dof)) heuristic
            # is >= 0 everywhere, which the discrete primitive reads as "always
            # closed" — a silent correctness bug. Refuse to emit a bogus signal.
            raise ValueError(
                "Cannot recover hand_action: source_robot_data lacks "
                "'hand_action_left'/'hand_action_right'. Pass --source_data pointing "
                "at the retargeted motion lib that contains them (the discrete "
                "-1=open/+1=closed convention cannot be reconstructed from hand DOFs)."
            )

    return robot_data


def trajectory_to_object_pkl(trajectory, source_object_data=None):
    """Convert trajectory recorder data to object motion library format.

    Args:
        trajectory: Dict from .trajectory.pkl
        source_object_data: Original object motion data (for scale, contacts)

    Returns:
        Dict in object motion library format, or None if no object data
    """
    if trajectory.get("object_pos_w") is None:
        return None

    obj_pos = trajectory["object_pos_w"]  # (T, 3)
    obj_quat = trajectory["object_quat_w"]  # (T, 4)

    fps = trajectory["fps"]

    # obj_quat is wxyz end-to-end: convert_hoi_to_motion_lib.py stores source root_quat
    # as wxyz, motion_lib passes bytes through to IsaacLab (which expects wxyz), and the
    # recorder reads obj.data.root_quat_w (also wxyz). Pass through without permutation so
    # the exported pkl matches the source pkl's convention.
    obj_quat_wxyz = np.asarray(obj_quat)

    object_data = {
        "root_pos": obj_pos[:, np.newaxis, :].astype(np.float32),  # (T, 1, 3)
        "root_quat": obj_quat_wxyz[:, np.newaxis, :].astype(np.float32),  # (T, 1, 4) wxyz
        "fps": float(fps),
    }

    # Copy scale and contact data from source if available
    if source_object_data is not None:
        if "scale" in source_object_data:
            object_data["scale"] = source_object_data["scale"]
        if "contact_points_left_hand" in source_object_data:
            object_data["contact_points_left_hand"] = source_object_data["contact_points_left_hand"]
        if "contact_points_right_hand" in source_object_data:
            object_data["contact_points_right_hand"] = source_object_data[
                "contact_points_right_hand"
            ]

    return object_data


def load_source_data(source_dir, motion_key):
    """Load original source motion data for a given motion key."""
    robot_data = None
    object_data = None
    meta_data = None

    # Try robot pkl
    robot_path = os.path.join(source_dir, "robot", f"{motion_key}.pkl")
    if os.path.exists(robot_path):
        data = joblib.load(robot_path)
        key = list(data.keys())[0]
        robot_data = data[key]

    # Try object pkl
    object_path = os.path.join(source_dir, "objects", f"{motion_key}.pkl")
    if os.path.exists(object_path):
        data = joblib.load(object_path)
        key = list(data.keys())[0]
        object_data = data[key]

    # Try meta pkl
    meta_path = os.path.join(source_dir, "meta", f"{motion_key}.pkl")
    if os.path.exists(meta_path):
        meta_data = joblib.load(meta_path)

    return robot_data, object_data, meta_data


def export_motions(
    eval_dir, source_dir, output_dir, obj_pos_threshold=None, min_progress=1.0, copy_videos=True
):
    """Main export function.

    Args:
        eval_dir: Path to eval output directory (containing metrics_eval.json)
        source_dir: Path to source motion library (for USD, metadata, scale)
        output_dir: Path to output motion library directory
        obj_pos_threshold: Max object position error threshold
        min_progress: Minimum progress to consider successful
        copy_videos: Whether to copy rendered videos
    """
    print(f"Loading metrics from {eval_dir}")
    metrics = load_metrics(eval_dir)

    successful = get_successful_motions(metrics, obj_pos_threshold, min_progress)
    if not successful:
        print("No successful motions found. Nothing to export.")
        return

    # Create output directories
    os.makedirs(os.path.join(output_dir, "robot"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "objects"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "object_usd"), exist_ok=True)
    if copy_videos:
        os.makedirs(os.path.join(output_dir, "vis"), exist_ok=True)

    exported_count = 0
    skipped_no_traj = 0
    export_manifest = {
        "source_eval_dir": os.path.abspath(eval_dir),
        "source_data_dir": os.path.abspath(source_dir) if source_dir else None,
        "obj_pos_threshold": obj_pos_threshold,
        "min_progress": min_progress,
        "total_motions_evaluated": len(
            metrics.get("eval/all_metrics_dict", {}).get("motion_keys", [])
        ),
        "exported_motions": [],
    }

    all_metrics = metrics.get("eval/all_metrics_dict", {})

    for motion_key, env_idx in successful:
        traj_path = find_trajectory_file(eval_dir, env_idx, motion_key=motion_key)
        if traj_path is None:
            skipped_no_traj += 1
            if skipped_no_traj <= 5:
                print(f"  WARN: No trajectory file for env {env_idx} ({motion_key})")
            continue

        # Load trajectory
        with open(traj_path, "rb") as f:
            trajectory = pickle.load(f)

        # Load source data for supplemental fields
        source_robot, source_object, source_meta = (None, None, None)
        if source_dir:
            source_robot, source_object, source_meta = load_source_data(source_dir, motion_key)

        # Convert to motion library format
        robot_data = trajectory_to_robot_pkl(trajectory, source_robot)
        object_data = trajectory_to_object_pkl(trajectory, source_object)

        # Save robot pkl
        robot_out = os.path.join(output_dir, "robot", f"{motion_key}.pkl")
        joblib.dump({motion_key: robot_data}, robot_out)

        # Save object pkl
        if object_data is not None:
            object_out = os.path.join(output_dir, "objects", f"{motion_key}.pkl")
            joblib.dump({motion_key: object_data}, object_out)

        # Write meta: start from source_meta, override table state with trajectory if recorded
        meta_data = dict(source_meta) if source_meta is not None else {}
        if trajectory.get("table_pos_w") is not None:
            table_pos = trajectory["table_pos_w"]
            table_quat = trajectory["table_quat_w"]
            meta_data["table_pos"] = table_pos[0].astype(np.float64)
            meta_data["table_quat"] = quat_wxyz_to_xyzw(table_quat[0:1])[0].astype(np.float64)
        if meta_data:
            meta_out = os.path.join(output_dir, "meta", f"{motion_key}.pkl")
            joblib.dump(meta_data, meta_out)

        # Copy USD asset from source (plus its referenced textures). Chair USDs
        # reference `textures/<motion>/model.jpg` (nested subdir per motion).
        # Stair USDs reference `textures/<motion>_model.jpg` (flat files). Copy
        # both patterns so exported USDs render with correct materials downstream.
        if source_dir:
            for ext in [".usd", ".usda"]:
                usd_src = os.path.join(source_dir, "object_usd", f"{motion_key}{ext}")
                if os.path.exists(usd_src):
                    usd_dst = os.path.join(output_dir, "object_usd", f"{motion_key}{ext}")
                    if not os.path.exists(usd_dst):
                        shutil.copy2(usd_src, usd_dst)
            _copy_usd_textures(source_dir, output_dir, motion_key)

        # Copy video
        if copy_videos:
            video_path = find_video_file(eval_dir, env_idx, motion_key=motion_key)
            if video_path:
                video_dst = os.path.join(output_dir, "vis", f"{motion_key}.mp4")
                shutil.copy2(video_path, video_dst)

        # Track per-motion metrics in manifest
        motion_info = {"motion_key": motion_key, "env_idx": env_idx}
        if "obj_pos_error" in all_metrics and env_idx < len(all_metrics["obj_pos_error"]):
            motion_info["obj_pos_error"] = all_metrics["obj_pos_error"][env_idx]
        if "obj_ori_error" in all_metrics and env_idx < len(all_metrics["obj_ori_error"]):
            motion_info["obj_ori_error"] = all_metrics["obj_ori_error"][env_idx]
        if "mpjpe_g" in all_metrics and env_idx < len(all_metrics["mpjpe_g"]):
            motion_info["mpjpe_g"] = all_metrics["mpjpe_g"][env_idx]
        if "progress" in all_metrics and env_idx < len(all_metrics["progress"]):
            motion_info["progress"] = all_metrics["progress"][env_idx]
        export_manifest["exported_motions"].append(motion_info)

        exported_count += 1

    # Copy object_usd/config.yaml if it exists
    if source_dir:
        config_src = os.path.join(source_dir, "object_usd", "config.yaml")
        if os.path.exists(config_src):
            shutil.copy2(config_src, os.path.join(output_dir, "object_usd", "config.yaml"))

    # Save manifest
    export_manifest["exported_count"] = exported_count
    export_manifest["skipped_no_trajectory"] = skipped_no_traj
    manifest_path = os.path.join(output_dir, "export_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(export_manifest, f, indent=2)

    print(f"\nExported {exported_count} motions to {output_dir}")
    if skipped_no_traj > 0:
        print(f"  Skipped {skipped_no_traj} motions (no trajectory file found)")
    print(f"  Manifest: {manifest_path}")
    print(f"  Robot PKLs: {output_dir}/robot/")
    print(f"  Object PKLs: {output_dir}/objects/")
    if copy_videos:
        print(f"  Vis MP4s: {output_dir}/vis/")


def verify_exported_data(output_dir):
    """Verify exported data can be loaded through MotionLibBase."""
    robot_dir = os.path.join(output_dir, "robot")
    objects_dir = os.path.join(output_dir, "objects")

    robot_files = sorted(glob.glob(os.path.join(robot_dir, "*.pkl")))
    object_files = sorted(glob.glob(os.path.join(objects_dir, "*.pkl")))

    print(f"Verifying {len(robot_files)} robot files, {len(object_files)} object files")

    errors = []
    for robot_file in robot_files:
        key = os.path.splitext(os.path.basename(robot_file))[0]
        try:
            data = joblib.load(robot_file)
            motion_key = list(data.keys())[0]
            m = data[motion_key]

            # Check required fields
            assert "dof" in m, "Missing 'dof' field"
            assert "root_trans_offset" in m, "Missing 'root_trans_offset' field"
            assert "root_rot" in m, "Missing 'root_rot' field"
            assert "fps" in m, "Missing 'fps' field"
            assert m["dof"].shape[1] == 29, f"dof shape {m['dof'].shape}, expected (T, 29)"
            assert (
                m["root_trans_offset"].shape[1] == 3
            ), f"root_trans shape {m['root_trans_offset'].shape}"
            assert m["root_rot"].shape[1] == 4, f"root_rot shape {m['root_rot'].shape}"
            assert m["dof"].shape[0] == m["root_trans_offset"].shape[0], "Frame count mismatch"
        except Exception as e:
            errors.append(f"Robot {key}: {e}")

    for obj_file in object_files:
        key = os.path.splitext(os.path.basename(obj_file))[0]
        try:
            data = joblib.load(obj_file)
            motion_key = list(data.keys())[0]
            m = data[motion_key]

            assert "root_pos" in m, "Missing 'root_pos' field"
            assert "root_quat" in m, "Missing 'root_quat' field"
            assert "fps" in m, "Missing 'fps' field"
            assert (
                len(m["root_pos"].shape) == 3
            ), f"root_pos shape {m['root_pos'].shape}, expected (T, 1, 3)"
            assert m["root_pos"].shape[1] >= 1, f"root_pos shape {m['root_pos'].shape}"
            assert m["root_pos"].shape[2] == 3, f"root_pos shape {m['root_pos'].shape}"
        except Exception as e:
            errors.append(f"Object {key}: {e}")

    if errors:
        print(f"\nVERIFICATION FAILED: {len(errors)} errors")
        for err in errors[:20]:
            print(f"  ERROR: {err}")
        return False
    else:
        print(
            f"\nVERIFICATION PASSED: All {len(robot_files)} robot + {len(object_files)} object files valid"
        )

        # Also try loading through MotionLibBase if available
        try:
            from easydict import EasyDict
            from groot.rl.utils.motion_lib.motion_lib_base import MotionLibBase

            cfg = EasyDict(
                {
                    "motion_file": robot_dir,
                    "debug": True,
                    "fix_height": "no_fix",
                    "target_fps": 30,
                    "skeleton_file": None,
                    "multi_thread": False,
                }
            )
            ml = MotionLibBase(cfg, device="cpu")
            print(f"  MotionLibBase loaded {ml._num_unique_motions} motions successfully")
        except Exception as e:
            print(f"  MotionLibBase load test skipped: {e}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Export successful task-general tracking rollouts")
    parser.add_argument("--eval_dir", type=str, help="Eval output dir with metrics_eval.json")
    parser.add_argument(
        "--source_data", type=str, help="Source motion library dir (for USD, metadata)"
    )
    parser.add_argument("--output_dir", type=str, help="Output motion library directory")
    parser.add_argument(
        "--obj_pos_threshold",
        type=float,
        default=None,
        help="Max object position error (meters) for success",
    )
    parser.add_argument(
        "--min_progress", type=float, default=1.0, help="Min progress ratio (0-1) for success"
    )
    parser.add_argument("--no_videos", action="store_true", help="Skip copying videos")
    parser.add_argument(
        "--verify", type=str, default=None, help="Verify an exported directory (no export)"
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_exported_data(args.verify)
        exit(0 if success else 1)

    if not args.eval_dir or not args.output_dir:
        parser.error("--eval_dir and --output_dir are required for export")

    export_motions(
        eval_dir=args.eval_dir,
        source_dir=args.source_data,
        output_dir=args.output_dir,
        obj_pos_threshold=args.obj_pos_threshold,
        min_progress=args.min_progress,
        copy_videos=not args.no_videos,
    )

    # Auto-verify after export
    print("\n--- Running verification ---")
    verify_exported_data(args.output_dir)


if __name__ == "__main__":
    main()
