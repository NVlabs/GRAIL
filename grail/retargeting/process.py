import argparse
import copy
import os
import os.path as osp
import pickle
import shutil

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d


def load_pickle(f):
    return pickle.load(open(f, "rb"))


#!/usr/bin/env python
"""
Packaged forward kinematics function for motion library data.

Usage:
    from forward_kinematics import forward_kinematics_from_motion

    # Load skeleton info once (stays fixed)
    skeleton_info = load_skeleton_info(meta_pkl_path)

    # Run FK on any motion
    body_positions = forward_kinematics_from_motion(motion_pkl_path, skeleton_info)
    # Returns: (T, 30, 3) body positions in world frame
"""

from pathlib import Path
from typing import Any, Dict, Union


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / (angle + 1e-8)
    K = np.zeros(axis_angle.shape[:-1] + (3, 3))
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    I = np.eye(3)
    sin_a = np.sin(angle)[..., None]
    cos_a = np.cos(angle)[..., None]
    R = I + sin_a * K + (1 - cos_a) * (K @ K)
    return R


def load_skeleton_info(meta_pkl_path):
    with open(meta_pkl_path, "rb") as f:
        meta = pickle.load(f)
    return meta["skeleton_info"]


def load_motion_data(motion_pkl_path):
    data = joblib.load(motion_pkl_path)
    motion_key = list(data.keys())[0]
    return data[motion_key]


def forward_kinematics_single_frame(dof_pos, root_rot_aa, root_trans, skeleton_info):
    dof_axis = skeleton_info["dof_axis"]
    parents_raw = skeleton_info["skeleton_parents"]
    offsets = skeleton_info["skeleton_offsets"]
    local_rot_mat = skeleton_info.get("local_rotation_mat", None)
    parents = [int(p.item()) if hasattr(p, "item") else int(p) for p in parents_raw]
    num_bodies = len(parents)
    offsets = offsets.squeeze(0)
    if hasattr(dof_axis, "numpy"):
        dof_axis = dof_axis.numpy()
    if hasattr(offsets, "numpy"):
        offsets = offsets.numpy()
    if local_rot_mat is not None and hasattr(local_rot_mat, "numpy"):
        local_rot_mat = local_rot_mat.numpy()
    root_mat = axis_angle_to_matrix(root_rot_aa)
    root_pos = root_trans
    pose_aa = dof_pos[:, None] * dof_axis
    pose_mat = axis_angle_to_matrix(pose_aa)
    positions = []
    rotations = []
    for i in range(num_bodies):
        if parents[i] == -1:
            positions.append(root_pos)
            rotations.append(root_mat)
        else:
            parent_rot = rotations[parents[i]]
            parent_pos = positions[parents[i]]
            jpos = parent_pos + parent_rot @ offsets[i]
            joint_rot = pose_mat[i - 1] if i > 0 else np.eye(3)
            if local_rot_mat is not None:
                local_r = local_rot_mat[0, i]
                rot = parent_rot @ local_r @ joint_rot
            else:
                rot = parent_rot @ joint_rot
            positions.append(jpos)
            rotations.append(rot)
    body_pos = np.stack(positions)
    return body_pos


def forward_kinematics_from_motion(motion_pkl_path, skeleton_info):
    motion = load_motion_data(motion_pkl_path)
    dof = motion["dof"]
    pose_aa = motion["pose_aa"]
    root_trans = motion["root_trans_offset"]
    root_rot_aa = pose_aa[:, 0, :]
    T = dof.shape[0]
    body_positions = []
    for t in range(T):
        body_pos = forward_kinematics_single_frame(
            dof_pos=dof[t],
            root_rot_aa=root_rot_aa[t],
            root_trans=root_trans[t],
            skeleton_info=skeleton_info,
        )
        body_positions.append(body_pos)
    return np.stack(body_positions)


def forward_kinematics_batch(dof, root_rot_aa, root_trans, skeleton_info):
    T = dof.shape[0]
    body_positions = []
    for t in range(T):
        body_pos = forward_kinematics_single_frame(
            dof_pos=dof[t],
            root_rot_aa=root_rot_aa[t],
            root_trans=root_trans[t],
            skeleton_info=skeleton_info,
        )
        body_positions.append(body_pos)
    return np.stack(body_positions)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def line_plane_intersection(p1, p2, plane_z):
    z1, z2 = p1[2], p2[2]
    if (z1 - plane_z) * (z2 - plane_z) > 0:
        return None
    if abs(z2 - z1) < 1e-8:
        return None
    t = (plane_z - z1) / (z2 - z1)
    if t < 0 or t > 1:
        return None
    x = p1[0] + t * (p2[0] - p1[0])
    y = p1[1] + t * (p2[1] - p1[1])
    return (x, y)


def _contact_has_points(points):
    return hasattr(points, "size") and points.size > 0


def _first_contact_frame(contact_points):
    if not contact_points:
        return None
    frames = sorted(k for k, v in contact_points.items() if _contact_has_points(v))
    return frames[0] if frames else None


def _has_contact_at_or_before(contact_points, frame):
    if not contact_points:
        return False
    return any(k <= frame and _contact_has_points(v) for k, v in contact_points.items())


def _hand_action_from_start(num_frames, start_frame):
    action = np.full(num_frames, -1.0, dtype=np.float32)
    if start_frame is not None:
        action[max(0, start_frame) :] = 1.0
    return action


SHOULDER_BLEND_JOINTS = {
    "left": {"pitch": 15, "roll": 16, "yaw": 17},
    "right": {"pitch": 22, "roll": 23, "yaw": 24},
}

SHOULDER_FIRST_FRAME_BLEND_DEFAULTS = {
    "side": "right",
    "axis": "roll",
    "degrees": -20.0,
    "blend_seconds": 4.0,
}


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _duration_seconds_to_source_frame_count(source_num_frames, source_fps, duration_seconds):
    """Convert a time window in seconds to a source-frame count."""
    if duration_seconds is None:
        return 0
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be >= 0")

    source_num_frames = int(source_num_frames)
    source_fps = float(source_fps)
    if source_num_frames <= 0:
        return 0
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")

    return min(int(np.ceil(float(duration_seconds) * source_fps)), source_num_frames)


def apply_first_frame_shoulder_dof_blend(
    motion,
    skeleton_info,
    *,
    side,
    axis,
    degrees,
    blend_frames=None,
    blend_seconds=None,
):
    """Offset shoulder DOF at frame 0 and linearly blend back to original motion."""
    if blend_frames is None and blend_seconds is None:
        raise ValueError("one of blend_frames or blend_seconds must be provided")
    if blend_frames is not None and blend_frames < 1:
        raise ValueError("blend_frames must be >= 1")

    sides = ("left", "right") if side == "both" else (side,)
    joint_idxs = [SHOULDER_BLEND_JOINTS[s][axis] for s in sides]

    dof = motion["dof"]
    pose_aa = motion["pose_aa"]
    dof_axis = _to_numpy(skeleton_info["dof_axis"])

    if blend_frames is None:
        num_frames = _duration_seconds_to_source_frame_count(
            dof.shape[0],
            motion.get("fps", 50),
            blend_seconds,
        )
    else:
        num_frames = min(int(blend_frames), int(dof.shape[0]))
    if num_frames <= 0:
        return joint_idxs

    if num_frames == 1:
        weights = np.ones(1, dtype=dof.dtype)
    else:
        weights = np.linspace(1.0, 0.0, num_frames, dtype=dof.dtype)
    offsets = np.deg2rad(degrees) * weights

    for joint_idx in joint_idxs:
        dof[:num_frames, joint_idx] += offsets
        pose_aa[:num_frames, joint_idx + 1, :] = (
            dof[:num_frames, joint_idx, None] * dof_axis[joint_idx]
        )
    return joint_idxs


def apply_default_first_frame_shoulder_dof_blend(motion, skeleton_info):
    """Apply the accepted shoulder first-frame fix."""
    return apply_first_frame_shoulder_dof_blend(
        motion,
        skeleton_info,
        **SHOULDER_FIRST_FRAME_BLEND_DEFAULTS,
    )


def _target_frame_count_to_source_frame_count(
    source_num_frames,
    source_fps,
    target_frame_count,
    *,
    target_fps=50,
):
    """Map a target-fps frame count to source frames using SONIC's replay resampling."""
    if target_frame_count is None:
        return 0
    if target_frame_count < 0:
        raise ValueError("target_frame_count must be >= 0")

    source_num_frames = int(source_num_frames)
    target_frame_count = int(target_frame_count)
    source_fps = float(source_fps)
    target_fps = float(target_fps)

    if target_frame_count == 0 or source_num_frames <= 0:
        return 0
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if source_fps == target_fps:
        return min(target_frame_count, source_num_frames)

    duration = (source_num_frames - 1) / source_fps
    target_num_frames = len(np.arange(0, duration, 1 / target_fps))
    if target_num_frames <= 0:
        return 0
    if target_frame_count >= target_num_frames:
        return source_num_frames

    # Match MotionLibBase.load_motions hand-action upsampling:
    # round(linspace(0, src_len - 1, num_target_frames)).
    source_indices = np.round(np.linspace(0, source_num_frames - 1, target_num_frames)).astype(int)
    return int(source_indices[:target_frame_count].max()) + 1


def apply_right_hand_closed_until_frame(motion, frame_count, *, target_fps=50):
    """Force right hand closed until ``frame_count`` replay/target-fps frames."""
    if frame_count is None:
        return 0
    if "hand_action_right" not in motion:
        raise KeyError("motion is missing hand_action_right")

    hand_action = motion["hand_action_right"]
    num_frames = _target_frame_count_to_source_frame_count(
        hand_action.shape[0],
        motion.get("fps", target_fps),
        frame_count,
        target_fps=target_fps,
    )
    hand_action[:num_frames] = 1.0
    return num_frames


def apply_right_hand_closed_until_seconds(motion, seconds):
    """Force right hand closed for the first ``seconds`` of source motion."""
    if seconds is None:
        return 0
    if "hand_action_right" not in motion:
        raise KeyError("motion is missing hand_action_right")

    hand_action = motion["hand_action_right"]
    num_frames = _duration_seconds_to_source_frame_count(
        hand_action.shape[0],
        motion.get("fps", 50),
        seconds,
    )
    hand_action[:num_frames] = 1.0
    return num_frames


def apply_right_hand_closed_until_first_close_margin_seconds(motion, seconds):
    """Force right hand closed until ``seconds`` before its first open-to-closed transition."""
    if seconds is None:
        return 0
    if seconds < 0:
        raise ValueError("seconds must be >= 0")
    if "hand_action_right" not in motion:
        raise KeyError("motion is missing hand_action_right")

    hand_action = motion["hand_action_right"]
    if hand_action.shape[0] <= 1:
        return 0

    close_transitions = np.flatnonzero((hand_action[:-1] < 0.0) & (hand_action[1:] > 0.0))
    if close_transitions.size == 0:
        return 0

    first_close_frame = int(close_transitions[0]) + 1
    margin_frames = _duration_seconds_to_source_frame_count(
        hand_action.shape[0],
        motion.get("fps", 50),
        seconds,
    )
    overwrite_end = max(0, first_close_frame - margin_frames)
    hand_action[:overwrite_end] = 1.0
    return overwrite_end


def process_existing_ha_per_object_dataset(
    base_dir,
    out_dir,
    skeleton_info,
    *,
    shoulder_first_frame_blend=False,
    right_hand_closed_until_frame=None,
    right_hand_closed_until_seconds=None,
    right_hand_closed_until_first_close_margin_seconds=None,
):
    """Copy an existing per-object HA dataset and apply robot-only post-processing."""
    if (
        right_hand_closed_until_first_close_margin_seconds is None
        and right_hand_closed_until_frame is not None
        and right_hand_closed_until_seconds is not None
    ):
        raise ValueError(
            "Use either right_hand_closed_until_seconds or right_hand_closed_until_frame, not both"
        )
    if (
        right_hand_closed_until_first_close_margin_seconds is None
        and right_hand_closed_until_seconds is None
        and right_hand_closed_until_frame is not None
    ):
        right_hand_closed_until_seconds = float(right_hand_closed_until_frame) / 50.0

    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    robot_in_dir = base_dir / "robot"
    robot_out_dir = out_dir / "robot"

    if not robot_in_dir.is_dir():
        raise FileNotFoundError(f"Missing robot input directory: {robot_in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    robot_out_dir.mkdir(parents=True, exist_ok=True)

    for item in base_dir.iterdir():
        if item.name == "robot":
            continue
        dest = out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    processed = 0
    for robot_path in sorted(robot_in_dir.glob("*.pkl")):
        motion_data = joblib.load(robot_path)
        for motion in motion_data.values():
            if shoulder_first_frame_blend:
                apply_default_first_frame_shoulder_dof_blend(motion, skeleton_info)
            if right_hand_closed_until_first_close_margin_seconds is not None:
                apply_right_hand_closed_until_first_close_margin_seconds(
                    motion,
                    right_hand_closed_until_first_close_margin_seconds,
                )
            else:
                apply_right_hand_closed_until_seconds(motion, right_hand_closed_until_seconds)
            processed += 1
        joblib.dump(motion_data, robot_out_dir / robot_path.name)

    print(f"Saved {processed} robot pkls to {robot_out_dir}")
    return processed


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Process GenHOI motion data for robot training")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory containing robot/, objects/, meta/ subdirs",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output directory for processed pkl files"
    )
    parser.add_argument(
        "--meta_pkl", type=str, required=True, help="Path to meta.pkl containing skeleton_info"
    )
    parser.add_argument(
        "--bottle_usd", type=str, default=None, help="Path to bottle.usd to copy (optional)"
    )
    parser.add_argument("--include_contact_points", action="store_true")
    parser.add_argument("--grasp_from_lift", action="store_true")
    parser.add_argument("--lift_threshold", type=float, default=0.02)
    parser.add_argument("--grasp_anticipation_frames", type=int, default=10)
    parser.add_argument(
        "--skip_no_lift",
        action="store_true",
        help="Skip motions where object never lifts above threshold (instead of crashing)",
    )
    parser.add_argument(
        "--per_object",
        action="store_true",
        help="Save one pkl per motion (robot/ and objects/ subdirs) instead of monolithic pkls",
    )
    parser.add_argument(
        "--processed_input_passthrough",
        action="store_true",
        help="Input is already processed HA; copy non-robot data unchanged and apply robot-only edits",
    )
    parser.add_argument(
        "--z_correction",
        action="store_true",
        help="Enable object Z offset correction for 10cm->15cm cylinder retarget (off by default)",
    )
    parser.add_argument(
        "--treat_hands_equally",
        "--treat-hands-equally",
        action="store_true",
        help=(
            "Handle left/right hands symmetrically: preserve both arms and derive each "
            "hand action from that hand's contacts. Default preserves legacy "
            "right-hand pickup behavior."
        ),
    )
    parser.add_argument(
        "--table_size",
        type=float,
        nargs=3,
        default=[1.5, 0.7, 0.04],
        metavar=("X", "Y", "Z"),
        help="Processed table size in meters",
    )
    parser.add_argument(
        "--shoulder_first_frame_blend",
        action="store_true",
        help="Enable accepted shoulder fix: right roll -20deg blended over the first 4 seconds",
    )
    parser.add_argument(
        "--right_hand_closed_until_seconds",
        type=float,
        default=None,
        help="Force hand_action_right closed for the first N seconds",
    )
    parser.add_argument(
        "--right_hand_closed_until_frame",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--right_hand_closed_until_first_close_margin_seconds",
        type=float,
        default=None,
        help="Force hand_action_right closed until N seconds before its first natural close",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    base_dir = args.input
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    print(f"Input:  {base_dir}")
    print(f"Output: {out_dir}")

    zero_out_left_arm = not args.treat_hands_equally
    rotate_table_90_deg = False
    rot_90_z_quat = np.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
    clip_object_z_to_initial = False
    cut_sequence_on_z_deviation = False
    z_deviation_threshold = 0.30
    smooth_robot_motion = False
    smooth_sigma = 4
    smooth_start_frame = 96
    global_scale = 1.0
    override_fps = None

    if args.bottle_usd and os.path.exists(args.bottle_usd):
        shutil.copy(args.bottle_usd, osp.join(out_dir, "bottle.usd"))

    skeleton_info = load_skeleton_info(args.meta_pkl)

    if args.processed_input_passthrough:
        if not args.per_object:
            parser.error("--processed_input_passthrough requires --per_object")
        if (
            args.right_hand_closed_until_first_close_margin_seconds is None
            and args.right_hand_closed_until_seconds is not None
            and args.right_hand_closed_until_frame is not None
        ):
            parser.error(
                "Use either --right_hand_closed_until_seconds or --right_hand_closed_until_frame, not both"
            )
        process_existing_ha_per_object_dataset(
            base_dir,
            out_dir,
            skeleton_info,
            shoulder_first_frame_blend=args.shoulder_first_frame_blend,
            right_hand_closed_until_frame=args.right_hand_closed_until_frame,
            right_hand_closed_until_seconds=args.right_hand_closed_until_seconds,
            right_hand_closed_until_first_close_margin_seconds=(
                args.right_hand_closed_until_first_close_margin_seconds
            ),
        )
        return

    left_arm_joint_idxs = [15, 16, 17, 18, 19, 20, 21]
    left_arm_zero_joints = np.array([0, 0.2, -0.2, 1.57, 0, 0, 0])

    fix_ankle_pitch = False
    ankle_pitch_joint_idxs = [4, 10]
    ankle_pitch_target_mean = -0.2
    fix_ankle_pitch_independent = True

    apply_table_object_size_adjustments = args.z_correction
    table_y_pos_offset = -0.05 * global_scale
    object_z_pos_offset = 0.025 * global_scale

    adjust_table_for_leg_intersection = True
    table_size = np.array(args.table_size, dtype=np.float32) * global_scale
    object_radius = 0.03 * global_scale
    table_leg_margin = 0.05 * global_scale
    lower_body_indices = set(range(16)) | {23}

    filter_motion_indices = None
    add_hand_actions = True
    include_contact_points = args.include_contact_points
    grasp_anticipation_frames = args.grasp_anticipation_frames

    processed_object_motions = {}
    processed_robot_motions = {}
    processed_meta_motions = {}

    all_motion_keys = sorted([f.split(".")[0] for f in os.listdir(osp.join(base_dir, "robot"))])
    if filter_motion_indices is not None:
        motion_keys_to_process = [
            all_motion_keys[i] for i in filter_motion_indices if i < len(all_motion_keys)
        ]
    else:
        motion_keys_to_process = all_motion_keys

    for f in motion_keys_to_process:
        try:
            meta = joblib.load(osp.join(base_dir, "meta", f + ".pkl"))
            object_motion = joblib.load(osp.join(base_dir, "objects", f + ".pkl"))
            robot_motion = joblib.load(osp.join(base_dir, "robot", f + ".pkl"))
        except:
            print("Skipping", f)
            continue

        if smooth_robot_motion:
            for key in ["root_trans_offset", "dof", "pose_aa"]:
                original = robot_motion[f][key].copy()
                smoothed = gaussian_filter1d(original, sigma=smooth_sigma, axis=0)
                robot_motion[f][key] = np.concatenate(
                    [original[:smooth_start_frame], smoothed[smooth_start_frame:]], axis=0
                )

        body_positions = forward_kinematics_from_motion(
            osp.join(base_dir, "robot", f + ".pkl"), skeleton_info
        )
        min_z = body_positions.min(axis=1)[:, 2].min()

        has_table = "table_pos" in meta
        if has_table:
            table_pos = np.array(meta["table_pos"]) * global_scale
            table_quat = np.array(meta["table_quat"])
            if rotate_table_90_deg:
                table_quat = quat_multiply(rot_90_z_quat, table_quat)
        else:
            table_pos = None
            table_quat = None

        if min_z < 0:
            if has_table:
                table_pos[2] -= min_z
            object_motion[f]["root_pos"][:, :, 2] -= min_z
            robot_motion[f]["root_trans_offset"][:, 2] -= min_z

        if apply_table_object_size_adjustments:
            if not has_table:
                raise ValueError(
                    f"{f}: --apply_table_object_size_adjustments requires table_pos in meta"
                )
            table_pos[1] += table_y_pos_offset
            object_motion[f]["root_pos"][:, :, 2] += object_z_pos_offset
            object_y_initial = object_motion[f]["root_pos"][0, 0, 1]
            min_allowed_table_y = object_y_initial - table_size[1] / 2 + object_radius
            if table_pos[1] < min_allowed_table_y:
                print(f"  WARNING: {f} - initial table offset clamped")
                table_pos[1] = min_allowed_table_y

        if clip_object_z_to_initial:
            initial_z = object_motion[f]["root_pos"][0, :, 2].copy()
            object_motion[f]["root_pos"][:, :, 2] = np.maximum(
                object_motion[f]["root_pos"][:, :, 2], initial_z[None, :]
            )

        if zero_out_left_arm:
            robot_motion[f]["dof"][:, left_arm_joint_idxs] = left_arm_zero_joints
            robot_motion[f]["pose_aa"][
                :, np.array(left_arm_joint_idxs) + 1, [1, 0, 2, 1, 0, 1, 2]
            ] = left_arm_zero_joints

        if args.shoulder_first_frame_blend:
            shoulder_joint_idxs = apply_default_first_frame_shoulder_dof_blend(
                robot_motion[f], skeleton_info
            )
            shoulder_defaults = SHOULDER_FIRST_FRAME_BLEND_DEFAULTS
            print(
                f"  Applied shoulder first-frame blend to DOFs {shoulder_joint_idxs}: "
                f"{shoulder_defaults['side']} {shoulder_defaults['axis']} "
                f"{shoulder_defaults['degrees']:g}deg over {shoulder_defaults['blend_seconds']:g}s"
            )

        if fix_ankle_pitch:
            if fix_ankle_pitch_independent:
                for joint_idx in ankle_pitch_joint_idxs:
                    current_mean = robot_motion[f]["dof"][:, joint_idx].mean()
                    offset = ankle_pitch_target_mean - current_mean
                    robot_motion[f]["dof"][:, joint_idx] += offset
                    robot_motion[f]["pose_aa"][:, joint_idx + 1, 1] += offset

        if adjust_table_for_leg_intersection and has_table:
            T = robot_motion[f]["dof"].shape[0]
            parents = skeleton_info["skeleton_parents"]
            parents = [int(p.item()) if hasattr(p, "item") else int(p) for p in parents]
            all_intersections = []
            table_z = table_pos[2] + table_size[2] / 2
            for t in range(T):
                body_pos = forward_kinematics_single_frame(
                    dof_pos=robot_motion[f]["dof"][t],
                    root_rot_aa=robot_motion[f]["pose_aa"][t, 0],
                    root_trans=robot_motion[f]["root_trans_offset"][t],
                    skeleton_info=skeleton_info,
                )
                for child_idx in range(len(parents)):
                    parent_idx = parents[child_idx]
                    if parent_idx == -1:
                        continue
                    if child_idx not in lower_body_indices or parent_idx not in lower_body_indices:
                        continue
                    intersection = line_plane_intersection(
                        body_pos[parent_idx], body_pos[child_idx], table_z
                    )
                    if intersection is not None:
                        ix, iy = intersection
                        table_x_min = table_pos[0] - table_size[0] / 2
                        table_x_max = table_pos[0] + table_size[0] / 2
                        table_y_min = table_pos[1] - table_size[1] / 2
                        table_y_max = table_pos[1] + table_size[1] / 2
                        if table_x_min <= ix <= table_x_max and table_y_min <= iy <= table_y_max:
                            all_intersections.append(intersection)
            if all_intersections:
                min_intersection = min(all_intersections, key=lambda p: p[1])
                ix_min, y_intersect = min_intersection
                desired_table_y = y_intersect - table_size[1] / 2 - table_leg_margin
                object_y_initial = object_motion[f]["root_pos"][0, 0, 1]
                min_allowed_table_y = object_y_initial - table_size[1] / 2 + object_radius
                if desired_table_y < min_allowed_table_y:
                    print(f"  WARNING: {f} - table adjustment clamped")
                    table_pos[1] = min_allowed_table_y
                else:
                    table_pos[1] = desired_table_y
                print(f"  Adjusted table Y: {table_pos[1]:.4f}")

        lift_frame = None
        if args.grasp_from_lift:
            obj_z = object_motion[f]["root_pos"][:, 0, 2]
            initial_z = obj_z[0]
            lifted_frames = np.where(obj_z > initial_z + args.lift_threshold)[0]
            if len(lifted_frames) == 0:
                if args.skip_no_lift:
                    print(f"  SKIP {f}: object never lifted above {args.lift_threshold}m")
                    continue
                raise ValueError(f"Motion {f} never lifts object above {args.lift_threshold}m")
            lift_frame = int(lifted_frames[0])
            print(f"  Lift frame: {lift_frame}")

        if add_hand_actions:
            T = robot_motion[f]["dof"].shape[0]
            if args.treat_hands_equally:
                hand_actions = {}
                for hand in ("left", "right"):
                    contact_points = object_motion[f].get(f"contact_points_{hand}_hand", {})
                    if lift_frame is not None:
                        hand_contact_frame = (
                            lift_frame
                            if _has_contact_at_or_before(contact_points, lift_frame)
                            else None
                        )
                    else:
                        hand_contact_frame = _first_contact_frame(contact_points)
                    if hand_contact_frame is None:
                        grasp_start_frame = None
                    else:
                        grasp_start_frame = hand_contact_frame - grasp_anticipation_frames
                    hand_actions[hand] = _hand_action_from_start(T, grasp_start_frame)
                hand_action_left = hand_actions["left"]
                hand_action_right = hand_actions["right"]
            else:
                if lift_frame is not None:
                    first_contact_frame = lift_frame
                else:
                    contact_points_right = object_motion[f].get("contact_points_right_hand", {})
                    if not contact_points_right:
                        raise ValueError(f"Motion {f} has no right hand contact points")
                    first_contact_frame = min(contact_points_right.keys())
                grasp_start_frame = max(0, first_contact_frame - grasp_anticipation_frames)
                hand_action_left = np.full(T, -1.0, dtype=np.float32)
                hand_action_right = np.full(T, -1.0, dtype=np.float32)
                hand_action_right[grasp_start_frame:] = 1.0

        output_fps = override_fps if override_fps is not None else object_motion[f]["fps"]
        processed_object_motions[f] = {
            "root_pos": object_motion[f]["root_pos"],
            "root_quat": object_motion[f]["root_quat"],
            "fps": output_fps,
        }
        if include_contact_points:
            for hand in ("left_hand", "right_hand"):
                key = f"contact_points_{hand}"
                raw = object_motion[f][key]
                if lift_frame is not None:
                    filtered = {k: copy.deepcopy(v) for k, v in raw.items() if k >= lift_frame}
                    processed_object_motions[f][key] = filtered
                else:
                    processed_object_motions[f][key] = copy.deepcopy(raw)

        processed_robot_motions[f] = {
            "root_trans_offset": robot_motion[f]["root_trans_offset"],
            "pose_aa": robot_motion[f]["pose_aa"],
            "dof": robot_motion[f]["dof"],
            "root_rot": robot_motion[f]["root_rot"],
            "smpl_joints": robot_motion[f]["smpl_joints"],
            "fps": output_fps,
        }
        if add_hand_actions:
            processed_robot_motions[f]["hand_action_left"] = hand_action_left
            processed_robot_motions[f]["hand_action_right"] = hand_action_right
            if args.right_hand_closed_until_first_close_margin_seconds is not None:
                apply_right_hand_closed_until_first_close_margin_seconds(
                    processed_robot_motions[f],
                    args.right_hand_closed_until_first_close_margin_seconds,
                )
            else:
                if (
                    args.right_hand_closed_until_seconds is not None
                    and args.right_hand_closed_until_frame is not None
                ):
                    parser.error(
                        "Use either --right_hand_closed_until_seconds or --right_hand_closed_until_frame, not both"
                    )
                if (
                    args.right_hand_closed_until_seconds is None
                    and args.right_hand_closed_until_frame is not None
                ):
                    right_hand_closed_until_seconds = (
                        float(args.right_hand_closed_until_frame) / 50.0
                    )
                else:
                    right_hand_closed_until_seconds = args.right_hand_closed_until_seconds
                apply_right_hand_closed_until_seconds(
                    processed_robot_motions[f],
                    right_hand_closed_until_seconds,
                )
        processed_meta_motions[f] = {
            "object_name": meta.get("object_name"),
        }
        if has_table:
            processed_meta_motions[f].update(
                {
                    "table_pos": table_pos,
                    "table_quat": table_quat,
                    "table_size": table_size,
                }
            )
        print(f, min_z)

    if args.per_object:
        robot_dir = osp.join(out_dir, "robot")
        objects_dir = osp.join(out_dir, "objects")
        meta_dir = osp.join(out_dir, "meta")
        os.makedirs(robot_dir, exist_ok=True)
        os.makedirs(objects_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        for key in processed_robot_motions:
            joblib.dump({key: processed_robot_motions[key]}, osp.join(robot_dir, f"{key}.pkl"))
        for key in processed_object_motions:
            joblib.dump({key: processed_object_motions[key]}, osp.join(objects_dir, f"{key}.pkl"))
        for key in processed_meta_motions:
            joblib.dump(processed_meta_motions[key], osp.join(meta_dir, f"{key}.pkl"))
        print(
            f"\nSaved {len(processed_robot_motions)} per-object pkls to {out_dir}/robot/, {out_dir}/objects/, {out_dir}/meta/"
        )
    else:
        joblib.dump(processed_object_motions, osp.join(out_dir, "processed_object_motions.pkl"))
        joblib.dump(processed_robot_motions, osp.join(out_dir, "processed_robot_motions.pkl"))
        joblib.dump(processed_meta_motions, osp.join(out_dir, "processed_meta_motions.pkl"))
        print(f"\nSaved {len(processed_object_motions)} motions to {out_dir}")


if __name__ == "__main__":
    main()
