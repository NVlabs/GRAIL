#!/usr/bin/env python3
"""Convert retargeting motion_lib format to the shard format expected by batch_render_replay.py.

Reads robot/*.pkl, objects/*.pkl, meta/*.pkl from a motion_lib directory and produces:
  - trajectories/{motion_key}.trajectory.pkl  (per-motion trajectory)
  - metrics_eval.json  (synthetic: marks all motions as successful)

Usage:
    python prepare_vis_shard.py \
        --data_dir data/motion_lib/my_output \
        --shard_dir /tmp/vis_shard_my_output \
        [--max_motions 16]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import joblib
import numpy as np

# MuJoCo → IsaacLab DOF reordering for G1 29-DOF body joints.
# Source: imports/SONIC/gear_sonic/envs/manager_env/robots/g1.py
# Both the retargeting pipeline and the data-export pipeline
# (export_successful_rollouts.py, which IsaacLab→MuJoCo-reorders before
# writing) store DOFs in MuJoCo joint order; batch_render_replay.py uses an
# IsaacLab articulation, so we always apply this reorder here.
G1_MUJOCO_TO_ISAACLAB_DOF = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]


def _detect_quat_convention(root_rot: np.ndarray) -> str:
    """Return 'wxyz' or 'xyzw' based on which slot stores the scalar w.

    The retargeting pipeline (grail/retargeting/retarget.py) writes root_rot
    as wxyz; the data-export pipeline (grail/data_export/export_successful_rollouts.py)
    writes it as xyzw. The renderer (batch_render_replay.py) consumes wxyz.

    Heuristic: for any robot motion that spends most of its time near-upright,
    the scalar component w (≈cos(θ/2)) carries the bulk of the quaternion's
    magnitude. Compare the energy of component 0 vs. component 3 — whichever
    is bigger is w.

    Reliable in practice because robot rotations rarely exceed ±90° on roll/
    pitch within a single motion clip; if they ever did, the comparison would
    flip and the caller can override with --quat_convention.
    """
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        return "wxyz"
    energy = np.mean(root_rot**2, axis=0)
    return "wxyz" if energy[0] >= energy[3] else "xyzw"


def convert_motion_lib_to_trajectories(
    data_dir: str,
    shard_dir: str,
    max_motions: int = 0,
    motion_filter: set[str] | None = None,
    quat_convention: str = "xyzw",
):
    """Read motion-lib output and write trajectory pkls + metrics_eval.json.

    Accepts both retarget-source (root_rot=wxyz) and data-export-source
    (root_rot=xyzw) motion libraries. Default is "xyzw" because the typical
    consumer is a data-export merged / public-release dir; pass "wxyz"
    explicitly for retargeting output (data/motion_lib/<name>/), or "auto"
    to use a magnitude-based heuristic (which can mis-classify motions that
    start with a non-upright pose).

    Hand DOFs: when the input robot pkl carries a (T, 14) ``hand_dof_pos``
    array (added in export_successful_rollouts.py as of 2026-06-01), it is
    appended to the body trajectory so the renderer can drive the gripper.
    For older pkls without this field, the renderer zero-pads the missing
    14 DOFs, leaving the gripper in its open pose.
    """
    robot_dir = os.path.join(data_dir, "robot")
    objects_dir = os.path.join(data_dir, "objects")
    meta_dir = os.path.join(data_dir, "meta")

    if not os.path.isdir(robot_dir):
        sys.exit(f"Error: robot directory not found: {robot_dir}")

    robot_files = sorted(f for f in os.listdir(robot_dir) if f.endswith(".pkl"))
    if not robot_files:
        sys.exit(f"Error: no .pkl files in {robot_dir}")

    motion_keys = [os.path.splitext(f)[0] for f in robot_files]
    if motion_filter:
        motion_keys = [k for k in motion_keys if k in motion_filter]
    if max_motions > 0 and len(motion_keys) > max_motions:
        rng = random.Random(42)
        motion_keys = sorted(rng.sample(motion_keys, max_motions))

    traj_dir = os.path.join(shard_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    successful_keys = []
    conv_counts = {"wxyz": 0, "xyzw": 0}
    hand_gt_count = 0

    for motion_key in motion_keys:
        robot_path = os.path.join(robot_dir, f"{motion_key}.pkl")
        object_path = os.path.join(objects_dir, f"{motion_key}.pkl")

        try:
            robot_data = joblib.load(robot_path)
        except Exception as e:
            print(f"  Skip {motion_key}: cannot load robot pkl: {e}")
            continue

        # robot pkl is {motion_key: {...}} dict
        if isinstance(robot_data, dict) and motion_key in robot_data:
            robot_entry = robot_data[motion_key]
        elif isinstance(robot_data, dict) and len(robot_data) == 1:
            robot_entry = next(iter(robot_data.values()))
        else:
            print(f"  Skip {motion_key}: unexpected robot pkl structure")
            continue

        root_pos = np.asarray(robot_entry["root_trans_offset"], dtype=np.float32)
        root_rot_raw = np.asarray(robot_entry["root_rot"], dtype=np.float32)
        # Canonicalize to wxyz (what batch_render_replay.py expects).
        conv = (
            quat_convention if quat_convention != "auto" else _detect_quat_convention(root_rot_raw)
        )
        if conv == "xyzw":
            root_rot = root_rot_raw[:, [3, 0, 1, 2]]
        else:
            root_rot = root_rot_raw
        conv_counts[conv] = conv_counts.get(conv, 0) + 1
        dof_pos_mujoco = np.asarray(robot_entry["dof"], dtype=np.float32)
        fps = float(robot_entry.get("fps", 30.0))
        total_frames = len(root_pos)

        # Reorder DOFs from MuJoCo order → IsaacLab order
        num_dofs = dof_pos_mujoco.shape[1]
        if num_dofs == 29:
            dof_pos = dof_pos_mujoco[:, G1_MUJOCO_TO_ISAACLAB_DOF]
        elif num_dofs >= 29:
            # 43-DOF (body+hands): reorder the first 29 body DOFs, keep hands as-is
            dof_pos = dof_pos_mujoco.copy()
            dof_pos[:, :29] = dof_pos_mujoco[:, :29][:, G1_MUJOCO_TO_ISAACLAB_DOF]
        else:
            print(f"  Warning: {motion_key} has {num_dofs} DOFs (expected >=29), skipping reorder")
            dof_pos = dof_pos_mujoco

        # Append 14-DOF hand trajectory when present (added in
        # export_successful_rollouts.py as of 2026-06-01). Without it the
        # renderer zero-pads → gripper stays in its open pose.
        if dof_pos.shape[1] == 29 and "hand_dof_pos" in robot_entry:
            h = np.asarray(robot_entry["hand_dof_pos"], dtype=np.float32)
            if h.ndim == 2 and h.shape[1] == 14:
                min_t = min(total_frames, h.shape[0])
                h = h[:min_t]
                if min_t < total_frames:
                    pad = np.tile(h[-1:], (total_frames - min_t, 1))
                    h = np.concatenate([h, pad], axis=0)
                dof_pos = np.concatenate([dof_pos, h], axis=1)
                hand_gt_count += 1
            else:
                print(
                    f"  Warning: {motion_key} hand_dof_pos shape {h.shape}, "
                    f"expected (T, 14); leaving gripper open"
                )

        # Object data
        object_pos_w = None
        object_quat_w = None
        object_scale = None
        if os.path.isfile(object_path):
            try:
                obj_data = joblib.load(object_path)
                if isinstance(obj_data, dict) and motion_key in obj_data:
                    obj_entry = obj_data[motion_key]
                elif isinstance(obj_data, dict) and len(obj_data) == 1:
                    obj_entry = next(iter(obj_data.values()))
                else:
                    obj_entry = None

                if obj_entry is not None:
                    obj_pos = np.asarray(obj_entry["root_pos"], dtype=np.float32)
                    obj_quat = np.asarray(obj_entry["root_quat"], dtype=np.float32)
                    # Shape: (T, 1, 3) → (T, 3); (T, 1, 4) → (T, 4)
                    if obj_pos.ndim == 3:
                        obj_pos = obj_pos[:, 0, :]
                    if obj_quat.ndim == 3:
                        obj_quat = obj_quat[:, 0, :]
                    # Ensure same frame count (object may not be padded identically)
                    min_t = min(total_frames, len(obj_pos))
                    object_pos_w = obj_pos[:min_t]
                    object_quat_w = obj_quat[:min_t]
                    if min_t < total_frames:
                        # Pad by repeating last frame
                        pad_n = total_frames - min_t
                        object_pos_w = np.concatenate(
                            [object_pos_w, np.tile(object_pos_w[-1:], (pad_n, 1))], axis=0
                        )
                        object_quat_w = np.concatenate(
                            [object_quat_w, np.tile(object_quat_w[-1:], (pad_n, 1))], axis=0
                        )

                    # Object meshes are stored at their source size; the motion-lib
                    # scale must be applied by the replay renderer. Scale is static
                    # and may be stored as a scalar, (3,), or (3, 1).
                    raw_scale = obj_entry.get("scale", np.ones(3))
                    object_scale = np.asarray(raw_scale, dtype=np.float32).reshape(-1)
                    if object_scale.size == 1:
                        object_scale = np.repeat(object_scale, 3)
                    if (
                        object_scale.size != 3
                        or not np.all(np.isfinite(object_scale))
                        or np.any(object_scale <= 0)
                    ):
                        print(
                            f"  Warning: {motion_key} has invalid object scale "
                            f"{np.asarray(raw_scale).shape}; using identity"
                        )
                        object_scale = np.ones(3, dtype=np.float32)
            except Exception as e:
                print(f"  Warning: cannot load object for {motion_key}: {e}")

        # Table/meta data
        table_pos_w = None
        meta_path = os.path.join(meta_dir, f"{motion_key}.pkl")
        if os.path.isfile(meta_path):
            try:
                meta = joblib.load(meta_path)
                if "table_pos" in meta:
                    tp = np.asarray(meta["table_pos"], dtype=np.float32)
                    # Broadcast static table pos to all frames
                    table_pos_w = np.tile(tp[np.newaxis, :], (total_frames, 1))
            except Exception as e:
                print(f"  Warning: cannot load meta for {motion_key}: {e}")

        # Build trajectory dict (same format batch_render_replay.py expects)
        traj = {
            "total_frames": total_frames,
            "root_pos_w": root_pos,
            "root_quat_w": root_rot,
            "dof_pos": dof_pos,
            "fps": fps,
        }
        if object_pos_w is not None:
            traj["object_pos_w"] = object_pos_w
            traj["object_quat_w"] = object_quat_w
            traj["object_scale"] = object_scale
        if table_pos_w is not None:
            traj["table_pos_w"] = table_pos_w

        # Write trajectory pkl
        traj_path = os.path.join(traj_dir, f"{motion_key}.trajectory.pkl")
        import pickle

        with open(traj_path, "wb") as f:
            pickle.dump(traj, f)

        successful_keys.append(motion_key)

    # Write synthetic metrics_eval.json (all motions marked as successful)
    metrics = {
        "eval/all_metrics_dict": {
            "motion_keys": successful_keys,
            "terminated": [False] * len(successful_keys),
            "mpjpe_l": [0.0] * len(successful_keys),
            "mpjpe_g": [0.0] * len(successful_keys),
            "obj_pos_error": [0.0] * len(successful_keys),
        }
    }
    metrics_path = os.path.join(shard_dir, "metrics_eval.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Prepared {len(successful_keys)} trajectories in {shard_dir}")
    print(f"  quat conventions (root_rot): wxyz={conv_counts['wxyz']} xyzw={conv_counts['xyzw']}")
    print(f"  hand DOFs from hand_dof_pos: {hand_gt_count} / {len(successful_keys)}")
    return successful_keys


def main():
    parser = argparse.ArgumentParser(
        description="Convert motion_lib to shard format for visualization"
    )
    parser.add_argument(
        "--data_dir", required=True, help="motion_lib directory (contains robot/, objects/, meta/)"
    )
    parser.add_argument("--shard_dir", required=True, help="Output shard directory")
    parser.add_argument("--max_motions", type=int, default=0, help="Max motions to convert (0=all)")
    parser.add_argument(
        "--motion_keys",
        type=str,
        default=None,
        help="Comma-separated motion keys to convert (default: all)",
    )
    parser.add_argument(
        "--quat_convention",
        choices=("auto", "wxyz", "xyzw"),
        default="xyzw",
        help="Convention of root_rot in the input robot pkls. Default is "
        "'xyzw' (matches data-export / public-release output). Use "
        "'wxyz' for retarget output (data/motion_lib/<name>/), or "
        "'auto' for magnitude-based detection (may mis-classify "
        "motions that don't start near-upright).",
    )
    args = parser.parse_args()

    motion_filter = set(args.motion_keys.split(",")) if args.motion_keys else None
    convert_motion_lib_to_trajectories(
        args.data_dir,
        args.shard_dir,
        args.max_motions,
        motion_filter=motion_filter,
        quat_convention=args.quat_convention,
    )


if __name__ == "__main__":
    main()
