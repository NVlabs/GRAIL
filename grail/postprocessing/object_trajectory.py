"""Smooth an exported object trajectory with GRAIL's existing pose filter.

Writes a separate, explicitly postprocessed artifact. Does not change the live
tracker, interpolate missing poses or retain stale tracker scores/state.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from grail.pose_est.utils import smooth_pose_matrices


def smooth_trajectory(
    poses, valid, frame_ids, window_length=9, polyorder=3, preserve_initial_pose=True
):
    """Filter contiguous valid runs; preserve missing rows and initial pose."""
    poses = np.asarray(poses, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    frame_ids = np.asarray(frame_ids)
    if poses.shape != (len(valid), 4, 4) or frame_ids.shape != valid.shape:
        raise ValueError("Expected poses [T,4,4], validity [T] and source frame IDs [T]")
    if not np.issubdtype(frame_ids.dtype, np.integer) or np.any(np.diff(frame_ids) <= 0):
        raise ValueError("Source frame IDs must be strictly increasing integers")
    if window_length < 3 or window_length % 2 != 1 or not 0 <= polyorder < window_length:
        raise ValueError("Invalid Savitzky-Golay window or polynomial order")
    checked = poses[valid]
    if (
        not np.isfinite(checked).all()
        or not np.allclose(checked[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(
            checked[:, :3, :3].transpose(0, 2, 1) @ checked[:, :3, :3], np.eye(3), atol=1e-3, rtol=0
        )
        or not np.allclose(np.linalg.det(checked[:, :3, :3]), 1, atol=1e-3, rtol=0)
    ):
        raise ValueError("Valid source poses must be finite proper rigid transforms")
    result = poses.copy()
    indices = np.flatnonzero(valid)
    breaks = np.flatnonzero((np.diff(indices) != 1) | (np.diff(frame_ids[indices]) != 1)) + 1
    for run in np.split(indices, breaks):
        if len(run) >= window_length:
            result[run] = np.asarray(
                smooth_pose_matrices(
                    list(poses[run]), window_length=window_length, polyorder=polyorder
                )
            )
    if preserve_initial_pose and len(valid) and valid[0]:
        result[0] = poses[0]
    if not np.isfinite(result[valid]).all():
        raise ValueError("Smoothing produced nonfinite poses")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == args.input.resolve() or output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Write a fresh output; preserve the original trajectory")
    config = yaml.safe_load(args.config.read_text())["postprocessing"]["object_trajectory"]
    if config["method"] != "savgol" or not config["enabled"]:
        raise ValueError("This command requires enabled savgol object postprocessing")
    settings = {k: config[k] for k in ("window_length", "polyorder", "preserve_initial_pose")}
    with np.load(args.input, allow_pickle=False) as data:
        poses, valid, frames = data["T_camera_object"], data["valid"], data["frame_ids"]
        fps = float(data["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("A positive original source FPS is required")
        result = smooth_trajectory(poses, valid, frames, **settings)
        dropped = [
            k for k in data.files if k not in ("T_camera_object", "valid", "frame_ids", "fps")
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, T_camera_object=result, valid=valid, frame_ids=frames, fps=fps)
    import grail.pose_est.utils as utils

    def binding(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    report = {
        "kind": "postprocessed_object_trajectory",
        "method": "grail.pose_est.utils.smooth_pose_matrices",
        "settings": settings,
        "frames": len(frames),
        "fps": fps,
        "raw_output_preserved": True,
        "missing_pose_fill": False,
        "dropped_source_state_fields": dropped,
        "scope": "Offline smoothing; no new tracking or contact enforcement. Scores and live state remain in the raw input only.",
        "bindings": [binding(p) for p in (args.input, args.config, __file__, utils.__file__)],
        "output": binding(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": len(frames), "fps": fps}))


if __name__ == "__main__":
    main()
