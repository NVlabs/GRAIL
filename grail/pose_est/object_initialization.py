"""Default dynamic-object initialization: MoSE3 proposals, native FP, then smoothing.

The proposal subprocess owns its model/runtime and exits before FoundationPose
loads. All intermediates are retained; only a complete trajectory is exported to
the reconstruction's poses_in_cam.pkl contract.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml

from grail.adapters.mose3_foundationpose import load_runtime, track_hybrid, valid_pose
from grail.core.io import load_init_rendering_data, save_object_pose_data
from grail.postprocessing.object_trajectory import smooth_trajectory

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PRESET = REPO / "configs/object_tracking/mose3_fp_hybrid.yaml"


def binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def resolve_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def resolve_runtime_python(value):
    """Resolve checkout-relative runtime paths without dereferencing a venv Python."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else REPO / path))


def load_settings(path=DEFAULT_PRESET):
    path = resolve_path(path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Object initialization preset must be a mapping")
    for key in ("runtime", "proposals", "pnp", "hybrid", "postprocessing"):
        if key not in config:
            raise ValueError(f"Missing object initialization section: {key}")
    return config


def rgb_frames(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode source video: {video}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def export_initialization(raw, valid, frames, fps, output, settings):
    """Retain raw poses even on failure; export no invalid sequence to Stage 1."""
    raw, valid, frames = np.asarray(raw), np.asarray(valid), np.asarray(frames)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Original FPS must be finite and positive")
    if (
        raw.ndim != 3
        or raw.shape[1:] != (4, 4)
        or len(raw) == 0
        or valid.shape != (len(raw),)
        or valid.dtype != bool
        or not np.issubdtype(frames.dtype, np.integer)
        or not np.array_equal(frames, np.arange(len(raw)))
    ):
        raise ValueError("Expected complete source-order poses, boolean validity and frame IDs")
    if any(not valid_pose(pose) for pose in raw[valid]):
        raise ValueError("A valid source pose is not a finite proper rigid transform")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "poses_raw.npz"
    pose_path = output / "poses_in_cam.pkl"
    processed_path = output / "poses_initialization.npz"
    if any(p.exists() for p in (raw_path, pose_path, processed_path)):
        raise FileExistsError("Use a fresh initialization directory; preserve previous poses")
    np.savez_compressed(raw_path, T_camera_object=raw, valid=valid, frame_ids=frames, fps=fps)
    if not np.asarray(valid).all():
        raise ValueError("Hybrid has invalid source frames; raw failure retained, no HOI export")
    smooth = settings["postprocessing"]["object_trajectory"]
    if smooth["method"] != "savgol":
        raise ValueError("Only repository Savitzky-Golay postprocessing is supported")
    poses = (
        smooth_trajectory(
            raw,
            valid,
            frames,
            window_length=smooth["window_length"],
            polyorder=smooth["polyorder"],
            preserve_initial_pose=smooth["preserve_initial_pose"],
        )
        if smooth["enabled"]
        else np.asarray(raw).copy()
    )
    np.savez_compressed(
        processed_path, T_camera_object=poses, valid=valid, frame_ids=frames, fps=fps
    )
    save_object_pose_data(list(poses), str(pose_path))
    return poses


def initialize(video, mesh_file, scene, masks, preset=DEFAULT_PRESET):
    """Run the adopted dynamic-object initializer on the original source timeline."""
    video, mesh_file, scene, masks = map(
        lambda p: Path(p).resolve(), (video, mesh_file, scene, masks)
    )
    preset = resolve_path(preset)
    settings = load_settings(preset)
    runtime = settings["runtime"]
    # Do not resolve a venv executable symlink: Python locates pyvenv.cfg through
    # the invoked path, and its base interpreter has different dependencies.
    python = resolve_runtime_python(os.environ.get("GRAIL_MOSE3_PYTHON", runtime["mose3_python"]))
    mose_root = resolve_path(os.environ.get("GRAIL_MOSE3_ROOT", runtime["mose3_root"]))
    checkpoint = resolve_path(os.environ.get("GRAIL_MOSE3_CHECKPOINT", runtime["mose3_checkpoint"]))
    fp_root = resolve_path(
        os.environ.get("GRAIL_FOUNDATIONPOSE_ROOT", runtime["foundationpose_root"])
    )
    for path in (
        python,
        mose_root / "mose3/models/mose3.py",
        checkpoint / "config.json",
        fp_root / "estimater.py",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Object initialization dependency missing: {path}")
    cap = cv2.VideoCapture(str(video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0 or count < 1:
        raise ValueError("Original source frame count and FPS are required")
    output = scene / "pose_estimation_output"
    if output.exists():
        raise FileExistsError(f"Use a fresh scene/output directory: {output}")
    output.mkdir(parents=True)
    start = time.monotonic()
    first_pose = scene / "first_frame_pose.pickle"
    K_path = scene / "cam_K.txt"
    obj_R, obj_t, scale, cam_R, cam_t, _ = load_init_rendering_data(first_pose)
    # Same Blender camera convention as the existing FoundationPose adapter.
    import torch

    from grail.rendering.camera import world_to_camera_matrix

    camera = world_to_camera_matrix(
        torch.from_numpy(cam_R).float(), torch.from_numpy(cam_t).float()
    ).numpy()
    obj_pose = np.eye(4)
    obj_pose[:3, :3], obj_pose[:3, 3] = obj_R, obj_t
    initial = np.diag([1.0, -1.0, -1.0, 1.0]) @ camera @ obj_pose
    K = np.loadtxt(K_path).reshape(3, 3)
    mesh = trimesh.load(str(mesh_file), force="mesh", process=False)
    mesh.apply_scale(scale)
    with np.load(masks, allow_pickle=False) as data:
        object_mask, human_mask = data["object_mask0"], data["human_mask0"]
    if object_mask.shape != (height, width) or human_mask.shape != (height, width):
        raise ValueError("Initial masks must use the original video resolution")
    init_path = output / "initialization_input.npz"
    np.savez_compressed(
        init_path,
        vertices_scaled_m=mesh.vertices,
        faces=mesh.faces,
        T_camera_object_initial=initial,
        K=K,
        object_mask0=object_mask,
        human_mask0=human_mask,
    )
    receipt = {
        "state": "running",
        "backend": "mose3_fp_hybrid",
        "settings": settings,
        "frames": count,
        "fps": fps,
        "width": width,
        "height": height,
        "object_scale": np.asarray(scale).tolist(),
        "triangle_count": len(mesh.faces),
        "bindings": [
            binding(p)
            for p in (video, mesh_file, first_pose, K_path, masks, preset, init_path, __file__)
        ],
        "runtime": {
            "mose3_python": str(python),
            "mose3_root": str(mose_root),
            "mose3_checkpoint": str(checkpoint),
            "foundationpose_root": str(fp_root),
        },
        "raw_and_processed_separate": True,
        "offline": True,
    }
    # Bind the actual native implementation and weights, not just its directory.
    native_files = sorted(
        set(fp_root.rglob("*.py"))
        | set(fp_root.rglob("*.so"))
        | {p for p in (fp_root / "weights").rglob("*") if p.is_file()}
    )
    receipt["foundationpose_bindings"] = [binding(path) for path in native_files]
    record_path = output / "initialization.json"

    def record():
        receipt["elapsed_seconds"] = time.monotonic() - start
        record_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

    record()
    try:
        proposals_dir = output / "mose3"
        command = [
            str(python),
            "-m",
            "grail.adapters.mose3_tracking",
            "--input",
            str(init_path),
            "--video",
            str(video),
            "--output-dir",
            str(proposals_dir),
            "--checkpoint",
            str(checkpoint),
            "--mose3-root",
            str(mose_root),
            "--fps",
            str(fps),
            "--pnp-config",
            str(preset),
        ]
        for key, value in settings["proposals"].items():
            command += ["--" + key.replace("_", "-"), str(value)]
        receipt["proposal_command"] = command
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with (output / "mose3.log").open("w") as log:
            subprocess.run(
                command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        with np.load(proposals_dir / "poses.npz", allow_pickle=False) as data:
            proposals, quality, ids = (
                data["T_camera_object"],
                data["quality_pass"],
                data["frame_ids"],
            )
            if (
                len(proposals) != count
                or not np.array_equal(ids, np.arange(count))
                or float(data["fps"]) != fps
            ):
                raise ValueError("MoSE3 output changed original source timeline")
        torch.set_num_threads(4)
        native = load_runtime(fp_root)
        tracked = track_hybrid(
            mesh,
            initial,
            K,
            rgb_frames(video),
            proposals,
            quality,
            settings=settings["hybrid"],
            runtime=native,
            debug_dir=output / "native_debug",
        )
        np.savez_compressed(
            output / "hybrid_hypotheses.npz",
            continuation=tracked.continuation_poses,
            refined_proposal=tracked.proposal_poses,
            scores=tracked.scores,
            selected_centered=tracked.selected_centered,
            live_centered=tracked.live_centered,
        )
        (output / "hybrid_selection.json").write_text(
            json.dumps(tracked.records, indent=2, allow_nan=False) + "\n"
        )
        export_initialization(tracked.poses, tracked.valid, ids, fps, output, settings)
        receipt.update(
            state="complete",
            valid_frames=int(tracked.valid.sum()),
            postprocessing=settings["postprocessing"]["object_trajectory"],
        )
        receipt["outputs"] = [
            binding(output / name)
            for name in (
                "poses_raw.npz",
                "poses_initialization.npz",
                "poses_in_cam.pkl",
                "hybrid_hypotheses.npz",
                "hybrid_selection.json",
            )
        ]
        receipt["code"] = [
            binding(REPO / name)
            for name in (
                "grail/adapters/mose3_foundationpose.py",
                "grail/adapters/mose3_tracking.py",
                "grail/adapters/mose3_pnp.py",
                "grail/postprocessing/object_trajectory.py",
                "grail/pose_est/utils.py",
            )
        ]
        for item in receipt["bindings"] + receipt["foundationpose_bindings"]:
            if binding(item["path"])["sha256"] != item["sha256"]:
                raise RuntimeError(f"Initialization input changed during execution: {item['path']}")
        record()
        return receipt
    except Exception as error:
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        record()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--preset", default=str(DEFAULT_PRESET))
    args = parser.parse_args()
    result = initialize(args.video, args.mesh_file, args.scene, args.masks, args.preset)
    print(json.dumps({"state": result["state"], "frames": result["frames"], "fps": result["fps"]}))


if __name__ == "__main__":
    main()
