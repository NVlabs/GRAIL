"""GRAIL-owned orchestration of public GEM-X components in an isolated process.

The only variation from GEM-X's body-video frontend is an optional, provenance-
validated crop adjustment between tracking and observation/feature extraction.
The upstream checkout is never patched or shadowed with legacy GEM packages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def _installed_soma_source():
    """Bind the installed FK/postprocess implementation, not only its version."""
    spec = importlib.util.find_spec("soma")
    if spec is None or spec.origin is None:
        raise RuntimeError("GEM-X postprocessing requires the installed SOMA source package")
    root = Path(spec.origin).resolve(strict=True).parent
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(
            part in {"__pycache__", ".git", ".pytest_cache"}
            for part in path.relative_to(root).parts
        )
        and path.suffix not in {".pyc", ".pyo"}
    }
    if "__init__.py" not in files:
        raise RuntimeError("GEM-X postprocessing requires an inspectable SOMA source package")
    return {
        "root": str(root),
        "files": files,
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def describe(request):
    from gem.inference.body_video import load_body_video_config

    config = load_body_video_config(request["config"])
    if request["static_camera"]:
        config.setdefault("camera", {})["mode"] = "static"
        config["camera"].pop("rotations", None)
    inputs = {}
    for section, keys in (
        ("pose", ("checkpoint",)),
        ("observations", ("checkpoint",)),
        ("detector", ("checkpoint",)),
        ("motion", ("checkpoint", "stats")),
        ("camera", ("rotations",)),
    ):
        for key in keys:
            value = config.get(section, {}).get(key)
            if value is not None:
                inputs[f"{section}.{key}"] = str(Path(value).resolve(strict=True))
    if request["postprocess"]:
        assets = Path(request["soma_assets"]).resolve(strict=True)
        for name in (
            "SOMA_neutral.npz",
            "SOMA_template_rig.usda",
            "SOMA_procedural_transforms.json",
            "correctives_model.pt",
        ):
            inputs[f"soma.{name}"] = str((assets / name).resolve(strict=True))
    versions = {}
    for package in (
        "torch",
        "numpy",
        "scipy",
        "transformers",
        "py-soma-x",
        "onnxruntime",
        "onnxruntime-gpu",
        "omegaconf",
        "opencv-python-headless",
        "einops",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": versions,
    }
    if request["postprocess"]:
        runtime["soma_source"] = _installed_soma_source()
    return {
        "config": config,
        "input_files": inputs,
        "postprocess": request["postprocess"],
        "device": request["device"],
        "runtime": runtime,
    }


def build_conditions(frames, config, *, device, normalized_video, crop_request, output_dir):
    """Use upstream stage APIs, retaining original tracking/camera-mask boxes."""
    from dataclasses import asdict

    import numpy as np
    import torch
    from gem.common.detectors import YOLOXONNX, detect_track
    from gem.common.pose2d import load_native77_detector
    from gem.inference.body_video import body_release, extract_body_features, motion_intrinsics
    from gem.inference.camera_motion import camera_motion
    from gem.motion.api import MotionConditionSpec
    from gem.motion.frontend.joint_sets import get_observation_schema
    from gem.pose.api import load_pose_checkpoint

    from grail.adapters.gem_x_soma import (
        OBSERVATION_SCHEMA,
        OBSERVATION_SOURCE,
        POSE_FEATURE_RELEASE,
    )

    expected = MotionConditionSpec(**config["motion"]["condition_spec"])
    release = body_release(config)
    actual = MotionConditionSpec(
        f"{release.pose_release_id}:pose_tokens",
        config["observations"]["release_id"],
        expected.observation_joint_set,
    )
    canonical = MotionConditionSpec(POSE_FEATURE_RELEASE, OBSERVATION_SOURCE, OBSERVATION_SCHEMA)
    schema = get_observation_schema(expected.observation_joint_set)
    if actual != expected or expected != canonical or schema.num_joints != 77:
        raise ValueError("GRAIL GEM-X requires the released native77 detector/feature contract")
    camera_config = config.get("camera", {})
    intrinsics = motion_intrinsics(frames.width, frames.height, camera_config.get("intrinsics"))
    print("Detecting and tracking primary person", flush=True)
    options = dict(config["detector"])
    detector = YOLOXONNX(options.pop("checkpoint"), **options)
    tracks = detect_track(frames, detector, **config.get("tracking", {}))
    del detector
    boxes = torch.from_numpy(tracks.boxes_xys).clone()
    if crop_request is not None:
        from grail.adapters.human_pose_crop import apply_crop_repair

        repaired = apply_crop_repair(
            {"bbx_xys": boxes, "camera_mask_xyxy": torch.from_numpy(tracks.boxes_xyxy)},
            normalized_video,
            crop_request,
            output_dir,
        )
        boxes = repaired["bbx_xys"]
    print("Extracting native77 observations", flush=True)
    observation_config = config["observations"]
    detector = load_native77_detector(
        observation_config["checkpoint"],
        release_id=observation_config["release_id"],
        device=device,
        batch_size=observation_config.get("batch_size", 4),
    )
    keypoints = detector.predict(frames, boxes)
    detector_release = detector.release_id
    del detector
    print("Estimating camera rotation", flush=True)
    rotations = camera_config.get("rotations")
    if rotations is not None:
        rotations = np.load(rotations)
    camera = camera_motion(
        frames,
        intrinsics.numpy(),
        tracks.boxes_xyxy,
        mode=camera_config.get("mode", "background"),
        rotations=rotations,
        keypoints=keypoints.numpy(),
        **camera_config.get("options", {}),
    )
    print("Extracting body Pose features", flush=True)
    pose = load_pose_checkpoint(config["pose"]["checkpoint"], release, device=device)
    features, feature_version = extract_body_features(
        frames,
        boxes,
        intrinsics,
        pose,
        batch_size=config["pose"].get("batch_size", 4),
        device=device,
    )
    del pose
    actual = MotionConditionSpec(feature_version, detector_release, schema.name)
    if actual != expected:
        raise ValueError("Extracted GEM-X conditions do not match the selected motion checkpoint")
    sequence = {
        "condition_spec": asdict(actual),
        "length": torch.tensor([len(frames)]),
        "f_imgseq": features.unsqueeze(0),
        "kp2d_native77": keypoints.unsqueeze(0),
        "bbx_xys": boxes.unsqueeze(0),
        "K_fullimg": intrinsics.expand(1, len(frames), 3, 3).clone(),
        "cam_angvel": camera["cam_angvel"].unsqueeze(0),
        "frame_times": (torch.arange(len(frames), dtype=torch.float64) / 30).unsqueeze(0),
        "sampling_fps": 30.0,
    }
    return sequence, tracks, camera


def _portable(value):
    """Save ordinary CPU tensors/scalars, loadable with weights_only=True."""
    import numpy as np
    import torch

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return value


def run(request, description):
    from dataclasses import asdict

    import torch
    from gem.inference.motion import load_predictor, run_motion
    from gem.inference.video import VideoFrames, normalize_video

    output = Path(request["output_dir"])
    normalized = output / "source.mp4"
    metadata = normalize_video(request["video"], normalized, start=0.0, duration=None)
    with VideoFrames(normalized) as frames:
        sequence, tracks, camera = build_conditions(
            frames,
            description["config"],
            device=request["device"],
            normalized_video=normalized,
            crop_request=request["human_crop_repair"],
            output_dir=output,
        )
    torch.save(_portable(sequence), output / "conditions.pt")
    motion = description["config"]["motion"]
    print("Predicting native temporal Motion", flush=True)
    predictor = load_predictor(
        motion["checkpoint"], motion, stats_path=motion.get("stats"), device=request["device"]
    )
    result = run_motion(predictor, sequence, device=request["device"])
    result = {
        **result,
        "format": "gem_body_video_predictions_v1",
        "video": metadata,
        "release": description["config"],
        "tracking": asdict(tracks),
        "camera": camera,
    }
    if request["postprocess"]:
        from gem.inference.postprocess import postprocess_motion_result

        result = postprocess_motion_result(
            result, soma_assets=request["soma_assets"], device=request["device"]
        )
    torch.save(_portable(result), output / "native_prediction.pt")
    print("GEM-X native prediction saved", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    source = args.source_root.resolve(strict=True)
    if not (source / "gem/inference/body_video.py").is_file():
        raise ValueError("GEM-X source root lacks the public body-video API")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(source))
    import gem

    if not Path(gem.__file__).resolve().is_relative_to(source):
        raise RuntimeError("The isolated worker imported another GEM implementation")
    request = json.loads(args.request.read_text())
    description = describe(request)
    path = Path(request["output_dir"]) / "description.json"
    if args.describe:
        path.write_text(json.dumps(description, indent=2, allow_nan=False) + "\n")
    else:
        if json.loads(path.read_text()) != description:
            raise ValueError("GEM-X config or runtime changed after request binding")
        run(request, description)


if __name__ == "__main__":
    main()
