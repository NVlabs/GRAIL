"""Isolated GEM-X native77 inference with source-timeline and cache bindings.

This adapter does not select a reconstruction geometry runtime or
apply a fitted identity. GEM-X code and its model dependencies are imported
only in the selected worker interpreter. Historical GEP/GEM caches are separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from pathlib import Path

from grail.adapters.human_pose_utils import (
    _bundle_component_hashes,
    _linear,
    _rotations,
    _sha256,
    _tensor,
    read_video_timing,
)

ADAPTER_SCHEMA = "grail.gem_x.soma03.native77.v1"
OBSERVATION_SCHEMA = "soma77_native_v1"
OBSERVATION_SOURCE = "vitpose77-dinov3-huge-metrosim-256x192-v1"
POSE_FEATURE_RELEASE = "gem-pose-body-s240000-v1:pose_tokens"
# Native SOMA joint indices, matching Soma03HumanModel's observation geometry.
# HeadEnd is only a placeholder for unavailable nose/ears: confidence is zero.
NATIVE77_TO_COCO17 = (7, 9, 10, 7, 7, 12, 40, 13, 41, 14, 42, 67, 72, 68, 73, 69, 74)
NATIVE77_TO_HAND50 = (*range(14, 39), *range(42, 67))
# Display-compatible wrist + first three articulated joints per finger.
_LEFT_HAND16 = (14, 15, 16, 17, 19, 20, 21, 24, 25, 26, 29, 30, 31, 34, 35, 36)
NATIVE77_TO_HAND32 = (*_LEFT_HAND16, *(index + 28 for index in _LEFT_HAND16))


def _timeline(source_fps, source_frame_count, source_timestamps, native_timestamps, length):
    import numpy as np

    if (
        not math.isfinite(source_fps)
        or source_fps <= 0
        or type(source_frame_count) is not int
        or source_frame_count < 2
        or length < 2
    ):
        raise ValueError("GEM-X requires positive FPS and at least two complete frames")
    target = np.asarray(
        (
            source_timestamps
            if source_timestamps is not None
            else np.arange(source_frame_count) / source_fps
        ),
        dtype=np.float64,
    )
    native = np.asarray(native_timestamps, dtype=np.float64)
    for name, values, count, fps in (
        ("source", target, source_frame_count, source_fps),
        ("native", native, length, 30.0),
    ):
        if (
            values.shape != (count,)
            or not np.isfinite(values).all()
            or not np.allclose(np.diff(values), 1 / fps, rtol=0, atol=max(1e-5, 0.01 / fps))
        ):
            raise ValueError(f"Invalid {name} timestamps; explicit constant-rate timeline required")
    origins = float(target[0]), float(native[0])
    target, native = target - origins[0], native - origins[1]
    if abs(float(target[-1] - native[-1])) > max(1 / source_fps, 1 / 30) + 1e-5:
        raise ValueError("Native prediction does not cover the complete source video timeline")
    return target, native, origins


def normalize_prediction(
    prediction,
    *,
    source_fps,
    source_frame_count,
    source_timestamps=None,
    native_timestamps=None,
    provenance=None,
):
    """Validate native77 results and preserve all native parameters and evidence.

    The 77 observations are native joints, not Sapiens surface landmarks.
    Confidence remains the detector's raw heatmap maximum, without sigmoid or
    threshold conversion. Source resampling interpolates confidence explicitly.
    """
    import numpy as np
    import torch

    if prediction.get("format") not in {
        "gem_body_video_predictions_v1",
        "gem_motion_predictions_v1",
    }:
        raise ValueError("Expected a public GEM-X native motion result")
    expected = {
        "pose_release_id": POSE_FEATURE_RELEASE,
        "observation_source": OBSERVATION_SOURCE,
        "observation_joint_set": OBSERVATION_SCHEMA,
    }
    conditions = prediction["conditions"]
    if prediction.get("condition_spec") != expected or conditions.get("condition_spec") != expected:
        raise ValueError("GEM-X result has an incompatible native77 observation/feature contract")
    lengths = torch.as_tensor(prediction["length"])
    if lengths.shape != (1,) or lengths.dtype.is_floating_point or int(lengths[0]) < 2:
        raise ValueError("Expected exactly one unpadded native GEM-X sequence")
    length = int(lengths[0])
    if not torch.equal(torch.as_tensor(conditions["length"]), lengths):
        raise ValueError("GEM-X condition length differs from its prediction")
    if float(conditions.get("sampling_fps", 0)) != 30:
        raise ValueError("GEM-X native inference must use the explicit 30 Hz clock")
    times = torch.as_tensor(conditions["frame_times"], dtype=torch.float64)
    if times.shape != (1, length):
        raise ValueError("Invalid GEM-X native frame_times shape")
    actual_times = times[0].cpu().numpy()
    if native_timestamps is not None:
        supplied = np.asarray(native_timestamps, dtype=np.float64)
        if supplied.shape != (length,) or not np.allclose(
            supplied - supplied[0], actual_times - actual_times[0], atol=1e-5, rtol=0
        ):
            raise ValueError("Decoded native timestamps differ from the predictor clock")
    else:
        native_timestamps = actual_times
    target, native, origins = _timeline(
        source_fps, source_frame_count, source_timestamps, native_timestamps, length
    )
    unchanged = len(target) == len(native) and np.allclose(target, native, atol=1e-5, rtol=0)

    def sample(value, rotation=False):
        return (
            value.clone()
            if unchanged
            else (_rotations if rotation else _linear)(value, native, target)
        )

    result = {}
    for space in ("incam", "global"):
        raw = prediction[f"pred_soma_params_{space}"]
        body = torch.as_tensor(raw["body_pose"])
        if body.shape == (1, length, 76, 3):
            body = body.reshape(1, length, 228)
        parameters = {}
        for name, value, width in (
            ("body_pose", body, 228),
            ("global_orient", raw["global_orient"], 3),
            ("transl", raw["transl"], 3),
            ("identity_coeffs", raw["identity_coeffs"], 128),
            ("scale_params", raw["scale_params"], 57),
        ):
            parameters[name] = _tensor(value, f"{space}.{name}", (1, length, width))[0]
        scales = parameters["scale_params"]
        if (
            not torch.equal(scales[:, 0], torch.ones_like(scales[:, 0]))
            or (scales[:, 1:] <= 0).any()
        ):
            raise ValueError("Raw GEM-X requires exact global scale one and positive local56")
        parameters = {
            key: sample(value, key in {"body_pose", "global_orient"})
            for key, value in parameters.items()
        }
        joints = parameters["body_pose"].reshape(source_frame_count, 76, 3)
        parameters["left_hand_pose"] = joints[:, 14:38].reshape(source_frame_count, 72).clone()
        parameters["right_hand_pose"] = joints[:, 42:66].reshape(source_frame_count, 72).clone()
        result[f"body_params_{space}"] = parameters
    observations = _tensor(
        conditions["kp2d_native77"], "native77 observations", (1, length, 77, 3)
    )[0]
    mapped = sample(observations)
    result["native77_keypoints_2d"] = mapped
    result["vitpose"] = mapped[:, NATIVE77_TO_COCO17].clone()
    result["vitpose"][:, [0, 3, 4], 2] = 0
    result["soma_hand_keypoints_2d"] = mapped[:, NATIVE77_TO_HAND50].clone()
    result["hand_keypoints_2d"] = mapped[:, NATIVE77_TO_HAND32].clone()
    result["native_observations"] = {
        "schema": OBSERVATION_SCHEMA,
        "source": OBSERVATION_SOURCE,
        "fps": 30.0,
        "keypoints_2d": observations.clone(),
        "confidence": "raw_heatmap_maximum",
    }
    result["K_fullimg"] = sample(
        _tensor(conditions["K_fullimg"], "K_fullimg", (1, length, 3, 3))[0]
    )
    logits = _tensor(prediction["static_conf_logits"], "static_conf_logits", (1, length, 6))[0]
    result["foot_contact_probs"] = sample(logits[:, :4].sigmoid())
    result["mocap_frame_rate"] = float(source_fps)
    result["provenance"] = {
        **(provenance or {}),
        "adapter_schema": ADAPTER_SCHEMA,
        "estimator": "GEM-X",
        "model_id": "gem_x_native77",
        "body_model": "soma03",
        "identity_layout": "identity128_scale57_fixed_gs1",
        "coordinate_frame_global": "world_y_up",
        "native_coordinates": prediction.get("coordinates"),
        "units": "meters",
        "observation_source": OBSERVATION_SOURCE,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_confidence": "raw_heatmap_maximum",
        "hand_observations_available": True,
        "observed_hand_landmarks": "native_wrist_and_24_finger_joints_per_hand",
        "postprocessing": prediction.get("postprocessing", False),
        "native_fps": 30.0,
        "native_frame_count": length,
        "source_fps": float(source_fps),
        "source_frame_count": source_frame_count,
        "source_first_pts_seconds": origins[0],
        "native_first_pts_seconds": origins[1],
        "source_timestamps_seconds": target.tolist(),
        "native_timestamps_seconds": native.tolist(),
        "resampling": (
            "none" if unchanged else "SO3_shortest_arc_linear_translation_shape_observations"
        ),
        "tail_clamped_source_indices": np.flatnonzero(target > native[-1] + 1e-5).tolist(),
        "timeline_origin": "first_decoded_frame",
    }
    return result


def prediction_to_motion(result):
    """Build the existing SOMA03 motion format without a fitted identity swap."""
    import torch

    output = {}
    for space in ("incam", "global"):
        parameters = result[f"body_params_{space}"]
        motion = {
            "poses": torch.cat((parameters["global_orient"], parameters["body_pose"]), dim=-1),
            "trans": parameters["transl"],
            **{
                key: parameters[key]
                for key in ("identity_coeffs", "scale_params", "left_hand_pose", "right_hand_pose")
            },
            "gender": "neutral",
            "mocap_frame_rate": result["mocap_frame_rate"],
            "model": "soma03",
            "model_type": "soma03",
            "estimator_provenance": result["provenance"],
        }
        if space == "incam":
            motion.update(
                {
                    key: result[key]
                    for key in (
                        "vitpose",
                        "hand_keypoints_2d",
                        "soma_hand_keypoints_2d",
                        "foot_contact_probs",
                        "K_fullimg",
                        "native77_keypoints_2d",
                        "native_observations",
                    )
                }
            )
        output[f"motion_{space}"] = motion
    return output


def _worker_command(python, source_root, request_path, *, describe=False):
    return [
        str(python),
        str(Path(__file__).with_name("gem_x_worker.py")),
        "--source-root",
        str(source_root),
        "--request",
        str(request_path),
        *(["--describe"] if describe else []),
    ]


def _native_cache_intact(result):
    """A converted cache retains its independently inspectable native evidence."""
    provenance = result.get("provenance", {})
    try:
        root = Path(provenance["native_output_dir"])
        return all(
            _sha256(root / name) == provenance[key]
            for name, key in (
                ("native_prediction.pt", "native_prediction_sha256"),
                ("conditions.pt", "native_conditions_sha256"),
                ("source.mp4", "native_source_sha256"),
            )
        )
    except (KeyError, OSError, ValueError):
        return False


def _signature(video, source_root, config_path, python, description, timing, crop_request):
    paths = {name: Path(value) for name, value in description["input_files"].items()}
    return {
        "adapter_schema": ADAPTER_SCHEMA,
        "adapter_sha256": _sha256(__file__),
        "worker_sha256": _sha256(Path(__file__).with_name("gem_x_worker.py")),
        "timeline_helpers_sha256": _sha256(Path(__file__).with_name("human_pose_utils.py")),
        "source_sha256": _sha256(video),
        "source_timing": timing,
        "source_root": str(source_root),
        "source_components": _bundle_component_hashes({"gem": source_root / "gem"}),
        "config_sha256": _sha256(config_path),
        "python": str(python),
        "runtime": description["runtime"],
        "config": description["config"],
        "postprocess": description["postprocess"],
        "device": description["device"],
        "model_files_sha256": {name: _sha256(path) for name, path in paths.items()},
        "human_crop_repair": crop_request,
    }


def infer_human_pose(
    video_path,
    cache_dir,
    *,
    source_root,
    config_path,
    python,
    device="cuda",
    is_static_cam=True,
    human_crop_repair=True,
    human_crop_masks=None,
    human_crop_receipt=None,
    human_crop_config=None,
    postprocess=False,
    soma_assets=None,
):
    """Run upstream components in an isolated interpreter and content-bound cache.

    Static-camera mode is an explicit override. False preserves the release
    configuration's background/provided camera mode. The selected GEM-X source
    tree is never modified. No fallback to old inference or stale outputs occurs.
    """
    import torch

    for name, value in (
        ("is_static_cam", is_static_cam),
        ("human_crop_repair", human_crop_repair),
        ("postprocess", postprocess),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if human_crop_repair and not human_crop_masks:
        raise ValueError("SAM2 crop repair requires source-bound human_crop_masks and receipt")
    if postprocess and not soma_assets:
        raise ValueError("GEM-X Contact/IK requires explicit matching soma_assets")
    video = Path(video_path).expanduser().resolve(strict=True)
    source = Path(source_root).expanduser().resolve(strict=True)
    config = Path(config_path).expanduser().resolve(strict=True)
    interpreter = Path(python).expanduser().absolute()
    if not interpreter.is_file():
        raise FileNotFoundError(interpreter)
    cache = Path(cache_dir).expanduser().resolve()
    timing = read_video_timing(video)
    if timing["frame_count"] / timing["fps"] > 60:
        raise ValueError("GEM-X accepts at most 60 seconds per complete source clip")
    crop = None
    if human_crop_repair:
        from grail.adapters.human_pose_crop import prepare_crop_repair

        crop = prepare_crop_repair(
            video, human_crop_masks, human_crop_receipt, timing=timing, config=human_crop_config
        )
    cache.mkdir(parents=True, exist_ok=True)
    run_dir = cache / f"native_gem_x_{uuid.uuid4().hex}"
    run_dir.mkdir()
    request = {
        "video": str(video),
        "config": str(config),
        "output_dir": str(run_dir),
        "device": device,
        "static_camera": is_static_cam,
        "postprocess": postprocess,
        "soma_assets": str(Path(soma_assets).resolve(strict=True)) if soma_assets else None,
        "human_crop_repair": crop,
    }
    request_path = run_dir / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    with (run_dir / "describe.log").open("w") as log:
        subprocess.run(
            _worker_command(interpreter, source, request_path, describe=True),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    description = json.loads((run_dir / "description.json").read_text())
    signature = _signature(video, source, config, interpreter, description, timing, crop)
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    converted = cache / f"gem_x_soma03_{fingerprint}.pt"
    seal_path = converted.with_suffix(".json")
    if converted.is_file() and seal_path.is_file():
        seal = json.loads(seal_path.read_text())
        if seal.get("fingerprint") == fingerprint and seal.get("sha256") == _sha256(converted):
            result = torch.load(converted, map_location="cpu", weights_only=True)
            if result.get("provenance", {}).get(
                "fingerprint"
            ) == fingerprint and _native_cache_intact(result):
                return result
    with (run_dir / "worker.log").open("w") as log:
        subprocess.run(
            _worker_command(interpreter, source, request_path),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if _signature(video, source, config, interpreter, description, timing, crop) != signature:
        raise ValueError("GEM-X inputs or implementation changed during inference")
    native_path = run_dir / "native_prediction.pt"
    native_result = torch.load(native_path, map_location="cpu", weights_only=True)
    native_timing = read_video_timing(run_dir / "source.mp4")
    result = normalize_prediction(
        native_result,
        source_fps=timing["fps"],
        source_frame_count=timing["frame_count"],
        source_timestamps=timing["timestamps"],
        native_timestamps=native_timing["timestamps"],
        provenance={
            **signature,
            "fingerprint": fingerprint,
            "native_output_dir": str(run_dir),
            "native_prediction_sha256": _sha256(native_path),
            "native_conditions_sha256": _sha256(run_dir / "conditions.pt"),
            "native_source_sha256": _sha256(run_dir / "source.mp4"),
        },
    )
    temporary = converted.with_suffix(f".{uuid.uuid4().hex}.tmp")
    torch.save(result, temporary)
    temporary.replace(converted)
    seal_path.write_text(
        json.dumps({"fingerprint": fingerprint, "sha256": _sha256(converted)}, indent=2) + "\n"
    )
    return result
