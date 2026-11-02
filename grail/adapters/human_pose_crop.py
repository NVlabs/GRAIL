"""Provenance and native-timeline bridge for optional SAM2 human crop repair."""

from __future__ import annotations

import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from grail.adapters.human_pose_utils import _sha256, read_video_timing
from grail.core.human_crop import HumanCropConfig, load_human_masks, repair_crops


def ffmpeg_source_indices(source, native):
    """Match ffmpeg ``fps=30:round=near`` (round source PTS, then hold).

    This differs from nearest-source-time sampling at 24→30 Hz. Integer PTS
    preserve exact half-tick rounding; floating ffprobe display times do not.
    The rational CFR fallback supports older timing records without integer PTS.
    """
    native_rate = Fraction(native.get("frame_rate", str(native["fps"]))).limit_denominator(1000000)
    if "frame_pts" in source and "time_base" in source:
        origin = source["frame_pts"][0]
        source_times = [
            Fraction(pts - origin) * Fraction(source["time_base"]) for pts in source["frame_pts"]
        ]
    else:
        source_rate = Fraction(source.get("frame_rate", str(source["fps"]))).limit_denominator(
            1000000
        )
        source_times = [Fraction(i) / source_rate for i in range(source["frame_count"])]
    quantized = []
    for timestamp in source_times:
        value = timestamp * native_rate + Fraction(1, 2)
        quantized.append(value.numerator // value.denominator)
    target_times = np.asarray(native["timestamps"]) - native["timestamps"][0]
    native_ticks = np.rint(target_times * float(native_rate)).astype(np.int64)
    return (np.searchsorted(quantized, native_ticks, side="right") - 1).clip(
        0, len(source_times) - 1
    )


def prepare_crop_repair(video, masks, receipt=None, *, timing=None, config=None):
    """Bind completed, original-source SAM2 masks before an estimator cache lookup.

    ``completion.json`` is produced by grail.preprocessing.ord_masks. Old or
    unbound masks are rejected instead of silently changing model inputs.
    """
    video = Path(video).resolve(strict=True)
    masks = Path(masks).resolve(strict=True)
    receipt = Path(receipt or masks.with_name("completion.json")).resolve(strict=True)
    record = json.loads(receipt.read_text())
    timing = timing or read_video_timing(video)
    if (
        record.get("operation") != "mask_predict"
        or record.get("status") != "MASKS_COMPLETE_REVIEW_PENDING"
    ):
        raise ValueError("Crop repair requires a completed SAM2 mask prediction receipt")
    video_hash, mask_hash = _sha256(video), _sha256(masks)
    if record.get("inputs", {}).get("video", {}).get("sha256") != video_hash:
        raise ValueError("SAM2 human masks belong to a different source video")
    if record.get("outputs", {}).get("masks", {}).get("sha256") != mask_hash:
        raise ValueError("SAM2 human mask bytes do not match their completion receipt")
    if (
        record.get("frame_count") != timing["frame_count"]
        or not np.isclose(record.get("fps", 0), timing["fps"], rtol=0, atol=1e-6)
        or record.get("source_width") != timing["width"]
        or record.get("source_height") != timing["height"]
    ):
        raise ValueError("SAM2 human masks do not match source count/FPS/resolution")
    policy = HumanCropConfig(**(config or {}))
    core = Path(__file__).resolve().parents[1] / "core/human_crop.py"
    return {
        "schema": "grail.human_pose.human_crop_request.v1",
        "source_video": str(video),
        "source_sha256": video_hash,
        "masks_path": str(masks),
        "masks_sha256": mask_hash,
        "receipt_path": str(receipt),
        "receipt_sha256": _sha256(receipt),
        "source_timing": timing,
        "human_id": 1,
        "config": asdict(policy),
        "bridge_sha256": _sha256(__file__),
        "utils_sha256": _sha256(Path(__file__).with_name("human_pose_utils.py")),
        "core_sha256": _sha256(core),
        "module_root": str(Path(__file__).resolve().parents[2]),
    }


def apply_crop_repair(conditions, clip, request, output_dir):
    """Repair the exact crops consumed by GEM-X Pose and native77 observations.

    Masks remain on their original source timeline; ffmpeg's source-PTS rounding
    maps them to the native 30 Hz video. No interpolation of binary masks and
    no modification of the selected camera-mask boxes or estimated camera.
    """
    import torch

    current = prepare_crop_repair(
        request["source_video"],
        request["masks_path"],
        request["receipt_path"],
        timing=request["source_timing"],
        config=request["config"],
    )
    if current != request:
        raise ValueError("Crop-repair inputs or implementation changed after cache binding")
    masks = load_human_masks(request["masks_path"], human_id=request["human_id"])
    source = request["source_timing"]
    if len(masks) != source["frame_count"]:
        raise ValueError("SAM2 mask frame count differs from completion receipt")
    if any(np.asarray(mask).shape != (source["height"], source["width"]) for mask in masks):
        raise ValueError("SAM2 masks differ from original source resolution")
    native = read_video_timing(clip)
    if (native["width"], native["height"]) != (
        source["width"] + source["width"] % 2,
        source["height"] + source["height"] % 2,
    ):
        raise ValueError("Native clip changed source resolution beyond even-size padding")
    target_times = np.asarray(native["timestamps"]) - native["timestamps"][0]
    source_times = np.asarray(source["timestamps"]) - source["timestamps"][0]
    if len(target_times) != len(conditions["bbx_xys"]):
        raise ValueError("Native timing and tracked crop counts differ")
    # ffmpeg fps conversion may extend less than one source frame at the tail.
    if target_times[-1] > source_times[-1] + 1 / source["fps"] + 1e-5:
        raise ValueError("Native clip exceeds the bound SAM2 source timeline")
    indices = ffmpeg_source_indices(source, native)
    aligned = [masks[int(index)] for index in indices]
    if native["width"] != source["width"] or native["height"] != source["height"]:
        aligned = [
            np.pad(
                mask,
                ((0, native["height"] - source["height"]), (0, native["width"] - source["width"])),
            )
            for mask in aligned
        ]
    original = conditions["bbx_xys"]
    repaired, report = repair_crops(
        conditions["camera_mask_xyxy"].cpu().numpy(),
        original.cpu().numpy(),
        aligned,
        config=HumanCropConfig(**request["config"]),
        mask_source_indices=indices,
    )
    report.update(
        {
            "request": request,
            "native_timing": native,
            "mask_source_indices": indices.tolist(),
            "alignment": "ffmpeg_fps_round_near_source_pts_then_hold",
            "camera_mask_xyxy_unchanged": True,
        }
    )
    Path(output_dir, "human_crop_repair.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    result = dict(conditions)
    result["bbx_xys_original"] = original.clone()
    result["bbx_xys"] = torch.as_tensor(repaired, dtype=original.dtype, device=original.device)
    return result
