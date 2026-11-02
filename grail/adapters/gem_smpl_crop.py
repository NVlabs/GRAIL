"""Source-bound crop policy and isolated caches for native GEM-SMPL.

The shared conservative repair changes the pose crop, never the selected person
track. An occluded/shrunken SAM2 mask cannot by itself recover an amodal body.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from grail.adapters.human_pose_crop import prepare_crop_repair
from grail.adapters.human_pose_utils import _sha256, read_video_timing
from grail.core.human_crop import HumanCropConfig, load_human_masks, repair_crops


def file_binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def request_key(request):
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prediction_request(
    video,
    vendor_root,
    checkpoint,
    model_config,
    *,
    source_timing,
    crop_request=None,
    static_cam=False,
):
    """Bind source, actual estimator assets, resolved Hydra config and crop code."""
    root = Path(vendor_root).resolve(strict=True)
    model_paths = [
        Path(checkpoint),
        *(
            root / name
            for name in (
                "inputs/checkpoints/yolo/yolov8x.pt",
                "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
                "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
                "inputs/checkpoints/vimo/vimo_checkpoint.pth.tar",
            )
        ),
    ]
    model_paths += [
        p
        for p in (root / "inputs/checkpoints/body_models").rglob("*")
        if p.is_file() and p.suffix in (".pt", ".pkl", ".npz", ".npy")
    ]
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("gem_smpl.py"),
        root / "tools/demo/demo_slam.py",
    ]
    for name in ("hmr4d", "motiondiff"):
        code_paths += [
            p
            for p in (root / name).rglob("*")
            if p.is_file() and p.suffix in (".py", ".yaml", ".yml")
        ]
    return {
        "schema": "grail.gem_smpl.prediction_request.v1",
        "source": file_binding(video),
        "source_timing": source_timing,
        "static_cam": bool(static_cam),
        "model_config": model_config,
        "models": [file_binding(p) for p in sorted(set(model_paths))],
        "code": [file_binding(p) for p in sorted(set(code_paths))],
        "human_crop_repair": crop_request,
    }


def prepare_run_directory(cache, request):
    """Old unbound caches and different policies never share preprocessed data."""
    directory = Path(cache).resolve() / "gem_smpl_runs" / request_key(request)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request.json"
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise ValueError("GEM-SMPL cache request changed")
    elif any(directory.iterdir()):
        raise ValueError("GEM-SMPL run contains unbound cached files")
    else:
        write_json(path, request)
    return directory


def completed_cache(directory, request):
    path = Path(directory) / "completion.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if (
        record.get("request_sha256") != request_key(request)
        or record.get("status") != "PREDICTION_COMPLETE"
    ):
        raise ValueError("GEM-SMPL completion does not match the requested inputs/policy")
    for role, binding in record.get("outputs", {}).items():
        if file_binding(binding["path"]) != binding:
            raise ValueError(f"Changed GEM-SMPL cached output: {role}")
    required = {
        "prediction",
        "input_video",
        "original_bbox",
        "crop_report",
        "bbx",
        "vitpose",
        "vit_features",
        "vimo_pred",
        "hmr4d_results",
    }
    if not request.get("static_cam", True):
        required.update(("slam", "cam_int"))
    if not required.issubset(record.get("outputs", {})):
        raise ValueError("GEM-SMPL completion lacks required bound outputs")
    return True


def preprocess_cache_paths(cfg, request):
    """Vendor inputs reused by preprocessing and native camera-motion loading."""
    paths = {
        name: Path(getattr(cfg.paths, name))
        for name in ("bbx", "vitpose", "vit_features", "vimo_pred")
    }
    if not request.get("static_cam", True):
        paths["slam"] = Path(cfg.paths.slam)
        paths["cam_int"] = paths["slam"].with_name("cam_int.npy")
    return paths


def validate_preprocess_cache(cfg, request):
    """Interrupted, unbound feature files cannot masquerade as valid crops."""
    path = Path(cfg.output_dir) / "preprocess_completion.json"
    paths = preprocess_cache_paths(cfg, request)
    # Original/effective boxes have their own receipts and can safely precede
    # preprocessing. Camera/features from an interrupted run need a receipt.
    feature_paths = [value for name, value in paths.items() if name != "bbx"]
    if not path.exists():
        if any(p.exists() for p in feature_paths) or Path(cfg.paths.hmr4d_results).exists():
            raise ValueError("Unbound partial GEM-SMPL feature cache; use a new cache directory")
        return
    record = json.loads(path.read_text())
    if record.get("request_sha256") != request_key(request):
        raise ValueError("Preprocessing cache differs from crop/input policy")
    expected = set(paths)
    if set(record.get("outputs", {})) != expected:
        raise ValueError("Incomplete GEM-SMPL preprocessing receipt")
    for name, original in record["outputs"].items():
        if file_binding(paths[name]) != original:
            raise ValueError(f"Changed crop-dependent preprocessing cache: {name}")


def record_preprocess_cache(cfg, request):
    write_json(
        Path(cfg.output_dir) / "preprocess_completion.json",
        {
            "request_sha256": request_key(request),
            "outputs": {
                name: file_binding(path)
                for name, path in preprocess_cache_paths(cfg, request).items()
            },
        },
    )


def repair_native_crops(raw, request):
    """Use source-rate SAM2 masks directly; GEM-SMPL retains the source frames."""
    import torch

    current = prepare_crop_repair(
        request["source_video"],
        request["masks_path"],
        request["receipt_path"],
        timing=request["source_timing"],
        config=request["config"],
    )
    if current != request:
        raise ValueError("Crop policy or source inputs changed after cache binding")
    masks = load_human_masks(request["masks_path"], human_id=request["human_id"])
    timing = request["source_timing"]
    if len(masks) != timing["frame_count"] or any(
        mask.shape != (timing["height"], timing["width"]) for mask in masks
    ):
        raise ValueError("GEM-SMPL mask count/resolution must match the source timeline")
    original = raw["bbx_xys"]
    repaired, report = repair_crops(
        raw["bbx_xyxy"].cpu().numpy(),
        original.cpu().numpy(),
        masks,
        config=HumanCropConfig(**request["config"]),
        mask_source_indices=np.arange(len(masks)),
    )
    result = dict(raw)
    result["bbx_xys_original"] = original.clone()
    result["bbx_xys"] = torch.as_tensor(repaired, dtype=original.dtype, device=original.device)
    report.update(
        request=request,
        alignment="unchanged_source_frames",
        selected_track_unchanged=True,
        limitation="Visible-mask repair cannot recover an occluded body absent from both inputs.",
    )
    return result, report


def seed_preprocess_crops(cfg, request, tracker_factory, box_to_crop):
    """Write the crop consumed by all vendor extractors before run_preprocess."""
    import torch

    output = Path(cfg.output_dir)
    raw_path = Path(cfg.preprocess_dir) / "bbx_original.pt"
    raw_receipt = raw_path.with_suffix(".json")
    signature = request_key(request)
    if raw_path.exists():
        if not raw_receipt.exists():
            raise ValueError("Original detector boxes have no source/cache binding")
        saved = json.loads(raw_receipt.read_text())
        if saved != {"request_sha256": signature, "output": file_binding(raw_path)}:
            raise ValueError("Original detector boxes changed after cache binding")
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    else:
        tracker = tracker_factory()
        boxes = tracker.get_one_track(cfg.video_path).detach().float().cpu()
        raw = {"bbx_xyxy": boxes, "bbx_xys": box_to_crop(boxes, base_enlarge=1.2).float()}
        del tracker
        torch.save(raw, raw_path)
        write_json(raw_receipt, {"request_sha256": signature, "output": file_binding(raw_path)})
    count = request["source_timing"]["frame_count"]
    if raw["bbx_xyxy"].shape != (count, 4) or raw["bbx_xys"].shape != (count, 3):
        raise ValueError("Detector boxes differ from source frame count")
    crop = request["human_crop_repair"]
    if crop is None:
        repaired = raw
        report = {"policy": "detector_only", "selected_track_unchanged": True}
    else:
        repaired, report = repair_native_crops(raw, crop)
    path = Path(cfg.paths.bbx)
    if path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=False)
        if set(previous) != set(repaired) or any(
            not torch.equal(previous[k], repaired[k]) for k in previous
        ):
            raise ValueError("Cached preprocessing crops differ from the bound crop policy")
    else:
        torch.save(repaired, path)
    report.update(
        request_sha256=signature,
        original_bbox=file_binding(raw_path),
        effective_bbox=file_binding(path),
    )
    write_json(output / "human_crop_repair.json", report)
