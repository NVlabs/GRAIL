"""Prepare hash-bound SAM2 masks before native human pose prediction."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from grail.adapters.human_pose_crop import prepare_crop_repair
from grail.adapters.human_pose_utils import _sha256, read_video_timing
from grail.core.human_crop import load_human_masks


def mask_receipt_path(masks):
    return Path(str(masks) + ".completion.json")


def _binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def ensure_human_crop_masks(
    video, masks, object_seed, human_seed, *, device="cuda", force=False, producer=None
):
    """Reuse validated masks or run SAM2; never certify an old unbound archive.

    Both original rendered seeds are bound because the existing SAM2 producer
    tracks object ID 0 and human ID 1 together. Empty propagated human masks are
    allowed: the crop policy records a per-frame detector fallback.
    """
    masks = Path(masks)
    receipt = mask_receipt_path(masks)
    inputs = {
        "video": _binding(video),
        "gt_object_mask": _binding(object_seed),
        "gt_human_mask": _binding(human_seed),
    }
    timing = read_video_timing(video)
    if masks.is_file() and receipt.is_file() and not force:
        try:
            record = json.loads(receipt.read_text())
            prepare_crop_repair(video, masks, receipt, timing=timing)
            if record.get("inputs") == inputs:
                return masks, receipt
        except (ValueError, OSError, KeyError, TypeError):
            pass

    if producer is None:
        from grail.preprocessing.preprocess import preprocess_masks

        producer = preprocess_masks
    masks.parent.mkdir(parents=True, exist_ok=True)
    staging = masks.with_name(f".{masks.stem}.{uuid.uuid4().hex}.npz")
    producer(
        video_path=str(video),
        first_frame_obj_mask_path=str(object_seed),
        first_frame_human_mask_path=str(human_seed),
        cache_file=str(staging),
        device=device,
    )
    human_masks = load_human_masks(staging)
    if len(human_masks) != timing["frame_count"] or any(
        mask.shape != (timing["height"], timing["width"]) for mask in human_masks
    ):
        raise ValueError("Generated SAM2 masks differ from source count/resolution")
    if any(_binding(value["path"]) != value for value in inputs.values()):
        raise ValueError("Source video or SAM2 seeds changed during mask prediction")
    previous = [path for path in (masks, receipt) if path.exists()]
    if previous:
        archive = masks.with_name(f"{masks.name}.previous-{uuid.uuid4().hex}")
        archive.mkdir()
        for path in previous:
            path.rename(archive / path.name)
        print(f"  Preserved previous source masks: {archive}")
    staging.replace(masks)
    record = {
        "operation": "mask_predict",
        "status": "MASKS_COMPLETE_REVIEW_PENDING",
        "inputs": inputs,
        "outputs": {"masks": _binding(masks)},
        "frame_count": timing["frame_count"],
        "fps": timing["fps"],
        "source_width": timing["width"],
        "source_height": timing["height"],
        "producer": "grail.preprocessing.preprocess.preprocess_masks",
        "fresh_inference": True,
    }
    temporary = receipt.with_suffix(receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(receipt)
    return masks, receipt


def prepare_pipeline_human_crop(video, model_config, args, video_id, origin_id):
    """Resolve the default crop input without changing the depth-stage order."""
    cfg = dict(model_config)
    if cfg.get("body_model") in ("smplx", "g1_smplx"):
        prefix = "gem_smpl"
    else:
        prefix = cfg.get("human_pose_backend", "gem_x")
        if prefix != "gem_x":
            raise ValueError(f"Unsupported SOMA03 human_pose_backend: {prefix!r}")
    key = f"{prefix}_human_crop"
    enabled = cfg.get(f"{key}_repair", prefix == "gem_x")
    if type(enabled) is not bool:
        raise ValueError(f"{key}_repair must be a boolean")
    if not enabled:
        return cfg
    explicit_masks = cfg.get(f"{key}_masks")
    if explicit_masks:
        prepare_crop_repair(
            video,
            explicit_masks,
            cfg.get(f"{key}_receipt"),
            config=cfg.get(f"{key}_config"),
        )
        prepared = getattr(args, "_human_crop_masks_prepared", {})
        prepared[video_id] = str(Path(explicit_masks).resolve())
        args._human_crop_masks_prepared = prepared
        return cfg
    if cfg.get(f"{key}_receipt"):
        raise ValueError(f"{key}_receipt requires {key}_masks")
    masks = Path(args.results_dir) / args.recon_cache_dir / "masks" / f"{video_id}.npz"
    seeds = Path(args.results_dir) / args.foundation_pose_dir / origin_id
    masks, receipt = ensure_human_crop_masks(
        video,
        masks,
        seeds / "masks/000000.png",
        seeds / "human_masks/000000.png",
        device=args.device,
        force=not args.skip_done,
    )
    cfg[f"{key}_masks"] = str(masks.resolve())
    cfg[f"{key}_receipt"] = str(receipt.resolve())
    prepared = getattr(args, "_human_crop_masks_prepared", {})
    prepared[video_id] = str(masks.resolve())
    args._human_crop_masks_prepared = prepared
    return cfg


def masks_prepared_this_invocation(args, video_id, masks):
    return getattr(args, "_human_crop_masks_prepared", {}).get(video_id) == str(
        Path(masks).resolve()
    )
