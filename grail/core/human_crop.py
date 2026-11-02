"""Deterministic, conservative SAM2-assisted human crop repair.

Only the crop passed to pose estimators changes. Person identities and camera
mask boxes remain detector outputs. SAM2 is supporting evidence, not a ground
truth body silhouette: failed geometric/temporal checks retain the input crop.
Coordinates use full-image pixels and half-open ``[x0, y0, x1, y1]`` boxes.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HumanCropConfig:
    min_component_pixels: int = 8
    min_component_area_fraction: float = 0.0001
    min_mask_pixels: int = 64
    max_mask_image_fraction: float = 0.65
    min_box_mask_overlap: float = 0.10
    max_mask_box_area_ratio: float = 12.0
    max_crop_scale: float = 3.0
    max_temporal_area_ratio: float = 2.5
    min_temporal_box_iou: float = 0.25
    temporal_radius: int = 2
    min_support_frames: int = 2
    max_uncovered_fraction: float = 0.005
    edge_tolerance_pixels: float = 3.0
    padding_fraction: float = 0.03
    smoothing_radius: int = 2
    # Opt-in offline occlusion guard. Unlike mask coverage, this uses the
    # surrounding selected-person crops as a scale prior, not a body label.
    temporal_size_floor_ratio: float = 0.0
    temporal_size_radius: int = 32
    temporal_size_min_support: int = 2
    temporal_size_center_tolerance: float = 0.5

    def __post_init__(self):
        for name in (
            "min_component_pixels",
            "min_mask_pixels",
            "min_support_frames",
            "temporal_size_min_support",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("temporal_radius", "smoothing_radius", "temporal_size_radius"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "min_component_area_fraction",
            "max_mask_image_fraction",
            "min_box_mask_overlap",
            "min_temporal_box_iou",
            "max_uncovered_fraction",
            "padding_fraction",
            "temporal_size_floor_ratio",
            "temporal_size_center_tolerance",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("max_mask_box_area_ratio", "max_crop_scale", "max_temporal_area_ratio"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if self.edge_tolerance_pixels < 0:
            raise ValueError("edge_tolerance_pixels must be nonnegative")


def load_human_masks(path: str | Path, human_id: int = 1) -> list[np.ndarray]:
    """Load an existing trusted local SAM2 ``masks.npz`` or binary ``.npy``.

    SAM2 archives store one dictionary per frame, indexed by object ID (human
    is 1). Numeric arrays must contain human-only binary masks, never a label
    image combining human and manipulated object. Empty human masks are valid
    inputs and cause an explicit per-frame fallback during crop repair.
    """
    path = Path(path)
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "masks" not in loaded:
                raise ValueError("Human mask archive requires the 'masks' key")
            sequence = loaded["masks"]
        finally:
            loaded.close()
    else:
        sequence = loaded
    if sequence.ndim == 0:
        sequence = sequence.item()
        if isinstance(sequence, dict):
            frame_keys = sorted(sequence, key=int)
            if [int(x) for x in frame_keys] != list(range(len(frame_keys))):
                raise ValueError("Mask frame IDs must be contiguous from zero")
            sequence = [sequence[key] for key in frame_keys]
    masks = []
    for entry in sequence:
        if isinstance(entry, dict):
            if human_id not in entry:
                raise ValueError(f"Human ID {human_id} missing from mask frame {len(masks)}")
            entry = entry[human_id]
        mask = np.asarray(entry)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2 or not np.isin(mask, (0, 1, 255)).all():
            raise ValueError("Expected binary human-only masks with shape [H, W]")
        masks.append(mask.astype(bool))
    if not masks or any(mask.shape != masks[0].shape for mask in masks):
        raise ValueError("Require nonempty mask sequence with consistent image size")
    return masks


def align_masks_to_timestamps(
    masks: Sequence[np.ndarray], source_fps: float, target_timestamps: Sequence[float]
) -> tuple[list[np.ndarray], np.ndarray]:
    """Nearest source-frame sampling, using explicit target times in seconds.

    Masks are discrete observations, so no interpolation of labels is used.
    A target outside the source duration by more than half a source frame is
    rejected; mismatched frame counts are never silently stretched to match.
    """
    times = np.asarray(target_timestamps, dtype=np.float64)
    if not len(masks) or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Require masks and a positive finite source fps")
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Target timestamps must be finite and monotonic")
    if np.any(times < 0) or np.any(times * source_fps > len(masks) - 0.5 + 1e-6):
        raise ValueError("Target timestamps exceed source mask coverage")
    indices = np.minimum(np.floor(times * source_fps + 0.5).astype(int), len(masks) - 1)
    return [masks[index] for index in indices], indices


def _crop_box(crop):
    return np.r_[crop[:2] - crop[2] / 2, crop[:2] + crop[2] / 2]


def _union(a, b):
    return np.r_[np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])]


def _square(box):
    return np.r_[(box[:2] + box[2:]) / 2, np.max(box[2:] - box[:2])]


def _iou(a, b):
    intersection = np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0))
    return intersection / max(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection, 1)


def _coverage(mask, box):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    inside = (xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3])
    return float(inside.mean())


def repair_crops(
    selected_boxes_xyxy: np.ndarray,
    original_crops_xys: np.ndarray,
    human_masks: Sequence[np.ndarray],
    config: HumanCropConfig | None = None,
    *,
    mask_source_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Expand incomplete square crops using temporally supported human masks.

    The selected track is never changed. Multiple sufficiently large mask
    components are retained, including legs separated by an occluding table.
    Healthy sequences are returned bit-for-bit unchanged. Repairs can only
    enlarge the original covered region, and post-smoothing enclosure prevents
    losing accepted mask pixels. Invalid/unsupported masks retain the original
    crop and appear explicitly in the JSON-safe report.
    """
    import cv2

    cfg = config or HumanCropConfig()
    boxes = np.asarray(selected_boxes_xyxy, dtype=np.float64)
    originals = np.asarray(original_crops_xys)
    count = len(originals)
    source_indices = (
        np.arange(count) if mask_source_indices is None else np.asarray(mask_source_indices)
    )
    if (
        source_indices.shape != (count,)
        or not np.issubdtype(source_indices.dtype, np.integer)
        or np.any(source_indices < 0)
        or np.any(np.diff(source_indices) < 0)
    ):
        raise ValueError("Mask source indices must be monotonic nonnegative integers [T]")
    if boxes.shape != (count, 4) or originals.shape != (count, 3) or len(human_masks) != count:
        raise ValueError("Boxes [T,4], crops [T,3] and masks must share the frame count")
    if not np.issubdtype(originals.dtype, np.floating):
        raise ValueError("Crop coordinates must be floating point")
    if not count or not np.isfinite(boxes).all() or not np.isfinite(originals).all():
        raise ValueError("Require finite nonempty crops and selected boxes")
    if np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any(originals[:, 2] <= 0):
        raise ValueError("Box extents and crop sizes must be positive")
    repaired = originals.copy()
    frames, cleaned, mask_boxes, requirements = [], [], [], []
    image_shape = np.asarray(human_masks[0]).shape
    if len(image_shape) != 2:
        raise ValueError("Human masks must have shape [H, W]")
    height, width = image_shape
    for index, raw_mask in enumerate(human_masks):
        raw = np.asarray(raw_mask)
        if raw.shape != image_shape or not np.isin(raw, (0, 1, 255)).all():
            raise ValueError("Require same-size binary human masks in source image coordinates")
        raw = (raw != 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        threshold = max(cfg.min_component_pixels, int(raw.sum() * cfg.min_component_area_fraction))
        keep = np.zeros(num, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= threshold
        mask = keep[labels]
        area = int(mask.sum())
        old_box = _crop_box(originals[index])
        record = {
            "frame": index,
            "mask_source_frame": int(source_indices[index]),
            "status": "unchanged",
            "reason": "already_covered",
            "raw_mask_pixels": int(raw.sum()),
            "mask_pixels": area,
            "removed_noise_pixels": int(raw.sum()) - area,
            "retained_components": int(keep.sum()),
            "coverage_before": _coverage(mask, old_box),
            "coverage_after": _coverage(mask, old_box),
            "original_crop_xys": originals[index].tolist(),
            "repaired_crop_xys": originals[index].tolist(),
        }
        reason = None
        mask_box = None
        requirement = None
        if area < cfg.min_mask_pixels:
            reason = "empty_or_tiny_mask"
        else:
            ys, xs = np.nonzero(mask)
            mask_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
            record["mask_box_xyxy"] = mask_box.tolist()
            detector = boxes[index]
            detector_area = np.prod(detector[2:] - detector[:2])
            overlap = np.sum(
                (xs >= detector[0]) & (xs < detector[2]) & (ys >= detector[1]) & (ys < detector[3])
            )
            record["box_mask_overlap"] = float(overlap / max(min(area, detector_area), 1))
            pad = cfg.padding_fraction * max(mask_box[2:] - mask_box[:2])
            padded = mask_box + np.array([-pad, -pad, pad, pad])
            padded = np.clip(padded, [0, 0, 0, 0], [width, height, width, height])
            requirement = _union(old_box, padded)
            if area / (height * width) > cfg.max_mask_image_fraction:
                reason = "mask_covers_too_much_image"
            elif record["box_mask_overlap"] < cfg.min_box_mask_overlap:
                reason = "mask_not_associated_with_selected_person"
            elif np.prod(mask_box[2:] - mask_box[:2]) / detector_area > cfg.max_mask_box_area_ratio:
                reason = "implausible_mask_box_extent"
            elif _square(requirement)[2] / originals[index, 2] > cfg.max_crop_scale:
                reason = "excessive_crop_expansion"
        if reason:
            record.update(status="fallback", reason=reason)
        frames.append(record)
        cleaned.append(mask)
        mask_boxes.append(mask_box)
        requirements.append(requirement)

    geometrically_valid = np.array([f["status"] != "fallback" for f in frames])
    proposals = {}
    for index, record in enumerate(frames):
        if not geometrically_valid[index]:
            continue
        old_box = _crop_box(originals[index])
        mask_box = mask_boxes[index]
        excess = max(np.max(old_box[:2] - mask_box[:2]), np.max(mask_box[2:] - old_box[2:]))
        if (
            record["coverage_before"] >= 1 - cfg.max_uncovered_fraction
            and excess <= cfg.edge_tolerance_pixels
        ):
            continue
        lo, hi = max(0, index - cfg.temporal_radius), min(count, index + cfg.temporal_radius + 1)
        # A 24-to-30 fps conversion duplicates some source masks. Duplicates
        # must not become independent temporal votes for a defective mask.
        distinct = {}
        for j in range(lo, hi):
            if geometrically_valid[j]:
                distinct.setdefault(int(source_indices[j]), j)
        neighbors = list(distinct.values())
        median_area = np.median([frames[j]["mask_pixels"] for j in neighbors])
        area_ratio = max(record["mask_pixels"] / median_area, median_area / record["mask_pixels"])
        support = sum(_iou(mask_box, mask_boxes[j]) >= cfg.min_temporal_box_iou for j in neighbors)
        record["temporal_support_frames"] = int(support)
        if area_ratio > cfg.max_temporal_area_ratio:
            record.update(status="fallback", reason="mask_area_temporal_outlier")
        elif support < cfg.min_support_frames:
            record.update(status="fallback", reason="insufficient_temporal_support")
        else:
            proposals[index] = requirements[index]

    for index, requirement in proposals.items():
        lo, hi = max(0, index - cfg.smoothing_radius), min(count, index + cfg.smoothing_radius + 1)
        # Average only accepted proposals. Failed masks never lend their extents
        # to adjacent frames; covered/invalid frames themselves remain unchanged.
        nearby = [proposals[j] for j in range(lo, hi) if j in proposals]
        envelope = _union(np.mean(nearby, axis=0), requirement)
        proposed_crop = _square(envelope)
        if proposed_crop[2] / originals[index, 2] > cfg.max_crop_scale:
            frames[index].update(status="fallback", reason="smoothed_expansion_exceeds_limit")
            continue
        repaired[index] = proposed_crop
        coverage = _coverage(cleaned[index], _crop_box(repaired[index]))
        # Float32 conversion can place a boundary fractionally inward. Expand
        # one representable unit if needed, without shrinking any input bounds.
        if coverage < 1:
            repaired[index, 2] = np.nextafter(repaired[index, 2], np.inf, dtype=repaired.dtype)
            coverage = _coverage(cleaned[index], _crop_box(repaired[index]))
        frames[index].update(
            status="repaired",
            reason="mask_outside_pose_crop",
            coverage_after=coverage,
            repaired_crop_xys=repaired[index].tolist(),
        )
    if cfg.temporal_size_floor_ratio > 0:
        _guard_temporal_crop_size(repaired, originals, frames, cleaned, source_indices, cfg)
    report = {
        "schema": "grail.human_mask_crop.v1",
        "config": asdict(cfg),
        "frame_count": count,
        "image_shape_hw": [height, width],
        "repaired_frames": sum(f["status"] == "repaired" for f in frames),
        "fallback_frames": sum(f["status"] == "fallback" for f in frames),
        "unchanged_frames": sum(f["status"] == "unchanged" for f in frames),
        "scope": "pose_crop_only; selected identity and camera mask boxes unchanged",
        "frames": frames,
    }
    return repaired, report


def _guard_temporal_crop_size(repaired, originals, frames, masks, source_indices, cfg):
    """Protect a contracted crop only with larger same-track support on both sides.

    This optional offline policy handles masks that also lose occluded limbs.
    It changes only the pose crop, keeps the current center, and never shrinks
    the mask-repaired crop. It cannot establish the location of hidden joints.
    Failed mask associations, sequence edges and unsupported scale changes do
    not borrow extents from other frames. Votes count distinct source frames.
    """
    valid = np.array([frame["status"] != "fallback" for frame in frames])
    for index, frame in enumerate(frames):
        if not valid[index]:
            continue
        lo = max(0, index - cfg.temporal_size_radius)
        hi = min(len(frames), index + cfg.temporal_size_radius + 1)
        nearby = np.arange(lo, hi)
        distance = np.linalg.norm(originals[nearby, :2] - originals[index, :2], axis=1)
        compatible = valid[nearby] & (
            distance <= cfg.temporal_size_center_tolerance * originals[nearby, 2]
        )
        distinct = {}
        for j in nearby[compatible]:
            distinct.setdefault(int(source_indices[j]), int(j))
        neighbors = np.array(list(distinct.values()), dtype=int)
        if not len(neighbors):
            continue
        reference = float(np.quantile(originals[neighbors, 2], 0.9))
        minimum = cfg.temporal_size_floor_ratio * reference
        if originals[index, 2] >= minimum:
            continue
        larger = neighbors[originals[neighbors, 2] >= minimum]
        before = larger[source_indices[larger] < source_indices[index]]
        after = larger[source_indices[larger] > source_indices[index]]
        if min(len(before), len(after)) < cfg.temporal_size_min_support:
            continue
        # Keep the selected-track center rather than inheriting a sudden
        # center shift when newly visible feet first expand the SAM2 mask.
        # Enclose that accepted mask crop at this unchanged center.
        size = max(
            minimum,
            float(
                repaired[index, 2] + 2 * np.max(np.abs(repaired[index, :2] - originals[index, :2]))
            ),
        )
        if size / originals[index, 2] > cfg.max_crop_scale:
            continue
        frame["mask_repair_reason"] = frame["reason"]
        frame["temporal_size_evidence"] = {
            "reference_size_p90": reference,
            "minimum_size": minimum,
            "support_before": source_indices[before].tolist(),
            "support_after": source_indices[after].tolist(),
            "interpretation": "offline crop scale prior; not observed hidden anatomy",
        }
        repaired[index, :2] = originals[index, :2]
        repaired[index, 2] = size
        frame.update(
            status="repaired",
            reason="bracketed_temporal_size_guard",
            coverage_after=_coverage(masks[index], _crop_box(repaired[index])),
            repaired_crop_xys=repaired[index].tolist(),
        )
