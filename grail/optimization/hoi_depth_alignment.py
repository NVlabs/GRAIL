"""Opt-in refreshed original-triangle regional camera-depth alignment.

Foreground targets are estimates, not ground truth. The caller must bind an
explicit background-calibrated depth cache. This module never estimates scale,
changes geometry, trims high residuals, or supplies contact attraction.
"""

from __future__ import annotations

import math

import torch

REGIONS = ("body", "left_hand", "right_hand", "left_forearm", "right_forearm", "object")


def exclude_body_observation_confidence(confidence, exclusions):
    """Return a masked copy; never edit raw detections or invent observations."""
    if confidence.ndim != 2:
        raise ValueError("Expected [frame, body-keypoint] confidence")
    result = confidence.clone()
    for entry in exclusions:
        if set(entry) != {"coco_id", "start", "end"}:
            raise ValueError("Observation exclusions need coco_id/start/end")
        joint, start, end = (entry[k] for k in ("coco_id", "start", "end"))
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (joint, start, end)):
            raise ValueError("Observation exclusion coordinates must be integers")
        if not (0 <= joint < confidence.shape[1] and 0 <= start <= end < confidence.shape[0]):
            raise ValueError("Observation exclusion is outside the source timeline")
        result[start : end + 1, joint] = 0
    return result


def pseudo_huber_squared(residual, delta_m):
    """Smooth robust squared-metre error, locally r² and never hard-trimmed."""
    if not math.isfinite(delta_m) or delta_m <= 0:
        raise ValueError("delta_m must be finite and positive")
    # Rational form avoids cancellation near zero.
    return 2 * residual.square() / (torch.sqrt(1 + (residual / delta_m).square()) + 1)


def regional_depth_reduction(
    depth, target, labels, source_human, source_object, *, delta_m=0.03, min_pixels=2
):
    """Return per-region/per-frame losses and diagnostic tensors.

    Only first-hit original surfaces with matching source segmentation are used.
    Invalid depth and source-mask overlap are excluded. Empty regions have count
    zero and are *not* an alignment certificate; the caller must inspect coverage.
    """
    if depth.shape != target.shape or labels.shape != depth.shape:
        raise ValueError("Depth, target and face-region image shapes differ")
    valid = torch.isfinite(target) & (target > 0) & torch.isfinite(depth) & (depth > 0)
    h = source_human.bool() & ~source_object.bool()
    o = source_object.bool() & ~source_human.bool()
    residual = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth)) - torch.where(
        torch.isfinite(target), target, torch.zeros_like(target)
    )
    per_pixel = pseudo_huber_squared(residual, delta_m)
    values, counts, rendered, signed, absolute = [], [], [], [], []
    dims = tuple(range(1, depth.ndim))
    for region in range(6):
        render = labels == region
        mask = render & valid & (o if region == 5 else h)
        count = mask.sum(dims)
        usable = count >= min_pixels
        denom = count.clamp_min(1)
        values.append(torch.where(usable, (torch.where(mask, per_pixel, 0)).sum(dims) / denom, 0))
        counts.append(count)
        rendered.append(render.sum(dims))
        signed.append(torch.where(mask, residual, 0).sum(dims) / denom)
        absolute.append(torch.where(mask, residual.abs(), 0).sum(dims) / denom)
    return {
        "loss_per_frame": torch.stack(values),
        "count_per_frame": torch.stack(counts),
        "rendered_count_per_frame": torch.stack(rendered),
        "signed_m_per_frame": torch.stack(signed),
        "mae_m_per_frame": torch.stack(absolute),
    }


def source_silhouette_dice(alpha, source_human, source_object, known):
    """Mean visible human/object Dice per frame; ambiguous pixels are unknown."""
    source = torch.stack([source_human, source_object], -1).to(alpha)
    known = known[..., None].to(alpha)
    alpha, source = alpha * known, source * known
    intersection = (alpha * source).sum((1, 2))
    denominator = alpha.sum((1, 2)) + source.sum((1, 2))
    return (1 - (2 * intersection + 1e-6) / (denominator + 1e-6)).mean(-1)


def screen_depth_to_clip(screen, z, full_height, full_width, near=0.01, far=100.0):
    """Convert existing calibrated screen pixels and view-Z into GL clip space."""
    return torch.stack(
        (
            (2 * screen[..., 0] / full_width - 1) * z,
            (1 - 2 * screen[..., 1] / full_height) * z,
            (far + near) / (far - near) * z - 2 * far * near / (far - near),
            z,
        ),
        -1,
    )


def coverage_retention(reference_counts, candidate_counts, *, min_fraction=0.8, pixel_tolerance=2):
    """Per-region/frame coverage check; missing difficult hands cannot pass.

    Returns a Boolean matrix so callers can either reject updates or reject an
    endpoint after independent evaluation. This does not fabricate gradients for
    pixel-count changes. Reference counts must be captured before fitting.
    """
    if reference_counts.shape != candidate_counts.shape:
        raise ValueError("Reference and candidate coverage shapes differ")
    if not 0 < min_fraction <= 1 or pixel_tolerance < 0:
        raise ValueError("Invalid coverage retention threshold")
    return candidate_counts >= torch.clamp(reference_counts * min_fraction - pixel_tolerance, min=0)


def capture_constrained_motion_reference(optimizer, data):
    """Capture one fixed stationary/source-seeded initializer before stage one.

    This explicit experiment helper is called by the stage-one runner *after*
    its source-supported root seed. It leaves the optimizer's legacy default
    reference initialization unchanged. Never call it during iterative fitting.
    """
    import hashlib

    if getattr(optimizer, "_constrained_motion_reference_captured", False):
        raise RuntimeError("Constrained motion reference may be captured only once")
    if (
        getattr(optimizer, "reviewed_iterations", None) != 0
        or getattr(optimizer, "reviewed_attempted_iterations", None) != 0
    ):
        raise RuntimeError(
            "Constrained motion reference requires zero attempted and accepted updates"
        )
    computer = optimizer.loss_computer
    previous = computer._motion_reference
    if previous is None:
        raise RuntimeError("Expected an existing raw-initializer motion reference")
    raw_object_poses = data.obj.poses.detach().clone()
    parameters = {name: value.detach().clone() for name, value in vars(optimizer.params).items()}
    with torch.no_grad():
        predicted = optimizer.forward(data, optimizer.params, apply_object_stationary=True)
        computer.set_motion_reference(data, predicted)
        expected = computer._motion_points(predicted)
    current = computer._motion_reference
    if not all(torch.equal(a, b) for a, b in zip(current, expected)):
        raise RuntimeError("Captured motion reference differs from constrained initializer")
    if not torch.equal(raw_object_poses, data.obj.poses):
        raise RuntimeError("Reference capture mutated raw object poses")
    if set(parameters) != set(vars(optimizer.params)) or any(
        not torch.equal(value, getattr(optimizer.params, name))
        for name, value in parameters.items()
    ):
        raise RuntimeError("Reference capture mutated optimizer parameters")

    def binding(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        }

    record = {
        "policy": "constrained_initializer",
        "capture_timing": "after_source_seed_before_first_stage_update",
        "apply_object_stationary": True,
        "accepted_updates_at_capture": 0,
        "attempted_updates_at_capture": 0,
        "reference_matches_constrained_initializer_exact": True,
        "raw_object_poses_unchanged": True,
        "optimizer_parameters_unchanged": True,
        "original_raw_reference": [binding(v) for v in previous],
        "constrained_reference": [binding(v) for v in current],
        "reference_rms_change_m": [
            float((a - b).square().mean().sqrt()) for a, b in zip(current, previous)
        ],
    }
    optimizer._constrained_motion_reference_captured = record
    return record
