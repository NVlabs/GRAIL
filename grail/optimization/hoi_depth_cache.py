"""Explicit, hash-bound depth input for reviewed reconstruction experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path

from grail.core.bindings import sha256


def validate_depth_override(spec, source_video=None):
    """Validate provenance before fitting; never fall back to a legacy cache."""
    required = {"path", "sha256", "metadata_sha256", "model", "frame_count"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Explicit depth override requires exact file/model/frame bindings")
    path = Path(spec["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Explicit depth cache must be an existing absolute path")
    metadata_path = Path(str(path) + ".json")
    if sha256(path) != spec["sha256"] or sha256(metadata_path) != spec["metadata_sha256"]:
        raise ValueError("Explicit depth cache or sidecar hash changed")
    metadata = json.loads(metadata_path.read_text())
    n = spec["frame_count"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Explicit depth frame count must be positive")
    if metadata.get("schema_version") != 1 or metadata.get("model") != spec["model"]:
        raise ValueError("Explicit depth model/settings mismatch")
    if metadata.get("frame_count") != n or spec["model"].get("depth_unit") != "meter":
        raise ValueError("Explicit depth timeline/unit mismatch")
    if source_video is not None and metadata.get("source_video_path") != str(source_video):
        raise ValueError("Explicit depth source-video path mismatch")
    calibration = metadata.get("gt_calibration", {})
    frames = calibration.get("frames", [])
    if (
        calibration.get("status") != "aligned"
        or calibration.get("method") != "per_frame_background_linear_depth_scale_shift"
        or len(frames) != n
    ):
        raise ValueError("Explicit depth must have complete background-GT alignment")
    for index, row in enumerate(frames):
        scale, shift = row.get("scale", float("nan")), row.get("shift", float("nan"))
        if (
            row.get("frame_idx") != index
            or row.get("status") != "aligned"
            or row.get("foreground_excluded") is not True
            or not math.isfinite(scale)
            or scale <= 0
            or not math.isfinite(shift)
        ):
            raise ValueError("Explicit depth contains an unaligned or invalid frame")
    return metadata
