"""Utilities for versioned RP2 reset-start caches."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
VALIDATION_LEVELS = {"candidate": 0, "reference": 1, "isaaclab": 2}


@dataclass(frozen=True)
class ResetStartRecord:
    """Validated reset ranges for one motion in runtime-frame coordinates."""

    motion_key: str
    runtime_num_steps: int
    start_ranges: tuple[tuple[int, int], ...]
    validation_level: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResetStartCache:
    """Parsed reset cache aligned to a requested motion-key order."""

    path: Path
    target_fps: float
    episode_window_frames: int | None
    records: tuple[ResetStartRecord | None, ...]
    payload: dict[str, Any]


def runtime_num_steps(source_num_frames: int, source_fps: float, target_fps: float) -> int:
    """Match MotionLib's interpolation length for one source clip."""
    if source_num_frames < 1:
        raise ValueError(f"source_num_frames must be positive, got {source_num_frames}")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError(f"fps values must be positive, got {source_fps}, {target_fps}")
    if math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1.0e-9):
        return int(source_num_frames)
    duration = (int(source_num_frames) - 1) / float(source_fps)
    return max(int(math.ceil(duration * float(target_fps) - 1.0e-9)), 1)


def valid_start_upper_bound(runtime_steps: int, episode_window_frames: int | None) -> int:
    """Return MotionLib-compatible exclusive upper bound for episode starts."""
    if runtime_steps < 1:
        raise ValueError(f"runtime_steps must be positive, got {runtime_steps}")
    if episode_window_frames is None:
        return int(runtime_steps)
    if episode_window_frames < 1:
        raise ValueError(
            f"episode_window_frames must be positive or None, got {episode_window_frames}"
        )
    return max(int(runtime_steps) - 1 - int(episode_window_frames), 1)


def precontact_start_ranges(
    *,
    first_contact_frame: int,
    valid_upper: int,
    minimum_lead_frames: int,
    episode_window_frames: int | None,
) -> tuple[tuple[int, int], ...]:
    """Build starts that put contact inside the episode after a clean lead-in."""
    if first_contact_frame < 0:
        raise ValueError(f"first_contact_frame must be non-negative, got {first_contact_frame}")
    if valid_upper < 1:
        raise ValueError(f"valid_upper must be positive, got {valid_upper}")
    if minimum_lead_frames < 0:
        raise ValueError(f"minimum_lead_frames must be non-negative, got {minimum_lead_frames}")
    lower = 0
    if episode_window_frames is not None:
        lower = max(int(first_contact_frame) - int(episode_window_frames), 0)
    upper = min(
        int(first_contact_frame) - int(minimum_lead_frames) + 1,
        int(valid_upper),
    )
    if upper <= lower:
        return ()
    return ((lower, upper),)


def _validation_rank(level: str) -> int:
    try:
        return VALIDATION_LEVELS[level]
    except KeyError as exc:
        raise ValueError(
            f"unknown reset-cache validation level {level!r}; "
            f"expected one of {sorted(VALIDATION_LEVELS)}"
        ) from exc


def _normalize_ranges(raw_ranges: Iterable[Iterable[Any]], motion_key: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for raw in raw_ranges:
        values = list(raw)
        if len(values) != 2:
            raise ValueError(f"{motion_key}: reset range must contain [start, end), got {values}")
        start, end = int(values[0]), int(values[1])
        if start < 0 or end <= start:
            raise ValueError(f"{motion_key}: invalid reset range [{start}, {end})")
        ranges.append((start, end))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"{motion_key}: overlapping reset ranges {previous} and {current}"
            )
    return tuple(ranges)


def load_reset_start_cache(
    path: str | Path,
    *,
    motion_keys: Iterable[str],
    target_fps: float,
    required_validation_level: str = "reference",
    require_all: bool = False,
) -> ResetStartCache:
    """Load and align a reset cache, rejecting stale or malformed records."""
    cache_path = Path(path).expanduser().resolve()
    with cache_path.open() as cache_file:
        payload = json.load(cache_file)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"{cache_path}: schema_version={payload.get('schema_version')} "
            f"does not match {SCHEMA_VERSION}"
        )
    cache_fps = float(payload.get("target_fps", -1.0))
    if not math.isclose(cache_fps, float(target_fps), rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(
            f"{cache_path}: target_fps={cache_fps} does not match runtime {target_fps}"
        )
    episode_window = payload.get("episode_window_frames")
    episode_window = int(episode_window) if episode_window is not None else None
    motions = payload.get("motions")
    if not isinstance(motions, dict):
        raise TypeError(f"{cache_path}: motions must be a dictionary")

    minimum_rank = _validation_rank(required_validation_level)
    aligned: list[ResetStartRecord | None] = []
    missing: list[str] = []
    for motion_key in motion_keys:
        raw_record = motions.get(motion_key)
        if not isinstance(raw_record, dict):
            aligned.append(None)
            missing.append(motion_key)
            continue
        level = str(raw_record.get("validation_level", "candidate"))
        if _validation_rank(level) < minimum_rank:
            aligned.append(None)
            missing.append(motion_key)
            continue
        ranges = _normalize_ranges(raw_record.get("start_ranges", ()), motion_key)
        if not ranges:
            aligned.append(None)
            missing.append(motion_key)
            continue
        runtime_steps = int(raw_record.get("runtime_num_steps", 0))
        if runtime_steps < 1:
            raise ValueError(f"{motion_key}: missing positive runtime_num_steps")
        if ranges[-1][1] > runtime_steps:
            raise ValueError(
                f"{motion_key}: reset range ends at {ranges[-1][1]} past "
                f"runtime_num_steps={runtime_steps}"
            )
        aligned.append(
            ResetStartRecord(
                motion_key=motion_key,
                runtime_num_steps=runtime_steps,
                start_ranges=ranges,
                validation_level=level,
                metadata=raw_record,
            )
        )
    if require_all and missing:
        raise ValueError(
            f"{cache_path}: {len(missing)}/{len(aligned)} loaded motions lack "
            f"{required_validation_level!r} reset ranges; first={missing[:8]}"
        )
    return ResetStartCache(
        path=cache_path,
        target_fps=cache_fps,
        episode_window_frames=episode_window,
        records=tuple(aligned),
        payload=payload,
    )
