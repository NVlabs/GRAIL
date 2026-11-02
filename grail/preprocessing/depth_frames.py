"""Lossless depth input frames bound to the selected source video."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2

from grail.core.depth_config import depth_file_binding


def _frames(directory):
    return sorted(directory.glob("*.png"))


def _bindings(directory):
    return [{**depth_file_binding(path), "path": path.name} for path in _frames(directory)]


def ensure_depth_frames(video, directory, expected_frames):
    """Reuse only complete, content-bound frames; preserve every superseded cache.

    Decode into a new sibling before replacing the selected directory. An
    interrupted extraction therefore cannot certify a partial set of frames or
    destroy the previous cache.
    """
    video, directory = Path(video).resolve(strict=True), Path(directory)
    source = depth_file_binding(video)
    if expected_frames < 1:
        raise ValueError("Depth frame extraction requires a nonempty source video")
    manifest = directory / "source.json"
    if directory.is_dir() and manifest.is_file():
        try:
            record = json.loads(manifest.read_text())
            if (
                isinstance(record, dict)
                and record.get("schema") == "grail.depth.source_frames.v1"
                and record.get("source") == source
                and record.get("frame_count") == expected_frames
                and len(_frames(directory)) == expected_frames
                and record.get("frames") == _bindings(directory)
            ):
                return [str(path) for path in _frames(directory)]
        except (OSError, ValueError, TypeError):
            pass
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(f"{directory.name}.extracting-{uuid.uuid4().hex}")
    staging.mkdir()
    capture = cv2.VideoCapture(str(video))
    count = 0
    shape = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode depth source video: {video}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if shape is None:
                shape = frame.shape
            if frame.shape != shape:
                raise ValueError("Depth source dimensions changed during decoding")
            if not cv2.imwrite(str(staging / f"{count:06d}.png"), frame):
                raise OSError("Cannot save lossless depth input frame")
            count += 1
    finally:
        capture.release()
    if count != expected_frames or depth_file_binding(video) != source:
        raise ValueError("Depth source changed or did not decode its complete frame count")
    record = {
        "schema": "grail.depth.source_frames.v1",
        "source": source,
        "frame_count": count,
        "frames": _bindings(staging),
    }
    (staging / "source.json").write_text(json.dumps(record, indent=2) + "\n")
    if directory.exists():
        previous = directory.with_name(f"{directory.name}.previous-{uuid.uuid4().hex}")
        directory.rename(previous)
        print(f"  Preserved previous depth frames: {previous}")
    staging.rename(directory)
    return [str(path) for path in _frames(directory)]
