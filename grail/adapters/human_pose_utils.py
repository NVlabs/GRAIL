"""Shared source-video timing, interpolation and content bindings for pose adapters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

_FILE_HASH_CACHE = {}


def _sha256(path):
    path = Path(path).resolve()

    def stamp(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    current = stamp(path.stat())
    cached = _FILE_HASH_CACHE.get(path)
    if cached is not None and cached[0] == current:
        return cached[1]
    # Bind the digest to the opened file, including ctime. A network-filesystem
    # metadata refresh or a one-time replacement may invalidate the first read;
    # retry it once, never cache unstable bytes or ignore repeated changes.
    observed = []
    for _ in range(2):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            before = stamp(os.fstat(handle.fileno()))
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = stamp(os.fstat(handle.fileno()))
        final = stamp(path.stat())
        if before == after == final:
            result = digest.hexdigest()
            _FILE_HASH_CACHE[path] = (final, result)
            return result
        observed.append((before, after, final))
    raise RuntimeError(
        f"File changed while computing its cache fingerprint after two reads: {path}; "
        f"observed (device, inode, size, mtime_ns, ctime_ns)={observed}"
    )


def _bundle_component_hashes(paths):
    """Bind every selected model, source tree, detector and native body asset.

    Hashes are memoized only in this process using inode/size/mtime/ctime. A
    previously published content manifest without those stamps cannot establish
    that a local file still has its published bytes, so the first call reads it.
    Generated Python caches are excluded from source-tree identities.
    """
    components = {}
    for name, value in sorted(paths.items()):
        path = Path(value)
        if path.is_file():
            components[name] = {"sha256": _sha256(path), "files": 1}
            continue
        files = {
            str(child.relative_to(path)): _sha256(child)
            for child in sorted(path.rglob("*"))
            if child.is_file()
            and not any(
                part in {"__pycache__", ".git", ".pytest_cache"}
                for part in child.relative_to(path).parts
            )
            and child.suffix not in {".pyc", ".pyo"}
        }
        if not files:
            raise ValueError(f"Bundle component has no source/assets: {name}: {path}")
        components[name] = {
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "files": len(files),
        }
    return components


def read_video_timing(path):
    """Read decoded timestamps and reject VFR, which GRAIL cannot represent."""
    import numpy as np

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp,best_effort_timestamp_time:stream=avg_frame_rate,time_base,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    stream = data["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    timestamps = np.asarray(
        [float(frame["best_effort_timestamp_time"]) for frame in data["frames"]],
        dtype=np.float64,
    )
    if not math.isfinite(fps) or fps <= 0 or len(timestamps) < 2:
        raise ValueError(
            "Pose estimation requires a video with positive FPS and at least two frames"
        )
    if not np.isfinite(timestamps).all() or not np.allclose(
        np.diff(timestamps), 1 / fps, atol=max(1e-5, 0.01 / fps), rtol=0
    ):
        raise ValueError("Variable or invalid frame timestamps need explicit video normalization")
    return {
        "fps": fps,
        "frame_count": len(timestamps),
        "timestamps": timestamps.tolist(),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        **(
            {
                "frame_pts": [int(frame["best_effort_timestamp"]) for frame in data["frames"]],
                "time_base": stream["time_base"],
                "frame_rate": stream["avg_frame_rate"],
            }
            if "time_base" in stream
            and all("best_effort_timestamp" in frame for frame in data["frames"])
            else {}
        ),
    }


def _tensor(value, name, shape):
    import torch

    result = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tuple(result.shape) != tuple(shape) or not bool(torch.isfinite(result).all()):
        raise ValueError(f"Invalid {name}: expected finite {shape}, got {tuple(result.shape)}")
    return result


def _linear(values, native_times, target_times):
    import numpy as np
    import torch

    source = values.numpy()
    flattened = source.reshape(len(native_times), -1)
    result = np.stack(
        [np.interp(target_times, native_times, column) for column in flattened.T], axis=-1
    ).reshape((len(target_times),) + source.shape[1:])
    return torch.from_numpy(result.astype(np.float32))


def _rotations(values, native_times, target_times):
    """Shortest-arc quaternion interpolation, including the +/-pi boundary."""
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    source = values.numpy().reshape(len(native_times), -1, 3)
    quaternions = (
        Rotation.from_rotvec(source.reshape(-1, 3)).as_quat().reshape(len(native_times), -1, 4)
    )
    times = np.clip(target_times, native_times[0], native_times[-1])
    upper = np.searchsorted(native_times, times, side="right").clip(1, len(native_times) - 1)
    lower = upper - 1
    weight = ((times - native_times[lower]) / (native_times[upper] - native_times[lower]))[
        :, None, None
    ]
    first, second = quaternions[lower], quaternions[upper]
    dot = (first * second).sum(axis=-1, keepdims=True)
    second = np.where(dot < 0, -second, second)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    safe_sine = np.where(sine > 1e-8, sine, 1)
    spherical = (np.sin((1 - weight) * angle) * first + np.sin(weight * angle) * second) / safe_sine
    linear = (1 - weight) * first + weight * second
    interpolated = np.where(dot > 0.9995, linear, spherical)
    interpolated /= np.linalg.norm(interpolated, axis=-1, keepdims=True)
    result = (
        Rotation.from_quat(interpolated.reshape(-1, 4))
        .as_rotvec()
        .reshape((len(target_times),) + values.shape[1:])
    )
    return torch.from_numpy(result.astype(np.float32))
