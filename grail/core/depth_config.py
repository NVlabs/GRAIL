"""Pinned production depth model and cache identity.

MoGe-2 caches remain under ``depth/``.  MoGe-3 has its own namespace so a
preprocessing skip or optimizer load cannot accidentally reuse those values.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from grail.core.bindings import bind

MOGE_MODEL_ID = "Ruicheng/moge-3-vitg"
MOGE_MODEL_REVISION = "6ef26c5a4b4148dab5ccaacdb08b72dc66380475"
MOGE_RESOLUTION_LEVEL = 9
MOGE_REFINE_STEPS = 3
DEPTH_CACHE_SUBDIR = "depth_moge3_vitg"
DEPTH_CACHE_SCHEMA_VERSION = 1


def depth_model_spec() -> dict:
    """Identity and inference settings shared by inference and cache validation."""
    return {
        "model_id": MOGE_MODEL_ID,
        "model_revision": MOGE_MODEL_REVISION,
        "resolution_level": MOGE_RESOLUTION_LEVEL,
        "refine_steps": MOGE_REFINE_STEPS,
        "cuda_use_fp16": True,
        "depth_unit": "meter",
    }


def depth_file_binding(path):
    """Content binding used by cache validation, independent of tensor libraries."""
    return bind(Path(path).expanduser())


def depth_input_bindings(paths):
    """Bind every selected source/calibration file, including explicitly absent inputs."""
    return {key: None if path is None else depth_file_binding(path) for key, path in paths.items()}


def preserve_depth_cache(cache_file):
    """Archive an existing tensor and sidecar together before publishing a replacement."""
    cache = Path(cache_file)
    previous = [p for p in (cache, depth_cache_metadata_path(cache)) if p.exists()]
    if not previous:
        return None
    archive = cache.with_name(f"{cache.name}.previous-{uuid.uuid4().hex}")
    archive.mkdir()
    for path in previous:
        path.rename(archive / path.name)
    return archive


def depth_cache_metadata_path(cache_file: str | os.PathLike) -> Path:
    return Path(f"{cache_file}.json")


def read_depth_cache_metadata(
    cache_file: str | os.PathLike, *, metadata_binding: dict | None = None
) -> dict:
    """Require current provenance, optionally relocated by a verified inventory.

    An explicit binding is authoritative: never fall back to a neighbouring
    sidecar when the bound metadata is missing or changed. Read and validate the
    same bytes so relocation does not require rewriting calibration provenance.
    """
    if metadata_binding is None:
        path = depth_cache_metadata_path(cache_file)
    else:
        if (
            not isinstance(metadata_binding, dict)
            or set(metadata_binding) != {"path", "sha256", "bytes"}
            or not isinstance(metadata_binding["path"], (str, os.PathLike))
            or not isinstance(metadata_binding["sha256"], str)
            or len(metadata_binding["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in metadata_binding["sha256"])
            or type(metadata_binding["bytes"]) is not int
            or metadata_binding["bytes"] < 1
        ):
            raise ValueError("Depth metadata binding requires path, sha256 and positive bytes")
        path = Path(metadata_binding["path"])
    try:
        payload = path.read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Depth cache provenance missing or unreadable: {path}") from exc
    if metadata_binding is not None and (
        len(payload) != metadata_binding["bytes"]
        or hashlib.sha256(payload).hexdigest() != metadata_binding["sha256"]
    ):
        raise ValueError(f"Depth cache provenance differs from bound metadata: {path}")
    try:
        metadata = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Depth cache provenance missing or unreadable: {path}") from exc
    if not isinstance(metadata, dict) or (
        metadata.get("schema_version") != DEPTH_CACHE_SCHEMA_VERSION
        or metadata.get("model") != depth_model_spec()
        or not isinstance(metadata.get("frame_count"), int)
        or metadata["frame_count"] < 1
    ):
        raise ValueError(f"Depth cache model/settings mismatch: {path}; rerun preprocessing")
    # Older inventory-bound sidecars remain readable; new sidecars additionally
    # seal the tensor bytes. A copied cache can retain its original provenance.
    if "cache_binding" in metadata:
        expected = metadata["cache_binding"]
        try:
            actual = depth_file_binding(cache_file)
        except OSError as exc:
            raise ValueError(f"Depth tensor cache missing or unreadable: {cache_file}") from exc
        if not isinstance(expected, dict) or any(
            actual[key] != expected.get(key) for key in ("sha256", "bytes")
        ):
            raise ValueError(f"Depth tensor cache differs from provenance: {cache_file}")
    return metadata


def depth_cache_is_current(cache_file: str | os.PathLike, *, input_paths=None) -> bool:
    """Cache hits require both unchanged tensor bytes and the current input selection.

    Model-only historical sidecars remain loadable for explicit recovery, but
    cannot establish a preprocessing cache hit for a newly selected source.
    """
    if not input_paths or not Path(cache_file).is_file():
        return False
    try:
        metadata = read_depth_cache_metadata(cache_file)
        return "cache_binding" in metadata and metadata.get(
            "input_bindings"
        ) == depth_input_bindings(input_paths)
    except (ValueError, OSError):
        return False


def write_depth_cache_metadata(cache_file: str | os.PathLike, metadata: dict) -> None:
    """Publish provenance last, after the tensor cache has been saved."""
    path = depth_cache_metadata_path(cache_file)
    payload = {
        **metadata,
        "schema_version": DEPTH_CACHE_SCHEMA_VERSION,
        "model": depth_model_spec(),
        "cache_binding": depth_file_binding(cache_file),
    }
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
