"""Share exact depth targets and visibility masks between paired fits.

The NPZ stores only numeric arrays. A JSON sidecar binds every frame to the
caller's input/model/camera/config context and the exact archive bytes. This
does not build or recalibrate targets; callers must verify native initialization
parity before assigning the loaded cache to a loss computer.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

_SCHEMA = "grail.depth_reference.v1"
_GROUPS = ("gt_human_pcs", "gt_obj_pcs", "human_vis_masks", "obj_vis_masks")
_CONTEXT_COUNTS = ("frame_count", "human_vertex_count", "object_vertex_count")


def _json_bytes(value):
    """Canonical, type-sensitive JSON; reject coercion and non-finite values."""

    def check(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("Context/metadata must contain only JSON values")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _context_bytes(context):
    if type(context) is not dict:
        raise ValueError("Depth reference context must be a JSON object")
    for key in _CONTEXT_COUNTS:
        if type(context.get(key)) is not int or context[key] <= 0:
            raise ValueError(f"Context {key} must be a positive integer")
    return _json_bytes(context)


def _paths(path):
    path = Path(path).absolute()
    if path.suffix != ".npz":
        raise ValueError("Depth reference path must end in .npz")
    return path, path.with_suffix(".json")


def _hash(payload):
    return hashlib.sha256(payload).hexdigest()


def _name(group, frame):
    return f"{group}__{frame:06d}"


def _validate_array(array, group, frame, context):
    label = f"{group}[{frame}]"
    if group.endswith("_pcs"):
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{label} must have shape [N,3]")
        if array.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"{label} must have float16/float32/float64 dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} contains non-finite points")
    else:
        count = context[
            "human_vertex_count" if group == "human_vis_masks" else "object_vertex_count"
        ]
        if array.shape != (count,):
            raise ValueError(f"{label} must have shape [{count}]")
        if array.dtype != np.dtype("bool"):
            raise ValueError(f"{label} must have bool dtype")


def _arrays_from_cache(cache, context):
    if not isinstance(cache, Mapping) or set(cache) != set(_GROUPS):
        raise ValueError(f"Depth cache must contain exactly {_GROUPS}")
    arrays = {}
    expected_frames = set(range(context["frame_count"]))
    for group in _GROUPS:
        frames = cache[group]
        if (
            not isinstance(frames, Mapping)
            or any(type(key) is not int for key in frames)
            or set(frames) != expected_frames
        ):
            raise ValueError(f"{group} must contain every integer frame key 0..T-1")
        for frame in range(context["frame_count"]):
            value = frames[frame]
            if isinstance(value, torch.Tensor):
                try:
                    array = value.detach().cpu().numpy()
                except (TypeError, RuntimeError) as error:
                    raise ValueError(f"Unsupported tensor in {group}[{frame}]") from error
            elif isinstance(value, np.ndarray):
                array = value
            else:
                raise ValueError(f"{group}[{frame}] must be a tensor or numeric array")
            _validate_array(array, group, frame, context)
            # Own the bytes, so a later mutation of the caller's cache is harmless.
            arrays[_name(group, frame)] = np.array(array, copy=True, order="C")
    return arrays


def _array_record(array):
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _hash(array.tobytes(order="C")),
    }


def _receipt(path, sidecar, payload, metadata_bytes, context_bytes):
    return {
        "schema": "grail.depth_reference_receipt.v1",
        "npz": {"path": str(path), "sha256": _hash(payload), "bytes": len(payload)},
        "sidecar": {
            "path": str(sidecar),
            "sha256": _hash(metadata_bytes),
            "bytes": len(metadata_bytes),
        },
        "context_sha256": _hash(context_bytes),
    }


def save_depth_reference(path, cache, context):
    """Save a new immutable-by-contract NPZ/JSON pair and return its hashes.

    ``context`` requires positive integer ``frame_count``,
    ``human_vertex_count`` and ``object_vertex_count``. The caller supplies all
    other input, depth-setting, camera and model bindings as JSON values.
    Existing archives or sidecars are never overwritten.
    """
    context_bytes = _context_bytes(context)
    path, sidecar = _paths(path)
    if any(p.exists() or p.is_symlink() for p in (path, sidecar)):
        raise FileExistsError(f"Depth reference already exists: {path} or {sidecar}")
    arrays = _arrays_from_cache(cache, context)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    metadata = {
        "schema": _SCHEMA,
        "context": json.loads(context_bytes),
        "context_sha256": _hash(context_bytes),
        "npz_sha256": _hash(payload),
        "npz_bytes": len(payload),
        "arrays": {name: _array_record(array) for name, array in arrays.items()},
    }
    metadata_bytes = _json_bytes(metadata) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for destination, content in ((path, payload), (sidecar, metadata_bytes)):
            with destination.open("xb") as stream:
                created.append(destination)
                stream.write(content)
    except BaseException:
        for destination in created:
            destination.unlink(missing_ok=True)
        raise
    return _receipt(path, sidecar, payload, metadata_bytes, context_bytes)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def load_depth_reference(path, context, device):
    """Validate an exact bound reference and return a tensor cache + receipt.

    No visibility rerasterization, random sampling or pickle deserialization
    occurs. The returned tensors have the saved dtypes on ``device``.
    """
    context_bytes = _context_bytes(context)
    path, sidecar = _paths(path)
    metadata_bytes = sidecar.read_bytes()
    metadata = json.loads(metadata_bytes, object_pairs_hook=_unique_object)
    required = {"schema", "context", "context_sha256", "npz_sha256", "npz_bytes", "arrays"}
    if type(metadata) is not dict or set(metadata) != required or metadata["schema"] != _SCHEMA:
        raise ValueError("Invalid depth reference metadata schema")
    _json_bytes(metadata)
    saved_context = _context_bytes(metadata["context"])
    if saved_context != context_bytes or metadata["context_sha256"] != _hash(context_bytes):
        raise ValueError("Depth reference context mismatch")
    payload = path.read_bytes()
    if (
        type(metadata["npz_bytes"]) is not int
        or len(payload) != metadata["npz_bytes"]
        or _hash(payload) != metadata["npz_sha256"]
    ):
        raise ValueError("Depth reference NPZ hash/byte count mismatch")
    expected = {_name(group, frame) for group in _GROUPS for frame in range(context["frame_count"])}
    if type(metadata["arrays"]) is not dict or set(metadata["arrays"]) != expected:
        raise ValueError("Depth reference metadata must contain all frame keys")
    cache = {group: {} for group in _GROUPS}
    # Load the same immutable bytes that were hashed, not a reopened path.
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError("Depth reference archive must contain all frame keys")
        for group in _GROUPS:
            for frame in range(context["frame_count"]):
                name = _name(group, frame)
                array = archive[name]
                _validate_array(array, group, frame, context)
                if _json_bytes(metadata["arrays"][name]) != _json_bytes(_array_record(array)):
                    raise ValueError(f"Depth reference array metadata mismatch: {name}")
                cache[group][frame] = torch.from_numpy(np.array(array, copy=True, order="C")).to(
                    device
                )
    return cache, _receipt(path, sidecar, payload, metadata_bytes, context_bytes)
