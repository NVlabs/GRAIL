"""Small, dependency-light contracts shared by the native and Blender workers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

MODEL_TYPES = ("soma03",)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a regular input file: {path}")
    return {"path": str(path), "sha256": digest(path)}


def verify_binding(record):
    if digest(record["path"]) != record["sha256"]:
        raise ValueError(f"Bound input changed: {record['path']}")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    """Atomically replace a receipt inside the already unique round directory."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=json_value)
        stream.write("\n")
    os.replace(temp, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def save_npz(path, arrays):
    for name, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"Object arrays are not portable native data: {name}")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _resolve(value, parent, *, executable=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    # Resolving a venv's bin/python symlink changes which environment starts.
    # Keep its absolute invocation path; data files use canonical target paths.
    path = path.absolute() if executable else path.resolve(strict=True)
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        raise ValueError(f"Expected {'executable' if executable else 'file'}: {path}")
    return str(path)


def read_case(path):
    path = Path(path).expanduser().resolve(strict=True)
    case = read_json(path)
    allowed = {
        "schema_version",
        "name",
        "input_native",
        "model",
        "runtime",
        "scene",
        "plan",
        "observation_frames",
        "review_frames",
        "source_video",
        "source_images",
        "contact_review",
        "evidence",
        "render",
        "newton",
        "repair",
        "iteration",
    }
    if not isinstance(case, dict) or set(case) - allowed:
        raise ValueError("Unknown refinement case fields")
    if type(case.get("schema_version")) is not int or case["schema_version"] != 1:
        raise ValueError("Expected refinement case schema_version=1")
    if not isinstance(case.get("name"), str) or not case["name"].strip():
        raise ValueError("A nonempty case name is required")
    model = case.get("model")
    if not isinstance(model, dict) or model.get("model_type") not in MODEL_TYPES:
        raise ValueError("Declare model_type as soma03")
    if model.setdefault("backend", "native") != "native":
        raise ValueError("SOMA03 requires the native evaluator backend")
    for name in ("input_native", "scene", "plan"):
        case[name] = _resolve(case[name], path.parent)
    for name in ("source_video", "contact_review"):
        if case.get(name):
            case[name] = _resolve(case[name], path.parent)
    if "repair" in case:
        if not isinstance(case["repair"], dict) or not case["repair"].get("contact_map"):
            raise ValueError("repair requires an explicit video-reviewed contact_map")
        case["repair"]["contact_map"] = _resolve(case["repair"]["contact_map"], path.parent)
    if "iteration" in case and not isinstance(case["iteration"], dict):
        raise ValueError("iteration must be an object")
    case["source_images"] = [
        _resolve(value, path.parent) for value in case.get("source_images", [])
    ]
    # Explicit model asset paths resolve against the case, not a worker's cwd.
    for key, value in list(model.items()):
        if key.endswith(("_path", "_dir")) or key == "gem_x_soma_assets":
            asset = Path(value).expanduser()
            model[key] = str(
                (path.parent / asset if not asset.is_absolute() else asset).resolve(strict=True)
            )
    runtime = case.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) - {"native_python", "blender", "timeout_s"}:
        raise ValueError("Declare native_python and blender in runtime")
    for name in ("native_python", "blender"):
        runtime[name] = _resolve(runtime[name], path.parent, executable=True)
    timeout = runtime.setdefault("timeout_s", 180)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("runtime.timeout_s must be finite and positive")
    for key in ("observation_frames", "review_frames"):
        frames = case.get(key, case.get("observation_frames"))
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{key} must contain explicit source frames")
        if any(
            isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or f < 0
            for f in frames
        ):
            raise ValueError(f"Invalid {key}")
        if frames != sorted(set(frames)):
            raise ValueError(f"{key} must be unique and increasing")
        case[key] = frames
    if not set(case["observation_frames"]) <= set(case["review_frames"]):
        raise ValueError("review_frames must include observation_frames")
    evidence = case.setdefault("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object")
    for name in ("fixed_depth", "source_keypoints"):
        if isinstance(evidence.get(name), dict) and "path" in evidence[name]:
            evidence[name]["path"] = _resolve(evidence[name]["path"], path.parent)
    if case.get("newton"):
        newton = case["newton"]
        newton["python"] = _resolve(newton["python"], path.parent, executable=True)
        value = Path(newton["repo"]).expanduser()
        newton["repo"] = str(
            (path.parent / value if not value.is_absolute() else value).resolve(strict=True)
        )
    return case


def validate_scene(scene):
    """Original object geometry and calibrated OpenCV camera-to-world pose."""
    required = {
        "object_vertices",
        "object_faces",
        "camera_matrix_world",
        "camera_intrinsics",
        "image_width",
        "image_height",
        "camera_convention",
    }
    if not required <= scene.keys():
        raise ValueError(f"Missing scene fields: {sorted(required - scene.keys())}")
    if str(np.asarray(scene["camera_convention"]).item()) != "opencv_camera_to_world":
        raise ValueError("Declare camera_convention=opencv_camera_to_world")
    vertices, faces = np.asarray(scene["object_vertices"]), np.asarray(scene["object_faces"])
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 3
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Object vertices must be finite [V,3]")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Object faces must index original vertices")
    matrix, intrinsic = np.asarray(scene["camera_matrix_world"]), np.asarray(
        scene["camera_intrinsics"]
    )
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix[3], [0, 0, 0, 1])
    ):
        raise ValueError("Camera-to-world matrix must be finite affine 4x4")
    rot = matrix[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rot), 1, atol=1e-6
    ):
        raise ValueError("Camera rotation must be a proper rotation without scale")
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0
        or not np.array_equal(intrinsic[2], [0, 0, 1])
    ):
        raise ValueError("Invalid calibrated camera intrinsics")
    if intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0:
        raise ValueError("Skewed cameras are unsupported by the Blender preview")
    for key in ("image_width", "image_height"):
        value = np.asarray(scene[key])
        if value.shape != () or value.dtype.kind not in "iu" or int(value) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return scene
