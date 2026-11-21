"""Convert explicitly bound SOMA Retargeter CSV output to native robot motion.

The upstream CSV uses centimeters and degrees, including extrinsic XYZ root
Euler angles. It does not contain a frame rate, object motion, or missing robot
joints. This adapter recovers only those quantities that have real input data.
It neither builds a Blender scene nor transfers a 29-DOF robot to a hand model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit.native_io import load_npz
from .native import file_sha256
from .robot import _model_dependencies
from .static_scene import static_scene_payload


ROOT_COLUMNS = ("Frame", "root_translateX", "root_translateY", "root_translateZ",
                "root_rotateX", "root_rotateY", "root_rotateZ")


def _binding(path: str | Path, expected: str | None = None) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Required input file is missing: {path}")
    digest = file_sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Input SHA-256 differs from the bound upstream file: {path}")
    return {"path": str(path), "sha256": digest}


def _bvh_timing(path: Path) -> tuple[int, float, float]:
    """Match the official loader's two-decimal sample-rate convention."""
    if path.suffix.lower() != ".bvh":
        raise ValueError("SOMA native conversion requires the actual source BVH for timing")
    frames = frame_time = None
    in_motion = False
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == "MOTION":
                in_motion = True
            if not in_motion:
                continue
            match = re.fullmatch(r"\s*Frames:\s*(\d+)\s*", line)
            if match:
                frames = int(match.group(1))
            match = re.fullmatch(r"\s*Frame Time:\s*([^\s]+)\s*", line)
            if match:
                frame_time = float(match.group(1))
                break
    if not frames or frame_time is None or not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("Source BVH is missing valid Frames / Frame Time metadata")
    return frames, round(1.0 / frame_time, 2), frame_time


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if tuple(header[:7]) != ROOT_COLUMNS or len(header) <= 7:
            raise ValueError("Expected SOMA Retargeter root columns and named joint columns")
        if len(set(header)) != len(header) or any(not name.strip() for name in header):
            raise ValueError("CSV columns must be nonempty and unique")
        rows = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("CSV rows must match the complete header")
            try:
                rows.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError("CSV motion cells must be numeric") from exc
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("CSV must contain nonempty finite motion")
    if not np.array_equal(values[:, 0], np.arange(len(values))):
        raise ValueError("CSV Frame must preserve consecutive source frames starting at zero")
    return header[7:], values


def _object_arrays(path: Path, source_sha256: str, world_frame: str,
                   frames: int, fps: float) -> tuple[dict, dict]:
    values = load_npz(path)
    required = {"obj_R", "obj_t", "object_scale", "fps", "timestamps", "world_frame",
                "source_sha256", "object_mesh_path", "object_mesh_sha256"}
    if not required <= values.keys():
        raise ValueError("Object sidecar is missing explicit bindings: " + ", ".join(sorted(required - values.keys())))
    for key, expected in (("source_sha256", source_sha256), ("world_frame", world_frame)):
        if values[key].shape != () or str(values[key].item()) != expected:
            raise ValueError(f"Object sidecar {key} differs from the robot source")
    times = np.arange(frames, dtype=float) / fps
    if values["fps"].shape != () or float(values["fps"]) != fps:
        raise ValueError("Object sidecar fps differs from the robot timeline")
    if values["timestamps"].shape != (frames,) or not np.allclose(values["timestamps"], times, rtol=0, atol=1e-9):
        raise ValueError("Object sidecar timestamps differ from the robot timeline")
    rotation = np.asarray(values["obj_R"], dtype=float)
    translation = np.asarray(values["obj_t"], dtype=float)
    scale = np.asarray(values["object_scale"], dtype=float)
    if rotation.shape != (frames, 3, 3) or translation.shape != (frames, 3):
        raise ValueError("Object transforms must cover the exact robot frame count")
    if scale.shape not in ((), (1,), (3,)) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Object scale must be explicitly positive and scalar or XYZ")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Object transforms must be finite")
    if not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), rtol=0, atol=2e-5) or not np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=2e-5):
        raise ValueError("Object rotations must be proper rotation matrices")
    for key in ("object_mesh_path", "object_mesh_sha256"):
        if values[key].shape != () or values[key].dtype.kind not in "US":
            raise ValueError(f"Object sidecar {key} must be a scalar string")
    mesh_path = Path(str(values["object_mesh_path"].item())).expanduser()
    mesh_path = mesh_path if mesh_path.is_absolute() else path.parent / mesh_path
    mesh = _binding(mesh_path, str(values["object_mesh_sha256"].item()))
    arrays = {key: values[key] for key in ("obj_R", "obj_t", "object_scale")}
    arrays.update(object_mesh_path=np.asarray(mesh["path"]), object_mesh_sha256=np.asarray(mesh["sha256"]))
    _, static_arrays = static_scene_payload(values, world_frame=world_frame)
    arrays.update(static_arrays)
    return arrays, mesh


def convert_soma_csv(csv_path: str | Path, output_path: str | Path, *,
                     model_xml: str | Path, fps: float, source_path: str | Path,
                     expected_csv_sha256: str, expected_model_sha256: str,
                     expected_source_sha256: str, expected_model_assets: dict[str, str], world_frame: str,
                     object_sidecar: str | Path | None = None,
                     optimized_object: dict | None = None) -> dict:
    """Export exact-model native NPZ with MuJoCo ``wxyz`` root quaternion.

    ``model_xml`` must be the actual upstream robot model, not a same-name
    replacement. Every hinge must appear exactly once in the CSV. No zero-filled
    finger coordinates or stationary object transforms are created. ``fps`` is
    checked against the BVH using SOMA Retargeter's rounding convention. Object
    transforms, when supplied, must already use the declared robot world frame;
    asserting that alignment is the handoff producer's responsibility.
    ``expected_model_assets`` maps actual file paths to upstream SHA-256 values
    and must cover the complete MJCF include/mesh/texture dependency closure.
    """
    import mujoco

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite native motion: {output}")
    if output.suffix.lower() != ".npz":
        raise ValueError("Native output must have an .npz suffix")
    if isinstance(fps, (bool, np.bool_)) or not isinstance(fps, (int, float, np.integer, np.floating)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Explicit fps must be a positive finite number")
    if not isinstance(world_frame, str) or not world_frame.strip():
        raise ValueError("Explicit world_frame is required; axes cannot be inferred from the CSV")
    for name, digest in (("expected_csv_sha256", expected_csv_sha256), ("expected_model_sha256", expected_model_sha256),
                         ("expected_source_sha256", expected_source_sha256)):
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError(f"{name} must bind a lowercase SHA-256 digest")
    fps = float(fps)
    csv_binding = _binding(csv_path, expected_csv_sha256)
    source_binding = _binding(source_path, expected_source_sha256)
    model_binding = _binding(model_xml, expected_model_sha256)
    if not isinstance(expected_model_assets, dict) or not expected_model_assets:
        raise ValueError("expected_model_assets must bind the complete original model asset set")
    asset_bindings = {}
    for path, digest in expected_model_assets.items():
        if not isinstance(path, str) or not path or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("expected_model_assets must map file paths to lowercase SHA-256 digests")
        binding = _binding(path, digest)
        if binding["path"] in asset_bindings:
            raise ValueError("expected_model_assets must not duplicate a resolved file path")
        asset_bindings[binding["path"]] = binding
    dependencies = [_binding(path) for path in _model_dependencies(model_binding["path"])]
    for binding in dependencies:
        if binding["path"] not in asset_bindings or binding["sha256"] != asset_bindings[binding["path"]]["sha256"]:
            raise ValueError(f"Actual model dependency is not covered by the original model asset binding: {binding['path']}")
    source_frames, source_fps, frame_time = _bvh_timing(Path(source_binding["path"]))
    if not np.isclose(fps, source_fps, rtol=0, atol=1e-8):
        raise ValueError(f"fps must match the source BVH/official runtime: {source_fps}")
    names, rows = _read_csv(Path(csv_binding["path"]))
    if len(rows) != source_frames:
        raise ValueError("CSV frame count differs from the source BVH; resampling is not implicit")
    model = mujoco.MjModel.from_xml_path(model_binding["path"])
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0 or len(hinges) != model.njnt - 1 or model.nq != 7 + len(hinges):
        raise ValueError("CSV adapter requires exactly one free root followed by scalar hinges")
    native_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(index)) for index in hinges]
    if any(name is None for name in native_names) or len(set(native_names)) != len(native_names):
        raise ValueError("Native model hinges require unique names")
    if set(names) != set(native_names):
        missing, extra = sorted(set(native_names) - set(names)), sorted(set(names) - set(native_names))
        raise ValueError(f"CSV/model joint mismatch; missing={missing}, extra={extra}. Missing joints, including fingers, cannot be invented.")
    qpos = np.empty((len(rows), model.nq), dtype=np.float64)
    qpos[:, :3] = rows[:, 1:4] * .01
    xyzw = Rotation.from_euler("xyz", rows[:, 4:7], degrees=True).as_quat()
    qpos[:, 3:7] = xyzw[:, [3, 0, 1, 2]]
    # Choose an equivalent continuous quaternion representation for native
    # interpolation. No orientation is changed by this sign adjustment.
    for frame in range(1, len(qpos)):
        if np.dot(qpos[frame - 1, 3:7], qpos[frame, 3:7]) < 0:
            qpos[frame, 3:7] *= -1
    column = {name: index + 7 for index, name in enumerate(names)}
    addresses = model.jnt_qposadr[hinges].copy()
    for name, address in zip(native_names, addresses):
        qpos[:, address] = np.deg2rad(rows[:, column[name]])
    # Keep CSV roundoff tolerance separate from motion acceptance; no clipping.
    for index, name, address in zip(hinges, native_names, addresses):
        if model.jnt_limited[index]:
            low, high = model.jnt_range[index]
            if np.any(qpos[:, address] < low - 1e-6) or np.any(qpos[:, address] > high + 1e-6):
                raise ValueError(f"CSV exceeds original native joint limits for {name!r}; values were not clipped")
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, compiled)
    model_hash = hashlib.sha256(compiled.tobytes()).hexdigest()
    provenance = {"schema_version": 1, "backend": "soma", "status": "UNREVIEWED",
                  "source": source_binding, "csv": csv_binding, "model_xml": model_binding,
                  "model_dependencies": dependencies, "upstream_model_assets": list(asset_bindings.values()),
                  "csv_units": {"translation": "centimeters", "rotation": "degrees"},
                  "root_euler_convention": "extrinsic_xyz", "native_quaternion_order": "wxyz",
                  "world_frame": world_frame, "source_frame_time_seconds": frame_time,
                  "fps": fps, "fps_convention": "round(1 / BVH Frame Time, 2)",
                  "finger_motion_synthesized": False, "object_motion_synthesized": False}
    arrays = dict(qpos=qpos, fps=np.asarray(fps), timestamps=np.arange(len(rows)) / fps,
                  dof_names=np.asarray(native_names), qposadr=addresses,
                  source_path=np.asarray(source_binding["path"]), source_sha256=np.asarray(source_binding["sha256"]),
                  model_xml_path=np.asarray(model_binding["path"]), model_xml_sha256=np.asarray(model_binding["sha256"]),
                  compiled_model_sha256=np.asarray(model_hash), world_frame=np.asarray(world_frame),
                  native_status=np.asarray("UNREVIEWED"), selected=np.asarray(False),
                  export_kind=np.asarray("soma_initialization_requires_native_contact_and_visual_review"))
    bindings = [csv_binding, source_binding, *asset_bindings.values()]
    missing_scene_inputs = ["bound Blender scene and native visual-body mapping", "source and diagnostic cameras"]
    if object_sidecar is not None:
        sidecar_binding = _binding(object_sidecar)
        object_arrays, mesh = _object_arrays(Path(sidecar_binding["path"]), source_binding["sha256"], world_frame, len(rows), fps)
        arrays.update(object_arrays)
        provenance.update(object_sidecar=sidecar_binding, object_mesh=mesh)
        bindings.extend((sidecar_binding, mesh))
    else:
        missing_scene_inputs.append("object motion, original mesh and original scale for HOI editing")
    if optimized_object is not None:
        from .soma_object_io import validate_object_result
        fields = {"path", "sha256", "reference_path", "reference_sha256", "config", "original_sidecar_sha256"}
        if object_sidecar is None or set(optimized_object) != fields or world_frame != "soma_retargeter_z_up":
            raise ValueError("Optimized object requires its original aligned sidecar and exact output/reference bindings")
        if sidecar_binding["sha256"] != optimized_object["original_sidecar_sha256"]:
            raise ValueError("Optimized object requires the bound original sidecar, including its mesh and static scene")
        object_binding = _binding(optimized_object["path"], optimized_object["sha256"])
        reference_binding = _binding(optimized_object["reference_path"], optimized_object["reference_sha256"])
        changed, object_audit = validate_object_result(object_binding["path"],
            reference_path=reference_binding["path"], reference_sha256=reference_binding["sha256"],
            config=optimized_object["config"], frame_count=len(rows), fps=fps)
        if (not np.array_equal(arrays["obj_R"], changed["reference_obj_R"])
                or not np.array_equal(arrays["obj_t"], changed["reference_obj_t"])
                or not np.array_equal(np.broadcast_to(arrays["object_scale"], (3,)), changed["object_scale"])):
            raise ValueError("Original object sidecar does not match optimized trajectory reference")
        arrays.update(obj_R=changed["obj_R"], obj_t=changed["obj_t"],
            object_reference_R=changed["reference_obj_R"], object_reference_t=changed["reference_obj_t"],
            object_optimization_json=np.asarray(json.dumps(optimized_object["config"], sort_keys=True)))
        provenance.update(object_motion_optimized=object_audit["motion_policy"]["optimization_enabled"],
            object_motion_policy=object_audit["motion_policy"], optimized_object=object_binding,
            original_object_trajectory=reference_binding, object_trajectory_audit=object_audit)
        bindings.extend((object_binding, reference_binding))
    provenance["missing_scene_inputs"] = missing_scene_inputs
    arrays["retarget_provenance_json"] = np.asarray(json.dumps(provenance, sort_keys=True, allow_nan=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        restored = load_npz(temporary)
        if restored.keys() != arrays.keys() or any(not np.array_equal(restored[key], value) for key, value in arrays.items()):
            raise ValueError("Native data changed during serialization")
        for binding in bindings:
            _binding(binding["path"], binding["sha256"])
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"schema_version": 1, "status": "UNREVIEWED", "selected": False,
            "path": str(output), "sha256": file_sha256(output), "frames": len(rows), "fps": fps,
            "native_qpos": model.nq, "joint_names": native_names, "compiled_model_sha256": model_hash,
            "object_motion_present": object_sidecar is not None, "blender_scene_created": False,
            "missing_scene_inputs": missing_scene_inputs, "provenance": provenance}
