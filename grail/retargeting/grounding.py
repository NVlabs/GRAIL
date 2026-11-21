"""One declared world translation before native retargeting.

The complete native human mesh at source frame zero defines one constant
Z offset shared by the human, object, static scene and camera.
These NumPy-only helpers do not change shape/pose or modify inputs in place.
"""

from __future__ import annotations

import copy

import numpy as np

BLENDER_CAMERA_CONVENTION = "Blender camera-to-world; local -Z forward and +Y up"


def _finite(value, shape, label):
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite numeric {shape}")
    return result.astype(np.float64)


def _offset(value):
    return _finite(value, (3,), "World offset")


def first_frame_ground_offset(vertices):
    """Return ``(offset, receipt)`` from full native frame-zero vertices [V,3].

    The same additive ``[0, 0, -min_z]`` offset must be used for every human
    frame, every object frame and the support scene. A negative initial minimum
    produces an upward correction; no clipping or horizontal recentering occurs.
    """
    points = np.asarray(vertices)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not len(points)
        or points.dtype.kind not in "fiu"
        or not np.isfinite(points).all()
    ):
        raise ValueError("Initial native human mesh must be nonempty finite numeric [V,3]")
    minimum = float(np.min(points[:, 2]))
    offset = np.array([0.0, 0.0, -minimum], dtype=np.float64)
    return offset, {
        "schema": "grail.retargeting.initial_frame_grounding.v1",
        "method": "full_native_human_mesh_initial_frame_min_z",
        "source": "first_frame_min",
        "delta_z_m": float(offset[2]),
        "source_frame": 0,
        "vertex_count": len(points),
        "source_world_frame": "grail_z_up_m",
        "initial_human_min_z_m": minimum,
        "target_min_z_m": 0.0,
        "world_offset": offset.tolist(),
        "sign_convention": "translated_position = original_position + world_offset",
        "scope": "one constant offset for the entire human, object and static scene",
        "limitations": (
            "Initial placement only; no foot-lock, per-frame grounding or stability claim."
        ),
    }


def translate_native_motion(motion, offset):
    """Copy native motion and translate only its human/object world positions."""
    if not isinstance(motion, dict):
        raise ValueError("Native motion must be a mapping")
    delta = _offset(offset)
    values = {}
    for name in ("trans", "obj_t"):
        if name not in motion:
            raise ValueError(f"Native motion requires {name}")
        value = np.asarray(motion[name])
        if (
            value.ndim != 2
            or value.shape[1:] != (3,)
            or not len(value)
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Native {name} must be nonempty finite floating [T,3]")
        values[name] = value
    if values["trans"].shape != values["obj_t"].shape:
        raise ValueError("Human and object must cover the same source timeline")
    translated = copy.deepcopy(motion)
    for name, value in values.items():
        shifted = value + delta.astype(value.dtype)
        if not np.isfinite(shifted).all():
            raise ValueError(f"World translation overflowed native {name}")
        translated[name] = shifted
    return translated


def _rotation(value):
    rotation = _finite(value, (3, 3), "Camera rotation")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-5, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-5, rtol=0
    ):
        raise ValueError("Camera rotation must be a proper rotation")
    return rotation


def translate_camera(camera, offset):
    """Copy an explicitly declared camera into the translated world.

    Supported JSON contracts are the existing Blender ``original_fields``
    camera, column-vector ``world_to_camera`` with root R/t, and row-vector
    ``pytorch3d_world_to_camera`` with root R/T. Intrinsics and rotations are
    unchanged. Source bindings still identify the original camera, while the
    added derivative metadata identifies its transformed effective fields.
    """
    if not isinstance(camera, dict):
        raise ValueError("Camera must be a mapping with an explicit convention")
    delta = _offset(offset)
    result = copy.deepcopy(camera)
    convention = camera.get("convention")
    changed = []
    if convention == BLENDER_CAMERA_CONVENTION:
        fields = result.get("original_fields")
        if not isinstance(fields, dict) or not {"cam_R", "cam_t"} <= fields.keys():
            raise ValueError("Blender camera requires original_fields.cam_R and cam_t")
        _rotation(fields["cam_R"])
        fields["cam_t"] = (_finite(fields["cam_t"], (3,), "Camera position") + delta).tolist()
        changed.append("original_fields.cam_t")
        if "obj_t" in fields:
            shape = np.asarray(fields["obj_t"]).shape
            if shape not in ((3,), (3, 1), (1, 3)):
                raise ValueError("Camera's recorded initial object position must be one XYZ point")
            position = _finite(fields["obj_t"], shape, "Recorded initial object position")
            fields["obj_t"] = (position + delta.reshape(shape)).tolist()
            changed.append("original_fields.obj_t")
    elif convention in {"world_to_camera", "pytorch3d_world_to_camera"}:
        key = "t" if convention == "world_to_camera" else "T"
        if "R" not in camera or key not in camera:
            raise ValueError(f"{convention} requires R and {key}")
        rotation = _rotation(camera["R"])
        translation = _finite(camera[key], (3,), "Camera translation")
        adjustment = rotation @ delta if key == "t" else delta @ rotation
        result[key] = (translation - adjustment).tolist()
        changed.append(key)
    else:
        raise ValueError(f"Unsupported explicit camera convention: {convention!r}")
    metadata = {
        "schema": "grail.retargeting.camera_world_translation.v1",
        "world_offset": delta.tolist(),
        "effective_fields_changed": changed,
        "policy": "Common world translation; preserve camera-space projection and intrinsics.",
        "source_bindings": "Inherited source bindings identify the original, unmodified camera.",
    }
    if "world_translation" in result:
        metadata["parent"] = result["world_translation"]
    result["world_translation"] = metadata
    return result
