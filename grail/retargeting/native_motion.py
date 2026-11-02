"""Canonical native motion I/O shared by public SOMA retargeting tools.

Pickle and joblib inputs must be trusted local GRAIL outputs. Canonical NPZ
inputs never enable pickle and must carry an explicit model tag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_TYPES = ("soma03", "smplx", "g1_smplx")
FACE_FIELDS = ("jaw_pose", "leye_pose", "reye_pose", "expression")


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _model_type(value):
    value = np.asarray(value)
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError("model_type must be an explicit scalar string")
    result = str(value.item())
    if result not in MODEL_TYPES:
        raise ValueError(f"Unsupported native model_type {result!r}; expected {MODEL_TYPES}")
    return result


def _finite(motion, name):
    if name not in motion:
        raise ValueError(f"Missing native field {name}")
    value = np.asarray(motion[name])
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return value


def _shape_parameter(motion, name, width, frames):
    value = _finite(motion, name)
    if value.shape not in ((width,), (1, width), (frames, width)):
        raise ValueError(f"{name} must be [{width}], [1,{width}] or [T,{width}]")
    return value


def validate_motion(motion):
    """Validate canonical native fields without inferring a body model."""
    required = {"model_type", "poses", "trans", "fps", "obj_R", "obj_t", "obj_scale"}
    missing = required - motion.keys()
    if missing:
        raise ValueError(f"Missing canonical fields: {sorted(missing)}")
    model_type = _model_type(motion["model_type"])
    for name, value in motion.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"{name}: object arrays are not canonical native fields")
    poses = _finite(motion, "poses")
    if poses.ndim not in (2, 3) or not len(poses) or poses.dtype.kind != "f":
        raise ValueError("poses must be a nonempty floating [T,J*3] or [T,J,3] array")
    if poses.ndim == 3 and poses.shape[-1] != 3:
        raise ValueError("poses last dimension must be three")
    frames = len(poses)
    width = int(np.prod(poses.shape[1:]))
    expected = (231,) if model_type == "soma03" else (66, 165)
    if width not in expected:
        raise ValueError(f"{model_type} pose width must be one of {expected}, got {width}")
    if _finite(motion, "trans").shape != (frames, 3):
        raise ValueError("trans must be [T,3]")
    fps = _finite(motion, "fps")
    if fps.shape != () or float(fps) <= 0:
        raise ValueError("fps must be a positive scalar")
    rotation = _finite(motion, "obj_R")
    if rotation.shape != (frames, 3, 3):
        raise ValueError("obj_R must be [T,3,3]")
    if not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-5):
        raise ValueError("obj_R must contain orthonormal rotations")
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("obj_R must contain proper rotations")
    if _finite(motion, "obj_t").shape != (frames, 3):
        raise ValueError("obj_t must be [T,3]")
    scale = _finite(motion, "obj_scale")
    if scale.shape not in ((), (3,)) or np.any(scale <= 0):
        raise ValueError("obj_scale must be a positive scalar or [3]")
    if model_type == "soma03":
        _shape_parameter(motion, "identity_coeffs", 128, frames)
        scales = _shape_parameter(motion, "scale_params", 57, frames)
        if np.any(scales <= 0):
            raise ValueError("SOMA03 native scales must be positive")
        if "human_scale" in motion and not np.all(_finite(motion, "human_scale") == 1):
            raise ValueError("SOMA03 disallows an additional display-mesh scale")
    else:
        _shape_parameter(motion, "betas", 10, frames)
        human_scale = _finite(motion, "human_scale")
        if human_scale.shape != () or float(human_scale) <= 0:
            raise ValueError("human_scale must be a positive scalar")
    return {
        "model_type": model_type,
        "frame_count": frames,
        "joint_count": width // 3,
        "fps": float(fps),
    }


def _check_tags(container, expected):
    for name in ("model_type", "body_model", "model"):
        if name in container:
            tag = np.asarray(container[name])
            if tag.shape == () and tag.dtype.kind in "US" and _model_type(tag) != expected:
                raise ValueError(f"Native {name} tag differs from requested {expected}")


def load_native(path, model_type):
    """Load canonical NPZ or trusted GRAIL human/object checkpoint."""
    model_type = _model_type(model_type)
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            motion = {name: archive[name].copy() for name in archive.files}
        metadata = validate_motion(motion)
        if metadata["model_type"] != model_type:
            raise ValueError("Canonical model_type differs from the explicit requested model_type")
        return motion
    if path.suffix not in (".pkl", ".pickle", ".joblib"):
        raise ValueError("Native input must be .npz or a trusted .pkl/.pickle/.joblib")
    import joblib

    original = joblib.load(path)
    human, obj = original["human_data"], original["obj_data"]
    _check_tags(original, model_type)
    _check_tags(human, model_type)
    names = ["poses", "trans"]
    names += (
        ["identity_coeffs", "scale_params"]
        if model_type == "soma03"
        else ["betas", "left_hand_pose", "right_hand_pose"]
    )
    names += [name for name in FACE_FIELDS if name in human and human[name] is not None]
    motion = {name: _array(human[name]) for name in names}
    motion.update({name: _array(obj[name]) for name in ("obj_R", "obj_t", "obj_scale")})
    if motion["obj_scale"].size == 1:
        motion["obj_scale"] = motion["obj_scale"].reshape(())
    elif motion["obj_scale"].size == 3:
        motion["obj_scale"] = motion["obj_scale"].reshape(3)
    motion["model_type"] = np.asarray(model_type)
    fps = human.get("mocap_frame_rate", human.get("fps", original.get("fps")))
    if fps is None:
        raise ValueError("Native checkpoint must explicitly declare source fps")
    motion["fps"] = _array(fps).reshape(())
    if model_type != "soma03" or "scale" in human:
        motion["human_scale"] = _array(human.get("scale", 1.0)).reshape(())
    validate_motion(motion)
    return motion
