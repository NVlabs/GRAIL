"""Native, explicitly tagged HOI motion I/O and isolated geometry evaluation.

Importing this module only needs NumPy. Pickle/joblib inputs must be trusted;
the interchange NPZ format never enables pickle. Geometry dependencies are
loaded only by :func:`evaluate_motion`, normally in ``native_worker``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

MODEL_TYPES = ("soma03",)
# Native full-pose indices. Wrists are body controls; individual fingers,
# face/eyes, terminal joints and the root remain frozen.
EDITABLE_BODY_JOINTS = {
    "soma03": (1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 39, 40, 41, 42, 67, 68, 69, 70, 72, 73, 74, 75),
}
DERIVED_FIELDS = (
    "vertices",
    "verts",
    "human_vertices",
    "joints",
    "native_joints",
    "geometry",
    "render_cache",
    "contact_cache",
    "eval_data",
)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _model_type(value):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError("model_type must be an explicit scalar string")
    value = str(array.item())
    if value not in MODEL_TYPES:
        raise ValueError(f"Unsupported native model_type {value!r}; expected {MODEL_TYPES}")
    return value


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
    """Validate canonical native fields without inferring identity from shape."""
    required = {"model_type", "poses", "trans", "fps", "obj_R", "obj_t", "obj_scale"}
    if required - motion.keys():
        raise ValueError(f"Missing canonical fields: {sorted(required - motion.keys())}")
    model_type = _model_type(motion["model_type"])
    for name, value in motion.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"{name}: object arrays are not canonical native fields")
    poses = _finite(motion, "poses")
    if poses.ndim not in (2, 3) or poses.shape[0] == 0 or poses.dtype.kind != "f":
        raise ValueError("poses must be a nonempty floating [T,J*3] or [T,J,3] array")
    count = len(poses)
    if poses.ndim == 3 and poses.shape[-1] != 3:
        raise ValueError("poses last dimension must be three")
    width = int(np.prod(poses.shape[1:]))
    if width != 231:
        raise ValueError(f"SOMA03 poses require 77 native joints (231 values); got {width}")
    if _finite(motion, "trans").shape != (count, 3):
        raise ValueError("trans must be [T,3]")
    fps = _finite(motion, "fps")
    if fps.shape != () or float(fps) <= 0:
        raise ValueError("fps must be a positive scalar")
    rotation = _finite(motion, "obj_R")
    if rotation.shape != (count, 3, 3):
        raise ValueError("obj_R must be [T,3,3]")
    if not np.allclose(
        rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-5
    ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("obj_R must contain proper rotation matrices")
    if _finite(motion, "obj_t").shape != (count, 3):
        raise ValueError("obj_t must be [T,3]")
    scale = _finite(motion, "obj_scale")
    if scale.shape not in ((), (3,)) or np.any(scale <= 0):
        raise ValueError("obj_scale must be a positive scalar or [3]")
    _shape_parameter(motion, "identity_coeffs", 128, count)
    scales = _shape_parameter(motion, "scale_params", 57, count)
    if np.any(scales <= 0):
        raise ValueError("SOMA03 native scales must be positive")
    if "human_scale" in motion and not np.all(_finite(motion, "human_scale") == 1):
        raise ValueError("SOMA03 disallows additional display-mesh scale")
    return dict(
        model_type=model_type,
        frame_count=count,
        joint_count=width // 3,
        fps=float(fps),
        editable_body_joints=list(EDITABLE_BODY_JOINTS[model_type]),
    )


def _check_tags(container, expected):
    for name in ("model_type", "body_model", "model"):
        if name in container:
            tag = np.asarray(container[name])
            if tag.shape == () and tag.dtype.kind in "US":
                if _model_type(tag) != expected:
                    raise ValueError(f"Native {name} tag differs from requested {expected}")


def load_native(path, model_type):
    """Load a canonical NPZ or trusted GRAIL human_data/obj_data checkpoint.

    An untagged historical pickle requires the caller's explicit model type;
    any existing native tag must agree. NPZ always requires its own tag.
    """
    model_type = _model_type(model_type)
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            motion = {name: archive[name].copy() for name in archive.files}
        metadata = validate_motion(motion)
        if metadata["model_type"] != model_type:
            raise ValueError("Canonical model_type differs from explicit requested model_type")
        return motion
    if path.suffix not in (".pkl", ".pickle", ".joblib"):
        raise ValueError("Native input must be .npz or a trusted .pkl/.pickle/.joblib")
    import joblib

    original = joblib.load(path)
    human, obj = original["human_data"], original["obj_data"]
    _check_tags(original, model_type)
    _check_tags(human, model_type)
    names = ["poses", "trans", "identity_coeffs", "scale_params"]
    motion = {name: _array(human[name]) for name in names}
    motion.update({name: _array(obj[name]) for name in ("obj_R", "obj_t", "obj_scale")})
    if motion["obj_scale"].size == 1:
        motion["obj_scale"] = motion["obj_scale"].reshape(())
    elif motion["obj_scale"].size == 3:
        motion["obj_scale"] = motion["obj_scale"].reshape(3)
    motion["model_type"] = np.asarray(model_type)
    frame_rate = human.get("mocap_frame_rate", human.get("fps", original.get("fps")))
    if frame_rate is None:
        raise ValueError("Native checkpoint must explicitly declare source fps")
    motion["fps"] = _array(frame_rate).reshape(())
    if "scale" in human:
        motion["human_scale"] = _array(human.get("scale", 1.0)).reshape(())
    validate_motion(motion)
    return motion


def _exact(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def export_native(parent_path, candidate_motion, output_path, model_type):
    """Write a new checkpoint, changing body poses only and removing stale caches.

    Frozen canonical fields and all undeclared pose joints must match the parent
    bit for bit. Unknown native metadata survives unchanged; no inherited quality
    claim becomes a review of this candidate. Returns a removal/roundtrip receipt.
    """
    parent = load_native(parent_path, model_type)
    meta = validate_motion(candidate_motion)
    if parent.keys() != candidate_motion.keys():
        raise ValueError("Candidate canonical fields differ from parent")
    for name in parent:
        if name != "poses" and not _exact(parent[name], candidate_motion[name]):
            raise ValueError(f"Candidate changed frozen field {name}")
    before, after = parent["poses"], np.asarray(candidate_motion["poses"])
    if before.shape != after.shape or before.dtype != after.dtype:
        raise ValueError("Candidate changed native pose shape or dtype")
    frozen = sorted(set(range(meta["joint_count"])) - set(meta["editable_body_joints"]))
    if not _exact(
        before.reshape(len(before), -1, 3)[:, frozen], after.reshape(len(after), -1, 3)[:, frozen]
    ):
        raise ValueError("Candidate changed frozen root/finger/face joints")
    destination = Path(output_path)
    parent_path = Path(parent_path)
    if destination.exists() or destination.resolve() == parent_path.resolve():
        raise FileExistsError(f"Preserve existing checkpoint: {destination}")
    removed = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    if parent_path.suffix == ".npz":
        if destination.suffix != ".npz":
            raise ValueError("Canonical parent export must remain .npz")
        payload = {}
        for name, value in candidate_motion.items():
            if name in DERIVED_FIELDS:
                removed.append(f"root.{name}")
            else:
                payload[name] = value
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **payload)
    else:
        if destination.suffix not in (".pkl", ".pickle", ".joblib"):
            raise ValueError("Native pickle export must remain .pkl/.pickle/.joblib")
        import joblib

        original = joblib.load(parent_path)
        result = copy.deepcopy(original)
        native_poses = original["human_data"]["poses"]
        if hasattr(native_poses, "detach"):
            import torch

            result["human_data"]["poses"] = torch.as_tensor(
                after.copy(), dtype=native_poses.dtype, device=native_poses.device
            )
        else:
            result["human_data"]["poses"] = after.copy()
        for label, container in (("root", result), ("human_data", result["human_data"])):
            for name in DERIVED_FIELDS:
                if name in container:
                    del container[name]
                    removed.append(f"{label}.{name}")
        with destination.open("xb") as stream:
            joblib.dump(result, stream)
    reloaded = load_native(destination, model_type)
    if any(
        not _exact(value, reloaded[name])
        for name, value in candidate_motion.items()
        if name not in DERIVED_FIELDS
    ):
        raise RuntimeError("Native export failed exact parameter roundtrip")
    return dict(
        output=str(destination.resolve()),
        removed_derived_fields=removed,
        native_reload_exact=True,
        status="unreviewed",
        parameters_only=True,
    )


def _assert_bound_identity(motion, identity, scales):
    for name, bound in (("identity_coeffs", identity), ("scale_params", scales)):
        actual, expected = np.asarray(motion[name]), _array(bound).reshape(1, -1)
        actual = actual.reshape(-1, expected.shape[-1])
        if not np.array_equal(actual, np.broadcast_to(expected, actual.shape)):
            raise ValueError(f"Frozen native {name} differs from the bound fitted identity")


def _cached_model(model_config, cache, create):
    if cache is None:
        return create()
    key = json.dumps(model_config, sort_keys=True, allow_nan=False)
    if key not in cache:
        cache[key] = create()
    return cache[key]


def evaluate_motion(motion, model_config, frames=None, *, model_cache=None):
    """Evaluate original native models; returns vertices/joints/faces/source frames.

    Use backend=native with gem_x_soma_assets and an optional bound identity.
    An optional caller-owned model_cache reuses loaded assets for a bounded
    worker invocation. Callers must keep those assets immutable while cached;
    motion/identity bindings are still validated on every evaluation.
    """
    from .interpolation import sample_motion

    metadata = validate_motion(motion)
    model_type = _model_type(model_config["model_type"])
    if model_type != metadata["model_type"]:
        raise ValueError("Evaluator model_type differs from motion")
    if frames is None:
        frames = np.arange(metadata["frame_count"], dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    sampled = sample_motion(motion, {"schema_version": 1, "windows": []}, frames)
    backend = model_config["backend"]
    if backend != "native":
        raise ValueError(f"Unknown explicit native evaluator backend: {backend!r}")
    device = model_config.get("device", "cpu")
    from grail.models.soma03_model import Soma03HumanModel

    model = _cached_model(
        model_config, model_cache, lambda: Soma03HumanModel(model_config, device=device)
    )
    if model_config.get("soma_identity_params_path"):
        bound = model.load_shape_params(model_config, None, device)
        _assert_bound_identity(motion, *bound)
    native = dict(sampled, scale=1.0)
    vertices, faces, joints = model.generate_mesh(native, output_joints=True)
    vertices, joints, faces = _array(vertices), _array(joints), _array(faces)
    return dict(
        vertices=np.asarray(vertices),
        joints=np.asarray(joints),
        faces=np.asarray(faces, dtype=np.int64),
        frames=frames.copy(),
        obj_R=sampled["obj_R"],
        obj_t=sampled["obj_t"],
        obj_scale=sampled["obj_scale"],
    )
