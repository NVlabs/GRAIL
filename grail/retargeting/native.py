"""Lossless native qpos sidecars for explicit G1-with-hands retargeting.

The training motion-library conversion can retain only actuator coordinates.
This export instead binds every native hinge by name/address, including all
fourteen finger coordinates, to its exact human/object source file.
With a contact config binding it records contact-aware IK output; otherwise it
records the original kinematic initialization. Neither is dynamics validation.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_native_trajectory(
    output_path,
    model,
    qpos,
    obj_R,
    obj_t,
    object_scale,
    fps,
    source_path,
    expected_source_sha256=None,
    scene_scale=1.0,
    height_offset=0.0,
    height_scale=1.0,
    pelvis_rotation=0.0,
    contact_config_sha256=None,
    static_scene=None,
):
    """Write/read back a named native NPZ without truncating non-actuated joints.

    `obj_R` and `obj_t` are the actual exported world transforms; `object_scale`
    is the scale relative to the original unscaled object vertices. Unknown
    scales must not be replaced by an assumed one. `fps` is the actual timeline
    of these arrays, not an instruction to resample them.
    """
    import mujoco

    output, source = Path(output_path), Path(source_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite native trajectory: {output}")
    digest = file_sha256(source)
    if expected_source_sha256 is not None and digest != expected_source_sha256:
        raise ValueError("Selected human/object source changed during retargeting")
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos):
        raise ValueError("qpos must retain all native model coordinates")
    frames = len(qpos)
    rotation, translation = np.asarray(obj_R), np.asarray(obj_t)
    if rotation.shape != (frames, 3, 3) or translation.shape != (frames, 3):
        raise ValueError("Native robot and object arrays must have the exact same frame count")
    if object_scale is None:
        raise ValueError("Native sidecar requires an explicitly recorded original object scale")
    scale = np.asarray(object_scale, dtype=float).reshape(-1)
    if scale.size not in (1, 3):
        raise ValueError("Original object scale must be scalar or contain three XYZ values")
    scale = np.broadcast_to(scale, (3,)).copy()
    if (
        not np.isfinite([scene_scale, height_offset, height_scale, pelvis_rotation]).all()
        or scene_scale <= 0
        or height_scale <= 0
    ):
        raise ValueError("Scene/height scales must be finite positive; offsets/rotations finite")
    for name, value in {
        "qpos": qpos,
        "obj_R": rotation,
        "obj_t": translation,
        "scale": scale,
    }.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite native export field: {name}")
    if np.any(scale <= 0) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Native object scale and source fps must be positive")
    if not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=2e-5) or np.any(
        np.linalg.det(rotation) < 0.99998
    ):
        raise ValueError("Native object rotations must be proper rotation matrices")
    scalar = np.flatnonzero(
        np.isin(model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE])
    )
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in scalar]
    if any(name is None for name in names):
        raise ValueError("Every native scalar joint must have a name")
    for j, kind in enumerate(model.jnt_type):
        start = int(model.jnt_qposadr[j]) + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
        if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            if not np.allclose(np.linalg.norm(qpos[:, start : start + 4], axis=1), 1, atol=1e-6):
                raise ValueError("Native quaternion must be unit length")
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, compiled)
    values = dict(
        qpos=qpos,
        fps=np.asarray(float(fps)),
        timestamps=np.arange(frames, dtype=float) / fps,
        obj_R=rotation,
        obj_t=translation,
        object_scale=scale,
        dof_names=np.asarray(names),
        qposadr=model.jnt_qposadr[scalar].copy(),
        source_path=np.asarray(str(source)),
        source_sha256=np.asarray(digest),
        compiled_model_sha256=np.asarray(hashlib.sha256(compiled.tobytes()).hexdigest()),
        scene_scale=np.asarray(float(scene_scale)),
        height_offset=np.asarray(float(height_offset)),
        height_scale=np.asarray(float(height_scale)),
        pelvis_rotation=np.asarray(float(pelvis_rotation)),
        export_kind=np.asarray("native_gmr_initialization_requires_contact_and_visual_review"),
        world_frame=np.asarray("grail_z_up_m"),
    )
    if static_scene is not None:
        if any(not key.startswith("static_scene_") for key in static_scene):
            raise ValueError("Static scene must not override native motion fields")
        values.update(static_scene)
    if contact_config_sha256 is not None:
        if (
            not isinstance(contact_config_sha256, str)
            or len(contact_config_sha256) != 64
            or any(char not in "0123456789abcdef" for char in contact_config_sha256)
        ):
            raise ValueError("Contact config binding must be a lowercase SHA-256 digest")
        values["contact_config_sha256"] = np.asarray(contact_config_sha256)
        values["export_kind"] = np.asarray(
            "native_gmr_reviewed_contact_requires_visual_and_collision_review"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
        with np.load(temporary, allow_pickle=False) as reloaded:
            for key, value in values.items():
                if not np.array_equal(reloaded[key], value):
                    raise ValueError(f"Native serialization changed {key}")
        if file_sha256(source) != digest:
            raise ValueError("Selected source changed while saving native sidecar")
        # Exclusive destination creation avoids silently replacing another run.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "frames": frames,
        "native_qpos": model.nq,
        "scalar_joints": len(names),
        "source_sha256": digest,
    }
