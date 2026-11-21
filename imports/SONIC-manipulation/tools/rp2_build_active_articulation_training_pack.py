#!/usr/bin/env python3
"""Build fixed-base, one-active-link articulation training data from RP2 exports.

The output preserves the GRAIL single-object policy contract: the moving link is
slot 0 in object observations/rewards, while its fixed mechanism frame and every
unrelated prop are represented by physics rather than padded object slots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

from rp2_build_static_context_usd import _write_static_context_usda


FAMILY_SPECS = {
    "classic_door": {
        "motion": "navigation_session_02_001__T016",
        "parent": r"^classic_door_frame__",
        "panel": r"^classic_door__",
        "joint_type": "revolute",
        "lower": 0.0,
        "upper": math.radians(110.0),
    },
    "sliding_door": {
        "motion": "navigation_session_02_001__T018",
        "parent": r"^sliding_door_frame__",
        "panel": r"^sliding_door__",
        "joint_type": "prismatic",
        "lower": 0.0,
        "upper": 0.75,
    },
    "classic_door_pair": {
        "motion": "navigation_session_36_001__T020",
        "parent": r"^classic_door_frame__",
        "panel": r"^classic_door__",
        "joint_type": "revolute",
        "lower": 0.0,
        "upper": math.radians(110.0),
    },
    "sliding_door_pair": {
        "motion": "navigation_session_36_001__T022",
        "parent": r"^sliding_door_frame__",
        "panel": r"^sliding_door__",
        "joint_type": "prismatic",
        "lower": 0.0,
        "upper": 0.75,
    },
    "laptop": {
        "motion": "shelf_laptop_mid_pickup_shift_weight_return_001__T029",
        "parent": r"^laptop_bottom__",
        "panel": r"^laptop_top__",
        "joint_type": "revolute",
        "lower": 0.0,
        "upper": math.radians(135.0),
    },
    "office_chair": {
        "motion": "chair_adjustment_session_08_001__T031",
        "parent": r"^office_chair_bottom__",
        "panel": r"^office_chair_top__",
        "joint_type": "revolute",
        "lower": math.radians(-170.0),
        "upper": math.radians(170.0),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-motion-lib", type=Path, required=True)
    parser.add_argument("--output-motion-lib", type=Path, required=True)
    parser.add_argument(
        "--robot-source-dir",
        type=Path,
        help="Optional directory of repaired robot PKLs; object data still comes from source.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILY_SPECS),
        choices=sorted(FAMILY_SPECS),
    )
    parser.add_argument("--clip-length", type=int, default=300)
    parser.add_argument("--collision-approximation", default="sdf")
    parser.add_argument("--parent-mass", type=float, default=10.0)
    parser.add_argument("--panel-mass", type=float, default=2.0)
    parser.add_argument(
        "--root-mode",
        choices=("spawn-config", "auto", "fixed", "dynamic"),
        default="spawn-config",
        help=(
            "Root mobility policy. 'spawn-config' preserves legacy runtime behavior; "
            "'auto' marks stationary fitted parents for mixed-mode runtime clamping."
        ),
    )
    parser.add_argument(
        "--enable-adjacent-self-collision",
        action="store_true",
        help="Allow collision between the articulation frame and moving panel.",
    )
    parser.add_argument(
        "--contact-validator",
        type=Path,
        help="Optional SOMA validator JSON used to gate G1/contact consistency.",
    )
    parser.add_argument(
        "--max-median-g1-contact-error",
        type=float,
        help="Reject a family whose tracked link exceeds this contact error in meters.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_entry(path: Path, key: str) -> dict[str, Any]:
    data = joblib.load(path)
    if isinstance(data, dict) and key in data:
        return data[key]
    if isinstance(data, dict) and len(data) == 1:
        return next(iter(data.values()))
    raise ValueError(f"Expected one entry for {key} in {path}")


def _dump_entry(path: Path, key: str, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({key: value}, path)


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _find_one_name(names: list[str], pattern: str) -> int:
    matches = [index for index, name in enumerate(names) if re.search(pattern, str(name))]
    if len(matches) != 1:
        raise ValueError(f"Pattern {pattern!r} matched indices {matches}; names={names}")
    return matches[0]


def _contact_quality(
    validator: dict[str, Any], motion: str, object_pattern: str
) -> dict[str, Any]:
    motion_records = [
        record for record in validator.get("records", []) if record.get("motion_key") == motion
    ]
    if len(motion_records) != 1:
        raise ValueError(
            f"Contact validator has {len(motion_records)} records for motion {motion!r}"
        )
    objects = [
        item
        for item in motion_records[0].get("objects", [])
        if re.search(object_pattern, str(item.get("name", "")))
    ]
    if len(objects) != 1:
        raise ValueError(
            f"Contact validator pattern {object_pattern!r} matched {len(objects)} objects "
            f"for {motion!r}"
        )
    contact = objects[0].get("soma_contact") or {}
    error_key = "median_g1_same_side_contact_vector_error_m"
    error = contact.get(error_key)
    if error is None:
        error_key = "median_g1_contact_vector_error_m"
        error = contact.get(error_key)
    if error is None:
        raise ValueError(
            f"Contact validator has no median G1 contact error for {motion}/{objects[0]['name']}"
        )
    contact_frames = int(
        contact.get("either_hand_contact_frames", contact.get("sampled_contact_frames", 0))
    )
    if contact_frames <= 0:
        raise ValueError(
            f"Contact validator has no hand-contact frames for {motion}/{objects[0]['name']}"
        )
    return {
        "validator_object_name": str(objects[0]["name"]),
        "contact_frames": contact_frames,
        "median_g1_contact_error_m": float(error),
        "median_g1_contact_error_metric": error_key,
    }


def _pose_matrices(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=-1)
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(pos), axis=0)
    matrices[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    matrices[:, :3, 3] = pos
    return matrices


def _quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)


def _pose_velocities(
    position: np.ndarray, quat_wxyz: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference world-frame linear and angular velocity."""
    if len(position) < 2:
        zeros = np.zeros((len(position), 3), dtype=np.float32)
        return zeros, zeros.copy()
    dt = 1.0 / float(fps)
    linear = np.gradient(position, dt, axis=0)
    rotations = _pose_matrices(position, quat_wxyz)[:, :3, :3]
    step_rotation = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    step_angular = Rotation.from_matrix(step_rotation).as_rotvec() / dt
    angular = np.empty_like(position)
    angular[0] = step_angular[0]
    angular[-1] = step_angular[-1]
    if len(position) > 2:
        angular[1:-1] = 0.5 * (step_angular[:-1] + step_angular[1:])
    return linear.astype(np.float32), angular.astype(np.float32)


def _axis_frame(axis: np.ndarray) -> np.ndarray:
    x_axis = np.asarray(axis, dtype=np.float64)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    helper = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, x_axis))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0])
    y_axis = np.cross(helper, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)
    return np.stack([x_axis, y_axis, np.cross(x_axis, y_axis)], axis=1)


def _fit_revolute(relative: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    rotations = relative[:, :3, :3]
    translations = relative[:, :3, 3]
    rest_rotation = rotations[0]
    rotvec = Rotation.from_matrix(rotations @ rest_rotation.T).as_rotvec()
    centered = rotvec - rotvec.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or float(singular_values[0]) < 1e-5:
        raise ValueError("Insufficient relative rotation to fit a revolute joint")
    axis = vh[0]
    q = np.unwrap(rotvec @ axis)
    q -= q[0]
    if abs(float(q.min())) > abs(float(q.max())):
        axis = -axis
        q = -q
    axis /= max(float(np.linalg.norm(axis)), 1e-9)

    sample = np.linspace(0, len(relative) - 1, min(500, len(relative)), dtype=np.int64)
    lhs = []
    rhs = []
    for frame in sample:
        lhs.append(np.concatenate([rotations[frame], -np.eye(3)], axis=1))
        rhs.append(-translations[frame])
    solution, _, _, _ = np.linalg.lstsq(np.concatenate(lhs), np.concatenate(rhs), rcond=None)
    pivot_child = solution[:3]
    pivot_parent = solution[3:]

    predicted_rotation = Rotation.from_rotvec(q[:, None] * axis[None]).as_matrix() @ rest_rotation
    predicted_translation = pivot_parent[None] - np.einsum(
        "nij,j->ni", predicted_rotation, pivot_child
    )
    position_error = np.linalg.norm(predicted_translation - translations, axis=-1)
    orientation_error = Rotation.from_matrix(
        predicted_rotation @ np.swapaxes(rotations, -1, -2)
    ).magnitude()
    fit = {
        "axis_parent": axis,
        "pivot_parent": pivot_parent,
        "pivot_child": pivot_child,
        "rest_rotation": rest_rotation,
        "rest_translation": translations[0],
        "position_fit_rms_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "position_fit_max_m": float(position_error.max()),
        "orientation_fit_rms_rad": float(np.sqrt(np.mean(np.square(orientation_error)))),
        "orientation_fit_max_rad": float(orientation_error.max()),
    }
    return fit, q


def _fit_prismatic(relative: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    rotations = relative[:, :3, :3]
    translations = relative[:, :3, 3]
    delta = translations - translations[0]
    centered = delta - delta.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or float(singular_values[0]) < 1e-5:
        raise ValueError("Insufficient relative translation to fit a prismatic joint")
    axis = vh[0]
    q = delta @ axis
    if abs(float(q.min())) > abs(float(q.max())):
        axis = -axis
        q = -q
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    predicted_translation = translations[0] + q[:, None] * axis[None]
    position_error = np.linalg.norm(predicted_translation - translations, axis=-1)
    orientation_error = Rotation.from_matrix(
        rotations @ np.swapaxes(rotations[:1], -1, -2)
    ).magnitude()
    fit = {
        "axis_parent": axis,
        "pivot_parent": translations[0],
        "pivot_child": np.zeros(3, dtype=np.float64),
        "rest_rotation": rotations[0],
        "rest_translation": translations[0],
        "position_fit_rms_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "position_fit_max_m": float(position_error.max()),
        "orientation_fit_rms_rad": float(np.sqrt(np.mean(np.square(orientation_error)))),
        "orientation_fit_max_rad": float(orientation_error.max()),
    }
    return fit, q


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            face = [int(token.split("/")[0]) for token in line.split()[1:]]
            face = [index - 1 if index > 0 else len(vertices) + index for index in face]
            for offset in range(1, len(face) - 1):
                triangles.append([face[0], face[offset], face[offset + 1]])
    if not vertices or not triangles:
        raise ValueError(f"OBJ contains no triangle mesh: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _mesh_path(source: Path, motion: str, object_name: str) -> Path:
    expected = source / "object_mesh_obj" / f"{motion}__{_safe_name(object_name)}.obj"
    if expected.is_file():
        return expected
    candidates = list((source / "object_mesh_obj").glob(f"{motion}__*.obj"))
    matches = [path for path in candidates if path.stem.endswith(_safe_name(object_name))]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not resolve mesh for {motion}/{object_name}; candidates={candidates}"
        )
    return matches[0]


def _set_xform(prim, rotation: np.ndarray, translation: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    quat = _quat_wxyz(rotation)
    xform.AddOrientOp().Set(
        Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    )


def _define_mesh_link(
    stage,
    path: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    mass: float,
    color: tuple[float, float, float],
    collision_approximation: str,
):
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    _set_xform(prim, rotation, translation)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/geometry")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    approximation_attr = mesh_collision.CreateApproximationAttr()
    approximation_attr.Set(collision_approximation)
    try:
        from pxr import PhysxSchema

        if collision_approximation == "sdf":
            approximation_attr.Set(PhysxSchema.Tokens.sdf)
            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim())
            sdf_api.CreateSdfResolutionAttr().Set(256)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    except Exception:
        if collision_approximation == "sdf":
            from pxr import Sdf

            approximation_attr.Set("sdf")
            schemas = list(mesh.GetPrim().GetAppliedSchemas())
            if "PhysxSDFMeshCollisionAPI" not in schemas:
                schemas.append("PhysxSDFMeshCollisionAPI")
                schema_list = Sdf.TokenListOp()
                schema_list.explicitItems = schemas
                mesh.GetPrim().SetMetadata("apiSchemas", schema_list)
            mesh.GetPrim().CreateAttribute(
                "physxSDFMeshCollision:sdfResolution",
                Sdf.ValueTypeNames.Int,
                custom=False,
            ).Set(256)
    return prim


def _write_articulation_usd(
    path: Path,
    *,
    parent_mesh: tuple[np.ndarray, np.ndarray],
    panel_mesh: tuple[np.ndarray, np.ndarray],
    fit: dict[str, Any],
    joint_type: str,
    lower: float,
    upper: float,
    parent_mass: float,
    panel_mass: float,
    collision_approximation: str,
    enable_adjacent_self_collision: bool,
    author_fixed_root: bool,
) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root_path = "/ActiveArticulation"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)

    frame = _define_mesh_link(
        stage,
        f"{root_path}/frame",
        *parent_mesh,
        np.eye(3),
        np.zeros(3),
        mass=parent_mass,
        color=(0.30, 0.36, 0.42),
        collision_approximation=collision_approximation,
    )
    # Keep the root API at one relative path for every active-entity USD. This
    # is required for one IsaacLab articulation tensor view over mixed assets.
    UsdPhysics.ArticulationRootAPI.Apply(root)
    try:
        from pxr import PhysxSchema

        PhysxSchema.PhysxArticulationAPI.Apply(root)
    except Exception:
        pass
    if author_fixed_root:
        root_joint = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/root_fixed_joint")
        root_joint.CreateBody1Rel().SetTargets([frame.GetPath()])
    panel = _define_mesh_link(
        stage,
        f"{root_path}/panel",
        *panel_mesh,
        fit["rest_rotation"],
        fit["rest_translation"],
        mass=panel_mass,
        color=(0.65, 0.36, 0.16),
        collision_approximation=collision_approximation,
    )

    joint_path = f"{root_path}/mechanism_joint"
    if joint_type == "revolute":
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateLowerLimitAttr(math.degrees(lower))
        joint.CreateUpperLimitAttr(math.degrees(upper))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    else:
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateLowerLimitAttr(lower)
        joint.CreateUpperLimitAttr(upper)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    joint.CreateBody0Rel().SetTargets([frame.GetPath()])
    joint.CreateBody1Rel().SetTargets([panel.GetPath()])
    joint.CreateAxisAttr("X")
    parent_axis_frame = _axis_frame(fit["axis_parent"])
    child_axis_frame = fit["rest_rotation"].T @ parent_axis_frame
    q0 = _quat_wxyz(parent_axis_frame)
    q1 = _quat_wxyz(child_axis_frame)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, fit["pivot_parent"])))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, fit["pivot_child"])))
    joint.CreateLocalRot0Attr(Gf.Quatf(float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])))
    joint.CreateLocalRot1Attr(Gf.Quatf(float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])))
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    if not enable_adjacent_self_collision:
        UsdPhysics.FilteredPairsAPI.Apply(frame).CreateFilteredPairsRel().AddTarget(
            panel.GetPath()
        )
    stage.GetRootLayer().Save()


def _activity_window(entry: dict[str, Any], object_index: int, clip_length: int) -> tuple[int, int]:
    num_frames = int(np.asarray(entry["root_pos"]).shape[0])
    score = np.zeros(num_frames, dtype=np.float64)
    motion_active = np.zeros(num_frames, dtype=bool)
    for key, weight in (("object_motion_mask", 1.0), ("object_interaction_mask", 0.5)):
        if key in entry:
            values = np.asarray(entry[key])
            if values.ndim == 2 and object_index < values.shape[1]:
                score += weight * values[:, object_index]
                if key == "object_motion_mask":
                    motion_active |= values[:, object_index] > 0.5
    if not motion_active.any():
        pos = np.asarray(entry["root_pos"], dtype=np.float64)[:, object_index]
        motion_delta = np.zeros(num_frames, dtype=np.float64)
        motion_delta[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
        motion_active = motion_delta > 1e-4
        score += motion_delta
    contact_active = np.zeros(num_frames, dtype=bool)
    for key in ("contact_points_left_hand", "contact_points_right_hand"):
        for frame in entry.get(key, {}):
            if 0 <= int(frame) < num_frames:
                score[int(frame)] += 2.0
                contact_active[int(frame)] = True
    if not motion_active.any():
        raise ValueError("Active articulation has no moving reference frames")
    if not contact_active.any():
        raise ValueError("Active articulation has no reference hand-contact frames")
    if clip_length <= 0 or num_frames <= clip_length:
        return 0, num_frames
    summed = np.convolve(score, np.ones(clip_length), mode="valid")
    motion_counts = np.convolve(
        motion_active.astype(np.int32), np.ones(clip_length, dtype=np.int32), mode="valid"
    )
    contact_counts = np.convolve(
        contact_active.astype(np.int32), np.ones(clip_length, dtype=np.int32), mode="valid"
    )
    contains_both = (motion_counts > 0) & (contact_counts > 0)
    if not contains_both.any():
        motion_frames = np.flatnonzero(motion_active)
        contact_frames = np.flatnonzero(contact_active)
        min_gap = int(np.min(np.abs(motion_frames[:, None] - contact_frames[None, :])))
        raise ValueError(
            "No clip can contain both active-object motion and reference contact: "
            f"clip_length={clip_length}, min_gap={min_gap}"
        )
    summed = np.where(contains_both, summed, -np.inf)
    start = int(np.argmax(summed))
    return start, start + clip_length


def _slice_value(value: Any, start: int, end: int, num_frames: int) -> Any:
    if isinstance(value, dict) and all(isinstance(key, (int, np.integer)) for key in value):
        return {int(key) - start: item for key, item in value.items() if start <= int(key) < end}
    if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == num_frames:
        return value[start:end].copy() if hasattr(value, "copy") else value[start:end].clone()
    return value


def _slice_entry(entry: dict[str, Any], start: int, end: int, num_frames: int) -> dict[str, Any]:
    return {key: _slice_value(value, start, end, num_frames) for key, value in entry.items()}


def _select_panel(entry: dict[str, Any], panel_index: int) -> dict[str, Any]:
    output = dict(entry)
    names = list(entry.get("object_names", []))
    panel_name = names[panel_index]
    output["object_name"] = panel_name
    output["object_names"] = [panel_name]
    num_objects = len(names)
    for key, value in list(entry.items()):
        if not hasattr(value, "shape") or len(value.shape) < 2:
            continue
        if value.shape[1] == num_objects:
            output[key] = value[:, panel_index : panel_index + 1].copy()
    return output


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def _build_family(
    source: Path,
    output: Path,
    family: str,
    *,
    robot_source_dir: Path,
    clip_length: int,
    parent_mass: float,
    panel_mass: float,
    root_mode: str,
    collision_approximation: str,
    enable_adjacent_self_collision: bool,
    contact_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    motion = spec["motion"]
    masked_path = source / "objects_with_masks" / f"{motion}.pkl"
    object_path = source / "objects" / f"{motion}.pkl"
    masked = _load_entry(masked_path, motion)
    objects = _load_entry(object_path, motion)
    names = list(masked["object_names"])
    parent_index = _find_one_name(names, spec["parent"])
    panel_index = _find_one_name(names, spec["panel"])
    root_pos = np.asarray(masked["root_pos"], dtype=np.float64)
    root_quat = np.asarray(masked["root_quat"], dtype=np.float64)
    parent_world = _pose_matrices(root_pos[:, parent_index], root_quat[:, parent_index])
    panel_world = _pose_matrices(root_pos[:, panel_index], root_quat[:, panel_index])
    relative = np.linalg.inv(parent_world) @ panel_world
    if spec["joint_type"] == "revolute":
        fit, joint_pos = _fit_revolute(relative)
        margin = math.radians(2.0)
    else:
        fit, joint_pos = _fit_prismatic(relative)
        margin = 0.01
    lower = min(float(spec["lower"]), float(joint_pos.min()) - margin)
    upper = max(float(spec["upper"]), float(joint_pos.max()) + margin)
    robot = _load_entry(robot_source_dir / f"{motion}.pkl", motion)
    if "fps" not in masked or "fps" not in objects or "fps" not in robot:
        raise KeyError(
            f"{motion}: paired robot/object reference is missing native fps"
        )
    fps = float(masked["fps"])
    objects_fps = float(objects["fps"])
    robot_fps = float(robot["fps"])
    if (
        not np.isfinite(fps)
        or fps <= 0.0
        or not np.isfinite(objects_fps)
        or objects_fps <= 0.0
        or not np.isfinite(robot_fps)
        or robot_fps <= 0.0
    ):
        raise ValueError(
            f"{motion}: invalid paired FPS "
            f"{robot_fps} / {fps} / {objects_fps}"
        )
    if (
        abs(fps - objects_fps) > 1.0e-9
        or abs(fps - robot_fps) > 1.0e-9
    ):
        raise ValueError(
            f"{motion}: paired FPS mismatch "
            f"{robot_fps} / {fps} / {objects_fps}"
        )
    joint_vel = np.gradient(joint_pos, 1.0 / fps).astype(np.float32)
    root_lin_vel, root_ang_vel = _pose_velocities(
        root_pos[:, parent_index], root_quat[:, parent_index], fps
    )

    start, end = _activity_window(masked, panel_index, clip_length)
    num_frames = len(root_pos)
    robot_frames = int(np.asarray(robot["root_trans_offset"]).shape[0])
    if robot_frames != num_frames:
        raise ValueError(
            f"Robot/object frame mismatch for {motion}: robot={robot_frames}, object={num_frames}"
        )
    robot_out = _slice_entry(robot, start, end, num_frames)

    clip_parent_world = parent_world[start:end]
    parent_translation_delta = np.linalg.norm(
        clip_parent_world[:, :3, 3] - clip_parent_world[0, :3, 3], axis=-1
    )
    parent_orientation_delta = Rotation.from_matrix(
        clip_parent_world[:, :3, :3]
        @ clip_parent_world[0, :3, :3].T
    ).magnitude()
    root_translation_max = float(parent_translation_delta.max())
    root_orientation_max = float(parent_orientation_delta.max())
    recommended_fix_root_link = (
        root_translation_max < 0.01 and root_orientation_max < math.radians(5.0)
    )
    author_fixed_root = root_mode == "fixed"
    root_is_dynamic = not (
        root_mode == "fixed"
        or (root_mode == "auto" and recommended_fix_root_link)
    )

    active_channels = {
        "active_articulation_root_pos": root_pos[start:end, parent_index].astype(np.float32),
        "active_articulation_root_quat": root_quat[start:end, parent_index].astype(np.float32),
        "active_articulation_root_lin_vel": root_lin_vel[start:end],
        "active_articulation_root_ang_vel": root_ang_vel[start:end],
        "active_articulation_joint_pos": joint_pos[start:end, None].astype(np.float32),
        "active_articulation_joint_vel": joint_vel[start:end, None].astype(np.float32),
        "active_articulation_joint_valid_mask": np.ones(
            (end - start, 1), dtype=np.float32
        ),
        "active_articulation_root_dynamic_mask": np.full(
            (end - start, 1), float(root_is_dynamic), dtype=np.float32
        ),
        "active_entity_tracking_valid": np.ones(
            (end - start, 1), dtype=np.float32
        ),
    }
    metadata = {
        "family": family,
        "parent_object_name": names[parent_index],
        "tracked_object_name": names[panel_index],
        "tracked_body_name": "panel",
        "tracked_prim_path": "{ENV_REGEX_NS}/ActiveArticulation/panel",
        "active_entity_abi": "frame/panel/mechanism_joint-v1",
        "articulation_root_prim_path": "/ActiveArticulation",
        "reference_joint_names": ["mechanism_joint"],
        "joint_type": spec["joint_type"],
        "joint_lower": lower,
        "joint_upper": upper,
        "parent_mass_kg": float(parent_mass),
        "panel_mass_kg": float(panel_mass),
        "root_translation_from_clip_start_max_m": root_translation_max,
        "root_orientation_from_clip_start_max_rad": root_orientation_max,
        "recommended_fix_root_link": recommended_fix_root_link,
        "root_mode": root_mode,
        "authored_fixed_root": author_fixed_root,
        "root_is_dynamic": root_is_dynamic,
        "runtime_root_clamp_required": not root_is_dynamic and not author_fixed_root,
        "fit": {
            key: value
            for key, value in fit.items()
            if isinstance(value, (float, int))
        },
        "source_frames": num_frames,
        "fps": fps,
        "robot_source": str((robot_source_dir / f"{motion}.pkl").resolve()),
        "clip_start": start,
        "clip_end": end,
        "adjacent_self_collision_enabled": bool(enable_adjacent_self_collision),
    }
    if contact_quality is not None:
        metadata["contact_quality"] = contact_quality

    for source_entry, subdir in ((objects, "objects"), (masked, "objects_with_masks")):
        selected = _select_panel(source_entry, panel_index)
        selected = _slice_entry(selected, start, end, num_frames)
        selected.update(active_channels)
        selected["active_articulation"] = metadata
        _dump_entry(output / subdir / f"{motion}.pkl", motion, selected)
    _dump_entry(output / "robot" / f"{motion}.pkl", motion, robot_out)
    meta_path = source / "meta" / f"{motion}.pkl"
    if meta_path.is_file():
        _link(meta_path, output / "meta" / meta_path.name)

    parent_mesh_path = _mesh_path(source, motion, names[parent_index])
    panel_mesh_path = _mesh_path(source, motion, names[panel_index])
    articulation_path = output / "active_articulation_usd" / f"{motion}.usd"
    _write_articulation_usd(
        articulation_path,
        parent_mesh=_read_obj(parent_mesh_path),
        panel_mesh=_read_obj(panel_mesh_path),
        fit=fit,
        joint_type=spec["joint_type"],
        lower=lower,
        upper=upper,
        parent_mass=parent_mass,
        panel_mass=panel_mass,
        collision_approximation=collision_approximation,
        enable_adjacent_self_collision=enable_adjacent_self_collision,
        author_fixed_root=author_fixed_root,
    )

    usd_list_path = source / "full_scene_object_usds" / f"{motion}.txt"
    usd_paths = [Path(line.strip()) for line in usd_list_path.read_text().splitlines() if line.strip()]
    if panel_index >= len(usd_paths):
        raise ValueError(
            f"USD list has no entry for active panel index {panel_index}: {usd_list_path}"
        )
    active_usd_list = output / "full_scene_object_usds" / usd_list_path.name
    active_usd_list.parent.mkdir(parents=True, exist_ok=True)
    active_usd_list.write_text(str(usd_paths[panel_index].resolve()) + "\n")
    context_path = output / "static_context_usd_grail_paired" / f"{motion}.usda"
    context_count = _write_static_context_usda(
        context_path,
        motion_key=motion,
        usd_paths=usd_paths,
        root_pos=root_pos[start],
        root_quat=root_quat[start],
        excluded_indices={parent_index, panel_index},
    )
    metadata["static_context_object_count"] = context_count
    metadata["active_articulation_usd"] = str(articulation_path.resolve())
    metadata["static_context_usd"] = str(context_path.resolve())

    bps_source = source / "bps" / f"{motion}__{_safe_name(names[panel_index])}.npy"
    if not bps_source.is_file():
        raise FileNotFoundError(bps_source)
    _link(bps_source, output / "bps" / f"{motion}.npy")
    for aux in ("_basis.npy",):
        if (source / "bps" / aux).is_file():
            _link(source / "bps" / aux, output / "bps" / aux)
    return metadata


def main() -> int:
    args = _parse_args()
    source = args.source_motion_lib.resolve()
    output = args.output_motion_lib.resolve()
    robot_source_dir = (
        args.robot_source_dir.resolve()
        if args.robot_source_dir is not None
        else source / "robot"
    )
    if not robot_source_dir.is_dir():
        raise FileNotFoundError(f"Robot source directory does not exist: {robot_source_dir}")
    if args.max_median_g1_contact_error is not None and args.contact_validator is None:
        raise ValueError(
            "--max-median-g1-contact-error requires --contact-validator"
        )
    contact_quality_by_family: dict[str, dict[str, Any]] = {}
    if args.contact_validator is not None:
        validator = json.loads(args.contact_validator.read_text())
        for family in args.families:
            spec = FAMILY_SPECS[family]
            quality = _contact_quality(validator, spec["motion"], spec["panel"])
            if (
                args.max_median_g1_contact_error is not None
                and quality["median_g1_contact_error_m"]
                > args.max_median_g1_contact_error
            ):
                raise ValueError(
                    f"{family}: median G1/reference contact error "
                    f"{quality['median_g1_contact_error_m']:.3f} m exceeds "
                    f"{args.max_median_g1_contact_error:.3f} m"
                )
            contact_quality_by_family[family] = quality
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records = []
    for family in args.families:
        records.append(
            _build_family(
                source,
                output,
                family,
                robot_source_dir=robot_source_dir,
                clip_length=args.clip_length,
                parent_mass=args.parent_mass,
                panel_mass=args.panel_mass,
                root_mode=args.root_mode,
                collision_approximation=args.collision_approximation,
                enable_adjacent_self_collision=args.enable_adjacent_self_collision,
                contact_quality=contact_quality_by_family.get(family),
            )
        )
    manifest = {
        "schema_version": 1,
        "source_motion_lib": str(source),
        "robot_source_dir": str(robot_source_dir),
        "output_motion_lib": str(output),
        "policy_contract": "one active entity plus static collision context",
        "active_entity_abi": "frame/panel/mechanism_joint-v1",
        "contact_validator": (
            str(args.contact_validator.resolve()) if args.contact_validator is not None else None
        ),
        "max_median_g1_contact_error_m": args.max_median_g1_contact_error,
        "adjacent_self_collision_enabled": bool(args.enable_adjacent_self_collision),
        "root_mode": args.root_mode,
        "records": records,
    }
    (output / "active_articulation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
