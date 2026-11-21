"""Reference G1 collision surfaces and canonical full-body contact geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.rp2.discrete_contact import (
    CONTACT_BODY_GROUPS,
    CONTACT_ENTITY_CLASSES,
    DEFAULT_CONTACT_FORBIDDEN_THRESHOLD_M,
    DEFAULT_CONTACT_REQUIRED_THRESHOLD_M,
    G1_CONTACT_GROUP_LINKS,
    canonical_contact_shape,
    discretize_contact_distances,
)
from gear_sonic.rp2.static_context import (
    StaticBox,
    points_to_static_box_contact_distance,
)
from gear_sonic.rp2.surface_relation import (


    closest_points_to_mesh,
    world_points_to_object_local,
)


G1_BODY_DOF_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Stored GRAIL hand_dof_pos and DEX31 dof[29:43] use this per-hand,
# breadth-first order. Runtime name-maps the same order to IsaacLab.
G1_HAND_DOF_NAMES_USD: tuple[str, ...] = (
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

G1_HAND_PRIMITIVE_NAMES: Mapping[str, tuple[str, ...]] = {
    "left": (
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
    ),
    "right": (
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
    ),
}

G1_HAND_PRIMITIVE_OPEN: Mapping[str, np.ndarray] = {
    side: np.zeros(7, dtype=np.float64) for side in ("left", "right")
}
G1_HAND_PRIMITIVE_CLOSED: Mapping[str, np.ndarray] = {
    "left": np.asarray(
        [-1.42857143, -2.14285714, -1.42857143, -2.14285714, 0.0, 1.0, 1.0],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [1.42857143, 2.14285714, 1.42857143, 2.14285714, 0.0, -1.0, -1.0],
        dtype=np.float64,
    ),
}

ZERO_POSE_POSITION_TOLERANCE_M = 1.0e-5
ZERO_POSE_ORIENTATION_TOLERANCE_RAD = 5.0e-4


def _kd_workers() -> int:
    """cKDTree worker count bounded by the CPUs this process may actually use.

    `workers=-1` means os.cpu_count() (188 on cpu_av nodes) and cKDTree spawns
    that many Python threads PER QUERY CALL: a 1-motion cProfile (2026-09-01)
    showed 1.47M Thread.start calls costing 170 of 320 s inside a 2-CPU cgroup.
    """
    import os as _os

    env = _os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        return max(1, len(_os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, _os.cpu_count() or 1)


@dataclass(frozen=True)
class KinematicJoint:
    name: str
    kind: str
    parent: str
    child: str
    parent_to_joint: np.ndarray
    child_to_joint: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class G1ReferenceTrajectory:
    root_position: np.ndarray
    root_quaternion_xyzw: np.ndarray
    joint_position: Mapping[str, np.ndarray]
    hand_source: str
    source_fps: float
    target_fps: float


@dataclass(frozen=True)
class G1CollisionModel:
    source_usd: Path
    source_sha256: str
    root_link: str
    joints: tuple[KinematicJoint, ...]
    surface_points_by_link: Mapping[str, np.ndarray]
    collider_rows: tuple[Mapping[str, Any], ...]
    zero_pose_position_error_max_m: float
    zero_pose_orientation_error_max_rad: float
    source_layer_sha256: Mapping[str, str] = field(default_factory=dict)
    source_composed_sha256: str = ""

    @property
    def revolute_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints if joint.kind == "revolute")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_layer_fingerprint(stage: Any) -> tuple[dict[str, str], str]:
    rows = {}
    for layer in stage.GetUsedLayers():
        real_path = getattr(layer, "realPath", "")
        if not real_path:
            continue
        path = Path(str(real_path))
        if path.is_file():
            rows[str(path.resolve())] = _sha256(path)
    if not rows:
        raise ValueError("composed G1 stage has no hashable local USD layers")
    digest = hashlib.sha256()
    for path, value in sorted(rows.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return rows, digest.hexdigest()


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _batch_transform(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    matrix = np.broadcast_to(
        np.eye(4, dtype=np.float64), rotation.shape[:-2] + (4, 4)
    ).copy()
    matrix[..., :3, :3] = rotation
    matrix[..., :3, 3] = translation
    return matrix


def _quat_wxyz_matrix(value: Any) -> np.ndarray:
    real = float(value.GetReal())
    imaginary = np.asarray(value.GetImaginary(), dtype=np.float64)
    return Rotation.from_quat(
        [imaginary[0], imaginary[1], imaginary[2], real]
    ).as_matrix()


def _gf_matrix_to_column(value: Any) -> np.ndarray:
    # Gf.Matrix4d uses row-vector point multiplication. Transpose it into the
    # column-vector convention used by the reference FK below.
    return np.asarray(value, dtype=np.float64).T


def _axis_vector(token: str) -> np.ndarray:
    axis = np.zeros(3, dtype=np.float64)
    try:
        axis[{"X": 0, "Y": 1, "Z": 2}[str(token).upper()]] = 1.0
    except KeyError as error:
        raise ValueError(f"unsupported USD joint/primitive axis {token!r}") from error
    return axis


def _joint_motion(joint: KinematicJoint, value: np.ndarray) -> np.ndarray:
    frames = len(value)
    if joint.kind == "fixed":
        return np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
    if joint.kind == "revolute":
        rotation = Rotation.from_rotvec(
            np.asarray(value)[:, None] * joint.axis[None]
        ).as_matrix()
        return _batch_transform(rotation, np.zeros((frames, 3)))
    if joint.kind == "prismatic":
        translation = np.asarray(value)[:, None] * joint.axis[None]
        return _batch_transform(
            np.broadcast_to(np.eye(3), (frames, 3, 3)), translation
        )
    raise ValueError(f"unsupported joint kind {joint.kind!r}")


def forward_kinematics(
    model: G1CollisionModel,
    trajectory: G1ReferenceTrajectory,
) -> dict[str, np.ndarray]:
    frames = len(trajectory.root_position)
    root_rotation = Rotation.from_quat(
        trajectory.root_quaternion_xyzw
    ).as_matrix()
    world = {
        model.root_link: _batch_transform(
            root_rotation, trajectory.root_position
        )
    }
    for joint in model.joints:
        if joint.parent not in world:
            raise RuntimeError(
                f"kinematic joint order is invalid at {joint.name}"
            )
        if joint.kind == "fixed":
            value = np.zeros(frames, dtype=np.float64)
        else:
            if joint.name not in trajectory.joint_position:
                raise KeyError(f"reference lacks joint {joint.name}")
            value = np.asarray(
                trajectory.joint_position[joint.name], dtype=np.float64
            )
            if value.shape != (frames,):
                raise ValueError(
                    f"{joint.name}: trajectory shape {value.shape} != {(frames,)}"
                )
        parent_to_joint = np.broadcast_to(
            joint.parent_to_joint, (frames, 4, 4)
        )
        joint_to_child = np.broadcast_to(
            np.linalg.inv(joint.child_to_joint), (frames, 4, 4)
        )
        world[joint.child] = (
            world[joint.parent]
            @ parent_to_joint
            @ _joint_motion(joint, value)
            @ joint_to_child
        )
    return world


def _interpolate_linear(
    values: np.ndarray,
    source_fps: float,
    target_times: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source_times = np.arange(len(values), dtype=np.float64) / float(source_fps)
    flat = values.reshape(len(values), -1)
    result = np.empty((len(target_times), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(
            target_times, source_times, flat[:, column]
        )
    return result.reshape((len(target_times),) + values.shape[1:])


def _interpolate_quaternion(
    quaternion_xyzw: np.ndarray,
    source_fps: float,
    target_times: np.ndarray,
) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12
    )
    if len(quaternion) == 1:
        return np.broadcast_to(quaternion, (len(target_times), 4)).copy()
    source_times = np.arange(len(quaternion), dtype=np.float64) / float(source_fps)
    return Slerp(source_times, Rotation.from_quat(quaternion))(
        np.clip(target_times, source_times[0], source_times[-1])
    ).as_quat()


def _primitive_hand_values(
    robot: Mapping[str, Any],
    *,
    frames: int,
    mode: str,
) -> tuple[dict[str, np.ndarray], str]:
    values: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        key = f"hand_action_{side}"
        if key not in robot:
            raise KeyError(f"reference lacks {key} and explicit finger DOFs")
        action = np.asarray(robot[key], dtype=np.float64)
        if action.shape != (frames,) or not np.isfinite(action).all():
            raise ValueError(f"{key}: invalid shape or values {action.shape}")
        if mode == "discrete":
            fraction = (action >= 0.0).astype(np.float64)
        elif mode == "linear":
            fraction = np.clip((action + 1.0) * 0.5, 0.0, 1.0)
        else:
            raise ValueError(f"unknown hand primitive mode {mode!r}")
        position = (
            G1_HAND_PRIMITIVE_OPEN[side][None]
            + fraction[:, None]
            * (
                G1_HAND_PRIMITIVE_CLOSED[side]
                - G1_HAND_PRIMITIVE_OPEN[side]
            )[None]
        )
        for column, name in enumerate(G1_HAND_PRIMITIVE_NAMES[side]):
            values[name] = position[:, column]
    return values, f"runtime_s3_{mode}_hand_primitive"


def reference_trajectory(
    robot: Mapping[str, Any],
    *,
    target_frames: int,
    target_fps: float,
    hand_primitive_mode: str = "discrete",
) -> G1ReferenceTrajectory:
    root_position = np.asarray(robot["root_trans_offset"], dtype=np.float64)
    root_quaternion = np.asarray(robot["root_rot"], dtype=np.float64)
    dof = np.asarray(robot["dof"], dtype=np.float64)
    source_fps = float(robot.get("fps", target_fps))
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("reference FPS must be positive")
    if (
        root_position.ndim != 2
        or root_position.shape[1] != 3
        or root_quaternion.shape != (len(root_position), 4)
        or dof.ndim != 2
        or dof.shape[0] != len(root_position)
        or dof.shape[1] < len(G1_BODY_DOF_NAMES)
    ):
        raise ValueError(
            "invalid G1 reference arrays: "
            f"root={root_position.shape}, quat={root_quaternion.shape}, "
            f"dof={dof.shape}"
        )
    if not (
        np.isfinite(root_position).all()
        and np.isfinite(root_quaternion).all()
        and np.isfinite(dof).all()
    ):
        raise ValueError("G1 reference contains non-finite values")
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    maximum_time = (len(root_position) - 1) / source_fps
    if len(target_times) and target_times[-1] > maximum_time + 1.5 / min(
        source_fps, target_fps
    ):
        raise ValueError("target object clock exceeds G1 reference clock")

    source_joint: dict[str, np.ndarray] = {
        name: dof[:, index]
        for index, name in enumerate(G1_BODY_DOF_NAMES)
    }
    explicit_hand = robot.get("hand_dof_pos")
    if explicit_hand is not None:
        hand = np.asarray(explicit_hand, dtype=np.float64)
        if hand.shape != (len(root_position), len(G1_HAND_DOF_NAMES_USD)):
            raise ValueError(f"hand_dof_pos has invalid shape {hand.shape}")
        for index, name in enumerate(G1_HAND_DOF_NAMES_USD):
            source_joint[name] = hand[:, index]
        hand_source = "stored_hand_dof_pos_usd_order"
    elif dof.shape[1] >= len(G1_BODY_DOF_NAMES) + len(G1_HAND_DOF_NAMES_USD):
        start = len(G1_BODY_DOF_NAMES)
        for index, name in enumerate(G1_HAND_DOF_NAMES_USD):
            source_joint[name] = dof[:, start + index]
        hand_source = "stored_dof43_usd_hand_order"
    else:
        primitive, hand_source = _primitive_hand_values(
            robot, frames=len(root_position), mode=hand_primitive_mode
        )
        source_joint.update(primitive)

    joint_position = {
        name: _interpolate_linear(
            values[:, None], source_fps, target_times
        )[:, 0]
        for name, values in source_joint.items()
    }
    return G1ReferenceTrajectory(
        root_position=_interpolate_linear(
            root_position, source_fps, target_times
        ),
        root_quaternion_xyzw=_interpolate_quaternion(
            root_quaternion, source_fps, target_times
        ),
        joint_position=joint_position,
        hand_source=hand_source,
        source_fps=source_fps,
        target_fps=float(target_fps),
    )


def _halton(index: int, base: int) -> float:
    value = 0.0
    fraction = 1.0
    while index:
        fraction /= base
        value += fraction * (index % base)
        index //= base
    return value


def _sample_triangle_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    maximum_points: int,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=-1,
    )
    valid = area > 1.0e-14
    triangles = triangles[valid]
    area = area[valid]
    if not len(triangles):
        raise ValueError("collision mesh has no nondegenerate triangles")
    count = max(1, int(maximum_points) - min(len(vertices), int(maximum_points) // 4))
    cdf = np.cumsum(area)
    targets = (np.arange(count, dtype=np.float64) + 0.5) / count * cdf[-1]
    face_index = np.searchsorted(cdf, targets, side="left")
    samples = np.empty((count, 3), dtype=np.float64)
    for sample_index, triangle_index in enumerate(face_index):
        u = _halton(sample_index + 1, 2)
        v = _halton(sample_index + 1, 3)
        root_u = np.sqrt(u)
        barycentric = np.asarray(
            [1.0 - root_u, root_u * (1.0 - v), root_u * v]
        )
        samples[sample_index] = (
            barycentric[:, None] * triangles[triangle_index]
        ).sum(axis=0)
    vertex_count = min(len(vertices), max(1, int(maximum_points) - count))
    vertex_index = np.linspace(
        0, len(vertices) - 1, vertex_count, dtype=np.int64
    )
    return np.concatenate((vertices[vertex_index], samples), axis=0)


def _triangulate_mesh(mesh: Any, time: Any) -> tuple[np.ndarray, np.ndarray]:
    points = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    if points is None or counts is None or indices is None:
        raise ValueError(f"{mesh.GetPath()}: missing mesh topology")
    vertices = np.asarray(points, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    faces = []
    cursor = 0
    for count in np.asarray(counts, dtype=np.int64):
        polygon = indices[cursor : cursor + int(count)]
        faces.extend(
            polygon[[0, index, index + 1]]
            for index in range(1, int(count) - 1)
        )
        cursor += int(count)
    if not faces:
        raise ValueError(f"{mesh.GetPath()}: no triangulated faces")
    return vertices, np.asarray(faces, dtype=np.int64)


def _ring_points(
    radius: float,
    height: float,
    axis: str,
    resolution: int,
    *,
    capsule: bool,
) -> np.ndarray:
    points = []
    half = 0.5 * float(height)
    for longitudinal in np.linspace(-half, half, max(2, resolution // 2)):
        for angle in np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False):
            points.append(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    longitudinal,
                ]
            )
    if capsule:
        for pole in (-1.0, 1.0):
            for phi in np.linspace(
                0.0, 0.5 * np.pi, max(3, resolution // 4)
            ):
                ring_radius = radius * np.sin(phi)
                z = pole * (half + radius * np.cos(phi))
                for angle in np.linspace(
                    0.0, 2.0 * np.pi, resolution, endpoint=False
                ):
                    points.append(
                        [
                            ring_radius * np.cos(angle),
                            ring_radius * np.sin(angle),
                            z,
                        ]
                    )
    else:
        for pole in (-1.0, 1.0):
            z = pole * half
            for radial in np.linspace(0.0, radius, max(3, resolution // 4)):
                for angle in np.linspace(
                    0.0, 2.0 * np.pi, resolution, endpoint=False
                ):
                    points.append(
                        [radial * np.cos(angle), radial * np.sin(angle), z]
                    )
    value = np.asarray(points, dtype=np.float64)
    if axis.upper() == "X":
        return value[:, [2, 0, 1]]
    if axis.upper() == "Y":
        return value[:, [1, 2, 0]]
    if axis.upper() == "Z":
        return value
    raise ValueError(f"unsupported primitive axis {axis!r}")


def _cube_points(size: float, resolution: int) -> np.ndarray:
    coordinate = np.linspace(-0.5 * size, 0.5 * size, max(2, resolution // 2))
    points = []
    for fixed_axis in range(3):
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        for sign in (-1.0, 1.0):
            for first in coordinate:
                for second in coordinate:
                    point = np.zeros(3)
                    point[fixed_axis] = sign * 0.5 * size
                    point[free_axes] = (first, second)
                    points.append(point)
    return np.asarray(points, dtype=np.float64)


def _sphere_points(radius: float, resolution: int) -> np.ndarray:
    count = max(32, resolution * resolution)
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    angle = np.pi * (1.0 + np.sqrt(5.0)) * index
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return radius * np.stack(
        (radial * np.cos(angle), radial * np.sin(angle), z), axis=-1
    )


def _collision_context(prim: Any, root: Any, collision_api: Any) -> Any | None:
    cursor = prim
    while cursor and cursor.IsValid():
        if cursor.HasAPI(collision_api):
            api = collision_api(cursor)
            enabled = api.GetCollisionEnabledAttr().Get()
            if enabled is False:
                return None
            return cursor
        if cursor == root:
            break
        cursor = cursor.GetParent()
    return None


def _nearest_link(prim: Any, link_paths: Mapping[str, Any]) -> str | None:
    by_path = {str(path): name for name, path in link_paths.items()}
    cursor = prim
    while cursor and cursor.IsValid():
        name = by_path.get(str(cursor.GetPath()))
        if name is not None:
            return name
        cursor = cursor.GetParent()
    return None


def _topological_joints(
    joints: Sequence[KinematicJoint], root_link: str
) -> tuple[KinematicJoint, ...]:
    pending = list(joints)
    ordered = []
    reached = {root_link}
    while pending:
        ready = [joint for joint in pending if joint.parent in reached]
        if not ready:
            raise ValueError(
                "USD joint graph is disconnected or cyclic; pending="
                f"{[joint.name for joint in pending[:12]]}"
            )
        for joint in sorted(ready, key=lambda item: item.name):
            if joint.child in reached:
                raise ValueError(f"USD joint graph has duplicate child {joint.child}")
            ordered.append(joint)
            reached.add(joint.child)
            pending.remove(joint)
    return tuple(ordered)


def load_collision_model(
    path: Path,
    *,
    maximum_points_per_collider: int = 256,
    primitive_resolution: int = 16,
) -> G1CollisionModel:
    """Load composed training USD joints and enabled collision surfaces."""
    from pxr import Usd, UsdGeom, UsdPhysics

    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"could not open G1 USD {source}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise ValueError(f"{source}: no valid default prim")
    source_layer_sha256, source_composed_sha256 = _stage_layer_fingerprint(
        stage
    )
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isclose(meters_per_unit, 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{source}: metersPerUnit={meters_per_unit}, expected 1")

    rigid_prims = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_prims[str(prim.GetPath())] = prim
    if not rigid_prims:
        raise ValueError(f"{source}: no rigid-body links")
    link_paths = {
        prim.GetName(): prim.GetPath()
        for prim in rigid_prims.values()
    }

    joints = []
    child_links = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        schema = UsdPhysics.Joint(prim)
        body0 = schema.GetBody0Rel().GetTargets()
        body1 = schema.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            raise ValueError(f"{prim.GetPath()}: joint must connect two bodies")
        parent_prim = stage.GetPrimAtPath(body0[0])
        child_prim = stage.GetPrimAtPath(body1[0])
        if not parent_prim.IsValid() or not child_prim.IsValid():
            raise ValueError(f"{prim.GetPath()}: invalid joint body target")
        if prim.IsA(UsdPhysics.RevoluteJoint):
            kind = "revolute"
            axis = _axis_vector(
                UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            )
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            kind = "prismatic"
            axis = _axis_vector(
                UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get()
            )
        elif prim.IsA(UsdPhysics.FixedJoint):
            kind = "fixed"
            axis = np.zeros(3)
        else:
            raise ValueError(
                f"{prim.GetPath()}: unsupported G1 joint type {prim.GetTypeName()}"
            )
        joint = KinematicJoint(
            name=prim.GetName(),
            kind=kind,
            parent=parent_prim.GetName(),
            child=child_prim.GetName(),
            parent_to_joint=_transform(
                _quat_wxyz_matrix(schema.GetLocalRot0Attr().Get()),
                np.asarray(schema.GetLocalPos0Attr().Get(), dtype=np.float64),
            ),
            child_to_joint=_transform(
                _quat_wxyz_matrix(schema.GetLocalRot1Attr().Get()),
                np.asarray(schema.GetLocalPos1Attr().Get(), dtype=np.float64),
            ),
            axis=axis,
        )
        joints.append(joint)
        child_links.add(joint.child)
    roots = sorted(set(link_paths) - child_links)
    if roots != ["pelvis"]:
        raise ValueError(f"{source}: unexpected kinematic roots {roots}")
    ordered_joints = _topological_joints(joints, roots[0])

    expected_actuated = set(G1_BODY_DOF_NAMES) | set(G1_HAND_DOF_NAMES_USD)
    actual_actuated = {
        joint.name for joint in ordered_joints if joint.kind != "fixed"
    }
    if actual_actuated != expected_actuated:
        raise ValueError(
            f"{source}: actuated joint contract mismatch; "
            f"missing={sorted(expected_actuated - actual_actuated)}, "
            f"extra={sorted(actual_actuated - expected_actuated)}"
        )

    time = Usd.TimeCode.Default()
    xform_cache = UsdGeom.XformCache(time)
    point_rows: dict[str, list[np.ndarray]] = {}
    collider_rows = []
    prim_range = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    for prim in prim_range:
        if not (
            prim.IsA(UsdGeom.Mesh)
            or prim.IsA(UsdGeom.Capsule)
            or prim.IsA(UsdGeom.Cylinder)
            or prim.IsA(UsdGeom.Cube)
            or prim.IsA(UsdGeom.Sphere)
        ):
            continue
        collision_holder = _collision_context(
            prim, root, UsdPhysics.CollisionAPI
        )
        if collision_holder is None:
            continue
        link = _nearest_link(prim, link_paths)
        if link is None:
            raise ValueError(f"{prim.GetPath()}: collider is not below a G1 link")

        approximation = None
        if prim.IsA(UsdGeom.Mesh):
            vertices, faces = _triangulate_mesh(UsdGeom.Mesh(prim), time)
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            approximation = mesh_api.GetApproximationAttr().Get()
            if approximation == "convexHull" and len(vertices) >= 4:
                hull = ConvexHull(vertices)
                faces = np.asarray(hull.simplices, dtype=np.int64)
            points = _sample_triangle_surface(
                vertices, faces, maximum_points_per_collider
            )
        elif prim.IsA(UsdGeom.Capsule):
            shape = UsdGeom.Capsule(prim)
            points = _ring_points(
                float(shape.GetRadiusAttr().Get()),
                float(shape.GetHeightAttr().Get()),
                str(shape.GetAxisAttr().Get()),
                primitive_resolution,
                capsule=True,
            )
        elif prim.IsA(UsdGeom.Cylinder):
            shape = UsdGeom.Cylinder(prim)
            points = _ring_points(
                float(shape.GetRadiusAttr().Get()),
                float(shape.GetHeightAttr().Get()),
                str(shape.GetAxisAttr().Get()),
                primitive_resolution,
                capsule=False,
            )
        elif prim.IsA(UsdGeom.Cube):
            points = _cube_points(
                float(UsdGeom.Cube(prim).GetSizeAttr().Get()),
                primitive_resolution,
            )
        else:
            points = _sphere_points(
                float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()),
                primitive_resolution,
            )

        link_prim = stage.GetPrimAtPath(link_paths[link])
        collider_world = _gf_matrix_to_column(
            xform_cache.GetLocalToWorldTransform(prim)
        )
        link_world = _gf_matrix_to_column(
            xform_cache.GetLocalToWorldTransform(link_prim)
        )
        collider_to_link = np.linalg.inv(link_world) @ collider_world
        points_link = (
            points @ collider_to_link[:3, :3].T
            + collider_to_link[:3, 3]
        )
        if not np.isfinite(points_link).all():
            raise ValueError(f"{prim.GetPath()}: non-finite collision points")
        point_rows.setdefault(link, []).append(points_link.astype(np.float32))
        collider_rows.append(
            {
                "path": str(prim.GetPath()),
                "link": link,
                "type": prim.GetTypeName(),
                "approximation": (
                    str(approximation) if approximation is not None else None
                ),
                "surface_points": int(len(points_link)),
            }
        )

    surface_by_link = {
        link: np.concatenate(rows, axis=0)
        for link, rows in point_rows.items()
    }
    missing_groups = {
        group: [
            link
            for link in links
            if link not in surface_by_link
        ]
        for group, links in G1_CONTACT_GROUP_LINKS.items()
        if not any(link in surface_by_link for link in links)
    }
    if missing_groups:
        raise ValueError(
            f"{source}: canonical groups have no collision geometry "
            f"{missing_groups}"
        )

    frames = 1
    zero_joint = {
        name: np.zeros(frames, dtype=np.float64)
        for name in expected_actuated
    }
    placeholder = G1CollisionModel(
        source_usd=source,
        source_sha256=_sha256(source),
        root_link=roots[0],
        joints=ordered_joints,
        surface_points_by_link=surface_by_link,
        collider_rows=tuple(collider_rows),
        zero_pose_position_error_max_m=0.0,
        zero_pose_orientation_error_max_rad=0.0,
        source_layer_sha256=source_layer_sha256,
        source_composed_sha256=source_composed_sha256,
    )
    zero_trajectory = G1ReferenceTrajectory(
        root_position=np.zeros((1, 3)),
        root_quaternion_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        joint_position=zero_joint,
        hand_source="zero_pose",
        source_fps=1.0,
        target_fps=1.0,
    )
    computed = forward_kinematics(placeholder, zero_trajectory)
    position_errors = []
    orientation_errors = []
    root_world = _gf_matrix_to_column(
        xform_cache.GetLocalToWorldTransform(
            stage.GetPrimAtPath(link_paths[roots[0]])
        )
    )
    root_inverse = np.linalg.inv(root_world)
    for link, matrix in computed.items():
        if link not in link_paths:
            continue
        authored = root_inverse @ _gf_matrix_to_column(
            xform_cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath(link_paths[link])
            )
        )
        position_errors.append(
            float(np.linalg.norm(matrix[0, :3, 3] - authored[:3, 3]))
        )
        relative = matrix[0, :3, :3].T @ authored[:3, :3]
        orientation_errors.append(
            float(Rotation.from_matrix(relative).magnitude())
        )
    position_max = max(position_errors, default=0.0)
    orientation_max = max(orientation_errors, default=0.0)
    if (
        position_max > ZERO_POSE_POSITION_TOLERANCE_M
        or orientation_max > ZERO_POSE_ORIENTATION_TOLERANCE_RAD
    ):
        raise ValueError(
            f"{source}: USD joint FK does not close against authored zero pose; "
            f"position={position_max:.6g} m, orientation={orientation_max:.6g} rad"
        )
    return G1CollisionModel(
        source_usd=source,
        source_sha256=placeholder.source_sha256,
        root_link=roots[0],
        joints=ordered_joints,
        surface_points_by_link=surface_by_link,
        collider_rows=tuple(collider_rows),
        zero_pose_position_error_max_m=position_max,
        zero_pose_orientation_error_max_rad=orientation_max,
        source_layer_sha256=source_layer_sha256,
        source_composed_sha256=source_composed_sha256,
    )


def _group_surface_world(
    transforms: Mapping[str, np.ndarray],
    model: G1CollisionModel,
    group_name: str,
) -> np.ndarray:
    rows = []
    for link in G1_CONTACT_GROUP_LINKS[group_name]:
        points = model.surface_points_by_link.get(link)
        transform = transforms.get(link)
        if points is None or transform is None:
            continue
        rows.append(
            np.einsum(
                "tij,pj->tpi", transform[:, :3, :3], points
            )
            + transform[:, None, :3, 3]
        )
    if not rows:
        raise ValueError(f"{group_name}: no transformed collision surface")
    return np.concatenate(rows, axis=1).astype(np.float32)


def _shortlist_body_surface_points(
    points_local: np.ndarray,
    group_slices: Sequence[slice],
    *,
    object_surface_tree: cKDTree | None,
    candidate_body_points_per_group: int,
) -> tuple[np.ndarray, list[slice]]:
    """Shortlist body points per frame/group using dense object samples."""
    points = np.asarray(points_local, dtype=np.float32)
    if candidate_body_points_per_group <= 0:
        return points, list(group_slices)
    if object_surface_tree is None:
        raise ValueError("candidate filtering requires an object surface tree")

    approximate_distance = object_surface_tree.query(
        points.reshape(-1, 3), k=1, workers=_kd_workers()
    )[0].reshape(points.shape[0], points.shape[1])
    selected_rows = []
    selected_slices = []
    selected_offset = 0
    for point_slice in group_slices:
        width = int(point_slice.stop - point_slice.start)
        if width <= 0:
            raise ValueError("body contact group has no surface points")
        count = min(int(candidate_body_points_per_group), width)
        local_index = np.argpartition(
            approximate_distance[:, point_slice],
            kth=count - 1,
            axis=1,
        )[:, :count]
        source_index = local_index + int(point_slice.start)
        selected_rows.append(
            np.take_along_axis(
                points,
                source_index[..., None],
                axis=1,
            )
        )
        selected_slices.append(
            slice(selected_offset, selected_offset + count)
        )
        selected_offset += count
    return np.concatenate(selected_rows, axis=1), selected_slices


def canonical_contact_from_g1_reference(
    *,
    robot: Mapping[str, Any],
    object_position: np.ndarray,
    object_quaternion_wxyz: np.ndarray,
    object_vertices_local: np.ndarray,
    object_faces: np.ndarray,
    object_surface_samples_local: np.ndarray | None = None,
    object_usd: Path,
    model: G1CollisionModel,
    object_fps: float,
    required_threshold_m: float = DEFAULT_CONTACT_REQUIRED_THRESHOLD_M,
    forbidden_threshold_m: float = DEFAULT_CONTACT_FORBIDDEN_THRESHOLD_M,
    minimum_required_frames: int,
    close_gap_frames: int,
    transition_neutral_frames: int,
    frame_batch_size: int = 8,
    candidate_body_points_per_group: int = 64,
    hand_primitive_mode: str = "discrete",
    ground_height_m: float = 0.0,
    static_context_boxes: Sequence[StaticBox] | None = None,
    reuse_proof: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build canonical active-object/ground contact from training geometry.

    Temporal frame counts must be realized from the shared duration contract
    at ``object_fps`` before this geometry routine is called.
    """
    object_position = np.asarray(object_position, dtype=np.float32)
    object_quaternion = np.asarray(object_quaternion_wxyz, dtype=np.float32)
    vertices = np.asarray(object_vertices_local, dtype=np.float32)
    faces = np.asarray(object_faces, dtype=np.int64)
    surface_samples = (
        np.asarray(object_surface_samples_local, dtype=np.float32)
        if object_surface_samples_local is not None
        else None
    )
    frames = len(object_position)
    if (
        object_position.shape != (frames, 3)
        or object_quaternion.shape != (frames, 4)
        or not np.isfinite(object_position).all()
        or not np.isfinite(object_quaternion).all()
    ):
        raise ValueError("invalid active-object trajectory")
    if frame_batch_size <= 0:
        raise ValueError("frame_batch_size must be positive")
    if candidate_body_points_per_group < 0:
        raise ValueError(
            "candidate_body_points_per_group must be non-negative"
        )
    if candidate_body_points_per_group > 0:
        if (
            surface_samples is None
            or surface_samples.ndim != 2
            or surface_samples.shape[1] != 3
            or not len(surface_samples)
            or not np.isfinite(surface_samples).all()
        ):
            raise ValueError(
                "candidate filtering requires finite object surface samples"
            )
        surface_tree = cKDTree(surface_samples)
    else:
        surface_tree = None
    trajectory = reference_trajectory(
        robot,
        target_frames=frames,
        target_fps=float(object_fps),
        hand_primitive_mode=hand_primitive_mode,
    )
    transforms = forward_kinematics(model, trajectory)
    distance = np.full(
        canonical_contact_shape(frames), np.inf, dtype=np.float32
    )
    valid = np.zeros(distance.shape, dtype=bool)
    active_axis = CONTACT_ENTITY_CLASSES.index("active_entity")
    static_axis = CONTACT_ENTITY_CLASSES.index("static_context")
    ground_axis = CONTACT_ENTITY_CLASSES.index("ground")
    static_boxes = tuple(static_context_boxes or ())
    point_counts = {}
    group_points: list[np.ndarray] = []
    group_slices: list[slice] = []
    point_offset = 0
    for group_index, group_name in enumerate(CONTACT_BODY_GROUPS):
        points_world = _group_surface_world(transforms, model, group_name)
        point_counts[group_name] = int(points_world.shape[1])
        group_points.append(points_world)
        group_slices.append(
            slice(point_offset, point_offset + points_world.shape[1])
        )
        point_offset += points_world.shape[1]
        valid[:, active_axis, group_index] = True
        if static_boxes:
            static_distance = np.stack(
                [
                    points_to_static_box_contact_distance(points_world, box)
                    for box in static_boxes
                ],
                axis=0,
            )
            distance[:, static_axis, group_index] = np.min(
                static_distance,
                axis=(0, 2),
            )
            valid[:, static_axis, group_index] = True
        distance[:, ground_axis, group_index] = np.min(
            np.abs(points_world[..., 2] - float(ground_height_m)), axis=1
        )
        valid[:, ground_axis, group_index] = True

    all_points_world = np.concatenate(group_points, axis=1)
    for start in range(0, frames, int(frame_batch_size)):
        end = min(frames, start + int(frame_batch_size))
        points_local = world_points_to_object_local(
            all_points_world[start:end],
            object_position[start:end],
            object_quaternion[start:end],
        )
        query_points, query_group_slices = _shortlist_body_surface_points(
            points_local,
            group_slices,
            object_surface_tree=surface_tree,
            candidate_body_points_per_group=(
                candidate_body_points_per_group
            ),
        )
        result = closest_points_to_mesh(
            query_points.reshape(-1, 3),
            vertices,
            faces,
            compute_signed_distance=False,
        )
        point_distance = np.asarray(result[0]).reshape(
            end - start, query_points.shape[1]
        )
        for group_index, point_slice in enumerate(query_group_slices):
            distance[start:end, active_axis, group_index] = np.min(
                point_distance[:, point_slice], axis=1
            )

    state = discretize_contact_distances(
        distance,
        geometry_valid=valid,
        required_threshold_m=float(required_threshold_m),
        forbidden_threshold_m=float(forbidden_threshold_m),
        minimum_required_frames=int(minimum_required_frames),
        close_gap_frames=int(close_gap_frames),
        transition_neutral_frames=int(transition_neutral_frames),
    )
    provenance = {
        "schema": "canonical_discrete_contact_v1",
        "generator": "gear_sonic.rp2.g1_collision_geometry",
        "source_reference": {
            "kind": "g1_reference_forward_kinematics",
            "source_fps": float(trajectory.source_fps),
            "target_fps": float(trajectory.target_fps),
            "hand_source": trajectory.hand_source,
            "joint_order": list(model.revolute_joint_names),
        },
        "body_geometry": {
            "kind": "training_g1_collision_surface_samples",
            "usd": str(model.source_usd),
            "usd_sha256": model.source_sha256,
            "composed_usd_sha256": model.source_composed_sha256,
            "layer_sha256": dict(model.source_layer_sha256),
            "group_links": {
                name: list(G1_CONTACT_GROUP_LINKS[name])
                for name in CONTACT_BODY_GROUPS
            },
            "group_surface_point_counts": point_counts,
            "zero_pose_position_error_max_m": (
                model.zero_pose_position_error_max_m
            ),
            "zero_pose_orientation_error_max_rad": (
                model.zero_pose_orientation_error_max_rad
            ),
            "colliders": [dict(row) for row in model.collider_rows],
        },
        "entity_geometry": {
            "active_entity": {
                "kind": "paired_object_triangle_surface",
                "usd": str(object_usd.resolve()),
                "usd_sha256": _sha256(object_usd.resolve()),
                "vertices": int(len(vertices)),
                "triangles": int(len(faces)),
            },
            "static_context": {
                "kind": (
                    "canonical_static_boxes_v1"
                    if static_boxes
                    else "unavailable"
                ),
                "valid": bool(static_boxes),
                "label_policy": (
                    "exact_obb_outside_distance_penetration_is_contact"
                    if static_boxes
                    else "neutral"
                ),
                "boxes": [box.as_dict() for box in static_boxes],
            },
            "ground": {
                "kind": "infinite_z_plane",
                "height_m": float(ground_height_m),
            },
        },
        "distance_metric": "minimum_surface_distance_m",
        "distance_query": (
            "actual_training_collision_surface_samples_to_certified_exact_"
            "object_triangle_surface; exact_static_obb_outside_distance; "
            "absolute_distance_to_ground_plane"
        ),
        "body_surface_candidate_filter": {
            "mode": (
                "dense_object_surface_shortlist_then_certified_triangle"
                if surface_tree is not None
                else "all_body_surface_points_certified_triangle"
            ),
            "candidate_body_points_per_group": int(
                candidate_body_points_per_group
            ),
            "object_surface_samples": (
                int(len(surface_samples))
                if surface_samples is not None
                else 0
            ),
        },
        "required_threshold_m": float(required_threshold_m),
        "forbidden_threshold_m": float(forbidden_threshold_m),
        "minimum_required_frames": int(minimum_required_frames),
        "close_gap_frames": int(close_gap_frames),
        "transition_neutral_frames": int(transition_neutral_frames),
        "reuse_proof": dict(reuse_proof or {}),
    }
    fields = {
        "reference_contact_state": state.astype(np.int8, copy=False),
        "reference_contact_distance_m": distance,
        "reference_contact_geometry_valid": valid,
        "reference_contact_entity_classes": list(CONTACT_ENTITY_CLASSES),
        "reference_contact_body_groups": list(CONTACT_BODY_GROUPS),
        "reference_contact_provenance": provenance,
    }
    audit = {
        "frames": frames,
        "hand_source": trajectory.hand_source,
        "body_surface_points": int(sum(point_counts.values())),
        "group_surface_point_counts": point_counts,
        "valid": int(np.count_nonzero(valid)),
        "required": int(np.count_nonzero(state == 1)),
        "neutral": int(np.count_nonzero(state == 0)),
        "forbidden": int(np.count_nonzero(state == -1)),
    }
    total_labels = int(state.size)
    audit["state_fractions"] = {
        name: (
            float(audit[name]) / float(total_labels)
            if total_labels
            else 0.0
        )
        for name in ("required", "neutral", "forbidden")
    }
    audit["valid_fraction"] = (
        float(audit["valid"]) / float(total_labels)
        if total_labels
        else 0.0
    )
    return fields, audit
