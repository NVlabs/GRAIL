"""Pure NumPy helpers for source-SOMA relational RP2 references."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


DEFAULT_SOURCE_RELATION_MAPPING: tuple[tuple[str, str], ...] = (
    ("pelvis", "Hips"),
    ("left_hip_roll_link", "LeftLeg"),
    ("left_knee_link", "LeftShin"),
    ("left_ankle_roll_link", "LeftFoot"),
    ("right_hip_roll_link", "RightLeg"),
    ("right_knee_link", "RightShin"),
    ("right_ankle_roll_link", "RightFoot"),
    ("torso_link", "Chest"),
    ("left_shoulder_roll_link", "LeftArm"),
    ("left_elbow_link", "LeftForeArm"),
    ("left_wrist_yaw_link", "LeftHand"),
    ("right_shoulder_roll_link", "RightArm"),
    ("right_elbow_link", "RightForeArm"),
    ("right_wrist_yaw_link", "RightHand"),
)


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized XYZW quaternions to rotation matrices."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4:
        raise ValueError(f"Expected XYZW quaternion(s), got {value.shape}")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("Landmark offset quaternion contains a zero norm")
    x, y, z, w = np.moveaxis(value / norm, -1, 0)
    matrix = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def load_source_relation_landmark_config(
    path: str | Path,
    *,
    expected_scene_scale: float,
    scale_tolerance: float = 1.0e-8,
    mapping: Sequence[tuple[str, str]] = DEFAULT_SOURCE_RELATION_MAPPING,
) -> dict[str, object]:
    """Load and fail closed on the calibrated NOVA-to-G1 landmark contract."""
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pairs = tuple(
        (str(item["robot_body"]), str(item["soma_joint"]))
        for item in config.get("landmarks", [])
    )
    if pairs != tuple(mapping):
        raise ValueError(
            f"Landmark configuration {config_path} does not exactly match "
            "the runtime source-relation mapping"
        )
    scene_scale = float(config["scene_scale"])
    if abs(scene_scale - float(expected_scene_scale)) > float(scale_tolerance):
        raise ValueError(
            f"Landmark configuration scene scale {scene_scale} does not match "
            f"the motion scene scale {expected_scene_scale}"
        )
    offset_scale = float(config.get("offset_translation_scale", 1.0))
    if offset_scale != 1.0:
        raise ValueError(
            "NOVA-to-G1 landmark translation offsets are target-frame "
            f"calibrations and must remain unscaled; got {offset_scale}"
        )
    config["path"] = str(config_path)
    return config


def apply_source_relation_landmark_offsets(
    soma_joint_positions: np.ndarray,
    joint_names: Sequence[str],
    soma_joint_rotations: np.ndarray,
    landmark_config: Mapping[str, object],
    mapping: Sequence[tuple[str, str]] = DEFAULT_SOURCE_RELATION_MAPPING,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the retargeter's static landmark offsets to scaled SOMA pivots.

    ``soma_joint_positions`` are already in the scaled scene. The translation
    offsets remain in their calibrated G1 target-frame metric units, matching
    the NOVA-to-G1 scaler. Each offset is first rotated by its configured
    target-frame orientation and then by the source joint's world rotation.
    """
    positions = np.asarray(soma_joint_positions, dtype=np.float32)
    rotations = np.asarray(soma_joint_rotations, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected SOMA positions [T, J, 3], got {positions.shape}")
    if rotations.shape != positions.shape[:2] + (3, 3):
        raise ValueError(
            "SOMA position/rotation shape mismatch: "
            f"positions={positions.shape}, rotations={rotations.shape}"
        )

    items = list(landmark_config.get("landmarks", []))
    pairs = tuple(
        (str(item["robot_body"]), str(item["soma_joint"])) for item in items
    )
    if pairs != tuple(mapping):
        raise ValueError("Landmark records do not match the canonical mapping order")

    joint_index = {str(name): index for index, name in enumerate(joint_names)}
    missing = [source_name for _, source_name in mapping if source_name not in joint_index]
    if missing:
        raise KeyError(f"Missing calibrated SOMA landmarks: {missing}")

    calibrated = positions.copy()
    records: list[dict[str, object]] = []
    for item in items:
        source_name = str(item["soma_joint"])
        index = joint_index[source_name]
        translation = np.asarray(item["translation_offset_m"], dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError(
                f"Landmark {source_name} has invalid translation {translation.shape}"
            )
        offset_rotation = _quaternion_xyzw_to_matrix(
            np.asarray(item["rotation_offset_xyzw"], dtype=np.float64)
        )
        target_frame_offset = offset_rotation @ translation
        delta = np.einsum(
            "tij,j->ti",
            rotations[:, index],
            target_frame_offset,
        )
        calibrated[:, index] += delta.astype(np.float32)
        records.append(
            {
                "robot_body": str(item["robot_body"]),
                "soma_joint": source_name,
                "offset_norm_m": float(np.linalg.norm(translation)),
            }
        )

    return calibrated, {
        "applied": True,
        "config": str(landmark_config.get("path", "")),
        "offset_translation_scale": 1.0,
        "landmarks": records,
    }


def select_full_scene_source_object_indices(
    num_source_objects: int,
    max_num_objects: int,
    *,
    target_object_index: int | None = None,
    declared_source_object_count: int | None = None,
) -> tuple[int, ...]:
    """Select source channels that correspond to simulated object slots.

    Full-scene RP2 labels may accompany a composed clip that simulates only
    the active object. Such clips explicitly reorder that object to index zero
    and retain the original scene object count as metadata. Keep arbitrary
    multi-object overflows as errors so they cannot be truncated silently.
    """
    num_source_objects = int(num_source_objects)
    max_num_objects = int(max_num_objects)
    if num_source_objects < 1 or max_num_objects < 1:
        raise ValueError(
            "Source and simulated object counts must be positive: "
            f"source={num_source_objects}, max={max_num_objects}"
        )
    if num_source_objects <= max_num_objects:
        return tuple(range(num_source_objects))
    if declared_source_object_count is None:
        raise ValueError(
            "Full-scene source tensor exceeds simulated object slots without "
            "declared source object count metadata"
        )
    if int(declared_source_object_count) != num_source_objects:
        raise ValueError(
            "Full-scene source object count does not match metadata: "
            f"tensor={num_source_objects}, declared={declared_source_object_count}"
        )
    if max_num_objects != 1 or target_object_index is None or int(target_object_index) != 0:
        raise ValueError(
            "Full-scene source tensor can only collapse for a one-object composed "
            "clip whose active object is explicitly stored at target index 0"
        )
    return (0,)


def _normalized_heading(heading_xy: np.ndarray) -> np.ndarray:
    heading = np.asarray(heading_xy, dtype=np.float32)
    if heading.ndim != 2 or heading.shape[1] != 2:
        raise ValueError(f"Expected heading_xy [T, 2], got {heading.shape}")
    norms = np.linalg.norm(heading, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-6):
        raise ValueError("Source heading contains a zero-length vector")
    return heading / norms


def world_vectors_to_heading_frame(vectors: np.ndarray, heading_xy: np.ndarray) -> np.ndarray:
    """Rotate Z-up world vectors into a yaw-only human heading frame."""
    vectors = np.asarray(vectors, dtype=np.float32)
    heading = _normalized_heading(heading_xy)
    if vectors.shape[0] != heading.shape[0] or vectors.shape[-1] != 3:
        raise ValueError(
            f"Vector/heading shape mismatch: vectors={vectors.shape}, heading={heading.shape}"
        )
    cos_yaw = heading[:, 0]
    sin_yaw = heading[:, 1]
    while cos_yaw.ndim < vectors.ndim - 1:
        cos_yaw = cos_yaw[..., None]
        sin_yaw = sin_yaw[..., None]
    x_world = vectors[..., 0]
    y_world = vectors[..., 1]
    output = np.empty_like(vectors, dtype=np.float32)
    output[..., 0] = cos_yaw * x_world + sin_yaw * y_world
    output[..., 1] = -sin_yaw * x_world + cos_yaw * y_world
    output[..., 2] = vectors[..., 2]
    return output


def reexpress_heading_frame_vectors(
    vectors: np.ndarray,
    from_heading_xy: np.ndarray,
    to_heading_xy: np.ndarray,
) -> np.ndarray:
    """Re-express yaw-local vectors from one heading frame in another."""
    vectors = np.asarray(vectors, dtype=np.float32)
    from_heading = _normalized_heading(from_heading_xy)
    to_heading = _normalized_heading(to_heading_xy)
    if vectors.shape[0] != from_heading.shape[0] or vectors.shape[0] != to_heading.shape[0]:
        raise ValueError(
            "Vector/heading shape mismatch: "
            f"vectors={vectors.shape}, from={from_heading.shape}, to={to_heading.shape}"
        )

    from_cos = from_heading[:, 0]
    from_sin = from_heading[:, 1]
    to_cos = to_heading[:, 0]
    to_sin = to_heading[:, 1]
    while from_cos.ndim < vectors.ndim - 1:
        from_cos = from_cos[..., None]
        from_sin = from_sin[..., None]
        to_cos = to_cos[..., None]
        to_sin = to_sin[..., None]

    x_world = from_cos * vectors[..., 0] - from_sin * vectors[..., 1]
    y_world = from_sin * vectors[..., 0] + from_cos * vectors[..., 1]
    output = np.empty_like(vectors, dtype=np.float32)
    output[..., 0] = to_cos * x_world + to_sin * y_world
    output[..., 1] = -to_sin * x_world + to_cos * y_world
    output[..., 2] = vectors[..., 2]
    return output


def build_human_centric_relations(
    soma_joint_positions: np.ndarray,
    joint_names: Sequence[str],
    object_positions: np.ndarray,
    heading_xy: np.ndarray,
    mapping: Sequence[tuple[str, str]] = DEFAULT_SOURCE_RELATION_MAPPING,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense link-to-object vectors in the source human heading frame.

    The output ordering follows ``mapping`` and therefore matches the configured
    G1 body ordering. The vector convention is object position minus landmark
    position, i.e. each vector points from the mapped body landmark to the object.
    """
    joints = np.asarray(soma_joint_positions, dtype=np.float32)
    objects = np.asarray(object_positions, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected SOMA joints [T, J, 3], got {joints.shape}")
    if objects.ndim == 3:
        if objects.shape[1] != 1:
            raise ValueError(
                "Source-relation training currently requires exactly one active object, "
                f"got {objects.shape[1]}"
            )
        objects = objects[:, 0]
    if objects.shape != (joints.shape[0], 3):
        raise ValueError(f"Object/joint shape mismatch: objects={objects.shape}, joints={joints.shape}")

    joint_index = {str(name): idx for idx, name in enumerate(joint_names)}
    missing = [source_name for _, source_name in mapping if source_name not in joint_index]
    if missing:
        raise KeyError(f"Missing mapped SOMA joints: {missing}")
    selected = np.stack([joints[:, joint_index[source_name]] for _, source_name in mapping], axis=1)
    world_vectors = objects[:, None, :] - selected
    relations = world_vectors_to_heading_frame(world_vectors, heading_xy)
    valid = np.isfinite(relations).all(axis=-1)
    relations = np.where(valid[..., None], relations, 0.0).astype(np.float32)
    return relations, valid.astype(np.float32)


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or output.size == 0:
        return output
    true_indices = np.flatnonzero(output)
    for left, right in zip(true_indices[:-1], true_indices[1:]):
        if 1 < right - left <= max_gap + 1:
            output[left : right + 1] = True
    return output


def smooth_contact_probability(
    mask: np.ndarray,
    *,
    ramp_frames: int = 10,
    max_gap_frames: int = 5,
) -> np.ndarray:
    """Convert a binary source-contact mask into a continuous symmetric ramp."""
    contact = _fill_short_false_gaps(np.asarray(mask, dtype=bool).reshape(-1), max_gap_frames)
    if ramp_frames <= 0:
        return contact.astype(np.float32)
    probability = contact.astype(np.float32)
    true_indices = np.flatnonzero(contact)
    if true_indices.size == 0:
        return probability
    frame_ids = np.arange(contact.size)
    distance = np.min(np.abs(frame_ids[:, None] - true_indices[None, :]), axis=1)
    outside = ~contact
    probability[outside] = np.clip(1.0 - distance[outside] / float(ramp_frames), 0.0, 1.0)
    return probability.astype(np.float32)


def heading_xy_to_yaw(heading_xy: np.ndarray) -> np.ndarray:
    """Return yaw angles for normalized XY heading vectors."""
    heading = _normalized_heading(heading_xy)
    return np.arctan2(heading[:, 1], heading[:, 0]).astype(np.float32)


def build_source_clock_motion(
    root_positions: np.ndarray,
    heading_xy: np.ndarray,
    *,
    fps: float,
    contact_probability: np.ndarray | None = None,
    num_dof: int = 29,
) -> dict[str, np.ndarray | float | bool | str]:
    """Create a neutral robot-shaped timing stream with no retargeted G1 pose."""
    roots = np.asarray(root_positions, dtype=np.float32)
    if roots.ndim != 2 or roots.shape[1] != 3:
        raise ValueError(f"Expected root positions [T, 3], got {roots.shape}")
    yaw = heading_xy_to_yaw(heading_xy)
    if yaw.shape[0] != roots.shape[0]:
        raise ValueError("Root position and heading lengths differ")
    pose_aa = np.zeros((roots.shape[0], num_dof + 1, 3), dtype=np.float32)
    pose_aa[:, 0, 2] = yaw
    root_rot_xyzw = np.zeros((roots.shape[0], 4), dtype=np.float32)
    root_rot_xyzw[:, 2] = np.sin(0.5 * yaw)
    root_rot_xyzw[:, 3] = np.cos(0.5 * yaw)

    hand_actions = np.full((roots.shape[0], 2), -1.0, dtype=np.float32)
    if contact_probability is not None:
        contact = np.asarray(contact_probability, dtype=np.float32)
        if contact.shape != hand_actions.shape:
            raise ValueError(
                f"Expected bilateral contact probability {hand_actions.shape}, got {contact.shape}"
            )
        hand_actions[contact >= 0.5] = 1.0
    return {
        "root_trans_offset": roots.copy(),
        "pose_aa": pose_aa,
        "dof": np.zeros((roots.shape[0], num_dof), dtype=np.float32),
        "root_rot": root_rot_xyzw,
        "smpl_joints": np.zeros((roots.shape[0], 24, 3), dtype=np.float32),
        "fps": float(fps),
        "hand_action_left": hand_actions[:, 0],
        "hand_action_right": hand_actions[:, 1],
        "source_clock_only": True,
        "reference_semantics": "uniform_soma_object_relation_only",
    }


def require_uniform_scale(
    metadata: Mapping[str, object],
    *,
    expected_scale: float,
    tolerance: float = 1.0e-6,
) -> None:
    """Validate source object geometry metadata for pose-and-shape scaling."""
    actual = metadata.get("object_geometry_scale")
    if actual is None:
        raise ValueError("Motion metadata is missing object_geometry_scale")
    if abs(float(actual) - float(expected_scale)) > float(tolerance):
        raise ValueError(
            f"Object geometry scale {actual} does not match required scene scale {expected_scale}"
        )
    mode = str(metadata.get("object_geometry_scale_mode", ""))
    if "uniform" not in mode:
        raise ValueError(f"Object geometry was not uniformly scaled: mode={mode!r}")
