"""Canonical payload ABI for joint multi-source active-entity training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from gear_sonic.rp2.discrete_contact import (
    CONTACT_BODY_GROUPS,
    CONTACT_ENTITY_CLASSES,
    DEFAULT_CONTACT_FORBIDDEN_THRESHOLD_M,
    DEFAULT_CONTACT_REQUIRED_THRESHOLD_M,
    validate_temporal_contact_contract,
    validate_reference_contact_state,
)
from gear_sonic.rp2.source_relation import DEFAULT_SOURCE_RELATION_MAPPING


JOINT_PAYLOAD_CONTRACT_NAME = "single_active_entity_training_payload_v1"
JOINT_PAYLOAD_SCHEMA_VERSION = 1
CANONICAL_SURFACE_RELATION_SEMANTICS = (
    "nearest_triangle_surface_displacement_heading_frame_watertight_"
    "signed_distance_positive_outside_open_mesh_unsigned_source_"
    "penetration_projected_v3"
)
CANONICAL_SURFACE_QUERY_METHODS = frozenset(
    {
        "nearest_triangle_with_oriented_surface_sign",
        "certified_nearest_triangle_with_oriented_surface_sign",
    }
)
HIPHI_SOURCE_HUMAN_SURFACE_RELATION_SEMANTICS = (
    "scaled_source_human_landmark_to_exact_object_triangle_surface_"
    "source_human_heading_frame_v1"
)
HIPHI_SOURCE_HUMAN_SURFACE_AUTHORITY = (
    "hiphi_source_human_reference_supervision"
)
HIPHI_SOURCE_HUMAN_REFERENCE = "hiphi_source_human_bvh_scaled_landmarks"
HIPHI_SOURCE_OBJECT_REFERENCE = "hiphi_exact_source_object_triangle_surface"
HIPHI_SOURCE_HUMAN_SURFACE_QUERY = (
    "exact_nearest_triangle_source_object_surface"
)
CANONICAL_CONTACT_SCHEMA = "canonical_discrete_contact_v1"
CANONICAL_CONTACT_DISTANCE_METRIC = "minimum_surface_distance_m"
CANONICAL_CONTACT_REQUIRED_THRESHOLD_M = DEFAULT_CONTACT_REQUIRED_THRESHOLD_M
CANONICAL_CONTACT_FORBIDDEN_THRESHOLD_M = DEFAULT_CONTACT_FORBIDDEN_THRESHOLD_M
LEGACY_FULL_SCENE_SOURCE_PREFIX = "rp2_full_scene_source_"

RELATION_ROBOT_BODY_NAMES = tuple(
    robot_name for robot_name, _ in DEFAULT_SOURCE_RELATION_MAPPING
)
RELATION_SOURCE_JOINT_NAMES = tuple(
    source_name for _, source_name in DEFAULT_SOURCE_RELATION_MAPPING
)
NUM_RELATION_LINKS = len(DEFAULT_SOURCE_RELATION_MAPPING)
MINIMUM_REFERENCE_BODY_DOFS = 29


def joint_payload_contract_document() -> dict[str, Any]:
    """Return the versioned ABI recorded in every materialized roster."""
    return {
        "name": JOINT_PAYLOAD_CONTRACT_NAME,
        "schema_version": JOINT_PAYLOAD_SCHEMA_VERSION,
        "active_entity_count": 1,
        "minimum_reference_body_dofs": MINIMUM_REFERENCE_BODY_DOFS,
        "active_entity_tracking_valid": "required_for_every_frame",
        "relation_link_count": NUM_RELATION_LINKS,
        "relation_robot_body_names": list(RELATION_ROBOT_BODY_NAMES),
        "relation_source_joint_names": list(RELATION_SOURCE_JOINT_NAMES),
        "surface_relation_semantics": CANONICAL_SURFACE_RELATION_SEMANTICS,
        "allowed_surface_relation_semantics": [
            CANONICAL_SURFACE_RELATION_SEMANTICS,
            HIPHI_SOURCE_HUMAN_SURFACE_RELATION_SEMANTICS,
        ],
        "hiphi_source_human_surface_contract": {
            "authority": HIPHI_SOURCE_HUMAN_SURFACE_AUTHORITY,
            "human_reference": HIPHI_SOURCE_HUMAN_REFERENCE,
            "object_reference": HIPHI_SOURCE_OBJECT_REFERENCE,
            "surface_query": HIPHI_SOURCE_HUMAN_SURFACE_QUERY,
            "g1_fk_used": False,
            "g1_robot_files_read": False,
            "slot_zero_prior_used": False,
        },
        "contact_schema": CANONICAL_CONTACT_SCHEMA,
        "contact_required_threshold_m": (
            CANONICAL_CONTACT_REQUIRED_THRESHOLD_M
        ),
        "contact_forbidden_threshold_m": (
            CANONICAL_CONTACT_FORBIDDEN_THRESHOLD_M
        ),
        "contact_entity_classes": list(CONTACT_ENTITY_CLASSES),
        "contact_body_groups": list(CONTACT_BODY_GROUPS),
        "legacy_full_scene_channels": "forbidden",
        "missing_policy_channels": "error",
        "invalid_contact_distance": (
            "non_finite_allowed_only_where_geometry_valid_is_false"
        ),
    }


def _positive_fps(entry: Mapping[str, Any], *, label: str) -> float:
    if "fps" not in entry:
        raise KeyError(f"{label}: missing fps")
    fps = float(entry["fps"])
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{label}: fps must be positive and finite")
    return fps


def _require_fields(
    entry: Mapping[str, Any],
    fields: Sequence[str],
    *,
    label: str,
) -> None:
    missing = [field for field in fields if field not in entry]
    if missing:
        raise KeyError(f"{label}: missing required fields {missing}")


def _numeric_array(
    entry: Mapping[str, Any],
    field: str,
    *,
    label: str,
    shape: tuple[int | None, ...],
    finite: bool = True,
) -> np.ndarray:
    value = np.asarray(entry[field])
    if value.ndim != len(shape):
        raise ValueError(
            f"{label}: {field} must have rank {len(shape)}, got {value.shape}"
        )
    for axis, expected in enumerate(shape):
        if expected is not None and value.shape[axis] != expected:
            raise ValueError(
                f"{label}: {field} must have shape {shape}, got {value.shape}"
            )
    if value.dtype.kind not in "biufc":
        raise TypeError(f"{label}: {field} must be numeric, got {value.dtype}")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{label}: {field} contains non-finite values")
    return value


def _binary_array(
    entry: Mapping[str, Any],
    field: str,
    *,
    label: str,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    value = _numeric_array(
        entry,
        field,
        label=label,
        shape=shape,
    )
    unique = set(np.unique(value).tolist())
    if not unique.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(
            f"{label}: {field} must be binary, got {sorted(unique)[:8]}"
        )
    return value


def _unit_quaternions(
    entry: Mapping[str, Any],
    field: str,
    *,
    label: str,
    shape: tuple[int | None, ...],
    atol: float = 1.0e-3,
) -> np.ndarray:
    value = _numeric_array(
        entry,
        field,
        label=label,
        shape=shape,
    )
    norms = np.linalg.norm(value, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=atol):
        raise ValueError(
            f"{label}: {field} must contain unit quaternions; "
            f"norm range=[{float(norms.min())}, {float(norms.max())}]"
        )
    return value


def _exact_names(
    entry: Mapping[str, Any],
    field: str,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    actual = tuple(str(value) for value in entry[field])
    expected_tuple = tuple(str(value) for value in expected)
    if actual != expected_tuple:
        raise ValueError(
            f"{label}: {field} order differs from the canonical ABI"
        )


def _validate_surface_relation_contract(
    objects: Mapping[str, Any],
    *,
    label: str,
) -> None:
    semantics = str(objects["rp2_source_surface_relation_semantics"]).strip()
    provenance = objects["rp2_source_surface_relation_provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError(
            f"{label}: surface-relation provenance must be a mapping"
        )

    if semantics == CANONICAL_SURFACE_RELATION_SEMANTICS:
        for field in ("human_reference", "object_reference", "surface_query"):
            if not str(provenance.get(field, "")).strip():
                raise ValueError(
                    f"{label}: surface-relation provenance is missing {field!r}"
                )
        surface_query = str(provenance["surface_query"]).strip()
        if surface_query not in CANONICAL_SURFACE_QUERY_METHODS:
            raise ValueError(
                f"{label}: unsupported surface query {surface_query!r}"
            )
        return

    if semantics == HIPHI_SOURCE_HUMAN_SURFACE_RELATION_SEMANTICS:
        for field, expected in (
            ("authority", HIPHI_SOURCE_HUMAN_SURFACE_AUTHORITY),
            ("human_reference", HIPHI_SOURCE_HUMAN_REFERENCE),
            ("object_reference", HIPHI_SOURCE_OBJECT_REFERENCE),
            ("surface_query", HIPHI_SOURCE_HUMAN_SURFACE_QUERY),
        ):
            actual = provenance.get(field)
            if actual != expected:
                raise ValueError(
                    f"{label}: HiPHI source-human surface provenance {field} "
                    f"must be {expected!r}, got {actual!r}"
                )
        if not str(provenance.get("source_reference_record", "")).strip():
            raise ValueError(
                f"{label}: HiPHI source-human surface provenance is missing "
                "'source_reference_record'"
            )
        for field in (
            "g1_fk_used",
            "g1_robot_files_read",
            "slot_zero_prior_used",
        ):
            if provenance.get(field) is not False:
                raise ValueError(
                    f"{label}: HiPHI source-human surface provenance {field} "
                    "must be exactly false"
                )
        return

    raise ValueError(
        f"{label}: surface-relation semantics {semantics!r} do not match "
        "an admitted exact contract"
    )


def _validate_reference_contact_contract(
    provenance: Any,
    *,
    fps: float,
    label: str,
) -> None:
    if not isinstance(provenance, Mapping):
        raise TypeError(
            f"{label}: reference_contact_provenance must be a mapping"
        )
    if provenance.get("schema") != CANONICAL_CONTACT_SCHEMA:
        raise ValueError(
            f"{label}: reference contact schema must be "
            f"{CANONICAL_CONTACT_SCHEMA!r}"
        )
    if provenance.get("distance_metric") != CANONICAL_CONTACT_DISTANCE_METRIC:
        raise ValueError(
            f"{label}: reference contact distance metric must be "
            f"{CANONICAL_CONTACT_DISTANCE_METRIC!r}"
        )
    for field, expected in (
        (
            "required_threshold_m",
            CANONICAL_CONTACT_REQUIRED_THRESHOLD_M,
        ),
        (
            "forbidden_threshold_m",
            CANONICAL_CONTACT_FORBIDDEN_THRESHOLD_M,
        ),
    ):
        actual = float(provenance.get(field, float("nan")))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"{label}: {field}={actual} does not match {expected}"
            )
    temporal = provenance.get("temporal_contact_contract")
    validate_temporal_contact_contract(temporal, fps=fps)


def _validate_no_legacy_full_scene_channels(
    objects: Mapping[str, Any],
    *,
    label: str,
) -> None:
    legacy_fields = sorted(
        str(field)
        for field in objects
        if str(field).startswith(LEGACY_FULL_SCENE_SOURCE_PREFIX)
    )
    if legacy_fields:
        raise ValueError(
            f"{label}: single-active-entity payload contains legacy "
            f"full-scene channels {legacy_fields}"
        )


def validate_joint_training_payload_pair(
    robot: Mapping[str, Any],
    objects: Mapping[str, Any],
    *,
    label: str,
    expected_source_id: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless a pair implements the complete shared training ABI."""
    if not isinstance(robot, Mapping) or not isinstance(objects, Mapping):
        raise TypeError(f"{label}: robot and object payloads must be mappings")

    robot_label = f"{label}/robot"
    object_label = f"{label}/objects_with_masks"
    _validate_no_legacy_full_scene_channels(objects, label=object_label)
    _require_fields(
        robot,
        (
            "fps",
            "root_trans_offset",
            "pose_aa",
            "dof",
            "root_rot",
            "smpl_joints",
        ),
        label=robot_label,
    )
    robot_fps = _positive_fps(robot, label=robot_label)
    root_trans = _numeric_array(
        robot,
        "root_trans_offset",
        label=robot_label,
        shape=(None, 3),
    )
    num_frames = int(root_trans.shape[0])
    if num_frames < 1:
        raise ValueError(f"{robot_label}: motion must contain at least one frame")
    pose_aa = _numeric_array(
        robot,
        "pose_aa",
        label=robot_label,
        shape=(num_frames, None, 3),
    )
    dof = _numeric_array(
        robot,
        "dof",
        label=robot_label,
        shape=(num_frames, None),
    )
    smpl_joints = _numeric_array(
        robot,
        "smpl_joints",
        label=robot_label,
        shape=(num_frames, None, 3),
    )
    if pose_aa.shape[1] < 2 or smpl_joints.shape[1] < 1:
        raise ValueError(f"{robot_label}: robot joint arrays are empty")
    if dof.shape[1] < MINIMUM_REFERENCE_BODY_DOFS:
        raise ValueError(
            f"{robot_label}: dof must contain at least "
            f"{MINIMUM_REFERENCE_BODY_DOFS} body joints, got {dof.shape[1]}"
        )
    _unit_quaternions(
        robot,
        "root_rot",
        label=robot_label,
        shape=(num_frames, 4),
    )

    _require_fields(
        objects,
        (
            "fps",
            "root_pos",
            "root_quat",
            "rp2_source_root_heading_xy",
            "rp2_source_root_pos",
            "rp2_source_relation_robot_body_names",
            "rp2_source_relation_soma_joint_names",
            "rp2_source_relation_vector",
            "rp2_source_relation_mask",
            "rp2_source_surface_relation_vector",
            "rp2_source_surface_relation_mask",
            "rp2_source_surface_point_local",
            "rp2_source_surface_signed_distance",
            "rp2_source_surface_penetration_mask",
            "rp2_source_surface_relation_semantics",
            "rp2_source_surface_relation_provenance",
            "rp2_source_scene_scale",
            "reference_contact_state",
            "reference_contact_distance_m",
            "reference_contact_geometry_valid",
            "reference_contact_entity_classes",
            "reference_contact_body_groups",
            "reference_contact_provenance",
            "active_articulation_root_pos",
            "active_articulation_root_quat",
            "active_articulation_root_lin_vel",
            "active_articulation_root_ang_vel",
            "active_articulation_joint_pos",
            "active_articulation_joint_vel",
            "active_articulation_joint_valid_mask",
            "active_articulation_root_dynamic_mask",
            "active_entity_tracking_valid",
            "active_articulation",
        ),
        label=object_label,
    )
    object_fps = _positive_fps(objects, label=object_label)
    if not math.isclose(robot_fps, object_fps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"{label}: paired robot/object FPS mismatch "
            f"{robot_fps} != {object_fps}"
        )

    _numeric_array(
        objects,
        "root_pos",
        label=object_label,
        shape=(num_frames, 1, 3),
    )
    _unit_quaternions(
        objects,
        "root_quat",
        label=object_label,
        shape=(num_frames, 1, 4),
    )
    heading = _numeric_array(
        objects,
        "rp2_source_root_heading_xy",
        label=object_label,
        shape=(num_frames, 2),
    )
    heading_norm = np.linalg.norm(heading, axis=-1)
    if not np.allclose(heading_norm, 1.0, rtol=0.0, atol=1.0e-3):
        raise ValueError(
            f"{object_label}: rp2_source_root_heading_xy must be unit length"
        )
    _numeric_array(
        objects,
        "rp2_source_root_pos",
        label=object_label,
        shape=(num_frames, 3),
    )

    _exact_names(
        objects,
        "rp2_source_relation_robot_body_names",
        RELATION_ROBOT_BODY_NAMES,
        label=object_label,
    )
    _exact_names(
        objects,
        "rp2_source_relation_soma_joint_names",
        RELATION_SOURCE_JOINT_NAMES,
        label=object_label,
    )
    for field, tail in (
        ("rp2_source_relation_vector", (NUM_RELATION_LINKS, 3)),
        ("rp2_source_surface_relation_vector", (NUM_RELATION_LINKS, 3)),
        ("rp2_source_surface_point_local", (NUM_RELATION_LINKS, 3)),
        ("rp2_source_surface_signed_distance", (NUM_RELATION_LINKS,)),
    ):
        _numeric_array(
            objects,
            field,
            label=object_label,
            shape=(num_frames, *tail),
        )
    for field in (
        "rp2_source_relation_mask",
        "rp2_source_surface_relation_mask",
        "rp2_source_surface_penetration_mask",
    ):
        _binary_array(
            objects,
            field,
            label=object_label,
            shape=(num_frames, NUM_RELATION_LINKS),
        )
    _validate_surface_relation_contract(objects, label=object_label)
    scene_scale = float(objects["rp2_source_scene_scale"])
    if not math.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError(
            f"{object_label}: rp2_source_scene_scale must be positive"
        )

    state = validate_reference_contact_state(
        np.asarray(objects["reference_contact_state"]),
        num_frames=num_frames,
    )
    expected_contact_shape = (
        num_frames,
        len(CONTACT_ENTITY_CLASSES),
        len(CONTACT_BODY_GROUPS),
    )
    distance = _numeric_array(
        objects,
        "reference_contact_distance_m",
        label=object_label,
        shape=expected_contact_shape,
        finite=False,
    )
    geometry_valid = _binary_array(
        objects,
        "reference_contact_geometry_valid",
        label=object_label,
        shape=expected_contact_shape,
    ).astype(bool)
    if not np.isfinite(distance[geometry_valid]).all():
        raise ValueError(
            f"{object_label}: valid contact distances must be finite"
        )
    if np.any(distance[geometry_valid] < 0.0):
        raise ValueError(
            f"{object_label}: valid contact distances must be non-negative"
        )
    if np.any(state[~geometry_valid] != 0):
        raise ValueError(
            f"{object_label}: invalid contact geometry must be neutral"
        )
    _exact_names(
        objects,
        "reference_contact_entity_classes",
        CONTACT_ENTITY_CLASSES,
        label=object_label,
    )
    _exact_names(
        objects,
        "reference_contact_body_groups",
        CONTACT_BODY_GROUPS,
        label=object_label,
    )
    _validate_reference_contact_contract(
        objects["reference_contact_provenance"],
        fps=object_fps,
        label=object_label,
    )

    _numeric_array(
        objects,
        "active_articulation_root_pos",
        label=object_label,
        shape=(num_frames, 3),
    )
    _unit_quaternions(
        objects,
        "active_articulation_root_quat",
        label=object_label,
        shape=(num_frames, 4),
    )
    for field in (
        "active_articulation_root_lin_vel",
        "active_articulation_root_ang_vel",
    ):
        _numeric_array(
            objects,
            field,
            label=object_label,
            shape=(num_frames, 3),
        )
    joint_pos = _numeric_array(
        objects,
        "active_articulation_joint_pos",
        label=object_label,
        shape=(num_frames, None),
    )
    if joint_pos.shape[1] < 1:
        raise ValueError(f"{object_label}: active articulation has no joint slot")
    joint_shape = tuple(joint_pos.shape)
    _numeric_array(
        objects,
        "active_articulation_joint_vel",
        label=object_label,
        shape=joint_shape,
    )
    _binary_array(
        objects,
        "active_articulation_joint_valid_mask",
        label=object_label,
        shape=joint_shape,
    )
    _binary_array(
        objects,
        "active_articulation_root_dynamic_mask",
        label=object_label,
        shape=(num_frames, 1),
    )
    tracking_valid = _binary_array(
        objects,
        "active_entity_tracking_valid",
        label=object_label,
        shape=(num_frames, 1),
    )
    if not np.all(tracking_valid == 1):
        raise ValueError(
            f"{object_label}: active_entity_tracking_valid must cover "
            "every reference frame"
        )

    articulation = objects["active_articulation"]
    if not isinstance(articulation, Mapping):
        raise TypeError(f"{object_label}: active_articulation must be a mapping")
    payload_source_id = str(objects.get("joint_training_source_id", "")).strip()
    articulation_source_id = str(articulation.get("source_id", "")).strip()
    expected_source = (
        str(expected_source_id).strip()
        if expected_source_id is not None
        else ""
    )
    declared_source_ids = {
        value
        for value in (payload_source_id, articulation_source_id)
        if value
    }
    if len(declared_source_ids) > 1:
        raise ValueError(
            f"{object_label}: payload source provenance disagrees: "
            f"{sorted(declared_source_ids)}"
        )
    if expected_source and declared_source_ids and declared_source_ids != {
        expected_source
    }:
        raise ValueError(
            f"{object_label}: payload source provenance "
            f"{sorted(declared_source_ids)!r} != {expected_source!r}"
        )

    return {
        "contract": JOINT_PAYLOAD_CONTRACT_NAME,
        "schema_version": JOINT_PAYLOAD_SCHEMA_VERSION,
        "fps": object_fps,
        "num_frames": num_frames,
        "robot_pose_joints": int(pose_aa.shape[1]),
        "robot_dofs": int(dof.shape[1]),
        "relation_links": NUM_RELATION_LINKS,
        "contact_shape": list(state.shape),
        "active_joint_slots": int(joint_pos.shape[1]),
        "source_id": expected_source or next(iter(declared_source_ids), ""),
    }
