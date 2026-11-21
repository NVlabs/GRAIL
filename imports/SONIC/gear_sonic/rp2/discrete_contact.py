"""Canonical discrete contact supervision shared by all training sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

import numpy as np

CONTACT_FORBIDDEN = np.int8(-1)
CONTACT_NEUTRAL = np.int8(0)
CONTACT_REQUIRED = np.int8(1)
CONTACT_STATE_VALUES = frozenset((-1, 0, 1))

# These durations preserve the original 3/2/5-frame contact morphology at
# 30 FPS while giving every source the same physical-time contract.
DEFAULT_MINIMUM_REQUIRED_SECONDS = 3.0 / 30.0
DEFAULT_CLOSE_GAP_SECONDS = 2.0 / 30.0
DEFAULT_TRANSITION_NEUTRAL_SECONDS = 5.0 / 30.0
DEFAULT_GRASP_ANTICIPATION_SECONDS = 10.0 / 30.0
DEFAULT_CONTACT_REQUIRED_THRESHOLD_M = 0.02
DEFAULT_CONTACT_FORBIDDEN_THRESHOLD_M = 0.06

CONTACT_ENTITY_CLASSES: tuple[str, ...] = (
    "active_entity",
    "static_context",
    "ground",
)

CONTACT_BODY_GROUPS: tuple[str, ...] = (
    "pelvis_core",
    "torso",
    "head",
    "left_thigh",
    "left_shin",
    "left_foot",
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_thigh",
    "right_shin",
    "right_foot",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
)

G1_CONTACT_GROUP_LINKS: Mapping[str, tuple[str, ...]] = {
    "pelvis_core": (
        "pelvis",
        "pelvis_contour_link",
        "waist_yaw_link",
        "waist_roll_link",
    ),
    "torso": ("torso_link",),
    "head": ("head_link",),
    "left_thigh": (
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
    ),
    "left_shin": ("left_knee_link",),
    "left_foot": (
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
    ),
    "left_upper_arm": (
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
    ),
    "left_forearm": (
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
    ),
    "left_hand": (
        "left_wrist_yaw_link",
        "left_hand_palm_link",
        "left_hand_index_0_link",
        "left_hand_index_1_link",
        "left_hand_middle_0_link",
        "left_hand_middle_1_link",
        "left_hand_thumb_0_link",
        "left_hand_thumb_1_link",
        "left_hand_thumb_2_link",
    ),
    "right_thigh": (
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
    ),
    "right_shin": ("right_knee_link",),
    "right_foot": (
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ),
    "right_upper_arm": (
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
    ),
    "right_forearm": (
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
    ),
    "right_hand": (
        "right_wrist_yaw_link",
        "right_hand_palm_link",
        "right_hand_index_0_link",
        "right_hand_index_1_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
        "right_hand_thumb_0_link",
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
    ),
}


@dataclass(frozen=True)
class TemporalContactFrames:
    """Record-FPS realization of the canonical temporal contact contract."""

    fps: float
    minimum_required_seconds: float
    close_gap_seconds: float
    transition_neutral_seconds: float
    minimum_required_frames: int
    close_gap_frames: int
    transition_neutral_frames: int

    def provenance(self) -> dict[str, float | int | str]:
        return {
            "time_base": "seconds_converted_from_record_fps",
            "rounding": "nearest_frame_half_up",
            **asdict(self),
        }


def duration_seconds_to_frames(
    duration_seconds: float,
    fps: float,
    *,
    at_least_one_if_positive: bool = False,
) -> int:
    """Convert a physical duration to frames without source-specific rules."""
    duration = float(duration_seconds)
    frame_rate = float(fps)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(
            f"duration_seconds must be finite and non-negative, got {duration}"
        )
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {frame_rate}")
    frames = int(math.floor(duration * frame_rate + 0.5))
    if at_least_one_if_positive and duration > 0.0:
        frames = max(1, frames)
    return frames


def temporal_contact_frames(
    fps: float,
    *,
    minimum_required_seconds: float = DEFAULT_MINIMUM_REQUIRED_SECONDS,
    close_gap_seconds: float = DEFAULT_CLOSE_GAP_SECONDS,
    transition_neutral_seconds: float = DEFAULT_TRANSITION_NEUTRAL_SECONDS,
) -> TemporalContactFrames:
    """Realize the shared temporal contact contract at one record's FPS."""
    return TemporalContactFrames(
        fps=float(fps),
        minimum_required_seconds=float(minimum_required_seconds),
        close_gap_seconds=float(close_gap_seconds),
        transition_neutral_seconds=float(transition_neutral_seconds),
        minimum_required_frames=duration_seconds_to_frames(
            minimum_required_seconds,
            fps,
            at_least_one_if_positive=True,
        ),
        close_gap_frames=duration_seconds_to_frames(close_gap_seconds, fps),
        transition_neutral_frames=duration_seconds_to_frames(
            transition_neutral_seconds,
            fps,
        ),
    )


def validate_temporal_contact_contract(
    contract: Mapping[str, object],
    *,
    fps: float,
) -> TemporalContactFrames:
    """Validate a duration-based contract and reproduce its frame realization."""
    if not isinstance(contract, Mapping):
        raise TypeError("temporal contact contract must be a mapping")
    if contract.get("time_base") != "seconds_converted_from_record_fps":
        raise ValueError(
            "temporal contact contract must use seconds converted from record FPS"
        )
    contract_fps = float(contract.get("fps", float("nan")))
    if not math.isclose(contract_fps, float(fps), rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(
            f"temporal contact FPS {contract_fps} does not match record FPS {fps}"
        )
    realized = temporal_contact_frames(
        fps,
        minimum_required_seconds=float(
            contract["minimum_required_seconds"]
        ),
        close_gap_seconds=float(contract["close_gap_seconds"]),
        transition_neutral_seconds=float(
            contract["transition_neutral_seconds"]
        ),
    )
    for name in (
        "minimum_required_frames",
        "close_gap_frames",
        "transition_neutral_frames",
    ):
        if int(contract[name]) != int(getattr(realized, name)):
            raise ValueError(
                f"temporal contact {name}={contract[name]} does not match "
                f"duration/FPS realization {getattr(realized, name)}"
            )
    return realized


def canonical_hand_contact_indices(
    hand: str,
) -> tuple[int, tuple[int, ...]]:
    """Return active-entity and canonical body indices for one/either hand."""
    hand_names = (
        ("left_hand", "right_hand")
        if hand == "either"
        else (str(hand),)
    )
    unknown = sorted(set(hand_names) - {"left_hand", "right_hand"})
    if unknown:
        raise ValueError(f"Unsupported canonical hand selector: {unknown}")
    return (
        CONTACT_ENTITY_CLASSES.index("active_entity"),
        tuple(CONTACT_BODY_GROUPS.index(name) for name in hand_names),
    )


def canonical_contact_shape(num_frames: int) -> tuple[int, int, int]:
    """Return the canonical reference contact tensor shape."""
    return (
        int(num_frames),
        len(CONTACT_ENTITY_CLASSES),
        len(CONTACT_BODY_GROUPS),
    )


def validate_reference_contact_state(
    state: np.ndarray,
    *,
    num_frames: int | None = None,
) -> np.ndarray:
    """Validate and return an ``int8`` canonical contact tensor."""
    value = np.asarray(state)
    if value.ndim != 3:
        raise ValueError(
            "Reference contact state must have shape [T, entity_class, body_group], "
            f"got {value.shape}"
        )
    expected_tail = (len(CONTACT_ENTITY_CLASSES), len(CONTACT_BODY_GROUPS))
    if value.shape[1:] != expected_tail:
        raise ValueError(
            f"Reference contact state tail must be {expected_tail}, got {value.shape[1:]}"
        )
    if num_frames is not None and value.shape[0] != int(num_frames):
        raise ValueError(
            f"Reference contact frame count is {value.shape[0]}, expected {num_frames}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("Reference contact state contains non-finite values")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded):
        raise ValueError("Reference contact state must contain discrete integer labels")
    unique = set(int(item) for item in np.unique(rounded))
    if not unique.issubset(CONTACT_STATE_VALUES):
        raise ValueError(
            "Reference contact state contains labels outside {-1, 0, +1}: "
            f"{sorted(unique)}"
        )
    return rounded.astype(np.int8, copy=False)


def _remove_short_true_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if minimum <= 1:
        return result
    flat = result.reshape(result.shape[0], -1)
    for column in range(flat.shape[1]):
        start = 0
        while start < flat.shape[0]:
            if not flat[start, column]:
                start += 1
                continue
            end = start + 1
            while end < flat.shape[0] and flat[end, column]:
                end += 1
            if end - start < minimum:
                flat[start:end, column] = False
            start = end
    return result


def _close_short_false_gaps(mask: np.ndarray, maximum: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if maximum < 1:
        return result
    flat = result.reshape(result.shape[0], -1)
    for column in range(flat.shape[1]):
        start = 0
        while start < flat.shape[0]:
            if flat[start, column]:
                start += 1
                continue
            end = start + 1
            while end < flat.shape[0] and not flat[end, column]:
                end += 1
            bounded = start > 0 and end < flat.shape[0]
            if bounded and end - start <= maximum:
                flat[start:end, column] = True
            start = end
    return result


def _temporal_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return value.copy()
    result = value.copy()
    for offset in range(1, radius + 1):
        result[offset:] |= value[:-offset]
        result[:-offset] |= value[offset:]
    return result


def discretize_contact_distances(
    distance_m: np.ndarray,
    *,
    geometry_valid: np.ndarray | None = None,
    required_threshold_m: float = DEFAULT_CONTACT_REQUIRED_THRESHOLD_M,
    forbidden_threshold_m: float = DEFAULT_CONTACT_FORBIDDEN_THRESHOLD_M,
    minimum_required_frames: int,
    close_gap_frames: int,
    transition_neutral_frames: int,
) -> np.ndarray:
    """Convert exact geometry distances into required/neutral/forbidden labels.

    The first axis is time. Remaining axes can be entity classes and body
    groups or any other fixed contact vocabulary. Temporal frame counts are
    deliberately required: callers must derive them from physical durations
    and the record FPS with :func:`temporal_contact_frames`.
    """
    distance = np.asarray(distance_m, dtype=np.float64)
    if distance.ndim < 1:
        raise ValueError(f"Contact distance needs a time axis, got {distance.shape}")
    if not 0.0 <= required_threshold_m < forbidden_threshold_m:
        raise ValueError(
            "Contact thresholds must satisfy "
            f"0 <= required < forbidden, got {required_threshold_m}, "
            f"{forbidden_threshold_m}"
        )
    if minimum_required_frames < 1 or close_gap_frames < 0:
        raise ValueError("Temporal contact parameters must be non-negative")
    if transition_neutral_frames < 0:
        raise ValueError("transition_neutral_frames must be non-negative")

    finite = np.isfinite(distance)
    if geometry_valid is None:
        valid = finite
    else:
        valid = np.asarray(geometry_valid, dtype=bool)
        if valid.shape != distance.shape:
            valid = np.broadcast_to(valid, distance.shape)
        valid = valid & finite

    required = valid & (distance <= float(required_threshold_m))
    required = _close_short_false_gaps(required, int(close_gap_frames))
    required = _remove_short_true_runs(required, int(minimum_required_frames))
    transition = _temporal_dilation(required, int(transition_neutral_frames))
    forbidden = (
        valid
        & (distance >= float(forbidden_threshold_m))
        & ~transition
        & ~required
    )

    state = np.full(distance.shape, CONTACT_NEUTRAL, dtype=np.int8)
    state[forbidden] = CONTACT_FORBIDDEN
    state[required] = CONTACT_REQUIRED
    return state


def contact_mismatch(
    reference_state: np.ndarray,
    measured_contact: np.ndarray,
) -> np.ndarray:
    """Return the InterMimic-style discrete contact mismatch.

    Required contact gives ``1-contact``, forbidden contact gives ``contact``,
    and neutral contact gives zero.
    """
    reference = np.asarray(reference_state)
    measured = np.asarray(measured_contact)
    reference = validate_reference_contact_state(reference)
    if measured.shape != reference.shape:
        raise ValueError(
            f"Measured/reference contact shape mismatch: {measured.shape} vs "
            f"{reference.shape}"
        )
    if not np.all(np.isfinite(measured)):
        raise ValueError("Measured contact contains non-finite values")
    measured_rounded = np.rint(measured)
    if not np.array_equal(measured, measured_rounded) or not set(
        int(item) for item in np.unique(measured_rounded)
    ).issubset({0, 1}):
        raise ValueError("Measured contact must contain binary 0/1 labels")
    reference_f = reference.astype(np.float32)
    measured_f = measured_rounded.astype(np.float32)
    return reference_f * ((reference_f + 1.0) * 0.5 - measured_f)


def aggregate_link_contact(
    link_contact: np.ndarray,
    link_names: Sequence[str],
) -> np.ndarray:
    """OR per-link binary contact into the canonical 15 body groups."""
    contact = np.asarray(link_contact)
    if contact.ndim < 1 or contact.shape[-1] != len(link_names):
        raise ValueError(
            f"Expected final link axis of width {len(link_names)}, got {contact.shape}"
        )
    rounded = np.rint(contact)
    if not np.array_equal(contact, rounded) or not set(
        int(item) for item in np.unique(rounded)
    ).issubset({0, 1}):
        raise ValueError("Per-link contact must contain binary 0/1 labels")

    link_index = {str(name): index for index, name in enumerate(link_names)}
    missing = sorted(
        {
            link
            for group in CONTACT_BODY_GROUPS
            for link in G1_CONTACT_GROUP_LINKS[group]
            if link not in link_index
        }
    )
    if missing:
        raise KeyError(f"Missing canonical G1 contact links: {missing}")

    grouped = np.zeros(
        contact.shape[:-1] + (len(CONTACT_BODY_GROUPS),),
        dtype=np.int8,
    )
    for group_index, group_name in enumerate(CONTACT_BODY_GROUPS):
        indices = [link_index[name] for name in G1_CONTACT_GROUP_LINKS[group_name]]
        grouped[..., group_index] = np.any(rounded[..., indices] != 0, axis=-1)
    return grouped
