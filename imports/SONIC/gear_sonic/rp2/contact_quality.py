"""Source-agnostic quality filters for canonical contact supervision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np

from gear_sonic.rp2.discrete_contact import (
    CONTACT_BODY_GROUPS,
    CONTACT_ENTITY_CLASSES,
    CONTACT_NEUTRAL,
    validate_reference_contact_state,
)


@dataclass(frozen=True)
class BilateralFloatingFootPolicy:
    """Neutralize sustained frames where both feet float above the ground."""

    clearance_threshold_m: float = 0.04
    minimum_duration_seconds: float = 0.25

    def validate(self) -> None:
        if (
            not math.isfinite(self.clearance_threshold_m)
            or self.clearance_threshold_m <= 0.0
        ):
            raise ValueError(
                "clearance_threshold_m must be finite and positive"
            )
        if (
            not math.isfinite(self.minimum_duration_seconds)
            or self.minimum_duration_seconds <= 0.0
        ):
            raise ValueError(
                "minimum_duration_seconds must be finite and positive"
            )


def _true_run_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 1:
        raise ValueError(f"run mask must be one-dimensional, got {value.shape}")
    intervals = []
    start = 0
    while start < len(value):
        if not value[start]:
            start += 1
            continue
        end = start + 1
        while end < len(value) and value[end]:
            end += 1
        intervals.append((start, end))
        start = end
    return intervals


def _retain_true_runs(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be positive")
    retained = np.zeros_like(value)
    for start, end in _true_run_intervals(value):
        if end - start >= minimum_frames:
            retained[start:end] = True
    return retained


def _mask_sha256(mask: np.ndarray) -> str:
    value = np.asarray(mask, dtype=bool)
    payload = np.asarray([value.size], dtype="<i8").tobytes()
    payload += np.packbits(value, bitorder="little").tobytes()
    return hashlib.sha256(payload).hexdigest()


def neutralize_sustained_bilateral_floating_ground_feet(
    reference_state: np.ndarray,
    contact_distance_m: np.ndarray,
    geometry_valid: np.ndarray,
    *,
    fps: float,
    policy: BilateralFloatingFootPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Turn unreliable bilateral floating-foot ground labels into neutral.

    The filter never creates required contact and never changes active-object
    or static-context labels. It is intended for a source preprocessing gate,
    while the runtime contact ABI remains identical for every dataset.
    """

    selected_policy = policy or BilateralFloatingFootPolicy()
    selected_policy.validate()
    frame_rate = float(fps)
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")

    state = validate_reference_contact_state(reference_state).copy()
    distance = np.asarray(contact_distance_m, dtype=np.float64)
    valid = np.asarray(geometry_valid, dtype=bool).copy()
    if distance.shape != state.shape or valid.shape != state.shape:
        raise ValueError(
            "contact state, distance, and validity must have identical shapes; "
            f"got {state.shape}, {distance.shape}, and {valid.shape}"
        )
    if not np.isfinite(distance[valid]).all():
        raise ValueError("valid contact distances must be finite")

    ground_index = CONTACT_ENTITY_CLASSES.index("ground")
    foot_indices = (
        CONTACT_BODY_GROUPS.index("left_foot"),
        CONTACT_BODY_GROUPS.index("right_foot"),
    )
    foot_distance = distance[:, ground_index, foot_indices]
    foot_valid = valid[:, ground_index, foot_indices]
    bilateral_clearance = np.all(
        foot_valid
        & (
            foot_distance
            > float(selected_policy.clearance_threshold_m)
        ),
        axis=1,
    )
    minimum_frames = max(
        1,
        int(
            math.ceil(
                selected_policy.minimum_duration_seconds * frame_rate
                - 1.0e-12
            )
        ),
    )
    neutralized_frames = _retain_true_runs(
        bilateral_clearance,
        minimum_frames,
    )

    before = state[:, ground_index, foot_indices].copy()
    frame_indices = np.flatnonzero(neutralized_frames)
    selection = np.ix_(
        frame_indices,
        [ground_index],
        list(foot_indices),
    )
    state[selection] = CONTACT_NEUTRAL
    valid[selection] = False
    intervals = _true_run_intervals(neutralized_frames)

    audit = {
        "contract": "bilateral_floating_ground_foot_neutralization_v1",
        "policy": asdict(selected_policy),
        "minimum_duration_frames": minimum_frames,
        "candidate_frames": int(np.count_nonzero(bilateral_clearance)),
        "neutralized_frames": int(np.count_nonzero(neutralized_frames)),
        "neutralized_values": int(np.count_nonzero(neutralized_frames) * 2),
        "state_values_changed": int(
            np.count_nonzero(before[neutralized_frames] != CONTACT_NEUTRAL)
        ),
        "validity_values_changed": int(
            np.count_nonzero(neutralized_frames) * 2
        ),
        "neutralized_intervals": [
            {
                "start_frame": int(start),
                "end_frame_exclusive": int(end),
                "frame_count": int(end - start),
                "minimum_bilateral_clearance_m": float(
                    np.min(foot_distance[start:end])
                ),
                "maximum_bilateral_clearance_m": float(
                    np.max(np.min(foot_distance[start:end], axis=1))
                ),
                "p95_bilateral_clearance_m": float(
                    np.quantile(
                        np.min(foot_distance[start:end], axis=1),
                        0.95,
                    )
                ),
            }
            for start, end in intervals
        ],
        "neutralized_frame_mask_sha256": _mask_sha256(
            neutralized_frames
        ),
        "forbidden_values_before": int(
            np.count_nonzero(
                before[neutralized_frames] == -1
            )
        ),
        "required_values_before": int(
            np.count_nonzero(
                before[neutralized_frames] == 1
            )
        ),
        "scope": {
            "entity_class": "ground",
            "body_groups": ["left_foot", "right_foot"],
        },
        "label_policy": (
            "quality-failed values become neutral by setting geometry_valid=false"
        ),
    }
    if audit["required_values_before"]:
        raise ValueError(
            "floating-foot quality filter would remove required contact"
        )
    return state, valid, audit
