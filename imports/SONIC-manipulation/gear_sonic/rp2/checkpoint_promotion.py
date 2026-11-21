"""Fail-closed promotion checks for a migrated joint multi-source checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ACTION_DELTA_CONTRACT = "joint_multisource_checkpoint_action_delta_v5"
REPLAY_REPORT_CONTRACT = "joint_multisource_m5_reference_replay_gate_v1"
ONSET_HAND_GATE_SEMANTICS = {
    "selection": "reference_required_hand_dofs",
    "dofs_per_hand": 7,
    "unrequired_hand_dofs": "audit_only",
    "noncontact_selection": "all_hand_dofs",
}
NONCONTACT_RMS_LIMIT = 1.0e-6
NONCONTACT_MAX_ABS_LIMIT = 1.0e-5
HAND_DOFS_PER_HAND = 7
HAND_ACTION_WIDTH = 14
SUPPORTED_SOURCES = frozenset(
    {
        "rp2_rigid",
        "rp2_revolute",
        "rp2_prismatic",
        "teleop",
        "grail_hf",
        "dex31",
    }
)
REQUIRED_TOPOLOGY_SOURCES = frozenset(
    {"rp2_rigid", "rp2_revolute", "rp2_prismatic"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path_value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(path_value or "")).expanduser()
    if not str(path_value or "").strip() or not path.is_file():
        raise FileNotFoundError(f"{label} report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} report must be a JSON mapping")
    return path.resolve(), value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_exact_float(value: Any, expected: float, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if not math.isclose(result, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label} must equal {expected}, got {result}")
    return result


def _validate_audit_shape(
    value: Any,
    *,
    label: str,
    frame_count: int,
    width: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or malformed")
    if _nonnegative_int(value.get("frame_count"), label=f"{label} frame_count") != (
        frame_count
    ):
        raise ValueError(f"{label} frame_count does not match its gate")
    if _nonnegative_int(value.get("width"), label=f"{label} width") != width:
        raise ValueError(f"{label} width must be {width}")
    return value


def _validate_partition_gate(
    value: Any,
    *,
    label: str,
    frame_count: int,
    rms_limit: float,
    max_abs_limit: float,
    hand_selection: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        raise ValueError(f"{label} is missing, malformed, or failed")
    if _nonnegative_int(value.get("frame_count"), label=f"{label} frame_count") != (
        frame_count
    ):
        raise ValueError(f"{label} frame_count does not match its gate")
    _require_exact_float(
        value.get("rms_limit"),
        rms_limit,
        label=f"{label} rms_limit",
    )
    _require_exact_float(
        value.get("max_abs_limit"),
        max_abs_limit,
        label=f"{label} max_abs_limit",
    )
    if hand_selection is not None and value.get("selection") != hand_selection:
        raise ValueError(
            f"{label} must select {hand_selection}, got "
            f"{value.get('selection')!r}"
        )
    if frame_count > 0:
        measured_rms = _finite_float(
            value.get("rms_max"),
            label=f"{label} rms_max",
        )
        measured_abs = _finite_float(
            value.get("max_abs"),
            label=f"{label} max_abs",
        )
        if measured_rms < 0.0 or measured_rms > rms_limit:
            raise ValueError(f"{label} rms_max exceeds its limit")
        if measured_abs < 0.0 or measured_abs > max_abs_limit:
            raise ValueError(f"{label} max_abs exceeds its limit")
    elif value.get("rms_max") is not None or value.get("max_abs") is not None:
        raise ValueError(f"{label} empty gate must report null maxima")
    return value


def _validate_action_delta_cases(
    action: Mapping[str, Any],
    *,
    expected_sources: set[str],
) -> None:
    thresholds = action.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("M5 action-delta thresholds are missing or malformed")
    onset_limits = {
        "body_rms": _finite_float(
            thresholds.get("onset_1n_body_rms"),
            label="onset body RMS threshold",
        ),
        "body_abs": _finite_float(
            thresholds.get("onset_1n_body_max_abs"),
            label="onset body max-abs threshold",
        ),
        "hand_rms": _finite_float(
            thresholds.get("onset_1n_hand_rms"),
            label="onset hand RMS threshold",
        ),
        "hand_abs": _finite_float(
            thresholds.get("onset_1n_hand_max_abs"),
            label="onset hand max-abs threshold",
        ),
    }
    if any(value < 0.0 for value in onset_limits.values()):
        raise ValueError("M5 onset action-delta thresholds cannot be negative")
    _require_exact_float(
        thresholds.get("noncontact_rms"),
        NONCONTACT_RMS_LIMIT,
        label="noncontact RMS threshold",
    )
    _require_exact_float(
        thresholds.get("noncontact_max_abs"),
        NONCONTACT_MAX_ABS_LIMIT,
        label="noncontact max-abs threshold",
    )

    cases = action.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("M5 action-delta report has no measured cases")

    case_sources: set[str] = set()
    measured_onset_hand_samples = 0
    for index, case in enumerate(cases):
        label = f"M5 action-delta case {index}"
        if not isinstance(case, Mapping) or case.get("passed") is not True:
            raise ValueError(f"{label} is malformed or failed")
        source = case.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{label} has no valid source")
        case_sources.add(source)

        onset = case.get("onset_1n_equivalence_gate")
        if not isinstance(onset, Mapping) or onset.get("passed") is not True:
            raise ValueError(f"{label} onset gate is missing, malformed, or failed")
        onset_frames = _nonnegative_int(
            onset.get("frame_count"),
            label=f"{label} onset frame_count",
        )
        _validate_audit_shape(
            onset.get("combined_43d_audit"),
            label=f"{label} onset combined audit",
            frame_count=onset_frames,
            width=43,
        )
        _validate_partition_gate(
            onset.get("body_29d"),
            label=f"{label} onset body gate",
            frame_count=onset_frames,
            rms_limit=onset_limits["body_rms"],
            max_abs_limit=onset_limits["body_abs"],
        )
        hand = _validate_partition_gate(
            onset.get("hand_14d"),
            label=f"{label} onset hand gate",
            frame_count=onset_frames,
            rms_limit=onset_limits["hand_rms"],
            max_abs_limit=onset_limits["hand_abs"],
            hand_selection="reference_required_hand_dofs",
        )
        if _nonnegative_int(
            hand.get("dofs_per_hand"),
            label=f"{label} onset hand dofs_per_hand",
        ) != HAND_DOFS_PER_HAND:
            raise ValueError(f"{label} onset hand gate is not seven DOFs per hand")
        selected_scalar_count = _nonnegative_int(
            hand.get("selected_scalar_count"),
            label=f"{label} onset selected_scalar_count",
        )
        required_flips = onset.get("required_hand_primitive_sign_flips")
        if (
            not isinstance(required_flips, Mapping)
            or required_flips.get("passed") is not True
        ):
            raise ValueError(f"{label} required-hand primitive gate failed")
        required_hand_samples = _nonnegative_int(
            required_flips.get("hand_sample_count"),
            label=f"{label} required hand sample count",
        )
        if selected_scalar_count != HAND_DOFS_PER_HAND * required_hand_samples:
            raise ValueError(
                f"{label} selected_scalar_count does not match seven DOFs "
                "per required hand sample"
            )
        if onset_frames == 0:
            if required_hand_samples != 0:
                raise ValueError(f"{label} empty onset gate has required hands")
        elif not onset_frames <= required_hand_samples <= 2 * onset_frames:
            raise ValueError(f"{label} required-hand sample count is inconsistent")
        measured_onset_hand_samples += required_hand_samples
        if _nonnegative_int(
            required_flips.get("flip_hand_sample_count"),
            label=f"{label} required-hand flip count",
        ) != 0:
            raise ValueError(f"{label} permits a required-hand primitive flip")
        _require_exact_float(
            required_flips.get("fraction"),
            0.0,
            label=f"{label} required-hand flip fraction",
        )
        _require_exact_float(
            required_flips.get("maximum_fraction"),
            0.0,
            label=f"{label} required-hand maximum flip fraction",
        )

        _validate_audit_shape(
            onset.get("all_hand_14d_audit"),
            label=f"{label} all-hand onset audit",
            frame_count=onset_frames,
            width=HAND_ACTION_WIDTH,
        )
        unrequired_audit = onset.get("unrequired_hand_dofs_audit")
        if not isinstance(unrequired_audit, Mapping):
            raise ValueError(f"{label} unrequired-hand audit is missing")
        unrequired_scalar_count = _nonnegative_int(
            unrequired_audit.get("selected_scalar_count"),
            label=f"{label} unrequired-hand selected scalar count",
        )
        if (
            selected_scalar_count + unrequired_scalar_count
            != onset_frames * HAND_ACTION_WIDTH
        ):
            raise ValueError(f"{label} required/unrequired hand audits do not partition")

        for gate_name in (
            "noncontact_equivalence_gate",
            "neutral_equivalence_gate",
        ):
            gate = case.get(gate_name)
            if not isinstance(gate, Mapping) or gate.get("passed") is not True:
                raise ValueError(f"{label} {gate_name} is missing or failed")
            frame_count = _nonnegative_int(
                gate.get("frame_count"),
                label=f"{label} {gate_name} frame_count",
            )
            _validate_audit_shape(
                gate.get("combined_43d_audit"),
                label=f"{label} {gate_name} combined audit",
                frame_count=frame_count,
                width=43,
            )
            _validate_partition_gate(
                gate.get("body_29d"),
                label=f"{label} {gate_name} body gate",
                frame_count=frame_count,
                rms_limit=NONCONTACT_RMS_LIMIT,
                max_abs_limit=NONCONTACT_MAX_ABS_LIMIT,
            )
            _validate_partition_gate(
                gate.get("hand_14d"),
                label=f"{label} {gate_name} hand gate",
                frame_count=frame_count,
                rms_limit=NONCONTACT_RMS_LIMIT,
                max_abs_limit=NONCONTACT_MAX_ABS_LIMIT,
                hand_selection="all_hand_dofs",
            )

    if case_sources != expected_sources:
        raise ValueError(
            "M5 action-delta case sources differ from the declared sources"
        )
    if measured_onset_hand_samples <= 0:
        raise ValueError("M5 action-delta report has no measured 1 N onset samples")


def validate_migrated_checkpoint_promotion(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require M5 evidence only for the one-time migrated warm start."""

    migration = checkpoint.get("joint_multisource_input_migration")
    if migration is None:
        return None
    if not isinstance(migration, Mapping):
        raise TypeError("joint multi-source migration metadata is malformed")
    require_m5 = bool(
        config.get(
            "require_joint_multisource_m5_for_migrated_checkpoint",
            False,
        )
    )
    allow_deferred = bool(
        config.get(
            "allow_deferred_migrated_checkpoint_warm_start",
            False,
        )
    )
    calibration_status = str(
        config.get("joint_multisource_calibration_status", "")
    )

    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    checkpoint_sha = _sha256(resolved_checkpoint)
    if allow_deferred:
        if require_m5:
            raise ValueError(
                "deferred migrated warm start cannot also claim the M5 "
                "promotion gate is required"
            )
        if calibration_status != "deferred_not_passed":
            raise ValueError(
                "deferred migrated warm start requires calibration status "
                "'deferred_not_passed'"
            )
        return {
            "contract": "joint_multisource_deferred_migration_warm_start_v1",
            "passed": False,
            "promotable": False,
            "status": calibration_status,
            "migrated_checkpoint_sha256": checkpoint_sha,
        }

    if not require_m5:
        raise ValueError(
            "migrated checkpoint loading is forbidden unless the resolved "
            "training config requires the M5 promotion gate"
        )

    action_path, action = _load_report(
        config.get("joint_multisource_m5_action_delta_report"),
        label="M5 action-delta",
    )
    replay_path, replay = _load_report(
        config.get("joint_multisource_m5_replay_report"),
        label="M5 replay",
    )

    if (
        action.get("contract") != ACTION_DELTA_CONTRACT
        or int(action.get("schema_version", -1)) != 5
        or action.get("passed") is not True
    ):
        raise ValueError("M5 action-delta report is not a passed contract")
    if action.get("onset_hand_gate_semantics") != ONSET_HAND_GATE_SEMANTICS:
        raise ValueError(
            "M5 action-delta report lacks the required-hand gating contract"
        )
    if action.get("migrated_checkpoint_sha256") != checkpoint_sha:
        raise ValueError(
            "M5 action-delta report does not certify this migrated checkpoint"
        )
    expected_sources = set(action.get("expected_sources") or [])
    observed_sources = set(action.get("observed_sources") or [])
    if not expected_sources or expected_sources != observed_sources:
        raise ValueError(
            "M5 action-delta expected and observed source sets differ"
        )
    if expected_sources - SUPPORTED_SOURCES:
        raise ValueError("M5 action-delta report contains unsupported sources")
    if REQUIRED_TOPOLOGY_SOURCES - expected_sources:
        raise ValueError(
            "M5 action-delta report lacks a topology-defining RP2 source"
        )
    if (action.get("contact_coverage") or {}).get("passed") is not True:
        raise ValueError(
            "M5 action-delta report lacks calibrated contact coverage"
        )
    if (
        action.get("calibration_sweep_coverage") or {}
    ).get("passed") is not True:
        raise ValueError(
            "M5 action-delta report lacks the complete 18-state force sweep"
        )
    thresholds = action.get("thresholds") or {}
    if thresholds.get("required_hand_primitive_sign_flip_fraction") != 0.0:
        raise ValueError(
            "M5 action-delta report permits required-hand primitive flips"
        )
    if (action.get("shared_action_transform") or {}).get("passed") is not True:
        raise ValueError(
            "M5 action-delta report does not bind one shared frozen ATM"
        )
    _validate_action_delta_cases(
        action,
        expected_sources=expected_sources,
    )

    if (
        replay.get("contract") != REPLAY_REPORT_CONTRACT
        or replay.get("passed") is not True
    ):
        raise ValueError("M5 replay report is not a passed contract")
    if int(replay.get("passed_source_count", -1)) != len(expected_sources):
        raise ValueError("M5 replay report passed-source count is inconsistent")
    replay_sources = {
        str(item.get("source", ""))
        for item in replay.get("source_metrics", [])
        if isinstance(item, Mapping)
    }
    if replay_sources != expected_sources:
        raise ValueError("M5 replay report source set differs from action gate")

    action_suite = action.get("suite_plan") or {}
    replay_suite = replay.get("suite_plan") or {}
    if (
        not action_suite.get("sha256")
        or action_suite.get("sha256") != replay_suite.get("sha256")
    ):
        raise ValueError(
            "M5 replay and action-delta reports use different suite plans"
        )

    return {
        "contract": "migrated_checkpoint_m5_promotion_v1",
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "action_delta_report": str(action_path),
        "action_delta_report_sha256": _sha256(action_path),
        "replay_report": str(replay_path),
        "replay_report_sha256": _sha256(replay_path),
        "suite_plan_sha256": str(action_suite["sha256"]),
        "sources": sorted(expected_sources),
        "passed": True,
    }
