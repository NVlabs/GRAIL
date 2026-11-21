"""Strict helpers for resolving promoted joint-training source resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
from typing import Any

from gear_sonic.rp2.discrete_contact import (
    DEFAULT_CLOSE_GAP_SECONDS,
    DEFAULT_CONTACT_FORBIDDEN_THRESHOLD_M,
    DEFAULT_CONTACT_REQUIRED_THRESHOLD_M,
    DEFAULT_MINIMUM_REQUIRED_SECONDS,
    DEFAULT_TRANSITION_NEUTRAL_SECONDS,
)


CANONICAL_HAND_JOINT_ORDER = {
    side: tuple(
        f"{side}_hand_{name}_joint"
        for name in (
            "index_0",
            "index_1",
            "middle_0",
            "middle_1",
            "thumb_0",
            "thumb_1",
            "thumb_2",
        )
    )
    for side in ("left", "right")
}

MECHANISM_RESISTANCE_CONTRACT_NAME = (
    "entity_kind_mechanism_resistance_v1"
)
MECHANISM_RESISTANCE_ENTITY_KINDS = (
    "rigid",
    "revolute",
    "prismatic",
)
MECHANISM_RESISTANCE_VALUE_KEYS = (
    "static_friction",
    "dynamic_friction",
    "viscous_friction",
    "damping",
)
MASS_PLAUSIBILITY_CONTRACT_NAME = (
    "registry_declared_active_panel_mass_v1"
)
MASS_PLAUSIBILITY_SEVERITIES = frozenset({"error", "warning"})


def resolve_active_entity_kind(record: Mapping[str, Any]) -> str:
    """Resolve one canonical entity kind from normalized manifest metadata."""
    if record.get("joint_valid") is False:
        return "rigid"
    active = record.get("active_entity") or {}
    metadata = record.get("metadata") or {}
    if not isinstance(active, Mapping):
        raise TypeError("active_entity manifest field must be a mapping")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata manifest field must be a mapping")
    return str(
        active.get("joint_type")
        or active.get("kind")
        or metadata.get("joint_type")
        or metadata.get("kind")
        or record.get("joint_type")
        or record.get("entity_kind")
        or record.get("kind")
        or "rigid"
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{label} keys must be exactly "
            f"{sorted(expected_keys, key=repr)}; "
            f"missing={sorted(expected_keys - actual_keys, key=repr)}, "
            f"extra={sorted(actual_keys - expected_keys, key=repr)}"
        )


def validate_mechanism_resistance_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize shared resistance values by entity kind."""
    if not isinstance(contract, Mapping):
        raise TypeError("mechanism_resistance_contract must be a mapping")
    _require_exact_keys(
        contract,
        ("name", "by_entity_kind"),
        label="mechanism_resistance_contract",
    )
    if contract.get("name") != MECHANISM_RESISTANCE_CONTRACT_NAME:
        raise ValueError("registry has an unsupported resistance contract")

    by_entity_kind = contract.get("by_entity_kind")
    if not isinstance(by_entity_kind, Mapping):
        raise TypeError(
            "mechanism_resistance_contract.by_entity_kind must be a mapping"
        )
    _require_exact_keys(
        by_entity_kind,
        MECHANISM_RESISTANCE_ENTITY_KINDS,
        label="mechanism_resistance_contract.by_entity_kind",
    )

    normalized: dict[str, dict[str, float]] = {}
    for entity_kind in MECHANISM_RESISTANCE_ENTITY_KINDS:
        values = by_entity_kind[entity_kind]
        if not isinstance(values, Mapping):
            raise TypeError(
                f"mechanism resistance for {entity_kind!r} must be a mapping"
            )
        _require_exact_keys(
            values,
            MECHANISM_RESISTANCE_VALUE_KEYS,
            label=f"mechanism resistance for {entity_kind!r}",
        )
        normalized_values = {}
        for key in MECHANISM_RESISTANCE_VALUE_KEYS:
            number = float(values[key])
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(
                    f"mechanism resistance {entity_kind}.{key} must be "
                    "finite and nonnegative"
                )
            normalized_values[key] = number
        normalized[entity_kind] = normalized_values
    return {
        "name": MECHANISM_RESISTANCE_CONTRACT_NAME,
        "by_entity_kind": normalized,
    }


def mechanism_resistance_for_entity_kind(
    contract: Mapping[str, Any],
    entity_kind: str,
) -> dict[str, float]:
    """Resolve one source-independent resistance profile."""
    normalized = validate_mechanism_resistance_contract(contract)
    if entity_kind not in MECHANISM_RESISTANCE_ENTITY_KINDS:
        raise ValueError(
            f"unsupported active-entity kind for resistance: {entity_kind!r}"
        )
    return dict(normalized["by_entity_kind"][entity_kind])


def _validate_mass_bounds(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    _require_exact_keys(
        value,
        ("minimum", "maximum"),
        label=label,
    )
    minimum = float(value["minimum"])
    maximum = float(value["maximum"])
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum <= minimum
    ):
        raise ValueError(
            f"{label} must contain finite positive bounds with "
            "maximum > minimum"
        )
    return {"minimum": minimum, "maximum": maximum}


def validate_source_mass_plausibility_contract(
    source_name: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one source-declared active-object mass screening contract."""
    if not isinstance(contract, Mapping):
        raise TypeError(
            f"source {source_name!r} mass_plausibility must be a mapping"
        )
    _require_exact_keys(
        contract,
        (
            "contract",
            "provenance",
            "target_body",
            "require_authored_mass",
            "source_bounds_kg",
            "category_rules",
            "identical_mass_batch",
        ),
        label=f"source {source_name!r} mass_plausibility",
    )
    if contract.get("contract") != MASS_PLAUSIBILITY_CONTRACT_NAME:
        raise ValueError(
            f"source {source_name!r} has an unsupported mass plausibility "
            "contract"
        )
    provenance = str(contract.get("provenance", "")).strip()
    if not provenance:
        raise ValueError(
            f"source {source_name!r} mass plausibility needs provenance"
        )
    target_body = str(contract.get("target_body", "")).strip()
    if not target_body.startswith("/"):
        raise ValueError(
            f"source {source_name!r} mass target_body must be an absolute "
            "USD prim path"
        )
    if not isinstance(contract.get("require_authored_mass"), bool):
        raise TypeError(
            f"source {source_name!r} require_authored_mass must be boolean"
        )
    source_bounds = _validate_mass_bounds(
        contract["source_bounds_kg"],
        label=f"source {source_name!r} source_bounds_kg",
    )

    raw_rules = contract.get("category_rules")
    if not isinstance(raw_rules, Sequence) or isinstance(
        raw_rules, (str, bytes)
    ):
        raise TypeError(
            f"source {source_name!r} category_rules must be a sequence"
        )
    normalized_rules: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        label = f"source {source_name!r} category rule {index}"
        if not isinstance(raw_rule, Mapping):
            raise TypeError(f"{label} must be a mapping")
        _require_exact_keys(
            raw_rule,
            (
                "name",
                "motion_key_regex",
                "minimum_kg",
                "maximum_kg",
                "provenance",
            ),
            label=label,
        )
        name = str(raw_rule.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"{label} has a missing or duplicate name")
        names.add(name)
        pattern = str(raw_rule.get("motion_key_regex", "")).strip()
        if not pattern:
            raise ValueError(f"{label} needs motion_key_regex")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(
                f"{label} has invalid motion_key_regex: {error}"
            ) from error
        minimum = float(raw_rule["minimum_kg"])
        maximum = float(raw_rule["maximum_kg"])
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum < source_bounds["minimum"]
            or maximum > source_bounds["maximum"]
            or maximum <= minimum
        ):
            raise ValueError(
                f"{label} bounds must lie inside source_bounds_kg"
            )
        rule_provenance = str(raw_rule.get("provenance", "")).strip()
        if not rule_provenance:
            raise ValueError(f"{label} needs provenance")
        normalized_rules.append(
            {
                "name": name,
                "motion_key_regex": pattern,
                "minimum_kg": minimum,
                "maximum_kg": maximum,
                "provenance": rule_provenance,
            }
        )

    identical = contract.get("identical_mass_batch")
    if not isinstance(identical, Mapping):
        raise TypeError(
            f"source {source_name!r} identical_mass_batch must be a mapping"
        )
    _require_exact_keys(
        identical,
        (
            "severity",
            "minimum_assets",
            "minimum_categories",
            "maximum_fraction",
            "absolute_tolerance_kg",
        ),
        label=f"source {source_name!r} identical_mass_batch",
    )
    severity = str(identical.get("severity", ""))
    if severity not in MASS_PLAUSIBILITY_SEVERITIES:
        raise ValueError(
            f"source {source_name!r} identical-mass severity must be one of "
            f"{sorted(MASS_PLAUSIBILITY_SEVERITIES)}"
        )
    minimum_assets = int(identical.get("minimum_assets", 0))
    minimum_categories = int(identical.get("minimum_categories", 0))
    maximum_fraction = float(identical.get("maximum_fraction", float("nan")))
    tolerance = float(
        identical.get("absolute_tolerance_kg", float("nan"))
    )
    if minimum_assets < 2:
        raise ValueError(
            f"source {source_name!r} minimum_assets must be at least 2"
        )
    if minimum_categories < 1:
        raise ValueError(
            f"source {source_name!r} minimum_categories must be positive"
        )
    if (
        not math.isfinite(maximum_fraction)
        or maximum_fraction <= 0.0
        or maximum_fraction > 1.0
    ):
        raise ValueError(
            f"source {source_name!r} maximum_fraction must lie in (0, 1]"
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"source {source_name!r} mass tolerance must be finite and "
            "nonnegative"
        )

    return {
        "contract": MASS_PLAUSIBILITY_CONTRACT_NAME,
        "provenance": provenance,
        "target_body": target_body,
        "require_authored_mass": bool(contract["require_authored_mass"]),
        "source_bounds_kg": source_bounds,
        "category_rules": normalized_rules,
        "identical_mass_batch": {
            "severity": severity,
            "minimum_assets": minimum_assets,
            "minimum_categories": minimum_categories,
            "maximum_fraction": maximum_fraction,
            "absolute_tolerance_kg": tolerance,
        },
    }


def validate_source_temporal_provenance(
    source_name: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one source clock and optional non-clock filter provenance."""
    if not isinstance(provenance, Mapping):
        raise TypeError(
            f"source {source_name!r} needs temporal_provenance"
        )
    sample_fps = float(provenance.get("sample_fps_hz", float("nan")))
    if not math.isfinite(sample_fps) or sample_fps <= 0.0:
        raise ValueError(
            f"source {source_name!r} has invalid sample_fps_hz"
        )
    reference_filter = provenance.get("reference_filter")
    if reference_filter is None:
        return dict(provenance)
    if not isinstance(reference_filter, Mapping):
        raise TypeError(
            f"source {source_name!r} reference_filter must be a mapping"
        )
    filter_kind = str(reference_filter.get("kind", "")).strip()
    if not filter_kind:
        raise ValueError(
            f"source {source_name!r} reference_filter needs a kind"
        )
    cutoff_hz = float(reference_filter.get("cutoff_hz", float("nan")))
    if (
        not math.isfinite(cutoff_hz)
        or cutoff_hz <= 0.0
        or cutoff_hz >= 0.5 * sample_fps
    ):
        raise ValueError(
            f"source {source_name!r} filter cutoff must lie below Nyquist"
        )
    if filter_kind == "zero_phase_butterworth_low_pass":
        order = float(reference_filter.get("order", float("nan")))
        if (
            not math.isfinite(order)
            or order <= 0.0
            or not order.is_integer()
        ):
            raise ValueError(
                f"source {source_name!r} Butterworth filter needs a "
                "positive integer order"
            )
        if reference_filter.get("zero_phase") is not True:
            raise ValueError(
                f"source {source_name!r} Butterworth filter must record "
                "zero_phase=true"
            )
    return dict(provenance)


SOURCE_LOCAL_BEHAVIOR_KEYS = frozenset(
    {
        "policy_contract",
        "scene_contract",
        "temporal_data_contract",
        "runtime_target_fps_hz",
        "mechanism_resistance_contract",
        "contact_contract",
        "hand_control_contract",
        "observation_overrides",
        "actor_observation_overrides",
        "critic_observation_overrides",
        "reward_overrides",
        "termination_overrides",
        "reset_overrides",
        "action_overrides",
        "runtime_overrides",
    }
)


def validate_shared_training_contracts(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate source-independent temporal-contact and hand-control contracts."""
    data_clock = registry.get("temporal_data_contract")
    if not isinstance(data_clock, Mapping):
        raise TypeError("registry temporal_data_contract must be a mapping")
    if (
        data_clock.get("name")
        != "native_paired_clock_to_common_runtime_50hz_v1"
    ):
        raise ValueError("registry has an unsupported temporal data contract")
    if data_clock.get("source_clock_field") != "fps":
        raise ValueError("source clock must come from each record's fps field")
    if data_clock.get("require_robot_object_clock_match") is not True:
        raise ValueError("paired robot/object clocks must match")
    if data_clock.get("temporal_constants_unit") != "seconds":
        raise ValueError("temporal constants must be source-independent seconds")
    runtime_fps = float(data_clock.get("runtime_target_fps_hz", float("nan")))
    if not math.isfinite(runtime_fps) or abs(runtime_fps - 50.0) > 1.0e-12:
        raise ValueError("the canonical policy runtime clock must be 50 Hz")
    if data_clock.get("filter_cutoff_is_not_sample_rate") is not True:
        raise ValueError("filter cutoff must not be interpreted as sample FPS")

    mechanism_resistance = validate_mechanism_resistance_contract(
        registry.get("mechanism_resistance_contract")
    )

    sources = registry.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("registry sources must be a mapping")
    for source_name, source in sources.items():
        if not isinstance(source, Mapping):
            raise TypeError(f"source {source_name!r} must be a mapping")
        if source.get("enabled") is not True:
            continue
        behavior_overrides = sorted(SOURCE_LOCAL_BEHAVIOR_KEYS.intersection(source))
        if behavior_overrides:
            raise ValueError(
                f"source {source_name!r} declares source-local policy/runtime "
                f"behavior {behavior_overrides}; joint training behavior must "
                "come from the shared top-level contracts"
            )
        validate_source_temporal_provenance(
            str(source_name),
            source.get("temporal_provenance"),
        )
        mass_plausibility = source.get("mass_plausibility")
        if mass_plausibility is not None:
            validate_source_mass_plausibility_contract(
                str(source_name),
                mass_plausibility,
            )

    contact = registry.get("contact_contract")
    if not isinstance(contact, Mapping):
        raise TypeError("registry contact_contract must be a mapping")
    if contact.get("name") != "ternary_full_body_contact_v1":
        raise ValueError("registry has an unsupported contact contract")
    expected_thresholds = {
        "required_distance_m": DEFAULT_CONTACT_REQUIRED_THRESHOLD_M,
        "forbidden_distance_m": DEFAULT_CONTACT_FORBIDDEN_THRESHOLD_M,
    }
    for name, expected in expected_thresholds.items():
        actual = float(contact.get(name, float("nan")))
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"contact {name}={actual} != canonical {expected}"
            )
    temporal = contact.get("temporal_filter")
    if not isinstance(temporal, Mapping):
        raise TypeError("contact temporal_filter must be a mapping")
    if (
        temporal.get("time_base")
        != "seconds_converted_from_each_record_fps"
    ):
        raise ValueError("contact temporal_filter must use each record's FPS")
    expected_durations = {
        "minimum_required_seconds": DEFAULT_MINIMUM_REQUIRED_SECONDS,
        "close_gap_seconds": DEFAULT_CLOSE_GAP_SECONDS,
        "transition_neutral_seconds": DEFAULT_TRANSITION_NEUTRAL_SECONDS,
    }
    for name, expected in expected_durations.items():
        actual = float(temporal.get(name, float("nan")))
        if abs(actual - expected) > 1.0e-12:
            raise ValueError(
                f"contact temporal_filter {name}={actual} != {expected}"
            )
    if temporal.get("rounding") != "nearest_frame_half_up":
        raise ValueError("contact temporal_filter rounding is not canonical")

    hand = registry.get("hand_control_contract")
    if not isinstance(hand, Mapping):
        raise TypeError("registry hand_control_contract must be a mapping")
    if hand.get("name") != "bilateral_discrete_yam_7dof_v1":
        raise ValueError("registry has an unsupported hand control contract")
    if int(hand.get("reference_body_dofs", -1)) != 29:
        raise ValueError("hand control contract must use 29 reference body DOFs")
    if hand.get("use_source_hand_actions") is not False:
        raise ValueError("source-specific hand actions must be disabled")
    if list(hand.get("policy_hand_actions") or []) != [
        "left_primitive",
        "right_primitive",
    ]:
        raise ValueError("hand policy actions are not bilateral and canonical")
    raw_order = hand.get("primitive_joint_order")
    if raw_order is not None:
        if not isinstance(raw_order, Mapping):
            raise TypeError("primitive_joint_order must be a mapping")
        for side, expected in CANONICAL_HAND_JOINT_ORDER.items():
            if tuple(raw_order.get(side) or ()) != expected:
                raise ValueError(
                    f"{side} primitive joint order is not canonical"
                )
    return {
        "temporal_data_contract": dict(data_clock),
        "mechanism_resistance_contract": mechanism_resistance,
        "contact_contract": dict(contact),
        "hand_control_contract": dict(hand),
    }


def resolve_repo_path(repo: Path, value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to the repository."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def source_pack(repo: Path, source: Mapping[str, Any]) -> Path:
    """Return the immutable promoted source pack."""
    value = source.get("source_motion_lib_dir") or source.get("motion_lib_dir")
    if not value:
        raise ValueError("source has no source_motion_lib_dir or motion_lib_dir")
    return resolve_repo_path(repo, value)


def load_motion_key_list(path: Path) -> tuple[str, ...]:
    """Load a duplicate-free, comment-aware motion-key list."""
    keys = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not keys:
        raise ValueError(f"{path}: motion-key list is empty")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: motion-key list contains duplicates")
    return keys


def source_selection(
    repo: Path,
    source: Mapping[str, Any],
) -> tuple[Path | None, tuple[str, ...] | None]:
    """Return an optional checked-in source-selection list."""
    value = source.get("selection_motion_list")
    if not value:
        return None, None
    path = resolve_repo_path(repo, value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, load_motion_key_list(path)


def resource_directory(
    repo: Path,
    pack: Path,
    source: Mapping[str, Any],
    resource_name: str,
) -> tuple[Path, dict[str, Any] | None]:
    """Resolve one resource, honoring a complete promoted override."""
    layout = source.get("resource_layout") or {}
    if resource_name not in layout:
        raise KeyError(f"resource_layout has no {resource_name!r}")
    overrides = source.get("resource_overrides") or {}
    override = overrides.get(resource_name)
    if override is not None:
        if not isinstance(override, Mapping):
            raise TypeError(f"resource override {resource_name!r} is not a mapping")
        directory = override.get("directory")
        if not directory:
            raise ValueError(
                f"resource override {resource_name!r} has no directory"
            )
        return resolve_repo_path(repo, directory), dict(override)
    return (pack / str(layout[resource_name])).resolve(), None


def resource_patch_directory(
    repo: Path,
    source: Mapping[str, Any],
    resource_name: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve an optional partial resource patch directory."""
    patches = source.get("resource_patches") or {}
    patch = patches.get(resource_name)
    if patch is None:
        return None
    if not isinstance(patch, Mapping):
        raise TypeError(f"resource patch {resource_name!r} is not a mapping")
    directory = patch.get("directory")
    if not directory:
        raise ValueError(f"resource patch {resource_name!r} has no directory")
    return resolve_repo_path(repo, directory), dict(patch)


def find_keyed_resource(
    directory: Path,
    key: str,
    suffixes: Sequence[str],
) -> Path | None:
    """Resolve exactly one keyed resource, failing on suffix ambiguity."""
    matches = [
        directory / f"{key}{suffix}"
        for suffix in suffixes
        if (directory / f"{key}{suffix}").is_file()
    ]
    if len(matches) > 1:
        raise ValueError(f"{directory}: ambiguous resources for {key}: {matches}")
    return matches[0] if matches else None


def resolve_patched_resource(
    source_resource: Path,
    patch_directory: Path | None,
    *,
    mode: str | None,
) -> Path:
    """Apply a declared partial patch to one logical source resource."""
    if patch_directory is None:
        return source_resource
    if mode != "replace_by_resolved_target_basename":
        raise ValueError(f"unsupported resource patch mode {mode!r}")
    target_name = source_resource.resolve().name
    candidate = patch_directory / target_name
    return candidate if candidate.is_file() else source_resource
