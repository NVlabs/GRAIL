"""Runtime validation and rank-local views of a complete joint-source roster."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from gear_sonic.rp2.source_registry import (
    mechanism_resistance_for_entity_kind,
    validate_mechanism_resistance_contract,
    validate_source_temporal_provenance,
)
from gear_sonic.rp2.joint_payload_contract import (
    joint_payload_contract_document,
)

RANK_TOPOLOGY_CONTRACT = "homogeneous_entity_kind_per_rank_v1"
SUPPORTED_ENTITY_KINDS = frozenset({"rigid", "revolute", "prismatic"})


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a mapping")
    if value.get("contract") != "persistent_complete_asset_compatible_roster_v1":
        raise ValueError(f"{path}: unsupported joint-roster contract")
    if value.get("persistent_complete_roster") is not True:
        raise ValueError(f"{path}: roster is not declared complete")
    return value


def validate_roster_temporal_contract(
    manifest: dict[str, Any],
    *,
    runtime_target_fps_hz: float | None = None,
) -> dict[str, Any]:
    """Validate one native-clock/common-runtime contract without source branching."""
    contract = manifest.get("temporal_data_contract")
    if not isinstance(contract, dict):
        raise TypeError("joint roster is missing temporal_data_contract")
    if (
        contract.get("name")
        != "native_paired_clock_to_common_runtime_50hz_v1"
    ):
        raise ValueError("joint roster has an unsupported temporal contract")
    if contract.get("source_clock_field") != "fps":
        raise ValueError("joint roster must use each record's native fps")
    if contract.get("require_robot_object_clock_match") is not True:
        raise ValueError("joint roster must require paired robot/object clocks")
    if contract.get("temporal_constants_unit") != "seconds":
        raise ValueError("joint roster temporal constants must use seconds")
    target_fps = float(contract.get("runtime_target_fps_hz", float("nan")))
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("joint roster has an invalid runtime target FPS")
    if runtime_target_fps_hz is not None and abs(
        float(runtime_target_fps_hz) - target_fps
    ) > 1.0e-9:
        raise ValueError(
            "joint roster/runtime target FPS mismatch: "
            f"{target_fps} vs {runtime_target_fps_hz}"
        )
    if contract.get("filter_cutoff_is_not_sample_rate") is not True:
        raise ValueError("filter cutoff cannot be used as source sample FPS")

    inventory = manifest.get("source_inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise TypeError("joint roster source_inventory must be a mapping")
    for source_name, source in inventory.items():
        provenance = (
            source.get("temporal_provenance")
            if isinstance(source, dict)
            else None
        )
        validate_source_temporal_provenance(
            str(source_name),
            provenance,
        )
    return dict(contract)


def validate_roster_mechanism_resistance_contract(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate the dataset-independent entity-kind resistance contract."""
    return validate_mechanism_resistance_contract(
        manifest.get("mechanism_resistance_contract")
    )


def validate_roster_payload_contract(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require the exact source-independent payload ABI used by the policy."""
    expected = joint_payload_contract_document()
    actual = manifest.get("payload_contract")
    if actual != expected:
        raise ValueError(
            "joint roster payload contract differs from the canonical ABI"
        )
    return dict(expected)


def resolve_rank_local_roster(
    manifest_path: str | Path,
    *,
    num_envs: int,
    rank: int,
    world_size: int,
    runtime_target_fps_hz: float | None = None,
) -> dict[str, Any]:
    """Validate a complete roster and return one rank's immutable slot view."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if num_envs <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(
            "Require num_envs > 0, world_size > 0, and 0 <= rank < world_size"
        )
    if json.loads(path.read_text(encoding="utf-8")).get("contract") == "unified_no_vof_cohort_v1":
        from gear_sonic.rp2.unified_cohort import resolve_unified_cohort

        return resolve_unified_cohort(
            path,
            num_envs=num_envs,
            rank=rank,
            world_size=world_size,
            runtime_target_fps_hz=runtime_target_fps_hz,
        )
    manifest = _load_manifest(path)
    temporal_contract = validate_roster_temporal_contract(
        manifest,
        runtime_target_fps_hz=runtime_target_fps_hz,
    )
    mechanism_resistance_contract = (
        validate_roster_mechanism_resistance_contract(manifest)
    )
    payload_contract = validate_roster_payload_contract(manifest)

    registry = Path(str(manifest.get("registry", ""))).expanduser().resolve()
    if not registry.is_file():
        raise FileNotFoundError(f"{path}: pinned registry is missing: {registry}")

    records = manifest.get("records")
    signatures = manifest.get("asset_signatures")
    slots = manifest.get("slot_assignments")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: roster records are empty")
    if not isinstance(signatures, dict) or not signatures:
        raise ValueError(f"{path}: asset signatures are empty")
    if not isinstance(slots, list) or not slots:
        raise ValueError(f"{path}: slot assignments are empty")

    signature_mechanism_resistance: dict[str, dict[str, Any]] = {}
    for signature, raw_signature in signatures.items():
        if not isinstance(raw_signature, dict):
            raise TypeError(f"{path}: invalid asset signature {signature!r}")
        if str(raw_signature.get("signature", "")) != str(signature):
            raise ValueError(
                f"{path}: asset signature key/payload mismatch for {signature!r}"
            )
        signature_contract = raw_signature.get("contract")
        if not isinstance(signature_contract, dict):
            raise TypeError(
                f"{path}: asset signature {signature!r} has no contract"
            )
        entity_kind = str(signature_contract.get("entity_kind", ""))
        expected_resistance = mechanism_resistance_for_entity_kind(
            mechanism_resistance_contract,
            entity_kind,
        )
        embedded_resistance = signature_contract.get(
            "mechanism_resistance"
        )
        if not isinstance(embedded_resistance, dict):
            raise TypeError(
                f"{path}: asset signature {signature!r} has no "
                "mechanism_resistance"
            )
        if embedded_resistance != expected_resistance:
            raise ValueError(
                f"{path}: asset signature {signature!r} resistance differs "
                f"from the shared {entity_kind!r} contract"
            )
        signature_mechanism_resistance[str(signature)] = {
            "entity_kind": entity_kind,
            "mechanism_resistance": expected_resistance,
        }

    records_by_alias = {}
    compatibility: dict[tuple[str, str], list[str]] = {}
    for raw in records:
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: invalid roster record")
        alias = str(raw.get("alias", ""))
        source = str(raw.get("source", ""))
        signature = str(raw.get("asset_signature", ""))
        revision = str(raw.get("promoted_revision", ""))
        if not alias or not source or not signature or not revision:
            raise ValueError(f"{path}: incomplete roster record {raw}")
        if alias in records_by_alias:
            raise ValueError(f"{path}: duplicate roster alias {alias}")
        if signature not in signatures:
            raise ValueError(f"{path}: record uses unknown signature {signature}")
        records_by_alias[alias] = raw
        compatibility.setdefault((source, signature), []).append(alias)

    source_inventory = manifest.get("source_inventory") or {}
    for source, inventory in source_inventory.items():
        revision = str((inventory or {}).get("promoted_revision", ""))
        source_records = [
            record
            for record in records
            if str(record.get("source")) == str(source)
        ]
        if not revision or any(
            str(record.get("promoted_revision")) != revision
            for record in source_records
        ):
            raise ValueError(
                f"{path}: source {source!r} has inconsistent promoted revisions"
            )
        if len(source_records) != int((inventory or {}).get("available_records", -1)):
            raise ValueError(
                f"{path}: source {source!r} record inventory is incomplete"
            )

    ordered_slots = sorted(slots, key=lambda item: int(item.get("slot", -1)))
    if [int(item.get("slot", -1)) for item in ordered_slots] != list(
        range(len(ordered_slots))
    ):
        raise ValueError(f"{path}: slot indices are not contiguous")
    expected_slots = int(num_envs) * int(world_size)
    declared_world_size = manifest.get("world_size")
    if (
        declared_world_size is not None
        and int(declared_world_size) != int(world_size)
    ):
        raise ValueError(
            f"{path}: roster world_size={declared_world_size}, runtime "
            f"world_size={world_size}"
        )
    if len(ordered_slots) != expected_slots:
        raise ValueError(
            f"{path}: roster has {len(ordered_slots)} slots, expected "
            f"{num_envs} x {world_size} = {expected_slots}"
        )
    if manifest.get("rank_topology_contract") != RANK_TOPOLOGY_CONTRACT:
        raise ValueError(
            f"{path}: roster does not declare {RANK_TOPOLOGY_CONTRACT}"
        )
    rank_entity_kinds = manifest.get("rank_entity_kinds")
    if (
        not isinstance(rank_entity_kinds, list)
        or len(rank_entity_kinds) != world_size
        or any(
            str(kind) not in SUPPORTED_ENTITY_KINDS
            for kind in rank_entity_kinds
        )
    ):
        raise ValueError(
            f"{path}: invalid rank_entity_kinds={rank_entity_kinds!r}"
        )
    rank_entity_kind = str(rank_entity_kinds[rank])

    start = rank * num_envs
    local_slots = ordered_slots[start : start + num_envs]
    compatible_aliases = set()
    normalized_slots = []
    for local_env_id, raw in enumerate(local_slots):
        source = str(raw.get("source", ""))
        signature = str(raw.get("asset_signature", ""))
        representative = str(raw.get("representative_alias", ""))
        aliases = sorted(compatibility.get((source, signature), []))
        if not aliases or representative not in aliases:
            raise ValueError(
                f"{path}: slot {raw.get('slot')} has no compatible motions"
            )
        active_usd = Path(
            str(raw.get("active_articulation_usd", ""))
        ).expanduser().resolve()
        static_usd = Path(
            str(raw.get("static_context_usd", ""))
        ).expanduser().resolve()
        if not active_usd.is_file() or not static_usd.is_file():
            raise FileNotFoundError(
                f"{path}: slot {raw.get('slot')} references missing USD assets"
            )
        signature_resistance = signature_mechanism_resistance.get(signature)
        if signature_resistance is None:
            raise ValueError(
                f"{path}: slot {raw.get('slot')} uses unknown signature "
                f"{signature}"
            )
        if signature_resistance["entity_kind"] != rank_entity_kind:
            raise ValueError(
                f"{path}: rank {rank} requires {rank_entity_kind!r} assets, "
                f"but slot {raw.get('slot')} uses "
                f"{signature_resistance['entity_kind']!r}"
            )
        declared_slot_kind = str(raw.get("rank_entity_kind", ""))
        if declared_slot_kind != rank_entity_kind:
            raise ValueError(
                f"{path}: slot {raw.get('slot')} rank topology "
                f"{declared_slot_kind!r} != {rank_entity_kind!r}"
            )
        compatible_aliases.update(aliases)
        normalized_slots.append(
            {
                **dict(raw),
                "local_env_id": local_env_id,
                "compatible_motion_aliases": aliases,
                **signature_resistance,
                "active_articulation_usd": str(active_usd),
                "static_context_usd": str(static_usd),
            }
        )

    topology_compatible_aliases = {
        alias
        for alias, record in records_by_alias.items()
        if signature_mechanism_resistance[
            str(record["asset_signature"])
        ]["entity_kind"]
        == rank_entity_kind
    }
    expected_motion_aliases: set[str] = set()
    for raw in local_slots:
        expected_motion_aliases.update(
            compatibility.get(
                (
                    str(raw.get("source", "")),
                    str(raw.get("asset_signature", "")),
                ),
                (),
            )
        )
    topology_mismatch_aliases = sorted(
        expected_motion_aliases - topology_compatible_aliases
    )
    if topology_mismatch_aliases:
        raise ValueError(
            f"{path}: rank {rank} slot-reachable motions violate its "
            f"{rank_entity_kind!r} topology: "
            f"{topology_mismatch_aliases[:8]}"
        )
    missing_motion_aliases = sorted(
        expected_motion_aliases - compatible_aliases
    )
    unexpected_motion_aliases = sorted(
        compatible_aliases - expected_motion_aliases
    )
    if missing_motion_aliases or unexpected_motion_aliases:
        raise ValueError(
            f"{path}: rank {rank} violates the slot-reachable "
            "zero-dropout motion contract: "
            f"expected={len(expected_motion_aliases)}, "
            f"resolved={len(compatible_aliases)}, "
            f"missing={missing_motion_aliases[:8]}, "
            f"unexpected={unexpected_motion_aliases[:8]}"
        )

    motion_dir = path.parent / "robot"
    object_motion_dir = path.parent / "objects_with_masks"
    bps_dir = path.parent / "bps"
    if not motion_dir.is_dir():
        raise FileNotFoundError(motion_dir)
    if not object_motion_dir.is_dir():
        raise FileNotFoundError(object_motion_dir)
    if not bps_dir.is_dir() or not (bps_dir / "_basis.npy").is_file():
        raise FileNotFoundError(
            f"{path}: roster BPS directory or shared basis is missing"
        )
    missing_motions = [
        alias
        for alias in sorted(compatible_aliases)
        if not (motion_dir / f"{alias}.pkl").is_file()
    ]
    if missing_motions:
        raise FileNotFoundError(
            f"{path}: compatible robot motions are missing: "
            f"{missing_motions[:8]}"
        )
    missing_objects = [
        alias
        for alias in sorted(compatible_aliases)
        if not (object_motion_dir / f"{alias}.pkl").is_file()
    ]
    if missing_objects:
        raise FileNotFoundError(
            f"{path}: compatible object motions are missing: "
            f"{missing_objects[:8]}"
        )
    missing_bps = [
        alias
        for alias in sorted(compatible_aliases)
        if not (bps_dir / f"{alias}.npy").is_file()
    ]
    if missing_bps:
        raise FileNotFoundError(
            f"{path}: compatible BPS vectors are missing: {missing_bps[:8]}"
        )

    for alias in sorted(compatible_aliases):
        record = records_by_alias[alias]
        resources = record.get("resources")
        if not isinstance(resources, dict):
            raise TypeError(f"{path}: {alias} has no resource path binding")
        # Legacy manifests may also carry per-resource fingerprint or stat fields
        # (e.g. "resource_sha256"); they are ignored. Only the actual resource
        # paths and their existence are checked here.
        for subdir, local_path in (
            ("robot", motion_dir / f"{alias}.pkl"),
            (
                "objects_with_masks",
                object_motion_dir / f"{alias}.pkl",
            ),
        ):
            expected_path = Path(
                str(resources.get(subdir, ""))
            ).expanduser().resolve()
            if expected_path != local_path.resolve():
                raise ValueError(
                    f"{path}: {alias}/{subdir} resource path drifted"
                )
            if not local_path.is_file():
                raise FileNotFoundError(f"{path}: {alias}/{subdir} missing: {local_path}")

    return {
        "manifest_path": str(path),
        "registry": str(registry),
        "temporal_data_contract": temporal_contract,
        "payload_contract": payload_contract,
        "mechanism_resistance_contract": mechanism_resistance_contract,
        "rank": rank,
        "rank_entity_kind": rank_entity_kind,
        "rank_entity_kinds": [str(kind) for kind in rank_entity_kinds],
        "rank_topology_contract": RANK_TOPOLOGY_CONTRACT,
        "world_size": world_size,
        "num_envs": num_envs,
        "motion_dir": str(motion_dir),
        "object_motion_dir": str(object_motion_dir),
        "bps_dir": str(bps_dir),
        "runtime_motion_resolution_contract": (
            "rank_slot_reachable_zero_dropout_v1"
        ),
        "topology_compatible_motion_count": len(
            topology_compatible_aliases
        ),
        "partitioned_out_motion_count": len(
            topology_compatible_aliases - expected_motion_aliases
        ),
        "expected_motion_aliases": sorted(expected_motion_aliases),
        "expected_motion_count": len(expected_motion_aliases),
        "motion_aliases": sorted(compatible_aliases),
        "resolved_motion_count": len(compatible_aliases),
        "dropout_motion_aliases": [],
        "dropout_motion_count": 0,
        "slots": normalized_slots,
    }


def build_asset_compatible_motion_groups(
    roster_view: dict[str, Any],
    loaded_motion_keys: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Map immutable simulator slots to compatible loaded motion indices."""
    motion_keys = [str(key) for key in loaded_motion_keys]
    if not motion_keys:
        raise ValueError("No motions were loaded for the joint roster")
    if len(set(motion_keys)) != len(motion_keys):
        raise ValueError("Loaded joint-roster motion keys are not unique")
    motion_id_by_alias = {
        alias: motion_id for motion_id, alias in enumerate(motion_keys)
    }

    expected_aliases = {str(alias) for alias in roster_view.get("motion_aliases", [])}
    missing = sorted(expected_aliases - set(motion_id_by_alias))
    extra = sorted(set(motion_id_by_alias) - expected_aliases)
    if missing or extra:
        raise ValueError(
            "Loaded motions differ from the rank-local roster: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    groups: list[dict[str, Any]] = []
    group_id_by_key: dict[tuple[str, str], int] = {}
    env_group_ids: list[int] = []
    representative_motion_ids: list[int] = []
    slots = list(roster_view.get("slots") or [])
    if len(slots) != int(roster_view.get("num_envs", -1)):
        raise ValueError(
            "Rank-local roster slot count differs from its declared num_envs"
        )

    for expected_env_id, slot in enumerate(slots):
        local_env_id = int(slot.get("local_env_id", -1))
        if local_env_id != expected_env_id:
            raise ValueError(
                "Rank-local roster slots are not ordered by local environment"
            )
        source = str(slot.get("source", ""))
        signature = str(slot.get("asset_signature", ""))
        aliases = tuple(
            sorted(str(alias) for alias in slot.get("compatible_motion_aliases", []))
        )
        if not source or not signature or not aliases:
            raise ValueError(f"Incomplete joint-roster slot {slot}")
        unknown = [alias for alias in aliases if alias not in motion_id_by_alias]
        if unknown:
            raise ValueError(
                f"Slot {local_env_id} references unloaded motions {unknown[:8]}"
            )
        group_key = (source, signature)
        group_id = group_id_by_key.get(group_key)
        if group_id is None:
            group_id = len(groups)
            group_id_by_key[group_key] = group_id
            groups.append(
                {
                    "source": source,
                    "asset_signature": signature,
                    "motion_aliases": list(aliases),
                    "motion_ids": [motion_id_by_alias[alias] for alias in aliases],
                }
            )
        elif tuple(groups[group_id]["motion_aliases"]) != aliases:
            raise ValueError(
                f"Slots for {source}/{signature} disagree on compatible motions"
            )

        representative = str(slot.get("representative_alias", ""))
        if representative not in aliases:
            raise ValueError(
                f"Slot {local_env_id} representative is not asset-compatible"
            )
        env_group_ids.append(group_id)
        representative_motion_ids.append(motion_id_by_alias[representative])

    return {
        "groups": groups,
        "env_group_ids": env_group_ids,
        "representative_motion_ids": representative_motion_ids,
    }
