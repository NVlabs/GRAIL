"""Small, source-independent runtime view of one no-VOF asset cohort."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def slot_repeat_factor() -> int:
    """Opt-in runtime repetition of each rank's materialized slots.

    ``UNIFIED_COHORT_SLOT_REPEAT=k`` (k >= 2) runs ``k x num_envs_per_rank`` envs per
    rank on the SAME immutable cohort: every original slot of the rank slice is
    repeated k times in order, preserving resource paths, aliases, asset
    signatures and each rank's ordered unique motion list. Only the runtime
    local/global env ids and counts change; no source motion, USD or surface is
    re-read. Exposure weights change (k envs per motion), so evaluations must use
    fixed unique-motion quotas.
    """
    raw = os.environ.get("UNIFIED_COHORT_SLOT_REPEAT", "1")
    try:
        value = int(str(raw).strip() or "1")
    except ValueError as exc:
        raise ValueError(f"UNIFIED_COHORT_SLOT_REPEAT must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"UNIFIED_COHORT_SLOT_REPEAT must be >= 1, got {value}")
    return value


ZERO_RESISTANCE = {
    "static_friction": 0.0,
    "dynamic_friction": 0.0,
    "viscous_friction": 0.0,
    "damping": 0.0,
}


def rank_interleave_enabled() -> bool:
    """Opt-in rank-balanced slot assignment on the SAME immutable cohort.

    ``UNIFIED_COHORT_RANK_INTERLEAVE=1``: rank r owns the global slots whose index
    satisfies ``index % world_size == r`` (order preserved) instead of the contiguous
    block ``[r*n, (r+1)*n)``. The global env population per PPO iteration is
    unchanged (every slot still exists exactly once), only which rank hosts which
    slot changes, so DDP gradient averaging sees the same data; the per-rank
    unique motion lists change (one-time motion-tensor cache fill per layout).
    Materialized cohorts are contiguous alias ranges sorted by motion key, i.e.
    content-clustered per rank; interleaving spreads heavy and light scenes over
    all ranks and removes the slowest-rank bottleneck.
    """
    return str(os.environ.get("UNIFIED_COHORT_RANK_INTERLEAVE", "0")).strip().lower() in ("1", "true", "yes")


def resolve_unified_cohort(
    manifest_path: str | Path,
    *,
    num_envs: int,
    rank: int,
    world_size: int,
    runtime_target_fps_hz: float | None = None,
) -> dict[str, Any]:
    """Resolve this rank's slot view of a materialized unified no-VOF cohort.

    The asset remains fixed during an episode; reset samples any motion whose
    active USD *and* static-context USD match the slot's exact resolved paths.

    A cohort materialized with ``world_size`` W holds ``num_envs * W`` slots in
    one contiguous list. Rank ``r`` owns slots ``[r * num_envs, (r + 1) * num_envs)``
    (or ``index % W == r`` with UNIFIED_COHORT_RANK_INTERLEAVE) and loads only the
    motions reachable from its own slots. Only the manifest's structure is checked
    here (contract, world size, slot count, duplicate aliases, this rank's
    motion/object binding); the rank-owned slots are selected FIRST and only their
    unique asset paths are normalized (memoized). There is no file-existence
    pre-scan over every pack file and no per-rank re-check of global coverage:
    allocation/coverage are the producer's outputs, and a missing or unreadable
    file is reported by the actual native load that needs it (2026-10-15, direct
    request: no redundant global preflight on the rank-plan miss path).
    """
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "unified_no_vof_cohort_v1":
        raise ValueError(f"{path}: unsupported cohort contract")
    if num_envs <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Require num_envs > 0, world_size > 0, and 0 <= rank < world_size")
    manifest_world_size = int(manifest.get("world_size", 1))
    if manifest_world_size != int(world_size):
        # Fixed-world-size cohorts only; elastic resume across GPU counts is
        # not supported by this path.
        raise ValueError(
            f"{path}: cohort was materialized for world_size={manifest_world_size}, "
            f"runtime world_size={world_size}"
        )
    repeat = slot_repeat_factor()
    if int(num_envs) % repeat:
        raise ValueError(
            f"{path}: NUM_ENVS={num_envs} is not a multiple of UNIFIED_COHORT_SLOT_REPEAT={repeat}"
        )
    base_num_envs = int(num_envs) // repeat
    total_envs = base_num_envs * int(world_size)
    if int(manifest.get("num_envs", -1)) != total_envs:
        raise ValueError(
            f"{path}: NUM_ENVS differs from materialized cohort "
            f"(manifest {manifest.get('num_envs')} slots, runtime {num_envs} x {world_size}"
            + (f" with slot repeat {repeat} -> base {base_num_envs} x {world_size}" if repeat > 1 else "")
            + ")"
        )
    if runtime_target_fps_hz is not None and float(runtime_target_fps_hz) != 50.0:
        raise ValueError("unified no-VOF runtime requires 50 Hz")
    records = manifest.get("records")
    slots = manifest.get("slots")
    if not isinstance(records, list) or not records or not isinstance(slots, list):
        raise ValueError(f"{path}: empty records or slots")
    if len(slots) != total_envs:
        raise ValueError(f"{path}: slot count differs from NUM_ENVS x world_size")
    by_alias = {str(record["alias"]): record for record in records}
    if len(by_alias) != len(records):
        raise ValueError(f"{path}: duplicate motion alias")
    root = path.parent
    start = int(rank) * base_num_envs
    stop = start + base_num_envs
    interleave = rank_interleave_enabled()
    if interleave:
        owned_ids = range(int(rank), total_envs, int(world_size))
    else:
        owned_ids = range(start, stop)
    resolved_paths: dict[str, str] = {}

    def normalize(value: Any) -> str:
        # one normalization per unique spelling (thousands of slots share a handful of USDs); no stat, no raise:
        # a missing file is reported by the native load/spawn that needs it
        text = str(value)
        hit = resolved_paths.get(text)
        if hit is None:
            hit = os.path.realpath(text)
            resolved_paths[text] = hit
        return hit

    rank_aliases: set[str] = set()
    normalized = []
    for env_id in owned_ids:
        slot = slots[env_id]
        if int(slot.get("local_env_id", -1)) != env_id:
            raise ValueError(f"{path}: nonsequential slot {env_id}")
        active = normalize(slot["active_articulation_usd"])
        context = normalize(slot["static_context_usd"])
        surface = slot.get("active_surface_path")
        if surface is not None:
            surface = normalize(surface)
        aliases = list(slot["compatible_motion_aliases"])
        if not aliases or slot["representative_alias"] not in aliases:
            raise ValueError(f"{path}: invalid compatible group in slot {env_id}")
        for alias in aliases:
            record = by_alias.get(alias)
            if record is None:
                raise ValueError(f"{path}: slot {env_id} references unknown alias {alias}")
            # this rank's motion/object binding: a motion may only be sampled into a slot whose assets are its own
            if (record["source"], record["asset_signature"]) != (
                slot["source"], slot["asset_signature"]
            ):
                raise ValueError(f"{path}: slot {env_id} mismatches motion {alias}")
            if normalize(record["active_articulation_usd"]) != active:
                raise ValueError(f"{path}: active USD differs for {alias}")
            if normalize(record["static_context_usd"]) != context:
                raise ValueError(f"{path}: context USD differs for {alias}")
        local_env_id = env_id // int(world_size) if interleave else env_id - start
        rank_aliases.update(aliases)
        normalized.append({
            **slot,
            "global_env_id": env_id,
            "local_env_id": local_env_id,
            "base_global_slot": env_id,
            "active_articulation_usd": active,
            "static_context_usd": context,
            "active_surface_path": surface,
            "entity_kind": "rigid",
            "mechanism_resistance": dict(ZERO_RESISTANCE),
        })
    if len(normalized) != base_num_envs:
        raise ValueError(f"{path}: rank {rank} resolved {len(normalized)} slots, expected {base_num_envs}")
    if repeat > 1:
        expanded = []
        for k in range(repeat):
            for i, slot in enumerate(normalized):
                local_env_id = k * base_num_envs + i
                expanded.append({
                    **slot,
                    "local_env_id": local_env_id,
                    "global_env_id": int(rank) * int(num_envs) + local_env_id,
                    "base_local_env_id": i,
                    "slot_repeat_index": k,
                })
        normalized = expanded
    return {
        "manifest_path": str(path),
        "motion_dir": str(root / "robot"),
        "object_motion_dir": str(root / "objects_with_masks"),
        "bps_dir": str(root / "bps"),
        "rank_entity_kind": "rigid",
        "rank": int(rank),
        "world_size": int(world_size),
        "num_envs": int(num_envs),
        "global_num_envs": total_envs * repeat,
        "base_num_envs": base_num_envs,
        "slot_repeat": repeat,
        "rank_interleave": interleave,
        "motion_aliases": sorted(rank_aliases),
        "slots": normalized,
        "resolved_motion_count": len(rank_aliases),
        "catalog_motion_count": len(by_alias),
        "dropout_motion_count": 0,
        # Default: no joint-source identity tracking (four-case behavior). A
        # cohort materialized with --track-source-stats opts in so the motion
        # library keeps source::motion_key identities and saves per-motion/bin
        # adaptive statistics in its checkpoint state.
        "skip_joint_source_tracking": not bool(manifest.get("track_source_stats", False)),
    }
