#!/usr/bin/env python3
"""Plan and materialize complete five-source, fixed-asset training cohorts.

Only small symlink libraries and JSON manifests are written. Source payloads,
USDs, and weights remain in their source packs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Mapping
import functools
import json
import os
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gear_sonic package (resolver is stdlib-only)
from gear_sonic.rp2.active_usd_resolver import (  # noqa: E402
    NATIVE_SUBDIR,
    describe_active_usd_selection,
    find_active_usd,
    selection_summary_line,
    sources_config_selections,
)


def _asset(pack: Path, subdir: str, key: str) -> Path:
    for suffix in (".usd", ".usda"):
        candidate = pack / subdir / f"{key}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{pack}/{subdir}/{key}.usd[a]")


PAIRED_STATIC_CONTEXT_SUBDIR = "static_context_usd_grail_paired"


def _read_pack_index(pack: Path, manifest_path: Path) -> tuple[dict, list[dict]]:
    """The pack index records to use. ``manifest_path`` is either a full index (``{"records": [...]}``) or a SPLIT
    selection (``{"motion_keys": [...]}``), in which case the pack's
    own index is filtered to those keys in the pack's order (a split that names a key the pack lacks is an error)."""
    manifest = json.loads(manifest_path.read_text())
    if isinstance(manifest, dict) and "records" in manifest:
        return manifest, list(manifest["records"])
    if isinstance(manifest, dict) and isinstance(manifest.get("motion_keys"), list):
        full = json.loads((pack / "active_entity_manifest.json").read_text())
        wanted = {str(k) for k in manifest["motion_keys"]}
        records = [r for r in full["records"] if str(r.get("motion_key")) in wanted]
        missing = wanted - {str(r.get("motion_key")) for r in records}
        if missing:
            raise ValueError(f"{manifest_path}: {len(missing)} selected keys are not in {pack}/active_entity_manifest.json (e.g. {sorted(missing)[:3]})")
        return {**{k: v for k, v in full.items() if k != "records"}, "split": manifest.get("split"), "split_source": str(manifest_path)}, records
    raise ValueError(f"{manifest_path}: neither a pack index with 'records' nor a split selection with 'motion_keys'")


def _static_context_asset(pack: Path, key: str, subdir: str | None) -> tuple[Path, str]:
    """``(path, subdir_used)``: the optional static-context store ``subdir`` (e.g. ``static_context_usd_support_v1``,
    written by tools/rp2_build_inferred_support_context.py for motions that need inferred support) when it has the
    key, else the pack's paired context. A motion without a file in the optional store simply keeps its paired
    context: the store is sparse by design."""
    if subdir and subdir != PAIRED_STATIC_CONTEXT_SUBDIR:
        store = Path(subdir)
        if store.is_absolute():            # a store outside the pack (e.g. a test-owned directory): <store>/<key>.usd
            for suffix in (".usd", ".usda"):
                candidate = store / f"{key}{suffix}"
                if candidate.is_file():
                    return candidate.resolve(), str(store)
        else:
            try:
                return _asset(pack, subdir, key), subdir
            except FileNotFoundError:
                pass
    return _asset(pack, PAIRED_STATIC_CONTEXT_SUBDIR, key), PAIRED_STATIC_CONTEXT_SUBDIR


def _object_key(record: dict, rule: dict | None) -> str:
    """Stable per-object label used for physics overrides and object statistics.

    Default: the pack's ``active_object`` field. Sources whose ``active_object``
    is a placeholder may declare ``{"from": "motion_key_segment", "separator":
    "__", "index": 1}`` in the source configuration. Falls back to the motion
    key when the requested field is empty.
    """
    rule = rule or {"from": "active_object"}
    origin = str(rule.get("from", "active_object"))
    if origin == "active_object":
        value = str(record.get("active_object", ""))
    elif origin == "motion_key_segment":
        parts = str(record["source_motion_key"]).split(str(rule.get("separator", "__")))
        index = int(rule.get("index", 0))
        value = parts[index] if -len(parts) <= index < len(parts) else ""
    elif origin == "motion_key":
        value = str(record["source_motion_key"])
    else:
        raise ValueError(f"Unsupported object_key rule: {rule}")
    return value or str(record["source_motion_key"])


SOURCES_CACHE_LOG = "[build_unified_no_vof] sources cache"
def _load_sources(
    config_path: Path, limit: int | None, cache_path: Path | None = None, active_usd_selection: str | None = None,
    *, return_decision: bool = False, static_context_subdir: str | None = None, boundary_support: str = "off", support_store: Path | None = None,
):
    """Load every source record (with its BPS vector) from the promoted packs.

    ``cache_path`` (optional) stores the loaded records so repeated plan/submit
    calls and natural re-layouts do not re-read ~50k index files; it is keyed on
    the exact source configuration content, ``limit``, the identity of every
    pack index and the resolved active-USD selection. It is a performance cache
    only: an unreadable / malformed cache file or a failed write is an ordinary
    miss or a skipped write and the native discovery result is used; a cache that
    reads but holds content the planner cannot use fails at USE in build_plan,
    which then discovers natively and rewrites it (no upfront per-record schema
    scan: user decision 2026-10-14). Configuration / index / payload errors of the
    native path stay fatal. One stderr line: ``[build_unified_no_vof] sources
    cache: path=... decision=hit|miss_absent|miss_key|miss_unreadable:<Exc>
    write=skipped_hit|written|failed:<Exc>``.
    """
    config = json.loads(config_path.read_text())
    # Cache key: the parsed config, the limit, the identity (size, mtime) of every
    # pack's active_entity_manifest.json AND the resolved active-USD directory per
    # source, so a re-materialized pack or a changed asset selection (default flip
    # with unchanged config text, launcher override) invalidates the cache naturally;
    # entries written before the selection field existed simply miss.
    index_identity = _pack_index_identity(config, active_usd_selection, static_context_subdir, boundary_support, support_store)
    decision = "miss_absent"
    if cache_path is not None:
        try:
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text())
                if (
                    _same_config_through_aliases(cached.get("config"), config)
                    and cached.get("limit") == limit
                    and cached.get("index_identity") == index_identity
                ):
                    sources = cached["sources"]
                    if _cached_support_failures(sources):
                        # a catalog cached before failed support decisions became errors: rediscover natively
                        decision = "miss_failed_support"
                    else:
                        print(f"{SOURCES_CACHE_LOG}: path={cache_path} decision=hit write=skipped_hit", file=sys.stderr)
                        return (sources, config, "hit") if return_decision else (sources, config)
                else:
                    decision = "miss_key"
        except (OSError, ValueError, TypeError, AttributeError, LookupError) as exc:
            # unreadable / malformed / foreign cache content: an ordinary miss, never a failure of the build
            decision = f"miss_unreadable:{type(exc).__name__}"
    sources, config = _load_sources_uncached(config, limit, active_usd_selection, static_context_subdir, boundary_support, support_store)
    if cache_path is not None:
        write = _write_sources_cache(cache_path, config, limit, index_identity, sources)
        print(f"{SOURCES_CACHE_LOG}: path={cache_path} decision={decision} write={write}", file=sys.stderr)
    return (sources, config, decision) if return_decision else (sources, config)


def _cached_support_failures(sources: Any) -> int:
    """Records of a cached catalog whose boundary-support decision failed or had no bounds (never valid for reuse)."""
    bad = 0
    if isinstance(sources, dict):
        for records in sources.values():
            for r in records if isinstance(records, list) else []:
                status = str((r.get("boundary_support") or {}).get("status", "")) if isinstance(r, dict) else ""
                if status.startswith("failed") or status == "no_bounds":
                    bad += 1
    return bad


def _canonical_paths_in(value: Any) -> Any:
    """``value`` with every absolute path string replaced by its canonical spelling (realpath of the directory part):
    two mount spellings can name the same storage, so a config that differs only in how a pack directory is
    spelled is the same configuration (direct human rule 2026-10-15: never repeat work because a directory spelling
    changed). Lookup-time comparison only: the cache file keeps the raw config, so caches written before this and by
    this code read each other."""
    if isinstance(value, str):
        return _canonical_asset_path(value) if value.startswith("/") else value
    if isinstance(value, dict):
        return {k: _canonical_paths_in(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonical_paths_in(v) for v in value]
    return value


def _same_config_through_aliases(cached_config: Any, config: Any) -> bool:
    return cached_config == config or _canonical_paths_in(cached_config) == _canonical_paths_in(config)


def _write_sources_cache(cache_path: Path, config: dict, limit: int | None, index_identity: dict, sources: dict) -> str:
    """Atomic optional write; a cache that cannot be written never fails the successfully loaded catalog."""
    import os

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"config": config, "limit": limit, "index_identity": index_identity, "sources": sources}))
        tmp.replace(cache_path)
        return "written"
    except (OSError, TypeError, ValueError) as exc:
        return f"failed:{type(exc).__name__}"


def _pack_index_identity(config: dict, active_usd_selection: str | None = None,
                         static_context_subdir: str | None = None, boundary_support: str = "off",
                         support_store: Path | None = None) -> dict[str, list]:
    out: dict[str, list] = {}
    selections = sources_config_selections(config, active_usd_selection)
    for source, spec in config.get("sources", {}).items():
        index = Path(spec["manifest"]) if spec.get("manifest") else Path(spec["pack"]) / "active_entity_manifest.json"
        try:
            stat = index.stat()
            out[source] = [int(stat.st_size), int(stat.st_mtime_ns)]
        except FileNotFoundError:
            out[source] = [-1, -1]
        out[source].append(selections[source]["subdir"])  # resolved active-USD store is part of the identity
        _ctx = spec.get("static_context_subdir") or static_context_subdir
        if _ctx:
            out[source].append(str(_ctx))  # an optional static-context store changes the records too
        if boundary_support != "off":
            out[source].append(f"boundary_support={boundary_support}:{support_store}")  # inferred contexts change the records
    return out


def _load_sources_uncached(
    config: dict, limit: int | None, active_usd_selection: str | None = None, static_context_subdir: str | None = None,
    boundary_support: str = "off", support_store: Path | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    if config.get("contract") != "unified_no_vof_sources_v1":
        raise ValueError("Unsupported source configuration")
    sources: dict[str, list[dict]] = {}
    shared_basis = None
    selections = sources_config_selections(config, active_usd_selection)
    for source, spec in config["sources"].items():
        pack = Path(spec["pack"])
        active_info = selections[source]
        active_subdir = active_info["subdir"]
        native_fallback_keys = 0
        # Optional per-source index override: a pack may ship several indices (e.g. OmniContact ``all`` vs ``train``);
        # ``"manifest": "<path>"`` selects one without copying the pack (default: the pack's active_entity_manifest.json).
        manifest_path = Path(spec["manifest"]) if spec.get("manifest") else pack / "active_entity_manifest.json"
        manifest, raw_records = _read_pack_index(pack, manifest_path)
        basis = np.load(pack / "bps" / "_basis.npy")
        if shared_basis is not None and (basis.shape != shared_basis.shape or not np.array_equal(basis, shared_basis)):
            raise ValueError(f"{source}: BPS basis differs from the common basis")
        shared_basis = basis
        seen = set()
        records = []
        for index, raw in enumerate(raw_records[:limit] if limit else raw_records):
            key = str(raw["motion_key"])
            if key in seen:
                raise ValueError(f"{source}: duplicate key {key}")
            seen.add(key)
            fps = float(raw.get("fps", spec["native_fps_hz"]))
            if abs(fps - float(spec["native_fps_hz"])) > 1.0e-9:
                raise ValueError(f"{source}/{key}: FPS {fps} differs from source declaration")
            active, from_native = find_active_usd(pack, key, subdir=active_subdir)
            native_fallback_keys += int(from_native)
            # per-source "static_context_subdir" in the sources config wins over the launcher-level value
            context, context_subdir = _static_context_asset(pack, key, spec.get("static_context_subdir") or static_context_subdir)
            record = {
                "alias": f"{source}_{index:06d}",
                "source": source,
                "source_motion_key": key,
                "source_sample_fps_hz": fps,
                "promoted_revision": str(raw.get("promoted_revision") or manifest.get("promoted_revision") or pack.name),
                "pack": str(pack),
                "active_articulation_usd": str(active.resolve()),
                "active_usd_subdir": NATIVE_SUBDIR if from_native else active_subdir,
                "active_usd_native_fallback": bool(from_native),
                "static_context_usd": str(context),
                "static_context_subdir": context_subdir,
                "active_object": str(raw.get("active_object", "")),
            }
            # Source-qualified: the same bare object name may exist in two packs
            # with different assets/masses (e.g. teleop and GRAIL "alcohol_11").
            record["object_key"] = f"{source}/{_object_key(record, spec.get('object_key'))}"
            bps_path = _bps_path(record)
            bps_vector = np.load(bps_path)
            if bps_vector.shape != (10,) or not np.isfinite(bps_vector).all():
                raise ValueError(f"{source}/{key}: invalid 10D BPS vector")
            record["bps_path"] = str(bps_path.resolve())
            record["bps_vector"] = bps_vector.astype(float).tolist()
            records.append(record)
        # summary is printed by the call; it is NOT stored in ``config`` (the cached config must stay byte-equal to the file)
        _apply_boundary_support_to_source(
            source, pack, records, {str(r.get("motion_key")): r for r in raw_records}, mode=boundary_support, support_store=support_store)
        sources[source] = records
    return sources, config


def build_plan(
    config_path: Path,
    *,
    num_envs: int,
    max_motions: int,
    max_group_motions: int,
    limit_per_source: int | None = None,
    sources_cache: Path | None = None,
    active_usd_selection: str | None = None,
    static_context_subdir: str | None = None,
    boundary_support: str = "off",
    support_store: Path | None = None,
) -> dict[str, Any]:
    if min(num_envs, max_motions, max_group_motions) <= 0:
        raise ValueError("num_envs, max_motions, and max_group_motions must be positive")
    sources, config, decision = _load_sources(config_path, limit_per_source, sources_cache, active_usd_selection,
                                              return_decision=True, static_context_subdir=static_context_subdir,
                                              boundary_support=boundary_support, support_store=support_store)
    try:
        return _plan_from_sources(sources, config, config_path=Path(config_path), num_envs=num_envs, max_motions=max_motions,
                                  max_group_motions=max_group_motions, active_usd_selection=active_usd_selection,
                                  static_context_subdir=static_context_subdir)
    except Exception as exc:  # noqa: BLE001 - only a CACHED catalog may be unusable; a natively discovered one re-raises
        if decision != "hit" or sources_cache is None or not Path(sources_cache).is_file():
            raise
        print(f"{SOURCES_CACHE_LOG}: path={sources_cache} cached catalog unusable at planning ({type(exc).__name__}: {exc}); "
              "discovering natively and rewriting the cache", file=sys.stderr)
    config = json.loads(Path(config_path).read_text())
    index_identity = _pack_index_identity(config, active_usd_selection, static_context_subdir, boundary_support, support_store)
    sources, config = _load_sources_uncached(config, limit_per_source, active_usd_selection, static_context_subdir, boundary_support, support_store)
    _write_sources_cache(Path(sources_cache), config, limit_per_source, index_identity, sources)
    return _plan_from_sources(sources, config, config_path=Path(config_path), num_envs=num_envs, max_motions=max_motions,
                              max_group_motions=max_group_motions, active_usd_selection=active_usd_selection,
                              static_context_subdir=static_context_subdir)


def _plan_from_sources(sources: dict, config: dict, *, config_path: Path, num_envs: int, max_motions: int,
                       max_group_motions: int, active_usd_selection: str | None,
                       static_context_subdir: str | None = None) -> dict[str, Any]:
    # Per-source active-USD selection as actually resolved (recorded in the plan and the cohort manifest).
    active_usd_selection_report = {
        name: dict(info, keys_from_native_fallback=sum(int(r.get("active_usd_native_fallback", False)) for r in sources.get(name, [])))
        for name, info in sources_config_selections(config, active_usd_selection).items()
    }
    for name, info in active_usd_selection_report.items():  # logged for cached and uncached loads alike
        print(selection_summary_line(name, info, info["keys_from_native_fallback"]), file=sys.stderr)
    queues: dict[str, deque[dict]] = {}
    all_aliases = set()
    for source, records in sources.items():
        groups: dict[tuple[str, str, tuple[float, ...]], list[dict]] = defaultdict(list)
        for record in records:
            groups[(record["active_articulation_usd"], record["static_context_usd"], tuple(record["bps_vector"]))].append(record)
            if record["alias"] in all_aliases:
                raise ValueError(f"Duplicate alias {record['alias']}")
            all_aliases.add(record["alias"])
        shards = deque()
        for group_index, ((active, context, _bps_vector), members) in enumerate(groups.items()):
            for start in range(0, len(members), max_group_motions):
                chunk = members[start : start + max_group_motions]
                signature = f"{source}_asset_{group_index:06d}_slice_{start // max_group_motions:06d}"
                for member in chunk:
                    member["asset_signature"] = signature
                shards.append({
                    "source": source,
                    "asset_signature": signature,
                    "active_articulation_usd": active,
                    "static_context_usd": context,
                    "records": chunk,
                })
        queues[source] = shards
    cohorts = []
    used = set()
    order = list(sources)
    cursor = 0
    while any(queues.values()):
        groups = []
        motion_count = 0
        idle = 0
        while len(groups) < num_envs and motion_count < max_motions and idle < len(order):
            source = order[cursor % len(order)]
            cursor += 1
            if not queues[source]:
                idle += 1
                continue
            next_group = queues[source][0]
            if motion_count + len(next_group["records"]) > max_motions:
                idle += 1
                continue
            groups.append(queues[source].popleft())
            motion_count += len(next_group["records"])
            idle = 0
        if not groups:
            raise ValueError("max_motions is smaller than one asset group's motion slice")
        for group in groups:
            for record in group["records"]:
                if record["alias"] in used:
                    raise ValueError(f"Motion repeated across cohorts: {record['alias']}")
                used.add(record["alias"])
        cohorts.append({"groups": groups, "motion_count": motion_count})
    if used != all_aliases:
        raise ValueError(f"Plan misses {len(all_aliases - used)} source motions")
    return {
        "contract": "unified_no_vof_plan_v1",
        "source_config": str(config_path.resolve()),
        "target_fps_hz": config["target_fps_hz"],
        "num_envs": num_envs,
        "max_motions": max_motions,
        "source_counts": {name: len(records) for name, records in sources.items()},
        "total_unique_motions": len(all_aliases),
        "active_usd_selection": active_usd_selection_report,
        "static_context_subdir": static_context_subdir or PAIRED_STATIC_CONTEXT_SUBDIR,
        "static_context_store_keys": {name: sum(int(r.get("static_context_subdir") not in (None, PAIRED_STATIC_CONTEXT_SUBDIR)) for r in sources.get(name, []))
                                      for name in sources},
        "cohorts": cohorts,
    }


def _bps_path(record: dict) -> Path:
    root = Path(record["pack"]) / "bps"
    key = record["source_motion_key"]
    exact = root / f"{key}.npy"
    if exact.is_file():
        return exact
    name = record["active_object"].replace("__", "_")
    named = root / f"{key}__{name}.npy"
    if name and named.is_file():
        return named
    candidates = list(root.glob(f"{key}__*.npy"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"{root}: no unambiguous active BPS for {key}")


def _link(target: Path, link: Path) -> None:
    if not target.is_file():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        raise FileExistsError(f"{link}: existing link points elsewhere")
    if link.exists():
        raise FileExistsError(link)
    link.symlink_to(target.resolve())


def _sample_active_panel_surface(usd_path: Path, count: int = 512) -> np.ndarray:
    """Sample the active object's mesh in its tracked panel frame, not scene space."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Cannot open active USD: {usd_path}")
    panel = next((prim for prim in stage.Traverse() if prim.GetName() == "panel"), None)
    if panel is None:
        raise ValueError(f"{usd_path}: missing tracked panel prim")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    panel_inverse = cache.GetLocalToWorldTransform(panel).GetInverse()
    triangles = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.GetPath().HasPrefix(panel.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        mesh_to_world = cache.GetLocalToWorldTransform(prim)
        vertices = np.asarray([
            tuple(panel_inverse.Transform(mesh_to_world.Transform(Gf.Vec3d(*point))))
            for point in points
        ], dtype=np.float64)
        cursor = 0
        for face_count in counts:
            face = indices[cursor : cursor + face_count]
            cursor += face_count
            for corner in range(1, face_count - 1):
                triangle = vertices[[face[0], face[corner], face[corner + 1]]]
                if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) > 1e-12:
                    triangles.append(triangle)
    if not triangles:
        raise ValueError(f"{usd_path}: no nondegenerate active-panel triangles")
    triangles = np.asarray(triangles, dtype=np.float64)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1
    )
    rng = np.random.default_rng(20261002)
    chosen = triangles[rng.choice(len(triangles), size=count, p=areas / areas.sum())]
    u = np.sqrt(rng.random((count, 1)))
    v = rng.random((count, 1))
    samples = (1.0 - u) * chosen[:, 0] + u * (1.0 - v) * chosen[:, 1] + u * v * chosen[:, 2]
    result = samples.astype(np.float32)
    if result.shape != (count, 3) or not np.isfinite(result).all():
        raise ValueError(f"{usd_path}: invalid surface samples")
    return result


def _apply_panel_mass(cohort: dict, output: Path, panel_mass_kg: float) -> None:
    """Point this cohort only at task-local USD copies with a new panel mass."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gear_sonic.rp2.panel_mass_override import write_panel_mass_copy, write_report

    copy_dir = output / "panel_mass_usd"
    reports = []
    for group in cohort["groups"]:
        source = Path(group["active_articulation_usd"])
        copy = copy_dir / f"{group['asset_signature']}.usd"
        reports.append({"asset_signature": group["asset_signature"], **write_panel_mass_copy(source, copy, panel_mass_kg)})
        group["active_articulation_usd"] = str(copy.resolve())
        for record in group["records"]:
            record["source_active_articulation_usd"] = record["active_articulation_usd"]
            record["active_articulation_usd"] = str(copy.resolve())
    write_report(copy_dir / "report.json", panel_mass_kg, reports)


# ---- object slot policy -----------------------------------------------------
# cycle: the historical allocation (groups[env_id % len(groups)], or the beta-capped weight
# redistribution when --object-slot-weights is given), byte-identical.
# motion_count (default for new cohorts): inside each source's FIXED budget (the slot count the
# default cycle gives that source) every object's share is proportional to the number of UNIQUE
# selected canonical motions (source::source_motion_key) in its asset groups; an object's slots are
# split over its groups by their motion members (coverage floor one slot per group). On a natural
# resume (tools/next_leg_layout.py) the same policy blends the current slots toward
# base x difficulty multiplier with eta and a per-leg bound; without a previous layout it IS the
# base (no multipliers, no old-slot baseline).
POLICY_CYCLE = "cycle"
POLICY_MOTION_COUNT = "motion_count"
POLICY_MOTION_COUNT_DIFFICULTY = "motion_count_difficulty"   # the next-leg helper's name for the same policy
MOTION_COUNT_POLICIES = (POLICY_MOTION_COUNT, POLICY_MOTION_COUNT_DIFFICULTY)
OBJECT_SLOT_POLICIES = (POLICY_CYCLE,) + MOTION_COUNT_POLICIES
DEFAULT_OBJECT_SLOT_POLICY = POLICY_MOTION_COUNT
DEFAULT_ETA = 0.10
DEFAULT_MAX_LEG_CHANGE_FRACTION = 0.10


def group_motion_count(group: dict) -> int:
    """Unique selected canonical motions of one asset group (never the slot count).

    Builder groups carry the selected ``records`` (counted by ``source::source_motion_key``,
    alias as the fallback key); manifest-derived groups (tools/next_leg_layout.py) carry
    ``member_count`` = the group's compatible aliases, once per group. A group without
    either is an ordinary error: nothing is assumed.
    """
    records = group.get("records")
    signature = group.get("asset_signature", "?")
    if records is not None:
        identities = set()
        for record in records:
            key = record.get("source_motion_key") or record.get("alias")
            if not key:
                raise ValueError(f"group {signature}: record without source_motion_key/alias")
            identities.add((str(record.get("source") or group.get("source")), str(key)))
        count = len(identities)
    elif group.get("member_count") is not None:
        count = int(group["member_count"])
    else:
        raise ValueError(
            f"group {signature}: neither records nor member_count; the {POLICY_MOTION_COUNT} policy needs the "
            "selected motion count of every group")
    if count <= 0:
        raise ValueError(f"group {signature}: selected motion count must be positive, got {count}")
    return count


def _multipliers_from_weights(keys: list[str], weights: dict[str, float]) -> dict[str, float]:
    """Legacy weight input under the motion_count policy: weight relative to the mean of the PROVIDED objects.

    Objects without a weight stay neutral (1.0); nothing is fabricated.
    """
    provided = {k: float(weights[k]) for k in keys if k in weights and float(weights[k]) > 0.0}
    if not provided:
        return {}
    mean = sum(provided.values()) / len(provided)
    return {k: v / mean for k, v in provided.items()}


def _previous_slots_for_source(previous: dict | None, source: str) -> dict[str, int]:
    """Previous slots of one source: nested ``{source: {object_key: n}}`` (preferred) or a flat ``{object_key: n}``."""
    if not previous:
        return {}
    nested = previous.get(source)
    if isinstance(nested, dict):
        return {str(k): int(v) for k, v in nested.items()}
    if any(isinstance(v, dict) for v in previous.values()):
        return {}  # nested form without this source: cold for this source
    return {str(k): int(v) for k, v in previous.items()}


def _allocate_motion_count_policy(groups: list[dict], num_envs: int, default_order: list[int], *, policy: str,
                                  beta: float, object_slot_weights: dict[str, float] | None,
                                  previous_object_slots: dict | None, difficulty_multipliers: dict[str, float] | None,
                                  eta: float, max_leg_change_fraction: float,
                                  previous_slot_assignment: list[Mapping[str, Any]] | None = None) -> tuple[list[int], dict[str, Any]]:
    import math
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gear_sonic.rp2.adaptive_sampling import allocate_bounded_slots

    count = len(groups)
    if num_envs < count:
        raise ValueError(
            f"{num_envs} slots cannot keep one slot per group for {count} groups; "
            f"the {policy} policy needs num_envs >= group count")
    eta = float(eta)
    if not (0.0 <= eta <= 1.0) or not math.isfinite(eta):
        raise ValueError(f"eta must be within [0, 1], got {eta}")
    max_leg_change_fraction = float(max_leg_change_fraction)
    if not (0.0 < max_leg_change_fraction <= 1.0) or not math.isfinite(max_leg_change_fraction):
        raise ValueError(f"max_leg_change_fraction must be within (0, 1], got {max_leg_change_fraction}")
    beta = float(beta)
    if not math.isfinite(beta) or beta < 1.0:
        raise ValueError(f"beta must be finite and >= 1, got {beta}")
    motion_counts = [group_motion_count(g) for g in groups]
    source_budget: dict[str, int] = defaultdict(int)
    for group_index in default_order:
        source_budget[groups[group_index]["source"]] += 1
    multipliers_in = {str(k): float(v) for k, v in (difficulty_multipliers or {}).items()}
    per_group = [0] * count
    report_sources: dict[str, Any] = {}
    any_changed = False
    all_cold = True
    for source in sorted(source_budget):
        indices = [i for i, group in enumerate(groups) if group["source"] == source]
        budget = int(source_budget[source])
        if budget < len(indices):
            raise ValueError(f"{source}: budget {budget} < {len(indices)} groups; cannot keep coverage")
        groups_by_key: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            groups_by_key[str(groups[i].get("object_key") or groups[i]["asset_signature"])].append(i)
        keys = sorted(groups_by_key)
        n_obj = len(keys)
        lower = [len(groups_by_key[k]) for k in keys]
        object_motions = [sum(motion_counts[i] for i in groups_by_key[k]) for k in keys]
        total_motions = sum(object_motions)
        base = [budget * m / total_motions for m in object_motions]
        # per-object cap around the object's OWN motion-count base: ceil(beta * base_o), never a uniform
        # share (an object holding 90 % of the selected motions keeps its 90 % baseline); the coverage
        # floor may exceed it for objects with many groups (reported as coverage overflow)
        caps = [min(budget, int(math.ceil(beta * b))) for b in base]
        coverage_overflow = [keys[j] for j in range(n_obj) if lower[j] > caps[j]]
        upper = [max(lower[j], caps[j]) for j in range(n_obj)]
        if sum(upper) < budget:
            raise ValueError(f"{source}: caps {upper} cannot fill budget {budget}")
        # multipliers: real-outcome multipliers of the next-leg helper, else derived from a legacy
        # weight file; unobserved objects stay 1.0; bounded to [1/beta, beta] around the base
        source_multipliers = {k: multipliers_in[k] for k in keys if k in multipliers_in}
        if not source_multipliers and object_slot_weights:
            source_multipliers = _multipliers_from_weights(keys, object_slot_weights)
        requested = [source_multipliers.get(k) for k in keys]
        for k, value in zip(keys, requested, strict=True):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{source}: difficulty multiplier of {k!r} must be finite and positive, got {value}")
        effective = [min(beta, max(1.0 / beta, float(v))) if v is not None else 1.0 for v in requested]
        raw_target = [b * m for b, m in zip(base, effective, strict=True)]
        target = [budget * t / sum(raw_target) for t in raw_target]
        previous = _previous_slots_for_source(previous_object_slots, source)
        cold = not any(k in previous for k in keys)
        rescaled = False
        # integer per-leg change bound: at most floor(max_leg_change_fraction * budget) slots moved per
        # object per leg, enforced in the FINAL integer allocator's bounds (intersection of the change
        # window with the coverage floor and the motion-base cap); a float clamp alone would not bound
        # the integers after normalization / rounding
        bound_slots = None if cold else int(math.floor(max_leg_change_fraction * budget))
        centers: list[int | None] = [None] * n_obj
        lo_bounds = list(lower)
        hi_bounds = list(upper)
        if cold:
            old: list[float | None] = [None] * n_obj
            desired = list(target)          # == base when no multiplier is known
        else:
            old_values = [float(previous.get(k, base[j])) for j, k in enumerate(keys)]
            old_total = sum(old_values)
            if old_total > 0.0 and abs(old_total - budget) > 1.0e-9:
                old_values = [v * budget / old_total for v in old_values]
                rescaled = True
            old = list(old_values)
            bound = max_leg_change_fraction * budget
            desired = []
            for j in range(n_obj):
                blended = (1.0 - eta) * old_values[j] + eta * target[j]
                desired.append(min(old_values[j] + bound, max(old_values[j] - bound, blended)))
                if keys[j] in previous:
                    center = int(round(old_values[j]))
                    centers[j] = center
                    lo_bounds[j] = max(lower[j], center - bound_slots)
                    hi_bounds[j] = min(upper[j], center + bound_slots)
                    if lo_bounds[j] > hi_bounds[j]:
                        raise ValueError(
                            f"{source}: change window of {keys[j]!r} [{center - bound_slots}, {center + bound_slots}] "
                            f"(previous {center}, bound {bound_slots} slots) does not intersect its feasible window "
                            f"[coverage floor {lower[j]}, cap {upper[j]}]; the current layout stays (no projection outside the bound)")
            if sum(lo_bounds) > budget or sum(hi_bounds) < budget:
                raise ValueError(
                    f"{source}: per-object windows {list(zip(keys, lo_bounds, hi_bounds, strict=True))} cannot fill budget {budget} "
                    f"(sum of lower bounds {sum(lo_bounds)}, sum of upper bounds {sum(hi_bounds)}); the current layout stays")
        if sum(desired) <= 0.0:
            raise ValueError(f"{source}: desired object shares have no mass")
        object_slots = allocate_bounded_slots([d / sum(desired) for d in desired], budget, lo_bounds, hi_bounds).tolist()
        for j in range(n_obj):
            if centers[j] is not None and abs(int(object_slots[j]) - centers[j]) > bound_slots:
                raise RuntimeError(f"{source}: final slots of {keys[j]!r} violate the change bound (internal error)")
        objects_report: dict[str, Any] = {}
        changed = False
        for j, key in enumerate(keys):
            members = groups_by_key[key]
            n_slots = int(object_slots[j])
            member_motions = [motion_counts[i] for i in members]
            split = allocate_bounded_slots(
                [m / sum(member_motions) for m in member_motions], n_slots, [1] * len(members), [n_slots] * len(members)
            ).tolist()
            for i, n in zip(members, split, strict=True):
                per_group[i] = int(n)
            old_int = None if old[j] is None else int(round(old[j]))
            previous_int = None if cold or key not in previous else int(previous[key])
            if previous_int is not None and previous_int != n_slots:
                changed = True
            center = centers[j]
            bound_active = (
                center is not None
                and abs(n_slots - center) >= bound_slots
                and abs(int(round(target[j])) - center) > bound_slots
            )
            objects_report[key] = {
                "groups": len(members),
                "motion_count": int(object_motions[j]),
                "base_slots": float(base[j]),
                "observed": requested[j] is not None,
                "multiplier_requested": None if requested[j] is None else float(requested[j]),
                "multiplier": float(effective[j]),
                "target_slots": float(target[j]),
                "previous_slots": previous_int,
                "old_slots_rescaled": None if old[j] is None else float(old[j]),
                "desired_slots": float(desired[j]),
                "new_slots": n_slots,
                "delta": None if previous_int is None else n_slots - previous_int,
                "change_center": center,
                "change_window": [int(lo_bounds[j]), int(hi_bounds[j])],
                "bounded_by_leg_change": bool(bound_active),
                "cap": int(upper[j]),
                "cap_from_base": int(caps[j]),
                "at_floor": n_slots <= lower[j],
                "at_cap": n_slots >= upper[j],
                "group_motion_counts": {groups[i]["asset_signature"]: int(motion_counts[i]) for i in members},
                "group_slots": {groups[i]["asset_signature"]: int(n) for i, n in zip(members, split, strict=True)},
                "old_slots": old_int,
            }
        final = [int(v) for v in object_slots]
        any_changed = any_changed or changed
        all_cold = all_cold and cold
        # objects of a PARTIALLY observed source that have no multiplier/weight keep the neutral 1.0
        # (reported like the legacy "missing keys" fallback); a fully cold source reports none
        observed_any = any(v is not None for v in requested)
        missing_keys = [k for k, v in zip(keys, requested, strict=True) if v is None] if observed_any else []
        report_sources[source] = {
            "mode": "object_slot_policy",
            "policy": policy,
            "budget_slots": budget,
            "groups": len(indices),
            "object_count": n_obj,
            "beta": beta,
            "eta": eta,
            "max_leg_change_fraction": max_leg_change_fraction,
            "cold_start": cold,
            "change_bound_slots": bound_slots,
            "change_bound_rounding": "floor(max_leg_change_fraction * budget_slots)",
            "object_change_windows": {k: [int(lo), int(hi)] for k, lo, hi in zip(keys, lo_bounds, hi_bounds, strict=True)},
            "previous_slots_rescaled_to_budget": rescaled,
            "motion_count_total": int(total_motions),
            # per-object caps (beta x the object's motion-count base); no scalar uniform cap in this policy
            "object_cap_configured": None,
            "object_caps": dict(zip(keys, (int(c) for c in caps), strict=True)),
            "object_upper_bounds": dict(zip(keys, (int(u) for u in upper), strict=True)),
            "object_lower_bound_groups": dict(zip(keys, lower, strict=True)),
            "coverage_forced_overflow_objects": coverage_overflow,
            "missing_object_keys": missing_keys,
            "final_object_slots": dict(zip(keys, final, strict=True)),
            "objects_at_configured_cap": [k for k, v, c in zip(keys, final, caps, strict=True) if v >= c],
            "max_object_slots": int(max(final)),
            "beta_cap_respected": bool(all(v <= c for v, c in zip(final, caps, strict=True))),
            "redistribution_freedom_slots": budget - sum(lower),
            "feasible": True,
            "allocation_changed": changed,
            "objects": objects_report,
        }
    order, placement = _slot_order(groups, per_group, num_envs, label="Policy", previous_slot_assignment=previous_slot_assignment)
    report = {
        "mode": "object_slot_policy",
        "policy": policy,
        "slot_placement": placement,
        "beta": beta,
        "eta": eta,
        "max_leg_change_fraction": max_leg_change_fraction,
        "cold_start": all_cold,
        "allocation_changed": any_changed,
        "rounding": "per source: object shares = motion count (x bounded multiplier; blended with the previous slots by eta and "
                    "a per-leg bound on a resume) -> allocate_bounded_slots (lower = groups of the object, upper = "
                    "max(lower, ceil(beta * motion-count base of the object))); object slots split over its groups by their "
                    "motion members (lower 1); largest remaining deficit first",
        "source_budgets_fixed_from_default_cycle": dict(sorted(source_budget.items())),
        "sources": report_sources,
        "slots_per_group": {groups[i]["asset_signature"]: per_group[i] for i in range(count)},
    }
    return order, report


def _load_object_slot_weights(path: Path | None) -> tuple[dict[str, float] | None, float | None]:
    """Return (weights, beta). ``beta`` is the cap carried by the export (default 3.0)."""
    if path is None:
        return None, None
    raw = json.loads(Path(path).read_text())
    weights = raw.get("weights", raw) if isinstance(raw, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{path}: expected a non-empty mapping object_key -> weight")
    if isinstance(raw, dict) and raw.get("available") is False:
        raise ValueError(f"{path}: export reports available=false; refusing stale redistribution input")
    out = {}
    for key, value in weights.items():
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{path}: weight for {key!r} must be finite and non-negative")
        out[str(key)] = value
    beta = float(raw.get("beta", 3.0)) if isinstance(raw, dict) else 3.0
    if not np.isfinite(beta) or beta < 1.0:
        raise ValueError(f"{path}: beta must be finite and >= 1")
    return out, beta


def _load_object_slot_policy_inputs(path: Path | None) -> dict[str, Any]:
    """The next-leg helper's policy block of a weights document (tools/next_leg_layout.py weights_document).

    Keys when present: ``policy``, ``eta``, ``max_leg_change_fraction``, ``difficulty_multipliers``
    (real-outcome multipliers per object key), ``previous_object_slots`` (current slots, keyed by
    source then object key, or flat). A plain ``{object_key: weight}`` file has none.
    """
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        return {}
    block: dict[str, Any]
    if isinstance(raw.get("policy"), dict):
        block = raw["policy"]                      # nested form (backwards compatible)
    elif isinstance(raw.get("policy"), str):
        block = raw                                # the document tools/next_leg_layout.py writes: top-level keys
    else:
        return {}
    out: dict[str, Any] = {}
    if block.get("policy") is not None:
        out["policy"] = str(block["policy"])
    if block.get("previous_manifest"):
        out["previous_manifest"] = str(block["previous_manifest"])
    for key in ("eta", "max_leg_change_fraction"):
        if block.get(key) is not None:
            out[key] = float(block[key])
    if isinstance(block.get("difficulty_multipliers"), dict):
        out["difficulty_multipliers"] = {str(k): float(v) for k, v in block["difficulty_multipliers"].items()}
    if isinstance(block.get("previous_object_slots"), dict):
        out["previous_object_slots"] = block["previous_object_slots"]
    return out


@functools.lru_cache(maxsize=65536)
def _canonical_directory(head: str) -> str:
    try:
        return os.path.realpath(head)
    except OSError:
        return head


BOUNDARY_SUPPORT_SUBDIR = "static_context_usd_inferred"
BOUNDARY_SUPPORT_MODES = ("inferred", "off")


def _pack_index_records(pack: Path, cache: dict) -> dict:
    """``motion_key -> index record`` of a pack (its own full index; loaded once per pack during a materialization)."""
    key = str(pack)
    if key not in cache:
        try:
            doc = json.loads((pack / "active_entity_manifest.json").read_text())
            cache[key] = {str(r.get("motion_key")): r for r in doc.get("records", [])}
        except (OSError, ValueError):
            cache[key] = {}
    return cache[key]


def _plan_record_support(args: tuple) -> dict:
    """Worker: classify one motion's start/end support (gear_sonic.rp2.inferred_support) and author its slabs.
    Returns the record update; any failure is reported, never raised (the motion keeps the pack's paired context)."""
    alias, key, objects_path, active_usd, context_usd, index_record, out_dir = args
    from gear_sonic.rp2.inferred_support import (
        authored_boxes_from_record, authored_boxes_from_usd, plan_boundary_support, record_local_aabb,
        write_support_context_usd,
    )
    import joblib
    try:
        boxes = authored_boxes_from_record((index_record or {}).get("static_context"))
        if not boxes and context_usd:
            boxes = authored_boxes_from_usd(context_usd)
        aabb = record_local_aabb(index_record or {}, active_usd)
        if aabb is None:
            return {"alias": alias, "status": "no_bounds"}
        payload = joblib.load(objects_path)
        plan = plan_boundary_support(key, payload, aabb[0], aabb[1], authored_boxes=boxes)
        result = {
            "alias": alias, "status": "ok",
            "decisions": {d.boundary: d.kind for d in plan.decisions},
            "authored_boxes": int(plan.authored_boxes), "slabs": len(plan.slabs),
            "merged": [list(m) for m in plan.merged_boundaries],
        }
        if plan.needs_slabs:
            usd_out = Path(out_dir) / f"{alias}.usd"
            write_support_context_usd(usd_out, plan.slabs, base_usd=context_usd or None)
            result["static_context_usd"] = str(usd_out)
            result["slab_boxes"] = [b.as_dict() for b in plan.slabs]
        return result
    except Exception as exc:  # noqa: BLE001 - reported per motion
        return {"alias": alias, "status": f"failed: {type(exc).__name__}: {exc}"}


def _apply_boundary_support_to_source(source: str, pack: Path, records: list[dict], raw_by_key: dict, *, mode: str,
                                      support_store: Path | None) -> dict:
    """Catalog-time boundary support for one source (gear_sonic/rp2/inferred_support.py; archived InterMimic rule):
    authored static boxes first; otherwise the object's lowest rotated-bounding-box point at the first/last frame with the
    hand-contact evidence decides floor / hand-held / unknown / inferred 0.5 x 0.5 x 0.06 m support; near-coincident
    start/end supports are merged. A motion that needs slabs gets ``<support_store>/<source>/<key>.usd`` (the pack's paired
    context referenced + the slab colliders tagged ``hoi:inferredSupport``) as its ``static_context_usd``. Because the
    static context is part of the asset-group key, such motions form their own slot groups and every slot's motions share
    one scene, which is what the runtime roster requires. Motions whose context already comes from a support store
    (``--static-context-subdir``) are left as they are. Returns the per-source summary."""
    summary = {"mode": mode, "motions": len(records)}
    if mode == "off":
        return summary
    if support_store is None:
        raise ValueError("boundary_support=inferred needs --support-store (default: <output-root>/static_context_usd_inferred)")
    out_dir = Path(support_store) / source
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    skipped_store = 0
    for record in records:
        if record.get("static_context_subdir") not in (None, PAIRED_STATIC_CONTEXT_SUBDIR):
            skipped_store += 1
            continue
        key = record["source_motion_key"]
        tasks.append((key, key, str(pack / "objects_with_masks" / f"{key}.pkl"), record["active_articulation_usd"],
                      record.get("static_context_usd"), raw_by_key.get(key), str(out_dir)))
    results: list[dict] = []
    if tasks:
        workers = max(1, min(len(tasks), int(os.environ.get("UNIFIED_SUPPORT_WORKERS", os.cpu_count() or 1))))
        try:
            import joblib
        except ImportError:  # pragma: no cover
            joblib = None
        if workers == 1 or joblib is None:
            results = [_plan_record_support(t) for t in tasks]
        else:
            results = joblib.Parallel(n_jobs=workers)(joblib.delayed(_plan_record_support)(t) for t in tasks)
    by_key = {r["alias"]: r for r in results}
    kinds: dict[str, int] = {}
    slabs = failed = no_bounds = merged = 0
    for record in records:
        r = by_key.get(record["source_motion_key"])
        if r is None:
            continue
        if r["status"] == "ok":
            for boundary, kind in r["decisions"].items():
                kinds[f"{boundary}:{kind}"] = kinds.get(f"{boundary}:{kind}", 0) + 1
            merged += len(r["merged"])
            record["boundary_support"] = {k: r[k] for k in ("decisions", "authored_boxes", "slabs", "merged")}
            if r.get("static_context_usd"):
                slabs += 1
                record["static_context_usd"] = r["static_context_usd"]
                record["static_context_subdir"] = BOUNDARY_SUPPORT_SUBDIR
                record["boundary_support"]["slab_boxes"] = r["slab_boxes"]
        elif r["status"] == "no_bounds":
            no_bounds += 1
            record["boundary_support"] = {"status": "no_bounds"}
        else:
            failed += 1
            record["boundary_support"] = {"status": r["status"]}
    summary.update(boundary_kinds=dict(sorted(kinds.items())), motions_with_slabs=slabs, near_coincident_merges=merged,
                   no_bounds=no_bounds, failed=failed, kept_store_context=skipped_store, store=str(out_dir))
    print(f"[boundary_support] {source}: {json.dumps(summary, sort_keys=True)}", flush=True)
    if failed or no_bounds:
        # A support decision that could not be made (missing pxr/asset, unreadable payload, no measurable bounds) is an
        # error of this build, not a silent "no table": the catalog is not returned and therefore never cached.
        problems = [f"{r['alias']}: {r['status']}" for r in results if r["status"] != "ok"]
        raise RuntimeError(f"{source}: boundary support failed for {len(problems)} motion(s), e.g. {problems[:3]}")
    return summary


def _count_boundary_kinds(records: list[dict]) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for r in records:
        for boundary, kind in ((r.get("boundary_support") or {}).get("decisions") or {}).items():
            kinds[f"{boundary}:{kind}"] = kinds.get(f"{boundary}:{kind}", 0) + 1
    return kinds


def _canonical_asset_path(path: Any) -> str:
    """One spelling for the same file reached through two mount spellings (realpath of the directory, cached per
    directory; the file itself is not stat'ed)."""
    text = str(path or "")
    if not text.startswith("/"):
        return text
    head, tail = os.path.split(text)
    return os.path.join(_canonical_directory(head), tail)


def _round_robin_slot_order(per_group: list[int], num_envs: int, *, label: str) -> list[int]:
    """The historical interleave: groups round-robin while they still have slots (contiguous per-rank slices stay mixed)."""
    count = len(per_group)
    remaining = list(per_group)
    order: list[int] = []
    while len(order) < num_envs:
        progressed = False
        for i in range(count):
            if remaining[i] > 0:
                order.append(i)
                remaining[i] -= 1
                progressed = True
        if not progressed:
            raise ValueError("Slot interleave exhausted groups before filling all slots")
    if len(order) != num_envs or sum(per_group) != num_envs:
        raise ValueError(f"{label} slot allocation does not fill num_envs exactly")
    return order


def stable_slot_order(groups: list[dict], per_group: list[int], num_envs: int,
                      previous_slot_assignment: list[Mapping[str, Any]]) -> tuple[list[int], dict[str, Any]]:
    """Slot order that keeps every previous slot's group while the group's NEW budget allows (user priority 2026-10-14).

    ``previous_slot_assignment[i]`` is the previous layout's slot record ``i`` (``source``, ``asset_signature``,
    ``active_articulation_usd``, ``static_context_usd``). A previous slot keeps its group when a current group has the
    same (source, asset_signature) AND the same canonical USD pair (a signature re-pointed at other assets is a
    different group) and that group still has quota; the lowest slot indices of a group are kept, so a shrinking
    group frees its HIGHEST slots and the few changed bindings concentrate in the tail ranks. Freed slots are then
    filled, ascending, by the remaining deficits in group order. Budgets (``per_group``) are untouched: this changes
    the slot ORDER only, so the number of re-bound env slots equals the half-L1 budget change instead of the
    thousands a round-robin rebuild shifts (370 -> 716 Generalist layout: 13678 re-bound by round-robin, 144 by this).
    Returns (order, placement report)."""
    count = len(groups)
    if sum(per_group) != num_envs:
        raise ValueError("stable slot order needs per-group counts that fill num_envs exactly")
    by_key: dict[tuple[str, str, str, str], int] = {}
    for index, group in enumerate(groups):
        key = (str(group.get("source")), str(group.get("asset_signature")),
               _canonical_asset_path(group.get("active_articulation_usd")), _canonical_asset_path(group.get("static_context_usd")))
        by_key.setdefault(key, index)
    quota = list(per_group)
    order: list[int | None] = [None] * num_envs
    freed: list[int] = []
    kept = 0
    previous_len = min(len(previous_slot_assignment), num_envs)
    for slot in range(previous_len):
        rec = previous_slot_assignment[slot]
        key = (str(rec.get("source")), str(rec.get("asset_signature")),
               _canonical_asset_path(rec.get("active_articulation_usd")), _canonical_asset_path(rec.get("static_context_usd")))
        index = by_key.get(key)
        if index is not None and quota[index] > 0:
            order[slot] = index
            quota[index] -= 1
            kept += 1
        else:
            freed.append(slot)
    freed.extend(range(previous_len, num_envs))
    fill = [index for index in range(count) for _ in range(quota[index])]
    if len(fill) != len(freed):
        raise ValueError(f"stable slot order: {len(fill)} deficit slots for {len(freed)} free positions")
    for slot, index in zip(freed, fill, strict=True):
        order[slot] = index
    final = [int(v) for v in order]  # type: ignore[arg-type]
    changed = [slot for slot in range(num_envs) if slot >= previous_len or order[slot] != by_key.get(
        (str(previous_slot_assignment[slot].get("source")), str(previous_slot_assignment[slot].get("asset_signature")),
         _canonical_asset_path(previous_slot_assignment[slot].get("active_articulation_usd")),
         _canonical_asset_path(previous_slot_assignment[slot].get("static_context_usd"))))]
    report = {
        "mode": "stable",
        "previous_slots": int(len(previous_slot_assignment)),
        "kept_slots": int(kept),
        "changed_slots": int(len(changed)),
        "first_changed_slot": changed[0] if changed else None,
        "last_changed_slot": changed[-1] if changed else None,
        "changed_slot_indices": changed,   # complete; materialize() turns it into exact per-rank counts and drops the list
    }
    return final, report


def _slot_order(groups: list[dict], per_group: list[int], num_envs: int, *, label: str,
                previous_slot_assignment: list[Mapping[str, Any]] | None) -> tuple[list[int], dict[str, Any]]:
    if previous_slot_assignment:
        return stable_slot_order(groups, per_group, num_envs, previous_slot_assignment)
    return _round_robin_slot_order(per_group, num_envs, label=label), {"mode": "round_robin"}


def allocate_cohort_slots(
    groups: list[dict],
    num_envs: int,
    object_slot_weights: dict[str, float] | None = None,
    *,
    beta: float | None = None,
    policy: str | None = None,
    previous_object_slots: dict | None = None,
    difficulty_multipliers: dict[str, float] | None = None,
    eta: float = DEFAULT_ETA,
    max_leg_change_fraction: float = DEFAULT_MAX_LEG_CHANGE_FRACTION,
    previous_slot_assignment: list[Mapping[str, Any]] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Return the group index for every slot plus an allocation report.

    ``policy`` = ``cycle`` / ``None`` is the historical interface below, byte-identical;
    ``motion_count`` (alias ``motion_count_difficulty``) is the object-slot policy documented at
    the top of this section (motion-count base inside fixed source budgets, bounded difficulty
    multipliers, eta blend with ``previous_object_slots`` and a per-leg bound on a resume;
    cold = base). The default for NEW cohorts is chosen by ``materialize`` / the CLI
    (``--object-slot-policy``, default ``motion_count``); this function keeps ``None`` = historical
    so existing callers and tests stay byte-identical.

    Default (no weights): ``groups[env_id % len(groups)]`` -- byte-identical to
    the historical cycling allocation.

    With weights (``object_key -> weight`` from the stats export, re-projected
    on THIS cohort's object set):

    * every source keeps exactly the slot budget B_s it receives from the
      default cycle (source budgets fixed);
    * inside a source, slots are allocated at the OBJECT level with the existing
      ``allocate_bounded_slots`` (largest remaining deficit first): lower bound
      = number of groups of the object (coverage: one slot per group, every
      selected sequence stays reachable), upper bound = integer beta cap
      ``cap_o = max(groups_o, ceil(beta * B_s / n_objects_s))``; keys absent
      from the weights get the source median weight (reported);
    * an object's slots are then split over its groups (lower 1, uniform);
    * feasibility: B_s >= groups_s (else error) and sum(cap_o) >= B_s (holds
      for beta >= 1). The report lists the caps, the final integer slots per
      object, and whether any object hit its cap, so the integer result can be
      checked against beta directly (normalized weights are not that proof).
    * slots are interleaved round-robin over groups so contiguous per-rank
      slices stay source- and object-mixed.
    """
    count = len(groups)
    if count <= 0 or num_envs <= 0:
        raise ValueError("Need at least one group and one slot")
    default_order = [env_id % count for env_id in range(num_envs)]
    if policy is not None and str(policy) not in OBJECT_SLOT_POLICIES:
        raise ValueError(f"unknown object slot policy {policy!r}; expected one of {OBJECT_SLOT_POLICIES}")
    if policy is not None and str(policy) in MOTION_COUNT_POLICIES:
        return _allocate_motion_count_policy(
            groups, num_envs, default_order, policy=str(policy), beta=3.0 if beta is None else float(beta),
            object_slot_weights=object_slot_weights, previous_object_slots=previous_object_slots,
            difficulty_multipliers=difficulty_multipliers, eta=eta, max_leg_change_fraction=max_leg_change_fraction,
            previous_slot_assignment=previous_slot_assignment)
    if object_slot_weights is None:
        return default_order, {"mode": "default_cycle"}
    if beta is None:
        beta = 3.0
    if num_envs < count:
        raise ValueError(
            f"{num_envs} slots cannot keep one slot per group for {count} groups; "
            "object slot weighting needs num_envs >= group count"
        )
    import math
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gear_sonic.rp2.adaptive_sampling import allocate_bounded_slots

    source_budget: dict[str, int] = defaultdict(int)
    for group_index in default_order:
        source_budget[groups[group_index]["source"]] += 1
    per_group = [0] * count
    report_sources = {}
    for source in sorted(source_budget):
        indices = [i for i, group in enumerate(groups) if group["source"] == source]
        budget = int(source_budget[source])
        if budget < len(indices):
            raise ValueError(f"{source}: budget {budget} < {len(indices)} groups; cannot keep coverage")
        groups_by_key: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            groups_by_key[str(groups[i].get("object_key") or groups[i]["asset_signature"])].append(i)
        keys = sorted(groups_by_key)
        provided = [object_slot_weights[k] for k in keys if k in object_slot_weights]
        fallback = float(np.median(provided)) if provided else 1.0
        missing = [k for k in keys if k not in object_slot_weights]
        weights = np.asarray([object_slot_weights.get(k, fallback) for k in keys], dtype=np.float64)
        if weights.sum() <= 0.0:
            weights = np.ones_like(weights)
        n_obj = len(keys)
        lower = [len(groups_by_key[k]) for k in keys]
        # Configured integer beta cap per object. Coverage (one slot per group)
        # may exceed it for objects with many groups; those objects keep every
        # selected sequence reachable and are reported as coverage overflow
        # instead of silently redefining beta.
        cap_configured = min(budget, int(math.ceil(float(beta) * budget / n_obj)))
        coverage_overflow = [keys[j] for j in range(n_obj) if lower[j] > cap_configured]
        upper = [max(lower[j], cap_configured) for j in range(n_obj)]
        if sum(upper) < budget:
            raise ValueError(f"{source}: caps {upper} cannot fill budget {budget}")
        object_slots = allocate_bounded_slots(weights / weights.sum(), budget, lower, upper).tolist()
        for key, n_slots in zip(keys, object_slots, strict=True):
            members = groups_by_key[key]
            split = allocate_bounded_slots(
                np.ones(len(members)) / len(members), int(n_slots), [1] * len(members), [int(n_slots)] * len(members)
            ).tolist()
            for i, n in zip(members, split, strict=True):
                per_group[i] = int(n)
        final = [int(v) for v in object_slots]
        report_sources[source] = {
            "budget_slots": budget,
            "groups": len(indices),
            "objects": n_obj,
            "beta": float(beta),
            "object_cap_configured": cap_configured,
            "object_lower_bound_groups": dict(zip(keys, lower, strict=True)),
            "coverage_forced_overflow_objects": coverage_overflow,
            "fallback_weight_for_missing_keys": fallback,
            "missing_object_keys": missing,
            "final_object_slots": dict(zip(keys, final, strict=True)),
            "objects_at_configured_cap": [k for k, v in zip(keys, final, strict=True) if v >= cap_configured],
            "max_object_slots": int(max(final)),
            # True only against the CONFIGURED cap; coverage overflow makes it False.
            "beta_cap_respected": bool(all(v <= cap_configured for v in final)),
            "redistribution_freedom_slots": budget - sum(lower),
        }
    order, placement = _slot_order(groups, per_group, num_envs, label="Weighted", previous_slot_assignment=previous_slot_assignment)
    report = {
        "mode": "object_slot_weights",
        "slot_placement": placement,
        "beta": float(beta),
        "rounding": "allocate_bounded_slots at object level (lower = groups of the object, upper = integer beta cap), then uniform split over the object's groups (lower 1); largest remaining deficit first, deterministic tie-break",
        "source_budgets_fixed_from_default_cycle": dict(sorted(source_budget.items())),
        "sources": report_sources,
        "slots_per_group": {groups[i]["asset_signature"]: per_group[i] for i in range(count)},
    }
    return order, report


def load_previous_slot_assignment(previous_manifest: Path | str | None) -> list[dict[str, Any]] | None:
    """The previous layout's per-slot records (source, asset_signature, USD pair) for the stable slot order; ``None``
    when no previous manifest is given or it cannot be read (cold placement, never a failure)."""
    if not previous_manifest:
        return None
    try:
        doc = json.loads(Path(previous_manifest).read_text())
        slots = doc.get("slots") if isinstance(doc, dict) else None
        if not isinstance(slots, list) or not slots:
            return None
        return [{k: rec.get(k) for k in ("source", "asset_signature", "active_articulation_usd", "static_context_usd")}
                for rec in slots if isinstance(rec, dict)]
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"[build_unified_no_vof] previous layout {previous_manifest} not usable for slot placement ({type(exc).__name__}: {exc}); cold placement",
              file=sys.stderr)
        return None


def materialize_allocation(groups: list[dict], num_envs: int, object_slot_weights_path: Path | None, *,
                           object_slot_policy: str | None = None,
                           previous_manifest: Path | str | None = None) -> tuple[list[int], dict[str, Any], str]:
    """The slot allocation of a cohort being materialized: ``(order, report, policy)``.

    Consumes the real ``--object-slot-weights`` document (weights + beta, and the next-leg
    helper's policy / eta / max_leg_change_fraction / difficulty_multipliers /
    previous_object_slots) exactly as the helper predicted it. Precedence of the policy name:
    explicit CLI value > the document's policy > the new-cohort default (``motion_count``);
    ``cycle`` is the historical allocation (weights -> beta-capped redistribution).
    """
    weights, beta = _load_object_slot_weights(object_slot_weights_path)
    policy_inputs = _load_object_slot_policy_inputs(object_slot_weights_path)
    policy = str(object_slot_policy or policy_inputs.get("policy") or DEFAULT_OBJECT_SLOT_POLICY)
    if policy not in OBJECT_SLOT_POLICIES:
        raise ValueError(f"unknown object slot policy {policy!r}; expected one of {OBJECT_SLOT_POLICIES}")
    previous_slots = load_previous_slot_assignment(previous_manifest or policy_inputs.get("previous_manifest"))
    if policy == POLICY_CYCLE:
        order, report = allocate_cohort_slots(groups, num_envs, weights, beta=beta, previous_slot_assignment=previous_slots)
    else:
        order, report = allocate_cohort_slots(
            groups, num_envs, weights, beta=beta, policy=policy,
            previous_object_slots=policy_inputs.get("previous_object_slots"),
            difficulty_multipliers=policy_inputs.get("difficulty_multipliers"),
            eta=float(policy_inputs.get("eta", DEFAULT_ETA)),
            max_leg_change_fraction=float(policy_inputs.get("max_leg_change_fraction", DEFAULT_MAX_LEG_CHANGE_FRACTION)),
            previous_slot_assignment=previous_slots)
    return order, report, policy


def materialize(
    plan: dict,
    cohort_id: int,
    output_root: Path,
    panel_mass_kg: float | None = None,
    *,
    world_size: int = 1,
    object_slot_weights_path: Path | None = None,
    track_source_stats: bool = False,
    object_slot_policy: str | None = None,
    previous_manifest: Path | str | None = None,
) -> Path:
    """Materialize one cohort: link each record's payload files and assets under the cohort directory and write
    the manifest. Boundary support (floor / authored surface / hand-held / inferred small slab) is decided per motion
    here from the pack payload and authored into the cohort's own static-context files (``boundary_support``). No per-record payload is loaded here (user decision 2026-10-14: no runtime check PPO does not
    need): a record's FPS comes from the promoted pack index and is compared with the source declaration when the
    catalog loads, and the motion loader reads the clip FPS itself when it loads the clips."""
    cohort = plan["cohorts"][cohort_id]
    output = output_root / f"cohort_{cohort_id:04d}"
    if world_size <= 0 or plan["num_envs"] % world_size:
        raise ValueError(f"num_envs={plan['num_envs']} must be divisible by world_size={world_size}")
    if panel_mass_kg is not None:
        _apply_panel_mass(cohort, output, panel_mass_kg)
    records = [record for group in cohort["groups"] for record in group["records"]]
    for record in records:
        pack = Path(record["pack"])
        key = record["source_motion_key"]
        alias = record["alias"]
        robot = pack / "robot" / f"{key}.pkl"
        objects = pack / "objects_with_masks" / f"{key}.pkl"
        _link(robot, output / "robot" / f"{alias}.pkl")
        _link(objects, output / "objects_with_masks" / f"{alias}.pkl")
        # The generic training launcher requires these legacy directories during its
        # preflight even though the cohort scene resolves both USDs from the
        # manifest, and does not spawn the legacy rigid Object asset.
        _link(Path(record["active_articulation_usd"]), output / "object_usd" / f"{alias}.usd")
        _link(Path(record["static_context_usd"]), output / "static_context_usd_grail_paired" / f"{alias}.usd")
        bps = Path(record["bps_path"])
        _link(bps, output / "bps" / f"{alias}.npy")
    first_pack = Path(records[0]["pack"])
    _link(first_pack / "bps" / "_basis.npy", output / "bps" / "_basis.npy")
    support_summary = {
        "contract": "inferred_boundary_support_v1",
        "motions_with_slabs": sum(1 for r in records if (r.get("boundary_support") or {}).get("slabs")),
        "boundary_kinds": dict(sorted(_count_boundary_kinds(records).items())),
        "note": "decided per motion when the source catalog was loaded (tools/build_unified_no_vof.py --boundary-support); "
                "slab motions carry static_context_subdir=static_context_usd_inferred",
    }
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "object_usd_grail_paired").mkdir(parents=True, exist_ok=True)
    slots = []
    groups = cohort["groups"]
    surface_dir = output / "active_surface"
    surface_dir.mkdir(parents=True, exist_ok=True)
    import os

    # Same file (hard link / same inode) -> identical deterministic samples, so
    # each distinct file is sampled once; distinct files are sampled in
    # parallel worker processes (same function, same seed, same result).
    pending: dict[tuple[int, int], str] = {}
    for group in groups:
        surface_path = surface_dir / f"{group['asset_signature']}.npy"
        if not surface_path.exists():
            stat = os.stat(group["active_articulation_usd"])
            pending.setdefault((stat.st_dev, stat.st_ino), group["active_articulation_usd"])
    surface_by_inode: dict[tuple[int, int], np.ndarray] = {}
    if pending:
        workers = max(1, min(len(pending), int(os.environ.get("UNIFIED_SURFACE_WORKERS", os.cpu_count() or 1))))
        keys = list(pending)
        try:
            import joblib  # optional parallelism for the surface sampling; the module was used without an import
        except ImportError:  # pragma: no cover - serial fallback
            joblib = None
        if workers == 1 or joblib is None:
            results = [_sample_active_panel_surface(Path(pending[key])) for key in keys]
        else:
            results = joblib.Parallel(n_jobs=workers)(
                joblib.delayed(_sample_active_panel_surface)(Path(pending[key])) for key in keys
            )
        surface_by_inode = dict(zip(keys, results, strict=True))
    for group in groups:
        surface_path = surface_dir / f"{group['asset_signature']}.npy"
        if not surface_path.exists():
            stat = os.stat(group["active_articulation_usd"])
            np.save(surface_path, surface_by_inode[(stat.st_dev, stat.st_ino)])
        surface = np.load(surface_path)
        if surface.shape != (512, 3) or not np.isfinite(surface).all():
            raise ValueError(f"Invalid active surface: {surface_path}")
        group["active_surface_path"] = str(surface_path)
    for group in groups:
        keys = sorted({str(record.get("object_key", "")) for record in group["records"]})
        if len(keys) == 1:
            group["object_key"] = keys[0]
        else:
            # Grouping is by exact asset; records that share the asset but
            # resolved different labels (e.g. empty active_object fell back to
            # the motion key) take one asset-derived label so planning and
            # materialization agree. Grouping itself is unchanged.
            group["object_key"] = f"{group['source']}/asset:{Path(group['active_articulation_usd']).stem}"
            for record in group["records"]:
                record["object_key"] = group["object_key"]
    slot_group_ids, allocation, policy = materialize_allocation(
        groups, plan["num_envs"], object_slot_weights_path, object_slot_policy=object_slot_policy,
        previous_manifest=previous_manifest)
    placement = allocation.get("slot_placement") if isinstance(allocation, dict) else None
    if isinstance(placement, dict) and placement.get("mode") == "stable":
        per_rank = plan["num_envs"] // int(world_size)
        by_rank: dict[str, int] = {}
        for slot in placement.get("changed_slot_indices") or []:
            by_rank[str(int(slot) // per_rank)] = by_rank.get(str(int(slot) // per_rank), 0) + 1
        placement["changed_by_rank"] = by_rank          # exact: computed from the complete list
        placement.pop("changed_slot_indices", None)   # the per-rank counts are the record; indices stay derivable from the two manifests
    for env_id, group_id in enumerate(slot_group_ids):
        group = groups[group_id]
        aliases = [record["alias"] for record in group["records"]]
        slots.append({
            "local_env_id": env_id,
            "source": group["source"],
            "asset_signature": group["asset_signature"],
            "object_key": group["object_key"],
            "active_articulation_usd": group["active_articulation_usd"],
            "static_context_usd": group["static_context_usd"],
            "active_surface_path": group["active_surface_path"],
            "compatible_motion_aliases": aliases,
            "representative_alias": aliases[0],
        })
    manifest = {
        "contract": "unified_no_vof_cohort_v1",
        "num_envs": plan["num_envs"],
        "world_size": int(world_size),
        "num_envs_per_rank": plan["num_envs"] // int(world_size),
        "target_fps_hz": 50,
        "cohort_id": cohort_id,
        "cohort_count": len(plan["cohorts"]),
        "catalog_total_unique_motions": plan["total_unique_motions"],
        "catalog_source_counts": dict(plan["source_counts"]),
        "sampling_plan": {"sources": {name: {} for name in plan["source_counts"]}},
        "slot_allocation": allocation,
        # Which active-USD store each source was resolved to (directory choice only); the
        # launcher's reuse check compares it with the current resolution so an older uniform
        # cohort is never silently reused once the tiered default applies.
        "active_usd_selection": dict(plan.get("active_usd_selection", {})),
        "static_context_subdir": plan.get("static_context_subdir", PAIRED_STATIC_CONTEXT_SUBDIR),
        "static_context_store_keys": dict(plan.get("static_context_store_keys", {})),
        "boundary_support": support_summary,
        "records": records,
        "slots": slots,
    }
    manifest["object_slot_policy"] = policy
    if object_slot_weights_path is not None:
        manifest["object_slot_weights_path"] = str(Path(object_slot_weights_path).resolve())
    if track_source_stats:
        manifest["track_source_stats"] = True
    if panel_mass_kg is not None:
        manifest["panel_mass_override_kg"] = panel_mass_kg
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing == manifest:
            return manifest_path
        if _same_cohort_ignoring_annotations(existing, manifest):
            # A cohort materialized by the earlier builder (no object_key /
            # allocation annotations) describes the same slots and motions;
            # keep it byte-identical so resumed runs see an unchanged roster.
            print(f"Reusing existing equivalent cohort manifest {manifest_path}")
            return manifest_path
        raise FileExistsError(f"{manifest_path}: different cohort already materialized")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


_ANNOTATION_KEYS = (
    "num_envs_per_rank", "cohort_count", "catalog_total_unique_motions", "catalog_source_counts",
    "slot_allocation", "object_slot_weights_path", "object_slot_policy",
)


def _same_cohort_ignoring_annotations(existing: dict, manifest: dict) -> bool:
    """True when both manifests describe identical slots/records/world size."""
    def _strip(doc: dict) -> dict:
        out = {k: v for k, v in doc.items() if k not in _ANNOTATION_KEYS and k not in ("records", "slots")}
        # promoted_revision is a provenance label, not a roster fact; a pack
        # re-tag must not refuse an otherwise identical cohort.
        out["records"] = [
            {k: v for k, v in r.items() if k not in ("object_key", "promoted_revision")}
            for r in doc.get("records", [])
        ]
        out["slots"] = [{k: v for k, v in s.items() if k != "object_key"} for s in doc.get("slots", [])]
        return out
    return _strip(existing) == _strip(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path(__file__).resolve().parents[1] / "configs/unified_no_vof_sources.json")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-motions", type=int, default=512)
    parser.add_argument("--max-group-motions", type=int, default=16)
    parser.add_argument("--limit-per-source", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cohort", type=int)
    parser.add_argument("--panel-mass-kg", type=float, help="Opt-in: task-local USD copies with this active-panel mass")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Ranks that will share this cohort; --num-envs is the TOTAL slot count (per-rank x world-size)")
    parser.add_argument("--object-slot-weights", type=Path,
                        help="Opt-in JSON {object_key: weight}; redistributes slots inside each source's fixed budget")
    parser.add_argument("--sources-cache", type=Path,
                        help="Opt-in JSON cache of loaded source records (keyed on the source config content)")
    parser.add_argument("--track-source-stats", action="store_true",
                        help="Opt-in: the motion library tracks source::motion_key identities for this cohort")
    parser.add_argument("--active-usd-selection", default=None,
                        help="Active-USD store for EVERY source: tiered (default: the tiered store when the pack has it, "
                             "else native) | native | <directory name>; overrides per-source 'active_usd_subdir'")
    parser.add_argument("--static-context-subdir", default=None,
                        help="Optional static-context store: a directory NAME inside each pack or an absolute directory; a "
                             "motion found there uses that USD (inferred boundary support, tools/rp2_build_inferred_support_context.py), "
                             "any other motion keeps the pack's paired context")
    # The command line (the real entry: train_unified_no_vof.sh, the layout producer) defaults to inferred; the Python API
    # defaults to off so in-process callers without a support store (tests, helpers) keep the plain catalog.
    parser.add_argument("--boundary-support", choices=list(BOUNDARY_SUPPORT_MODES), default="inferred",
                        help="Per-motion boundary support at materialization (default inferred: authored surface first, else the "
                             "InterMimic small-slab rule from the object's lowest bounding-box point and hand-contact evidence; "
                             "slabs authored under the cohort's static_context_usd_inferred/ and linked as the motion's static context)")
    parser.add_argument("--support-store", type=Path, default=None,
                        help="where inferred support contexts are authored (<store>/<source>/<key>.usd); default "
                             "<output-root>/static_context_usd_inferred, else next to --sources-cache")
    parser.add_argument("--previous-manifest", type=Path, default=None,
                        help="previous layout manifest: slots keep their previous group while the new budgets allow (stable slot order)")
    parser.add_argument("--object-slot-policy", choices=list(OBJECT_SLOT_POLICIES), default=None,
                        help=f"Object slot policy of a NEW cohort (default {DEFAULT_OBJECT_SLOT_POLICY}: object shares inside "
                             "each source's fixed budget proportional to its selected motion count; cycle = the historical "
                             "allocation, byte-identical). A next-leg weights document may carry its own policy block.")
    args = parser.parse_args()
    support_store = args.support_store
    if support_store is None and args.boundary_support != "off":
        if args.output_root is not None:
            support_store = Path(args.output_root) / BOUNDARY_SUPPORT_SUBDIR
        elif args.sources_cache is not None:
            support_store = Path(args.sources_cache).parent / BOUNDARY_SUPPORT_SUBDIR
    plan = build_plan(args.sources, num_envs=args.num_envs, max_motions=args.max_motions,
                      max_group_motions=args.max_group_motions, limit_per_source=args.limit_per_source,
                      sources_cache=args.sources_cache, active_usd_selection=args.active_usd_selection,
                      static_context_subdir=args.static_context_subdir, boundary_support=args.boundary_support,
                      support_store=support_store)
    print(json.dumps({k: v for k, v in plan.items() if k != "cohorts"} | {"cohort_count": len(plan["cohorts"])}, indent=2))
    if args.cohort is not None:
        if args.output_root is None:
            parser.error("--cohort requires --output-root")
        manifest = materialize(
            plan, args.cohort, args.output_root, args.panel_mass_kg,
            world_size=args.world_size, object_slot_weights_path=args.object_slot_weights,
            track_source_stats=args.track_source_stats, object_slot_policy=args.object_slot_policy,
            previous_manifest=args.previous_manifest,
        )
        print(f"Materialized {manifest}")


if __name__ == "__main__":
    main()
