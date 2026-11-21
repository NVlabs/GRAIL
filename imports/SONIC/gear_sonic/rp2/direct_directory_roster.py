"""Per-rank joint-roster views computed in memory from a delivered pack directory.

No manifest, no materialization, no symlink cohort, no hash/READY/certificate: the
pack directory (``robot/``, ``objects_with_masks/``, ``bps/``, ``active_articulation_usd/``,
``static_context_usd_grail_paired/`` and the optional ``meta/``) is the input and the
runtime view is derived from its listing at every start. The view carries exactly the
key set of ``resolve_unified_cohort`` (gear_sonic/rp2/unified_cohort.py) so every
consumer (env cfg slot binding, motion command, mechanism resistance, physics
overrides, ``build_asset_compatible_motion_groups``) is unchanged.

Rules (docs/DIRECT_DIRECTORY_ROSTER.md):

* motion key == robot pkl stem (MotionLib rule, motion_lib_base.py: recursive ``*.pkl``
  under ``motion_file`` minus ``metadata.pkl``); every companion is required by stem;
* asset groups use the materialization key (active USD realpath, static-context USD
  realpath, BPS vector) but are ORDERED by their first sorted member stem, never by
  realpath strings (login-node and container realpaths differ in alias spelling);
* slots per group: lower-bounded proportional to the group weight (default: motion
  count), floor one slot per group (coverage), largest remainder, signature tie-break;
  ``group_weight`` is the only hook left for the sampling owner's per-group weights;
* ranks: exactly ``num_envs`` slots each, deterministic min-load placement with integer
  (exact ``Fraction``) loads, groups split across ranks only when unavoidable;
* every decision is a pure function of directory listings and file contents that are
  identical on all ranks; an IO or parse error on an existing meta file is an error,
  never a rank-local fallback.
"""

from __future__ import annotations

import glob
import math
import os
import pickle
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gear_sonic.rp2.active_usd_resolver import (
    NATIVE_SUBDIR,
    ROSTER_ENV_SELECTION,
    describe_active_usd_selection,
    resolve_active_usd_subdir,
)
from gear_sonic.rp2.unified_cohort import (
    ZERO_RESISTANCE,
    rank_interleave_enabled,
    slot_repeat_factor,
)

ROSTER_KIND = "direct_directory_v1"
LAYOUT_CONTRACT = "direct_directory_layout_v1"
# In-memory identity contract consumed by the sampling owner's command/MotionLib consumer (no manifest file):
# one record per alias of the WHOLE layout, identical on every rank. Fields mirror the unified cohort manifest
# records that MotionLib.init_joint_source_tracking already parses (alias / source / source_motion_key /
# source_sample_fps_hz / world_size at the view level) plus the real group identity. fps is None when
# meta/<stem>.pkl is absent (never a filled default); the consumer derives it from the robot payload.
IDENTITY_CONTRACT = "direct_roster_identity_v1"
# The active-USD store is checked through the shared resolver (tiered store if present, else native).
REQUIRED_ENTRIES = (
    "robot",
    "objects_with_masks",
    "bps/_basis.npy",
    "static_context_usd_grail_paired",
)
# Clip-length report thresholds (frames): clips shorter than each value are counted.
SHORT_CLIP_FRAME_THRESHOLDS = (2, 5, 10, 30, 60)
EXTRA_SLOT_LOG_LIMIT = 16
_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_BUDGET_SUFFIX_RE = re.compile(r"^(.*)@(\d+)$")
_ACTIVE_SUBDIR_SUFFIX_RE = re.compile(r"^(.*?):active_usd_subdir=([A-Za-z0-9_.-]+)$")
_USD_SUFFIXES = (".usd", ".usda")

GroupWeight = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class DirectSource:
    name: str
    root: str
    budget_slots: int | None = None
    # Active-USD store selection (gear_sonic/rp2/active_usd_resolver.py): None = default (the tiered store when
    # the pack has it, else native); "native"; or an explicit directory name inside the pack.
    active_usd_subdir: str | None = None


# --------------------------------------------------------------------------- spec


def normalize_direct_spec(value: Any) -> str | None:
    """``None``/``''``/``[]`` -> ``None``; a list of items -> one comma-joined string."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__"):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items) or None
    text = str(value).strip()
    return text or None


def parse_direct_sources(spec: Any) -> list[DirectSource]:
    """``name=path[@slots][,name=path[@slots]]`` -> sources with realpath roots."""
    text = normalize_direct_spec(spec)
    if not text:
        raise ValueError(
            "JOINT_ROSTER_DIRECT_DIRS is empty; expected name=path[@slots][,name=path[@slots]]"
        )
    sources: list[DirectSource] = []
    seen: set[str] = set()
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"direct roster item {item!r} is not name=path[@slots]")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        budget: int | None = None
        active_usd_subdir: str | None = None
        match = _ACTIVE_SUBDIR_SUFFIX_RE.match(path)
        if match:
            path, active_usd_subdir = match.group(1).strip(), match.group(2)
        if active_usd_subdir is None:
            env_selection = os.environ.get(ROSTER_ENV_SELECTION, "").strip()
            active_usd_subdir = env_selection or None
        match = _BUDGET_SUFFIX_RE.match(path)
        if match:
            path, budget = match.group(1).strip(), int(match.group(2))
            if budget <= 0:
                raise ValueError(f"direct roster source {name}: @slots must be positive, got {budget}")
        if not _SOURCE_NAME_RE.match(name):
            raise ValueError(
                f"direct roster source name {name!r} must match [A-Za-z0-9_.-]+ (item {item!r})"
            )
        if name in seen:
            raise ValueError(f"direct roster source name {name!r} given twice")
        if not path:
            raise ValueError(f"direct roster source {name}: empty path")
        root = os.path.realpath(os.path.expanduser(path))
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"direct roster source {name}: {path} is not a directory (realpath {root})"
            )
        for entry in REQUIRED_ENTRIES:
            if not os.path.exists(os.path.join(root, entry)):
                raise FileNotFoundError(
                    f"direct roster source {name}: missing {entry} under {root}"
                )
        active_subdir = resolve_active_usd_subdir(root, active_usd_subdir)  # explicit missing dir -> ValueError
        if not os.path.isdir(os.path.join(root, active_subdir)):
            raise FileNotFoundError(
                f"direct roster source {name}: missing {active_subdir} (active-USD store) under {root}"
            )
        seen.add(name)
        sources.append(DirectSource(name=name, root=root, budget_slots=budget, active_usd_subdir=active_usd_subdir))
    if not sources:
        raise ValueError("JOINT_ROSTER_DIRECT_DIRS holds no name=path item")
    return sources


# ------------------------------------------------------------------------ catalog


def _listing(path: str) -> set[str]:
    return set(os.listdir(path)) if os.path.isdir(path) else set()


def _usd_name(names: set[str], stem: str) -> str | None:
    for suffix in _USD_SUFFIXES:
        candidate = f"{stem}{suffix}"
        if candidate in names:
            return candidate
    return None


def _fps_fraction(value: Any) -> Fraction:
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {value!r}")
    return Fraction(repr(fps))


def _read_meta_record(source_name: str, path: str, stem: str) -> dict[str, Any]:
    """Plain-pickle ``{stem: {...}}``; any IO/parse/content problem is an error."""
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - every failure class is reported the same way
        raise ValueError(
            f"direct roster source {source_name}: cannot read meta/{stem}.pkl "
            f"({type(exc).__name__}: {exc}); meta/ is present for every motion, so every rank "
            "must read the same durations and labels (no rank-local fallback)"
        ) from exc
    record = payload.get(stem) if isinstance(payload, dict) else None
    if record is None and isinstance(payload, dict) and len(payload) == 1:
        record = next(iter(payload.values()))
    if not isinstance(record, dict):
        raise ValueError(
            f"direct roster source {source_name}: meta/{stem}.pkl is not a {{stem: {{...}}}} record"
        )
    try:
        num_frames = int(record["num_frames"])
        fps = _fps_fraction(record["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"direct roster source {source_name}: meta/{stem}.pkl lacks a valid num_frames/fps "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if num_frames <= 0:
        raise ValueError(
            f"direct roster source {source_name}: meta/{stem}.pkl has num_frames={num_frames}"
        )
    label: str | None = None
    for field in ("object_name", "active_object"):
        value = record.get(field)
        if value is not None and str(value).strip():
            label = str(value).strip()
            break
    return {"num_frames": num_frames, "fps": fps, "label": label}


def _clip_length_summary(records: Mapping[str, dict[str, Any]], balance: str) -> dict[str, Any]:
    if balance != "duration_s":
        return {"available": False, "reason": "count balance (no complete meta/)"}
    frames = sorted(int(rec["num_frames"]) for rec in records.values())
    total = sum((rec["load"] for rec in records.values()), Fraction(0))
    under_1s = sum(1 for rec in records.values() if Fraction(rec["num_frames"]) < rec["fps"])
    return {
        "available": True,
        "motions": len(frames),
        "min_frames": frames[0],
        "clips_at_min_frames": sum(1 for value in frames if value == frames[0]),
        "median_frames": frames[len(frames) // 2],
        "max_frames": frames[-1],
        "total_duration_s": float(total),
        "under_1s": under_1s,
        "frames_lt": {
            str(threshold): sum(1 for value in frames if value < threshold)
            for threshold in SHORT_CLIP_FRAME_THRESHOLDS
        },
    }


def scan_direct_source(source: DirectSource) -> dict[str, Any]:
    """Catalog one pack directory from listings; strict USD realpaths; optional meta."""
    root = source.root
    motion_dir = os.path.join(root, "robot")
    stems: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(motion_dir, "**", "*.pkl"), recursive=True)):
        if path.endswith("metadata.pkl"):
            continue  # same exclusion as MotionLib's key discovery
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in stems:
            raise ValueError(
                f"direct roster source {source.name}: duplicate motion stem {stem!r} "
                f"({stems[stem]} and {path}); MotionLib keys motions by pkl stem"
            )
        stems[stem] = path
    if not stems:
        raise ValueError(f"direct roster source {source.name}: no robot/*.pkl under {motion_dir}")
    ordered = sorted(stems)

    objects_names = _listing(os.path.join(root, "objects_with_masks"))
    bps_names = _listing(os.path.join(root, "bps"))
    # Active-USD store: shared resolver contract (explicit > tiered-if-present > native); per key the
    # native store backs a file the chosen store lacks (counted, never an error while native has it).
    active_info = describe_active_usd_selection(root, source.active_usd_subdir)
    active_subdir = active_info["subdir"]
    active_names = _listing(os.path.join(root, active_subdir))
    native_names = _listing(os.path.join(root, NATIVE_SUBDIR)) if active_subdir != NATIVE_SUBDIR else set()
    native_fallback_keys = 0
    context_names = _listing(os.path.join(root, "static_context_usd_grail_paired"))
    meta_dir = os.path.join(root, "meta")
    meta_names = _listing(meta_dir)
    surface_dir = os.path.join(root, "active_surface")
    surface_names = _listing(surface_dir)

    missing: dict[str, list[str]] = {
        "objects_with_masks": [],
        "bps": [],
        "active_articulation_usd": [],
        "static_context_usd_grail_paired": [],
    }
    active_entry: dict[str, str] = {}
    context_entry: dict[str, str] = {}
    for stem in ordered:
        if f"{stem}.pkl" not in objects_names:
            missing["objects_with_masks"].append(stem)
        if f"{stem}.npy" not in bps_names:
            missing["bps"].append(stem)
        name = _usd_name(active_names, stem)
        entry_dir = active_subdir
        if name is None and native_names:
            name = _usd_name(native_names, stem)
            if name is not None:
                entry_dir = NATIVE_SUBDIR
                native_fallback_keys += 1
        if name is None:
            missing["active_articulation_usd"].append(stem)
        else:
            active_entry[stem] = os.path.join(root, entry_dir, name)
        name = _usd_name(context_names, stem)
        if name is None:
            missing["static_context_usd_grail_paired"].append(stem)
        else:
            context_entry[stem] = os.path.join(root, "static_context_usd_grail_paired", name)
    suffixes = {
        "objects_with_masks": ".pkl",
        "bps": ".npy",
        "active_articulation_usd": ".usd|.usda",
        "static_context_usd_grail_paired": ".usd|.usda",
    }
    problems = [
        f"{subdir}: {len(items)} motions without {subdir}/<stem>{suffixes[subdir]} (first: {items[:8]})"
        for subdir, items in missing.items()
        if items
    ]
    if problems:
        raise FileNotFoundError(
            f"direct roster source {source.name} ({root}): " + "; ".join(problems)
        )

    records: dict[str, dict[str, Any]] = {}
    for stem in ordered:
        active_realpath = os.path.realpath(active_entry[stem])
        if not os.path.isfile(active_realpath):
            raise FileNotFoundError(
                f"direct roster source {source.name}: {active_entry[stem]} -> {active_realpath} does not exist"
            )
        context_realpath = os.path.realpath(context_entry[stem])
        if not os.path.isfile(context_realpath):
            raise FileNotFoundError(
                f"direct roster source {source.name}: {context_entry[stem]} -> {context_realpath} does not exist"
            )
        records[stem] = {
            "active_entry": active_entry[stem],
            "active_realpath": active_realpath,
            "context_entry": context_entry[stem],
            "context_realpath": context_realpath,
        }

    # Balance/label mode is a listing fact (identical on every rank): meta/<stem>.pkl for
    # EVERY motion -> duration balance + labels; otherwise count balance for the whole
    # source and the stem as label. A read failure on a present meta file is an error.
    meta_missing = [stem for stem in ordered if f"{stem}.pkl" not in meta_names]
    if os.path.isdir(meta_dir) and not meta_missing:
        balance = "duration_s"
        for stem in ordered:
            meta = _read_meta_record(source.name, os.path.join(meta_dir, f"{stem}.pkl"), stem)
            records[stem].update(
                num_frames=meta["num_frames"],
                fps=meta["fps"],
                load=Fraction(meta["num_frames"]) / meta["fps"],
                label=meta["label"],
                object_key=f"{source.name}/{meta['label'] or stem}",
            )
        meta_note = (
            f"meta/<stem>.pkl present for all {len(ordered)} motions: duration balance, "
            "object_key from object_name/active_object (stem when both are empty)"
        )
    else:
        balance = "count"
        for stem in ordered:
            records[stem].update(
                num_frames=None, fps=None, load=Fraction(1), label=None,
                object_key=f"{source.name}/{stem}",
            )
        if not os.path.isdir(meta_dir):
            meta_note = "meta/ absent: count balance, object_key from stem"
        else:
            meta_note = (
                f"meta/<stem>.pkl missing for {len(meta_missing)} of {len(ordered)} motions "
                f"(first: {meta_missing[:4]}): count balance for the whole source, object_key from stem"
            )
    return {
        "source": source.name,
        "root": root,
        "budget_slots": source.budget_slots,
        "motion_dir": motion_dir,
        "object_motion_dir": os.path.join(root, "objects_with_masks"),
        "bps_dir": os.path.join(root, "bps"),
        "stems": ordered,
        "records": records,
        "balance": balance,
        "meta_note": meta_note,
        "clip_length": _clip_length_summary(records, balance),
        "surface_dir": surface_dir if os.path.isdir(surface_dir) else None,
        "surface_names": surface_names,
        "active_usd_subdir": active_subdir,
        "active_usd_selection": active_info,
        "active_usd_native_fallback_keys": native_fallback_keys,
    }


# ------------------------------------------------------------------------- groups


def _numpy():
    try:
        import numpy as np  # noqa: PLC0415 - only needed for shared-asset packs
    except ImportError as exc:  # pragma: no cover - login nodes without numpy
        raise RuntimeError(
            "numpy is required to split asset groups that share USD files by BPS vector"
        ) from exc
    return np


def _bps_key(np: Any, bps_dir: str, stem: str) -> tuple[float, ...]:
    vector = np.asarray(np.load(os.path.join(bps_dir, f"{stem}.npy")))
    if vector.shape != (10,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid BPS vector {bps_dir}/{stem}.npy: shape {vector.shape}, finite required")
    return tuple(float(value) for value in vector.tolist())


def build_direct_groups(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Exact-asset groups, ordered by first member stem; ``{source}_asset_{index:06d}``."""
    source = str(catalog["source"])
    records = catalog["records"]
    by_path: dict[tuple[str, str], list[str]] = {}
    for stem in catalog["stems"]:
        rec = records[stem]
        by_path.setdefault((rec["active_realpath"], rec["context_realpath"]), []).append(stem)
    groups: list[dict[str, Any]] = []
    for (active_realpath, context_realpath), members in by_path.items():
        if len(members) == 1:
            parts: list[tuple[tuple[float, ...] | None, list[str]]] = [(None, members)]
        else:
            np = _numpy()
            by_bps: dict[tuple[float, ...], list[str]] = {}
            for stem in members:
                by_bps.setdefault(_bps_key(np, str(catalog["bps_dir"]), stem), []).append(stem)
            parts = sorted(by_bps.items(), key=lambda item: item[1][0])
        for bps_key, stems in parts:
            stems = sorted(stems)
            keys = sorted({str(records[stem]["object_key"]) for stem in stems})
            object_key = keys[0] if len(keys) == 1 else f"{source}/asset:{Path(active_realpath).stem}"
            surface = None
            if catalog.get("surface_dir") and f"{stems[0]}.npy" in catalog.get("surface_names", ()):
                surface = os.path.realpath(os.path.join(str(catalog["surface_dir"]), f"{stems[0]}.npy"))
            num_frames = [records[stem]["num_frames"] for stem in stems]
            groups.append({
                "source": source,
                "asset_signature": "",
                "object_key": object_key,
                "active_articulation_usd": active_realpath,
                "static_context_usd": context_realpath,
                "active_surface_path": surface,
                "motion_aliases": stems,
                "representative_alias": stems[0],
                "motion_count": len(stems),
                "load": sum((records[stem]["load"] for stem in stems), Fraction(0)),
                "num_frames_total": None if any(value is None for value in num_frames) else int(sum(num_frames)),
                "bps_key": None if bps_key is None else list(bps_key),
            })
    groups.sort(key=lambda group: group["motion_aliases"][0])
    for index, group in enumerate(groups):
        group["asset_signature"] = f"{source}_asset_{index:06d}"
    return groups


# --------------------------------------------------------------------- allocation


def _as_weight(value: Any, what: str) -> Fraction:
    if isinstance(value, Fraction):
        weight = value
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what}: weight must be a number, got {value!r}")
    else:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{what}: weight must be finite, got {value!r}")
        weight = Fraction(value)
    if weight < 0:
        raise ValueError(f"{what}: weight must be >= 0, got {value!r}")
    return weight


def allocate_lower_bounded_proportional(
    items: Sequence[tuple[str, Any, int]], total: int
) -> dict[str, int]:
    """Integer shares proportional to weight with per-key lower bounds.

    ``target_k = total * w_k / sum(w)``; keys whose target is below their lower bound are
    pinned to it and the rest is redistributed proportionally among the remaining keys
    (repeated until stable); the final free shares are floored and the leftover goes by
    largest remainder, ties broken by item order. Exact ``Fraction`` arithmetic: the
    result is identical on every rank and every resume.
    """
    keys = [str(key) for key, _, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError("allocation keys must be unique")
    position = {key: index for index, key in enumerate(keys)}
    lowers = {str(key): int(lower) for key, _, lower in items}
    weights = {str(key): _as_weight(weight, f"allocation key {key}") for key, weight, _ in items}
    if any(lower < 0 for lower in lowers.values()):
        raise ValueError("allocation lower bounds must be >= 0")
    if total < sum(lowers.values()):
        raise ValueError(
            f"{total} slots cannot satisfy lower bounds summing to {sum(lowers.values())}"
        )
    fixed: dict[str, int] = {}
    free = list(keys)
    shares: dict[str, Fraction] = {}
    remaining = total
    while free:
        remaining = total - sum(fixed.values())
        weight_sum = sum((weights[key] for key in free), Fraction(0))
        if weight_sum == 0:
            shares = {key: Fraction(remaining, len(free)) for key in free}
        else:
            shares = {key: Fraction(remaining) * weights[key] / weight_sum for key in free}
        below = [key for key in free if shares[key] < lowers[key]]
        if not below:
            break
        for key in below:
            fixed[key] = lowers[key]
        free = [key for key in free if key not in fixed]
    result = dict(fixed)
    if free:
        floors = {key: int(math.floor(shares[key])) for key in free}
        leftover = remaining - sum(floors.values())
        order = sorted(free, key=lambda key: (-(shares[key] - floors[key]), position[key]))
        for key in order[:leftover]:
            floors[key] += 1
        result.update(floors)
    if sum(result.values()) != total:  # pragma: no cover - arithmetic guard
        raise RuntimeError("allocation does not sum to the requested total")
    return result


def default_group_weight(group: Mapping[str, Any]) -> int:
    """Agreed default: slots proportional to the group's motion count."""
    return len(group["motion_aliases"])


def _coerce_group_weight(group_weight: GroupWeight | Mapping[str, Any] | None) -> GroupWeight:
    if group_weight is None:
        return default_group_weight
    if callable(group_weight):
        return group_weight
    if isinstance(group_weight, Mapping):
        mapping = group_weight

        def from_mapping(group: Mapping[str, Any]) -> Any:
            if group["asset_signature"] in mapping:
                return mapping[group["asset_signature"]]
            if group["object_key"] in mapping:
                return mapping[group["object_key"]]
            return default_group_weight(group)

        return from_mapping
    raise TypeError("group_weight must be None, a callable(group) or a mapping")


def allocate_direct_slots(
    groups_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    total_slots: int,
    *,
    source_budgets: Mapping[str, int | None] | None = None,
    group_weight: GroupWeight | Mapping[str, Any] | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Slots per asset signature: per-source budgets, then lower-bounded proportional groups."""
    names = list(groups_by_source)
    if not names:
        raise ValueError("no sources to allocate")
    total_slots = int(total_slots)
    group_counts = {name: len(groups_by_source[name]) for name in names}
    motion_counts = {
        name: sum(len(group["motion_aliases"]) for group in groups_by_source[name]) for name in names
    }
    group_total = sum(group_counts.values())
    motion_total = sum(motion_counts.values())
    if group_total > total_slots:
        raise ValueError(
            f"direct roster: {group_total} distinct asset groups ({motion_total} motions) exceed "
            f"total slots {total_slots}; every group needs at least one slot"
        )
    explicit = {
        str(name): int(budget)
        for name, budget in (source_budgets or {}).items()
        if budget is not None
    }
    unknown = sorted(set(explicit) - set(names))
    if unknown:
        raise ValueError(f"budgets given for unknown sources {unknown}")
    for name, budget in explicit.items():
        if budget < group_counts[name]:
            raise ValueError(
                f"direct roster source {name}: budget {budget} slots is below its "
                f"{group_counts[name]} asset groups (coverage needs one slot per group)"
            )
    fixed_sum = sum(explicit.values())
    unspecified = [name for name in sorted(names) if name not in explicit]
    budgets = dict(explicit)
    if not unspecified:
        if fixed_sum != total_slots:
            raise ValueError(
                f"explicit per-source budgets sum to {fixed_sum}, total slots are {total_slots}"
            )
    else:
        remainder = total_slots - fixed_sum
        need = sum(group_counts[name] for name in unspecified)
        if remainder < need:
            raise ValueError(
                f"explicit budgets leave {remainder} slots for sources {unspecified}, which need "
                f"at least {need} (one per asset group)"
            )
        budgets.update(
            allocate_lower_bounded_proportional(
                [(name, motion_counts[name], group_counts[name]) for name in unspecified], remainder
            )
        )
    weight_fn = _coerce_group_weight(group_weight)
    slots_by_signature: dict[str, int] = {}
    for name in names:
        items = [
            (str(group["asset_signature"]), weight_fn(group), 1)
            for group in groups_by_source[name]
        ]
        slots_by_signature.update(allocate_lower_bounded_proportional(items, budgets[name]))
    extra = sorted(
        ((signature, slots) for signature, slots in slots_by_signature.items() if slots > 1),
        key=lambda item: item[0],
    )
    report = {
        "total_slots": total_slots,
        "group_count": group_total,
        "motion_count": motion_total,
        "extra_slots": total_slots - group_total,
        "weight": "motion_count" if group_weight is None else "custom",
        "source_budgets": {
            name: {
                "budget": budgets[name],
                "groups": group_counts[name],
                "motions": motion_counts[name],
                "explicit": name in explicit,
            }
            for name in names
        },
        "extra_slot_signatures": [[signature, slots] for signature, slots in extra],
    }
    return slots_by_signature, report


# ------------------------------------------------------------------------- layout


def layout_ranks(
    groups: Sequence[Mapping[str, Any]],
    slots_by_signature: Mapping[str, int],
    world_size: int,
    num_envs: int,
) -> list[dict[str, Any]]:
    """Exactly ``num_envs`` slots per rank; min-load placement; split only when forced."""
    world_size, num_envs = int(world_size), int(num_envs)
    total = num_envs * world_size
    slots = [int(slots_by_signature[str(group["asset_signature"])]) for group in groups]
    if sum(slots) != total:
        raise ValueError(f"slots sum to {sum(slots)}, expected {total}")
    order = sorted(
        range(len(groups)),
        key=lambda index: (-slots[index], -groups[index]["load"], str(groups[index]["asset_signature"])),
    )
    ranks: list[dict[str, Any]] = [
        {"rank": rank, "remaining": num_envs, "load": Fraction(0), "entries": {}, "shared": []}
        for rank in range(world_size)
    ]

    def place(entry: dict[str, Any], index: int, count: int) -> None:
        if index not in entry["entries"]:
            entry["load"] += groups[index]["load"]
        entry["entries"][index] = entry["entries"].get(index, 0) + count
        entry["remaining"] -= count

    for index in order:
        count = slots[index]
        candidates = [entry for entry in ranks if entry["remaining"] >= count]
        if candidates:
            place(min(candidates, key=lambda entry: (entry["load"], entry["rank"])), index, count)
            continue
        left = count
        while left > 0:
            target = max(ranks, key=lambda entry: (entry["remaining"], -entry["rank"]))
            take = min(target["remaining"], left)
            if take <= 0:  # pragma: no cover - sum(slots) == total guarantees capacity
                raise RuntimeError("rank capacity exhausted during split")
            place(target, index, take)
            target["shared"].append(str(groups[index]["asset_signature"]))
            left -= take
    for entry in ranks:
        if entry["remaining"] != 0:  # pragma: no cover - arithmetic guard
            raise RuntimeError(f"rank {entry['rank']} ends with {entry['remaining']} free slots")
        ordered = sorted(entry["entries"], key=lambda index: str(groups[index]["asset_signature"]))
        counts = {index: entry["entries"][index] for index in ordered}
        sequence: list[int] = []
        while len(sequence) < num_envs:
            for index in ordered:
                if counts[index] > 0:
                    sequence.append(index)
                    counts[index] -= 1
        entry["slot_sequence"] = sequence
        entry["group_indices"] = ordered
        entry["unique_motions"] = len({alias for index in ordered for alias in groups[index]["motion_aliases"]})
        entry["shared"] = sorted(set(entry["shared"]))
    return ranks


def _load_value(value: Fraction) -> float:
    return float(value)


def _check_runtime_mapping_defaults() -> None:
    repeat = slot_repeat_factor()
    if repeat != 1 or rank_interleave_enabled():
        raise ValueError(
            "direct roster computes its own balanced per-rank layout; UNIFIED_COHORT_SLOT_REPEAT "
            f"(={repeat}) and UNIFIED_COHORT_RANK_INTERLEAVE must stay at their defaults "
            "(slot repeat / interleave for direct directories is a follow-up)"
        )


def _validate_rank_args(num_envs: int, rank: int, world_size: int, runtime_target_fps_hz: float | None) -> None:
    if int(num_envs) <= 0 or int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
        raise ValueError("Require num_envs > 0, world_size > 0, and 0 <= rank < world_size")
    if runtime_target_fps_hz is not None and float(runtime_target_fps_hz) != 50.0:
        raise ValueError("unified no-VOF runtime requires 50 Hz")


def single_source_error(sources: Sequence[DirectSource]) -> ValueError:
    names = [source.name for source in sources]
    return ValueError(
        "direct roster v1 loads one pack directory per launch (MotionLib reads one "
        f"motion_dir/object_motion_dir/bps_dir); {len(sources)} sources given: {names}. Run one "
        "source per experiment or wait for the alias->path loader change."
    )


def build_direct_layout(
    sources: Sequence[DirectSource],
    *,
    num_envs: int,
    world_size: int,
    group_weight: GroupWeight | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Scan + group + allocate + lay out once; identical on every rank (no RNG, no timing)."""
    num_envs, world_size = int(num_envs), int(world_size)
    if num_envs <= 0 or world_size <= 0:
        raise ValueError("Require num_envs > 0 and world_size > 0")
    total_slots = num_envs * world_size
    catalogs = [scan_direct_source(source) for source in sources]
    groups_by_source = {catalog["source"]: build_direct_groups(catalog) for catalog in catalogs}
    groups = [group for catalog in catalogs for group in groups_by_source[catalog["source"]]]
    group_total = len(groups)
    motion_total = sum(len(catalog["stems"]) for catalog in catalogs)
    if group_total > total_slots:
        per_source = ", ".join(
            f"{catalog['source']}: {len(groups_by_source[catalog['source']])} groups/{len(catalog['stems'])} motions"
            for catalog in catalogs
        )
        needed = -(-group_total // world_size)
        raise ValueError(
            f"direct roster: {group_total} distinct asset groups ({motion_total} motions; per source: "
            f"{per_source}) exceed total slots {total_slots} = NUM_ENVS {num_envs} x WORLD_SIZE "
            f"{world_size}. Coverage of every motion in one leg needs NUM_ENVS >= {needed} per rank "
            f"(next multiple of 4: {-(-needed // 4) * 4}), or a pack layout that shares asset files "
            "across motions. Per-leg asset rotation is not implemented (follow-up; explicit opt-in only)."
        )
    slots_by_signature, allocation = allocate_direct_slots(
        groups_by_source,
        total_slots,
        source_budgets={source.name: source.budget_slots for source in sources},
        group_weight=group_weight,
    )
    ranks = layout_ranks(groups, slots_by_signature, world_size, num_envs)
    covered = {
        alias
        for entry in ranks
        for index in entry["group_indices"]
        for alias in groups[index]["motion_aliases"]
    }
    catalog_aliases = {stem for catalog in catalogs for stem in catalog["stems"]}
    if covered != catalog_aliases:  # pragma: no cover - one slot per group guarantees coverage
        raise RuntimeError(
            f"{len(catalog_aliases - covered)} motions have no slot on any rank"
        )
    shared = sorted({signature for entry in ranks for signature in entry["shared"]})
    report = {
        "contract": LAYOUT_CONTRACT,
        "num_envs": num_envs,
        "world_size": world_size,
        "total_slots": total_slots,
        "catalog_motion_count": motion_total,
        "group_count": group_total,
        "extra_slots": allocation["extra_slots"],
        "extra_slot_signatures": allocation["extra_slot_signatures"],
        "weight": allocation["weight"],
        "source_budgets": allocation["source_budgets"],
        "sources": [
            {
                "name": catalog["source"],
                "root": catalog["root"],
                "motion_count": len(catalog["stems"]),
                "group_count": len(groups_by_source[catalog["source"]]),
                "budget_slots": catalog["budget_slots"],
                "balance": catalog["balance"],
                "meta_note": catalog["meta_note"],
                "clip_length": catalog["clip_length"],
                "first_stem": catalog["stems"][0],
                "last_stem": catalog["stems"][-1],
                "active_usd_subdir": catalog["active_usd_subdir"],
                "active_usd_selection": catalog["active_usd_selection"],
                "active_usd_native_fallback_keys": catalog["active_usd_native_fallback_keys"],
            }
            for catalog in catalogs
        ],
        "global_load": _load_value(sum((group["load"] for group in groups), Fraction(0))),
        "shared_groups": shared,
        "ranks": [
            {
                "rank": entry["rank"],
                "slots": num_envs,
                "groups": len(entry["group_indices"]),
                "unique_motions": entry["unique_motions"],
                "load": _load_value(entry["load"]),
                "shared_groups": entry["shared"],
            }
            for entry in ranks
        ],
    }
    return {
        "sources": list(sources),
        "catalogs": catalogs,
        "groups": groups,
        "slots_by_signature": slots_by_signature,
        "ranks": ranks,
        "report": report,
        "num_envs": num_envs,
        "world_size": world_size,
    }


def direct_identity_records(layout: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Global per-alias identity records of a direct layout (identical on every rank; never written to disk).

    Minimal in-memory counterpart of the unified cohort manifest records read by
    ``MotionLib.init_joint_source_tracking``: ``alias`` (== MotionLib key == ``robot/<stem>.pkl`` stem),
    ``source`` (the explicit DirectSource name from JOINT_ROSTER_DIRECT_DIRS), ``source_motion_key`` (the original
    stem; identity = ``"<source>::<stem>"``), ``source_sample_fps_hz`` (float from ``meta/<stem>.pkl`` when present,
    else None - never a filled 30/50; the consumer falls back to the robot payload's own FPS semantics),
    ``num_frames`` (int or None), the LAYOUT-LOCAL grouping labels of this layout (``object_key`` - label-derived or
    a ``<source>/<stem>`` / ``<source>/asset:<usd basename>`` fallback; ``asset_signature`` =
    ``<source>_asset_<index>`` over the sorted stems of THIS layout; ``group_index``), the physical grouping basis
    (``active_articulation_usd`` / ``static_context_usd`` realpaths; the BPS vector is ``bps_dir/<stem>.npy``) and
    ``ranks`` (every rank whose slot sequence contains the group, so one alias may be loaded by several ranks).
    The only identity that persists across layouts is ``"<source>::<source_motion_key>"``: statistics must be keyed
    by it and re-aggregated per layout from the real active/context realpaths + BPS grouping, never restored by a
    stale ``asset_signature`` / ``group_index``. Sorted by alias, one record per alias; slot order, rank slices and
    every existing view key are unchanged. No manifest is serialized and no data is invented.
    """
    catalog = layout["catalogs"][0]
    records = catalog["records"]
    source = str(catalog["source"])
    ranks_by_group: dict[int, set[int]] = {}
    for rank, entry in enumerate(layout["ranks"]):
        for index in entry["slot_sequence"]:
            ranks_by_group.setdefault(int(index), set()).add(int(rank))
    out: list[dict[str, Any]] = []
    for index, group in enumerate(layout["groups"]):
        for stem in group["motion_aliases"]:
            record = records[stem]
            fps = record.get("fps")
            frames = record.get("num_frames")
            out.append({
                "alias": stem,
                "source": source,
                "source_motion_key": stem,
                "source_sample_fps_hz": None if fps is None else float(fps),
                "num_frames": None if frames is None else int(frames),
                "object_key": group["object_key"],
                "asset_signature": group["asset_signature"],
                "group_index": int(index),
                "active_articulation_usd": group["active_articulation_usd"],
                "static_context_usd": group["static_context_usd"],
                "ranks": sorted(ranks_by_group.get(index, ())),
            })
    out.sort(key=lambda item: item["alias"])
    return out


def rank_view_from_layout(
    layout: Mapping[str, Any],
    rank: int,
    *,
    runtime_target_fps_hz: float | None = None,
) -> dict[str, Any]:
    """The unified-view key set for one rank (unified_cohort.py:188-211) plus layout info."""
    sources: Sequence[DirectSource] = layout["sources"]
    if len(sources) != 1:
        raise single_source_error(sources)
    num_envs, world_size = int(layout["num_envs"]), int(layout["world_size"])
    _validate_rank_args(num_envs, rank, world_size, runtime_target_fps_hz)
    _check_runtime_mapping_defaults()
    rank = int(rank)
    catalog = layout["catalogs"][0]
    groups = layout["groups"]
    entry = layout["ranks"][rank]
    slots_by_signature = layout["slots_by_signature"]
    seen_per_group: dict[int, int] = {}
    slots: list[dict[str, Any]] = []
    aliases: set[str] = set()
    for local_env_id, index in enumerate(entry["slot_sequence"]):
        group = groups[index]
        slot_index = seen_per_group.get(index, 0)
        seen_per_group[index] = slot_index + 1
        aliases.update(group["motion_aliases"])
        slots.append({
            "local_env_id": local_env_id,
            "global_env_id": rank * num_envs + local_env_id,
            "base_global_slot": rank * num_envs + local_env_id,
            "source": group["source"],
            "asset_signature": group["asset_signature"],
            "object_key": group["object_key"],
            "active_articulation_usd": group["active_articulation_usd"],
            "static_context_usd": group["static_context_usd"],
            "active_surface_path": group["active_surface_path"],
            "compatible_motion_aliases": list(group["motion_aliases"]),
            "representative_alias": group["representative_alias"],
            "entity_kind": "rigid",
            "mechanism_resistance": dict(ZERO_RESISTANCE),
            "group_slot_index": slot_index,
            "group_rank_slots": int(entry["entries"][index]),
            "group_total_slots": int(slots_by_signature[group["asset_signature"]]),
            "group_shared": group["asset_signature"] in entry["shared"],
            "group_motion_count": int(group["motion_count"]),
            "group_load": _load_value(group["load"]),
        })
    if len(slots) != num_envs:  # pragma: no cover - layout_ranks guarantees this
        raise RuntimeError(f"rank {rank} resolved {len(slots)} slots, expected {num_envs}")
    report = layout["report"]
    return {
        # Label only (never opened): commands.py passes it into motion_lib_cfg.joint_roster_manifest,
        # which the motion library uses as a truthy flag and the tensor cache as a path identity.
        "manifest_path": str(catalog["root"]),
        "motion_dir": str(catalog["motion_dir"]),
        "object_motion_dir": str(catalog["object_motion_dir"]),
        "bps_dir": str(catalog["bps_dir"]),
        "rank_entity_kind": "rigid",
        "rank": rank,
        "world_size": world_size,
        "num_envs": num_envs,
        "global_num_envs": num_envs * world_size,
        "base_num_envs": num_envs,
        "slot_repeat": 1,
        "rank_interleave": False,
        "motion_aliases": sorted(aliases),
        "slots": slots,
        "resolved_motion_count": len(aliases),
        "catalog_motion_count": int(report["catalog_motion_count"]),
        "dropout_motion_count": 0,
        # No joint-source identity file exists: the motion library never opens one. The identities themselves are
        # carried IN MEMORY (direct_roster_identity_v1) for the sampling consumer: identical on every rank.
        "skip_joint_source_tracking": True,
        "identity_contract": IDENTITY_CONTRACT,
        "identity_source_order": [str(catalog["source"])],
        "identity_records": direct_identity_records(layout),
        "roster_kind": ROSTER_KIND,
        "direct_sources": report["sources"],
        "layout": report,
        "group_count": int(report["group_count"]),
    }


def resolve_direct_directory_roster(
    spec: Any,
    *,
    num_envs: int,
    rank: int,
    world_size: int,
    runtime_target_fps_hz: float | None = None,
    group_weight: GroupWeight | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Runtime entry: one pack directory -> this rank's slot view (global layout computed in memory)."""
    sources = parse_direct_sources(spec)
    _validate_rank_args(num_envs, rank, world_size, runtime_target_fps_hz)
    _check_runtime_mapping_defaults()
    if len(sources) != 1:
        raise single_source_error(sources)
    layout = build_direct_layout(sources, num_envs=num_envs, world_size=world_size, group_weight=group_weight)
    return rank_view_from_layout(layout, rank, runtime_target_fps_hz=runtime_target_fps_hz)


# ------------------------------------------------------------------- env cfg glue


def _format_clip_length(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return "clips_under_1s=n/a clip_frames_lt=n/a"
    frames_lt = ",".join(f"{key}:{value}" for key, value in summary["frames_lt"].items())
    return (
        f"clips_under_1s={summary['under_1s']} clip_frames_lt={{{frames_lt}}} "
        f"clip_frames_min/median/max={summary['min_frames']}/{summary['median_frames']}/{summary['max_frames']} "
        f"clips_at_min_frames={summary['clips_at_min_frames']}"
    )


def direct_roster_startup_line(view: Mapping[str, Any], *, elapsed_s: float | None = None) -> str:
    """One greppable line with the global numbers every rank must print identically."""
    report = view["layout"]
    source = report["sources"][0] if report["sources"] else {}
    rank_report = report["ranks"][int(view["rank"])]
    extra = report["extra_slot_signatures"]
    extra_text = ",".join(f"{signature}x{slots}" for signature, slots in extra[:EXTRA_SLOT_LOG_LIMIT])
    if len(extra) > EXTRA_SLOT_LOG_LIMIT:
        extra_text += f",...(+{len(extra) - EXTRA_SLOT_LOG_LIMIT})"
    parts = [
        "[JointRoster] startup_phase=direct_directory_roster",
        f"rank={view['rank']}",
        f"world_size={view['world_size']}",
        f"num_envs={view['num_envs']}",
        f"source={source.get('name')}",
        f"root={source.get('root')}",
        f"active_usd_subdir={source.get('active_usd_subdir')}",
        f"keys_from_native_fallback={source.get('active_usd_native_fallback_keys')}",
        f"balance={source.get('balance')}",
        f"catalog_motions={report['catalog_motion_count']}",
        f"first_stem={source.get('first_stem')}",
        f"last_stem={source.get('last_stem')}",
        f"groups={report['group_count']}",
        f"total_slots={report['total_slots']}",
        f"extra_slots={report['extra_slots']}",
        f"extra_slot_groups={len(extra)}",
        f"extra_slot_signatures={extra_text or 'none'}",
        f"global_load={report['global_load']:.3f}",
        f"rank_load={rank_report['load']:.3f}",
        f"rank_motions={rank_report['unique_motions']}",
        f"rank_groups={rank_report['groups']}",
        f"shared_groups={len(report['shared_groups'])}",
        _format_clip_length(source.get("clip_length", {})),
        "plan_cache=not_used",
    ]
    if elapsed_s is not None:
        parts.append(f"elapsed_s={float(elapsed_s):.3f}")
    return " ".join(parts)


def resolve_direct_roster_for_env_cfg(
    motion_lib_cfg: Mapping[str, Any],
    *,
    num_envs: int,
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any], list[str]]:
    """Env-cfg entry for the direct branch: native in-memory resolution, never the plan cache.

    The rank scene-plan cache keys on a manifest FILE identity; a directory stat is not an
    honest identity for a listing-derived view and a persisted rank plan would be exactly
    the kind of stale-able roster index this route exists to avoid. The requested cache
    mode is therefore only reported, never used.
    """
    spec = normalize_direct_spec(motion_lib_cfg.get("joint_roster_direct_dirs"))
    if not spec:
        raise ValueError("motion_lib_cfg.joint_roster_direct_dirs is empty")
    started = time.perf_counter()
    view = resolve_direct_directory_roster(
        spec,
        num_envs=int(num_envs),
        rank=int(rank),
        world_size=int(world_size),
        runtime_target_fps_hz=float(motion_lib_cfg.get("target_fps", 50) or 50),
    )
    elapsed = time.perf_counter() - started
    lines = [direct_roster_startup_line(view, elapsed_s=elapsed)]
    mode = str(motion_lib_cfg.get("joint_roster_scene_plan_cache_mode", "disabled") or "disabled").strip().lower()
    if mode != "disabled":
        lines.append(
            "[JointRoster] rank scene-plan cache not used for direct directories "
            f"(requested mode={mode}, dir={motion_lib_cfg.get('joint_roster_scene_plan_cache_dir')}): "
            "the slot view is derived in memory from the directory listing at every start; "
            f"native in-memory resolution took {elapsed:.1f} s"
        )
    return view, lines


__all__ = [
    "DirectSource",
    "ROSTER_KIND",
    "LAYOUT_CONTRACT",
    "allocate_direct_slots",
    "allocate_lower_bounded_proportional",
    "build_direct_groups",
    "build_direct_layout",
    "default_group_weight",
    "direct_roster_startup_line",
    "layout_ranks",
    "normalize_direct_spec",
    "parse_direct_sources",
    "rank_view_from_layout",
    "resolve_direct_directory_roster",
    "resolve_direct_roster_for_env_cfg",
    "scan_direct_source",
    "single_source_error",
]
