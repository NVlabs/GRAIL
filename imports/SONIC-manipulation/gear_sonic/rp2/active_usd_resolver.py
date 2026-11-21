"""Per-source selection of the active-articulation USD directory (pure stdlib; writes nothing).

Human directive 2026-10-14 ("默认采用分层资产"): the existing tiered collision assets are the shared default.
A pack may carry two authored stores for the same keys: ``active_articulation_usd`` (native; e.g. SDF256
for every object) and ``active_articulation_usd_tiered_v1`` (authored tiers: convex decompositions for large
objects, SDF128 for mid-size ones, thin/small objects kept at SDF256). Only the DIRECTORY choice is made here;
masses, physics overrides, visual meshes, motion coverage, source budgets and every per-asset policy inside the
chosen store are untouched. Both the global cohort builder (tools/build_unified_no_vof.py) and the
direct-directory roster (gear_sonic/rp2/direct_directory_roster.py) use this one contract:

  explicit directory name (per-source config ``active_usd_subdir`` / spec ``:active_usd_subdir=<dir>``)
    > ``"tiered"`` (also ``None`` / ``""`` / ``"default"``): the tiered store when the pack has it, else native
    > ``"native"``: the native store.

Per key: when the chosen store lacks ``<key>.usd|.usda`` but the native store has it, the native file is used
and counted (``keys_from_native_fallback``); a missing tiered file is never an error while the native one
exists. A changed selection changes the resolved paths, so every path-identified cache (sources cache, cohort
manifest reuse check, authored-stage cache) misses and rebuilds naturally; nothing is gated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

NATIVE_SUBDIR = "active_articulation_usd"
TIERED_SUBDIR = "active_articulation_usd_tiered_v1"
DEFAULT_SELECTION = "tiered"
# Launcher-level selection applied to EVERY source (train_global_teleop_grail.sh / train_unified_no_vof.sh).
ENV_SELECTION = "ACTIVE_USD_SELECTION"
# Direct-directory roster: applies to spec items that carry no ``:active_usd_subdir=`` of their own.
ROSTER_ENV_SELECTION = "JOINT_ROSTER_ACTIVE_USD_SUBDIR"
USD_SUFFIXES = (".usd", ".usda")
_ALIASES = {"": DEFAULT_SELECTION, "default": DEFAULT_SELECTION, "tiered": "tiered", "native": "native"}
_SUFFIX_FOR_NON_NATIVE = "_tiered"


def normalize_selection(selection: Any) -> str:
    text = "" if selection is None else str(selection).strip()
    return _ALIASES.get(text.lower(), text)


def resolve_active_usd_subdir(pack_root: str | os.PathLike[str], selection: Any = None) -> str:
    """Directory name (relative to the pack) that holds the active USDs for this source."""
    root = Path(pack_root)
    sel = normalize_selection(selection)
    if sel == "native":
        return NATIVE_SUBDIR
    if sel == "tiered":
        return TIERED_SUBDIR if (root / TIERED_SUBDIR).is_dir() else NATIVE_SUBDIR
    if "/" in sel or sel in (".", ".."):
        raise ValueError(f"active_usd_subdir must be a directory NAME inside the pack, got {sel!r}")
    if not (root / sel).is_dir():
        raise ValueError(f"active_usd_subdir {sel!r} does not exist under pack {root}")
    return sel


def describe_active_usd_selection(pack_root: str | os.PathLike[str], selection: Any = None) -> dict[str, Any]:
    root = Path(pack_root)
    sel = normalize_selection(selection)
    return {
        "selection": sel,
        "subdir": resolve_active_usd_subdir(root, sel),
        "tiered_available": (root / TIERED_SUBDIR).is_dir(),
        "native_available": (root / NATIVE_SUBDIR).is_dir(),
    }


def find_active_usd(
    pack_root: str | os.PathLike[str], key: str, selection: Any = None, *, subdir: str | None = None
) -> tuple[Path, bool]:
    """``(path, from_native_fallback)`` for one key; the native store backs a key the chosen store lacks."""
    root = Path(pack_root)
    chosen = subdir or resolve_active_usd_subdir(root, selection)
    for directory, fallback in ((chosen, False), (NATIVE_SUBDIR, True)):
        if fallback and directory == chosen:
            break
        for suffix in USD_SUFFIXES:
            candidate = root / directory / f"{key}{suffix}"
            if candidate.is_file():
                return candidate, fallback
    raise FileNotFoundError(f"{root}/{chosen}/{key}.usd[a] (no {NATIVE_SUBDIR} fallback either)")


def selection_summary_line(source: str, info: Mapping[str, Any], fallback_count: int) -> str:
    return (
        f"[active_usd] source={source} active_usd_subdir={info['subdir']} selection={info['selection']} "
        f"tiered_available={str(bool(info.get('tiered_available'))).lower()} keys_from_native_fallback={int(fallback_count)}"
    )


def sources_config_selections(config: Mapping[str, Any], override: Any = None) -> dict[str, dict[str, Any]]:
    """Per-source selection of a sources config; ``override`` (launcher ACTIVE_USD_SELECTION) wins for every source."""
    out: dict[str, dict[str, Any]] = {}
    for name, spec in (config.get("sources") or {}).items():
        # launcher-level override (ACTIVE_USD_SELECTION) applies to every source; otherwise the per-source field
        selection = spec.get("active_usd_subdir") if override in (None, "") else override
        out[name] = describe_active_usd_selection(spec["pack"], selection)
    return out


def cohort_root_suffix(config: Mapping[str, Any], override: Any = None) -> str:
    """``_tiered`` when any configured source resolves to a non-native store, else ``""``."""
    selections = sources_config_selections(config, override)
    return _SUFFIX_FOR_NON_NATIVE if any(v["subdir"] != NATIVE_SUBDIR for v in selections.values()) else ""


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("suffix", "selections"):
        p = sub.add_parser(name)
        p.add_argument("--sources", required=True, help="sources config JSON (unified_no_vof_sources_v1)")
        p.add_argument("--selection", default=None, help="launcher-level selection for every source (tiered|native|<dir>)")
    r = sub.add_parser("resolve")
    r.add_argument("--pack", required=True)
    r.add_argument("--selection", default=None)
    args = parser.parse_args(argv)
    if args.command == "resolve":
        print(resolve_active_usd_subdir(args.pack, args.selection))
        return 0
    config = json.loads(Path(args.sources).read_text())
    if args.command == "suffix":
        print(cohort_root_suffix(config, args.selection))
    else:
        print(json.dumps(sources_config_selections(config, args.selection), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
