"""Optional authored-USD-stage cache for heterogeneous Isaac Lab scenes (NEW8 port).

What is cached: USD scene description only -- the root-layer prim specs under
``/World/envs`` (every per-env Robot / ActiveArticulation / StaticContext authored by
``spawn_multi_asset`` and the ``clone`` decorator) and ``/World/collisions`` (the
per-env collision groups written by ``filter_collisions``). Every leg still creates a
fresh SimulationContext, PhysX scene, asset/sensor wrappers, managers and reset; nothing
about optimizer/RNG/episode/live PhysX state is ever stored.

Attach mode: ``root_copy`` only. The cold leg copies the two scopes out of the live
root layer with ``Sdf.CopySpec`` into ``rank_<r>.usdc`` (asset references made
absolute) and writes ``rank_<r>.json`` (ordinary identity metadata + env origins +
stage size/mtime). The warm leg validates the receipt before touching the stage, copies
the two scopes back into the live root layer, re-applies the physics-scene collision
filter attribute that the cold ``filter_collisions`` authored, and then runs the
unchanged ``InteractiveScene._add_entities_from_cfg`` with ``spawn=None`` for every
env-scoped asset (terrain, lights and every global prim are authored natively; the live
``/physicsScene`` authored by SimulationContext is never replaced). ``root_transfer`` is
deliberately not ported; a launcher that still passes it is mapped to ``root_copy``.

Identity: plain metadata already in memory at the install seam (rank, world size,
num_envs, env spacing, ordered per-slot USD paths with size+mtime, robot USD with
size+mtime, the scene entity roster). Dict equality decides hit/miss. Any miss, any
receipt/stage problem found BEFORE the stage is mutated, or any export problem -> the
unchanged native construction. A failure AFTER the copy started is rolled back at the
Sdf level, the entry is quarantined by a sibling marker (nothing is deleted or renamed)
and the native construction runs on the clean stage; if the rollback cannot be verified
the process fails loudly instead of continuing on a half-attached stage.

No hashes, digests, manifests, READY tokens or full-prim stage traversals. pxr /
isaaclab / isaacsim / carb / torch are imported inside functions so this module imports
on plain CPU python (see prep/fixtures/test_authored_stage_cache_cpu.py).
"""

from __future__ import annotations

import contextlib
import copy
import enum
import datetime as _dt
import json
import os
import time
import uuid
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


SCHEMA = "gear_sonic_authored_stage_cache_new8_root_copy_v2"
SUPPORTED_MODES = frozenset({"disabled", "read_write", "read_only"})
DEFAULT_ATTACH_MODE = "root_copy"
# Launcher values that are accepted and mapped onto the only ported mode.
_ATTACH_MODE_ALIASES = {"root_copy": "root_copy", "root_transfer": "root_copy"}
ENV_NS = "/World/envs"
ENV_REGEX_TOKEN = "{ENV_REGEX_NS}"
# Changed-slot reuse (user priority 2026-10-14: the slowest rank's full native rebuild after a small re-layout): when a
# receipt differs from the current identity ONLY in the ordered per-env USD records of env-scoped assets, the cached
# whole-rank layer is attached and just those env prims are re-authored through the asset's own spawn function - for
# any number of changed envs (no threshold: a crossover to the batch native build would need measured evidence; the
# native build remains the fallback for actual incompatibility or failure only).
# Only the per-env scope is cached. /World/collisions is re-authored natively on every leg by
# InteractiveScene.filter_collisions (cheap, O(n) Sdf specs) because the cloner also authors
# physxScene:invertCollisionGroupFilter on the LIVE physics-scene prim, which is never exported.
CACHED_SCOPES = ("/World/envs",)
INVERT_FILTER_ATTR = "physxScene:invertCollisionGroupFilter"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOG_PREFIX = "[authored_stage]"


def _log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}", flush=True)


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict / DictConfig / attribute bag."""
    if mapping is None:
        return default
    getter = getattr(mapping, "get", None)
    if callable(getter):
        try:
            value = getter(key, default)
        except Exception:  # noqa: BLE001 - struct DictConfig etc.
            value = default
        return default if value is None else value
    return getattr(mapping, key, default)


# ----------------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------------


class AuthoredStageCacheError(RuntimeError):
    """Raised for read_only misses and unrecoverable attach states."""


class StageNotFreshError(AuthoredStageCacheError):
    """The live root layer already carries /World/envs or /World/collisions (never silently continue)."""


@dataclass(frozen=True)
class AuthoredStageCacheSpec:
    mode: str
    cache_dir: Path
    rank: int
    world_size: int
    num_envs: int
    attach_mode: str = DEFAULT_ATTACH_MODE
    requested_attach_mode: str = DEFAULT_ATTACH_MODE

    @classmethod
    def from_mapping(
        cls,
        raw: Any,
        *,
        rank: int,
        world_size: int,
        num_envs: int,
    ) -> "AuthoredStageCacheSpec":
        mode = str(_get(raw, "mode", "disabled") or "disabled").strip().lower()
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"authored_stage_cache.mode={mode!r} not in {sorted(SUPPORTED_MODES)}"
            )
        requested = str(_get(raw, "attach_mode", DEFAULT_ATTACH_MODE) or DEFAULT_ATTACH_MODE).strip().lower()
        attach_mode = _ATTACH_MODE_ALIASES.get(requested)
        if attach_mode is None:
            raise ValueError(
                f"authored_stage_cache.attach_mode={requested!r} not supported "
                f"(accepted: {sorted(_ATTACH_MODE_ALIASES)}; all map to {DEFAULT_ATTACH_MODE})"
            )
        cache_dir_raw = _get(raw, "cache_dir", None)
        if mode != "disabled" and not cache_dir_raw:
            raise ValueError("authored_stage_cache.cache_dir is required when mode != disabled")
        cache_dir = Path(str(cache_dir_raw or ".")).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = Path.cwd() / cache_dir
        return cls(
            mode=mode,
            cache_dir=cache_dir,
            rank=int(rank),
            world_size=int(world_size),
            num_envs=int(num_envs),
            attach_mode=attach_mode,
            requested_attach_mode=requested,
        )

    @property
    def stage_path(self) -> Path:
        return self.cache_dir / f"rank_{self.rank}.usdc"

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / f"rank_{self.rank}.json"

    @property
    def lock_path(self) -> Path:
        return self.cache_dir / f".rank_{self.rank}.lock"

    def failed_marker_path(self, utc: str) -> Path:
        return self.cache_dir / f"rank_{self.rank}.failed-{utc}.json"

    def failed_marker_glob(self) -> str:
        return f"rank_{self.rank}.failed-*.json"


# ----------------------------------------------------------------------------------
# Pure-python helpers (CPU testable)
# ----------------------------------------------------------------------------------


def file_state(path: str | Path) -> dict[str, int | None]:
    """size+mtime(+ctime) of ``path`` (``None`` values when it cannot be stat'ed).

    ``ctime_ns`` is recorded alongside ``mtime_ns`` because on some parallel filesystems a file that was copied with preserved
    timestamps and then rewritten in place carries two mtimes (MDT: the preserved one, OST: the write) and a
    client may report either; the ctime is MDT-authoritative. Identity comparison ignores ``ctime_ns`` except
    as the tie-breaker in ``identity_mtime_flip_tolerated`` (2026-10-14, Generalist rank 4 miss).
    """
    try:
        stat = os.stat(path)
    except OSError:
        return {"size_bytes": None, "mtime_ns": None, "ctime_ns": None}
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "ctime_ns": int(stat.st_ctime_ns)}


def default_search_roots() -> list[Path]:
    """Where a RELATIVE asset path (the robot USD) is anchored: process cwd, then repo root."""
    roots = [Path.cwd(), _REPO_ROOT]
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def is_url(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(("http://", "https://", "omniverse://", "s3://"))


def absolutize_asset_path(path: str, search_roots: list[Path] | None = None) -> str:
    """Decide the absolute form of one asset path.

    Absolute paths and URLs are returned unchanged. A relative path is anchored at the
    first search root under which it exists (default: cwd, then repo root); if it exists
    nowhere it is anchored at the first search root (the native spawner would fail on it
    as well, so the identity still records a stable string).
    """
    if not path or is_url(path):
        return path
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    roots = search_roots or default_search_roots()
    for root in roots:
        probe = Path(root) / candidate
        if probe.exists():
            return str(probe)
    return str(Path(roots[0]) / candidate)


def is_env_scoped_prim_path(prim_path: str) -> bool:
    return ENV_REGEX_TOKEN in prim_path or prim_path.startswith(ENV_NS + "/") or prim_path == ENV_NS


def _asset_records(usd_path: Any, search_roots: list[Path] | None) -> list[dict[str, Any]]:
    paths = list(usd_path) if isinstance(usd_path, (list, tuple)) else [usd_path]
    records = []
    for raw in paths:
        resolved = absolutize_asset_path(str(raw), search_roots)
        record = {"path": resolved}
        record.update(file_state(resolved))
        records.append(record)
    return records


_READABLE_SKIP_KEYS = frozenset({"func", "usd_path"})


def readable_cfg(value: Any, *, depth: int = 0) -> Any:
    """JSON-friendly, deterministic view of an already-resolved config value (no hashing).

    Dataclass/configclass objects become sorted dicts (``to_dict`` when available, else
    ``vars``), tuples/lists become lists, enums/paths/numbers/strings pass through, numpy
    and torch values become lists, callables and ``usd_path``/``func`` keys are dropped
    (the ordered USD paths are recorded separately with their file states).
    """
    if depth > 8:
        return "<depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return readable_cfg(value.value, depth=depth + 1)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): readable_cfg(v, depth=depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0])) if str(k) not in _READABLE_SKIP_KEYS}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else list(value)
        return [readable_cfg(v, depth=depth + 1) for v in items]
    if callable(value) and not hasattr(value, "__dict__"):
        return "<callable>"
    for attr in ("tolist", "item"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return readable_cfg(fn(), depth=depth + 1)
            except Exception:  # noqa: BLE001
                break
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return readable_cfg(to_dict(), depth=depth + 1)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        return {str(k): readable_cfg(v, depth=depth + 1) for k, v in sorted(vars(value).items()) if not str(k).startswith("_") and str(k) not in _READABLE_SKIP_KEYS and not callable(v)}
    return repr(value)


def build_identity(
    scene_cfg: Any,
    *,
    rank: int,
    world_size: int,
    attach_mode: str = DEFAULT_ATTACH_MODE,
    sim_cfg: Any = None,
    search_roots: list[Path] | None = None,
    isaaclab_version: str | None = None,
) -> dict[str, Any]:
    """Ordinary-metadata identity of the scene this process is about to author.

    Duck-typed: ``scene_cfg`` needs ``num_envs``, ``env_spacing``, ``replicate_physics``
    and entity attributes with ``prim_path`` (and optionally ``spawn.usd_path`` which
    may be a str or an ordered list). ``_tmp_*``/private attributes are ignored.
    """
    entities: list[list[str]] = []
    usd_assets: dict[str, Any] = {}
    bound_settings: dict[str, Any] = {}
    for name, asset_cfg in sorted(vars(scene_cfg).items(), key=lambda item: item[0]):
        if name.startswith("_") or asset_cfg is None:
            continue
        prim_path = getattr(asset_cfg, "prim_path", None)
        if prim_path is None:
            continue
        entities.append([name, type(asset_cfg).__name__, str(prim_path)])
        spawn = getattr(asset_cfg, "spawn", None)
        usd_path = getattr(spawn, "usd_path", None) if spawn is not None else None
        if usd_path:
            usd_assets[name] = {
                "prim_path": str(prim_path),
                "ordered": isinstance(usd_path, (list, tuple)),
                "usd": _asset_records(usd_path, search_roots),
            }
        # The cached /World/envs specs also carry what the spawner authored FROM the cfg
        # (scale, rigid/collision/mass/articulation props, contact-report activation,
        # materials) and the init pose; a hit re-uses them, so they are part of the identity
        # for every entity whose spawn is nulled on the warm path (env-scoped prims).
        if spawn is not None and is_env_scoped_prim_path(str(prim_path)):
            bound_settings[name] = {
                "spawn": readable_cfg(spawn),
                "init_state": readable_cfg(getattr(asset_cfg, "init_state", None)),
                "collision_group": readable_cfg(getattr(asset_cfg, "collision_group", None)),
                "articulation_root_prim_path": readable_cfg(getattr(asset_cfg, "articulation_root_prim_path", None)),
            }
    identity: dict[str, Any] = {
        "schema": SCHEMA,
        "attach_mode": attach_mode,
        "rank": int(rank),
        "world_size": int(world_size),
        "num_envs": int(getattr(scene_cfg, "num_envs")),
        "env_spacing": float(getattr(scene_cfg, "env_spacing")),
        "replicate_physics": bool(getattr(scene_cfg, "replicate_physics", False)),
        "filter_collisions": bool(getattr(scene_cfg, "filter_collisions", True)),
        "physics_prim_path": str(_get(sim_cfg, "physics_prim_path", "/physicsScene")),
        "entities": sorted(entities),
        "usd_assets": usd_assets,
        "bound_settings": bound_settings,
        "isaaclab_version": isaaclab_version,
    }
    return identity


# Assets shipped INSIDE the code checkout (``gear_sonic/data/...``: the robot USD) live at a different absolute path
# with a fresh checkout mtime in every pinned worktree. For compatibility they are identified by repo-relative path +
# byte size + the last commit that touched the asset's directory tree (gear_sonic/utils/repo_asset_identity.py): a
# relocated unchanged checkout attaches, an equal-size edit or a changed sibling/referenced layer does not. When that
# identity cannot be established on either side (git failed once in Generalist 36593474 rank 1 while the other ranks
# and the same process later resolved it) the two checkouts' asset directory trees are compared byte for byte
# (repo_asset_trees_identical); only an actual difference, a rewritten recorded file or a removed checkout misses.
from gear_sonic.utils.repo_asset_identity import (  # noqa: E402
    comparable_asset_record as _pin_independent_asset_record,
    decision_scope as _identity_decision_scope,
    repo_asset_trees_identical as _repo_asset_trees_identical,
)


def _comparable_record(record: Any) -> Any:
    """Asset record as compared: repository-internal assets pin-independent; ``ctime_ns`` never part of equality."""
    rec = _pin_independent_asset_record(record)
    if isinstance(rec, Mapping) and "ctime_ns" in rec:
        rec = {k: v for k, v in rec.items() if k != "ctime_ns"}
    return rec


def _repo_asset_bytes_equal(recorded: Any, current: Any) -> bool:
    """Second chance for one repository-internal asset record pair when the git fact was unavailable on at least one
    side; two established and different facts are a real change and are never compared by bytes."""
    view_a, view_b = _comparable_record(recorded), _comparable_record(current)
    if (
        isinstance(view_a, Mapping) and isinstance(view_b, Mapping)
        and "asset_revision" in view_a and "asset_revision" in view_b
    ):
        return False
    return _repo_asset_trees_identical(recorded, current)


def comparable_identity(identity: Any) -> Any:
    """Identity view used for receipt comparison: repository-internal assets pin-independent; everything else exact
    (path + size + mtime for data-pack assets; the recorded ctime is not part of equality, so receipts written before
    it existed compare as before)."""
    if not isinstance(identity, Mapping):
        return identity
    view = dict(identity)
    assets = view.get("usd_assets")
    if isinstance(assets, Mapping):
        normalized = {}
        for name, entry in assets.items():
            if isinstance(entry, Mapping) and isinstance(entry.get("usd"), list):
                normalized[name] = dict(entry, usd=[_comparable_record(r) for r in entry["usd"]])
            else:
                normalized[name] = entry
        view["usd_assets"] = normalized
    return view


def identity_mtime_flip_tolerated(recorded: Any, identity: Any) -> int | None:
    """Number of asset records whose ONLY difference is the mtime of an otherwise identical, unmodified file; None otherwise.

    A parallel filesystem can report two mtimes for one file (a copy with preserved timestamps that was rewritten in place: MDT
    mtime vs OST mtime), so an exact path+size+mtime identity can flip between legs without any change to the
    file. A record pair is tolerated when path and size are equal, both carry ``ctime_ns`` and the ctimes are
    equal (a real modification or replacement changes the ctime too); any other difference anywhere in the
    identity returns None (ordinary miss). Receipts without ``ctime_ns`` are never tolerated.
    """
    if not isinstance(recorded, Mapping) or not isinstance(identity, Mapping):
        return None
    if set(recorded) != set(identity):
        return None
    tolerated = 0
    for key in recorded:
        a, b = recorded[key], identity[key]
        if key != "usd_assets":
            if a != b:
                return None
            continue
        if not isinstance(a, Mapping) or not isinstance(b, Mapping) or set(a) != set(b):
            return None
        for asset in a:
            ea, eb = a[asset], b[asset]
            if ea == eb:
                continue
            if not (isinstance(ea, Mapping) and isinstance(eb, Mapping)):
                return None
            ua, ub = ea.get("usd"), eb.get("usd")
            if {k: v for k, v in ea.items() if k != "usd"} != {k: v for k, v in eb.items() if k != "usd"}:
                return None
            if not (isinstance(ua, list) and isinstance(ub, list) and len(ua) == len(ub)):
                return None
            for ra, rb in zip(ua, ub):
                if ra == rb:
                    continue
                ca, cb = _comparable_record(ra), _comparable_record(rb)
                if ca == cb:
                    continue
                if not (isinstance(ra, Mapping) and isinstance(rb, Mapping)):
                    return None
                same_file = (
                    ra.get("path") == rb.get("path")
                    and ra.get("size_bytes") is not None
                    and ra.get("size_bytes") == rb.get("size_bytes")
                    and ra.get("ctime_ns") is not None
                    and ra.get("ctime_ns") == rb.get("ctime_ns")
                    and ra.get("mtime_ns") != rb.get("mtime_ns")
                )
                if not same_file:
                    return None
                tolerated += 1
    return tolerated


def same_file_by_identity(path_a: Any, path_b: Any) -> bool:
    """Two path spellings name the same file: both stat and have equal inode, size and mtime_ns.

    Two mount spellings can name the same storage (login node: symlink; container: bind mount), so an
    asset record written through one spelling must match the same asset seen through the other (user decision
    2026-10-14: alias spelling alone never invalidates a cache). Never raises.
    """
    if not isinstance(path_a, str) or not isinstance(path_b, str) or not path_a or not path_b:
        return False
    if path_a == path_b:
        return True
    try:
        sa, sb = os.stat(path_a), os.stat(path_b)
    except OSError:
        return False
    return (sa.st_ino, sa.st_size, sa.st_mtime_ns) == (sb.st_ino, sb.st_size, sb.st_mtime_ns)


def _alias_equal_values(a: Any, b: Any) -> bool:
    """Structural equality where two differing strings are equal when they name the same file (same_file_by_identity)."""
    if a == b:
        return True
    if isinstance(a, str) and isinstance(b, str):
        return same_file_by_identity(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_alias_equal_values(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_alias_equal_values(x, y) for x, y in zip(a, b))
    return False


def identity_alias_equivalent(recorded: Any, identity: Any, details: list[str] | None = None) -> int | None:
    """Number of asset records / setting values that differ ONLY by the spelling of a path naming the same file
    (inode + size + mtime_ns equal); None when any other difference exists (ordinary miss).

    Applied after exact comparison and the filesystem mtime-flip tolerance: every non-asset key must be equal exactly except
    ``bound_settings`` and ``entities`` (compared with path-alias equality for nested strings); asset records must be
    equal as compared, or differ only in ``path`` (and the never-compared ``ctime_ns``) with size and mtime equal and the
    two paths naming the same file - or be a repository-internal asset whose git identity was unavailable and whose
    asset directory tree is byte-identical in both checkouts (``details`` receives one line per such record). Real
    content, order, count or setting changes are never tolerated.
    """
    if not isinstance(recorded, Mapping) or not isinstance(identity, Mapping) or set(recorded) != set(identity):
        return None
    tolerated = 0
    for key in recorded:
        a, b = recorded[key], identity[key]
        if key in ("bound_settings", "entities"):
            if a == b:
                continue
            if not _alias_equal_values(a, b):
                return None
            tolerated += 1
            continue
        if key != "usd_assets":
            if a != b:
                return None
            continue
        if not isinstance(a, Mapping) or not isinstance(b, Mapping) or set(a) != set(b):
            return None
        for asset in a:
            ea, eb = a[asset], b[asset]
            if ea == eb:
                continue
            if not (isinstance(ea, Mapping) and isinstance(eb, Mapping)):
                return None
            if {k: v for k, v in ea.items() if k != "usd"} != {k: v for k, v in eb.items() if k != "usd"}:
                return None
            ua, ub = ea.get("usd"), eb.get("usd")
            if not (isinstance(ua, list) and isinstance(ub, list) and len(ua) == len(ub)):
                return None
            for index, (ra, rb) in enumerate(zip(ua, ub)):
                if ra == rb or _comparable_record(ra) == _comparable_record(rb):
                    continue
                if not (isinstance(ra, Mapping) and isinstance(rb, Mapping)):
                    return None
                same = (
                    ra.get("path") != rb.get("path")
                    and ra.get("size_bytes") is not None
                    and ra.get("size_bytes") == rb.get("size_bytes")
                    and ra.get("mtime_ns") == rb.get("mtime_ns")
                    and {k: v for k, v in ra.items() if k not in ("path", "ctime_ns")} == {k: v for k, v in rb.items() if k not in ("path", "ctime_ns")}
                    and same_file_by_identity(ra.get("path"), rb.get("path"))
                )
                if same:
                    tolerated += 1
                    continue
                # last resort, repository-internal assets only (data-pack records never reach it): git fact missing
                if _repo_asset_bytes_equal(ra, rb):
                    if details is not None:
                        details.append(
                            f"usd_assets.{asset}[{index}]: repository asset tree byte-identical in both checkouts, "
                            "git identity unavailable"
                        )
                    tolerated += 1
                    continue
                return None
    return tolerated


def metadata_compatible(
    metadata: Any,
    identity: dict[str, Any],
    *,
    stage_state: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Decide whether a loaded receipt may be attached. Never raises."""
    if not isinstance(metadata, dict):
        return False, "receipt is not a mapping"
    if metadata.get("schema") != SCHEMA:
        return False, f"schema {metadata.get('schema')!r} != {SCHEMA!r}"
    recorded = metadata.get("identity")
    note = ""
    # Exactly equal identities (same pin, same spelling, unchanged files) need no git fact and no tolerance branch.
    # Otherwise ONE decision scope: a git lookup that failed is not repeated by the later branches of this decision.
    if recorded != identity:
        with _identity_decision_scope():
            if comparable_identity(recorded) != comparable_identity(identity):
                tolerated = identity_mtime_flip_tolerated(recorded, identity)
                if tolerated:
                    note = f" (filesystem mtime flip tolerated on {tolerated} asset record(s): same path, size and ctime)"
                else:
                    details: list[str] = []
                    aliased = identity_alias_equivalent(recorded, identity, details)
                    if not aliased:
                        return False, "identity differs: " + describe_identity_difference(recorded, identity)
                    note = (
                        f" (path alias equivalence on {aliased} record(s)/setting(s): same file by inode, size and mtime"
                        " through another spelling" + (f"; {'; '.join(details)}" if details else "") + ")"
                    )
    origins = metadata.get("env_origins")
    if not isinstance(origins, list) or len(origins) != identity["num_envs"]:
        return False, f"env_origins length {None if not isinstance(origins, list) else len(origins)} != {identity['num_envs']}"
    if stage_state is not None:
        if (
            metadata.get("stage_size_bytes") != stage_state.get("size_bytes")
            or metadata.get("stage_mtime_ns") != stage_state.get("mtime_ns")
        ):
            return False, "stage file size/mtime differ from the receipt"
    if metadata.get("physics_scene_path") is not None and not isinstance(metadata.get("physics_scene_path"), str):
        return False, "physics_scene_path is not a string"
    return True, "ok" + note


def describe_identity_difference(recorded: Any, identity: dict[str, Any]) -> str:
    if not isinstance(recorded, dict):
        return "recorded identity missing"
    keys = sorted(set(recorded) | set(identity))
    diffs = []
    for key in keys:
        a, b = recorded.get(key), identity.get(key)
        if a == b:
            continue
        if key == "usd_assets" and isinstance(a, dict) and isinstance(b, dict):
            for asset in sorted(set(a) | set(b)):
                ra, rb = a.get(asset), b.get(asset)
                if ra == rb:
                    continue
                if isinstance(ra, dict) and isinstance(rb, dict):
                    ua, ub = ra.get("usd") or [], rb.get("usd") or []
                    if len(ua) != len(ub):
                        diffs.append(f"usd_assets.{asset}: {len(ua)} vs {len(ub)} files")
                        continue
                    for index, (xa, xb) in enumerate(zip(ua, ub)):
                        if xa != xb:
                            diffs.append(f"usd_assets.{asset}[{index}]: {xa} vs {xb}")
                            break
                else:
                    diffs.append(f"usd_assets.{asset}: present={ra is not None} vs {rb is not None}")
        else:
            diffs.append(f"{key}: {a!r} vs {b!r}")
        if len(diffs) >= 4:
            break
    return "; ".join(diffs) if diffs else "unknown"


def read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _temporary_sibling(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}{suffix}")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """temp sibling + fsync + os.replace (the only writes this module does besides the usdc)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path, ".json")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def durable_replace(temporary: Path, destination: Path) -> None:
    try:
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


class BestEffortLock:
    """flock that degrades to unlocked operation on any OSError (network-filesystem safe)."""

    def __init__(self, path: Path, *, exclusive: bool) -> None:
        self.path, self.exclusive, self.handle = path, exclusive, None

    def __enter__(self):
        if not self.exclusive and not self.path.exists():
            return self
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(self.path, "a+" if self.exclusive else "r", encoding="utf-8")
            if fcntl is not None:
                fcntl.flock(self.handle, fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH)
        except OSError:
            if self.handle is not None:
                self.handle.close()
            self.handle = None
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self.handle, fcntl.LOCK_UN)
            except OSError:
                pass
            self.handle.close()
        return False


def write_quarantine_marker(spec: AuthoredStageCacheSpec, metadata: dict[str, Any] | None, reason: str) -> Path | None:
    """Diagnostic sibling record of a failed attach (reason, stage size/mtime); the usdc/json entry is never deleted or
    renamed and - since 2026-10-15 - the record is NOT a veto: the next leg attempts the actual open/attach again and the
    existing rollback/native fallback decides (a later code fix must be able to attach an unchanged stage; the native
    build that follows a failed attach re-exports the entry anyway)."""
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    marker = spec.failed_marker_path(utc)
    payload = {
        "schema": SCHEMA,
        "rank": spec.rank,
        "stage_path": str(spec.stage_path),
        "stage_size_bytes": (metadata or {}).get("stage_size_bytes"),
        "stage_mtime_ns": (metadata or {}).get("stage_mtime_ns"),
        "reason": reason[:4000],
        "utc": utc,
        "pid": os.getpid(),
    }
    try:
        atomic_write_json(marker, payload)
    except OSError as exc:
        _log(f"quarantine marker could not be written: {exc!r}")
        return None
    return marker


def _asset_records_equivalent(ra: Any, rb: Any) -> bool:
    """Same asset as compared: exact (pin-independent, ctime ignored), the filesystem mtime flip (same path, size and ctime),
    or another spelling of the same file (same inode, size and mtime)."""
    if _comparable_record(ra) == _comparable_record(rb):
        return True
    if not (isinstance(ra, Mapping) and isinstance(rb, Mapping)):
        return False
    rest_a = {k: v for k, v in ra.items() if k not in ("path", "mtime_ns", "ctime_ns")}
    rest_b = {k: v for k, v in rb.items() if k not in ("path", "mtime_ns", "ctime_ns")}
    if rest_a != rest_b or ra.get("size_bytes") is None:
        return False
    if ra.get("path") == rb.get("path"):
        return ra.get("ctime_ns") is not None and ra.get("ctime_ns") == rb.get("ctime_ns")
    return ra.get("mtime_ns") == rb.get("mtime_ns") and same_file_by_identity(ra.get("path"), rb.get("path"))


def changed_slot_plan(recorded: Any, identity: Any) -> tuple[dict[str, list[int]] | None, str]:
    """Per-asset env ids whose bound USD differs when that is the ONLY difference between a receipt and the current identity.

    Every key other than ``usd_assets`` must be equal (``bound_settings``/``entities``: alias-equal); per asset the
    prim path / ordered flag must match and the record lists have the same length; only env-scoped ordered assets may
    differ. Returns (None, reason) for any other difference or for no difference; any number of changed envs is a plan.
    """
    if not isinstance(recorded, Mapping) or not isinstance(identity, Mapping) or set(recorded) != set(identity):
        return None, "identity keys differ"
    for key in recorded:
        if key == "usd_assets":
            continue
        a, b = recorded[key], identity[key]
        if a == b or (key in ("bound_settings", "entities") and _alias_equal_values(a, b)):
            continue
        return None, f"{key} differs"
    ua, ub = recorded.get("usd_assets"), identity.get("usd_assets")
    if not isinstance(ua, Mapping) or not isinstance(ub, Mapping) or set(ua) != set(ub):
        return None, "usd_assets entries differ"
    plan: dict[str, list[int]] = {}
    envs: set[int] = set()
    for name in sorted(ua):
        ea, eb = ua[name], ub[name]
        if ea == eb:
            continue
        if not (isinstance(ea, Mapping) and isinstance(eb, Mapping)):
            return None, f"usd_assets.{name}: not a mapping"
        if {k: v for k, v in ea.items() if k != "usd"} != {k: v for k, v in eb.items() if k != "usd"}:
            return None, f"usd_assets.{name}: prim path / ordering differ"
        la, lb = ea.get("usd"), eb.get("usd")
        if not (isinstance(la, list) and isinstance(lb, list) and len(la) == len(lb)):
            return None, f"usd_assets.{name}: {len(la) if isinstance(la, list) else '?'} vs {len(lb) if isinstance(lb, list) else '?'} files"
        # Records are compared AS COMPARED (pin-independent repository asset, filesystem mtime flip, other spelling of the
        # same file): an asset whose records differ only that way is unchanged and never vetoes the plan. Generalist
        # 36608659 ranks 6/7 (2026-10-15): the robot USD recorded through pin f42dd08 and seen through pin 63eaa91 -
        # byte-identical, no commit touching gear_sonic/data - was rejected here as "not a per-env ordered asset"
        # before its records were compared, forcing a 2600 s native rebuild for a handful of real slot changes.
        changed = [index for index, (ra, rb) in enumerate(zip(la, lb)) if not _asset_records_equivalent(ra, rb)]
        if not changed:
            continue
        if not ea.get("ordered") or not is_env_scoped_prim_path(str(ea.get("prim_path", ""))):
            return None, f"usd_assets.{name}: not a per-env ordered asset"
        plan[name] = changed
        envs.update(changed)
    if not plan:
        return None, "no per-slot difference"
    num_envs = int(identity.get("num_envs") or 0)
    if num_envs <= 0 or any(env >= num_envs for env in envs):
        return None, "slot index outside num_envs"
    return plan, f"{len(envs)} of {num_envs} env slot(s) differ in {sorted(plan)}"


def load_receipt_for_partial(
    spec: AuthoredStageCacheSpec, identity: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, list[int]] | None, str]:
    """The receipt of a stage whose ONLY difference from ``identity`` is a bounded set of per-env USD bindings.
    Returns (metadata, plan, reason) or (None, None, reason). Receipt level only (no USD access)."""
    if not spec.metadata_path.is_file() or not spec.stage_path.is_file():
        return None, None, "no entry"
    with BestEffortLock(spec.lock_path, exclusive=False):
        metadata = read_json(spec.metadata_path)
        if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
            return None, None, "receipt unreadable or other schema"
        stage_state = file_state(spec.stage_path)
        if metadata.get("stage_size_bytes") != stage_state.get("size_bytes") or metadata.get("stage_mtime_ns") != stage_state.get("mtime_ns"):
            return None, None, "stage file size/mtime differ from the receipt"
        origins = metadata.get("env_origins")
        if not isinstance(origins, list) or len(origins) != int(identity.get("num_envs") or -1):
            return None, None, "env_origins length differs"
        plan, reason = changed_slot_plan(metadata.get("identity"), identity)
    if plan is None:
        return None, None, reason
    return metadata, plan, reason


def _concrete_env_prim_path(prim_path: str, env_id: int, env_ns: str = ENV_NS) -> str:
    if ENV_REGEX_TOKEN in prim_path:
        return prim_path.replace(ENV_REGEX_TOKEN, f"{env_ns}/env_{env_id}")
    return prim_path.replace(f"{env_ns}/env_.*", f"{env_ns}/env_{env_id}")


def reauthor_env_prims(
    scene_cfg: Any, root_layer: Any, plan: Mapping[str, list[int]], identity: Mapping[str, Any], *, env_ns: str = ENV_NS
) -> list[tuple[str, int, str, str]]:
    """Re-author the env prims named by ``plan`` with the assets of the CURRENT identity: remove the cached prim spec and
    spawn the asset's own ``spawn`` (its usd list narrowed to the env's file) at the concrete env path, with the cfg's
    initial pose exactly as the asset class does natively. Raises on any asset without a callable spawn function."""
    done: list[tuple[str, int, str, str]] = []
    for name in sorted(plan):
        asset_cfg = getattr(scene_cfg, name, None)
        spawn = getattr(asset_cfg, "spawn", None)
        func = getattr(spawn, "func", None)
        if asset_cfg is None or spawn is None or not callable(func):
            raise AuthoredStageCacheError(f"{name}: no callable spawn function for the changed-slot re-author")
        records = ((identity.get("usd_assets") or {}).get(name) or {}).get("usd") or []
        prim_pattern = str(getattr(asset_cfg, "prim_path"))
        init_state = getattr(asset_cfg, "init_state", None)
        translation = getattr(init_state, "pos", None)
        orientation = getattr(init_state, "rot", None)
        usd_list = isinstance(getattr(spawn, "usd_path", None), (list, tuple))
        for env_id in plan[name]:
            new_path = str(records[int(env_id)]["path"])
            prim_path = _concrete_env_prim_path(prim_pattern, int(env_id), env_ns)
            remove_root_layer_prim(root_layer, prim_path)
            single = copy.copy(spawn)
            single.usd_path = [new_path] if usd_list else new_path
            func(prim_path, single, translation=translation, orientation=orientation)
            done.append((name, int(env_id), prim_path, new_path))
    return done


def load_compatible_metadata(
    spec: AuthoredStageCacheSpec, identity: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    """Receipt-level validation only (no USD access). Returns (metadata | None, reason)."""
    if not spec.metadata_path.is_file():
        return None, f"no receipt {spec.metadata_path}"
    if not spec.stage_path.is_file():
        return None, f"no stage {spec.stage_path}"
    with BestEffortLock(spec.lock_path, exclusive=False):
        metadata = read_json(spec.metadata_path)
        if metadata is None:
            return None, "receipt unreadable"
        stage_state = file_state(spec.stage_path)
        ok, reason = metadata_compatible(metadata, identity, stage_state=stage_state)
        if not ok:
            return None, reason
    return metadata, reason


def publish_entry(
    spec: AuthoredStageCacheSpec,
    temporary_stage: Path,
    metadata_without_stage_state: dict[str, Any],
) -> dict[str, Any]:
    """Atomic publish: usdc first (os.replace), then json; returns the final metadata."""
    with BestEffortLock(spec.lock_path, exclusive=True):
        durable_replace(temporary_stage, spec.stage_path)
        state = file_state(spec.stage_path)
        metadata = dict(metadata_without_stage_state)
        metadata["stage_path"] = str(spec.stage_path)
        metadata["stage_size_bytes"] = state["size_bytes"]
        metadata["stage_mtime_ns"] = state["mtime_ns"]
        atomic_write_json(spec.metadata_path, metadata)
    return metadata


def bind_existing_cfg(
    scene_cfg: Any,
    *,
    skip_names: set[str] | frozenset[str] = frozenset(),
) -> tuple[Any, list[str]]:
    """Deep-copy ``scene_cfg`` and set ``spawn=None`` on every env-scoped asset.

    Duck-typed on ``prim_path``/``spawn`` (ArticulationCfg, RigidObjectCfg, AssetBaseCfg,
    spawning sensors such as cameras, and the members of a RigidObjectCollectionCfg).
    Terrain (no ``spawn``), contact sensors (no ``spawn``) and global prims (lights) are
    left untouched so they are authored natively. Returns (cfg, bound asset names).
    """
    cfg = copy.deepcopy(scene_cfg)
    bound: list[str] = []
    for name, asset_cfg in vars(cfg).items():
        if name in skip_names or name.startswith("_") or asset_cfg is None:
            continue
        members = getattr(asset_cfg, "rigid_objects", None)
        if isinstance(members, dict):
            for member_name, member_cfg in members.items():
                if hasattr(member_cfg, "spawn") and is_env_scoped_prim_path(str(getattr(member_cfg, "prim_path", ""))):
                    if getattr(member_cfg, "spawn", None) is not None:
                        member_cfg.spawn = None
                        bound.append(f"{name}.{member_name}")
            continue
        if not hasattr(asset_cfg, "spawn") or not hasattr(asset_cfg, "prim_path"):
            continue
        if is_env_scoped_prim_path(str(asset_cfg.prim_path)) and asset_cfg.spawn is not None:
            asset_cfg.spawn = None
            bound.append(name)
    return cfg, bound


def _env_prim_names(num_envs: int) -> list[str]:
    return [f"env_{index}" for index in range(num_envs)]


# ----------------------------------------------------------------------------------
# USD-side helpers (pxr imported lazily; stage/layer objects are duck-typed so the CPU
# tests can pass fakes)
# ----------------------------------------------------------------------------------


def _sdf():
    from pxr import Sdf

    return Sdf


def define_world_prim(layer: Any) -> None:
    Sdf = _sdf()
    world = Sdf.CreatePrimInLayer(layer, "/World")
    world.specifier = Sdf.SpecifierDef
    world.typeName = "Xform"


def _change_block():
    """Sdf.ChangeBlock when available (batches stage change notices), else a no-op context."""
    Sdf = _sdf()
    block = getattr(Sdf, "ChangeBlock", None)
    return block() if block is not None else contextlib.nullcontext()


def copy_scopes(source_layer: Any, destination_layer: Any, *, require_collisions: bool = False) -> list[str]:
    """Sdf.CopySpec of /World/envs (required) and /World/collisions (optional) between layers."""
    Sdf = _sdf()
    copied: list[str] = []
    for prim_path in CACHED_SCOPES:
        if source_layer.GetPrimAtPath(prim_path) is None:
            if prim_path == "/World/envs" or require_collisions:
                raise AuthoredStageCacheError(f"source layer has no {prim_path} prim spec")
            continue
        if not Sdf.CopySpec(source_layer, Sdf.Path(prim_path), destination_layer, Sdf.Path(prim_path)):
            raise AuthoredStageCacheError(f"Sdf.CopySpec failed for {prim_path}")
        copied.append(prim_path)
    return copied


def remove_root_layer_prim(root_layer: Any, prim_path: str) -> None:
    """Remove one prim spec from ``root_layer`` (Sdf level); silently ok when absent."""
    Sdf = _sdf()
    if root_layer.GetPrimAtPath(prim_path) is None:
        return
    errors = []
    try:
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Remove(Sdf.Path(prim_path)))
        if root_layer.Apply(edit) and root_layer.GetPrimAtPath(prim_path) is None:
            return
        errors.append("BatchNamespaceEdit.Remove did not remove the spec")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"BatchNamespaceEdit: {exc!r}")
    try:
        spec = root_layer.GetPrimAtPath(prim_path)
        parent = spec.nameParent if spec.nameParent is not None else root_layer.pseudoRoot
        del parent.nameChildren[spec.name]
        if root_layer.GetPrimAtPath(prim_path) is None:
            return
        errors.append("nameChildren delete did not remove the spec")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"nameChildren delete: {exc!r}")
    raise AuthoredStageCacheError(f"could not remove {prim_path} from the live root layer: {errors}")


def snapshot_toplevel(root_layer: Any) -> dict[str, list[str]]:
    """Names of root prims and of /World children in the live root layer (a handful of strings)."""
    try:
        root_names = [spec.name for spec in root_layer.rootPrims]
    except Exception:  # noqa: BLE001
        root_names = []
    world = root_layer.GetPrimAtPath("/World")
    try:
        world_names = [spec.name for spec in world.nameChildren] if world is not None else []
    except Exception:  # noqa: BLE001
        world_names = []
    return {"root": root_names, "world": world_names}


def rollback_root_layer(stage: Any, root_layer: Any, snapshot: dict[str, list[str]]) -> list[str]:
    """Remove the cached scopes and anything newly authored at the top level since ``snapshot``.

    Verifies both cached scopes are gone from the composed stage; raises when not.
    """
    Sdf = _sdf()
    removed: list[str] = []
    to_remove = list(CACHED_SCOPES)
    world = root_layer.GetPrimAtPath("/World")
    if world is not None:
        try:
            for spec in list(world.nameChildren):
                path = f"/World/{spec.name}"
                if spec.name not in snapshot.get("world", []) and path not in to_remove:
                    to_remove.append(path)
        except Exception:  # noqa: BLE001
            pass
    try:
        for spec in list(root_layer.rootPrims):
            path = f"/{spec.name}"
            if spec.name not in snapshot.get("root", []) and spec.name != "World":
                to_remove.append(path)
    except Exception:  # noqa: BLE001
        pass
    change_block = getattr(Sdf, "ChangeBlock", None)
    if change_block is not None:
        with change_block():
            for path in to_remove:
                remove_root_layer_prim(root_layer, path)
                removed.append(path)
    else:
        for path in to_remove:
            remove_root_layer_prim(root_layer, path)
            removed.append(path)
    still_present = []
    for path in CACHED_SCOPES:
        if root_layer.GetPrimAtPath(path) is not None:
            still_present.append(path + " (root layer spec)")
            continue
        prim = stage.GetPrimAtPath(path)
        if prim is not None and bool(prim.IsValid()):
            still_present.append(path + " (composed prim)")
    if still_present:
        raise AuthoredStageCacheError(
            "rollback verification failed; refusing to continue on a half-attached stage: "
            f"{still_present}"
        )
    return removed


def make_asset_paths_absolute(layer: Any, *, resolver_context: Any, search_roots: list[Path] | None) -> dict[str, int]:
    """Rewrite every relative asset path in the cache layer to an absolute local path.

    Absolute paths and URLs pass through unchanged. Unresolvable relative paths raise
    (the caller turns that into an export failure; the live stage is never touched).
    """
    from pxr import Ar, UsdUtils

    resolver = Ar.GetResolver()
    counts = {"visited": 0, "rewritten": 0, "urls": 0, "absolute": 0}
    unresolved: list[str] = []
    roots = search_roots or default_search_roots()

    def resolve(asset_path: str) -> str:
        counts["visited"] += 1
        if not asset_path:
            return asset_path
        if is_url(asset_path):
            counts["urls"] += 1
            return asset_path
        if Path(asset_path).expanduser().is_absolute():
            counts["absolute"] += 1
            return asset_path
        resolved = ""
        try:
            resolved = str(resolver.Resolve(asset_path))
        except Exception:  # noqa: BLE001
            resolved = ""
        if resolved and Path(resolved).is_absolute() and Path(resolved).is_file():
            counts["rewritten"] += 1
            return str(Path(resolved))
        for root in roots:
            probe = Path(root) / asset_path
            if probe.is_file():
                counts["rewritten"] += 1
                return str(probe)
        unresolved.append(asset_path)
        return asset_path

    if resolver_context is not None:
        with Ar.ResolverContextBinder(resolver_context):
            UsdUtils.ModifyAssetPaths(layer, resolve)
    else:
        UsdUtils.ModifyAssetPaths(layer, resolve)
    if unresolved:
        raise AuthoredStageCacheError(f"unresolved relative asset paths in the exported layer: {sorted(set(unresolved))[:4]}")
    return counts


def read_invert_filter_attr(stage: Any, physics_scene_path: str | None) -> bool | None:
    """Value of physxScene:invertCollisionGroupFilter on the live physics-scene prim (None if unauthored)."""
    if not physics_scene_path:
        return None
    try:
        prim = stage.GetPrimAtPath(physics_scene_path)
        if prim is None or not prim.IsValid():
            return None
        attr = prim.GetAttribute(INVERT_FILTER_ATTR)
        if not attr or not attr.IsValid() or not attr.HasAuthoredValue():
            return None
        return bool(attr.Get())
    except Exception:  # noqa: BLE001
        return None


def apply_invert_filter_attr(stage: Any, physics_scene_path: str, value: bool) -> None:
    """Re-author the collision-filter inversion that the cold ``filter_collisions`` wrote on /physicsScene."""
    from pxr import PhysxSchema

    prim = stage.GetPrimAtPath(physics_scene_path)
    if prim is None or not prim.IsValid():
        raise AuthoredStageCacheError(f"live physics scene prim {physics_scene_path} not found")
    api = PhysxSchema.PhysxSceneAPI(prim)
    if not api:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
    api.CreateInvertCollisionGroupFilterAttr().Set(bool(value))


def _discard_entities(scene: Any) -> None:
    """Best-effort: unsubscribe timeline callbacks of wrappers created by a failed attach."""
    for attr in ("_articulations", "_deformable_objects", "_rigid_objects", "_rigid_object_collections", "_sensors", "_surface_grippers"):
        entities = getattr(scene, attr, None) or {}
        for entity in list(entities.values()):
            clear = getattr(entity, "_clear_callbacks", None)
            if callable(clear):
                try:
                    clear()
                except Exception:  # noqa: BLE001
                    pass
        setattr(scene, attr, {})
    scene._extras = {}
    scene._terrain = None
    # wrapper-added instance flags (gear_sonic.physx_replication) must not survive into the native retry
    scene.__dict__.pop("_physx_replication_done", None)


def _isaaclab_version() -> str | None:
    try:
        import isaaclab

        return str(getattr(isaaclab, "__version__", None) or None)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------
# Export (cold path)
# ----------------------------------------------------------------------------------


def repo_asset_root_map(recorded: Any, identity: Any) -> dict[str, str]:
    """{old checkout root: current checkout root} for every repository-internal asset record (``.../gear_sonic/data/...``)
    whose checkout root differs between the receipt and the current identity. Only called after the receipt was accepted
    as equivalent, i.e. the current checkout's asset tree is the same asset (git fact / byte-identical)."""
    from gear_sonic.utils.repo_asset_identity import split_repo_asset_path

    roots: dict[str, str] = {}
    try:
        ua, ub = recorded.get("usd_assets", {}), identity.get("usd_assets", {})
        for name in ua:
            la = (ua.get(name) or {}).get("usd") or []
            lb = (ub.get(name) or {}).get("usd") or []
            for ra, rb in zip(la, lb):
                pa = split_repo_asset_path(str((ra or {}).get("path") or ""))
                pb = split_repo_asset_path(str((rb or {}).get("path") or ""))
                if pa and pb and pa[1] == pb[1] and pa[0] != pb[0]:
                    roots[pa[0]] = pb[0]
    except (AttributeError, TypeError):
        return {}
    return roots


def rebase_repo_asset_references(layer: Any, recorded: Any, identity: Any) -> int:
    """Lightweight reference rebase at attach time (direct human 2026-10-15: a directory change must never repeat
    work): the cached root-copy layer references repository-internal assets (the robot USD tree) by the ABSOLUTE
    path of the pin that exported it. In the OPENED layer only (never saved: the cache file is untouched), every
    external reference under an old checkout root is re-pointed to the same repo-relative asset under the current
    checkout root, so the live stage composes the current pin's identical asset and the cache no longer depends on
    the exporting pin's directory. Pack-asset references (data packs) are not touched. Returns the number of
    references rebased; never raises (0 on any problem -> the old absolute references stay, as before)."""
    roots = repo_asset_root_map(recorded, identity)
    if not roots:
        return 0
    try:
        getter = getattr(layer, "GetCompositionAssetDependencies", None) or getattr(layer, "GetExternalReferences", None)
        updater = getattr(layer, "UpdateCompositionAssetDependency", None) or getattr(layer, "UpdateExternalReference", None)
        if getter is None or updater is None:
            return 0
        count = 0
        for ref in list(getter()):
            text = str(ref)
            for old_root, new_root in roots.items():
                if text.startswith(old_root + "/"):
                    new = new_root + text[len(old_root):]
                    if updater(text, new):
                        count += 1
                    break
        return count
    except Exception as exc:  # noqa: BLE001 - optional accelerator detail; the attach proceeds with the recorded paths
        _log(f"asset reference rebase skipped: {type(exc).__name__}: {exc}")
        return 0


def export_authored_stage(
    scene: Any,
    spec: AuthoredStageCacheSpec,
    identity: dict[str, Any],
    *,
    search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Copy the two scopes of the live root layer into rank_<r>.usdc and publish rank_<r>.json.

    Returns a status dict; raises only AuthoredStageCacheError/OSError (caller logs, never
    propagates). Nothing in the live stage is modified.
    """
    Sdf = _sdf()
    spec.cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(spec.stage_path, ".usdc")
    try:
        layer = Sdf.Layer.CreateNew(str(temporary))
        if layer is None:
            raise AuthoredStageCacheError(f"Sdf.Layer.CreateNew failed for {temporary}")
        root_layer = scene.stage.GetRootLayer()
        define_world_prim(layer)
        copied = copy_scopes(root_layer, layer)
        resolver_context = None
        try:
            resolver_context = scene.stage.GetPathResolverContext()
        except Exception:  # noqa: BLE001
            resolver_context = None
        path_counts = make_asset_paths_absolute(layer, resolver_context=resolver_context, search_roots=search_roots)
        layer.defaultPrim = "World"
        if not layer.Save():
            raise AuthoredStageCacheError(f"layer.Save failed for {temporary}")
        # (the per-env prim-spec re-check of the layer just written was verification-only work: removed, user
        #  decision 2026-10-14 - the receipt records the stage size/mtime and the next leg opens the layer itself)
        origins = getattr(scene, "_default_env_origins", None)
        if origins is None:
            raise AuthoredStageCacheError("scene has no _default_env_origins (replicate_physics scene?)")
        env_origins = [[float(v) for v in row] for row in origins.detach().cpu().tolist()]
        if len(env_origins) != spec.num_envs:
            raise AuthoredStageCacheError(f"{len(env_origins)} env origins for {spec.num_envs} envs")
        physics_scene_path = getattr(scene, "_physics_scene_path", None) or identity.get("physics_prim_path")
        invert = read_invert_filter_attr(scene.stage, physics_scene_path)
        metadata = {
            "schema": SCHEMA,
            "identity": identity,
            "env_origins": env_origins,
            "copied_scopes": copied,
            "physics_scene_path": physics_scene_path,
            "invert_collision_group_filter": invert,
            "asset_path_counts": path_counts,
            "written_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "writer_pid": os.getpid(),
        }
        layer = None
        published = publish_entry(spec, temporary, metadata)
        return {"stage_bytes": published.get("stage_size_bytes"), "copied": copied, "path_counts": path_counts}
    finally:
        # Only this process' own in-flight temp sibling is removed (os.replace consumed it on success).
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


# ----------------------------------------------------------------------------------
# Cached scene class
# ----------------------------------------------------------------------------------


def make_cached_scene_class(
    base_class: type,
    spec: AuthoredStageCacheSpec,
    identity: dict[str, Any],
    *,
    search_roots: list[Path] | None = None,
) -> type:
    """Subclass of InteractiveScene: warm = attach + bind existing; miss = native + export.

    Must NOT override ``_add_entities_from_cfg`` (physx_replication / scene_timing wrap the
    base attribute and must run on the warm path too).
    """

    class AuthoredStageCachedInteractiveScene(base_class):  # type: ignore[misc,valid-type]
        """InteractiveScene that binds wrappers to a cached authored root-copy layer."""

        _authored_stage_spec = spec

        def __init__(self, cfg):
            started = time.perf_counter()
            prepare_started = time.perf_counter()
            metadata, reason = load_compatible_metadata(spec, identity)
            if metadata is None:
                partial_metadata, plan, partial_reason = load_receipt_for_partial(spec, identity)
                if partial_metadata is not None and not bool(getattr(cfg, "replicate_physics", False)):
                    # the receipt differs only in a bounded set of per-env USD bindings: attach the cached layer and
                    # re-author just those env prims instead of rebuilding the whole rank natively
                    self._authored_stage_partial_reauthor(cfg, partial_metadata, plan, started, partial_reason)
                    return
                # The already-computed reasons are printed NOW (the 'mode=miss' line follows only after the native
                # build, up to ~45 min later): same information, no extra check.
                _log(f"mode=miss_selected rank={spec.rank} reason={reason!r} partial={partial_reason!r} stage={spec.stage_path}")
                # Any miss/incompatibility/IO problem enters the native path; read_only only
                # suppresses the export (an optional cache can never make training depend on a receipt).
                self._authored_stage_native_then_export(cfg, started, reason, export=spec.mode != "read_only")
                return
            if bool(getattr(cfg, "replicate_physics", False)):
                self._authored_stage_native_then_export(cfg, started, "cfg.replicate_physics=True is not a heterogeneous scene")
                return
            # ---- validation before any stage mutation -------------------------------
            try:
                stage, root_layer, cache_layer, bound_cfg = self._authored_stage_prepare(cfg, metadata)
                prepare_seconds = time.perf_counter() - prepare_started
            except StageNotFreshError:
                raise
            except AuthoredStageCacheError as exc:
                _log(f"mode=incompatible rank={spec.rank} reason={exc}")
                self._authored_stage_native_then_export(cfg, started, str(exc), export=spec.mode != "read_only")
                return
            except Exception as exc:  # noqa: BLE001
                _log(f"mode=incompatible rank={spec.rank} reason={exc!r}")
                self._authored_stage_native_then_export(cfg, started, repr(exc), export=spec.mode != "read_only")
                return
            # ---- mutation: copy scopes into the live root layer ----------------------
            snapshot = snapshot_toplevel(root_layer)
            attach_started = time.perf_counter()
            try:
                # Batch the copy into ONE change notification: copying into the live root layer
                # without an Sdf.ChangeBlock fires per-prim notices and recomposition (measured
                # 20.8 s for 64 envs on smoke leg 1304223, i.e. as slow as native spawning).
                with _change_block():
                    if root_layer.GetPrimAtPath("/World") is None:
                        define_world_prim(root_layer)
                    copy_scopes(cache_layer, root_layer)
                copy_seconds = time.perf_counter() - attach_started
                # (the composed per-env IsValid traversal after the copy was verification-only: removed, user decision
                #  2026-10-14; the entity binding below resolves the env prims physics actually needs)
                check_seconds = 0.0
                attach_seconds = time.perf_counter() - attach_started
                self._authored_stage_set_carb_flag()
                bind_started = time.perf_counter()
                filter_seconds = 0.0
                if self._is_scene_setup_from_cfg():
                    self._add_entities_from_cfg()
                    bind_seconds = time.perf_counter() - bind_started
                    # /World/collisions is NOT cached: the native filter authors the per-env
                    # collision groups and the live physics-scene invert attribute every leg.
                    filter_started = time.perf_counter()
                    if (not self.cfg.replicate_physics and self.cfg.filter_collisions) or self.device == "cpu":
                        self.filter_collisions(self._global_prim_paths)
                    filter_seconds = time.perf_counter() - filter_started
                else:
                    bind_seconds = time.perf_counter() - bind_started
            except Exception as exc:  # noqa: BLE001
                self._authored_stage_rollback(cfg, stage, root_layer, snapshot, metadata, exc, started)
                return
            _log(
                f"mode=hit rank={spec.rank} attach_seconds={attach_seconds:.3f} "
                f"prepare_seconds={prepare_seconds:.3f} copy_seconds={copy_seconds:.3f} "
                f"compose_check_seconds={check_seconds:.3f} bind_seconds={bind_seconds:.3f} "
                f"filter_seconds={filter_seconds:.3f} "
                f"total_seconds={time.perf_counter() - started:.3f} envs={spec.num_envs} "
                f"bound={len(getattr(self, '_authored_stage_bound', []))} stage={spec.stage_path}"
                + ("" if reason == "ok" else f" receipt={reason}")
            )

        # -- helpers ------------------------------------------------------------------

        def _authored_stage_prepare(self, cfg, metadata):
            import torch
            from isaacsim.core.cloner import GridCloner

            from isaaclab.sim import SimulationContext
            from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id

            Sdf = _sdf()
            # the receipt lookup already selected this stage file (size/mtime recorded at its atomic publication); the open
            # itself is the only thing left to do - no second stat comparison, no per-env prim-spec scan (user decision
            # 2026-10-14: a layer that cannot be opened or lacks /World/envs fails at open / copy_scopes -> native)
            cache_layer = Sdf.Layer.FindOrOpen(str(spec.stage_path))
            if cache_layer is None:
                raise AuthoredStageCacheError("cache layer could not be opened")
            rebased = rebase_repo_asset_references(cache_layer, metadata.get("identity") or {}, identity)
            if rebased:
                _log(f"rank={spec.rank} rebased {rebased} repository asset reference(s) from the exporting pin to the current checkout (in memory)")
            bound_cfg, bound = bind_existing_cfg(cfg)
            bound_cfg.validate()
            self.cfg = bound_cfg
            self._authored_stage_bound = bound
            self._terrain = None
            self._articulations = dict()
            self._deformable_objects = dict()
            self._rigid_objects = dict()
            self._rigid_object_collections = dict()
            self._sensors = dict()
            self._surface_grippers = dict()
            self._extras = dict()
            self.sim = SimulationContext.instance()
            self.stage = get_current_stage()
            self.stage_id = get_current_stage_id()
            self._physics_scene_path = None
            self.cloner = GridCloner(spacing=self.cfg.env_spacing, stage=self.stage)
            self.env_prim_paths = [f"{self.env_ns}/{name}" for name in _env_prim_names(self.cfg.num_envs)]
            self._ALL_INDICES = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
            self._default_env_origins = torch.tensor(metadata["env_origins"], device=self.device, dtype=torch.float32)
            self._global_prim_paths = list()
            root_layer = self.stage.GetRootLayer()
            for prim_path in CACHED_SCOPES:
                if root_layer.GetPrimAtPath(prim_path) is not None:
                    raise StageNotFreshError(f"live root layer already contains {prim_path}; stage is not fresh")
            physics_scene_path = metadata.get("physics_scene_path")
            if physics_scene_path:
                prim = self.stage.GetPrimAtPath(physics_scene_path)
                prim_ok = prim is not None and bool(prim.IsValid())
                if metadata.get("invert_collision_group_filter") and not prim_ok:
                    raise AuthoredStageCacheError(f"live physics scene prim {physics_scene_path} missing before attach")
                if prim_ok:
                    self._physics_scene_path = physics_scene_path
            return self.stage, root_layer, cache_layer, bound_cfg

        def _authored_stage_partial_reauthor(self, cfg, metadata, plan, started, reason):
            prepare_started = time.perf_counter()
            try:
                stage, root_layer, cache_layer, bound_cfg = self._authored_stage_prepare(cfg, metadata)
                prepare_seconds = time.perf_counter() - prepare_started
            except StageNotFreshError:
                raise
            except Exception as exc:  # noqa: BLE001
                _log(f"mode=incompatible rank={spec.rank} reason={exc!r}")
                self._authored_stage_native_then_export(cfg, started, repr(exc), export=spec.mode != "read_only")
                return
            snapshot = snapshot_toplevel(root_layer)
            attach_started = time.perf_counter()
            try:
                with _change_block():
                    if root_layer.GetPrimAtPath("/World") is None:
                        define_world_prim(root_layer)
                    copy_scopes(cache_layer, root_layer)
                copy_seconds = time.perf_counter() - attach_started
                reauthor_started = time.perf_counter()
                reauthored = reauthor_env_prims(cfg, root_layer, plan, identity)   # cfg = the original (spawns intact)
                reauthor_seconds = time.perf_counter() - reauthor_started
                self._authored_stage_set_carb_flag()
                bind_started = time.perf_counter()
                filter_seconds = 0.0
                if self._is_scene_setup_from_cfg():
                    self._add_entities_from_cfg()
                    bind_seconds = time.perf_counter() - bind_started
                    filter_started = time.perf_counter()
                    if (not self.cfg.replicate_physics and self.cfg.filter_collisions) or self.device == "cpu":
                        self.filter_collisions(self._global_prim_paths)
                    filter_seconds = time.perf_counter() - filter_started
                else:
                    bind_seconds = time.perf_counter() - bind_started
            except Exception as exc:  # noqa: BLE001
                self._authored_stage_rollback(cfg, stage, root_layer, snapshot, metadata, exc, started)
                return
            export_note = "skipped_read_only"
            if spec.mode != "read_only":
                try:
                    status = export_authored_stage(self, spec, identity, search_roots=search_roots)
                    export_note = f"published stage_bytes={status.get('stage_bytes') or 0}"
                except Exception as exc:  # noqa: BLE001 - the attached scene is complete; the next leg re-authors again
                    export_note = f"failed error={exc!r}"
            _log(
                f"mode=partial_hit rank={spec.rank} changed_envs={len({env for _, env, _, _ in reauthored})} "
                f"reauthored_prims={len(reauthored)} entities={sorted(plan)} prepare_seconds={prepare_seconds:.3f} "
                f"copy_seconds={copy_seconds:.3f} reauthor_seconds={reauthor_seconds:.3f} bind_seconds={bind_seconds:.3f} "
                f"filter_seconds={filter_seconds:.3f} total_seconds={time.perf_counter() - started:.3f} envs={spec.num_envs} "
                f"export={export_note} reason={reason!r} stage={spec.stage_path}"
            )

        def _authored_stage_set_carb_flag(self):
            try:
                import carb

                carb.settings.get_settings().set_bool("/isaaclab/spawn/multi_assets", True)
            except Exception:  # noqa: BLE001 - informational flag only
                pass

        def _authored_stage_native_then_export(self, cfg, started, reason, export=True):
            native_started = time.perf_counter()
            super().__init__(cfg)
            native_seconds = time.perf_counter() - native_started
            export_started = time.perf_counter()
            status: dict[str, Any] = {}
            error = None
            if export:
                try:
                    status = export_authored_stage(self, spec, identity, search_roots=search_roots)
                except Exception as exc:  # noqa: BLE001 - export never breaks a complete native scene
                    error = exc
            export_seconds = time.perf_counter() - export_started
            line = (
                f"mode={'miss' if export else 'miss_read_only'} rank={spec.rank} native_seconds={native_seconds:.3f} "
                f"export_seconds={export_seconds:.3f} stage_bytes={status.get('stage_bytes') or 0} "
                f"reason={reason!r} stage={spec.stage_path}"
            )
            if error is not None:
                line += f" export=failed error={error!r}"
            _log(line)

        def _authored_stage_rollback(self, cfg, stage, root_layer, snapshot, metadata, exc, started):
            _log(f"attach failed after stage mutation rank={spec.rank}: {exc!r}; rolling back")
            _discard_entities(self)
            removed = rollback_root_layer(stage, root_layer, snapshot)  # raises when not verifiable
            marker = write_quarantine_marker(spec, metadata, f"{type(exc).__name__}: {exc}")
            self._physics_scene_path = None
            native_started = time.perf_counter()
            super().__init__(cfg)
            _log(
                f"mode=rollback_native rank={spec.rank} removed={removed} marker={marker} "
                f"native_seconds={time.perf_counter() - native_started:.3f} "
                f"total_seconds={time.perf_counter() - started:.3f} reason={exc!r}"
            )

    AuthoredStageCachedInteractiveScene.__name__ = "AuthoredStageCachedInteractiveScene"
    AuthoredStageCachedInteractiveScene.__qualname__ = "AuthoredStageCachedInteractiveScene"
    return AuthoredStageCachedInteractiveScene


# ----------------------------------------------------------------------------------
# Install / restore
# ----------------------------------------------------------------------------------


def install_authored_stage_cache(
    raw_cfg: Any,
    *,
    env_cfg: Any,
    rank: int,
    world_size: int,
) -> Callable[[], None]:
    """Swap ``InteractiveScene`` for the cached subclass for one env construction.

    ``raw_cfg`` is ``manager_env.config.authored_stage_cache`` (None/absent or
    ``mode: disabled`` -> no-op). ``env_cfg`` is the instantiated ManagerBasedRLEnvCfg
    (identity built from ``env_cfg.scene`` / ``env_cfg.sim``). Returns the restore
    callable; call it in the ``finally`` after the env is constructed.
    """
    scene_cfg = getattr(env_cfg, "scene")
    spec = AuthoredStageCacheSpec.from_mapping(
        raw_cfg, rank=rank, world_size=world_size, num_envs=int(getattr(scene_cfg, "num_envs"))
    )
    if spec.mode == "disabled":
        return lambda: None
    if spec.requested_attach_mode != spec.attach_mode:
        _log(f"attach_mode={spec.requested_attach_mode} is mapped to {spec.attach_mode} (root_transfer is not ported; the live /physicsScene stays native)")
    search_roots = default_search_roots()
    identity_started = time.perf_counter()
    identity = build_identity(
        scene_cfg,
        rank=spec.rank,
        world_size=spec.world_size,
        attach_mode=spec.attach_mode,
        sim_cfg=getattr(env_cfg, "sim", None),
        search_roots=search_roots,
        isaaclab_version=_isaaclab_version(),
    )
    import isaaclab.scene as scene_module
    from isaaclab.envs import manager_based_env

    base_class = manager_based_env.InteractiveScene
    cached = make_cached_scene_class(base_class, spec, identity, search_roots=search_roots)
    original_env_attr = manager_based_env.InteractiveScene
    original_scene_attr = scene_module.InteractiveScene
    manager_based_env.InteractiveScene = cached
    scene_module.InteractiveScene = cached

    def restore() -> None:
        manager_based_env.InteractiveScene = original_env_attr
        scene_module.InteractiveScene = original_scene_attr

    slots = {name: len(entry.get("usd") or []) for name, entry in identity["usd_assets"].items()}
    _log(
        f"installed mode={spec.mode} attach_mode={spec.attach_mode} rank={spec.rank} world_size={spec.world_size} "
        f"num_envs={spec.num_envs} identity_seconds={time.perf_counter() - identity_started:.3f} "
        f"usd_assets={slots} cache_dir={spec.cache_dir}"
    )
    return restore


__all__ = [
    "SCHEMA",
    "SUPPORTED_MODES",
    "AuthoredStageCacheError",
    "AuthoredStageCacheSpec",
    "StageNotFreshError",
    "absolutize_asset_path",
    "atomic_write_json",
    "bind_existing_cfg",
    "build_identity",
    "describe_identity_difference",
    "export_authored_stage",
    "rebase_repo_asset_references",
    "repo_asset_root_map",
    "file_state",
    "install_authored_stage_cache",
    "is_env_scoped_prim_path",
    "load_compatible_metadata",
    "make_cached_scene_class",
    "metadata_compatible",
    "publish_entry",
    "read_json",
    "rollback_root_layer",
    "write_quarantine_marker",
]
