"""Run-local cache for one rank's resolved roster/scene plan (metadata keyed).

Lazy first-warmup flavor (2026-10-13): ``read_write`` tries a compatible cached
plan; on a miss it runs the unchanged resolver and atomically publishes the
plan JSON plus a receipt bound to the manifest file identity (inode, size,
mtime), rank, world size, env layout and clock. The manifest is keyed by inode
rather than by its absolute path so that two spellings of the same network
filesystem tree (for example a login-node symlink and a container bind mount)
resolve to the same entry; the path
actually seen is recorded in the receipt for humans only. Cache IO failures are reported in the returned
status and never replace a successful native resolution. The cached object is
the plain Python plan only; it does not keep a live PhysX scene across jobs.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


CACHE_CONTRACT = "rank_scene_plan_metadata_cache_new8_v2"
# v1 receipts (written before 2026-10-13T16Z) keyed the manifest by absolute path. They
# stay readable only when every alias-free field matches AND the recorded path is the same
# file under a configured mount-alias mapping (full normalized
# path compared after alias canonicalization; same inode whenever the recorded path is
# visible here). Anything else is a native miss, never a basename match.
_LEGACY_CONTRACT_V1 = "rank_scene_plan_metadata_cache_new8_v1"
# Deployment-specific mount-alias prefixes ((alias, canonical) pairs, longest first). The public
# release ships none; add pairs here if the same storage is visible under two spellings.
_KNOWN_ALIAS_PREFIXES: tuple[tuple[str, str], ...] = ()


def _canonical_alias(path: str) -> str:
    """Normalize ``path`` and map configured mount aliases onto their canonical form."""
    normalized = os.path.normpath(path)
    for alias, canonical in _KNOWN_ALIAS_PREFIXES:
        if normalized == alias or normalized.startswith(alias + "/"):
            return canonical + normalized[len(alias):]
    return normalized
SUPPORTED_MODES = frozenset({"disabled", "read_write", "read_only"})
_RESOLVER_TAG = "gear_sonic.rp2.joint_roster.resolve_rank_local_roster"


class RankScenePlanCacheMiss(FileNotFoundError):
    """Kept for callers that still catch it; no longer raised - a miss in any mode resolves natively."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON mapping: {path}")
    return value


def build_rank_scene_plan_identity(
    manifest_path: str | Path,
    *,
    num_envs: int,
    rank: int,
    world_size: int,
    runtime_target_fps_hz: float | None,
) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    stat = manifest.stat()
    identity = {
        "contract": CACHE_CONTRACT,
        "resolver": _RESOLVER_TAG,
        # Alias-stable file identity (same file through two mount spellings):
        # name + inode + size + mtime. No content hashing.
        "manifest_name": manifest.name,
        "manifest_inode": int(stat.st_ino),
        "manifest_size_bytes": int(stat.st_size),
        "manifest_mtime_ns": int(stat.st_mtime_ns),
        "num_envs": int(num_envs),
        "rank": int(rank),
        "world_size": int(world_size),
        "runtime_target_fps_hz": None if runtime_target_fps_hz is None else float(runtime_target_fps_hz),
    }
    # Runtime slot-mapping mode (gear_sonic/rp2/unified_cohort.py): the cached plan
    # is the resolved slot view itself, so a plan computed for the default
    # contiguous mapping must never be reused for a repeated/interleaved mapping
    # of the same manifest and env count. Default mapping keeps the original key.
    mapping = _slot_mapping_mode()
    identity["slot_repeat"] = int(mapping["slot_repeat"])
    identity["rank_interleave"] = bool(mapping["rank_interleave"])
    suffix = ""
    if identity["rank_interleave"]:
        suffix += "-il"
    if identity["slot_repeat"] != 1:
        suffix += f"-rep{identity['slot_repeat']}"
    identity["cache_key"] = f"plan-v1-w{int(world_size)}-r{int(rank)}-env{int(num_envs)}{suffix}"
    return identity


class InvalidSlotMappingMode(ValueError):
    """UNIFIED_COHORT_SLOT_REPEAT / RANK_INTERLEAVE could not be parsed; the cache must not
    serve any plan - the native resolver raises the authoritative error."""


def _slot_mapping_mode() -> dict[str, Any]:
    """Current runtime slot-mapping options (shared definition with unified_cohort)."""
    from gear_sonic.rp2.unified_cohort import rank_interleave_enabled, slot_repeat_factor

    try:
        return {"slot_repeat": slot_repeat_factor(), "rank_interleave": rank_interleave_enabled()}
    except ValueError as exc:
        raise InvalidSlotMappingMode(str(exc)) from exc


class _BestEffortLock:
    def __init__(self, path: Path, *, exclusive: bool) -> None:
        self.path, self.exclusive, self.handle = path, exclusive, None

    def __enter__(self):
        try:
            if not self.exclusive and not self.path.exists():
                return self  # a reader never creates lock files (read_only callers may be on a login node)
            self.handle = self.path.open("a+" if self.exclusive else "r", encoding="utf-8")
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


RANK_INPUTS_SCHEMA = "rank_inputs_v1"
_MANIFEST_FILE_KEYS = ("manifest_name", "manifest_inode", "manifest_size_bytes", "manifest_mtime_ns")


@functools.lru_cache(maxsize=65536)
def _canonical_directory(head: str) -> str:
    try:
        return os.path.realpath(head)
    except OSError:
        return head


def _canonical_file_path(path: Any) -> str:
    """One spelling for the same file reached through two mount spellings (realpath of the directory, cached)."""
    text = str(path or "")
    if not text.startswith("/"):
        return text
    head, tail = os.path.split(text)
    return os.path.join(_canonical_directory(head), tail)


def _file_identity(path: Any) -> dict[str, int] | None:
    try:
        st = os.stat(str(path))
    except OSError:
        return None
    return {"inode": int(st.st_ino), "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def rank_slot_inputs(
    manifest_path: str | Path, *, rank: int, world_size: int, num_envs: int
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The inputs the native resolver actually consumes for ONE rank under the default contiguous mapping, read from the
    manifest itself: the manifest scalars and, for the rank's env slots, source / asset_signature / object_key / the two
    USDs (canonical spelling) / compatible aliases / representative alias / the active-surface file identity.
    Returns (inputs, rebase) or None when the manifest cannot be read in that shape (ordinary native path).
    ``rebase`` carries what a reused plan must be re-pointed to: this layout's root and the rank's surface paths."""
    try:
        path = Path(manifest_path).expanduser().resolve(strict=True)
        manifest = json.loads(path.read_text())
        slots = manifest.get("slots")
        if not isinstance(slots, list):
            return None
        start, stop = int(rank) * int(num_envs), (int(rank) + 1) * int(num_envs)
        if stop > len(slots):
            return None
        owned = slots[start:stop]
        inputs = {
            "schema": RANK_INPUTS_SCHEMA,
            "scalars": {
                "num_envs": manifest.get("num_envs"), "world_size": manifest.get("world_size"),
                "num_envs_per_rank": manifest.get("num_envs_per_rank"), "target_fps_hz": manifest.get("target_fps_hz"),
                "catalog_total_unique_motions": manifest.get("catalog_total_unique_motions"),
                "catalog_source_counts": manifest.get("catalog_source_counts"),
                "record_count": len(manifest.get("records") or []),
                "track_source_stats": bool(manifest.get("track_source_stats", False)),
                "slot_count": len(slots),
            },
            "slots": [
                {
                    "source": slot.get("source"), "asset_signature": slot.get("asset_signature"), "object_key": slot.get("object_key"),
                    "active_articulation_usd": _canonical_file_path(slot.get("active_articulation_usd")),
                    "static_context_usd": _canonical_file_path(slot.get("static_context_usd")),
                    "compatible_motion_aliases": list(slot.get("compatible_motion_aliases") or []),
                    "representative_alias": slot.get("representative_alias"),
                    "active_surface": _file_identity(slot.get("active_surface_path")) if slot.get("active_surface_path") else None,
                }
                for slot in owned
            ],
        }
        rebase = {"root": str(path.parent), "manifest_path": str(path), "surface_paths": [slot.get("active_surface_path") for slot in owned]}
        return inputs, rebase
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def rebase_plan(plan: dict[str, Any], rebase: Mapping[str, Any]) -> dict[str, Any] | None:
    """A cached plan re-pointed to the CURRENT layout root: manifest_path / motion_dir / object_motion_dir / bps_dir and
    each slot's active_surface_path (taken from the current manifest, slot by slot). None when the shapes disagree."""
    slots = plan.get("slots")
    surfaces = rebase.get("surface_paths") or []
    if not isinstance(slots, list) or len(slots) != len(surfaces):
        return None
    root = str(rebase["root"])
    out = dict(plan)
    out["manifest_path"] = str(rebase["manifest_path"])
    out["motion_dir"] = os.path.join(root, "robot")
    out["object_motion_dir"] = os.path.join(root, "objects_with_masks")
    out["bps_dir"] = os.path.join(root, "bps")
    out["slots"] = [dict(slot, active_surface_path=surface) for slot, surface in zip(slots, surfaces)]
    return out


def _differs_only_in_manifest_file(recorded: Any, identity: Mapping[str, Any]) -> bool:
    if not isinstance(recorded, Mapping) or set(recorded) != set(identity):
        return False
    return all(recorded[k] == identity[k] for k in identity if k not in _MANIFEST_FILE_KEYS)


def _receipt_matches(
    receipt: dict[str, Any], identity: dict[str, Any], manifest_paths: tuple[str, ...] = ()
) -> str | None:
    """Return the matched contract name when ``receipt`` is compatible with ``identity``.

    ``manifest_paths`` are the as-given (normalized) and resolved paths of the manifest in
    this process; they are only consulted for legacy v1 receipts.
    """
    recorded = receipt.get("identity")
    if not isinstance(recorded, dict):
        return None
    contract = receipt.get("contract")
    if contract == CACHE_CONTRACT:
        if recorded == identity:
            return CACHE_CONTRACT
        # Legacy v2 receipts written before the slot-mapping fields existed describe
        # the DEFAULT mapping only: accept them for a default-mapping identity by
        # filling the missing fields with their defaults. A non-default identity
        # (repeat/interleave) requires the fields to be recorded explicitly.
        if (
            identity.get("slot_repeat", 1) == 1
            and identity.get("rank_interleave", False) is False
            and "slot_repeat" not in recorded
            and "rank_interleave" not in recorded
        ):
            filled = dict(recorded)
            filled["slot_repeat"] = 1
            filled["rank_interleave"] = False
            return CACHE_CONTRACT if filled == identity else None
        return None
    if contract != _LEGACY_CONTRACT_V1 or recorded.get("contract") != _LEGACY_CONTRACT_V1:
        return None
    expected = {k: v for k, v in identity.items() if k not in ("contract", "manifest_name", "manifest_inode")}
    seen = {k: v for k, v in recorded.items() if k not in ("contract", "manifest")}
    if seen != expected:
        return None
    recorded_path = recorded.get("manifest")
    if not isinstance(recorded_path, str) or not recorded_path:
        return None
    if _canonical_alias(recorded_path) not in {_canonical_alias(p) for p in manifest_paths}:
        return None  # not the same file under the known alias mapping -> native miss
    try:
        if os.stat(recorded_path).st_ino != identity["manifest_inode"]:
            return None
    except OSError:
        pass  # recorded alias not mounted here; the canonical full path already matched
    return _LEGACY_CONTRACT_V1


def _publish_plan(root: Path, entry: Path, payload_path: Path, receipt_path: Path, lock_path: Path, plan: dict[str, Any],
                  identity: dict[str, Any], manifest: Path, *, rank_inputs: dict[str, Any] | None) -> None:
    """Atomically publish the plan JSON and its receipt (manifest identity + the rank's own inputs for the re-layout reuse)."""
    root.mkdir(parents=True, exist_ok=True)
    with _BestEffortLock(lock_path, exclusive=True):
        entry.mkdir(parents=True, exist_ok=True)
        _atomic_json(payload_path, plan)
        payload_stat = payload_path.stat()
        receipt = {
            "contract": CACHE_CONTRACT,
            "identity": identity,
            # Informational only (not compared): the path alias this writer saw.
            "manifest_path_seen": str(manifest),
            "payload": str(payload_path),
            "payload_size_bytes": payload_stat.st_size,
            "payload_mtime_ns": payload_stat.st_mtime_ns,
        }
        if rank_inputs is not None:
            receipt["rank_inputs"] = rank_inputs
        _atomic_json(receipt_path, receipt)


def resolve_rank_scene_plan_cached(
    resolver: Callable[..., dict[str, Any]],
    manifest_path: str | Path,
    *,
    num_envs: int,
    rank: int,
    world_size: int,
    runtime_target_fps_hz: float | None,
    cache_mode: str = "disabled",
    cache_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one rank plan, reusing a metadata-compatible cached plan when enabled."""
    mode = str(cache_mode or "disabled").strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported rank scene-plan cache mode {cache_mode!r}")
    started = time.perf_counter()
    resolver_args = {
        "num_envs": num_envs,
        "rank": rank,
        "world_size": world_size,
        "runtime_target_fps_hz": runtime_target_fps_hz,
    }
    if mode == "disabled":
        plan = resolver(manifest_path, **resolver_args)
        return plan, {"mode": "disabled", "elapsed_seconds": time.perf_counter() - started}
    if cache_dir is None:
        raise ValueError("Rank scene-plan caching requires cache_dir")

    try:
        mapping_probe = _slot_mapping_mode()  # noqa: F841 - validates the mapping request first
    except InvalidSlotMappingMode:
        # Never serve a cached (default) plan for an unparseable mapping request:
        # the native resolver reports the configuration error itself.
        plan = resolver(manifest_path, **resolver_args)
        return plan, {"mode": "computed_invalid_mapping_request", "elapsed_seconds": time.perf_counter() - started}
    try:
        identity = build_rank_scene_plan_identity(
            manifest_path, num_envs=num_envs, rank=rank, world_size=world_size, runtime_target_fps_hz=runtime_target_fps_hz
        )
        manifest = Path(manifest_path).expanduser().resolve(strict=True)
        manifest_paths = (os.path.abspath(os.path.expanduser(str(manifest_path))), str(manifest))
    except FileNotFoundError:
        raise  # the native resolver would fail on the same missing manifest
    except OSError as exc:
        # Optional cache bookkeeping could not stat the manifest; resolve natively in every
        # mode (read_only only means "never publish"; it is never permission to resolve).
        plan = resolver(manifest_path, **resolver_args)
        return plan, {
            "mode": "computed_cache_unavailable",
            "identity_error": repr(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }
    try:
        root = Path(cache_dir).expanduser().resolve()
        entry = root / identity["cache_key"]
        payload_path = entry / "scene_plan.json"
        receipt_path = entry / "receipt.json"
        lock_path = root / f"{identity['cache_key']}.lock"
        root_is_dir = root.is_dir()
    except (OSError, RuntimeError) as exc:
        # Unreadable / looping / permission-denied cache root: the optional cache is simply
        # unavailable. One native resolution; the resolver's own data errors stay visible.
        plan = resolver(manifest_path, **resolver_args)
        return plan, {
            "mode": "computed_cache_unavailable",
            "root_error": repr(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }

    # -- read ---------------------------------------------------------------
    cached_plan = None
    read_error = None
    matched_contract = None
    if root_is_dir:
        try:
            with _BestEffortLock(lock_path, exclusive=False):
                if payload_path.is_file() and receipt_path.is_file():
                    receipt = _load_json(receipt_path, label="rank scene-plan cache receipt")
                    payload_stat = payload_path.stat()
                    matched_contract = _receipt_matches(receipt, identity, manifest_paths)
                    if (
                        matched_contract is not None
                        and receipt.get("payload_size_bytes") == payload_stat.st_size
                        and receipt.get("payload_mtime_ns") == payload_stat.st_mtime_ns
                    ):
                        plan = _load_json(payload_path, label="cached rank scene plan")
                        if (
                            all(plan.get(key) == identity[key] for key in ("rank", "world_size", "num_envs"))
                            and int(plan.get("slot_repeat", 1) or 1) == identity["slot_repeat"]
                            and bool(plan.get("rank_interleave", False)) == identity["rank_interleave"]
                        ):
                            cached_plan = plan
        except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            read_error = repr(exc)
            cached_plan = None
    # -- second chance: a NEW manifest file (re-layout) whose inputs for THIS rank are unchanged -----------
    rebased_from = None
    if cached_plan is None and root_is_dir and identity["slot_repeat"] == 1 and not identity["rank_interleave"]:
        try:
            with _BestEffortLock(lock_path, exclusive=False):
                receipt = _load_json(receipt_path, label="rank scene-plan cache receipt") if receipt_path.is_file() and payload_path.is_file() else None
                recorded_inputs = receipt.get("rank_inputs") if isinstance(receipt, dict) else None
                if (
                    isinstance(receipt, dict)
                    and receipt.get("contract") == CACHE_CONTRACT
                    and isinstance(recorded_inputs, dict)
                    and _differs_only_in_manifest_file(receipt.get("identity"), identity)
                ):
                    payload_stat = payload_path.stat()
                    if receipt.get("payload_size_bytes") == payload_stat.st_size and receipt.get("payload_mtime_ns") == payload_stat.st_mtime_ns:
                        current = rank_slot_inputs(manifest, rank=rank, world_size=world_size, num_envs=num_envs)
                        if current is not None and current[0] == recorded_inputs:
                            plan = _load_json(payload_path, label="cached rank scene plan")
                            rebased = rebase_plan(plan, current[1])
                            if rebased is not None and all(rebased.get(key) == identity[key] for key in ("rank", "world_size", "num_envs")):
                                cached_plan = rebased
                                rebased_from = (receipt.get("identity") or {}).get("manifest_inode")
        except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            read_error = repr(exc) if read_error is None else read_error
            cached_plan = None
    if cached_plan is not None and rebased_from is not None:
        status = {
            "mode": "reused_rank_inputs_rebased",
            "cache_key": identity["cache_key"],
            "entry": str(entry),
            "previous_manifest_inode": rebased_from,
        }
        if mode == "read_write":
            # publish the rebased plan under the new manifest identity so the next leg hits on the receipt alone
            try:
                _publish_plan(root, entry, payload_path, receipt_path, lock_path, cached_plan, identity, manifest, rank_inputs=current[0])
                status["receipt_upgraded"] = True
            except (OSError, TypeError, ValueError) as exc:
                status["receipt_upgrade_error"] = repr(exc)
        status["elapsed_seconds"] = time.perf_counter() - started
        return cached_plan, status
    if cached_plan is not None:
        status = {
            "mode": "reused_metadata_checked",
            "cache_key": identity["cache_key"],
            "entry": str(entry),
            "receipt_contract": matched_contract,
        }
        # Receipts written before the rank-input identity (50e7979) carry no rank_inputs. They are NOT enriched here:
        # that would re-read the manifest and stat every owned slot on the hit path (startup work with no training
        # consumer). They gain the inputs when the native resolver runs anyway (their next miss = the first re-layout),
        # so that first re-layout recomputes such a rank's plan once and the one after it reuses.
        if matched_contract == _LEGACY_CONTRACT_V1 and mode == "read_write":
            # Rewrite only the receipt in the current shape so later readers compare v2 directly.
            try:
                with _BestEffortLock(lock_path, exclusive=True):
                    payload_stat = payload_path.stat()
                    upgraded = {
                        "contract": CACHE_CONTRACT,
                        "identity": identity,
                        "manifest_path_seen": str(manifest),
                        "payload": str(payload_path),
                        "payload_size_bytes": payload_stat.st_size,
                        "payload_mtime_ns": payload_stat.st_mtime_ns,
                        "upgraded_from": _LEGACY_CONTRACT_V1,
                    }
                    _atomic_json(receipt_path, upgraded)   # (no rank-input read here: the hit path adds no I/O)
                status["receipt_upgraded"] = True
            except (OSError, TypeError, ValueError) as exc:
                status["receipt_upgrade_error"] = repr(exc)
        status["elapsed_seconds"] = time.perf_counter() - started
        return cached_plan, status
    # -- miss: unchanged native resolver in every mode, then optional publish (read_write only)
    plan = resolver(manifest_path, **resolver_args)
    resolved_seconds = time.perf_counter() - started
    if mode == "read_only":
        status = {
            "mode": "miss_native_fallback",
            "cache_key": identity["cache_key"],
            "entry": str(entry),
            "resolver_seconds": resolved_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if read_error is not None:
            status["read_error"] = read_error
        return plan, status
    status: dict[str, Any] = {
        "mode": "computed_and_cached",
        "cache_key": identity["cache_key"],
        "entry": str(entry),
        "resolver_seconds": resolved_seconds,
    }
    if read_error is not None:
        status["read_error"] = read_error
    try:
        rank_inputs = None
        if identity["slot_repeat"] == 1 and not identity["rank_interleave"]:
            computed = rank_slot_inputs(manifest, rank=rank, world_size=world_size, num_envs=num_envs)
            rank_inputs = computed[0] if computed is not None else None
        _publish_plan(root, entry, payload_path, receipt_path, lock_path, plan, identity, manifest, rank_inputs=rank_inputs)
    except (OSError, TypeError, ValueError) as exc:
        # Optional cache IO only: the native plan is already resolved and is returned unchanged.
        status["mode"] = "computed_cache_write_failed"
        status["write_error"] = repr(exc)
    status["elapsed_seconds"] = time.perf_counter() - started
    return plan, status
