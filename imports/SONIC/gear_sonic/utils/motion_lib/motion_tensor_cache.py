"""Run-local MotionLib reference-tensor cache keyed by direct source metadata.

Lazy first-warmup flavor: the first ordinary initialization in
``read_write`` mode runs the unchanged native loader and then publishes one
immutable CPU tensor bundle per rank; later same-layout initializations restore
those reference tensors to the device instead of re-reading every motion file.
Compatibility is decided by ordinary existing fields (rank/world/env layout,
ordered motion aliases, resolved source paths and their stat metadata, clocks,
mappings, numeric tensor schema). No code-content or data-content hashing, no
promotion token, no READY/sidecar admission file. Cache IO is optional: callers treat
read/write failures as a miss and keep the successfully initialized native
state. Only immutable preprocessed reference tensors are cached; sampler,
episode counters, RNG, environment and optimizer state stay on the ordinary
checkpoint path.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import pickle
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch

try:  # flock is best-effort on network filesystems; never required for correctness.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


CACHE_CONTRACT = "motionlib_rank_tensor_cache_new8_metadata_v1"
SUPPORTED_MODES = frozenset({"disabled", "read_write", "read_only"})
_TENSOR_FILE = "tensors.pt"
_METADATA_FILE = "metadata.json"
_LEGACY_READY_FILE = "READY.json"  # read-only compatibility with entries written before 2026-10-13T19Z; never written
_LOG = logging.getLogger(__name__)


class MotionTensorCacheMiss(FileNotFoundError):
    """An explicitly read-only cache has no compatible entry."""


# Non-"_motion_*" reference tensors assembled by MotionLibBase.load_motions.
REFERENCE_TENSOR_NAMES = frozenset(
    {
        "body_ang_vel_w",
        "body_ang_vel_w_full",
        "body_lin_vel_w",
        "body_lin_vel_w_full",
        "body_pos_b",
        "body_pos_w",
        "body_pos_w_full",
        "body_quat_w",
        "body_quat_w_full",
        "dof_pos",
        "dof_vel",
        "feet_l",
        "feet_r",
        "hand_dof_pos",
        "length_starts",
        "motion_has_object",
        "motion_has_smpl",
        "motion_has_soma",
        "motion_ids",
        "root_ang_vel_w",
        "root_linv_vel_w",
    }
)
# Plain-Python structural metadata assembled by load_motions.
REFERENCE_METADATA_NAMES = frozenset(
    {
        "_motion_object_name_candidates",
        "_num_motions",
        "curr_motion_keys",
        "has_action",
        "num_bodies",
        "num_bodies_full",
        "num_joints",
    }
)
# Reference attributes that load_motions may legitimately leave as None.
NULLABLE_REFERENCE_TENSOR_NAMES = frozenset({"_motion_object_bps", "hand_dof_pos"})
# Set by load_motions but re-derived by the caller on a hit (sampler-side state), never stored in a bundle.
# With adaptive sampling ON (the formal default) load_motions ends with
# update_adaptive_sampling_motion_frames(), which the cached hit path calls again after the restore:
#   adp_samp_active_motion_bins, orig_motion_id_to_motion_ids  - the loaded batch's bin index and orig->loaded motion id
#       map, derived from _curr_motion_ids and the immutable orig_motion_id_to_bins built at construction;
#   adp_sampling_active_prob, adp_samp_active_failure_rate     - RUNTIME sampling state computed by
#       update_adaptive_sampling_probabilities() from the per-bin episode / failure counters (checkpoint-owned, evolving;
#       a cached copy would freeze stale probabilities, so they are re-derived, never restored).
DYNAMIC_AFTER_LOAD_NAMES = frozenset(
    {
        "_curr_motion_ids",
        "_sampling_batch_prob",
        "res_non_nav_dataset",
        "_tracked_body_subset_idx",
        "adp_samp_active_motion_bins",
        "orig_motion_id_to_motion_ids",
        "adp_sampling_active_prob",
        "adp_samp_active_failure_rate",
    }
)
# Identity fields that only describe the env-slot layout, not the per-motion payload.
LAYOUT_ONLY_IDENTITY_KEYS = ("cache_key", "num_envs")
# load_data() products that share the "_motion_" prefix but are not tensors.
_NON_REFERENCE_MOTION_PREFIX_NAMES = frozenset(
    {"_motion_data_keys", "_motion_data_list", "_motion_data_load"}
)
# Load-time randomness whose sampled realization must not be frozen into a cache.
_STOCHASTIC_BOOLEAN_OPTIONS = ("randomize_heading",)
_STOCHASTIC_PROBABILITY_OPTIONS = (
    ("freeze_frame_aug", ("freeze_frame_prob", "freeze_frame_aug_prob"), 0.1),
    ("cat_upper_body_poses", ("cat_upper_body_poses_prob",), 0.0),
    ("randomize_wrist_poses", ("randomize_wrist_prob",), 0.3),
)
# Config keys that do not change the produced reference tensors.
_IDENTITY_EXCLUDED_CONFIG_KEYS = frozenset(
    {
        "adaptive_sampling",
        "debug",
        "filter_motion_keys",  # represented by ordered_motion_aliases
        "joint_roster_direct_dirs",  # spec string; data identity = ordered_motion_aliases + motion_file
        "joint_source_tracking",  # sampling identity switch (source::motion keys); never changes the reference tensors
        # NOTE: joint_source_manifest is deliberately NOT excluded: its presence changes loader payload semantics
        "joint_roster_scene_plan_cache_dir",
        "joint_roster_scene_plan_cache_mode",
        "joint_source_roster",  # in-memory identity records (sampling only); never changes the reference tensors
        "motion_loader_num_workers",
        "motion_tensor_cache",
        "multi_thread",
        "use_parallel_fk",
    }
)


def normalize_cache_mode(value: Any) -> str:
    mode = str(value or "disabled").strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported MotionLib tensor cache mode {value!r}; expected one of "
            f"{sorted(SUPPORTED_MODES)}"
        )
    return mode


def nondeterministic_load_options(
    motion_lib_cfg: Mapping[str, Any], *, loads_all_rank_motions: bool
) -> list[str]:
    """Return the load-time options that make the produced tensors non-cacheable."""
    problems = []
    if not loads_all_rank_motions:
        problems.append("random or partial rank-local motion selection")
    for option in _STOCHASTIC_BOOLEAN_OPTIONS:
        if bool(motion_lib_cfg.get(option, False)):
            problems.append(option)
    for enabled_option, probability_options, default_probability in _STOCHASTIC_PROBABILITY_OPTIONS:
        if not bool(motion_lib_cfg.get(enabled_option, False)):
            continue
        probability = default_probability
        for probability_option in probability_options:
            if probability_option in motion_lib_cfg:
                probability = float(motion_lib_cfg[probability_option])
                break
        if probability > 0.0:
            problems.append(f"{enabled_option} (probability={probability})")
    return problems


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    raise TypeError(f"MotionLib tensor cache identity cannot encode {type(value).__name__}")


def _path_identity(value: Any, *, strict: bool) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    try:
        resolved = path.resolve(strict=strict)
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    entry: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if resolved.is_file():
        stat = resolved.stat()
        entry["size_bytes"] = int(stat.st_size)
        entry["mtime_ns"] = int(stat.st_mtime_ns)
    return entry


def build_motion_tensor_cache_identity(
    motion_lib: Any, *, rank: int, world_size: int, num_envs: int
) -> dict[str, Any]:
    """Describe the exact rank-local load from existing config/data fields only."""
    cfg = motion_lib.m_cfg
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError(f"Invalid MotionLib cache rank/world_size: {rank}/{world_size}")
    aliases = [str(key) for key in motion_lib._motion_data_keys.tolist()]
    if not aliases or len(set(aliases)) != len(aliases):
        raise ValueError("MotionLib tensor cache identity requires unique ordered motion aliases")
    skeleton_file = Path(str(cfg.asset.assetRoot)) / str(cfg.asset.assetFileName)
    preprocessing = {
        str(key): _json_value(value)
        for key, value in cfg.items()
        if str(key) not in _IDENTITY_EXCLUDED_CONFIG_KEYS
    }
    identity = {
        "contract": CACHE_CONTRACT,
        # Reference-channel schema: bundles written before the per-clip label presence / provenance channels
        # (2026-10-13) cannot reproduce the hand reward's source semantics (an Omni valid native 0 would read as
        # neutral), so they are an ordinary identity miss and the native loader rebuilds them. Plain metadata.
        "label_channels": "presence_provenance_v1",
        "cache_key": f"layout-v1-world-{int(world_size)}-env{int(num_envs)}",
        "rank": int(rank),
        "world_size": int(world_size),
        "num_envs": int(num_envs),
        "ordered_motion_aliases": aliases,
        "motion_file": _path_identity(cfg.get("motion_file"), strict=True),
        "object_motion_file": _path_identity(cfg.get("object_motion_file"), strict=False),
        "bps_dir": _path_identity(cfg.get("bps_dir"), strict=False),
        "joint_roster_manifest": _path_identity(cfg.get("joint_roster_manifest"), strict=False),
        "joint_source_manifest": _path_identity(cfg.get("joint_source_manifest"), strict=False),
        "skeleton_asset": _path_identity(skeleton_file, strict=True),
        "target_fps": float(motion_lib.target_fps),
        "motion_fps_scale": float(motion_lib.motion_fps_scale),
        "sim_fps": float(motion_lib._sim_fps),
        "fix_height": _json_value(getattr(motion_lib, "fix_height", None)),
        "num_unique_motions": int(motion_lib._num_unique_motions),
        "preprocessing_config": preprocessing,
        "torch_version": str(torch.__version__),
        "stored_device": "cpu",
    }
    return _json_value(identity)


def _is_reference_tensor_name(name: str) -> bool:
    if name in _NON_REFERENCE_MOTION_PREFIX_NAMES:
        return False
    return name.startswith("_motion_") or name in REFERENCE_TENSOR_NAMES



def capture_motionlib_reference_state(
    motion_lib: Any, *, attributes_before_load: set[str] | frozenset[str] | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Capture immutable reference tensors plus the minimal structural metadata.

    When ``attributes_before_load`` is given, every attribute that load_motions
    added must be accounted for (tensor, metadata, nullable or caller-derived);
    otherwise the capture refuses, so a later hit can never miss an attribute.
    """
    namespace = vars(motion_lib)
    tensors = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in namespace.items()
        if torch.is_tensor(value) and _is_reference_tensor_name(name)
    }
    if not tensors:
        raise ValueError("MotionLib produced no cacheable reference tensors")
    attributes = {}
    for name in sorted(REFERENCE_METADATA_NAMES):
        if name in namespace:
            attributes[name] = _json_value(namespace[name])
    null_attributes = sorted(
        name
        for name in NULLABLE_REFERENCE_TENSOR_NAMES
        if name in namespace and namespace[name] is None
    )
    missing = sorted({"_num_motions", "curr_motion_keys", "num_joints"} - attributes.keys())
    if missing:
        raise ValueError(f"MotionLib reference capture is missing structural metadata: {missing}")
    if attributes_before_load is not None:
        added = set(namespace) - set(attributes_before_load)
        unexplained = sorted(
            name
            for name in added
            if name not in tensors
            and name not in attributes
            and name not in null_attributes
            and name not in DYNAMIC_AFTER_LOAD_NAMES
        )
        if unexplained:
            raise ValueError(
                "MotionLib load_motions produced attributes the reference cache does "
                f"not model; refusing to cache: {unexplained}"
            )
    metadata = {"attributes": attributes, "null_tensor_attributes": null_attributes}
    return tensors, metadata


def restore_motionlib_reference_state(
    motion_lib: Any,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    device: str | torch.device,
) -> None:
    """Restore immutable references only; sampler/curriculum state is untouched."""
    attributes = metadata.get("attributes")
    null_attributes = metadata.get("null_tensor_attributes", [])
    if not isinstance(attributes, Mapping) or not isinstance(null_attributes, list):
        raise TypeError("Invalid MotionLib tensor cache state metadata")
    unknown_attributes = set(attributes) - REFERENCE_METADATA_NAMES
    unknown_nulls = set(null_attributes) - NULLABLE_REFERENCE_TENSOR_NAMES
    unknown_tensors = {str(name) for name in tensors if not _is_reference_tensor_name(str(name))}
    if unknown_attributes or unknown_nulls or unknown_tensors:
        raise ValueError(
            "MotionLib tensor cache contains non-reference state: "
            f"attributes={sorted(unknown_attributes)}, nulls={sorted(unknown_nulls)}, "
            f"tensors={sorted(unknown_tensors)}"
        )
    for name, value in tensors.items():
        if not torch.is_tensor(value) or value.device.type != "cpu":
            raise ValueError(f"Cached MotionLib tensor {name!r} is not a CPU tensor")
    for name, value in attributes.items():
        setattr(motion_lib, str(name), value)
    for name in null_attributes:
        setattr(motion_lib, str(name), None)
    for name, value in tensors.items():
        setattr(motion_lib, str(name), value.to(device=device))


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON mapping: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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


_FP16_FLAG = "fp16_reference_tensors"
from gear_sonic.utils.repo_asset_identity import comparable_asset_record  # skeleton ships inside the checkout: pin-independent identity
# Effective PAYLOAD MODE: the loader semantics that change the reference tensors beyond the config/data identity.
# Today both flags follow the presence of a joint-source / roster manifest; a consumer that enables the paired
# reference clock from in-memory identity records (file manifest None) exposes the same mode through
# ``motion_lib.effective_payload_mode`` (attribute or zero-arg callable) and the cache compares THAT, never the
# file path alone. Receipts record the mode; receipts written before it is derived from their recorded identity.
PAYLOAD_MODE_KEYS = ("paired_reference_clock", "manifest_contact_tensors")


def effective_payload_mode(motion_lib: Any) -> dict[str, bool]:
    """Payload mode of a live MotionLib: paired reference clock (hand_action padding, canonical-contact fallback,
    retarget-init handling) and manifest-forced contact tensors, as the loader will actually apply them."""
    cfg = motion_lib.m_cfg
    manifest_present = bool(cfg.get("joint_source_manifest"))
    mode = {
        "paired_reference_clock": bool(getattr(motion_lib, "require_paired_reference_clock", manifest_present)),
        "manifest_contact_tensors": bool(cfg.get("joint_source_manifest") or cfg.get("joint_roster_manifest")),
    }
    explicit = getattr(motion_lib, "effective_payload_mode", None)
    if callable(explicit):
        explicit = explicit()
    if isinstance(explicit, Mapping):
        mode.update({str(k): bool(v) for k, v in explicit.items() if str(k) in PAYLOAD_MODE_KEYS})
    return mode


def derived_payload_mode(identity: Mapping[str, Any]) -> dict[str, bool]:
    """Payload mode implied by a recorded identity (receipts written before the mode was recorded)."""
    pre = identity.get("preprocessing_config") or {}
    source_present = bool(identity.get("joint_source_manifest")) or bool(pre.get("joint_source_manifest")) if isinstance(pre, Mapping) else bool(identity.get("joint_source_manifest"))
    roster_present = bool(identity.get("joint_roster_manifest")) or (isinstance(pre, Mapping) and bool(pre.get("joint_roster_manifest")))
    return {"paired_reference_clock": source_present, "manifest_contact_tensors": bool(source_present or roster_present)}


def receipt_payload_mode(metadata: Mapping[str, Any] | None, identity: Mapping[str, Any]) -> dict[str, bool]:
    """Recorded ``payload_mode`` of a receipt when present and complete, else the mode derived from its identity."""
    recorded = metadata.get("payload_mode") if isinstance(metadata, Mapping) else None
    if isinstance(recorded, Mapping) and all(key in recorded for key in PAYLOAD_MODE_KEYS):
        return {key: bool(recorded[key]) for key in PAYLOAD_MODE_KEYS}
    return derived_payload_mode(identity)


def _normalized_payload_identity(identity: Mapping[str, Any]) -> dict[str, Any] | None:
    """Identity view used ONLY by the compatibility matcher (no hashing, no gates).

    * layout-only fields (cache_key/num_envs) removed so a slot-repeated layout of
      the same rank-local payload can reuse an existing bundle;
    * ``preprocessing_config.fp16_reference_tensors`` absent -> False (older bundles
      predate the flag) and reported separately as ``fp16``;
    * everything else - including ``skeleton_asset`` exactly as recorded by the
      existing identity builder (resolved path, size, mtime) - stays strict. A
      different skeleton path is simply "not this bundle": the caller falls back to
      the native load. Point the run at the same immutable skeleton asset path
      (launcher MOTION_SKELETON_ASSET_ROOT) when source snapshots move.
    """
    view = dict(_json_value(identity))
    for key in LAYOUT_ONLY_IDENTITY_KEYS:
        view.pop(key, None)
    # Joint-source manifest: the loader keys several payload branches on its PRESENCE only
    # (motion_lib_base: require_paired_reference_clock = bool(cfg.joint_source_manifest) -> missing hand_action
    # padded to -1, canonical-contact fallback refused, retarget-init handling; contact tensors forced when a
    # joint-source or roster manifest is present). The manifest's path is location only, and its CONTENT (sources,
    # FPS, world size) is validated by the loader (init_joint_source_tracking, load_motions ~4115-4141) but does not
    # change the reference tensors a SUCCESSFUL load produces - equivalence here is successful-payload equivalence,
    # not absence of validation (a failing validation still fails natively). The strict identity keeps the
    # recorded path (top level) and the raw config value (nested copy) so existing receipts stay exact hits; the
    # compatibility view reduces BOTH to one presence flag: a rename / equivalent-content manifest reuses the bundle,
    # None <-> path does NOT (payload semantics may differ) and falls back to the native loader as usual.
    top_manifest = view.pop("joint_source_manifest", None)
    pre = dict(view.get("preprocessing_config") or {})
    nested_manifest = pre.pop("joint_source_manifest", None)
    view["_joint_source_manifest_present"] = bool(top_manifest) or bool(nested_manifest)
    pre.pop("joint_source_tracking", None)  # identity-tracking switch: sampling only, never a payload input
    # Only the skeleton DIRECTORY string (asset.assetRoot: relative in older receipts,
    # absolute stable path in newer launches) is location, not payload identity - the
    # resolved skeleton file itself is compared strictly through ``skeleton_asset``.
    # assetFileName / urdfFileName and every other asset field stay strict.
    if isinstance(pre.get("asset"), Mapping):
        asset = dict(pre["asset"]); asset.pop("assetRoot", None); pre["asset"] = asset
    raw = pre.pop(_FP16_FLAG, False)
    fp16 = raw.strip().lower() in ("1", "true", "yes", "on") if isinstance(raw, str) else bool(raw)
    view["preprocessing_config"] = pre
    if not isinstance(view.get("skeleton_asset"), Mapping):
        return None
    # The skeleton file ships inside the code checkout: a new pinned worktree has a new absolute path and mtime
    # for byte-identical content. Compare it pin-independently (repo-relative path + size + last-touch commit of its
    # directory tree, gear_sonic/utils/repo_asset_identity.py); the data-pack paths of the identity stay exact.
    view["skeleton_asset"] = comparable_asset_record(view["skeleton_asset"])
    # The roster manifest's PATH is layout identity (a re-materialized cohort of the same motions gets a new manifest
    # file); the reference tensors depend only on its PRESENCE (forced contact tensors, compared again through the
    # payload mode) - keep presence, drop the path/stat.
    view["joint_roster_manifest"] = bool(view.get("joint_roster_manifest"))
    if "joint_roster_manifest" in pre:  # the raw config value is copied into preprocessing_config as well
        pre["joint_roster_manifest"] = bool(pre.get("joint_roster_manifest"))
    view["preprocessing_config"] = pre
    view["_fp16"] = fp16
    return view


def payload_compatibility(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    current_mode: Mapping[str, Any] | None = None,
    candidate_mode: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the compatibility class of ``candidate`` (a receipt identity) for ``current``.

    ``current_mode`` / ``candidate_mode`` are effective payload modes (``effective_payload_mode`` /
    ``receipt_payload_mode``); when omitted they are derived from the identities. Different modes are never
    compatible, whatever the file paths say.

    ``"exact"``: identical payload identity (same precision); ``"convert_fp32_to_fp16"``:
    identical payload except the candidate stores the eligible channels FP32 while the
    current run opted into FP16 (restore + explicit compression is exact-to-spec);
    ``None``: incompatible, including the lossy direction FP16 source -> FP32 target.
    """
    a = _normalized_payload_identity(current)
    b = _normalized_payload_identity(candidate)
    if a is None or b is None:
        return None
    mode_a = dict(current_mode) if isinstance(current_mode, Mapping) else derived_payload_mode(current)
    mode_b = dict(candidate_mode) if isinstance(candidate_mode, Mapping) else derived_payload_mode(candidate)
    if {k: bool(mode_a.get(k)) for k in PAYLOAD_MODE_KEYS} != {k: bool(mode_b.get(k)) for k in PAYLOAD_MODE_KEYS}:
        return None
    fa, fb = a.pop("_fp16"), b.pop("_fp16")
    if a != b:
        return None
    if fa == fb:
        return "exact"
    if fb is False and fa is True:
        return "convert_fp32_to_fp16"
    return None


_DIR_IDENTITY_KEYS = ("motion_file", "object_motion_file", "bps_dir")
_DIR_ALIAS_SUFFIX = {"motion_file": ".pkl", "object_motion_file": ".pkl", "bps_dir": ".npy"}


def _same_file(path_a: Path, path_b: Path) -> bool:
    """Same file by identity (inode + size + mtime_ns); a cohort that links the same payload under another directory or
    the same directory spelled through another mount alias both satisfy it. Never raises."""
    try:
        sa, sb = os.stat(path_a), os.stat(path_b)
    except OSError:
        return False
    return (sa.st_ino, sa.st_size, sa.st_mtime_ns) == (sb.st_ino, sb.st_size, sb.st_mtime_ns)


def _dir_identity_path(identity: Mapping[str, Any], key: str) -> str | None:
    value = identity.get(key)
    if isinstance(value, Mapping):
        value = value.get("path")
    return str(value) if value else None


def payload_dirs_equivalent(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """The three payload directories of ``candidate`` hold, alias by alias, the SAME files as those of ``current``.

    The rank-local reference tensors depend on the payload files behind each ordered motion alias, not on the
    directory they are reached through: a re-materialized cohort of the same catalog (new layout root) links the
    same pack files under new directories, and a cohort reached through another mount alias has a different
    spelling of the same directory. Equivalence is decided per alias on the file identity (inode + size + mtime_ns)
    of ``<motion_file>/<alias>.pkl``, ``<object_motion_file>/<alias>.pkl`` and ``<bps_dir>/<alias>.npy``; one
    different, missing or unreadable file means not equivalent. Returns the counts when equivalent, else None.
    No hashing, nothing written.
    """
    aliases = current.get("ordered_motion_aliases")
    if not isinstance(aliases, list) or not aliases or aliases != candidate.get("ordered_motion_aliases"):
        return None
    checked = 0
    for key in _DIR_IDENTITY_KEYS:
        cur_dir, cand_dir = _dir_identity_path(current, key), _dir_identity_path(candidate, key)
        if cur_dir is None and cand_dir is None:
            continue
        if cur_dir is None or cand_dir is None:
            return None
        if cur_dir == cand_dir:
            # the same spelling: equivalent only when the recorded identity (size/mtime) is unchanged; a payload
            # rewritten in place has new content behind the same path and the alias files cannot tell it apart
            if _dir_identity_record(current, key) != _dir_identity_record(candidate, key):
                return None
            continue
        cur_path, cand_path = Path(cur_dir), Path(cand_dir)
        if cur_path.is_file() or cand_path.is_file():
            # single-file payload (MotionlibMode.file / one object pkl): one file identity, AND the candidate's recorded
            # size/mtime must still be the live file's (its bundle was computed from the file as recorded; a file
            # rewritten since is the same inode with new content)
            if not (cur_path.is_file() and cand_path.is_file() and _same_file(cur_path, cand_path)):
                return None
            cand_record = _dir_identity_record(candidate, key)
            try:
                live = cand_path.stat()
            except OSError:
                return None
            if not isinstance(cand_record, Mapping) or cand_record.get("size_bytes") != int(live.st_size) or cand_record.get("mtime_ns") != int(live.st_mtime_ns):
                return None
            checked += 1
            continue
        if cur_path.is_dir() and cand_path.is_dir() and _same_file(cur_path, cand_path):
            # The SAME directory through another spelling (a container bind mount, or a symlinked
            # root): the ordinary same-spelling rule applies - the recorded directory identity, spelling aside, decides;
            # no per-alias payload stat scan (2 stats x aliases x 3 directories otherwise).
            if _without_path(_dir_identity_record(current, key)) != _without_path(_dir_identity_record(candidate, key)):
                return None
            checked += 1
            continue
        suffix = _DIR_ALIAS_SUFFIX[key]
        for alias in aliases:
            if not _same_file(cur_path / f"{alias}{suffix}", cand_path / f"{alias}{suffix}"):
                return None
            checked += 1
        if key != "bps_dir":
            # directory loading merges <dir>/metadata.pkl and <dir>/<sub>/metadata.pkl into every record (fps and
            # per-clip fields): the same set of metadata files, each the same file, or no metadata on either side
            cur_meta, cand_meta = _metadata_pkls(cur_path), _metadata_pkls(cand_path)
            if set(cur_meta) != set(cand_meta):
                return None
            for rel in cur_meta:
                if not _same_file(cur_path / rel, cand_path / rel):
                    return None
                checked += 1
    return {"aliases": len(aliases), "files_compared": checked}


def _without_path(record: Any) -> Any:
    """A recorded path identity minus its spelling (two spellings of one directory compare on the rest)."""
    if isinstance(record, Mapping):
        return {k: v for k, v in record.items() if k != "path"}
    return record


def _dir_identity_record(identity: Mapping[str, Any], key: str) -> Any:
    """The recorded identity entry (path + size + mtime) of a payload directory/file, wherever it is stored."""
    value = identity.get(key)
    if value is None:
        pre = identity.get("preprocessing_config")
        if isinstance(pre, Mapping):
            value = pre.get(key)
    return _json_value(value)


def _metadata_pkls(directory: Path) -> list[str]:
    """Relative paths of metadata.pkl at the top level and in the immediate subdirectories (sorted)."""
    found = []
    try:
        if (directory / "metadata.pkl").is_file():
            found.append("metadata.pkl")
        for child in directory.iterdir():
            if child.is_dir() and (child / "metadata.pkl").is_file():
                found.append(f"{child.name}/metadata.pkl")
    except OSError:
        return found
    return sorted(found)


def _without_payload_dirs(view: dict[str, Any]) -> dict[str, Any]:
    out = dict(view)
    for key in _DIR_IDENTITY_KEYS:
        out.pop(key, None)
    pre = out.get("preprocessing_config")
    if isinstance(pre, Mapping):
        out["preprocessing_config"] = {k: v for k, v in pre.items() if k not in _DIR_IDENTITY_KEYS}
    return out


def payload_compatibility_with_equivalent_dirs(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    current_mode: Mapping[str, Any] | None = None,
    candidate_mode: Mapping[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """``payload_compatibility`` first; when it fails ONLY because the payload directories differ (every other
    normalized field equal, same precision class) and the directories are equivalent file by file, the candidate is
    compatible with the same class it would have had. Returns (class or None, equivalence note)."""
    compat = payload_compatibility(current, candidate, current_mode=current_mode, candidate_mode=candidate_mode)
    if compat is not None:
        return compat, None
    a = _normalized_payload_identity(current)
    b = _normalized_payload_identity(candidate)
    if a is None or b is None:
        return None, None
    mode_a = dict(current_mode) if isinstance(current_mode, Mapping) else derived_payload_mode(current)
    mode_b = dict(candidate_mode) if isinstance(candidate_mode, Mapping) else derived_payload_mode(candidate)
    if {k: bool(mode_a.get(k)) for k in PAYLOAD_MODE_KEYS} != {k: bool(mode_b.get(k)) for k in PAYLOAD_MODE_KEYS}:
        return None, None
    fa, fb = a.pop("_fp16"), b.pop("_fp16")
    if _without_payload_dirs(a) != _without_payload_dirs(b):
        return None, None
    note = payload_dirs_equivalent(current, candidate)
    if note is None:
        return None, None
    if fa == fb:
        return "exact", note
    if fb is False and fa is True:
        return "convert_fp32_to_fp16", note
    return None, None


def find_reusable_entry(
    cache_dir: str | Path, identity: Mapping[str, Any], *, payload_mode: Mapping[str, Any] | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]] | None:
    """Reuse a compatible bundle of the same rank/world after a normal miss.

    Candidates are every ``layout-v1-world-<W>-env*`` entry of this rank; each
    receipt identity is compared with ``payload_compatibility`` and, when
    compatible, loaded through the ordinary read path with ITS OWN identity
    (identity + effective payload mode binding only). The caller restores the
    tensors and then applies the configured storage precision/dedup, so an FP32
    source can serve an FP16-opted run; an FP16 source is never offered to an
    FP32 run. Nothing is written here.
    """
    root = Path(cache_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    world = int(identity.get("world_size", 1))
    rank = int(identity.get("rank", 0))
    prefix = f"layout-v1-world-{world}-env"
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for entry_root in candidates:
        entry = entry_root / f"rank-{rank:03d}"
        receipt_identity = None
        receipt_metadata: Mapping[str, Any] | None = None
        try:
            # recorded identity lives in metadata.json; entries written before 2026-10-13T19Z kept it in
            # a READY.json that is read as plain legacy metadata (never written anew, never required)
            metadata_json = None
            if (entry / _METADATA_FILE).is_file():
                metadata_json = receipt_metadata = _load_json(entry / _METADATA_FILE, label="MotionLib cache metadata")
                receipt_identity = receipt_metadata.get("identity")
            if receipt_identity is None and (entry / _LEGACY_READY_FILE).is_file():
                receipt_metadata = _load_json(entry / _LEGACY_READY_FILE, label="MotionLib cache legacy metadata")
                receipt_identity = receipt_metadata.get("identity")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(receipt_identity, Mapping):
            continue
        candidate_mode = receipt_payload_mode(receipt_metadata, receipt_identity)
        if receipt_identity.get("cache_key") == identity.get("cache_key") and _json_value(receipt_identity) == _json_value(identity):
            continue  # that is the normal path's own entry, already a miss
        compat, equivalence = payload_compatibility_with_equivalent_dirs(
            identity, receipt_identity, current_mode=payload_mode, candidate_mode=candidate_mode
        )
        if compat is None:
            continue
        try:
            other = MotionTensorCache(cache_dir=root, identity=receipt_identity, mode="read_only", payload_mode=candidate_mode)
            loaded = other.load(metadata_json if metadata_json is not None and metadata_json.get("identity") is not None else None)
        except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError):
            continue
        if loaded is None:
            continue
        tensors, state_metadata, status = loaded
        status = dict(status)
        status["mode"] = "reused_compatible_entry"
        status["compatibility"] = compat
        status["source_cache_key"] = str(receipt_identity.get("cache_key"))
        status["requested_cache_key"] = str(identity.get("cache_key"))
        if equivalence is not None:
            status["payload_dirs_equivalent"] = equivalence   # same files alias by alias under other directories / spellings
        return tensors, state_metadata, status
    return None


class BackgroundPublish:
    """The optional cache publication (torch.save of the CPU copies + fsync + atomic rename) running on a daemon thread
    so useful PPO starts as soon as the native reference state is ready (wall time first).
    The tensors handed in are the CPU copies the capture already made; nothing else of the motion library is touched.
    ``status`` holds 'computed_publish_pending' until the thread finishes, then the write's own status
    ('computed_and_cached') or 'write_failed'; ``join`` waits (tests, deliberate flushes)."""

    def __init__(self, cache: "MotionTensorCache", tensors: Mapping[str, torch.Tensor], state_metadata: Mapping[str, Any],
                 *, on_done: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.status: dict[str, Any] = {"mode": "computed_publish_pending", "entry": str(cache.entry), "cache_key": cache.cache_key}
        self._cache, self._tensors, self._state, self._on_done = cache, tensors, state_metadata, on_done
        self._thread = threading.Thread(target=self._run, name=f"motion-tensor-cache-publish-rank-{cache.rank:03d}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        started = time.perf_counter()
        try:
            status = dict(self._cache.write(self._tensors, self._state))
        except Exception as exc:  # noqa: BLE001 - optional cache IO only; native state is already initialized
            status = {"mode": "write_failed", "error": repr(exc), "entry": str(self._cache.entry)}
        status["publish_seconds"] = time.perf_counter() - started
        self._tensors = None  # release the CPU copies
        self.status = status
        if self._on_done is not None:
            try:
                self._on_done(status)
            except Exception:  # noqa: BLE001
                pass

    def join(self, timeout: float | None = None) -> dict[str, Any]:
        self._thread.join(timeout)
        return self.status

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()


class _BestEffortLock:
    """flock when the filesystem supports it; otherwise proceed unlocked."""

    def __init__(self, path: Path, *, exclusive: bool, create: bool) -> None:
        self.path = path
        self.exclusive = exclusive
        self.create = create
        self.handle = None

    def __enter__(self):
        try:
            self.handle = self.path.open("a+" if self.create else "r", encoding="utf-8")
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


class MotionTensorCache:
    """Read and atomically publish one immutable rank-local tensor bundle."""

    def __init__(
        self, *, cache_dir: str | Path, identity: Mapping[str, Any], mode: str, payload_mode: Mapping[str, Any] | None = None
    ) -> None:
        self.mode = normalize_cache_mode(mode)
        if self.mode == "disabled":
            raise ValueError("MotionTensorCache is only constructed for enabled modes")
        self.identity = _json_value(identity)
        # effective payload mode of THIS run (effective_payload_mode); derived from the identity when not given
        self.payload_mode = (
            {key: bool(payload_mode.get(key)) for key in PAYLOAD_MODE_KEYS}
            if isinstance(payload_mode, Mapping) and all(key in payload_mode for key in PAYLOAD_MODE_KEYS)
            else derived_payload_mode(self.identity)
        )
        if self.identity.get("contract") != CACHE_CONTRACT:
            raise ValueError("Incompatible MotionLib cache contract")
        self.rank = int(self.identity["rank"])
        self.cache_key = str(self.identity["cache_key"])
        self.root = Path(cache_dir).expanduser().resolve()
        self.entry = self.root / self.cache_key / f"rank-{self.rank:03d}"
        self.tensor_path = self.entry / _TENSOR_FILE
        self.metadata_path = self.entry / _METADATA_FILE
        self.legacy_ready_path = self.entry / _LEGACY_READY_FILE
        self.lock_path = self.root / f"{self.cache_key}.rank-{self.rank:03d}.lock"

    # -- entry binding ----------------------------------------------------
    def _bound_metadata(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        """metadata.json of this entry when it is bound to THIS run (recorded identity and effective payload mode),
        else ``None``. Entries published before 2026-10-13T19Z kept identity in READY.json; read as plain metadata,
        never written anew. No sizes, inventories, re-stats or hashes."""
        if not (self.tensor_path.is_file() and self.metadata_path.is_file()):
            return None  # partial bundle is never a hit
        if metadata is None:
            metadata = _load_json(self.metadata_path, label="MotionLib cache metadata")
        recorded_identity = metadata.get("identity")
        mode_source: Mapping[str, Any] = metadata
        if recorded_identity is None and self.legacy_ready_path.is_file():
            mode_source = _load_json(self.legacy_ready_path, label="MotionLib cache legacy metadata")
            recorded_identity = mode_source.get("identity")
        if recorded_identity != self.identity:
            return None
        # same identity but a different EFFECTIVE payload mode (e.g. paired reference clock enabled from in-memory
        # identity records while the file manifest stays None) is not this bundle: ordinary miss, native loader
        if receipt_payload_mode(mode_source, recorded_identity) != self.payload_mode:
            return None
        return dict(metadata)

    def _read_entry(self, metadata: Mapping[str, Any] | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
        """Tensors + state metadata of the bound entry (``None`` when unbound or unreadable: the native loader runs)."""
        bound = self._bound_metadata(metadata)
        if bound is None:
            return None
        try:
            tensors = torch.load(str(self.tensor_path), map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, EOFError, pickle.UnpicklingError, ValueError, OSError):
            return None  # truncated/corrupt bundle -> miss
        if not isinstance(tensors, dict) or not tensors:
            return None
        return tensors, bound.get("state_metadata")

    # -- public API -------------------------------------------------------
    def load(self, metadata: Mapping[str, Any] | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]] | None:
        """Return the bound entry or ``None`` on a miss (any mode); the caller runs the native loader.
        ``metadata`` (already read by a caller that enumerated entries) avoids a second metadata read."""
        started = time.perf_counter()
        loaded = None
        if self.root.is_dir():
            with _BestEffortLock(self.lock_path, exclusive=False, create=self.mode == "read_write"):
                try:
                    loaded = self._read_entry(metadata)
                except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError, json.JSONDecodeError):
                    loaded = None
        if loaded is not None:
            tensors, state_metadata = loaded
            return tensors, state_metadata, {
                "mode": "reused_metadata_checked",
                "cache_key": self.cache_key,
                "entry": str(self.entry),
                "elapsed_seconds": time.perf_counter() - started,
            }
        if self.mode == "read_only":
            # read_only only means "never publish"; a miss is still the native loader, never an error
            _LOG.warning("No compatible MotionLib tensor cache at %s (read_only); using the native loader", self.entry)
        return None

    def write(self, tensors: Mapping[str, torch.Tensor], state_metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Publish a CPU-only bundle: tensors first, then metadata.json (identity + file size), then one atomic directory
        rename that replaces any previous entry (write follows an actual miss; nothing is re-read to decide)."""
        started = time.perf_counter()
        if self.mode != "read_write":
            raise ValueError("MotionTensorCache.write requires read_write mode")
        cpu_tensors = {}
        for name, value in tensors.items():
            if not torch.is_tensor(value) or value.device.type != "cpu":
                raise ValueError(f"MotionLib cache write requires CPU tensor {name!r}")
            cpu_tensors[str(name)] = value.detach().contiguous()
        if not cpu_tensors:
            raise ValueError("Refusing to write an empty MotionLib tensor cache")
        inventory = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype), "device": "cpu"}
            for name, value in sorted(cpu_tensors.items())
        }
        normalized_state = _json_value(state_metadata)
        metadata = {
            "contract": CACHE_CONTRACT,
            "identity": self.identity,
            "payload_mode": dict(self.payload_mode),
            "tensor_inventory": inventory,
            "state_metadata": normalized_state,
            "tensor_bytes_total": int(sum(v.numel() * v.element_size() for v in cpu_tensors.values())),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        parent = self.entry.parent
        parent.mkdir(parents=True, exist_ok=True)
        # The bundle is prepared and made durable OUTSIDE the publication lock: torch.save + fsync of a 16-21 GB file
        # takes minutes, and a reader of the complete previous entry (shared lock) must never wait behind it. The
        # exclusive lock covers only the atomic swap, so a reader that holds the shared lock finishes its read before
        # the entry changes under it. write() runs only after this run's load() missed: the freshly computed tensors
        # REPLACE whatever is at the entry path; nothing is re-read to decide.
        temporary = Path(tempfile.mkdtemp(prefix=f".rank-{self.rank:03d}.", suffix=".tmp", dir=parent))
        try:
            tensor_path = temporary / _TENSOR_FILE
            metadata_path = temporary / _METADATA_FILE
            torch.save(cpu_tensors, str(tensor_path))
            with tensor_path.open("rb") as handle:
                os.fsync(handle.fileno())
            metadata["tensor_size_bytes"] = tensor_path.stat().st_size
            _write_json(metadata_path, metadata)  # written last: a complete bundle is tensors + metadata
            _fsync_directory(temporary)
            stale = None
            with _BestEffortLock(self.lock_path, exclusive=True, create=True):
                # the lock covers only the two renames that swap the entry; durability of the swap (parent fsync) and the
                # removal of the displaced previous bundle happen after the lock is released
                if self.entry.exists():
                    stale = self.entry.with_name(f".{self.entry.name}.invalid-{uuid.uuid4().hex}")
                    self.entry.replace(stale)
                temporary.replace(self.entry)
            _fsync_directory(parent)
            if stale is not None:
                shutil.rmtree(stale, ignore_errors=True)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        return {
            "mode": "computed_and_cached",
            "cache_key": self.cache_key,
            "entry": str(self.entry),
            "tensor_bytes_total": metadata["tensor_bytes_total"],
            "elapsed_seconds": time.perf_counter() - started,
        }
