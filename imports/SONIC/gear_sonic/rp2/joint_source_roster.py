"""In-memory joint-source roster: the identity consumer seam for rosters without a manifest FILE.

A unified cohort carries a ``manifest.json`` that the motion library opens to
build ``source::source_motion_key`` identities. The published direct-directory
roster (``gear_sonic/rp2/direct_directory_roster.py``, contract
``direct_roster_identity_v1``) has no such file: its rank view labels
``manifest_path`` with the pack ROOT DIRECTORY (never opened) and carries the
identities IN MEMORY as ``identity_records`` (one record per alias of the whole
layout, sorted by alias, identical on every rank) with ``identity_source_order``.
This module wraps those records into the mapping the motion library consumes
through ``motion_lib_cfg.joint_source_roster`` with the same validations as a
manifest file. Pure python (no torch), deterministic, identical on every rank.
No contract label is required to consume a roster (no admission gate).

Persistence rule of the producer contract: the only identity stable across
layouts is ``"<source>::<source_motion_key>"``. ``asset_signature`` and
``group_index`` are LAYOUT-LOCAL labels of the direct layout, so they are NOT
carried into the motion library's identity -> object records (which are
checkpointed and gathered across ranks); ``object_key`` (the pack's own label
or its documented fallback) is.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

IDENTITY_RECORDS_KEY = "identity_records"
IDENTITY_SOURCE_ORDER_KEY = "identity_source_order"
IDENTITY_CONTRACT_KEY = "identity_contract"
_RECORD_TEXT_FIELDS = ("alias", "source", "source_motion_key")
# Layout-local labels of direct_roster_identity_v1: consumed for nothing persistent.
LAYOUT_LOCAL_FIELDS = ("asset_signature", "group_index")


def _text(value: Any, *, label: str, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{label} is missing")
        return None
    text = str(value).strip()
    if not text:
        if required:
            raise ValueError(f"{label} is empty")
        return None
    return text


def _fps(value: Any, *, label: str) -> float | None:
    """Real meta fps (finite, positive) or None when the producer had no meta; never a filled default."""
    if value is None:
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: source_sample_fps_hz {value!r} is not a number") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{label}: source_sample_fps_hz {fps} must be finite and positive")
    return fps


def normalize_record(record: Any, *, label: str, keep_asset_signature: bool) -> dict[str, Any]:
    """One validated identity record (plain dict). Ordinary errors for missing/empty identity fields."""
    if not isinstance(record, Mapping):
        raise TypeError(f"{label}: roster record must be a mapping, got {type(record).__name__}")
    out: dict[str, Any] = {}
    for field in _RECORD_TEXT_FIELDS:
        out[field] = _text(record.get(field), label=f"{label}: record field {field!r}", required=True)
    object_key = _text(record.get("object_key"), label=f"{label}: {out['alias']} object_key", required=False)
    if object_key is not None:
        out["object_key"] = object_key
    if keep_asset_signature:
        signature = _text(record.get("asset_signature"), label=f"{label}: {out['alias']} asset_signature", required=False)
        if signature is not None:
            out["asset_signature"] = signature
    fps = _fps(record.get("source_sample_fps_hz"), label=f"{label}: {out['alias']}")
    if fps is not None:
        out["source_sample_fps_hz"] = fps
    frames = record.get("num_frames")
    if frames is not None:
        out["num_frames"] = int(frames)
    return out


def normalize_joint_source_roster(roster: Any) -> dict[str, Any]:
    """Validate an in-memory roster mapping -> plain dict with ordered sources.

    Keys: ``records`` (non-empty sequence, unique aliases, required), ``label`` (text;
    the directory label, never opened; required), ``world_size`` (positive int,
    required), optional ``sources`` (explicit order covering every record source),
    optional ``complete_roster`` (default True: records may list motions this rank
    does not load, exactly like unified cohorts), optional informational
    ``contract``. ``keep_asset_signature`` (default False) controls whether the
    layout-local ``asset_signature`` is kept in the records.
    """
    if not isinstance(roster, Mapping):
        raise TypeError(f"joint_source_roster must be a mapping, got {type(roster).__name__}")
    label = _text(roster.get("label"), label="joint_source_roster.label", required=True)
    try:
        world_size = int(roster.get("world_size"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: joint_source_roster.world_size must be a positive integer") from exc
    if world_size <= 0:
        raise ValueError(f"{label}: joint_source_roster.world_size must be positive, got {world_size}")
    raw_records = roster.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)) or len(raw_records) == 0:
        raise ValueError(f"{label}: no roster records")
    keep_signature = bool(roster.get("keep_asset_signature", False))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in raw_records:
        normalized = normalize_record(record, label=label, keep_asset_signature=keep_signature)
        if normalized["alias"] in seen:
            raise ValueError(f"{label}: duplicate alias {normalized['alias']}")
        seen.add(normalized["alias"])
        records.append(normalized)
    record_sources = list(dict.fromkeys(record["source"] for record in records))
    explicit = roster.get("sources")
    if explicit is None:
        sources = record_sources
    else:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            raise TypeError(f"{label}: joint_source_roster.sources must be a sequence of names")
        sources = [str(name) for name in explicit]
        if len(set(sources)) != len(sources):
            raise ValueError(f"{label}: joint_source_roster.sources has duplicates: {sources}")
        unknown = [name for name in record_sources if name not in sources]
        if unknown:
            raise ValueError(f"{label}: record sources without an allocation: {unknown}")
    out = {
        "label": label,
        "world_size": world_size,
        "sources": sources,
        "records": records,
        "complete_roster": bool(roster.get("complete_roster", True)),
    }
    contract = roster.get("contract")
    if contract is not None:
        out["contract"] = str(contract)
    return out


def joint_source_roster_from_view(view: Mapping[str, Any]) -> dict[str, Any]:
    """The in-memory roster of a resolved rank view that carries ``identity_records``.

    Producer contract ``direct_roster_identity_v1``: ``identity_records`` (whole
    layout, sorted by alias, identical on every rank), ``identity_source_order``
    (the explicit DirectSource names), existing ``world_size`` and ``manifest_path``
    (directory label, never opened). The records are passed through as they are:
    alias == MotionLib key == original robot stem, ``source_motion_key`` == original
    stem, ``source_sample_fps_hz`` real or None (the motion library then reads the
    robot payload's own fps), ``object_key`` kept; the layout-local
    ``asset_signature`` / ``group_index`` are not promoted into persistent records.
    Slot order, rank slices and the motion library's key order are untouched. A view
    without ``identity_records`` cannot provide identities (ordinary error); no
    derivation from slots or any other guess is attempted.
    """
    if not isinstance(view, Mapping):
        raise TypeError("joint roster view must be a mapping")
    label = _text(view.get("manifest_path"), label="view.manifest_path", required=True)
    records = view.get(IDENTITY_RECORDS_KEY)
    if records is None:
        raise ValueError(
            f"{label}: joint roster view carries no {IDENTITY_RECORDS_KEY} "
            "(direct_roster_identity_v1) and manifest_path is not a manifest file; "
            "source::motion identities cannot be tracked for this roster"
        )
    roster: dict[str, Any] = {
        "label": label,
        "world_size": view.get("world_size"),
        "records": list(records),
        "complete_roster": True,
        "keep_asset_signature": False,
    }
    order = view.get(IDENTITY_SOURCE_ORDER_KEY)
    if order is not None:
        roster["sources"] = list(order)
    contract = view.get(IDENTITY_CONTRACT_KEY)
    if contract is not None:
        roster["contract"] = str(contract)
    return normalize_joint_source_roster(roster)


__all__ = [
    "IDENTITY_CONTRACT_KEY",
    "IDENTITY_RECORDS_KEY",
    "IDENTITY_SOURCE_ORDER_KEY",
    "LAYOUT_LOCAL_FIELDS",
    "joint_source_roster_from_view",
    "normalize_joint_source_roster",
    "normalize_record",
]
