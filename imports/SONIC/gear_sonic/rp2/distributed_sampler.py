"""Pure helpers for heterogeneous-rank adaptive-sampler synchronization.

The payloads in this module contain only Python primitives, so callers can pass
them directly to ``torch.distributed.all_gather_object``.  Bins are identified
by source, motion alias, and a half-open source-frame interval.  No operation
depends on rank-local tensor positions or tensor shapes.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


AGGREGATE_PAYLOAD_CONTRACT = "joint_adaptive_sampler_aggregate_v1"
CHECKPOINT_CONTRACT = "joint_adaptive_sampler_checkpoint_v1"
MOTION_IDENTITY_METADATA_KEY = "motion_identity"

_BIN_RECORD_FIELDS = frozenset(
    {
        "source",
        "motion_alias",
        "start_frame",
        "end_frame",
        "episode_count",
        "failure_count",
        "terminal_failure_count",
        "metadata",
    }
)


def _require_nonnegative_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return normalized


def _canonicalize_metadata(value: object, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite value")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} keys must be strings")
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            normalized[key] = _canonicalize_metadata(
                value[key], path=f"{path}.{key}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must contain only JSON-compatible Python primitives, "
        f"not {type(value).__name__}"
    )


def _metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    return json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_kind(value: object) -> str:
    # the categories JSON text distinguishes: true/false, integers, floats, strings, null, objects, arrays
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__


def canonical_metadata_equal(left: object, right: object) -> bool:
    """``_metadata_fingerprint(left) == _metadata_fingerprint(right)`` for canonicalizable values, decided
    structurally without building the two JSON strings: same JSON category (``1``, ``1.0`` and ``True`` differ),
    equal key sets and element counts, floats by their JSON rendering (``-0.0`` differs from ``0.0``), strings and
    integers by value; key order never matters (the fingerprint sorts keys). Used by the restore-time roster
    comparison over every bin (2026-10-15: two fingerprints per bin were 26 s of the restore at 706,802 bins)."""
    kind = _json_kind(left)
    if kind != _json_kind(right):
        return False
    if kind == "object":
        if left.keys() != right.keys():
            return False
        return all(canonical_metadata_equal(left[key], right[key]) for key in left)
    if kind == "array":
        return len(left) == len(right) and all(
            canonical_metadata_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if kind == "float":
        return float.__repr__(left) == float.__repr__(right)
    if kind == "int":
        return int(left) == int(right)
    return left == right


def _require_world_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("world_size must be an integer")
    if value <= 0:
        raise ValueError("world_size must be positive")
    return value


@dataclass(frozen=True, order=True)
class BinKey:
    """Stable identity for one half-open adaptive-sampling frame bin."""

    source: str
    motion_alias: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a nonempty string")
        if not isinstance(self.motion_alias, str) or not self.motion_alias.strip():
            raise ValueError("motion_alias must be a nonempty string")
        if (
            isinstance(self.start_frame, bool)
            or not isinstance(self.start_frame, int)
            or isinstance(self.end_frame, bool)
            or not isinstance(self.end_frame, int)
        ):
            raise TypeError("start_frame and end_frame must be integers")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError(
                "Bin frames must satisfy 0 <= start_frame < end_frame"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "BinKey":
        return cls(
            source=record.get("source"),
            motion_alias=record.get("motion_alias"),
            start_frame=record.get("start_frame"),
            end_frame=record.get("end_frame"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "motion_alias": self.motion_alias,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


@dataclass(frozen=True)
class BinStatistics:
    """Mergeable sufficient statistics and immutable metadata for one bin."""

    key: BinKey
    episode_count: float
    failure_count: float
    terminal_failure_count: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.key, BinKey):
            raise TypeError("key must be a BinKey")
        _require_nonnegative_finite(self.episode_count, label="episode_count")
        _require_nonnegative_finite(self.failure_count, label="failure_count")
        _require_nonnegative_finite(
            self.terminal_failure_count, label="terminal_failure_count"
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        _canonicalize_metadata(self.metadata)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "BinStatistics":
        if not isinstance(record, Mapping):
            raise TypeError("Each bin record must be a mapping")
        unknown = set(record) - _BIN_RECORD_FIELDS
        if unknown:
            raise ValueError(f"Unknown bin-record fields: {sorted(unknown)}")
        return cls(
            key=BinKey.from_record(record),
            episode_count=_require_nonnegative_finite(
                record.get("episode_count"), label="episode_count"
            ),
            failure_count=_require_nonnegative_finite(
                record.get("failure_count"), label="failure_count"
            ),
            terminal_failure_count=_require_nonnegative_finite(
                record.get("terminal_failure_count", 0.0),
                label="terminal_failure_count",
            ),
            metadata=_canonicalize_metadata(record.get("metadata", {})),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            **self.key.to_record(),
            "episode_count": _require_nonnegative_finite(
                self.episode_count, label="episode_count"
            ),
            "failure_count": _require_nonnegative_finite(
                self.failure_count, label="failure_count"
            ),
            "terminal_failure_count": _require_nonnegative_finite(
                self.terminal_failure_count, label="terminal_failure_count"
            ),
            "metadata": _canonicalize_metadata(self.metadata),
        }


def _coerce_statistics(value: BinStatistics | Mapping[str, Any]) -> BinStatistics:
    if isinstance(value, BinStatistics):
        return BinStatistics.from_record(value.to_record())
    return BinStatistics.from_record(value)


def _merge_bin_statistics(
    records: Iterable[BinStatistics | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the bin records of one aggregate: unique keys, sorted, canonical metadata."""
    grouped: dict[BinKey, BinStatistics] = {}
    for value in records:
        statistics = _coerce_statistics(value)
        if statistics.key in grouped:
            raise ValueError(f"Duplicate aggregate bin key: {statistics.key}")
        grouped[statistics.key] = statistics
    return [grouped[key].to_record() for key in sorted(grouped)]


class CanonicalAggregate:
    """PRIVATE owned keyed-sampler state (not a checkpoint type, never handed out as a Mapping).

    Built from a fully validated aggregate payload (``from_payload``: a restored checkpoint) or by the
    MotionLib's lifecycle-start table; keys are sorted, metadata dicts are canonical copies owned by
    this object and never exposed without a copy. ``apply_slot_deltas`` is the steady-state update.
    ``export()`` produces the ordinary aggregate payload with FRESH record dicts and metadata copies.
    Consecutive states whose key set did not change share the ``keys`` tuple (so a consumer may cache
    positions on its identity).
    """

    __slots__ = ("world_size", "keys", "episodes", "failures", "terminals", "metadata", "_index")

    def __init__(self, world_size, keys, episodes, failures, terminals, metadata):
        self.world_size = int(world_size)
        self.keys = tuple(keys)
        self.episodes = list(episodes)
        self.failures = list(failures)
        self.terminals = list(terminals)
        self.metadata = tuple(metadata)
        self._index = None

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def from_normalized(cls, normalized: Mapping[str, Any]) -> "CanonicalAggregate":
        """Private state from a payload that ``_normalize_aggregate_payload`` (deserialize / rebase) has JUST
        produced: its records are fresh ``BinStatistics.to_record()`` dicts - unique sorted keys, validated
        counts, canonical metadata copies - so they are taken over as the owned state without a second
        normalization and validation pass. Equal to ``from_payload(normalized)`` by construction; the caller
        hands the records over (it must not keep mutating references to them). The restore path uses it once
        (2026-10-15: the public setter's ``from_payload`` re-normalized 706,802 already-normalized bins, 21 s)."""
        keys, episodes, failures, terminals, metadata = [], [], [], [], []
        for record in normalized["bins"]:
            keys.append(BinKey(record["source"], record["motion_alias"], record["start_frame"], record["end_frame"]))
            episodes.append(float(record["episode_count"]))
            failures.append(float(record["failure_count"]))
            terminals.append(float(record["terminal_failure_count"]))
            metadata.append(record["metadata"])
        return cls(normalized["world_size"], keys, episodes, failures, terminals, metadata)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CanonicalAggregate":
        """Full validation of an ordinary aggregate payload, then a private copy of its state."""
        normalized = _normalize_aggregate_payload(payload)
        keys, episodes, failures, terminals, metadata = [], [], [], [], []
        for record in normalized["bins"]:
            statistics = BinStatistics.from_record(record)
            keys.append(statistics.key)
            episodes.append(float(statistics.episode_count))
            failures.append(float(statistics.failure_count))
            terminals.append(float(statistics.terminal_failure_count))
            metadata.append(_canonicalize_metadata(statistics.metadata))
        return cls(normalized["world_size"], keys, episodes, failures, terminals, metadata)

    def apply_slot_deltas(self, contributions: Sequence[Sequence[Sequence[float]]]) -> "CanonicalAggregate":
        """Accumulate per-rank count deltas addressed by GLOBAL SLOT (position in ``keys``).

        ``contributions`` holds, in rank order, one ``(slots, episodes, failures, terminals)`` quadruple of
        equal-length sequences per rank (an idle rank contributes empty sequences). This is the steady-state
        synchronization path: the key set and metadata of this state are static for the roster lifecycle,
        so no record dicts, normalization or metadata comparison happen here. The result equals the public
        former per-rank merge + accumulate path exactly: a slot contributed by one
        rank adds that value directly (``state + v``), a slot contributed by several ranks adds their
        correctly rounded sum (``math.fsum``, as the public merge does) once; priors never re-enter because
        every slot already exists. Values are expected non-negative and finite (the caller clamps/checks the
        tensors once); a slot outside the table is an ordinary error.
        """
        episodes, failures, terminals = list(self.episodes), list(self.failures), list(self.terminals)
        size = len(self.keys)
        pending: dict[int, list[list[float]]] = {}
        for slots, delta_episodes, delta_failures, delta_terminals in contributions:
            for slot, de, df, dt in zip(slots, delta_episodes, delta_failures, delta_terminals, strict=True):
                slot = int(slot)
                if not 0 <= slot < size:
                    raise ValueError(f"Sampler delta slot {slot} is outside the global key table of {size} bins")
                bucket = pending.get(slot)
                if bucket is None:
                    pending[slot] = [[float(de)], [float(df)], [float(dt)]]
                else:
                    bucket[0].append(float(de))
                    bucket[1].append(float(df))
                    bucket[2].append(float(dt))
        for slot, (es, fs, ts) in pending.items():
            episodes[slot] += es[0] if len(es) == 1 else math.fsum(es)
            failures[slot] += fs[0] if len(fs) == 1 else math.fsum(fs)
            terminals[slot] += ts[0] if len(ts) == 1 else math.fsum(ts)
        return CanonicalAggregate._with_counts(self, episodes, failures, terminals)

    @classmethod
    def _with_counts(cls, previous: "CanonicalAggregate", episodes, failures, terminals) -> "CanonicalAggregate":
        """A new state with the same key set: keys/metadata/index are shared (immutable here), counts are new lists."""
        out = cls.__new__(cls)
        out.world_size = previous.world_size
        out.keys = previous.keys
        out.episodes, out.failures, out.terminals = episodes, failures, terminals
        out.metadata = previous.metadata
        out._index = previous._index
        return out

    def index(self) -> dict[BinKey, int]:
        if self._index is None:
            self._index = {key: position for position, key in enumerate(self.keys)}
        return self._index

    def export(self) -> dict[str, Any]:
        """Ordinary aggregate payload (the checkpoint aggregate schema): fresh record dicts and
        fresh canonical metadata copies every call, so no caller can reach this object's state through it."""
        bins = []
        for position, key in enumerate(self.keys):
            bins.append({
                **key.to_record(),
                "episode_count": _require_nonnegative_finite(self.episodes[position], label="episode_count"),
                "failure_count": _require_nonnegative_finite(self.failures[position], label="failure_count"),
                "terminal_failure_count": _require_nonnegative_finite(self.terminals[position], label="terminal_failure_count"),
                "metadata": _canonicalize_metadata(self.metadata[position]),
            })
        return {
            "contract": AGGREGATE_PAYLOAD_CONTRACT,
            "world_size": self.world_size,
            "contributing_ranks": list(range(self.world_size)),
            "bins": bins,
        }


def serialize_canonical_checkpoint(canonical: CanonicalAggregate, *, rng_state: Any) -> dict[str, Any]:
    """Checkpoint of a private canonical state (ordinary ``CHECKPOINT_CONTRACT`` schema).

    Equal to ``serialize_sampler_checkpoint(canonical.export(), rng_state=...)`` without the
    second full normalization; the aggregate records and nested metadata are fresh copies.
    """
    if not isinstance(canonical, CanonicalAggregate):
        raise TypeError("serialize_canonical_checkpoint needs a CanonicalAggregate")
    if rng_state is None:
        raise ValueError("rng_state is required for sampler checkpointing")
    return {
        "contract": CHECKPOINT_CONTRACT,
        "rng_state": copy.deepcopy(rng_state),
        "aggregate": canonical.export(),
    }


def _normalize_aggregate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Aggregate sampler payload must be a mapping")
    if payload.get("contract") != AGGREGATE_PAYLOAD_CONTRACT:
        raise ValueError("Unsupported adaptive-sampler aggregate contract")
    world_size = _require_world_size(payload.get("world_size"))
    contributing_ranks = payload.get("contributing_ranks")
    if (
        not isinstance(contributing_ranks, Sequence)
        or isinstance(contributing_ranks, (str, bytes))
        or list(contributing_ranks) != list(range(world_size))
    ):
        raise ValueError(
            "Aggregate contributing_ranks must cover every rank in order"
        )
    bins = payload.get("bins")
    if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes)):
        raise TypeError("Aggregate bins must be a sequence")
    return {
        "contract": AGGREGATE_PAYLOAD_CONTRACT,
        "world_size": world_size,
        "contributing_ranks": list(contributing_ranks),
        "bins": _merge_bin_statistics(bins),
    }


def serialize_sampler_checkpoint(
    aggregate_payload: Mapping[str, Any],
    *,
    rng_state: Any,
) -> dict[str, Any]:
    """Build a checkpoint-safe sampler state (keyed aggregate + RNG state)."""

    if rng_state is None:
        raise ValueError("rng_state is required for sampler checkpointing")
    normalized = _normalize_aggregate_payload(aggregate_payload)
    return {
        "contract": CHECKPOINT_CONTRACT,
        "rng_state": copy.deepcopy(rng_state),
        "aggregate": normalized,
    }


def deserialize_sampler_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and restore a sampler checkpoint from its actual contents.

    Legacy checkpoints that still carry a ``roster_manifest_sha256`` field load
    normally; the field is ignored. Whether the restored bins match the active
    roster is decided by the caller from the bin keys and metadata.
    """

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Sampler checkpoint must be a mapping")
    if checkpoint.get("contract") != CHECKPOINT_CONTRACT:
        raise ValueError("Unsupported adaptive-sampler checkpoint contract")
    if "rng_state" not in checkpoint or checkpoint["rng_state"] is None:
        raise ValueError("Sampler checkpoint is missing rng_state")
    aggregate = _normalize_aggregate_payload(checkpoint.get("aggregate"))
    return {
        "contract": CHECKPOINT_CONTRACT,
        "rng_state": copy.deepcopy(checkpoint["rng_state"]),
        "aggregate": aggregate,
    }


def _require_motion_identity(
    value: object,
    *,
    source: str,
    label: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing semantic motion identity")
    identity = value.strip()
    if not identity.startswith(f"{source}::"):
        raise ValueError(
            f"{label} motion identity {identity!r} does not belong to "
            f"source {source!r}"
        )
    return identity


def _transition_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _canonicalize_metadata(metadata)
    if not isinstance(normalized, dict):
        raise TypeError("Sampler bin metadata must be a mapping")
    normalized.pop(MOTION_IDENTITY_METADATA_KEY, None)
    # Revision labels and locator paths are generation-local labels, not bin
    # content. A motion is treated as a corrected replacement (statistics reset)
    # when its remaining bin metadata actually changed, or when the caller names
    # it explicitly; a label rename alone keeps the keyed statistics.
    normalized.pop("promoted_revision", None)
    return normalized


def migrate_sampler_checkpoint_roster_transition(
    checkpoint: Mapping[str, Any],
    *,
    candidate_bins: Iterable[BinStatistics | Mapping[str, Any]],
    parent_motion_identity_by_alias: Mapping[str, str],
    initial_episode_prior: float,
    initial_failure_prior: float,
    replacement_motion_identities: Iterable[str] = (),
) -> dict[str, Any]:
    """Migrate keyed sampler history across one roster generation.

    Unchanged motions retain keyed statistics. Added motions and replaced
    motions receive one global prior per bin. A motion counts as replaced when
    it is named in ``replacement_motion_identities`` or when its bin metadata
    (other than the identity and revision labels) differs between the parent
    checkpoint and the candidate roster. Replacements must retain semantic
    identity, alias, and frame-bin layout, so a data correction cannot silently
    change the training horizon or topology while reusing optimizer state.
    The returned checkpoint carries a ``transition`` summary (added / replaced
    motion identities) for logging.
    """

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Sampler checkpoint must be a mapping")
    explicit_replacements: set[str] = set()
    for value in replacement_motion_identities:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "replacement_motion_identities must contain nonempty strings"
            )
        explicit_replacements.add(value.strip())
    episode_prior = _require_nonnegative_finite(
        initial_episode_prior,
        label="initial_episode_prior",
    )
    failure_prior = _require_nonnegative_finite(
        initial_failure_prior,
        label="initial_failure_prior",
    )
    parent = deserialize_sampler_checkpoint(checkpoint)

    if not isinstance(parent_motion_identity_by_alias, Mapping):
        raise TypeError("parent_motion_identity_by_alias must be a mapping")
    parent_identities: dict[str, str] = {}
    for alias, identity in parent_motion_identity_by_alias.items():
        if not isinstance(alias, str) or not alias:
            raise ValueError("Parent motion aliases must be nonempty strings")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Parent semantic motion identities must be nonempty strings")
        parent_identities[alias] = identity

    candidate_by_semantic_bin: dict[
        tuple[str, str, int, int], BinStatistics
    ] = {}
    candidate_alias_by_identity: dict[str, str] = {}
    candidate_identity_by_source_alias: dict[tuple[str, str], str] = {}
    for value in candidate_bins:
        statistics = _coerce_statistics(value)
        identity = _require_motion_identity(
            statistics.metadata.get(MOTION_IDENTITY_METADATA_KEY),
            source=statistics.key.source,
            label=f"candidate bin {statistics.key}",
        )
        source_alias = (statistics.key.source, statistics.key.motion_alias)
        previous_identity = candidate_identity_by_source_alias.setdefault(
            source_alias,
            identity,
        )
        if previous_identity != identity:
            raise ValueError(
                "Candidate alias maps to multiple semantic motion identities: "
                f"{source_alias}"
            )
        previous_alias = candidate_alias_by_identity.setdefault(
            identity,
            statistics.key.motion_alias,
        )
        if previous_alias != statistics.key.motion_alias:
            raise ValueError(
                "Candidate semantic motion identity maps to multiple aliases: "
                f"{identity}"
            )
        semantic_key = (
            statistics.key.source,
            identity,
            statistics.key.start_frame,
            statistics.key.end_frame,
        )
        existing = candidate_by_semantic_bin.get(semantic_key)
        if existing is not None:
            if (
                existing.key != statistics.key
                or _metadata_fingerprint(
                    _canonicalize_metadata(existing.metadata)
                )
                != _metadata_fingerprint(
                    _canonicalize_metadata(statistics.metadata)
                )
            ):
                raise ValueError(
                    "Candidate ranks disagree on duplicate semantic bin "
                    f"{semantic_key}"
                )
            continue
        candidate_by_semantic_bin[semantic_key] = statistics
    if not candidate_by_semantic_bin:
        raise ValueError("Candidate roster has no adaptive-sampling bins")

    parent_by_semantic_bin: dict[
        tuple[str, str, int, int], BinStatistics
    ] = {}
    parent_alias_by_identity: dict[str, str] = {}
    for record in parent["aggregate"]["bins"]:
        statistics = BinStatistics.from_record(record)
        embedded_identity = statistics.metadata.get(
            MOTION_IDENTITY_METADATA_KEY
        )
        mapped_identity = parent_identities.get(statistics.key.motion_alias)
        if embedded_identity is None:
            identity = _require_motion_identity(
                mapped_identity,
                source=statistics.key.source,
                label=f"parent bin {statistics.key}",
            )
        else:
            identity = _require_motion_identity(
                embedded_identity,
                source=statistics.key.source,
                label=f"parent bin {statistics.key}",
            )
            if mapped_identity is not None and mapped_identity != identity:
                raise ValueError(
                    "Parent roster identity map disagrees with checkpoint "
                    f"metadata for {statistics.key.motion_alias!r}"
                )
        previous_alias = parent_alias_by_identity.setdefault(
            identity,
            statistics.key.motion_alias,
        )
        if previous_alias != statistics.key.motion_alias:
            raise ValueError(
                "Parent semantic motion identity maps to multiple aliases: "
                f"{identity}"
            )
        semantic_key = (
            statistics.key.source,
            identity,
            statistics.key.start_frame,
            statistics.key.end_frame,
        )
        if semantic_key in parent_by_semantic_bin:
            raise ValueError(
                f"Parent checkpoint has duplicate semantic bin {semantic_key}"
            )
        parent_by_semantic_bin[semantic_key] = statistics

    removed_bins = sorted(
        set(parent_by_semantic_bin) - set(candidate_by_semantic_bin)
    )
    if removed_bins:
        removed_motions = sorted({key[1] for key in removed_bins})
        raise ValueError(
            "Roster transition removed existing sampler bins: "
            f"{removed_motions[:8]}"
        )

    parent_motion_keys = set(parent_alias_by_identity)
    candidate_motion_keys = set(candidate_alias_by_identity)
    added_motion_keys = sorted(candidate_motion_keys - parent_motion_keys)
    extra_existing_bins = sorted(
        key
        for key in set(candidate_by_semantic_bin) - set(parent_by_semantic_bin)
        if key[1] in parent_motion_keys
    )
    if extra_existing_bins:
        mutated = sorted({key[1] for key in extra_existing_bins})
        raise ValueError(
            "Roster transition changed existing motion bin "
            f"layout: {mutated[:8]}"
        )

    unknown_replacements = explicit_replacements - (
        parent_motion_keys & candidate_motion_keys
    )
    if unknown_replacements:
        raise ValueError(
            "replacement_motion_identities are not existing motions: "
            f"{sorted(unknown_replacements)[:8]}"
        )

    metadata_mutations = set()
    for semantic_key, parent_statistics in parent_by_semantic_bin.items():
        candidate_statistics = candidate_by_semantic_bin[semantic_key]
        identity = semantic_key[1]
        if (
            parent_statistics.key.motion_alias
            != candidate_statistics.key.motion_alias
        ):
            raise ValueError(
                "Roster transition changed the alias for "
                f"{identity}"
            )
        if _metadata_fingerprint(
            _transition_metadata(parent_statistics.metadata)
        ) != _metadata_fingerprint(
            _transition_metadata(candidate_statistics.metadata)
        ):
            metadata_mutations.add(identity)

    # Replaced = explicitly named or actually changed bin content.
    replacement_motion_keys = explicit_replacements | metadata_mutations

    migrated_bins = []
    for semantic_key in sorted(candidate_by_semantic_bin):
        candidate_statistics = candidate_by_semantic_bin[semantic_key]
        parent_statistics = parent_by_semantic_bin.get(semantic_key)
        replaced = semantic_key[1] in replacement_motion_keys
        migrated_bins.append(
            BinStatistics(
                key=candidate_statistics.key,
                episode_count=(
                    parent_statistics.episode_count
                    if parent_statistics is not None and not replaced
                    else episode_prior
                ),
                failure_count=(
                    parent_statistics.failure_count
                    if parent_statistics is not None and not replaced
                    else failure_prior
                ),
                terminal_failure_count=(
                    parent_statistics.terminal_failure_count
                    if parent_statistics is not None and not replaced
                    else 0.0
                ),
                metadata=candidate_statistics.metadata,
            ).to_record()
        )
    candidate_aggregate = {
        "contract": AGGREGATE_PAYLOAD_CONTRACT,
        "world_size": parent["aggregate"]["world_size"],
        "contributing_ranks": list(
            range(parent["aggregate"]["world_size"])
        ),
        "bins": migrated_bins,
    }
    migrated = serialize_sampler_checkpoint(
        candidate_aggregate,
        rng_state=parent["rng_state"],
    )
    migrated["transition"] = {
        "added_motion_identities": added_motion_keys,
        "replaced_motion_identities": sorted(replacement_motion_keys),
    }
    return migrated


PROVENANCE_KEY = "adaptive_sampling_provenance"
COUNTING_MODES = ("episode", "step")
RESTORE_RETAIN = "retain"
RESTORE_REBASE = "rebase"
RESTORE_NEUTRAL = "neutral"


def provenance_counting_mode(provenance: object) -> str:
    """``episode`` / ``step`` / ``unknown`` from a saved ``adaptive_sampling_provenance``.

    Older checkpoints carry no provenance (-> ``unknown``); a malformed value is
    never an error, it is simply unknown.
    """
    if not isinstance(provenance, Mapping):
        return "unknown"
    mode = provenance.get("counting_mode")
    if isinstance(mode, str) and mode in COUNTING_MODES:
        return mode
    flag = provenance.get("update_on_episode_end_only")
    if isinstance(flag, bool):
        return "episode" if flag else "step"
    return "unknown"


def _provenance_number(provenance: Mapping[str, Any], key: str, default: float) -> float:
    value = provenance.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


ADAPTIVE_COUNT_STATE_KEYS = (
    "joint_distributed_sampler",
    "adaptive_sampling_by_motion",
    "adp_samp_num_episodes",
)


def plan_adaptive_restore(
    saved_provenance: object,
    live_provenance: Mapping[str, Any],
    *,
    saved_counts_present: bool = True,
) -> dict[str, Any]:
    """Decide how saved adaptive counts relate to the LIVE counting semantics.

    Pure and deterministic (every rank reaches the same decision from the same
    checkpoint and config; no collective). ``saved_counts_present=False`` (the
    checkpoint carries none of ``ADAPTIVE_COUNT_STATE_KEYS``, e.g. a resume from
    a non-adaptive run) has nothing to compare: ``retain`` with saved mode
    ``none``, so the existing no-statistics handling applies unchanged. Returns
    ``{"action", "reason", "saved_mode", "live_mode", "prior_delta",
    "failure_scale"}`` with action:

    * ``retain``  -- same counting mode, bin credit, bin size, prior and failure
      multiplier: the saved counts ARE live-semantics counts. A checkpoint
      WITHOUT provenance is retained only under live ``step`` counting (the
      legacy semantics of every pre-provenance run; the prior cannot be
      compared, counts are kept as before).
    * ``rebase``  -- same mode/credit/bin size but a different prior and/or
      ``failure_counts_multiplier``: exact counter rebase (the prior is one
      constant added to every bin's episode AND failure count; the multiplier
      scales the observed failures linearly).
    * ``neutral`` -- the saved counts cannot be expressed in the live semantics
      (unknown or different counting mode, different bin credit or bin size, a
      non-positive saved multiplier): start from the live neutral prior with a
      plain diagnostic. Old step/unknown history never becomes completed-episode
      warmup, and the next checkpoint is labeled with counts that really are
      live-semantics counts.
    """
    live_mode = provenance_counting_mode(live_provenance)
    saved_mode = provenance_counting_mode(saved_provenance)
    plan = {
        "saved_mode": saved_mode,
        "live_mode": live_mode,
        "prior_delta": 0.0,
        "failure_scale": 1.0,
    }
    if not saved_counts_present:
        return {
            **plan,
            "saved_mode": "none",
            "action": RESTORE_RETAIN,
            "reason": "checkpoint carries no adaptive per-bin statistics; nothing to compare",
        }
    if saved_mode == "unknown":
        if live_mode == "episode":
            return {
                **plan,
                "action": RESTORE_NEUTRAL,
                "reason": "checkpoint carries no counting provenance; its counts cannot be "
                "confirmed as completed-episode counts (live update_on_episode_end_only)",
            }
        return {
            **plan,
            "action": RESTORE_RETAIN,
            "reason": "checkpoint carries no counting provenance; live step counting keeps the "
            "saved counts as before (saved prior unknown, not compared)",
        }
    if saved_mode != live_mode:
        return {
            **plan,
            "action": RESTORE_NEUTRAL,
            "reason": f"saved counting mode {saved_mode!r} differs from the live mode {live_mode!r}",
        }
    saved = saved_provenance  # a Mapping here (saved_mode is known)
    for key, label in (
        ("credit_sampled_start_bin", "bin credit (credit_sampled_start_bin)"),
        ("bin_size", "bin size"),
    ):
        if key in saved and key in live_provenance and saved[key] != live_provenance[key]:
            return {
                **plan,
                "action": RESTORE_NEUTRAL,
                "reason": f"saved {label} {saved[key]!r} differs from the live value {live_provenance[key]!r}",
            }
    live_prior = _provenance_number(live_provenance, "init_num_failures", 1.0)
    saved_prior = _provenance_number(saved, "init_num_failures", live_prior)
    live_multiplier = _provenance_number(live_provenance, "failure_counts_multiplier", 1.0)
    saved_multiplier = _provenance_number(saved, "failure_counts_multiplier", live_multiplier)
    if not all(math.isfinite(v) for v in (live_prior, saved_prior, live_multiplier, saved_multiplier)):
        return {**plan, "action": RESTORE_NEUTRAL, "reason": "saved prior/multiplier provenance is not numeric"}
    if saved_prior == live_prior and saved_multiplier == live_multiplier:
        return {**plan, "action": RESTORE_RETAIN, "reason": "saved counting semantics equal the live semantics"}
    if saved_multiplier <= 0.0 or live_multiplier <= 0.0:
        return {
            **plan,
            "action": RESTORE_NEUTRAL,
            "reason": f"failure_counts_multiplier {saved_multiplier} -> {live_multiplier} cannot be rebased exactly",
        }
    plan["prior_delta"] = live_prior - saved_prior
    plan["failure_scale"] = live_multiplier / saved_multiplier
    plan["saved_prior"] = saved_prior
    plan["live_prior"] = live_prior
    return {
        **plan,
        "action": RESTORE_REBASE,
        "reason": f"prior {saved_prior} -> {live_prior}, failure_counts_multiplier "
        f"{saved_multiplier} -> {live_multiplier}: exact counter rebase",
    }


def rebase_counts(
    episode_count: float,
    failure_count: float,
    *,
    plan: Mapping[str, Any],
) -> tuple[float, float]:
    """One bin's (episode, failure) counts moved from the saved to the live prior/multiplier."""
    if plan.get("action") != RESTORE_REBASE:
        return float(episode_count), float(failure_count)
    saved_prior = float(plan["saved_prior"])
    live_prior = float(plan["live_prior"])
    scale = float(plan["failure_scale"])
    episodes = float(episode_count) + float(plan["prior_delta"])
    observed_failures = max(0.0, float(failure_count) - saved_prior) * scale
    return max(0.0, episodes), live_prior + observed_failures


def rebase_sampler_checkpoint_counts(
    checkpoint: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """A validated sampler checkpoint with every bin's counts rebased per ``plan`` (new dict)."""
    restored = deserialize_sampler_checkpoint(checkpoint)
    if plan.get("action") != RESTORE_REBASE:
        return restored
    bins = []
    for record in restored["aggregate"]["bins"]:
        episodes, failures = rebase_counts(record["episode_count"], record["failure_count"], plan=plan)
        bins.append({**record, "episode_count": episodes, "failure_count": failures})
    restored["aggregate"]["bins"] = _merge_bin_statistics(bins)
    return restored


OBJECT_RECORD_FIELDS = ("source", "alias", "object_key", "asset_signature")
_OBJECT_RECORD_REQUIRED_FIELDS = ("source", "alias")


def normalize_object_records(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Validate one ``identity -> object record`` mapping (primitives only).

    ``identity`` is ``"<source>::<source_motion_key>"``.  A record carries the
    roster fields ``source`` and ``alias`` and, when the manifest provides
    them, ``object_key`` and ``asset_signature``.  Absent optional fields stay
    absent (nothing is fabricated); unknown fields, empty values and a
    ``source`` that does not own the identity are rejected.
    """

    if not isinstance(records, Mapping):
        raise TypeError("Object records must be a mapping keyed by motion identity")
    normalized: dict[str, dict[str, str]] = {}
    for identity, record in records.items():
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Object record identities must be nonempty strings")
        if not isinstance(record, Mapping):
            raise TypeError(f"Object record for {identity!r} must be a mapping")
        unknown = set(record) - set(OBJECT_RECORD_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown object-record fields for {identity!r}: {sorted(unknown)}"
            )
        entry: dict[str, str] = {}
        for name in OBJECT_RECORD_FIELDS:
            if name not in record:
                continue
            value = record[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Object record field {name!r} for {identity!r} must be a "
                    "nonempty string"
                )
            entry[name] = value
        missing = [name for name in _OBJECT_RECORD_REQUIRED_FIELDS if name not in entry]
        if missing:
            raise ValueError(
                f"Object record for {identity!r} lacks required fields {missing}"
            )
        if not identity.startswith(f"{entry['source']}::"):
            raise ValueError(
                f"Object record identity {identity!r} does not belong to source "
                f"{entry['source']!r}"
            )
        normalized[identity] = entry
    return normalized


def merge_object_records_by_identity(
    records_by_rank: Iterable[Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, str]]:
    """Merge per-rank ``identity -> object record`` mappings into one mapping.

    Ranks that share a motion must agree on its record exactly; differing
    values for the same identity are a configuration error.  The result is
    keyed by identity in sorted order and covers every rank's motions.
    """

    merged: dict[str, dict[str, str]] = {}
    for records in records_by_rank:
        for identity, record in normalize_object_records(records).items():
            existing = merged.get(identity)
            if existing is None:
                merged[identity] = dict(record)
            elif existing != record:
                raise ValueError(
                    "Conflicting object records for motion identity "
                    f"{identity!r}: {existing} != {record}"
                )
    return {identity: merged[identity] for identity in sorted(merged)}


__all__ = [
    "AGGREGATE_PAYLOAD_CONTRACT",
    "CHECKPOINT_CONTRACT",
    "MOTION_IDENTITY_METADATA_KEY",
    "OBJECT_RECORD_FIELDS",
    "BinKey",
    "BinStatistics",
    "deserialize_sampler_checkpoint",
    "merge_object_records_by_identity",
    "migrate_sampler_checkpoint_roster_transition",
    "normalize_object_records",
    "serialize_sampler_checkpoint",
]
