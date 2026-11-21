"""Compile visually reviewed per-hand intervals into fixed material contacts.

This module does not infer contact from RGB, object motion, or current geometry.
An agent or person inspects the source video and authors the review before a
Stage 2 solve. Each hand covers the complete source timeline explicitly;
``unknown`` records uncertainty and contributes no contact attraction.
"""

from __future__ import annotations

import hashlib
import json
from urllib.parse import quote

SCHEMA = "grail.contact_review.v2"
LEGACY_SCHEMA = "grail.contact_review.v1"
_TOP_FIELDS = {"schema", "binding", "reviewed_by", "hands"}
_INTERVAL_FIELDS = {
    "id",
    "frames",
    "state",
    "description",
    "evidence_frames",
    "contacts",
}
_PAIR_FIELDS = {
    "id",
    "human_vertex_id",
    "confidence",
    "hand_surface",
    "object_surface",
}
_HANDS = {"left_hand": "L_Hand", "right_hand": "R_Hand"}


def fixed_schema_for_review(review):
    """Preserve the authored representation; never snap old anchors to vertices."""
    from grail.optimization.fixed_contact import LEGACY_SCHEMA as legacy
    from grail.optimization.fixed_contact import SCHEMA as current

    schema = review.get("schema") if isinstance(review, dict) else None
    if schema == SCHEMA:
        return current
    if schema == LEGACY_SCHEMA:
        return legacy
    raise ValueError(f"Contact review schema must be {SCHEMA} or {LEGACY_SCHEMA}")


def compile_review(review, binding):
    """Validate a complete review and return detached fixed-contact records.

    All frame intervals are zero-based and end-exclusive. Fixed point identity
    and activation are derived from each containing interval; the caller must
    still validate the selected original vertices against actual geometry.
    Changing a hand's material contact requires a new interval. The returned
    list is empty for a fully reviewed no-contact/unknown sequence.
    """
    # Reuse the shared source/model binding contract. The local import allows
    # FixedContactLoss to validate embedded reviews without an import cycle.
    from grail.optimization.fixed_contact import (
        _fields,
        _integer,
        _number,
        _text,
        _validate_binding,
        anchor_fields,
        validate_anchor,
    )

    _validate_binding(binding)
    _fields(review, _TOP_FIELDS, "contact review")
    schema = fixed_schema_for_review(review)
    pair_fields = _PAIR_FIELDS | anchor_fields(schema)
    _validate_binding(review["binding"])
    if review["binding"] != binding:
        raise ValueError(
            "Contact review binding differs from the current sequence/model/timeline/geometry"
        )
    _text(review["reviewed_by"], "reviewed_by")
    _fields(review["hands"], set(_HANDS), "hands")
    compiled = []
    for hand, body_part in _HANDS.items():
        intervals = review["hands"][hand]
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"{hand} must contain a nonempty list of intervals")
        next_frame, interval_ids = 0, set()
        for index, interval in enumerate(intervals):
            label = f"{hand}[{index}]"
            _fields(interval, _INTERVAL_FIELDS, label)
            interval_id = _text(interval["id"], f"{label}.id")
            if interval_id in interval_ids:
                raise ValueError(f"Duplicate interval id in {hand}: {interval_id}")
            interval_ids.add(interval_id)
            frames = interval["frames"]
            if not isinstance(frames, list) or len(frames) != 2:
                raise ValueError(f"{label}.frames must be [start, end_exclusive]")
            start, end = [_integer(value, f"{label}.frames") for value in frames]
            if not 0 <= start < end <= binding["frame_count"]:
                raise ValueError(f"{label}.frames must be nonempty and inside the source timeline")
            if start != next_frame:
                raise ValueError(
                    f"{hand} intervals must cover the timeline in order without gaps or overlaps"
                )
            next_frame = end
            state = interval["state"]
            if state not in ("contact", "no_contact", "unknown"):
                raise ValueError(f"{label}.state must be contact, no_contact, or unknown")
            _text(interval["description"], f"{label}.description")
            evidence = interval["evidence_frames"]
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(f"{label}.evidence_frames must be a nonempty list")
            inspected = set()
            for value in evidence:
                frame = _integer(value, f"{label}.evidence_frames")
                if not start <= frame < end:
                    raise ValueError(f"{label}.evidence_frames must be inside its interval")
                if frame in inspected:
                    raise ValueError(f"{label}.evidence_frames must be unique")
                inspected.add(frame)
            pairs = interval["contacts"]
            if not isinstance(pairs, list):
                raise ValueError(f"{label}.contacts must be a list")
            if state != "contact":
                if pairs:
                    raise ValueError(f"{label}: {state} intervals must have no contacts")
                continue
            pair_ids, positive_confidence = set(), False
            for pair_index, pair in enumerate(pairs):
                pair_label = f"{label}.contacts[{pair_index}]"
                _fields(pair, pair_fields, pair_label)
                pair_id = _text(pair["id"], f"{pair_label}.id")
                if pair_id in pair_ids:
                    raise ValueError(f"Duplicate contact id in {label}: {pair_id}")
                pair_ids.add(pair_id)
                for field in ("hand_surface", "object_surface"):
                    _text(pair[field], f"{pair_label}.{field}")
                human_id = _integer(pair["human_vertex_id"], f"{pair_label}.human_vertex_id")
                if human_id >= binding["human_vertex_count"]:
                    raise ValueError(f"{pair_label}: human vertex is out of range")
                anchor = validate_anchor(pair, binding, schema, pair_label)
                confidence = _number(pair["confidence"], f"{pair_label}.confidence")
                if not 0 <= confidence <= 1:
                    raise ValueError(f"{pair_label}.confidence must be between zero and one")
                positive_confidence |= confidence > 0
                # Escape delimiters as well as '%' to avoid ambiguous compound
                # IDs when author-provided IDs themselves contain colons.
                compiled.append(
                    {
                        "id": f"{hand}:{quote(interval_id, safe='')}:{quote(pair_id, safe='')}",
                        "body_part": body_part,
                        "human_vertex_id": human_id,
                        **anchor,
                        "active_frames": [start, end],
                        "confidence": confidence,
                    }
                )
            if not positive_confidence:
                raise ValueError(
                    f"{label}: contact intervals need at least one positive-confidence contact"
                )
        if next_frame != binding["frame_count"]:
            raise ValueError(f"{hand} intervals must cover the complete source timeline")
    return compiled


def review_sha256(review):
    """Hash review content canonically; call compile_review to validate it first."""
    payload = json.dumps(
        review, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
