"""Per-clip provenance of native binary hand-contact labels (pure Python, no torch).

Decides whether a clip's legacy/native binary hand label (``object_contact_mask_*`` / ``object_in_contact_*``) is
AUTHORITATIVE, i.e. a valid 0 means "known no contact" even where the canonical three-state conversion recorded a
neutral 0. Today that is true for OmniContact packs, whose wrist labels come from the native ``contact_info``
channels. Recognised from the pack's real fields first (robot or object record), then from an explicit flag, then
from the documented motion-key prefix fallback, which applies ONLY when no trustworthy source field exists (an
explicit FORM/teleop/other source under an ``omni*`` alias stays non-authoritative). No file is scanned, nothing is
written, nothing is gated.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Real producer fields: joint_training_source_id == "omnicontact", the raw
# omnicontact_contact_info channel, object_contact_mask_provenance naming the native channel.
NATIVE_SOURCE_IDS = frozenset({"omnicontact", "omni_contact", "omni"})
NATIVE_PROVENANCE_TOKENS = ("contact_info", "omnicontact", "native")
EXPLICIT_FLAG_KEY = "object_contact_mask_native_authoritative"
DEFAULT_KEY_PREFIXES = ("omni",)


def native_contact_label_authority(
    motion_key: str,
    *records: Mapping[str, Any] | None,
    key_prefixes: tuple[str, ...] | list[str] | None = None,
) -> tuple[bool, str]:
    """Return ``(authoritative, reason)`` for one clip.

    ``records`` are the clip's robot and object dictionaries (any may be None). Precedence:
      1. explicit boolean ``object_contact_mask_native_authoritative`` in any record;
      2. ``joint_training_source_id`` in NATIVE_SOURCE_IDS, or an ``omnicontact_contact_info`` field, or an
         ``object_contact_mask_provenance`` / ``contact_label_source`` string naming the native channel;
      3. an explicit NON-native source (``joint_training_source_id`` such as ``form``/``form_hoi``/``teleop``, or a
         provenance string that does not name the native channel) is final: ``(False, reason)``; the motion-key
         prefix is NOT consulted, so a FORM clip under an ``omni*`` alias keeps canonical semantics;
      4. only when no trustworthy source field exists at all: the motion key starts with one of ``key_prefixes``
         (default ``omni``), the documented fallback for renamed aliases that lost their fields.
    """
    for record in records:
        if isinstance(record, Mapping) and EXPLICIT_FLAG_KEY in record:
            value = record[EXPLICIT_FLAG_KEY]
            if hasattr(value, "item"):
                value = value.item()
            return bool(value), "explicit_flag"
    known_non_native: str | None = None
    for record in records:
        if not isinstance(record, Mapping):
            continue
        source_id = str(record.get("joint_training_source_id", "") or "").strip().lower()
        if source_id in NATIVE_SOURCE_IDS:
            return True, f"joint_training_source_id={source_id}"
        if "omnicontact_contact_info" in record:
            return True, "omnicontact_contact_info"
        for field in ("object_contact_mask_provenance", "contact_label_source"):
            text = record.get(field)
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8", "replace")
            if isinstance(text, str) and text.strip():
                if any(token in text.lower() for token in NATIVE_PROVENANCE_TOKENS):
                    return True, f"{field}={text}"
                if known_non_native is None:
                    known_non_native = f"{field}={text}"
        if source_id and known_non_native is None:
            known_non_native = f"joint_training_source_id={source_id}"
        elif source_id:
            # An explicit source id is the most specific statement; prefer it in the reason.
            known_non_native = f"joint_training_source_id={source_id}"
    if known_non_native is not None:
        # Trustworthy source present and not native: final answer, no alias/prefix guessing.
        return False, known_non_native
    prefixes = tuple(str(p).lower() for p in (key_prefixes or DEFAULT_KEY_PREFIXES))
    if str(motion_key).lower().startswith(prefixes):
        return True, "motion_key_prefix"
    return False, "none"
