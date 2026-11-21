"""Resolution of per-motion BPS lookup stems.

RP2 packs store one BPS file per scene object named
``<motion_key>__<object_name>.npy``, where double underscores inside the raw
object name (e.g. ``documents_box__P0046``) are collapsed to a single
underscore in the filename (``documents_box_P0046``). Legacy packs store a
single ``<motion_key>.npy`` per motion. This module resolves which lookup stem
belongs to a motion's active object without importing torch or IsaacLab so the
logic stays unit-testable outside the training container.
"""

from __future__ import annotations

import os
from collections.abc import Iterable


def normalize_object_name(name: str) -> str:
    """Collapse the pack's ``__`` object-name separator to the filename form."""
    return str(name).replace("__", "_")


def resolve_object_bps_stem(
    motion_key: str,
    object_names: Iterable[str] | None,
    available_stems: Iterable[str],
) -> str | None:
    """Return the BPS lookup stem for a motion's active object.

    Tries, in order: the bare motion key (legacy single-file packs), each
    candidate object name as ``<motion_key>__<normalized_name>``, then a
    unique ``<motion_key>__`` prefix match. Returns ``None`` when nothing
    matches unambiguously — callers decide whether that is fatal.
    """
    stems = set(available_stems)
    stem = os.path.basename(motion_key)
    if stem in stems:
        return stem
    for name in object_names or []:
        candidate = f"{stem}__{normalize_object_name(name)}"
        if candidate in stems:
            return candidate
    prefix = f"{stem}__"
    matches = [s for s in stems if s.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None
