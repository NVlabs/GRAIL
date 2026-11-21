"""In-memory record of checkpoint files completed at a training step (rank 0 of the live process only).

ModelSaveCallback (periodic ``model_step_*.pt`` / ``last.pt``) and TimeBudgetCallback (boundary ``last.pt``) can both be
due at the same step. The second writer reuses what the first completed: the same checkpoint is serialized once, a
second name is published as a hard link / copy of the completed file. Nothing is read back, hashed or certified; the
record lives only in this process and is consulted only for the current step. stdlib only.
"""
from __future__ import annotations

import os

_COMPLETED: dict[int, dict[str, str]] = {}


def record(step: int, kind: str, path: str) -> None:
    """Remember that ``path`` (kind 'last' or 'regular') was completed at ``step``."""
    _COMPLETED.setdefault(int(step), {})[str(kind)] = str(path)
    for old in [s for s in _COMPLETED if s < int(step)]:  # only the current step matters
        _COMPLETED.pop(old, None)


def completed(step: int) -> dict[str, str]:
    """{'last': path, 'regular': path} completed so far at ``step`` (copy; empty when nothing was written)."""
    return dict(_COMPLETED.get(int(step), {}))


def same_file_name(path_a: str, path_b: str) -> bool:
    """Two spellings of the same file name (realpath), without touching the file contents."""
    try:
        return os.path.realpath(str(path_a)) == os.path.realpath(str(path_b))
    except OSError:
        return str(path_a) == str(path_b)
