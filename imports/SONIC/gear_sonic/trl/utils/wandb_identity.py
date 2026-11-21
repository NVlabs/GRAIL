"""Persistent Weights & Biases run identity: choose the id to resume/record. No checks, no checkpoint reads
(the only checkpoint read is the mandatory restore)."""

from __future__ import annotations

from typing import Any


def normalize_wandb_run_id(value: Any) -> str | None:
    """A non-empty string id, else ``None`` (non-string values are ignored)."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def choose_wandb_run_id(*candidates: Any) -> str | None:
    """First usable id in precedence order (explicit config id, then this run's meta.yaml, then the live run)."""
    for candidate in candidates:
        run_id = normalize_wandb_run_id(candidate)
        if run_id is not None:
            return run_id
    return None
