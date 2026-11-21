"""Fail-closed deterministic action-prefix interventions for evaluation.

The helpers in this module are deliberately independent of IsaacLab so the
reviewed intervention contract can be unit-tested without launching a simulator.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_zero_latent_prefix_config(
    *,
    action_mode: str,
    prime_policy_before_loop: bool,
    motion_key: str | None,
    prefix_steps: int,
    latent_dim: int,
    expected_hand_dim: int,
) -> bool:
    """Validate an optional zero-latent launch-prefix intervention.

    Returns ``True`` when enabled. An empty motion key with zero prefix steps is
    the only disabled state, which keeps every existing evaluator unchanged.
    """

    key = str(motion_key or "")
    steps = int(prefix_steps)
    if not key and steps == 0:
        return False
    if not key or steps <= 0:
        raise ValueError(
            "zero-latent prefix requires both a non-empty motion key and positive prefix steps"
        )
    if action_mode != "deterministic_mean":
        raise ValueError("zero-latent prefix is reviewed only for deterministic_mean evaluation")
    if prime_policy_before_loop:
        raise ValueError(
            "zero-latent prefix requires prime_policy_before_loop=false for one causal inference per step"
        )
    if int(latent_dim) <= 0:
        raise ValueError("zero-latent prefix latent_dim must be positive")
    if int(expected_hand_dim) != 2:
        raise ValueError("zero-latent prefix requires exactly two untouched hand primitive outputs")
    return True


def resolve_exact_motion_id(motion_keys: Sequence[str], target_motion_key: str) -> int:
    """Resolve one exact motion key, rejecting missing or duplicate matches."""

    matches = [index for index, key in enumerate(motion_keys) if key == target_motion_key]
    if len(matches) != 1:
        raise ValueError(
            f"zero-latent prefix expected exactly one {target_motion_key!r}; found {len(matches)}"
        )
    return matches[0]


def apply_zero_latent_prefix(
    actions: torch.Tensor,
    *,
    motion_ids: torch.Tensor,
    reference_steps: torch.Tensor,
    motion_keys: Sequence[str],
    target_motion_key: str,
    prefix_steps: int,
    latent_dim: int = 64,
    expected_hand_dim: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Zero only the residual-latent slice for the reviewed launch prefix.

    The output is cloned. The final ``expected_hand_dim`` components are never
    modified. ``reference_steps`` must be the absolute motion step
    (``motion_start_time_steps + time_steps``), not the callback loop counter.
    """

    if actions.ndim != 2:
        raise ValueError(f"zero-latent prefix expects [env, action] actions; got {actions.shape}")
    if int(prefix_steps) <= 0:
        raise ValueError("zero-latent prefix_steps must be positive")
    if int(latent_dim) <= 0:
        raise ValueError("zero-latent latent_dim must be positive")
    if int(expected_hand_dim) != 2:
        raise ValueError("zero-latent prefix requires exactly two hand primitive outputs")
    expected_action_dim = int(latent_dim) + int(expected_hand_dim)
    if actions.shape[-1] != expected_action_dim:
        raise ValueError(
            "zero-latent prefix action ABI mismatch: "
            f"expected {latent_dim} latent + {expected_hand_dim} hand = "
            f"{expected_action_dim}, got {actions.shape[-1]}"
        )

    motion_ids = torch.as_tensor(motion_ids, device=actions.device).reshape(-1)
    reference_steps = torch.as_tensor(reference_steps, device=actions.device).reshape(-1)
    if motion_ids.shape[0] != actions.shape[0] or reference_steps.shape[0] != actions.shape[0]:
        raise ValueError(
            "zero-latent prefix batch mismatch: "
            f"actions={actions.shape[0]}, motion_ids={motion_ids.shape[0]}, "
            f"reference_steps={reference_steps.shape[0]}"
        )
    if torch.any(reference_steps < 0):
        raise ValueError("zero-latent prefix reference steps must be non-negative")

    target_motion_id = resolve_exact_motion_id(motion_keys, target_motion_key)
    mask = (motion_ids == target_motion_id) & (reference_steps < int(prefix_steps))
    overridden = actions.clone()
    hands_before = overridden[:, int(latent_dim) :].clone()
    overridden[mask, : int(latent_dim)] = 0
    if not torch.equal(overridden[:, int(latent_dim) :], hands_before):
        raise RuntimeError("zero-latent prefix unexpectedly modified hand primitive outputs")
    return overridden, mask, target_motion_id
