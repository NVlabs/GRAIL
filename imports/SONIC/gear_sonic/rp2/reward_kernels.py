"""Small tensor helpers shared by RP2 source-reference rewards."""

from __future__ import annotations

import torch


def tolerance_tracking_score(
    error: torch.Tensor,
    *,
    tolerance: float,
    sigma: float,
    kernel: str,
) -> torch.Tensor:
    """Map a non-negative error to a smooth score in ``[0, 1]``."""
    if tolerance < 0.0:
        raise ValueError("tracking tolerance must be non-negative")
    if sigma <= 0.0:
        raise ValueError("tracking sigma must be positive")
    effective_error = torch.clamp(error - float(tolerance), min=0.0)
    normalized_error = effective_error / float(sigma)
    if kernel == "cauchy":
        return 1.0 / (1.0 + normalized_error.square())
    if kernel == "gaussian":
        return torch.exp(-0.5 * normalized_error.square())
    raise ValueError(f"Unknown tracking kernel {kernel!r}")


def bilateral_reference_contact_gate(reference: torch.Tensor) -> torch.Tensor:
    """Return a soft either-hand gate from ``[..., left, right]`` references."""
    if reference.shape[-1] != 2:
        raise ValueError(
            "bilateral contact reference must end with left/right channels, "
            f"got {tuple(reference.shape)}"
        )
    return reference.clamp(0.0, 1.0).amax(dim=-1)


def reference_aware_support_contact_penalty(
    body_contact: torch.Tensor,
    active_support_contact: torch.Tensor,
    reference_distance: torch.Tensor,
    reference_valid: torch.Tensor,
    role_eligible: torch.Tensor,
    *,
    proximity_threshold: float,
    proximity_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize support-link contact except where the source reference expects it.

    The exemption is deliberately narrow: the active target must have support
    semantics, the same robot link must contact that target, and the source
    link-to-object relation must place the link near the target. The sigmoid
    keeps the transition into and out of a seated/support phase smooth.
    """
    if not (
        body_contact.shape
        == active_support_contact.shape
        == reference_distance.shape
        == reference_valid.shape
    ):
        raise ValueError("support contact tensors must have identical shapes")
    if role_eligible.shape != body_contact.shape[:-1]:
        raise ValueError("role_eligible must match the contact batch dimensions")
    if proximity_threshold < 0.0:
        raise ValueError("support proximity threshold must be non-negative")
    if proximity_temperature <= 0.0:
        raise ValueError("support proximity temperature must be positive")

    dtype = reference_distance.dtype
    proximity_gate = torch.sigmoid(
        (float(proximity_threshold) - reference_distance)
        / float(proximity_temperature)
    )
    exemption = (
        role_eligible[..., None].to(dtype)
        * reference_valid.to(dtype)
        * active_support_contact.to(dtype)
        * proximity_gate
    )
    penalty = body_contact.to(dtype) * (1.0 - exemption)
    return penalty.sum(dim=-1), exemption


def update_bilateral_contact_failure_counter(
    reference: torch.Tensor,
    simulated: torch.Tensor,
    counter: torch.Tensor,
    *,
    reference_threshold: float,
    simulated_threshold: float,
    grace_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance per-hand missing-contact counters and return env termination flags."""
    if reference.shape != simulated.shape or reference.shape != counter.shape:
        raise ValueError(
            "reference, simulated, and counter tensors must have identical shapes"
        )
    if reference.shape[-1] != 2:
        raise ValueError("contact failure tracking requires left/right channels")
    if grace_steps < 0:
        raise ValueError("contact failure grace_steps must be non-negative")
    required = reference >= float(reference_threshold)
    satisfied = simulated >= float(simulated_threshold)
    missing = required & ~satisfied
    next_counter = torch.where(missing, counter + 1, torch.zeros_like(counter))
    terminate = (next_counter > int(grace_steps)).any(dim=-1)
    return next_counter, terminate, required
