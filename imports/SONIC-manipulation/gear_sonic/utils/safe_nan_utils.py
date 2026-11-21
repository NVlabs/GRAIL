"""Utility functions for detecting and recovering from NaN/Inf in RL environments.

Used by manager_env_wrapper when safe_nan=True to detect invalid observations,
sanitize them, and merge reset observations for force-reset environments.
"""

from __future__ import annotations

import torch


def tensor_bad_env_ids(tensor: torch.Tensor, abs_limit: float | None = None) -> list[int]:
    """Find environment indices where the tensor contains NaN, Inf, or values exceeding abs_limit.

    Args:
        tensor: Batched tensor with shape (num_envs, ...).
        abs_limit: Optional absolute value threshold. Finite values exceeding this are treated as invalid.

    Returns:
        Sorted list of environment indices with invalid values.
    """
    if not isinstance(tensor, torch.Tensor):
        return []
    flat = tensor.detach().reshape(tensor.shape[0], -1).float()
    bad_mask = torch.isnan(flat) | torch.isinf(flat)
    if abs_limit is not None:
        bad_mask = bad_mask | (torch.isfinite(flat) & (torch.abs(flat) > abs_limit))
    if not bad_mask.any():
        return []
    return bad_mask.any(dim=1).nonzero(as_tuple=False).flatten().detach().cpu().tolist()


def collect_bad_env_ids_from_obs(
    obs: dict, abs_limit: float | None = None
) -> list[int]:
    """Scan an observation dict (possibly nested) for environments with invalid values.

    Args:
        obs: Observation dictionary, possibly with nested dicts per group.
        abs_limit: Optional absolute value threshold.

    Returns:
        Sorted list of environment indices with invalid observations.
    """
    bad_env_ids: set[int] = set()
    for group_value in obs.values():
        if isinstance(group_value, dict):
            for term_value in group_value.values():
                bad_env_ids.update(tensor_bad_env_ids(term_value, abs_limit=abs_limit))
        else:
            bad_env_ids.update(tensor_bad_env_ids(group_value, abs_limit=abs_limit))
    return sorted(bad_env_ids)


def merge_reset_obs(dst_obs, src_obs, env_ids: torch.Tensor):
    """Merge reset observations into the main observation dict for specific environments.

    Recursively handles nested dicts. For tensor leaves, copies values from src to dst
    at the given env_ids.

    Args:
        dst_obs: Destination observation dict (modified in-place).
        src_obs: Source observation dict from reset.
        env_ids: Tensor of environment indices to merge.
    """
    if isinstance(dst_obs, dict) and isinstance(src_obs, dict):
        for key in dst_obs.keys():
            if key in src_obs:
                merge_reset_obs(dst_obs[key], src_obs[key], env_ids)
        return

    if not isinstance(dst_obs, torch.Tensor) or not isinstance(src_obs, torch.Tensor):
        return

    if src_obs.shape[0] == dst_obs.shape[0]:
        dst_obs[env_ids] = src_obs[env_ids]
    elif src_obs.shape[0] == env_ids.shape[0]:
        dst_obs[env_ids] = src_obs


def sanitize_obs_for_env_ids(
    obs: dict, env_ids: torch.Tensor, abs_max: float = 1.0e6
):
    """Replace NaN/Inf values with zeros and clamp to abs_max for specific environments.

    Args:
        obs: Observation dictionary (modified in-place).
        env_ids: Tensor of environment indices to sanitize.
        abs_max: Clamp threshold for sanitized values.
    """
    for group_value in obs.values():
        if isinstance(group_value, dict):
            for term_name, term_value in group_value.items():
                if isinstance(term_value, torch.Tensor) and torch.is_floating_point(term_value):
                    sanitized = torch.nan_to_num(
                        term_value[env_ids], nan=0.0, posinf=0.0, neginf=0.0
                    )
                    group_value[term_name][env_ids] = torch.clamp(
                        sanitized, min=-abs_max, max=abs_max
                    )
        elif isinstance(group_value, torch.Tensor) and torch.is_floating_point(group_value):
            sanitized = torch.nan_to_num(
                group_value[env_ids], nan=0.0, posinf=0.0, neginf=0.0
            )
            group_value[env_ids] = torch.clamp(sanitized, min=-abs_max, max=abs_max)
