"""Deterministic helpers for opt-in stochastic policy evaluation.

The normal imitation-policy evaluator executes the action mean.  These helpers
support a diagnostic A/B where one or more environments keep that mean while
the remaining environments draw from the same action distribution using an
independent, explicitly seeded generator per environment.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


DETERMINISTIC_MEAN = "deterministic_mean"
MEAN_VS_SEEDED_SAMPLE = "mean_vs_seeded_sample"
VALID_EVAL_ACTION_MODES = frozenset({DETERMINISTIC_MEAN, MEAN_VS_SEEDED_SAMPLE})


def validate_eval_action_sampling_config(
    *,
    mode: str,
    num_envs: int,
    deterministic_env_count: int,
    log_steps: int,
) -> None:
    """Fail closed on an invalid evaluator action-sampling contract."""

    if mode not in VALID_EVAL_ACTION_MODES:
        raise ValueError(
            f"unsupported eval action mode {mode!r}; expected one of "
            f"{sorted(VALID_EVAL_ACTION_MODES)}"
        )
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if log_steps < 0:
        raise ValueError("action log_steps must be non-negative")
    if mode == DETERMINISTIC_MEAN:
        return
    if not 1 <= deterministic_env_count < num_envs:
        raise ValueError(
            "mean_vs_seeded_sample requires at least one deterministic mean env "
            "and at least one sampled env"
        )
    if log_steps <= 0:
        raise ValueError("mean_vs_seeded_sample requires positive action log_steps")


def per_env_sample_seeds(
    *, base_seed: int, num_envs: int, deterministic_env_count: int
) -> list[int | None]:
    """Return the reviewed seed assigned to each environment.

    Deterministic environments do not consume a generator and therefore carry
    ``None``.  Sampled environment ``i`` uses ``base_seed + i`` so adding more
    environments never changes an existing environment's random stream.
    """

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not 0 <= deterministic_env_count <= num_envs:
        raise ValueError("deterministic_env_count is outside [0, num_envs]")
    return [
        None if env_id < deterministic_env_count else base_seed + env_id
        for env_id in range(num_envs)
    ]


def make_per_env_generators(
    seeds: Sequence[int | None], *, device: torch.device | str
) -> list[torch.Generator | None]:
    """Create independent generators without modifying PyTorch's global RNG."""

    generators: list[torch.Generator | None] = []
    for seed in seeds:
        if seed is None:
            generators.append(None)
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)
    return generators


def mean_vs_seeded_sample(
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    *,
    deterministic_env_count: int,
    generators: Sequence[torch.Generator | None],
) -> torch.Tensor:
    """Keep leading environments at the mean and seed-sample the remainder."""

    if action_mean.ndim != 2:
        raise ValueError(f"action_mean must be [num_envs, action_dim], got {action_mean.shape}")
    try:
        expanded_std = torch.broadcast_to(action_std, action_mean.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"action_std shape {tuple(action_std.shape)} does not broadcast to "
            f"action_mean shape {tuple(action_mean.shape)}"
        ) from exc
    num_envs = action_mean.shape[0]
    if len(generators) != num_envs:
        raise ValueError(f"expected {num_envs} per-env generators, got {len(generators)}")
    if not 1 <= deterministic_env_count < num_envs:
        raise ValueError("diagnostic requires both deterministic and sampled environments")

    actions = action_mean.clone()
    for env_id in range(deterministic_env_count, num_envs):
        generator = generators[env_id]
        if generator is None:
            raise ValueError(f"sampled env {env_id} has no generator")
        noise = torch.randn(
            action_mean[env_id].shape,
            dtype=action_mean.dtype,
            device=action_mean.device,
            generator=generator,
        )
        actions[env_id] = action_mean[env_id] + expanded_std[env_id] * noise
    return actions
