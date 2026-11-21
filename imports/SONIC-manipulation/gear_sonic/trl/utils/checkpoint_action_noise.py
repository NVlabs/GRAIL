"""Opt-in action-noise reset for checkpoint weight warm starts.

The normal checkpoint path preserves the learned action standard deviation.
Adaptation runs can instead request a fresh exploration scale after loading
policy weights. Full resume paths preserve the run-local learned scale, so the
reset is applied exactly once before ordinary autoresume takes over.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _config_get(config, key: str, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def maybe_reset_action_noise_after_checkpoint_load(
    policy,
    algo_config,
    *,
    resume: bool,
    warm_resume: bool,
) -> dict | None:
    """Reset ``std``/``log_std`` after a weights-only checkpoint load.

    This is disabled unless ``reset_noise_std_on_checkpoint_load`` is true.
    ``freeze_noise_std`` controls whether the reset value remains fixed; its
    default is false, matching GRAIL-SONIC's trainable exploration parameter.
    Full resume paths skip the reset because their checkpoint already belongs
    to the destination run and includes the intended optimizer/noise state.
    """

    if not bool(_config_get(algo_config, "reset_noise_std_on_checkpoint_load", False)):
        return None
    if resume or warm_resume:
        return None

    init_noise_std = float(_config_get(algo_config, "init_noise_std"))
    if not math.isfinite(init_noise_std) or init_noise_std <= 0.0:
        raise ValueError(f"init_noise_std must be finite and positive, got {init_noise_std}")

    if hasattr(policy, "log_std"):
        parameter = policy.log_std
        fill_value = math.log(init_noise_std)
        parameterization = "log_std"
    elif hasattr(policy, "std"):
        parameter = policy.std
        fill_value = init_noise_std
        parameterization = "std"
    else:
        raise AttributeError("policy has neither std nor log_std action-noise parameter")

    freeze_noise_std = bool(_config_get(algo_config, "freeze_noise_std", False))
    with torch.no_grad():
        parameter.fill_(fill_value)
    parameter.requires_grad_(not freeze_noise_std)

    return {
        "parameterization": parameterization,
        "init_noise_std": init_noise_std,
        "frozen": freeze_noise_std,
    }
