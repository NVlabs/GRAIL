"""Pure-torch pieces of the PPO update hot path that replaced check-only or gather-heavy work (no runtime
work PPO does not need). Semantics preserved where the computation is required; tested in isolation."""
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch


def normalize_advantages_distributed(advantages: torch.Tensor, reduce_sum: Callable[[torch.Tensor], torch.Tensor], *,
                                     eps: float = 1e-8) -> torch.Tensor:
    """Global (all-rank) advantage normalization over dims (0, 1) without gathering the tensor.

    Equal to ``(a - gathered.mean(dim=(0, 1))) / (gathered.std(dim=(0, 1)) + eps)`` where ``gathered`` concatenates every
    rank's advantages: the per-rank sum, sum of squares and count are accumulated in float64 and summed across ranks by
    ``reduce_sum`` (``accelerator.reduce(x, reduction="sum")``); the variance is the unbiased one torch.std uses.
    """
    adv = advantages.double()
    n_local = float(adv.shape[0] * adv.shape[1])
    s = adv.sum(dim=(0, 1))
    local = torch.stack([s, (adv * adv).sum(dim=(0, 1)), torch.full_like(s, n_local)])
    total = reduce_sum(local)
    count = total[2]
    mean = total[0] / count
    var = (total[1] - count * mean * mean) / (count - 1.0).clamp_min(1.0)
    std = var.clamp_min(0.0).sqrt()
    return ((adv - mean) / (std + eps)).to(advantages.dtype)


def nonfinite_update_skip(grad_norm, grads: list[torch.Tensor]) -> bool:
    """True when this update must be skipped: the clipping result is not finite (one host sync), or - without clipping -
    any gradient holds a non-finite value (one fused host sync). The only remaining gradient check of the optimizer step."""
    if grad_norm is not None:
        return not bool(torch.isfinite(torch.as_tensor(grad_norm, dtype=torch.float64)).all().item())
    if not grads:
        return False
    return not bool(torch.all(torch.stack([torch.isfinite(g).all() for g in grads])).item())


def packed_means(stats: Mapping[str, torch.Tensor], gather_fn: Callable[[torch.Tensor], torch.Tensor],
                 var_of: tuple[str, ...] = ()) -> dict[str, float]:
    """Means (and, for ``var_of``, unbiased variances) of per-rank statistic tensors across ranks in ONE gather.

    Each rank contributes its local mean (and for ``var_of`` its local sum / sum of squares / count); the packed vector
    is gathered once (``gather_fn``, e.g. ``accelerator.gather_for_metrics``), moved to the host once, and the per-metric
    mean over ranks is returned (equal counts per rank: identical to the mean of the gathered tensor). The variance of a
    ``var_of`` statistic is the unbiased variance over all ranks' elements, as ``gathered.var()`` would return.
    """
    names = list(stats)
    parts = [stats[name].detach().double().mean().reshape(1) for name in names]
    for name in var_of:
        t = stats[name].detach().double()
        parts.append(torch.stack([t.sum(), (t * t).sum(), torch.tensor(float(t.numel()), dtype=torch.float64, device=t.device)]))
    packed = torch.cat(parts)
    gathered = gather_fn(packed).detach().double().cpu().reshape(-1, packed.numel())
    out: dict[str, float] = {}
    cursor = 0
    for name in names:
        out[name] = float(gathered[:, cursor].mean()); cursor += 1
    for name in var_of:
        sums = gathered[:, cursor:cursor + 3].sum(dim=0); cursor += 3
        count = sums[2]
        mean = sums[0] / count
        var = (sums[1] - count * mean * mean) / max(float(count - 1.0), 1.0)
        out[f"{name}__var"] = float(max(var, 0.0))
    return out
