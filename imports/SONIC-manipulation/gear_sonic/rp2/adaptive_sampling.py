"""Pure helpers for bounded multi-source curriculum sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def beta_capped_difficulty_distribution(
    success_rate: Sequence[float],
    previous_success_rate: Sequence[float],
    episode_count: Sequence[float],
    *,
    beta: float = 3.0,
    soft_success_floor: float = 0.10,
    minimum_weight: float = 0.05,
    progress_weight: float = 1.0,
    uncertainty_weight: float = 0.25,
    warmup_completed_episodes: float = 100.0,
) -> np.ndarray:
    """Build a capped per-motion or per-time-bin sampling distribution.

    This keeps the useful inverse-difficulty and beta-cap behavior from the
    InterMimic AMASS sampler, while down-weighting persistently zero-success
    items that are more likely invalid or currently unlearnable than useful
    frontier examples.
    """
    success = np.asarray(success_rate, dtype=np.float64)
    previous = np.asarray(previous_success_rate, dtype=np.float64)
    episodes = np.asarray(episode_count, dtype=np.float64)
    if success.ndim != 1 or success.shape != previous.shape or success.shape != episodes.shape:
        raise ValueError(
            "success_rate, previous_success_rate, and episode_count must be "
            "equal-length vectors"
        )
    if success.size == 0:
        raise ValueError("At least one adaptive-sampling item is required")
    if np.any(~np.isfinite(success)) or np.any((success < 0.0) | (success > 1.0)):
        raise ValueError("success_rate must be finite and within [0, 1]")
    if np.any(~np.isfinite(previous)) or np.any(
        (previous < 0.0) | (previous > 1.0)
    ):
        raise ValueError("previous_success_rate must be finite and within [0, 1]")
    if np.any(~np.isfinite(episodes)) or np.any(episodes < 0.0):
        raise ValueError("episode_count must be finite and non-negative")
    if beta < 1.0:
        raise ValueError("beta must be at least one")
    if not 0.0 < soft_success_floor <= 1.0:
        raise ValueError("soft_success_floor must be within (0, 1]")
    if minimum_weight <= 0.0:
        raise ValueError("minimum_weight must be positive")
    if warmup_completed_episodes < 0.0:
        raise ValueError("warmup_completed_episodes must be non-negative")

    if episodes.sum() < float(warmup_completed_episodes):
        return np.full(success.shape, 1.0 / success.size, dtype=np.float64)

    inverse_difficulty = 1.0 - success
    learnability = np.minimum(success / float(soft_success_floor), 1.0)
    progress = np.abs(success - previous)
    uncertainty = 1.0 / np.sqrt(episodes + 1.0)
    raw = (
        float(minimum_weight)
        + inverse_difficulty * learnability
        + float(progress_weight) * progress
        + float(uncertainty_weight) * uncertainty
    )
    capped = np.minimum(raw, float(beta) * raw.mean())
    probability_cap = min(1.0, float(beta) / success.size)
    return bounded_probability_distribution(
        capped,
        np.zeros(success.shape, dtype=np.float64),
        np.full(success.shape, probability_cap, dtype=np.float64),
    )


def learning_progress_priority(
    success_rate: Sequence[float],
    previous_success_rate: Sequence[float],
    episode_count: Sequence[float],
    *,
    frontier_weight: float = 1.0,
    progress_weight: float = 1.0,
    uncertainty_weight: float = 0.25,
    uncertainty_prior_episodes: float = 100.0,
    minimum_priority: float = 0.05,
) -> np.ndarray:
    """Score sources without letting mastered or impossible data dominate.

    The frontier term peaks at 50% success. Absolute learning progress favors
    sources that are still improving, while uncertainty gives new or
    under-sampled sources a temporary boost. ``minimum_priority`` maintains
    coverage after the bounds are applied.
    """
    success = np.asarray(success_rate, dtype=np.float64)
    previous = np.asarray(previous_success_rate, dtype=np.float64)
    episodes = np.asarray(episode_count, dtype=np.float64)
    if success.shape != previous.shape or success.shape != episodes.shape:
        raise ValueError(
            "success_rate, previous_success_rate, and episode_count must have "
            f"identical shapes; got {success.shape}, {previous.shape}, {episodes.shape}"
        )
    if np.any(~np.isfinite(success)) or np.any((success < 0.0) | (success > 1.0)):
        raise ValueError("success_rate must be finite and within [0, 1]")
    if np.any(~np.isfinite(previous)) or np.any(
        (previous < 0.0) | (previous > 1.0)
    ):
        raise ValueError("previous_success_rate must be finite and within [0, 1]")
    if np.any(~np.isfinite(episodes)) or np.any(episodes < 0.0):
        raise ValueError("episode_count must be finite and non-negative")
    if uncertainty_prior_episodes <= 0.0:
        raise ValueError("uncertainty_prior_episodes must be positive")
    if minimum_priority <= 0.0:
        raise ValueError("minimum_priority must be positive")

    frontier = 4.0 * success * (1.0 - success)
    progress = np.abs(success - previous)
    uncertainty = uncertainty_prior_episodes / (
        uncertainty_prior_episodes + episodes
    )
    return (
        minimum_priority
        + float(frontier_weight) * frontier
        + float(progress_weight) * progress
        + float(uncertainty_weight) * uncertainty
    )


def bounded_probability_distribution(
    scores: Sequence[float],
    minimum_probabilities: Sequence[float],
    maximum_probabilities: Sequence[float],
) -> np.ndarray:
    """Project positive source scores onto per-source probability bounds."""
    score = np.asarray(scores, dtype=np.float64)
    lower = np.asarray(minimum_probabilities, dtype=np.float64)
    upper = np.asarray(maximum_probabilities, dtype=np.float64)
    if score.ndim != 1 or lower.shape != score.shape or upper.shape != score.shape:
        raise ValueError("scores and probability bounds must be equal-length vectors")
    if np.any(~np.isfinite(score)) or np.any(score < 0.0):
        raise ValueError("scores must be finite and non-negative")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("probability bounds must be finite")
    if np.any(lower < 0.0) or np.any(upper > 1.0) or np.any(lower > upper):
        raise ValueError("probability bounds must satisfy 0 <= lower <= upper <= 1")
    if lower.sum() > 1.0 + 1.0e-10 or upper.sum() < 1.0 - 1.0e-10:
        raise ValueError(
            "Infeasible probability bounds: lower bounds exceed one or upper "
            "bounds cannot sum to one"
        )

    probability = lower.copy()
    capacity = upper - probability
    remaining = 1.0 - probability.sum()
    tolerance = 1.0e-12
    while remaining > tolerance:
        active = capacity > tolerance
        if not np.any(active):
            raise ValueError("Probability bounds exhausted before reaching sum one")
        active_score = np.where(active, score, 0.0)
        if active_score.sum() <= tolerance:
            active_score = active.astype(np.float64)
        proposal = remaining * active_score / active_score.sum()
        increment = np.minimum(proposal, capacity)
        assigned = increment.sum()
        if assigned <= tolerance:
            raise ValueError("Could not make progress while projecting probabilities")
        probability += increment
        capacity -= increment
        remaining -= assigned

    probability /= probability.sum()
    return probability


def allocate_bounded_slots(
    probabilities: Sequence[float],
    num_slots: int,
    minimum_slots: Sequence[int],
    maximum_slots: Sequence[int],
) -> np.ndarray:
    """Convert probabilities to integer environment slots under hard bounds."""
    probability = np.asarray(probabilities, dtype=np.float64)
    lower = np.asarray(minimum_slots, dtype=np.int64)
    upper = np.asarray(maximum_slots, dtype=np.int64)
    if probability.ndim != 1 or lower.shape != probability.shape or upper.shape != probability.shape:
        raise ValueError("probabilities and slot bounds must be equal-length vectors")
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")
    if np.any(lower < 0) or np.any(lower > upper):
        raise ValueError("slot bounds must satisfy 0 <= lower <= upper")
    if lower.sum() > num_slots or upper.sum() < num_slots:
        raise ValueError("Infeasible slot bounds for requested num_slots")
    if np.any(~np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if probability.sum() <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    probability = probability / probability.sum()

    slots = lower.copy()
    target = probability * float(num_slots)
    remaining = int(num_slots - slots.sum())
    for _ in range(remaining):
        eligible = slots < upper
        if not np.any(eligible):
            raise ValueError("Slot bounds exhausted before allocation completed")
        deficit = target - slots
        deficit[~eligible] = -np.inf
        slots[int(np.argmax(deficit))] += 1
    return slots


def build_source_sampling_plan(
    source_specs: Mapping[str, Mapping[str, float | int]],
    source_stats: Mapping[str, Mapping[str, float]],
    *,
    num_slots: int,
) -> dict[str, object]:
    """Build a bounded, adaptive source allocation for a joint-training leg."""
    names = tuple(source_specs)
    if not names:
        raise ValueError("At least one source is required")

    success = []
    previous = []
    episodes = []
    lower_probability = []
    upper_probability = []
    lower_slots = []
    upper_slots = []
    for name in names:
        spec = source_specs[name]
        stats = source_stats.get(name, {})
        success.append(float(stats.get("success_rate", 0.5)))
        previous.append(
            float(stats.get("previous_success_rate", stats.get("success_rate", 0.5)))
        )
        episodes.append(float(stats.get("episode_count", 0.0)))
        lower_probability.append(float(spec.get("minimum_probability", 0.0)))
        upper_probability.append(float(spec.get("maximum_probability", 1.0)))
        lower_slots.append(int(spec.get("minimum_slots", 1)))
        upper_slots.append(int(spec.get("maximum_slots", num_slots)))

    priority = learning_progress_priority(success, previous, episodes)
    probability = bounded_probability_distribution(
        priority,
        lower_probability,
        upper_probability,
    )
    slots = allocate_bounded_slots(
        probability,
        num_slots,
        lower_slots,
        upper_slots,
    )
    return {
        "num_slots": int(num_slots),
        "sources": {
            name: {
                "priority": float(priority[index]),
                "probability": float(probability[index]),
                "slots": int(slots[index]),
                "success_rate": float(success[index]),
                "previous_success_rate": float(previous[index]),
                "episode_count": float(episodes[index]),
            }
            for index, name in enumerate(names)
        },
    }
