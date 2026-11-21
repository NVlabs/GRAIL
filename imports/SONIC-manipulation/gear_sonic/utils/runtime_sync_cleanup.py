"""Opt-in helpers for removing avoidable training-loop host synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class RuntimeSyncCleanupOptions:
    """Resolved wrapper options; the absent/default configuration is a no-op."""

    enabled: bool = False
    omit_unused_env_actions_cpu_copy: bool = False
    batch_joint_source_metrics: bool = False
    sync_free_contact_cache_invalidation: bool = False
    benchmark_log_interval_steps: int = 0

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None
    ) -> "RuntimeSyncCleanupOptions":
        if not config or not bool(config.get("enabled", False)):
            return cls()
        interval = int(config.get("benchmark_log_interval_steps", 0) or 0)
        if interval < 0:
            raise ValueError(
                "runtime_sync_cleanup.benchmark_log_interval_steps must be >= 0"
            )
        return cls(
            enabled=True,
            omit_unused_env_actions_cpu_copy=bool(
                config.get("omit_unused_env_actions_cpu_copy", True)
            ),
            batch_joint_source_metrics=bool(
                config.get("batch_joint_source_metrics", True)
            ),
            sync_free_contact_cache_invalidation=bool(
                config.get(
                    "sync_free_contact_cache_invalidation",
                    False,
                )
            ),
            benchmark_log_interval_steps=interval,
        )


def joint_source_metric_snapshot(
    num_episodes: torch.Tensor,
    num_failures: torch.Tensor,
    previous_success_rate: torch.Tensor,
) -> torch.Tensor:
    """Snapshot episode, failure, and success metrics without a host sync.

    The returned ``(num_sources, 3)`` tensor is newly allocated. Its rows do
    not alias counters that continue changing during rollout collection.
    """

    if num_episodes.ndim != 1 or num_failures.ndim != 1:
        raise ValueError("joint-source episode and failure counters must be 1-D")
    if previous_success_rate.ndim != 1:
        raise ValueError("joint-source previous success rate must be 1-D")
    if not (
        num_episodes.shape
        == num_failures.shape
        == previous_success_rate.shape
    ):
        raise ValueError("joint-source metric tensors must have equal shapes")

    observed_success = 1.0 - num_failures / num_episodes.clamp_min(1.0)
    success_rate = torch.where(
        num_episodes > 0,
        observed_success,
        previous_success_rate,
    )
    return torch.stack(
        (num_episodes, num_failures, success_rate),
        dim=-1,
    ).detach()


@dataclass
class RuntimeSyncCleanupCounters:
    """Synchronization-free accounting for the optimized wrapper path."""

    steps: int = 0
    env_action_cpu_copies_omitted: int = 0
    env_action_cpu_bytes_omitted: int = 0
    source_metric_batches: int = 0
    source_scalar_host_syncs_avoided: int = 0

    def record_step(self) -> None:
        self.steps += 1

    def record_omitted_action_copy(self, actions: torch.Tensor) -> None:
        self.env_action_cpu_copies_omitted += 1
        self.env_action_cpu_bytes_omitted += (
            int(actions.numel()) * int(actions.element_size())
        )

    def record_batched_source_metrics(self, num_sources: int) -> None:
        self.source_metric_batches += 1
        # The legacy dictionary performs four .item() calls per source,
        # including previous_success_rate, which its caller does not log.
        self.source_scalar_host_syncs_avoided += 4 * int(num_sources)

    def as_dict(self) -> dict[str, int]:
        return {
            "steps": self.steps,
            "env_action_cpu_copies_omitted": self.env_action_cpu_copies_omitted,
            "env_action_cpu_bytes_omitted": self.env_action_cpu_bytes_omitted,
            "source_metric_batches": self.source_metric_batches,
            "source_scalar_host_syncs_avoided": (
                self.source_scalar_host_syncs_avoided
            ),
        }
