"""Completion check for auto-resumed PPO runs.

A requeued leg that resumes from a checkpoint already at the target update
must not run another update. The training loop only stops after an update, so
without this check a resume at step N == target trains (and never saves)
update N + 1.
"""

from __future__ import annotations

# Printed by the trainer and matched by an external launcher; keep them in sync.
ALREADY_COMPLETE_MARKER = "== Training already complete on resume =="


def training_already_complete(global_step: int, num_total_batches: int) -> bool:
    """True when the restored step has reached the configured number of updates."""
    return int(num_total_batches) > 0 and int(global_step) >= int(num_total_batches)
