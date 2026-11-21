"""Exact-resume verification for the ContactFirst manipulation finetune.

Enabled by ``algo.config.resume_verify_exact=true``: after ``load_checkpoint``
restores a full checkpoint, check that the live actor, critic, optimizer and
scheduler equal the saved state tensor by tensor and write
``resume_initialization.json`` next to the trainer output. Never active for the
released task-general tracking configs.
"""
from __future__ import annotations

import json
from pathlib import Path


def _equal(expected, actual):
    import torch

    if isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and expected.shape == actual.shape
                and expected.dtype == actual.dtype
                and torch.equal(expected.cpu(), actual.detach().cpu()))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(_equal(v, actual[k]) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (isinstance(actual, type(expected)) and len(expected) == len(actual)
                and all(_equal(a, b) for a, b in zip(expected, actual)))
    return expected == actual


def _require(value, message):
    if not value:
        raise ValueError(message)


def verify_exact_resume(trainer, checkpoint, output_dir) -> dict:
    """Check actor/critic/optimizer/scheduler against the restored checkpoint."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    _require(_equal(checkpoint["policy_state_dict"], model.policy.state_dict()),
             "Restored actor differs")
    _require(model.value_model is not None
             and _equal(checkpoint["value_state_dict"], model.value_model.state_dict()),
             "Restored critic differs")
    _require(_equal(checkpoint["optimizer_state_dict"], trainer.optimizer.state_dict()),
             "Restored optimizer differs")
    _require(_equal(checkpoint["lr_scheduler_state_dict"], trainer.lr_scheduler.state_dict()),
             "Restored scheduler differs")
    receipt = {
        "schema": "contactfirst.resume_initialization.v2",
        "status": "verified_on_load",
        "global_step": int(trainer.state.global_step),
        "exact_actor": True, "exact_critic": True,
        "exact_optimizer": True, "exact_scheduler": True,
        "optimizer_group_lrs": [float(g["lr"]) for g in trainer.optimizer.param_groups],
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "resume_initialization.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("[resume_initialization] exact actor/value/optimizer/scheduler verified: " + str(path),
          flush=True)
    return receipt
