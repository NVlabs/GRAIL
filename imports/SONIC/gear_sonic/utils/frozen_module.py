"""Utilities for making a pretrained module an immutable runtime component."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class FrozenModuleSummary:
    parameter_count: int
    buffer_count: int
    submodule_count: int


def assert_module_frozen(module: nn.Module, *, module_name: str) -> FrozenModuleSummary:
    """Require every nested module to be in eval mode with no trainable parameters."""
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    training = [
        name or "<root>"
        for name, submodule in module.named_modules()
        if submodule.training
    ]
    if trainable or training:
        details = []
        if trainable:
            details.append(f"trainable parameters={trainable[:8]}")
        if training:
            details.append(f"training-mode modules={training[:8]}")
        raise RuntimeError(f"{module_name} is not fully frozen: " + "; ".join(details))

    return FrozenModuleSummary(
        parameter_count=sum(parameter.numel() for parameter in module.parameters()),
        buffer_count=sum(buffer.numel() for buffer in module.buffers()),
        submodule_count=sum(1 for _ in module.modules()),
    )


def freeze_module(module: nn.Module, *, module_name: str) -> FrozenModuleSummary:
    """Freeze a complete module tree and verify the resulting invariant."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return assert_module_frozen(module, module_name=module_name)
