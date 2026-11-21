"""Pure helpers for binding checkpoint migration to live observation schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


OBSERVATION_SCHEMA_CONTRACT = "resolved_observation_manager_terms_v1"


def _term_width(shape: Sequence[int]) -> int:
    width = math.prod(int(value) for value in shape)
    if width <= 0:
        raise ValueError(f"invalid observation term shape {list(shape)}")
    return width


def expected_runtime_schema(
    group: Mapping[str, Any],
    *,
    side: str,
) -> list[dict[str, Any]]:
    """Return the ordered runtime names and widths declared by one layout."""

    if side not in {"source", "target"}:
        raise ValueError(f"unsupported observation layout side {side!r}")
    raw_terms = group.get(f"{side}_terms")
    if not isinstance(raw_terms, Sequence) or isinstance(
        raw_terms, (str, bytes)
    ):
        raise TypeError(f"{side}_terms must be a sequence")
    aliases = group.get(f"{side}_runtime_aliases", {})
    if not isinstance(aliases, Mapping):
        raise TypeError(f"{side}_runtime_aliases must be a mapping")

    result: list[dict[str, Any]] = []
    semantic_names: set[str] = set()
    runtime_names: set[str] = set()
    for raw in raw_terms:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError(f"invalid {side} observation term {raw!r}")
        expected_fields = 2 if side == "source" else 3
        if len(raw) != expected_fields:
            raise ValueError(
                f"{side} observation term {raw!r} has {len(raw)} fields; "
                f"expected {expected_fields}"
            )
        semantic_name = str(raw[0])
        runtime_name = str(aliases.get(semantic_name, semantic_name))
        width = int(raw[1])
        if (
            not semantic_name
            or not runtime_name
            or semantic_name in semantic_names
            or runtime_name in runtime_names
            or width <= 0
        ):
            raise ValueError(f"invalid or duplicate {side} term {raw!r}")
        semantic_names.add(semantic_name)
        runtime_names.add(runtime_name)
        result.append(
            {
                "semantic_name": semantic_name,
                "runtime_name": runtime_name,
                "width": width,
            }
        )
    return result


def validate_observation_layout(layout: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize both source and target observation schemas."""

    if layout.get("schema_contract") != OBSERVATION_SCHEMA_CONTRACT:
        raise ValueError(
            "observation layout must declare "
            f"{OBSERVATION_SCHEMA_CONTRACT!r}"
        )
    groups = layout.get("groups")
    if not isinstance(groups, Mapping) or set(groups) != {"actor", "critic"}:
        raise ValueError("observation layout must contain actor and critic groups")

    report: dict[str, Any] = {}
    runtime_groups: set[str] = set()
    for group_name in ("actor", "critic"):
        group = groups[group_name]
        if not isinstance(group, Mapping):
            raise TypeError(f"layout group {group_name!r} must be a mapping")
        runtime_group = str(group.get("runtime_group", ""))
        if not runtime_group or runtime_group in runtime_groups:
            raise ValueError(
                f"layout group {group_name!r} has invalid runtime_group"
            )
        runtime_groups.add(runtime_group)
        source = expected_runtime_schema(group, side="source")
        target = expected_runtime_schema(group, side="target")
        source_width = sum(item["width"] for item in source)
        target_width = sum(item["width"] for item in target)
        if source_width != int(group.get("source_width", -1)):
            raise ValueError(f"{group_name}: source width mismatch")
        if target_width != int(group.get("target_width", -1)):
            raise ValueError(f"{group_name}: target width mismatch")
        report[group_name] = {
            "runtime_group": runtime_group,
            "source": source,
            "target": target,
            "source_width": source_width,
            "target_width": target_width,
        }
    return report


def normalize_live_schema(
    names: Sequence[Any],
    dims: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Normalize IsaacLab observation-manager metadata for serialization."""

    if len(names) != len(dims):
        raise ValueError(
            f"live observation names/dims length mismatch: "
            f"{len(names)} != {len(dims)}"
        )
    result = []
    for name, shape in zip(names, dims, strict=True):
        normalized_shape = [int(value) for value in shape]
        result.append(
            {
                "runtime_name": str(name),
                "shape": normalized_shape,
                "width": _term_width(normalized_shape),
            }
        )
    return result


def validate_live_target_schema(
    layout: Mapping[str, Any],
    live_groups: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail unless captured target names, order, and widths match the layout."""

    layout_report = validate_observation_layout(layout)
    group_reports: dict[str, Any] = {}
    for group_name, expected_group in layout_report.items():
        runtime_group = expected_group["runtime_group"]
        live = live_groups.get(runtime_group)
        if not isinstance(live, Sequence) or isinstance(live, (str, bytes)):
            raise KeyError(
                f"capture is missing live observation group {runtime_group!r}"
            )
        expected = [
            {
                "runtime_name": item["runtime_name"],
                "width": item["width"],
            }
            for item in expected_group["target"]
        ]
        observed = [
            {
                "runtime_name": str(item.get("runtime_name", "")),
                "width": int(item.get("width", -1)),
            }
            for item in live
            if isinstance(item, Mapping)
        ]
        if observed != expected:
            raise ValueError(
                f"live {runtime_group!r} observation schema differs from "
                f"the migration layout: expected={expected}, "
                f"observed={observed}"
            )
        group_reports[group_name] = {
            "runtime_group": runtime_group,
            "term_count": len(observed),
            "width": sum(item["width"] for item in observed),
            "passed": True,
        }
    return group_reports


def reconstruct_source_observation(
    target: Any,
    source_only: Mapping[str, Any],
    group: Mapping[str, Any],
) -> Any:
    """Reconstruct one legacy input from a paired target observation.

    Copied terms are selected by semantic name, so this remains correct when
    target-only terms are inserted or inherited observation order changes.
    Removed source terms must be supplied explicitly by the live capture.
    """

    import torch

    if not isinstance(target, torch.Tensor) or target.ndim != 2:
        raise ValueError(
            "target observation must be a rank-2 torch tensor, got "
            f"{type(target).__name__} {getattr(target, 'shape', None)}"
        )

    source_terms = expected_runtime_schema(group, side="source")
    target_terms = expected_runtime_schema(group, side="target")
    source_width = int(group["source_width"])
    target_width = int(group["target_width"])
    if target.shape[1] != target_width:
        raise ValueError(
            f"target observation width {target.shape[1]} != {target_width}"
        )

    target_actions = {
        str(raw[0]): str(raw[2]) for raw in group["target_terms"]
    }
    target_slices: dict[str, slice] = {}
    offset = 0
    for item in target_terms:
        width = int(item["width"])
        target_slices[str(item["semantic_name"])] = slice(
            offset, offset + width
        )
        offset += width

    source = torch.empty(
        target.shape[0],
        source_width,
        dtype=target.dtype,
        device=target.device,
    )
    offset = 0
    for item in source_terms:
        name = str(item["semantic_name"])
        width = int(item["width"])
        destination = slice(offset, offset + width)
        target_slice = target_slices.get(name)
        if (
            target_slice is not None
            and target_actions.get(name) == "copy"
        ):
            source[:, destination] = target[:, target_slice]
        else:
            value = source_only.get(name)
            if not isinstance(value, torch.Tensor):
                raise KeyError(
                    f"paired capture is missing source-only term {name!r}"
                )
            value = value.to(device=target.device, dtype=target.dtype)
            expected_shape = (target.shape[0], width)
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"source-only term {name!r} has shape "
                    f"{tuple(value.shape)}, expected {expected_shape}"
                )
            source[:, destination] = value
        offset += width

    if not torch.isfinite(source).all():
        raise ValueError("reconstructed source observation is non-finite")
    return source
