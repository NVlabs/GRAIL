"""Validate explicit foot-support loss inputs without preparing geometry or exports."""

from __future__ import annotations

import copy
import math


def bind_scene_support_losses(losses, manifest, *, source_fps=None, probe=False):
    """Bind data to YAML losses without choosing or overriding solver weights.

    Disabled support terms are omitted from the effective objective. A geometry
    probe may omit unbound support terms because it performs no optimization;
    a positive-iteration fit must supply the reviewed support data.
    """
    losses = copy.deepcopy(losses)
    names = ("human_scene_support", "human_scene_support_temporal")
    reviewed_contact = losses.get("human_foot_contact", {}).get("label_source") == "scene_support"
    if reviewed_contact:
        names += ("human_foot_contact",)
    for name in names:
        if name not in losses:
            continue
        term = losses[name]
        weight = term.get("weight") if isinstance(term, dict) else None
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(f"{name}.weight must be finite and nonnegative")
        if weight == 0 or (probe and manifest is None):
            del losses[name]
            continue
        if manifest is None:
            raise ValueError(f"Enabled {name} requires a reviewed support manifest")
        if term.get("manifest") is not None and term["manifest"] != manifest:
            raise ValueError(f"{name} manifest differs from the bound support input")
        term["manifest"] = copy.deepcopy(manifest)

    support = losses.get("human_scene_support", {})
    for name in ("height_blend_frames", "tangent_blend_frames"):
        value = support.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Support blend frames must be nonnegative integers")
    temporal = losses.get("human_scene_support_temporal")
    if temporal is not None:
        if not support:
            raise ValueError("Temporal support requires enabled human_scene_support in YAML")
        if (
            losses.get("human_foot_contact", {}).get("weight", 0) != 0
            and temporal.get("component_weights", {}).get("no_slip", 0) != 0
        ):
            raise ValueError("Set scene temporal no_slip to 0 when human_foot_contact is enabled")
    timed_terms = [temporal] if temporal is not None else []
    contact = losses.get("human_foot_contact") if reviewed_contact else None
    if contact is not None:
        if not support:
            raise ValueError("Reviewed foot contact requires enabled human_scene_support in YAML")
        if contact.get("velocity_space") != "planar" or contact.get("point_mode") != "foot_parts":
            raise ValueError("Reviewed foot contact requires planar foot_parts")
        if contact.get("options", {}).get("normal_velocity_weight") != 0:
            raise ValueError("Planar reviewed foot contact requires normal_velocity_weight: 0")
        timed_terms.append(contact)
    for term in timed_terms:
        options = term.get("options")
        if not isinstance(options, dict):
            raise ValueError("Temporal support options must be a YAML mapping")
        if options.get("tangent_blend_frames") != support.get("tangent_blend_frames"):
            raise ValueError("Tangential support and no-slip must share their blend settings")
        if source_fps is not None and (
            isinstance(source_fps, bool)
            or not isinstance(source_fps, (int, float))
            or not math.isfinite(source_fps)
            or source_fps <= 0
        ):
            raise ValueError("Source FPS must be finite and positive")
        if options.get("fps") == "source":
            if source_fps is None:
                raise ValueError("fps: source requires the source FPS")
            options["fps"] = float(source_fps)
        fps = options.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("Temporal support FPS must be finite and positive")
        if source_fps is not None and float(fps) != float(source_fps):
            raise ValueError("Temporal support FPS differs from the unchanged source timeline")
    return losses


def validate_foot_support_config(config):
    """Enforce the mode authored by a config family across every numerical stage.

    There is deliberately no case-level mode, mode CLI option, geometry-driven
    mode selection, or fallback. Old configs without a declared mode remain
    inspectable, but the maintained manipulation/locomotion YAMLs declare one.
    """
    mode = config.get("foot_support_mode")
    if mode is None:
        return None
    if mode not in ("plane", "terrain"):
        raise ValueError("foot_support_mode must be plane or terrain in the family YAML")
    if "scene_support" in config:
        raise ValueError(
            "Optional scene_support overlays are incompatible with fixed support modes"
        )
    required = (
        ("human_foot_ground", "human_foot_ground_temporal")
        if mode == "plane"
        else ("human_scene_support", "human_scene_support_temporal")
    )
    forbidden = (
        ("human_scene_support", "human_scene_support_temporal")
        if mode == "plane"
        else ("human_foot_ground", "human_foot_ground_temporal")
    )
    stages = config["opt_stage_specs"]
    if mode == "terrain" and set(stages) != {"init_opt"}:
        raise ValueError("Terrain locomotion uses only the existing init_opt stage")
    for name, stage in stages.items():
        if "foot_support_mode" in stage:
            raise ValueError("Stage-specific support mode overrides are forbidden")
        losses = stage.get("loss_cfg", {})
        for key in (*required, "human_foot_contact"):
            weight = losses.get(key, {}).get("weight", 0)
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError(f"{mode} mode requires enabled {key} in {name}")
        if any(key in losses for key in forbidden):
            raise ValueError(f"{name} contains losses from the other support mode")
        if losses["human_foot_contact"].get("velocity_space") != "planar":
            raise ValueError("Both support modes retain planar human_foot_contact")
        expected_source = "scene_support" if mode == "terrain" else "gem"
        if losses["human_foot_contact"].get("label_source") != expected_source:
            raise ValueError(f"{mode} mode requires foot-contact label_source: {expected_source}")
    return mode


def configure_loss_support(config, fps):
    """Resolve loss-local observations and source timing for numerical fitting.

    Plane coefficients/provenance are supplied directly in the ground loss;
    terrain uses the existing reviewed manifest. No scene files are loaded,
    geometry is inferred, or native-output fields are added here.
    """
    mode = validate_foot_support_config(config)
    stages = config.get("opt_stage_specs", {})
    planes = []
    for spec in stages.values():
        losses = spec.get("loss_cfg", {})
        ground = losses.get("human_foot_ground", {})
        if ground.get("weight", 0) > 0:
            import torch

            from grail.optimization.foot_ground import normalized_ground_plane

            # Reuse the original plane/provenance contract; no file receipt is required.
            normalized_ground_plane(ground.get("plane"), torch.empty(0))
            planes.append(ground["plane"])
        temporal = losses.get("human_foot_ground_temporal", {})
        if temporal.get("weight", 0) > 0:
            if (
                isinstance(fps, bool)
                or not isinstance(fps, (int, float))
                or not math.isfinite(fps)
                or fps <= 0
            ):
                raise ValueError("Foot transition smoothing requires the actual source FPS")
            options = temporal.get("options")
            if not isinstance(options, dict):
                raise ValueError("Foot transition smoothing requires explicit options")
            if ground.get("weight", 0) > 0 and options.get("threshold", 0.5) != ground.get(
                "support_threshold", 0.5
            ):
                raise ValueError("Ground attraction and foot transitions must use the same labels")
            value = options.get("fps")
            if value != "source" and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value != fps
            ):
                raise ValueError("Foot transition smoothing FPS differs from the source")
            options["fps"] = float(fps)
        if mode == "terrain":
            manifest = losses.get("human_scene_support", {}).get("manifest")
            spec["loss_cfg"] = bind_scene_support_losses(losses, manifest, source_fps=fps)
    if mode == "plane" and any(plane != planes[0] for plane in planes[1:]):
        raise ValueError("Manipulation stages must use the same explicit ground plane")
    return mode
