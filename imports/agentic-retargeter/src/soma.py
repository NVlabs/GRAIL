"""Foreground adapter for the official NVIDIA SOMA-base BVH retargeter.

The heavy Newton/Warp runtime is external. Its robot CSV uses centimetres and
Euler/joint degrees and contains no FPS; this adapter preserves those semantics
and binds the source timing and model beside every unreviewed conversion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .gmr import _file, _path, _read, _sha
from .soma_runtime import validate_scale

ROOT_COLUMNS = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
                "root_rotateX", "root_rotateY", "root_rotateZ"]
_MARKER = "AGENTIC_SOMA_RUNTIME="


def _asset(path: Path) -> dict[str, str]:
    entry = _file(path)
    with path.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Unresolved Git LFS asset; fetch this file first: {path}")
    return entry


def _tree(path: Path, *, python_only: bool = False) -> dict[str, dict[str, str]]:
    entries = {str(p.relative_to(path)): _asset(p) for p in sorted(path.rglob("*"))
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
               and (not python_only or p.suffix == ".py")}
    if not entries:
        raise ValueError(f"Required tree is empty or missing: {path}")
    return entries


def inspect_bvh(path: str | Path) -> dict[str, Any]:
    """Validate SOMA's BVH dialect without importing Newton or executing pickle.

    Names/parent relationships are later compared with the bound upstream SOMA
    reference. Offsets may differ with subject proportions. Both ordinary
    three-value and SOMA six-value OFFSET records are supported.
    """
    path = Path(path)
    _asset(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines or lines[0] != "HIERARCHY" or "MOTION" not in lines:
        raise ValueError("Input must be a SOMA-base BVH with HIERARCHY and MOTION")
    split = lines.index("MOTION")
    stack: list[str | None] = []
    pending: str | None = None
    awaiting = False
    joints: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    roots = 0
    for line in lines[1:split]:
        parts = line.split()
        if parts[0] in {"ROOT", "JOINT"}:
            if len(parts) != 2 or awaiting or (parts[0] == "JOINT" and not stack):
                raise ValueError("Malformed BVH joint declaration")
            name = parts[1]
            if name in by_name or (stack and stack[-1] is None):
                raise ValueError("Duplicate joint name or joint inside End Site")
            if parts[0] == "ROOT":
                roots += 1
                if stack:
                    raise ValueError("BVH ROOT cannot have a parent")
            item = {"name": name, "parent": stack[-1] if stack else None, "channels": []}
            joints.append(item)
            by_name[name] = item
            pending, awaiting = name, True
        elif line == "End Site":
            if awaiting or not stack:
                raise ValueError("Malformed BVH End Site")
            pending, awaiting = None, True
        elif line == "{":
            if not awaiting:
                raise ValueError("Unmatched BVH opening brace")
            stack.append(pending)
            awaiting = False
        elif line == "}":
            if not stack or awaiting:
                raise ValueError("Unmatched BVH closing brace")
            stack.pop()
        elif parts[0] == "OFFSET":
            if not stack or len(parts) not in {4, 7}:
                raise ValueError("BVH OFFSET requires 3 or 6 finite values")
            if not all(math.isfinite(float(v)) for v in parts[1:]):
                raise ValueError("Non-finite BVH OFFSET")
        elif parts[0] == "CHANNELS":
            if not stack or stack[-1] is None or len(parts) < 2:
                raise ValueError("Malformed BVH CHANNELS")
            count = int(parts[1])
            channels = parts[2:]
            valid = {axis + kind for axis in "XYZ" for kind in ("position", "rotation")}
            if count not in {3, 6} or count != len(channels) or len(set(channels)) != count or not set(channels) <= valid:
                raise ValueError("Unsupported BVH channels")
            if {c for c in channels if c.endswith("rotation")} != {a + "rotation" for a in "XYZ"}:
                raise ValueError("Each BVH joint requires three rotation channels")
            if by_name[stack[-1]]["channels"]:
                raise ValueError("Duplicate BVH CHANNELS")
            by_name[stack[-1]]["channels"] = channels
        else:
            raise ValueError(f"Unsupported BVH hierarchy record: {parts[0]}")
    if roots != 1 or stack or awaiting or not joints or any(not j["channels"] for j in joints):
        raise ValueError("Incomplete BVH hierarchy")
    if len(lines) < split + 4:
        raise ValueError("BVH has no motion frames")
    count_match = re.fullmatch(r"Frames:\s*(\d+)", lines[split + 1])
    time_match = re.fullmatch(r"Frame Time:\s*(\S+)", lines[split + 2])
    if not count_match or not time_match:
        raise ValueError("Malformed BVH frame count/time")
    frame_count, frame_time = int(count_match[1]), float(time_match[1])
    if frame_count < 1 or not math.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("BVH frame count and time must be positive and finite")
    channels = sum(len(j["channels"]) for j in joints)
    frames = lines[split + 3:]
    if len(frames) != frame_count:
        raise ValueError("BVH declared frame count differs from motion rows")
    for index, line in enumerate(frames):
        values = line.split()
        if len(values) != channels or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"Malformed/non-finite BVH motion row {index}")
    # Official BVHImporter.get_frame_range rounds to two decimals.
    fps = round(1 / frame_time, 2)
    if fps <= 0:
        raise ValueError("BVH effective frame rate is zero")
    return {"frame_count": frame_count, "raw_frame_time": frame_time, "fps": fps,
            "fps_policy": "upstream_round_reciprocal_to_2_decimals",
            "channel_count": channels, "joints": joints}


def _target(root: Path, robot: str, extra_paths: list[Path] | None = None) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for directory in [root / "soma_retargeter/assets/robotics", *(extra_paths or [])]:
        for manifest in sorted(directory.rglob("manifest.json")):
            data = _read(manifest)
            for entry in data.get("targets", [data]):
                if entry.get("name") == robot:
                    candidates.append((manifest, entry))
    if len(candidates) != 1:
        raise ValueError(f"Expected one unambiguous target manifest for {robot}; got {len(candidates)}")
    return candidates[0]


def _load_external_robot_hands(config, *, base, model, frame_count, fps,
                              rotations, positions, scale):
    """Validate pose-only inputs before permitting an empty moving contact set."""
    import xml.etree.ElementTree as ET
    from .soma_robot_hand_targets import load_robot_hand_targets, validate_robot_hand_target_config
    config = validate_robot_hand_target_config(config)
    for key in ("handoff_json", "targets_npz"):
        config[key] = str(_path(config[key], base))
    tree = ET.parse(model)
    names = lambda tag: [item.get("name") for item in tree.iter(tag) if item.get("name")]
    return load_robot_hand_targets(config, fps=fps, frame_count=frame_count,
        object_positions=positions, object_rotations=rotations, object_scale=scale,
        available_joint_names=names("joint"), available_body_names=names("body"))


def _load_pose_keys(config, *, base, model, source, frame_count, fps):
    import mujoco
    from .soma_pose_keyframes import load
    native = mujoco.MjModel.from_xml_path(str(model))
    limits = {native.joint(i).name: tuple(native.jnt_range[i])
        for i in range(native.njnt) if native.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
    return load(config, model_xml=model, source_bvh=source, frame_count=frame_count,
                fps=fps, joint_limits=limits, base_path=base)


def plan(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Bind one source and one original-mesh robot; never execute or write a plan."""
    config_path = Path(config_path).resolve()
    cfg = _read(config_path)
    base = config_path.parent
    root, source = _path(cfg["soma_root"], base), _path(cfg["source"], base)
    if source.suffix.lower() != ".bvh":
        raise ValueError("SOMA route requires SOMA-base .bvh; NPZ/body parameters require a model-bound upstream BVH export")
    if cfg.get("source_units") != "cm":
        raise ValueError("Declare source_units='cm'; official BVH importer always scales centimetres to metres")
    facing = cfg.get("source_facing_direction")
    if facing not in {"Mujoco", "Maya"}:
        raise ValueError("Explicit source_facing_direction must be Mujoco or Maya")
    robot = cfg.get("robot", "unitree_g1")
    if not isinstance(robot, str) or not robot:
        raise ValueError("robot must be a registry target name")
    human_scale = validate_scale(cfg.get("human_scale_ratio", 1.0))
    from .soma_temporal import default_forward_device
    device = cfg.get("device", default_forward_device(cfg.get("temporal_optimization")))
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    extra = cfg.get("extra_robot_paths", [])
    if not isinstance(extra, list) or any(not isinstance(p, str) or not p for p in extra):
        raise ValueError("extra_robot_paths must be a list of existing directories")
    extra_paths = [_path(p, base) for p in extra]
    if any(not p.is_dir() for p in extra_paths) or len(set(extra_paths)) != len(extra_paths):
        raise ValueError("extra_robot_paths must be distinct existing directories")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output must be a new path: {output}")
    python_value = cfg.get("python", "python")
    if not isinstance(python_value, str) or not python_value:
        raise ValueError("python must be an executable path")
    # Keep the venv executable path: resolving its symlink to /usr/bin/python
    # would silently discard the configured virtual environment.
    python_path = Path(python_value).expanduser()
    python = (shutil.which(python_value) if "/" not in python_value else
              os.path.abspath(python_path if python_path.is_absolute() else base / python_path))
    if not python or not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Python executable not found: {python_value}")
    script = root / "app/bvh_to_csv_converter.py"
    # The manifest API is current upstream; fail explicitly for the older GEM-SOMA vendored snapshot.
    _file(root / "soma_retargeter/robotics/robot_registry.py")
    manifest, entry = _target(root, robot, extra_paths)
    desc = entry["desc"]
    model_keys = [key for key in ("xml_path", "urdf_path") if key in desc]
    if "newton_asset" in desc or len(model_keys) != 1:
        raise ValueError("This adapter requires one local MJCF or URDF model; runtime downloads are not bound")
    model_key = model_keys[0]
    model = _path(desc[model_key], manifest.parent)
    if manifest.parent not in model.parents:
        raise ValueError("Robot model must stay within its bound manifest asset tree")
    retarget_path = _path(entry["retarget_configs"]["soma"], manifest.parent)
    if manifest.parent not in retarget_path.parents:
        raise ValueError("Retarget config escapes its bound asset tree")
    _file(retarget_path)
    source_info = inspect_bvh(source)
    reference = root / "soma_retargeter/assets/soma/soma_zero_frame0.bvh"
    reference_info = inspect_bvh(reference)
    signature = lambda info: [(j["name"], j["parent"]) for j in info["joints"]]
    if signature(source_info) != signature(reference_info):
        raise ValueError("Input hierarchy/naming/order differs from the bound SOMA-base skeleton")
    if "fps" in cfg and (not isinstance(cfg["fps"], (int, float)) or isinstance(cfg["fps"], bool)
                         or not math.isclose(cfg["fps"], source_info["fps"], rel_tol=0, abs_tol=1e-8)):
        raise ValueError("Configured FPS disagrees with upstream BVH effective FPS")
    bindings_cfg = cfg.get("bindings", {})
    if not isinstance(bindings_cfg, dict) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", k) for k in bindings_cfg):
        raise ValueError("bindings must map simple names to original source/object/camera/model evidence files")
    bindings = {name: _asset(_path(value, base)) for name, value in bindings_cfg.items()}
    for protected in (root, source, config_path, model, *(Path(v["path"]) for v in bindings.values())):
        if output == protected or output in protected.parents or (protected.is_dir() and protected in output.parents):
            raise ValueError(f"Output overlaps a protected input/upstream tree: {protected}")
    converter = {"import_folder": str(output / "input"), "export_folder": str(output / "raw"),
                 "batch_size": 1, "retargeter": "Newton", "retarget_source": "soma", "retarget_target": robot,
                 "retarget_source_facing_direction": facing, "human_scale_ratio": human_scale,
                 "agentic_upstream_root": str(root), "agentic_scale_report": str(output / "scaling.json")}
    if extra_paths:
        converter["extra_robot_paths"] = [str(p) for p in extra_paths]
    code = {"app/bvh_to_csv_converter.py": _file(script)}
    code.update({"soma_retargeter/" + k: v for k, v in _tree(root / "soma_retargeter", python_only=True).items()})
    for name in ("pyproject.toml", "uv.lock"):
        if (root / name).is_file():
            code[name] = _file(root / name)
    proposal = {"schema_version": 1, "backend": "soma", "source_model": "soma", "status": "PLANNED_UNREVIEWED",
            "config": _file(config_path), "python": _file(Path(python)), "cwd": str(root), "output": str(output),
            "adapter_code": {name: _file(Path(__file__).with_name(name))
                             for name in ("soma.py", "gmr.py", "soma_runtime.py")},
            "commands": [[python, "-u", str(Path(__file__).with_name("soma_runtime.py")), "--config", str(output / "converter.json"),
                          "--viewer", "null", "--device", device]], "converter_config": converter,
            "source": _asset(source), "source_info": source_info, "source_units": "cm",
            "source_facing_direction": facing, "robot": robot, "model_description": _asset(model),
            "model_asset_key": model_key, "human_scale_ratio": human_scale,
            "robot_assets": _tree(manifest.parent), "soma_assets": _tree(reference.parent),
            "upstream_code": code, "bindings": bindings, "blockers": [],
            "limitations": ["SOMA-base BVH only; generic BVH and NPZ/body-parameter files are not interchangeable.",
                            "Robot DOFs and finger support are determined by the bound model and IK match table; no fingers are synthesized.",
                            "Upstream contact processing concerns locomotion/feet; this is not an HOI grasp solve.",
                            "CSV has no FPS; use the recorded BVH effective rate, not the CSV loader default.",
                            "Object/camera evidence remains separately bound; initial conversion does not accept motion."]}
    proposal["model_xml" if model_key == "xml_path" else "model_urdf"] = proposal["model_description"]
    if "temporal_optimization" in cfg and "contact_constraints" not in cfg:
        raise ValueError("temporal_optimization currently requires the explicit contact_constraints bridge")
    if "hand_target_optimization" in cfg and "contact_constraints" not in cfg:
        raise ValueError("hand_target_optimization currently requires the explicit contact_constraints bridge")
    if "pose_keyframes" in cfg and "contact_constraints" not in cfg:
        raise ValueError("pose_keyframes requires the explicit contact_constraints bridge")
    if "robot_hand_targets" in cfg and "contact_constraints" not in cfg:
        raise ValueError("robot_hand_targets requires the explicit contact_constraints bridge")
    if "wrist_keyframes" in cfg and ("robot_hand_targets" not in cfg or "contact_constraints" not in cfg):
        raise ValueError("wrist_keyframes requires unchanged robot_hand_targets and contact_constraints")
    if "object_optimization" in cfg and ("contact_constraints" not in cfg or "temporal_optimization" not in cfg):
        raise ValueError("object_optimization requires contact_constraints and temporal_optimization")
    if "contact_constraints" in cfg:
        if model_key != "xml_path":
            raise ValueError("contact_constraints currently requires a bound MJCF target")
        from .soma_contacts import compile_contacts
        spec_path = _path(cfg["contact_constraints"], base)
        robot_hands = None
        if "robot_hand_targets" in cfg:
            import numpy as np
            contact_spec = _read(spec_path)
            trajectory = _path(contact_spec["object_trajectory"], spec_path.parent)
            with np.load(trajectory, allow_pickle=False) as reference:
                robot_hands = _load_external_robot_hands(cfg["robot_hand_targets"], base=base,
                    model=model, frame_count=source_info["frame_count"], fps=source_info["fps"],
                    rotations=reference["obj_R"], positions=reference["obj_t"],
                    scale=reference["object_scale"])
        schedule, inputs, summary = compile_contacts(spec_path, frame_count=source_info["frame_count"],
            fps=source_info["fps"], model_xml=model, include_object_scene="object_optimization" in cfg,
            allow_empty_moving=robot_hands is not None)
        if robot_hands is not None:
            proposal["robot_hand_targets"] = robot_hands["config"]
            proposal["robot_hand_target_summary"] = robot_hands["report"]
            proposal["robot_hand_target_inputs"] = {
                key: _asset(Path(value["path"])) for key,value in robot_hands["report"]["input_files"].items()}
            for entry in proposal["robot_hand_target_inputs"].values():
                protected = Path(entry["path"])
                if output == protected or output in protected.parents:
                    raise ValueError("Output overlaps a protected native hand target input")
            converter["agentic_robot_hand_targets"] = robot_hands["config"]
            for filename in ("soma_robot_hand_targets.py", "soma_robot_hand_runtime.py"):
                proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
            proposal["limitations"].append(
                "External native wrist and finger trajectories are unaccepted soft objectives, tracked "
                "in live-object coordinates on declared contact phases and fixed world coordinates otherwise. "
                "Original human wrist/finger objective weights are explicitly scaled only for enabled hands. "
                "Body targets, timing and collision objectives are retained; pose tracking is not contact acceptance.")
        if "wrist_keyframes" in cfg:
            from .soma_wrist_keyframes import load as load_wrist_keys
            wrists = load_wrist_keys(cfg["wrist_keyframes"], base_path=base, model_xml=model,
                source_bvh=source, frame_count=source_info["frame_count"], fps=source_info["fps"],
                hand_targets=robot_hands)
            proposal["wrist_keyframes"] = wrists["config"]
            proposal["wrist_keyframe_summary"] = wrists["report"]
            proposal["wrist_keyframe_inputs"] = {
                name:_asset(Path(entry["path"])) for name,entry in wrists["report"]["input_files"].items()}
            for entry in proposal["wrist_keyframe_inputs"].values():
                protected = Path(entry["path"])
                if output == protected or output in protected.parents:
                    raise ValueError("Output overlaps a protected wrist keyframe input")
            converter["agentic_wrist_keyframes"] = dict(config=wrists["config"], model_xml=str(model), source_bvh=str(source))
            for filename in ("soma_wrist_keyframes.py", "soma_wrist_keyframe_runtime.py"):
                proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
            proposal["limitations"].append("Sparse authored object-relative wrist keys are additive soft costs in forward/global SOMA. Original hand goals remain active; no root/object pins or collision acceptance are implied.")
        if "pose_keyframes" in cfg:
            keys = _load_pose_keys(cfg["pose_keyframes"], base=base, model=model, source=source,
                frame_count=source_info["frame_count"], fps=source_info["fps"])
            proposal["pose_keyframes"] = keys["config"]
            proposal["pose_keyframe_summary"] = keys["report"]
            proposal["pose_keyframe_inputs"] = {"keys": _asset(Path(keys["config"]["path"]))}
            protected = Path(keys["config"]["path"])
            if output == protected or output in protected.parents:
                raise ValueError("Output overlaps a protected pose keyframe input")
            converter["agentic_pose_keyframes"] = dict(config=keys["config"],
                model_xml=str(model), source_bvh=str(source))
            for filename in ("soma_pose_keyframes.py", "soma_pose_keyframe_runtime.py", "soma_temporal_runtime.py"):
                proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
            proposal["limitations"].append("Authored scalar hinge keys are sparse soft objectives in forward and global SOMA, not hard pins; final deviations and independent collisions require review.")
        proposal["contact_inputs"] = {k: _asset(v) for k, v in inputs.items()}
        proposal["contact_constraints"] = summary
        for entry in proposal["contact_inputs"].values():
            protected = Path(entry["path"])
            if output == protected or output in protected.parents:
                raise ValueError("Output overlaps a protected contact input")
        for filename in ("soma_contacts.py", "soma_contact_runtime.py", "soma_object_policy.py"):
            proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
        if "collision_avoidance" in summary:
            for filename in ("soma_collision.py", "soma_collision_runtime.py", "soma_collision_witnesses.py", "repair_geometry.py", "static_scene.py"):
                proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
        proposal["commands"][0][2] = str(Path(__file__).with_name("soma_contact_runtime.py"))
        converter.update(agentic_upstream_root=str(root), agentic_contacts=str(output / "contact_targets.npz"),
                         agentic_contact_report=str(output / "contact_initialization.json"))
        proposal["limitations"][2] = ("Authored hand/object soft objectives participate in SOMA IK; collision/source/temporal review is still required."
            if summary["contact_loss_policy"]["enabled"] else
            "Static source object: hand/object position and normal contact losses are disabled in forward and temporal IK. Authored contacts are diagnostic only; collision/source/temporal review is still required.")
        proposal["limitations"].append("Contact mode disables upstream postprocessing, including foot stabilization, to prevent wrist overwrite.")
        if "object_optimization" in cfg:
            from .soma_object import ObjectTrajectory, validate_object_config
            import numpy as np
            if "collision_avoidance" not in summary:
                raise ValueError("object_optimization requires configured object/robot collision avoidance")
            if "object" not in bindings:
                raise ValueError("object_optimization requires bindings.object for the original native scene sidecar")
            with np.load(bindings["object"]["path"], allow_pickle=False) as original:
                if (not np.array_equal(original["obj_R"], schedule["object_reference_rotations"])
                        or not np.array_equal(original["obj_t"], schedule["object_reference_positions"])
                        or not np.array_equal(np.broadcast_to(original["object_scale"], (3,)), schedule["object_scale"])):
                    raise ValueError("Bound original object sidecar differs from contact trajectory reference")
            object_config = validate_object_config(cfg["object_optimization"])
            object_controller = ObjectTrajectory(schedule, object_config,
                allow_empty_material=robot_hands is not None)
            proposal["object_optimization"] = object_config
            proposal["object_motion_policy"] = object_controller.motion_policy
            converter.update(agentic_object_optimization=object_config,
                agentic_object_reference_sha256=proposal["contact_inputs"]["object_trajectory"]["sha256"])
            for filename in ("soma_object.py", "soma_object_io.py", "soma_temporal.py"):
                proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
            proposal["limitations"].append("Moving-source object and robot blocks alternate from the first sequence sweep; contact gaps are diagnostics, the moving-source precontact interval shares one optimized SE(3) pose, and wholly static source object paths pass through unchanged. Original-pose fidelity is a soft objective; deviation/rate caps apply only when explicitly configured. Contacts and clearance remain sampled soft objectives.")
    if "temporal_optimization" in cfg:
        from .soma_temporal import validate_temporal_config
        temporal = validate_temporal_config(cfg["temporal_optimization"])
        if source_info["frame_count"] < 4:
            raise ValueError("SOMA temporal optimization requires at least four source frames")
        if "relative_hand_temporal" in temporal:
            if "robot_hand_targets" not in proposal:
                raise ValueError("relative_hand_temporal requires explicit robot_hand_targets")
            proposal["adapter_code"]["soma_relative_temporal.py"] = _file(Path(__file__).with_name("soma_relative_temporal.py"))
        if "world_hand_temporal" in temporal:
            if "robot_hand_targets" not in proposal:
                raise ValueError("world_hand_temporal requires explicit robot_hand_targets")
            for filename in ("soma_world_temporal.py", "soma_relative_temporal.py"):
                proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
        if "hinge_speed" in temporal:
            proposal["adapter_code"]["soma_hinge_speed.py"] = _file(Path(__file__).with_name("soma_hinge_speed.py"))
        if "refinement_scope" in temporal:
            proposal["adapter_code"]["soma_refinement_scope.py"] = _file(Path(__file__).with_name("soma_refinement_scope.py"))
            proposal["limitations"].append("Explicit refinement_scope restricts optimized native scalar hinges and source-frame windows; the actual prior root, object and all other coordinates stay byte-exact. All trajectory objectives and boundary stencils remain. Convergence is only over the declared subset, not unrestricted whole-body convergence or collision acceptance.")
        if "prior_global_seed" in temporal:
            if "robot_hand_targets" not in proposal or "object_optimization" not in proposal:
                raise ValueError("prior_global_seed requires explicit hand targets and object optimization")
            from .soma_prior_global_seed import load as load_prior_seed
            seeded = load_prior_seed(temporal["prior_global_seed"], base_path=base,
                model_sha256=_sha(model), source_sha256=_sha(source),
                frame_count=source_info["frame_count"], fps=source_info["fps"],
                reference_rotations=schedule["object_reference_rotations"],
                reference_positions=schedule["object_reference_positions"],
                object_scale=schedule["object_scale"], object_config=object_config)
            temporal["prior_global_seed"] = seeded["config"]
            proposal["prior_global_seed_inputs"] = {
                key:_asset(Path(value["path"])) for key,value in seeded["files"].items()}
            for entry in proposal["prior_global_seed_inputs"].values():
                protected = Path(entry["path"])
                if output == protected or output in protected.parents:
                    raise ValueError("Output overlaps a protected prior-global seed input")
            proposal["adapter_code"]["soma_prior_global_seed.py"] = _file(Path(__file__).with_name("soma_prior_global_seed.py"))
            proposal["limitations"].append("Fresh forward initialization is retained as separate evidence; the global solve explicitly starts from a bound prior global checkpoint. Original targets remain unchanged; warm start does not prevent far-field drift or imply collision acceptance.")
        proposal["temporal_optimization"] = temporal
        converter["agentic_temporal_optimization"] = temporal
        for filename in ("soma_temporal.py", "soma_temporal_runtime.py"):
            proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
        if temporal.get("method") == "synchronous_sparse":
            for filename in ("soma_sparse_sequence.py", "soma_sequence_object.py",
                             "soma_sequence_runtime.py", "soma_synchronous_result.py",
                             "soma_sequence_cuda.py", "soma_sequence_device.py",
                             "soma_forward_cuda.py"):
                proposal["adapter_code"][filename] = _file(Path(__file__).with_name(filename))
            forward_contract = {"requested_device": device,
                "source_frame_count": source_info["frame_count"]}
            retarget_settings = _read(retarget_path)
            warmup_keys = ("num_initialization_frames", "num_stabilization_frames")
            if all(key in retarget_settings for key in warmup_keys):
                if any(type(retarget_settings[key]) is not int or retarget_settings[key] < 0
                       for key in warmup_keys):
                    raise ValueError("SOMA warmup frame counts must be nonnegative integers")
                forward_contract["warmup_frame_count"] = sum(retarget_settings[key] for key in warmup_keys)
            proposal["forward_execution_contract"] = forward_contract
            proposal["limitations"].append(
                "All source-frame native robot and editable object tangent variables are solved in one "
                "sparse simultaneous problem. Accepted global steps retain original source/contact/collision "
                "and reference/temporal costs; nonlinear convergence and motion acceptance remain separate.")
        if "sequence_warm_start" in temporal:
            proposal["adapter_code"]["soma_sequence_warm_start.py"] = _file(
                Path(__file__).with_name("soma_sequence_warm_start.py"))
            proposal["limitations"].append(
                "Declared global hand-sequence warm start changes only the solver starting hinges; "
                "its quadratic fidelity is to the forward robot solution, not human observations. "
                "Original contact/source/collision/object objectives remain active in subsequent sweeps; "
                "seed smoothness is not final contact or motion acceptance.")
        proposal["limitations"].append(
            "Temporal mode revisits neighboring source frames in the same SOMA objective set; "
            "hinge derivatives are soft costs, with no root-manifold smoothing or hard clearance guarantee.")
    if "hand_target_optimization" in cfg:
        from .soma_hand_targets import validate_hand_target_config
        hand_targets = validate_hand_target_config(cfg["hand_target_optimization"])
        proposal["hand_target_optimization"] = hand_targets
        converter["agentic_hand_target_optimization"] = hand_targets
        proposal["adapter_code"]["soma_hand_targets.py"] = _file(Path(__file__).with_name("soma_hand_targets.py"))
        proposal["limitations"].append(
            "Optional finger-only source-prior smoothing and weight ablation preserve original body/wrist "
            "targets and contact inputs; derivative priors are not the unchanged human observations, "
            "anatomically constrained hand reconstruction, or motion acceptance.")
    return proposal


def _environment(root: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _bound_files(proposal: dict[str, Any]) -> list[dict[str, str]]:
    values = [proposal[key] for key in ("config", "source", "python", "model_description")]
    for key in ("adapter_code", "upstream_code", "robot_assets", "soma_assets", "bindings"):
        values.extend(proposal[key].values())
    values.extend(proposal.get("contact_inputs", {}).values())
    values.extend(proposal.get("robot_hand_target_inputs", {}).values())
    values.extend(proposal.get("pose_keyframe_inputs", {}).values())
    values.extend(proposal.get("wrist_keyframe_inputs", {}).values())
    values.extend(proposal.get("prior_global_seed_inputs", {}).values())
    return values


def _assert_unchanged(proposal: dict[str, Any]) -> None:
    for entry in _bound_files(proposal):
        path = Path(entry["path"])
        if not path.is_file() or _sha(path) != entry["sha256"]:
            raise ValueError(f"Bound input changed: {path}")


def preflight(proposal: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
    """Test the exact external CLI and native model joint order without solving."""
    _assert_unchanged(proposal)
    command = proposal["commands"][0]
    checks = []
    runtime = None
    help_argv = command[:3] + ["--help"]
    script = "\n".join([
        "import importlib.metadata,json,pathlib,newton,soma_retargeter",
        "from soma_retargeter.pipelines.utils import create_robot_builder,get_target_asset_path",
        "from soma_retargeter.utils.newton_utils import get_filtered_joint_names",
        "from soma_retargeter.robotics.robot_registry import register_robots_path",
        f"[register_robots_path(p) for p in {proposal['converter_config'].get('extra_robot_paths', [])!r}]",
        f"target={proposal['robot']!r}",
        "builder=create_robot_builder(target)",
        "data={'module':str(pathlib.Path(soma_retargeter.__file__).resolve()),'model':get_target_asset_path(target),"
        "'joint_names':get_filtered_joint_names(builder,[newton.JointType.REVOLUTE]),"
        "'versions':{n:importlib.metadata.version(n) for n in ('newton','warp-lang','numpy','scipy')}}",
        f"print({_MARKER!r}+json.dumps(data))",
    ])
    for argv in (help_argv, [command[0], "-c", script]):
        try:
            result = subprocess.run(argv, cwd=proposal["cwd"], env=_environment(proposal["cwd"]),
                                    capture_output=True, text=True, timeout=timeout, check=False)
            ok = result.returncode == 0
            diagnostic = result.stderr[-4000:] if not ok else ""
            if argv == help_argv:
                missing = [flag for flag in ("--config", "--viewer", "--device") if flag not in result.stdout]
                ok = ok and not missing
                if missing:
                    diagnostic += " Missing CLI options: " + ", ".join(missing)
            elif ok:
                lines = [line[len(_MARKER):] for line in result.stdout.splitlines() if line.startswith(_MARKER)]
                if len(lines) != 1:
                    raise ValueError("External model probe produced no unique metadata record")
                runtime = json.loads(lines[0])
                expected = Path(proposal["cwd"]) / "soma_retargeter/__init__.py"
                names = runtime.get("joint_names")
                if (Path(runtime["module"]).resolve() != expected.resolve()
                    or Path(runtime["model"][proposal["model_asset_key"]]).resolve() != Path(proposal["model_description"]["path"])
                    or not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
                    or len(names) != len(set(names))):
                    raise ValueError("External SOMA module/model/joint order is not the bound installation")
            checks.append({"kind": "cli" if argv == help_argv else "model", "ok": ok,
                           "returncode": result.returncode, "diagnostic": diagnostic})
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            checks.append({"kind": "cli" if argv == help_argv else "model", "ok": False, "diagnostic": str(exc)})
    _assert_unchanged(proposal)
    return {"schema_version": 1, "ok": all(c["ok"] for c in checks) and not proposal["blockers"],
            "checks": checks, "blockers": proposal["blockers"], "runtime": runtime}


def validate_csv(path: str | Path, joint_names: list[str], frame_count: int) -> dict[str, Any]:
    """Reject truncated, nonfinite or differently ordered robot CSV output."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ROOT_COLUMNS + joint_names:
            raise ValueError("CSV header does not match the bound runtime robot joint order")
        rows = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"CSV column count mismatch at row {rows}")
            values = [float(v) for v in row]
            if not all(math.isfinite(v) for v in values) or values[0] != rows:
                raise ValueError(f"CSV nonfinite value or nonconsecutive source frame at row {rows}")
            rows += 1
    if rows != frame_count:
        raise ValueError(f"CSV frame count {rows} differs from source {frame_count}")
    return {"frame_count": rows, "joint_names": joint_names, "root_translation_units": "cm",
            "root_rotation": "extrinsic_xyz_degrees", "joint_units": "degrees",
            "frame_index_start": 0, "fps_embedded": False}


def _validate_forward_execution(proposal, contact_report):
    """Validate reported forward execution against the bound request and code.

    New synchronous plans require telemetry. Historical plans without the CUDA
    adapter binding and non-synchronous runs retain their old receipt contract.
    This checks execution provenance, not trajectory quality or timing fairness.
    """
    synchronous = proposal.get("temporal_optimization", {}).get("method") == "synchronous_sparse"
    binding = proposal.get("adapter_code", {}).get("soma_forward_cuda.py")
    evidence = contact_report.get("forward_execution")
    if not synchronous or (evidence is None and binding is None):
        return None
    message = "Missing/inconsistent SOMA forward execution evidence"
    if not isinstance(evidence, dict):
        raise ValueError(message)
    command = proposal["commands"][0]
    try:
        requested = command[command.index("--device") + 1]
    except (ValueError, IndexError):
        raise ValueError(message + ": missing requested device") from None
    contract = proposal.get("forward_execution_contract", {})
    if contract.get("requested_device", requested) != requested:
        raise ValueError(message + ": request device differs from contract")
    actual = evidence.get("device")
    if (not isinstance(actual, str) or not re.fullmatch(r"cpu|cuda:\d+", actual)
            or (requested == "cuda" and not actual.startswith("cuda:"))
            or (requested != "cuda" and actual != requested)
            or contact_report.get("device", actual) != actual):
        raise ValueError(message + ": actual device differs from request")
    cuda = actual.startswith("cuda:")
    expected_backend = ("cuda_normal_equations_then_small_cholesky" if cuda else "upstream_newton_lm")
    if evidence.get("backend") != expected_backend:
        raise ValueError(message + ": unexpected forward backend")
    solver = evidence.get("solver")
    if cuda:
        if (not isinstance(binding, dict) or not isinstance(solver, dict)
                or solver.get("schema") != "soma_forward_cuda_normal_equations.v1"
                or solver.get("backend") != expected_backend or solver.get("device") != actual
                or solver.get("adapter_sha256") != binding.get("sha256")
                or not isinstance(solver.get("adapter_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", solver["adapter_sha256"])
                or solver.get("upstream_residuals_jacobian_damping_acceptance_integration_unchanged") is not True
                or type(solver.get("n_batch")) is not int or solver["n_batch"] != 1
                or type(solver.get("n_dofs")) is not int or not 1 <= solver["n_dofs"] <= 64
                or type(solver.get("n_residuals")) is not int or solver["n_residuals"] < 1
                or not isinstance(solver.get("upstream_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", solver["upstream_sha256"])):
            raise ValueError(message + ": unbound or changed CUDA solver")
    elif solver is not None:
        raise ValueError(message + ": CPU forward unexpectedly declares CUDA solver")
    timing_keys = ("pipeline_wall_seconds", "pass_setup_capture_seconds",
                   "python_dispatch_seconds", "python_extra_target_update_seconds")
    for name in timing_keys:
        value = evidence.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(message + ": invalid forward timing")
    for name in ("diagnostic_wall_seconds", "pipeline_execution_wall_seconds"):
        if name in contact_report:
            value = contact_report[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(message + ": invalid diagnostic timing")
    frames = proposal["source_info"]["frame_count"]
    if (contract.get("source_frame_count", frames) != frames
            or contact_report.get("frame_count") != frames):
        raise ValueError(message + ": source frame count mismatch")
    dispatched = evidence.get("dispatched_frames_including_warmup")
    if type(dispatched) is not int or dispatched < frames:
        raise ValueError(message + ": incomplete forward frames")
    warmup = contact_report.get("warmup_frames_without_contacts")
    expected_warmup = contract.get("warmup_frame_count")
    if expected_warmup is not None and warmup != expected_warmup:
        raise ValueError(message + ": warmup differs from bound retarget configuration")
    if warmup is not None and (type(warmup) is not int or warmup < 0 or dispatched != frames + warmup):
        raise ValueError(message + ": inconsistent forward warmup frames")
    return dict(evidence, requested_device=requested, source_frame_count=frames,
        warmup_frame_count=warmup, code_and_device_verified=True, motion_accepted=False)


def _validate_hand_target_result(proposal, contact_report, output):
    """Validate actual derivative arrays and weight evidence, not a config claim."""
    import numpy as np
    from .soma_hand_targets import hand_target_weight_multipliers, smooth_hand_targets
    evidence = contact_report.get("hand_target_optimization")
    message = "Missing/inconsistent SOMA hand-target evidence"
    if not isinstance(evidence, dict) or evidence.get("config") != proposal["hand_target_optimization"]:
        raise ValueError(message)
    path = output / "hand_target_priors.npz"
    if not path.is_file():
        raise ValueError(message + ": derivative arrays missing")
    with np.load(path, allow_pickle=False) as saved:
        required = {"original_targets", "derived_targets", "mapped_joints", "first_source_frame", "fps"}
        if not required <= set(saved.files):
            raise ValueError(message + ": incomplete derivative arrays")
        original, derived = saved["original_targets"], saved["derived_targets"]
        names = saved["mapped_joints"].tolist()
        first, fps = saved["first_source_frame"].item(), saved["fps"].item()
    # Replay this small deterministic CPU transform against its saved input.
    # This verifies the declared source-prior transform, not anatomical quality.
    expected, expected_report = smooth_hand_targets(original, names, fps,
        proposal["hand_target_optimization"], first_source_frame=first)
    position, rotation = hand_target_weight_multipliers(names, proposal["hand_target_optimization"])
    if (derived.dtype != expected.dtype or not np.array_equal(derived, expected)
            or expected_report["source_frame_count"] != proposal["source_info"]["frame_count"]
            or fps != proposal["source_info"]["fps"]
            or any(evidence.get(key) != expected_report[key] for key in (
                "config", "source_frame_count", "first_source_frame", "fps", "original_target_sha256",
                "derived_target_sha256", "actual_targets_changed", "finger_target_count", "finger_names",
                "protected_body_wrist_targets_unchanged", "warmup_targets_unchanged", "contact_targets_modified"))
            or evidence.get("applied_weight_multipliers") != {
                "position": position.tolist(), "rotation": rotation.tolist()}):
        raise ValueError(message)
    return {"arrays": _file(path), "original_target_sha256": expected_report["original_target_sha256"],
            "derived_target_sha256": expected_report["derived_target_sha256"],
            "original_source_priors_unchanged": not expected_report["actual_targets_changed"],
            "original_source_objective_weights_unchanged": bool(np.all(position == 1) and np.all(rotation == 1)),
            "protected_body_wrist_targets_unchanged": True, "warmup_targets_unchanged": True,
            "contact_targets_modified": False, "derivative_transform_verified": True}


def _global_forward_state(proposal, output, global_state):
    """Select independently retained fresh-forward evidence for explicit seeds."""
    if "prior_global_seed" not in proposal.get("temporal_optimization", {}):
        return {key:global_state[key] for key in global_state.files if key.startswith("initial_")}
    import numpy as np
    p = Path(output)/"prior_global_seed_states.npz"
    sync = _read(Path(output)/"synchronous_optimization.json")
    ev = sync.get("prior_global_seed", {})
    if (ev.get("binding") != proposal["temporal_optimization"]["prior_global_seed"]
            or not p.is_file() or ev.get("states_sha256") != _sha(p)
            or ev.get("fresh_forward_executed") is not True):
        raise ValueError("Missing/hash-inconsistent separate fresh-forward seed evidence")
    with np.load(p, allow_pickle=False) as z:
        return {"initial_joint_q":z["fresh_q"].copy(),
                "initial_obj_R":z["fresh_obj_R"].copy(),"initial_obj_t":z["fresh_obj_t"].copy()}


def _validate_robot_hand_target_result(proposal, contact_report, output, schedule):
    """Bind native target execution, forward output and global starting state."""
    import numpy as np
    evidence = contact_report.get("robot_hand_targets")
    message = "Missing/inconsistent native robot hand target execution evidence"
    config = proposal["robot_hand_targets"]
    if (not isinstance(evidence, dict) or evidence.get("config") != config
            or evidence.get("binding") != proposal["robot_hand_target_summary"]
            or evidence.get("active_in_forward") is not True
            or evidence.get("active_in_global") is not ("temporal_optimization" in proposal)
            or evidence.get("source_hand_weight_multiplier") != config["source_hand_weight_multiplier"]
            or evidence.get("original_body_weights_unchanged") is not True):
        raise ValueError(message)
    target = _load_external_robot_hands(config, base=Path(proposal["config"]["path"]).parent,
        model=proposal["model_xml"]["path"], frame_count=proposal["source_info"]["frame_count"],
        fps=proposal["source_info"]["fps"], rotations=schedule["object_reference_rotations"],
        positions=schedule["object_reference_positions"], scale=schedule["object_scale"])
    sides, count, fps = list(target["hands"]), proposal["source_info"]["frame_count"], proposal["source_info"]["fps"]
    source_weights = evidence.get("source_mapping_weights")
    attributes = {"mapped_body_link_pos_data", "mapped_body_link_rot_data"}
    if not isinstance(source_weights, dict) or set(source_weights) != attributes:
        raise ValueError(message + ": missing source weight rows")
    previous_names = None
    for rows in source_weights.values():
        if not isinstance(rows, list) or not rows:
            raise ValueError(message + ": empty source weight rows")
        names = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"source", "original", "effective"}:
                raise ValueError(message + ": invalid source weight row")
            name, before, after = row["source"], row["original"], row["effective"]
            if (not isinstance(name, str) or not name
                    or any(isinstance(v, bool) or not isinstance(v, (int,float))
                           or not math.isfinite(v) or v < 0 for v in (before,after))):
                raise ValueError(message + ": invalid source mapping weight")
            enabled = any(name.startswith(side.capitalize()+"Hand") for side in sides)
            expected = before*(config["source_hand_weight_multiplier"] if enabled else 1.)
            if not math.isclose(after, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(message + ": protected or replacement source weight differs")
            names.append(name)
        if (len(set(names)) != len(names) or (previous_names is not None and names != previous_names)
                or any(not any(name.startswith(side.capitalize()+"Hand") for name in names) for side in sides)):
            raise ValueError(message + ": missing/ambiguous source hand mapping")
        previous_names = names
    seconds = evidence.get("forward_seconds")
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(message + ": invalid forward timing")
    for field in ("forward_observations", "final_observations"):
        rows = evidence.get(field)
        if not isinstance(rows,list) or len(rows) != count*len(sides):
            raise ValueError(message + ": incomplete mounted measurements")
        observed = set()
        for row in rows:
            if not isinstance(row,dict) or type(row.get("frame")) is not int or row.get("side") not in sides:
                raise ValueError(message + ": invalid mounted measurement")
            frame,side = row["frame"],row["side"]
            if not 0 <= frame < count or (frame,side) in observed:
                raise ValueError(message + ": duplicate/out-of-clock mounted measurement")
            observed.add((frame,side))
            if row.get("object_relative") is not bool(target["hands"][side]["contact_mask"][frame]):
                raise ValueError(message + ": incorrect phase coordinate mode")
            keys = ("wrist_translation_m", "wrist_rotation_rad", "finger_rms_rad", "finger_max_rad")
            if any(isinstance(row.get(key),bool) or not isinstance(row.get(key),(int,float))
                   or not math.isfinite(row[key]) or row[key] < 0 for key in keys):
                raise ValueError(message + ": invalid mounted error")
            if row["wrist_rotation_rad"] > math.pi+1e-6 or row["finger_rms_rad"] > row["finger_max_rad"]+1e-8:
                raise ValueError(message + ": inconsistent mounted error")
    checkpoint = output / "forward_stage_states.npz"
    if (evidence.get("forward_checkpoint") != str(checkpoint)
            or not checkpoint.is_file() or evidence.get("forward_checkpoint_sha256") != _sha(checkpoint)):
        raise ValueError(message + ": missing/mutated forward checkpoint")
    with np.load(checkpoint,allow_pickle=False) as saved:
        if set(saved.files) != {"joint_q", "object_rotations", "object_positions", "fps", "timestamps"}:
            raise ValueError(message + ": invalid forward checkpoint fields")
        q, r, p = saved["joint_q"],saved["object_rotations"],saved["object_positions"]
        if (q.ndim != 2 or q.shape[0] != count or q.shape[1] < 8 or not np.isfinite(q).all()
                or r.shape != (count,3,3) or p.shape != (count,3)
                or not np.isfinite(r).all() or not np.isfinite(p).all()
                or not np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-5,rtol=0)
                or not np.allclose(np.linalg.det(r),1.,atol=1e-5,rtol=0)
                or saved["fps"].shape != () or float(saved["fps"]) != fps
                or not np.array_equal(saved["timestamps"],target["timestamps"])):
            raise ValueError(message + ": invalid forward clock/state")
    global_binding = None
    if proposal.get("temporal_optimization",{}).get("method") == "synchronous_sparse":
        global_path = output / "temporal_sequence_states.npz"
        with np.load(global_path,allow_pickle=False) as global_state:
            forward_state = _global_forward_state(proposal, output, global_state)
            if not np.array_equal(forward_state["initial_joint_q"],q):
                raise ValueError(message + ": global solve did not start from saved forward state")
            if "object_optimization" in proposal and (
                    not np.array_equal(forward_state["initial_obj_R"],r)
                    or not np.array_equal(forward_state["initial_obj_t"],p)):
                raise ValueError(message + ": global object seed differs from forward state")
        global_binding = _file(global_path)
    return dict(forward_checkpoint=_file(checkpoint), global_checkpoint=global_binding,
        inputs=proposal["robot_hand_target_inputs"], enabled_hands=sides,
        forward_seconds=float(seconds), observations_per_stage=count*len(sides),
        source_hand_weight_multiplier=config["source_hand_weight_multiplier"],
        active_in_forward=True, active_in_global="temporal_optimization" in proposal,
        original_body_weights_unchanged=True, motion_accepted=False,
        measurement_validation="Schema/clock/finite checks; mounted FK error values are runtime diagnostics.")


def _validate_wrist_keyframe_result(proposal, contact_report, output, schedule):
    import numpy as np
    from .soma_wrist_keyframes import load, measure, measurements_match
    evidence = contact_report.get("wrist_keyframes")
    if (not isinstance(evidence, dict) or evidence.get("active_in_forward") is not True
            or evidence.get("active_in_global") is not ("temporal_optimization" in proposal)
            or evidence.get("binding") != proposal["wrist_keyframe_summary"]):
        raise ValueError("Missing/inconsistent wrist keyframe execution")
    hands = _load_external_robot_hands(proposal["robot_hand_targets"], base=Path(proposal["config"]["path"]).parent,
        model=proposal["model_xml"]["path"], frame_count=proposal["source_info"]["frame_count"],
        fps=proposal["source_info"]["fps"], rotations=schedule["object_reference_rotations"],
        positions=schedule["object_reference_positions"], scale=schedule["object_scale"])
    loaded = load(proposal["wrist_keyframes"], model_xml=proposal["model_xml"]["path"],
        source_bvh=proposal["source"]["path"], frame_count=proposal["source_info"]["frame_count"],
        fps=proposal["source_info"]["fps"], hand_targets=hands)
    path = output / "wrist_keyframe_states.npz"
    if evidence.get("states_path") != str(path) or evidence.get("states_sha256") != _sha(path):
        raise ValueError("Wrist keyframe state evidence changed")
    with np.load(path, allow_pickle=False) as data:
        required = {"hands", "timestamps"} | {stage+"_"+suffix for stage in ("forward", "final")
            for suffix in ("positions", "rotations", "joint_q", "obj_R", "obj_t")}
        if set(data.files) != required or data["hands"].tolist() != loaded["report"]["enabled_hands"]:
            raise ValueError("Invalid wrist keyframe state evidence arrays/hands")
        if not np.array_equal(data["timestamps"], loaded["timestamps"]):
            raise ValueError("Wrist keyframe state clock differs")
        for stage in ("forward", "final"):
            if not measurements_match(measure(loaded, data[stage+"_positions"], data[stage+"_rotations"]), evidence.get(stage)):
                raise ValueError("Wrist keyframe measured error differs from saved states")
        if proposal.get("temporal_optimization", {}).get("method") == "synchronous_sparse":
            with np.load(output/"temporal_sequence_states.npz", allow_pickle=False) as states:
                forward_state = _global_forward_state(proposal, output, states)
                for stage, prefix in (("forward", "initial"), ("final", "refined")):
                    selected = forward_state if stage == "forward" else states
                    if not np.array_equal(data[stage+"_joint_q"], selected[prefix+"_joint_q"]):
                        raise ValueError("Wrist keyframe joint states differ from actual global states")
                    if "object_optimization" in proposal:
                        for field in ("obj_R", "obj_t"):
                            if not np.array_equal(data[stage+"_"+field], selected[prefix+"_"+field]):
                                raise ValueError("Wrist keyframe object states differ from actual global states")
    return dict(evidence, state_file=_file(path), original_hand_targets_retained=True,
                measurement_validation="Saved actual solve states bound to global checkpoints; mounted FK errors are runtime diagnostics.")


def _validate_pose_keyframe_result(proposal, contact_report, output, csv_path):
    import numpy as np
    from .soma_pose_keyframes import measure
    evidence = contact_report.get("pose_keyframes")
    if (not isinstance(evidence, dict) or evidence.get("active_in_forward") is not True
            or evidence.get("active_in_global") is not ("temporal_optimization" in proposal)
            or evidence.get("binding") != proposal["pose_keyframe_summary"]):
        raise ValueError("Missing or inconsistent pose keyframe execution")
    loaded = _load_pose_keys(proposal["pose_keyframes"], base=Path(proposal["config"]["path"]).parent,
        model=proposal["model_xml"]["path"], source=proposal["source"]["path"],
        frame_count=proposal["source_info"]["frame_count"], fps=proposal["source_info"]["fps"])
    path = output / "pose_keyframe_states.npz"
    if evidence.get("states_path") != str(path) or evidence.get("states_sha256") != _sha(path):
        raise ValueError("Pose keyframe state evidence changed")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"forward_hinges", "final_hinges", "joint_names", "timestamps"}:
            raise ValueError("Invalid pose keyframe state evidence arrays")
        if (data["joint_names"].tolist() != loaded["joint_names"]
                or not np.allclose(data["timestamps"], np.arange(loaded["report"]["frame_count"])/loaded["report"]["fps"], atol=1e-8,rtol=0)):
            raise ValueError("Pose keyframe state names or clock differ")
        for stage in ("forward", "final"):
            if measure(loaded, data[stage+"_hinges"]) != evidence.get(stage):
                raise ValueError("Pose keyframe measured error differs from saved state")
        if proposal.get("temporal_optimization", {}).get("method") == "synchronous_sparse":
            with np.load(output/"temporal_sequence_states.npz", allow_pickle=False) as global_state:
                forward_state = _global_forward_state(proposal, output, global_state)
                names = [str(name).rsplit("/", 1)[-1] for name in global_state["joint_names"]]
                if len(names) != len(set(names)):
                    raise ValueError("Ambiguous pose keyframe global hinge names")
                coords = [global_state["coordinate_indices"][names.index(name)] for name in loaded["joint_names"]]
                for stage, field in (("forward", "initial_joint_q"), ("final", "refined_joint_q")):
                    selected = forward_state if stage == "forward" else global_state
                    if not np.array_equal(selected[field][:,coords], data[stage+"_hinges"]):
                        raise ValueError("Pose keyframe states differ from actual global solve states")
        with Path(csv_path).open() as stream:
            rows = list(csv.reader(stream))
        values = np.asarray(rows[1:], dtype=float)
        columns = [rows[0].index(name) for name in loaded["joint_names"]]
        if not np.allclose(np.deg2rad(values[:,columns]), data["final_hinges"], atol=2e-6,rtol=0):
            raise ValueError("Pose keyframe state differs from exported native CSV")
    return dict(binding=evidence["binding"], forward=evidence["forward"], final=evidence["final"],
        states=_file(path), active_in_forward=True, active_in_global=evidence["active_in_global"],
        soft_constraints=True, motion_accepted=False)


def execute(proposal: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    """Stage exactly one bound BVH, run headlessly and validate a new candidate."""
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive and finite")
    if proposal["blockers"]:
        raise ValueError("; ".join(proposal["blockers"]))
    checks = preflight(proposal)
    if not checks["ok"]:
        raise ValueError("External SOMA preflight failed: " + json.dumps(checks))
    _assert_unchanged(proposal)
    output = Path(proposal["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(proposal, status="RUNNING_UNREVIEWED", preflight=checks)
    report_path = output / "run.json"
    save = lambda: report_path.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        (output / "input").mkdir()
        staged_source = output / "input/source.bvh"
        shutil.copyfile(proposal["source"]["path"], staged_source)
        if _sha(staged_source) != proposal["source"]["sha256"]:
            raise ValueError("Staged source differs from bound BVH")
        evidence_bindings = []
        for name, entry in proposal["bindings"].items():
            dest = output / "evidence" / (name + Path(entry["path"]).suffix)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(entry["path"], dest)
            if _sha(dest) != entry["sha256"]:
                raise ValueError(f"Copied evidence differs from bound input: {name}")
            evidence_bindings.append(_file(dest))
        converter_path = output / "converter.json"
        converter_path.write_text(json.dumps(proposal["converter_config"], indent=2) + "\n")
        generated_bindings = [_file(p) for p in (staged_source, converter_path)] + evidence_bindings
        if "contact_inputs" in proposal:
            import numpy as np
            from .soma_contacts import compile_contacts
            schedule, inputs, _ = compile_contacts(proposal["contact_inputs"]["spec"]["path"],
                frame_count=proposal["source_info"]["frame_count"], fps=proposal["source_info"]["fps"],
                model_xml=proposal["model_xml"]["path"], include_object_scene="object_optimization" in proposal,
                allow_empty_moving="robot_hand_targets" in proposal)
            if "robot_hand_targets" in proposal:
                loaded = _load_external_robot_hands(proposal["robot_hand_targets"], base=Path(proposal["config"]["path"]).parent,
                    model=proposal["model_xml"]["path"], frame_count=proposal["source_info"]["frame_count"],
                    fps=proposal["source_info"]["fps"], rotations=schedule["object_reference_rotations"],
                    positions=schedule["object_reference_positions"], scale=schedule["object_scale"])
                if loaded["report"] != proposal["robot_hand_target_summary"]:
                    raise ValueError("Native hand target preparation differs from the bound plan")
                for key, entry in proposal["robot_hand_target_inputs"].items():
                    source = Path(entry["path"])
                    dest = output / "evidence" / ("robot_hand_" + key + source.suffix)
                    dest.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source, dest)
                    if _sha(dest) != entry["sha256"]:
                        raise ValueError("Staged native hand target changed")
                    generated_bindings.append(_file(dest))
            target_path = output / "contact_targets.npz"
            np.savez_compressed(target_path, **schedule)
            generated_bindings.append(_file(target_path))
            for key, path in inputs.items():
                dest = output / "evidence" / ("contact_" + key + path.suffix)
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, dest)
                generated_bindings.append(_file(dest))
            _assert_unchanged(proposal)
        if "pose_keyframe_inputs" in proposal:
            entry = proposal["pose_keyframe_inputs"]["keys"]
            dest = output / "evidence" / "pose_keyframes.json"
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(entry["path"], dest)
            if _sha(dest) != entry["sha256"]:
                raise ValueError("Staged pose keyframe input changed")
            generated_bindings.append(_file(dest))
        for name, entry in proposal.get("wrist_keyframe_inputs", {}).items():
            dest = output / "evidence" / ("wrist_key_"+name+Path(entry["path"]).suffix)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(entry["path"], dest)
            if _sha(dest) != entry["sha256"]:
                raise ValueError("Staged wrist keyframe input changed")
        for name, entry in proposal.get("prior_global_seed_inputs", {}).items():
            dest = output / "evidence" / ("prior_global_"+name+Path(entry["path"]).suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry["path"], dest)
            if _sha(dest) != entry["sha256"]:
                raise ValueError("Staged prior-global seed input changed")
            generated_bindings.append(_file(dest))
        log = output / "conversion.log"
        with log.open("w") as stream:
            result = subprocess.run(proposal["commands"][0], cwd=proposal["cwd"], env=_environment(proposal["cwd"]),
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        report["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"SOMA converter exited {result.returncode}; see {log}")
        csv_path = output / "raw/source.csv"
        if sorted((output / "raw").rglob("*.csv")) != [csv_path]:
            raise ValueError("Expected exactly one CSV for the staged source")
        report["csv_validation"] = validate_csv(csv_path, checks["runtime"]["joint_names"], proposal["source_info"]["frame_count"])
        scaling = _read(output / "scaling.json")
        applied = scaling.get("scalers")
        if (not isinstance(applied, list) or not applied or any(
                not isinstance(record, dict) or record.get("robot") != proposal["robot"]
                or validate_scale(record.get("human_scale_ratio")) != proposal["human_scale_ratio"]
                for record in applied)):
            raise ValueError("Missing/inconsistent effective SOMA scale evidence")
        report["scaling"] = scaling
        if "contact_inputs" in proposal:
            contact_report = _read(output / "contact_initialization.json")
            if (contact_report.get("same_soma_ik_solve") is not True
                    or contact_report.get("measurements_after_joint_limit_clamp") is not True
                    or contact_report.get("frame_count") != proposal["source_info"]["frame_count"]):
                raise ValueError("Missing/inconsistent contact initialization evidence")
            forward_execution = _validate_forward_execution(proposal, contact_report)
            if forward_execution is not None:
                report["forward_execution"] = forward_execution
            if "hand_target_optimization" in proposal:
                report["hand_target_execution"] = _validate_hand_target_result(proposal, contact_report, output)
            if "robot_hand_targets" in proposal:
                report["robot_hand_target_execution"] = _validate_robot_hand_target_result(
                    proposal, contact_report, output, schedule)
            if "pose_keyframes" in proposal:
                report["pose_keyframe_execution"] = _validate_pose_keyframe_result(proposal, contact_report, output, csv_path)
            if "wrist_keyframes" in proposal:
                report["wrist_keyframe_execution"] = _validate_wrist_keyframe_result(proposal, contact_report, output, schedule)
            if "collision_avoidance" in proposal["contact_constraints"]:
                collision_report = contact_report.get("collision_avoidance", {})
                if (collision_report.get("active_in_same_solve") is not True
                        or collision_report.get("failed_queries") != 0
                        or len(collision_report.get("post_clamp", [])) != proposal["source_info"]["frame_count"]):
                    raise ValueError("Missing/inconsistent collision-objective evidence")
            if "temporal_optimization" in proposal:
                temporal = contact_report.get("temporal_optimization", {})
                if (temporal.get("same_soma_objectives") is not True
                        or temporal.get("source_targets_unchanged") is not True
                        or temporal.get("frame_count") != proposal["source_info"]["frame_count"]
                        or temporal.get("config") != proposal["temporal_optimization"]
                        or (proposal["temporal_optimization"].get("method") != "synchronous_sparse"
                            and temporal.get("completed_sweeps", 0) < 1)):
                    raise ValueError("Missing/inconsistent SOMA temporal-objective evidence")
                if proposal["temporal_optimization"].get("method") == "synchronous_sparse":
                    from .soma_synchronous_result import validate_synchronous_result
                    report["synchronous_execution"] = validate_synchronous_result(output, temporal)
                    report["synchronous_csv_binding"] = _validate_synchronous_csv_checkpoint(
                        csv_path, output/"temporal_sequence_states.npz")
                if "sequence_warm_start" in proposal["temporal_optimization"]:
                    report["sequence_warm_start_evidence"] = _validate_sequence_warm_start_result(
                        output, temporal)
                elif temporal.get("sequence_warm_start") is not None:
                    raise ValueError("Undeclared global hand warm-start execution")
            if "object_optimization" in proposal:
                from .soma_object import resolve_object_policy
                from .soma_object_io import validate_object_result
                object_report = contact_report.get("object_optimization", {})
                policy = resolve_object_policy(schedule["object_reference_rotations"],
                    schedule["object_reference_positions"], proposal["object_optimization"])
                joint_sweeps = temporal.get("completed_joint_sweeps")
                object_history = temporal.get("object_history")
                editable_frames = list(range(policy["first_edit_frame"], proposal["source_info"]["frame_count"]))
                precontact_frames = list(range(policy["contact_frame"]))

                def shared_prefix_evidence(row):
                    update = row.get("precontact_update")
                    if not precontact_frames:
                        return update is None
                    return (isinstance(update, dict) and update.get("shared_pose") is True
                            and update.get("frames") == precontact_frames
                            and type(update.get("accepted")) is bool
                            and isinstance(update.get("reason"), str) and bool(update["reason"]))
                if (object_report.get("config") != proposal["object_optimization"]
                        or proposal.get("object_motion_policy") != policy
                        or object_report.get("motion_policy") != policy
                        or temporal.get("object_motion_policy") != policy
                        or object_report.get("audit", {}).get("passed") is not True):
                    raise ValueError("Missing/inconsistent bounded object-optimization evidence")
                synchronous = proposal["temporal_optimization"].get("method") == "synchronous_sparse"
                if synchronous:
                    if (joint_sweeps != 0 or temporal.get("completed_sweeps") != 0
                            or object_history != []
                            or temporal.get("global_problem", {}).get("object_variable_count", -1) != (
                                0 if not policy["optimization_enabled"] else
                                6*(proposal["source_info"]["frame_count"]-policy["contact_frame"]
                                   + int(policy["contact_frame"] > 0)))):
                        raise ValueError("Inconsistent synchronous object variable policy")
                elif (type(joint_sweeps) is not int or not isinstance(object_history, list)
                        or len(object_history) != joint_sweeps
                        or joint_sweeps > temporal["completed_sweeps"]
                        or (policy["optimization_enabled"] and (
                            joint_sweeps != temporal["completed_sweeps"] or joint_sweeps < 1 or any(
                            not isinstance(row, dict) or type(row.get("sweep")) is not int
                            or row["sweep"] != index
                            or not shared_prefix_evidence(row)
                            or not isinstance(row.get("updates"), list)
                            or any(not isinstance(update, dict) or type(update.get("frame")) is not int
                                   for update in row["updates"])
                            or sorted(update["frame"] for update in row["updates"]) != editable_frames
                            for index, row in enumerate(object_history, 1))))
                        or (not policy["optimization_enabled"] and joint_sweeps != 0)):
                    raise ValueError("Missing/inconsistent actual object sweep evidence for resolved motion policy")
                validate_object_result(output / "optimized_object.npz",
                    reference_path=proposal["contact_inputs"]["object_trajectory"]["path"],
                    reference_sha256=proposal["contact_inputs"]["object_trajectory"]["sha256"],
                    config=proposal["object_optimization"], frame_count=proposal["source_info"]["frame_count"],
                    fps=proposal["source_info"]["fps"])
                report["optimized_object"] = _file(output / "optimized_object.npz")
        for entry in generated_bindings:
            if _sha(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("External converter changed its staged source/config")
        _assert_unchanged(proposal)
        model_field = "model_xml" if proposal["model_asset_key"] == "xml_path" else "model_urdf"
        report["native_handoff"] = {"csv_path": str(csv_path), model_field: proposal["model_description"]["path"],
                                    "fps": proposal["source_info"]["fps"], "source_path": proposal["source"]["path"],
                                    "expected_csv_sha256": _sha(csv_path),
                                    "expected_source_sha256": proposal["source"]["sha256"],
                                    "expected_model_sha256": proposal["model_description"]["sha256"],
                                    "expected_model_assets": {entry["path"]: entry["sha256"] for entry in proposal["robot_assets"].values()},
                                    "world_frame": "soma_retargeter_z_up"}
        report["outputs"] = {str(p.relative_to(output)): _file(p) for p in sorted(output.rglob("*"))
                             if p.is_file() and p != report_path}
        report["status"] = "CONVERTED_UNREVIEWED"
        report["bound_inputs_unchanged"] = True
        save()
        return report
    except BaseException as exc:
        report["status"], report["error"] = "FAILED_UNREVIEWED", str(exc)
        try:
            _assert_unchanged(proposal)
            report["bound_inputs_unchanged"] = True
        except ValueError as input_error:
            report["bound_inputs_unchanged"], report["input_error"] = False, str(input_error)
        save()
        raise


def _validate_synchronous_csv_checkpoint(csv_path, checkpoint):
    """Bind the exported original CSV semantics to the actual global result."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    with np.load(checkpoint, allow_pickle=False) as data:
        q = data["refined_joint_q"]
    values = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    if (q.ndim != 2 or q.shape[1] < 8 or values.shape != q.shape
            or not np.isfinite(values).all() or not np.isfinite(q).all()
            or not np.array_equal(values[:, 0], np.arange(len(q)))):
        raise ValueError("CSV and synchronous native timeline/dimensions differ")
    translation = np.max(np.abs(values[:, 1:4]/100.-q[:, :3]))
    joints = np.max(np.abs(np.deg2rad(values[:, 7:])-q[:, 7:]))
    rotation = (Rotation.from_euler("xyz", values[:, 4:7], degrees=True)
                * Rotation.from_quat(q[:, 3:7]).inv()).magnitude().max()
    if translation > 1e-5 or joints > 1e-6 or rotation > 5e-6:
        raise ValueError("Exported CSV differs from accepted synchronous robot state")
    return dict(passed=True, checkpoint_sha256=_sha(Path(checkpoint)),
        csv_sha256=_sha(Path(csv_path)), frame_count=len(q),
        maximum_translation_error_m=float(translation), maximum_hinge_error_rad=float(joints),
        maximum_orientation_error_rad=float(rotation),
        scope="Original upstream float32 centimetre/degree CSV rounding; no trajectory changes")


def _validate_sequence_warm_start_result(output, temporal):
    """Replay the declared initializer without trusting its status strings."""
    import numpy as np
    from .soma_sequence_warm_start import _hash_array, sequence_warm_start
    from .soma_temporal import temporal_metrics

    # Runtime and adapter may use different NumPy/BLAS builds. Only derived
    # float64 diagnostics receive tolerance; saved bytes, discrete metadata,
    # native seeds and checkpoint bindings remain exact. Actual 3.12/NumPy2.4
    # versus 3.11/NumPy1.26 replay differs by <6e-15 rad / <9e-14 energy units.
    def same_measurement(recorded, recomputed):
        if isinstance(recomputed, dict):
            return (isinstance(recorded, dict) and set(recorded) == set(recomputed)
                    and all(same_measurement(recorded[key], value)
                            for key, value in recomputed.items()))
        if isinstance(recomputed, list):
            return (isinstance(recorded, list) and len(recorded) == len(recomputed)
                    and all(same_measurement(a, b) for a, b in zip(recorded, recomputed)))
        if isinstance(recomputed, float):
            return (isinstance(recorded, (float, int)) and not isinstance(recorded, bool)
                    and np.isfinite(recorded) and np.isfinite(recomputed)
                    and bool(np.isclose(recorded, recomputed, rtol=1e-12, atol=1e-11)))
        return type(recorded) is type(recomputed) and recorded == recomputed

    evidence = Path(output) / "hand_sequence_warm_start.npz"
    saved = temporal.get("sequence_warm_start")
    if not isinstance(saved, dict) or saved.get("evidence_sha256") != _sha(evidence):
        raise ValueError("Missing/mismatched global hand warm-start evidence")
    with np.load(evidence, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    config = json.loads(str(arrays["temporal_config_json"].item()))
    names = arrays["joint_names"].tolist()
    if config != temporal["config"] or names != temporal["joint_names"]:
        raise ValueError("Warm-start settings/joint names differ from executed temporal solve")
    seeded, replay, replay_arrays = sequence_warm_start(
        arrays["initial_hinge_q"], float(arrays["fps"]), config, names,
        arrays["lower_limits"], arrays["upper_limits"])
    if saved.get("array_sha256") != {key: _hash_array(value) for key, value in arrays.items()}:
        raise ValueError("Warm-start retained array hashes do not match saved bytes")
    if set(arrays) != set(replay_arrays):
        raise ValueError("Warm-start retained array fields do not exactly replay")
    for key, value in arrays.items():
        expected = replay_arrays[key]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError("Warm-start retained array shape/dtype does not exactly replay")
        if key == "unconstrained_hinge_q":
            matches = np.allclose(value, expected, rtol=1e-12, atol=1e-12, equal_nan=False)
        else:
            matches = np.array_equal(value, expected)
        if not matches:
            raise ValueError("Warm-start retained arrays do not exactly replay within float64 tolerance")
    if any(not same_measurement(saved.get(key), value)
           for key, value in replay.items() if key != "array_sha256"):
        raise ValueError("Warm-start reported measurements do not replay within float64 tolerance")
    coordinates = np.asarray(temporal["coordinate_indices"], dtype=int)
    with np.load(Path(output) / "temporal_sequence_states.npz", allow_pickle=False) as checkpoint:
        initial = checkpoint["initial_joint_q"]
        warm = checkpoint["warm_started_joint_q"]
        if (not np.array_equal(initial[:, coordinates], arrays["initial_hinge_q"])
                or not np.array_equal(warm[:, coordinates], seeded)):
            raise ValueError("Warm-start evidence differs from actual temporal checkpoint")
        editable = set(coordinates[arrays["selected_indices"]].tolist())
        protected = [i for i in range(initial.shape[1]) if i not in editable]
        if not np.array_equal(initial[:, protected], warm[:, protected]):
            raise ValueError("Warm-start changed protected body/root coordinates")
    if (not same_measurement(saved.get("after_seed_temporal_metrics"), temporal_metrics(
            seeded, float(arrays["fps"]), config, joint_names=names))
            or saved.get("source_targets_unchanged") is not True):
        raise ValueError("Warm-start temporal/source execution evidence differs")
    return _file(evidence)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "run"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args(argv)
    try:
        proposal = plan(args.config, args.output)
        result = proposal if args.action == "plan" else preflight(proposal) if args.action == "preflight" else execute(proposal, args.timeout)
        print(json.dumps(result, indent=2))
        return int(args.action == "preflight" and not result["ok"])
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"SOMA adapter: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
