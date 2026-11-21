#!/usr/bin/env python3
"""Wrap rigid RP2 active objects in the normalized active-entity articulation ABI.

The generated physics asset has the same ``frame/panel/mechanism_joint`` names
as the one-DoF articulated packs. Its joint is locked and its reference validity
mask is zero, so one policy can consume rigid and articulated clips without a
second observation, reward, or termination layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any

import joblib
import numpy as np

from rp2_build_active_articulation_training_pack import (
    _define_mesh_link,
    _dump_entry,
    _link,
    _load_entry,
    _pose_velocities,
    _read_obj,
    _safe_name,
    _select_panel,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-motion-lib", type=Path, required=True)
    parser.add_argument("--output-motion-lib", type=Path, required=True)
    parser.add_argument(
        "--motion-list",
        type=Path,
        help="Optional text file with one motion key per line; default is every robot PKL.",
    )
    parser.add_argument(
        "--object-mass", type=float, default=0.1,
        help="Uniform simulated object mass in kilograms (default: user-selected 0.1; not an RP2 measurement).",
    )
    parser.add_argument(
        "--object-physics-overrides",
        type=Path,
        help=(
            "Optional calibrated physics manifest. Exact records or an "
            "unambiguous parent-motion/object-index consensus take precedence "
            "over --object-mass."
        ),
    )
    parser.add_argument(
        "--require-object-physics-match",
        action="store_true",
        help=(
            "Fail when --object-physics-overrides does not resolve a positive "
            "mass for every selected motion."
        ),
    )
    parser.add_argument("--collision-approximation", default="sdf")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_motion_key(motion_key: str) -> str:
    """Recover the pre-split source key used by linked mesh filenames."""
    source = re.sub(r"_(?:boxonly|rigidactive|articactive)_c\d+_f\d+_\d+_o\d+$", "", motion_key)
    return re.sub(r"_T(\d+)$", r"__T\1", source)


def _parent_motion_key(motion_key: str) -> str:
    return re.sub(
        r"_(?:boxonly|rigidactive|articactive)_c\d+_f\d+_\d+_o\d+$",
        "",
        motion_key,
    )


def _load_object_physics_overrides(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON mapping")
    for field in ("default", "objects", "rules"):
        value.setdefault(field, {} if field != "rules" else [])
    if not isinstance(value["default"], dict):
        raise TypeError(f"{path}: default must be a mapping")
    if not isinstance(value["objects"], dict):
        raise TypeError(f"{path}: objects must be a mapping")
    if not isinstance(value["rules"], list):
        raise TypeError(f"{path}: rules must be a list")
    return value


def _positive_mass(value: Any, *, label: str) -> float:
    mass = float(value)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{label}: mass must be positive and finite, got {value!r}")
    return mass


def _rule_matches(rule: dict[str, Any], *labels: str) -> bool:
    contains = rule.get("contains")
    pattern = rule.get("match")
    if contains and any(str(contains) in label for label in labels):
        return True
    if pattern and any(re.search(str(pattern), label) is not None for label in labels):
        return True
    return False


def _resolve_object_mass(
    motion_key: str,
    *,
    object_name: str,
    active_index: int,
    overrides: dict[str, Any] | None,
    fallback_mass: float | None,
    require_override_match: bool,
) -> tuple[float, str]:
    if overrides is not None:
        objects = overrides["objects"]
        exact = objects.get(motion_key)
        if isinstance(exact, dict) and "mass" in exact:
            return (
                _positive_mass(exact["mass"], label=f"{motion_key}.mass"),
                f"physics_override:objects.{motion_key}.mass",
            )

        # The RP2 provider identifies a physical prop consistently by P####.
        # A measured per-prop table can therefore cover every retargeted clip
        # without enumerating thousands of generated clip keys.
        prop_match = re.search(r"P\d{4}", object_name, re.IGNORECASE)
        prop_id = prop_match.group(0).upper() if prop_match else None
        for label in (object_name, prop_id):
            if label is None:
                continue
            measured = objects.get(label)
            if isinstance(measured, dict) and "mass" in measured:
                return (
                    _positive_mass(measured["mass"], label=f"{label}.mass"),
                    f"physics_override:objects.{label}.mass",
                )

        if require_override_match:
            raise ValueError(
                f"{motion_key}: measured mass is missing for {object_name} "
                f"({prop_id or 'no prop ID'}); expected an exact key or prop entry"
            )

        parent = _parent_motion_key(motion_key)
        target_suffix = f"_o{active_index:02d}"
        candidates: list[tuple[str, float]] = []
        for key, value in objects.items():
            if not (
                isinstance(key, str)
                and key.startswith(f"{parent}_actorowned_")
                and key.endswith(target_suffix)
                and isinstance(value, dict)
                and "mass" in value
            ):
                continue
            candidates.append(
                (
                    key,
                    _positive_mass(value["mass"], label=f"{key}.mass"),
                )
            )
        candidate_masses = sorted({mass for _, mass in candidates})
        if len(candidate_masses) == 1:
            return (
                candidate_masses[0],
                (
                    "physics_override:parent_motion_object_index_consensus:"
                    f"{parent}:o{active_index:02d}:{len(candidates)}_records"
                ),
            )
        if len(candidate_masses) > 1:
            examples = ", ".join(
                f"{key}={mass:g}" for key, mass in candidates[:8]
            )
            raise ValueError(
                f"{motion_key}: ambiguous calibrated masses for "
                f"{parent}/o{active_index:02d}: {examples}"
            )

        profile = dict(overrides["default"])
        matched_rules = 0
        for rule in overrides["rules"]:
            if not isinstance(rule, dict) or not _rule_matches(
                rule, motion_key, parent, object_name
            ):
                continue
            matched_rules += 1
            profile.update(
                {
                    key: value
                    for key, value in rule.items()
                    if key not in {"contains", "match"}
                }
            )
        if "mass" in profile:
            source = "default" if matched_rules == 0 else f"{matched_rules}_matching_rules"
            return (
                _positive_mass(profile["mass"], label=f"{motion_key}.mass"),
                f"physics_override:{source}",
            )
    elif require_override_match:
        raise ValueError(
            "--require-object-physics-match requires --object-physics-overrides"
        )

    if fallback_mass is None:
        raise ValueError(
            f"{motion_key}: no measured physics entry or explicit --object-mass"
        )
    return (
        _positive_mass(fallback_mass, label="--object-mass"),
        "user_selected_uniform_simulation_mass",
    )


def _load_motion_metadata(source: Path, motion_key: str) -> dict[str, Any]:
    meta_path = source / "meta" / f"{motion_key}.pkl"
    if not meta_path.is_file():
        return {}
    meta = joblib.load(meta_path)
    if not isinstance(meta, dict):
        return {}
    nested = meta.get(motion_key)
    if isinstance(nested, dict):
        return nested
    return meta


def _metadata_mesh_stems(source: Path, motion_key: str) -> list[str]:
    meta = _load_motion_metadata(source, motion_key)
    stems: list[str] = []
    task_stem = meta.get("full_scene_task_object_stem")
    if isinstance(task_stem, str) and task_stem:
        stems.append(task_stem)
    object_stems = meta.get("full_scene_object_stems")
    task_index = meta.get("full_scene_task_object_index")
    if isinstance(object_stems, (list, tuple)):
        if task_index is not None:
            try:
                index = int(task_index)
                if 0 <= index < len(object_stems):
                    stems.append(str(object_stems[index]))
            except Exception:
                pass
        stems.extend(str(stem) for stem in object_stems if stem)
    out: list[str] = []
    seen: set[str] = set()
    for stem in stems:
        if stem in seen:
            continue
        seen.add(stem)
        out.append(stem)
    return out


def _resolve_active_mesh(source: Path, motion_key: str, object_name: str) -> Path:
    safe_object = _safe_name(object_name)
    metadata = _load_motion_metadata(source, motion_key)
    parent_motion_key = metadata.get("rp2_parent_motion_key")
    if parent_motion_key is not None and (
        not isinstance(parent_motion_key, str)
        or not parent_motion_key
        or Path(parent_motion_key).name != parent_motion_key
    ):
        raise ValueError(
            f"{motion_key}: invalid rp2_parent_motion_key "
            f"{parent_motion_key!r}"
        )

    stems = [
        *_metadata_mesh_stems(source, motion_key),
        f"{motion_key}__{safe_object}",
        f"{_source_motion_key(motion_key)}__{safe_object}",
    ]
    if parent_motion_key:
        # Composed clips may normalize punctuation in their output key.  The
        # persisted parent key is the lossless, motion-local provenance needed
        # to recover the original shared asset stem.
        stems.insert(0, f"{parent_motion_key}__{safe_object}")

    unique_stems = list(dict.fromkeys(stems))
    invalid_stems = [
        stem
        for stem in unique_stems
        if not isinstance(stem, str)
        or not stem
        or Path(stem).name != stem
    ]
    if invalid_stems:
        raise ValueError(
            f"{motion_key}: invalid motion-local mesh stems "
            f"{invalid_stems!r}"
        )

    candidates = [
        source / "object_mesh_obj" / f"{stem}.obj"
        for stem in unique_stems
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve one active mesh for {motion_key}/{object_name}; "
        f"motion_local_candidates={candidates}. Refusing cross-motion "
        "object-name fallback."
    )


def _write_rigid_wrapper_usd(
    path: Path,
    *,
    mesh: tuple[np.ndarray, np.ndarray],
    object_mass: float,
    collision_approximation: str,
    fix_root_link: bool = False,
) -> None:
    """Author a rigid active entity with a locked, semantically invalid joint."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root_path = "/ActiveArticulation"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)

    frame = UsdGeom.Xform.Define(stage, f"{root_path}/frame").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(frame)
    UsdPhysics.MassAPI.Apply(frame).CreateMassAttr(0.01)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    try:
        from pxr import PhysxSchema

        PhysxSchema.PhysxArticulationAPI.Apply(root)
    except ImportError:
        pass
    if fix_root_link:
        root_joint = UsdPhysics.FixedJoint.Define(
            stage, f"{root_path}/root_joint"
        )
        root_joint.CreateBody1Rel().SetTargets([frame.GetPath()])

    panel = _define_mesh_link(
        stage,
        f"{root_path}/panel",
        *mesh,
        np.eye(3),
        np.zeros(3),
        mass=object_mass,
        color=(0.45, 0.55, 0.65),
        collision_approximation=collision_approximation,
    )

    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/mechanism_joint")
    joint.CreateBody0Rel().SetTargets([frame.GetPath()])
    joint.CreateBody1Rel().SetTargets([panel.GetPath()])
    joint.CreateAxisAttr("X")
    # A tiny nonzero interval keeps this joint in the PhysX articulation tensor
    # while making its motion physically negligible.
    joint.CreateLowerLimitAttr(-1.0e-4)
    joint.CreateUpperLimitAttr(1.0e-4)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    identity = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    joint.CreateLocalRot0Attr(identity)
    joint.CreateLocalRot1Attr(identity)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(1.0e6)
    drive.CreateDampingAttr(1.0e4)
    UsdPhysics.FilteredPairsAPI.Apply(frame).CreateFilteredPairsRel().AddTarget(
        panel.GetPath()
    )
    stage.GetRootLayer().Save()


def _motion_keys(source: Path, motion_list: Path | None) -> list[str]:
    if motion_list is None:
        return sorted(path.stem for path in (source / "robot").glob("*.pkl"))
    keys = [
        line.strip()
        for line in motion_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Motion list contains duplicate keys: {motion_list}")
    return keys


def _link_required(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    _link(source, target)


def _copy_auxiliary_files(source: Path, output: Path, motion_key: str) -> None:
    for subdir, suffixes in (
        ("robot", (".pkl",)),
        ("meta", (".pkl",)),
        ("object_usd_grail_paired", (".usd", ".usda")),
        ("static_context_usd_grail_paired", (".usd", ".usda")),
    ):
        matches = [source / subdir / f"{motion_key}{suffix}" for suffix in suffixes]
        existing = next((path for path in matches if path.exists()), None)
        if existing is None:
            raise FileNotFoundError(f"Missing {subdir} entry for {motion_key}: {matches}")
        _link(existing, output / subdir / existing.name)

    exact_bps = source / "bps" / f"{motion_key}.npy"
    _link_required(exact_bps, output / "bps" / exact_bps.name)


def _build_motion(
    source: Path,
    output: Path,
    motion_key: str,
    *,
    object_mass: float,
    object_physics_overrides: dict[str, Any] | None = None,
    require_object_physics_match: bool = False,
    collision_approximation: str,
) -> dict[str, Any]:
    masked = _load_entry(source / "objects_with_masks" / f"{motion_key}.pkl", motion_key)
    objects = _load_entry(source / "objects" / f"{motion_key}.pkl", motion_key)
    names = list(masked.get("object_names", []))
    active_name = str(masked.get("object_name", ""))
    if not active_name or active_name not in names:
        raise ValueError(
            f"{motion_key}: object_name must identify the one active object; "
            f"object_name={active_name!r}, names={names}"
        )
    active_index = names.index(active_name)
    object_mass, object_mass_provenance = _resolve_object_mass(
        motion_key,
        object_name=active_name,
        active_index=active_index,
        overrides=object_physics_overrides,
        fallback_mass=object_mass,
        require_override_match=require_object_physics_match,
    )
    root_pos = np.asarray(masked["root_pos"], dtype=np.float64)[:, active_index]
    root_quat = np.asarray(masked["root_quat"], dtype=np.float64)[:, active_index]
    if "fps" not in masked or "fps" not in objects:
        raise KeyError(
            f"{motion_key}: paired active-object reference is missing fps"
        )
    fps = float(masked["fps"])
    objects_fps = float(objects["fps"])
    if (
        not np.isfinite(fps)
        or fps <= 0.0
        or not np.isfinite(objects_fps)
        or objects_fps <= 0.0
    ):
        raise ValueError(
            f"{motion_key}: invalid active-object fps {fps}/{objects_fps}"
        )
    if abs(fps - objects_fps) > 1.0e-9:
        raise ValueError(
            f"{motion_key}: object/masked FPS mismatch "
            f"{objects_fps} vs {fps}"
        )
    root_lin_vel, root_ang_vel = _pose_velocities(root_pos, root_quat, fps)
    num_frames = len(root_pos)
    active_channels = {
        "active_articulation_root_pos": root_pos.astype(np.float32),
        "active_articulation_root_quat": root_quat.astype(np.float32),
        "active_articulation_root_lin_vel": root_lin_vel,
        "active_articulation_root_ang_vel": root_ang_vel,
        "active_articulation_joint_pos": np.zeros((num_frames, 1), dtype=np.float32),
        "active_articulation_joint_vel": np.zeros((num_frames, 1), dtype=np.float32),
        "active_articulation_joint_valid_mask": np.zeros(
            (num_frames, 1), dtype=np.float32
        ),
        "active_articulation_root_dynamic_mask": np.ones(
            (num_frames, 1), dtype=np.float32
        ),
        "active_entity_tracking_valid": np.ones(
            (num_frames, 1), dtype=np.float32
        ),
    }
    metadata = {
        "kind": "rigid",
        "source_object_name": active_name,
        "tracked_body_name": "panel",
        "tracked_prim_path": "{ENV_REGEX_NS}/ActiveArticulation/panel",
        "reference_joint_names": ["mechanism_joint"],
        "joint_valid": False,
        "fix_root_link": False,
        "object_mass_kg": float(object_mass),
        "object_mass_provenance": object_mass_provenance,
    }
    for source_entry, subdir in (
        (objects, "objects"),
        (masked, "objects_with_masks"),
    ):
        selected = _select_panel(source_entry, active_index)
        if subdir == "objects_with_masks":
            selected["rp2_original_object_names"] = names
            for mask_name in (
                "rp2_original_object_motion_mask",
                "rp2_original_object_interaction_mask",
            ):
                if mask_name in masked:
                    selected[mask_name] = np.asarray(masked[mask_name]).copy()
        selected.update(active_channels)
        selected["active_articulation"] = metadata
        _dump_entry(output / subdir / f"{motion_key}.pkl", motion_key, selected)

    mesh_path = _resolve_active_mesh(source, motion_key, active_name)
    wrapper_path = output / "active_articulation_usd" / f"{motion_key}.usd"
    _write_rigid_wrapper_usd(
        wrapper_path,
        mesh=_read_obj(mesh_path),
        object_mass=object_mass,
        collision_approximation=collision_approximation,
    )
    _copy_auxiliary_files(source, output, motion_key)
    source_usd_list = source / "full_scene_object_usds" / f"{motion_key}.txt"
    usd_paths = [
        Path(line.strip())
        for line in source_usd_list.read_text().splitlines()
        if line.strip()
    ]
    if active_index >= len(usd_paths):
        raise ValueError(
            f"USD list has no entry for active object index {active_index}: {source_usd_list}"
        )
    active_usd_list = output / "full_scene_object_usds" / f"{motion_key}.txt"
    active_usd_list.parent.mkdir(parents=True, exist_ok=True)
    active_usd_list.write_text(str(usd_paths[active_index].resolve()) + "\n")
    return {
        "motion_key": motion_key,
        "active_object": active_name,
        "source_mesh": str(mesh_path.resolve()),
        "active_articulation_usd": str(wrapper_path.resolve()),
        "num_frames": num_frames,
        "fps": fps,
        "joint_valid": False,
        "object_mass_kg": float(object_mass),
        "object_mass_provenance": object_mass_provenance,
    }


def main() -> int:
    args = _parse_args()
    source = args.source_motion_lib.resolve()
    output = args.output_motion_lib.resolve()
    physics_path = (
        args.object_physics_overrides.resolve()
        if args.object_physics_overrides is not None
        else None
    )
    object_physics_overrides = _load_object_physics_overrides(physics_path)
    keys = _motion_keys(source, args.motion_list)
    if not keys:
        raise ValueError(f"No motions selected from {source}")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    basis = source / "bps" / "_basis.npy"
    if basis.exists():
        _link(basis, output / "bps" / basis.name)
    roles = source / "object_roles_rigid.json"
    if roles.is_file():
        shutil.copy2(roles, output / roles.name)
    records = [
        _build_motion(
            source,
            output,
            key,
            object_mass=args.object_mass,
            object_physics_overrides=object_physics_overrides,
            require_object_physics_match=args.require_object_physics_match,
            collision_approximation=args.collision_approximation,
        )
        for key in keys
    ]
    manifest = {
        "schema_version": 1,
        "source_motion_lib": str(source),
        "output_motion_lib": str(output),
        "active_entity_abi": "frame/panel/mechanism_joint-v1",
        "rigid_joint_policy": "locked dummy joint; validity=0",
        "object_physics_overrides": (
            {
                "path": str(physics_path),
                "required_match": bool(args.require_object_physics_match),
            }
            if physics_path is not None
            else None
        ),
        "records": records,
    }
    (output / "active_entity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
