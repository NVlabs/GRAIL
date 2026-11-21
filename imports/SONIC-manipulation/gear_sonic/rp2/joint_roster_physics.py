"""Apply per-slot object physics overrides to a live articulation PhysX view.

Pure torch helper so the logic is unit-testable without Isaac Sim. The motion
command calls :func:`apply_joint_roster_physics_overrides` once after sim start
in joint-roster mode. Supported per-slot fields:

* ``mass`` -- tracked link mass. The link's inertia tensor is scaled by the same
  ratio (authored mass distribution preserved, centre of mass untouched), which
  matches the legacy rigid-object override and the panel-mass ablation copies.
* ``static_friction`` / ``dynamic_friction`` / ``restitution`` -- material of
  the tracked link's collision shapes (optional; only when the view exposes
  per-shape paths).

Every write is read back from the view and verified.
"""

from __future__ import annotations

import math
from typing import Any

import torch

MASS_FIELDS = ("mass",)
MATERIAL_FIELDS = ("static_friction", "dynamic_friction", "restitution")


def apply_joint_roster_physics_overrides(
    view: Any,
    *,
    body_names: list[str],
    tracked_body: str,
    per_env: list[dict],
    num_envs: int,
) -> dict[str, Any] | None:
    """Apply and verify overrides; return a compact report (None when nothing to do)."""
    if not per_env or not any(per_env):
        return None
    if len(per_env) != int(num_envs):
        raise ValueError(
            "Joint-roster physics override count differs from environment count: "
            f"{len(per_env)} vs {num_envs}"
        )
    if tracked_body not in body_names:
        raise ValueError(
            f"Tracked body {tracked_body!r} not among active articulation bodies {body_names}"
        )
    body = list(body_names).index(tracked_body)
    unsupported = sorted(
        {key for physics in per_env for key in physics if key not in MASS_FIELDS + MATERIAL_FIELDS}
    )
    if unsupported:
        raise ValueError(
            "Joint-roster physics overrides support mass/static_friction/dynamic_friction/"
            f"restitution only; got {unsupported}"
        )

    report: dict[str, Any] = {"tracked_body": tracked_body, "mass": None, "material": None}
    mass_env_ids = [i for i, physics in enumerate(per_env) if physics.get("mass") is not None]
    if mass_env_ids:
        masses = view.get_masses().clone()
        inertias_raw = view.get_inertias().clone()
        if masses.shape[0] != num_envs or masses.shape[1] != len(body_names):
            raise ValueError(f"Unexpected articulation mass tensor shape {tuple(masses.shape)}")
        inertias = inertias_raw.reshape(num_envs, len(body_names), -1)
        before = masses[:, body].clone()
        changed_ids = []
        for env_id in mass_env_ids:
            target = float(per_env[env_id]["mass"])
            if not math.isfinite(target) or target <= 0.0:
                raise ValueError(f"Slot {env_id}: override mass must be finite and positive")
            # Cast to the simulator mass dtype BEFORE forming the ratio so a
            # restatement that rounds to the same stored mass leaves the
            # inertia exactly unscaled.
            target_sim = torch.tensor(target, dtype=masses.dtype)
            old_sim = masses[env_id, body].clone()
            if float(old_sim) <= 0.0:
                raise ValueError(f"Slot {env_id}: simulator reports non-positive mass {float(old_sim)}")
            if torch.equal(target_sim, old_sim):
                continue
            ratio = (target_sim.double() / old_sim.double()).to(inertias.dtype)
            masses[env_id, body] = target_sim
            inertias[env_id, body] *= ratio
            changed_ids.append(env_id)
        if changed_ids:
            # Write only the environments whose stored mass actually changes; a
            # pure restatement leaves PhysX untouched.
            indices = torch.tensor(changed_ids, dtype=torch.int32)
            view.set_masses(masses, indices)
            view.set_inertias(inertias.reshape(inertias_raw.shape), indices)
        readback_mass = view.get_masses()
        readback_inertia = view.get_inertias().reshape(num_envs, len(body_names), -1)
        observed = readback_mass[mass_env_ids, body].float()
        expected = masses[mass_env_ids, body].float()
        if not torch.allclose(observed, expected, rtol=1.0e-5, atol=1.0e-7):
            raise RuntimeError("Simulator mass readback differs from the joint-roster physics overrides")
        if not torch.allclose(
            readback_inertia[mass_env_ids, body].float(),
            inertias[mass_env_ids, body].float(),
            rtol=1.0e-4,
            atol=1.0e-9,
        ):
            raise RuntimeError("Simulator inertia readback differs from the mass-scaled joint-roster inertia")
        changed = int((before[mass_env_ids].float() != observed).sum())
        report["mass"] = {
            "envs": len(mass_env_ids),
            "changed_envs": changed,
            "before_min_max": (float(before[mass_env_ids].min()), float(before[mass_env_ids].max())),
            "after_min_max": (float(observed.min()), float(observed.max())),
        }
        report["runtime_tracked_mass"] = readback_mass[:, body].detach().float().cpu()

    material_env_ids = [
        i for i, physics in enumerate(per_env) if any(physics.get(k) is not None for k in MATERIAL_FIELDS)
    ]
    if material_env_ids:
        materials = view.get_material_properties().clone()
        shape_paths = getattr(view, "shape_paths", None)
        if not shape_paths:
            raise RuntimeError("Joint-roster material overrides need per-shape paths from the PhysX view")
        for env_id in material_env_ids:
            physics = per_env[env_id]
            paths = list(shape_paths[env_id]) if env_id < len(shape_paths) else []
            shape_ids = [
                i for i, path in enumerate(paths)
                if f"/{tracked_body}/" in str(path) or str(path).endswith(f"/{tracked_body}")
            ]
            if not shape_ids:
                raise ValueError(f"Slot {env_id}: no collision shapes found for tracked body {tracked_body!r}")
            for column, key in enumerate(MATERIAL_FIELDS):
                value = physics.get(key)
                if value is None:
                    continue
                value = float(value)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"Slot {env_id}: {key} must be finite and non-negative")
                for shape_id in shape_ids:
                    materials[env_id, shape_id, column] = value
        indices = torch.tensor(material_env_ids, dtype=torch.int32)
        view.set_material_properties(materials, indices)
        readback = view.get_material_properties()
        if not torch.allclose(
            readback[material_env_ids].float(), materials[material_env_ids].float(), rtol=1.0e-5, atol=1.0e-7
        ):
            raise RuntimeError("Simulator material readback differs from the joint-roster physics overrides")
        report["material"] = {"envs": len(material_env_ids)}
    return report
