"""Optional original-surface local collision-regression acceptance checks.

Every original human vertex is checked at every frame against every original
closed object component. Existing seed collisions are documented, not validated.
Point occupancy cannot detect all triangle crossings or certify physical motion.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np


def _validate_guidance_margin_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {"policy", "fraction"}:
        raise ValueError("guidance_margin_policy requires explicit policy and fraction")
    if policy["policy"] != "detached_slack_limited":
        raise ValueError("Unsupported collision guidance margin policy")
    fraction = policy["fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction < 1
    ):
        raise ValueError(
            "Collision guidance fraction must be finite and strictly between zero and one"
        )


def validate_collision_guard_config(config):
    if config is None:
        return None
    keys = {
        "schema_version",
        "frame_count",
        "human_vertex_count",
        "reference_policy",
        "penetration_floor_m",
        "numerical_margin_m",
        "query_scale",
    }
    if (
        not isinstance(config, dict)
        or not keys <= set(config)
        or set(config) - keys - {"proposal_projection"}
    ):
        raise ValueError(
            f"Collision guard requires {sorted(keys)} and optional proposal_projection"
        )
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported collision guard schema")
    for name in ("frame_count", "human_vertex_count"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be positive integer")
    if config["reference_policy"] != "first_stage_seeded":
        raise ValueError("Collision reference must be first_stage_seeded")
    for name, upper in [
        ("penetration_floor_m", 0.001),
        ("numerical_margin_m", 0.0001),
        ("query_scale", 10000.0),
    ]:
        value = config[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite numeric")
        lower = 1.0 if name == "query_scale" else 0.0
        if not lower <= value <= upper:
            raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    if "proposal_projection" in config:
        projection = config["proposal_projection"]
        projection_keys = {"max_active_constraints", "activation_margin_m", "safety_margin_m"}
        if (
            not isinstance(projection, dict)
            or not projection_keys <= set(projection)
            or set(projection) - projection_keys - {"working_set", "guidance_margin_policy"}
        ):
            raise ValueError(
                "Collision proposal_projection requires explicit row budget and margins"
            )
        rows = projection["max_active_constraints"]
        if type(rows) is not int or not 1 <= rows <= 2048:
            raise ValueError("Collision proposal row budget must be integer1..2048")
        for key in ("activation_margin_m", "safety_margin_m"):
            value = projection[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 0.001
            ):
                raise ValueError(f"{key} must be finite0..0.001metres")
        if "guidance_margin_policy" in projection:
            _validate_guidance_margin_policy(projection["guidance_margin_policy"])
        if "working_set" in projection:
            working = projection["working_set"]
            if not isinstance(working, dict) or set(working) != {
                "max_rechecks",
                "max_total_constraints",
                "max_new_constraints_per_recheck",
                "conditional_radial_shrink",
            }:
                raise ValueError(
                    "Collision working_set requires explicit recheck/total/addition bounds and conditional_radial_shrink"
                )
            for key, lower, upper in (
                ("max_rechecks", 1, 8),
                ("max_total_constraints", rows, 4096),
                ("max_new_constraints_per_recheck", 1, 512),
            ):
                value = working[key]
                if type(value) is not int or not lower <= value <= upper:
                    raise ValueError(f"{key} must be integer {lower}..{upper}")
            if working["max_new_constraints_per_recheck"] > working["max_total_constraints"]:
                raise ValueError("Collision per-recheck additions exceed total row budget")
            if type(working["conditional_radial_shrink"]) is not bool:
                raise ValueError("conditional_radial_shrink must be boolean")
    return copy.deepcopy(config)


class LocalCollisionGuard:
    def __init__(self, config, reference_depth_m):
        self.config = validate_collision_guard_config(config)
        if self.config is None:
            raise ValueError("An explicit collision config is required")
        ref = np.asarray(reference_depth_m, dtype=np.float64)
        shape = (self.config["frame_count"], self.config["human_vertex_count"])
        if (
            ref.ndim != 3
            or ref.shape[0] < 1
            or ref.shape[1:] != shape
            or not np.isfinite(ref).all()
            or (ref < 0).any()
        ):
            raise ValueError(
                "Reference must be finite nonnegative depth [component,frame,all_original_vertices]"
            )
        self.reference_depth_m = ref.copy()
        self.allowance_m = (
            np.maximum(ref, self.config["penetration_floor_m"]) + self.config["numerical_margin_m"]
        )
        self.reference_depth_m.setflags(write=False)
        self.allowance_m.setflags(write=False)

    def arrays(self):
        return {
            "reference_depth_m": self.reference_depth_m.copy(),
            "allowance_m": self.allowance_m.copy(),
        }

    def evaluate(self, candidate_depth_m):
        depth = np.asarray(candidate_depth_m, dtype=np.float64)
        if (
            depth.shape != self.reference_depth_m.shape
            or not np.isfinite(depth).all()
            or (depth < 0).any()
        ):
            return dict(feasible=False, violations=["collision_nonfinite_or_shape"], summaries={})
        excess = depth - self.allowance_m
        violated = excess > 0
        index = np.unravel_index(int(np.argmax(excess)), excess.shape)
        c, f, v = map(int, index)
        summaries = {
            "component_count": depth.shape[0],
            "frame_count": depth.shape[1],
            "human_vertex_count": depth.shape[2],
            "violation_count": int(violated.sum()),
            "violating_frame_count": int(violated.any(axis=(0, 2)).sum()),
            "maximum_excess_m": float(excess[index]),
            "maximum_depth_m": float(depth.max()),
            "reference_maximum_depth_m": float(self.reference_depth_m.max()),
            "reference_positive_component_vertex_frames": int((self.reference_depth_m > 0).sum()),
            "worst": {
                "component": c,
                "frame": f,
                "human_vertex_id": v,
                "depth_m": float(depth[index]),
                "reference_depth_m": float(self.reference_depth_m[index]),
                "allowance_m": float(self.allowance_m[index]),
            },
            "scope": "all original human vertices; separate original closed object components; every source frame",
            "limitation": "Point occupancy only; seed intersections remain unvalidated; no complete mesh or physical validation",
        }
        return dict(
            feasible=not bool(violated.any()),
            violations=["local_collision_regression"] if violated.any() else [],
            summaries=summaries,
        )


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        from .hoi_feasibility import FeasibilityRuntimeBudget

        raise FeasibilityRuntimeBudget(
            "Original-surface collision check exceeded shared fitting budget"
        )


def original_component_depths(human_vertices, object_R, object_t, mesh, *, deadline=None):
    """Return nonnegative [component,frame,vertex] depths, retaining original IDs.

    Safe AABB broadphase skips only points outside the complete object's bounds.
    All remaining points query every original shell; no contact/hand subset is used.
    """
    import torch

    _check_deadline(deadline)
    h, r, t = human_vertices, object_R, object_t
    if not all(torch.is_tensor(x) for x in (h, r, t)):
        raise ValueError("Native geometry must be Torch tensors")
    if (
        h.ndim != 3
        or h.shape[-1] != 3
        or min(h.shape[:2]) < 1
        or r.shape != (len(h), 3, 3)
        or t.shape != (len(h), 3)
    ):
        raise ValueError("Native geometry shape mismatch")
    if not all(torch.isfinite(x).all() for x in (h, r, t)):
        raise ValueError("Native collision geometry must be finite")
    if not all(
        x.dtype in (torch.float32, torch.float64) and x.device == h.device and x.dtype == h.dtype
        for x in (h, r, t)
    ):
        raise ValueError("Native collision geometry dtype/device mismatch")
    identity = torch.eye(3, dtype=r.dtype, device=r.device)
    if not torch.allclose(
        r @ r.transpose(-1, -2), identity.expand_as(r), atol=1e-5, rtol=0
    ) or not torch.allclose(
        torch.linalg.det(r), torch.ones(len(r), device=r.device, dtype=r.dtype), atol=1e-5, rtol=0
    ):
        raise ValueError("Object rotations must be proper rigid matrices")
    count = len(mesh.component_face_ids)
    if count < 1:
        raise ValueError("Original object must have at least one closed component")
    with torch.no_grad():
        local = (h - t[:, None, :]) @ r
        verts = torch.as_tensor(mesh.vertices, dtype=h.dtype, device=h.device)
        # Expand broadphase by a few float32 ULPs, never contract the original box.
        pad = torch.finfo(h.dtype).eps * (verts.abs().max() + 1) * 8
        inside_box = (
            (local >= verts.min(0).values - pad) & (local <= verts.max(0).values + pad)
        ).all(-1)
        depth = torch.zeros((count, *h.shape[:2]), device=h.device, dtype=h.dtype)
        if inside_box.any():
            values = mesh.query(local[inside_box], return_details=True)[
                "component_signed_distance_m"
            ]
            if values.shape != (count, int(inside_box.sum())) or not torch.isfinite(values).all():
                raise ValueError("Original component query returned invalid full-component depths")
            depth[:, inside_box] = torch.clamp(-values, min=0)
        result = depth.detach().cpu().numpy()
    _check_deadline(deadline)
    return result


def audit_native_collision(optimizer, data, output, reread, human_vertices):
    """Rebuild original geometry checker using only saved native/reference arrays."""
    import json
    from pathlib import Path

    import joblib
    import torch

    from grail.core.bindings import bind, loaded_model_hashes, sha256

    from .hoi_original_mesh import OriginalMeshPenetration

    binding = optimizer.human_collision_reference
    for name in ("native", "arrays", "config"):
        if sha256(binding[name]["path"]) != binding[name]["sha256"]:
            raise ValueError(f"Collision reference {name} changed during optimization")
    reference = joblib.load(binding["native"]["path"])
    config = optimizer.cfg["human_collision_guard"]
    with np.load(binding["arrays"]["path"]) as stored:
        guard = LocalCollisionGuard(config, stored["reference_depth_m"])
        checks = {
            "config_exact": json.loads(Path(binding["config"]["path"]).read_text()) == config,
            "object_scale_exact": np.array_equal(
                reread["obj_data"]["obj_scale"], reference["obj_data"]["obj_scale"]
            )
            and np.array_equal(
                reread["obj_data"]["obj_scale"], data.obj.scale.detach().cpu().numpy()
            ),
            "allowance_exact": np.array_equal(guard.allowance_m, stored["allowance_m"]),
            "object_vertices_exact": np.array_equal(
                stored["object_vertices"], data.obj.verts.detach().cpu().numpy()
            ),
            "object_faces_exact": np.array_equal(
                stored["object_faces"], data.obj.faces.detach().cpu().numpy()
            ),
            "human_faces_exact": np.array_equal(
                stored["human_faces"], data.human.faces.detach().cpu().numpy()
            ),
            "loaded_models_unchanged": binding["loaded_model_hashes"]
            == loaded_model_hashes(optimizer.human_model),
        }
        mesh = OriginalMeshPenetration(
            stored["object_vertices"],
            stored["object_faces"],
            device=str(human_vertices.device),
            query_scale=config["query_scale"],
        )
    checks["component_topology_exact"] = mesh.topology == binding["object_topology"]
    with torch.no_grad():
        reference_human = {
            key: (
                torch.as_tensor(value, device=optimizer.device)
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in reference["human_data"].items()
        }
        reference_vertices, _ = optimizer.human_model.generate_mesh(
            reference_human, require_grad=False
        )

        def measure(vertices, native):
            obj = native["obj_data"]
            r = torch.as_tensor(obj["obj_R"], dtype=vertices.dtype, device=vertices.device)
            t = torch.as_tensor(obj["obj_t"], dtype=vertices.dtype, device=vertices.device)
            return original_component_depths(vertices, r, t, mesh)

        reference_depth = measure(reference_vertices, reference)
        depth = measure(human_vertices, reread)
    checks["native_reference_depth"] = bool(
        np.allclose(
            reference_depth, guard.reference_depth_m, rtol=0, atol=config["numerical_margin_m"]
        )
    )
    report = guard.evaluate(depth)
    arrays_path = output / "human_collision_native_arrays.npz"
    np.savez_compressed(
        arrays_path,
        component_depth_m=depth,
        component_excess_m=depth - guard.allowance_m,
        recomputed_reference_depth_m=reference_depth,
    )
    report.update(
        reference_bindings=copy.deepcopy(binding),
        reference_roundtrip_checks=checks,
        all_frame_diagnostics=bind(arrays_path),
        maximum_reference_roundtrip_error_m=float(
            np.abs(reference_depth - guard.reference_depth_m).max()
        ),
    )
    return report


def select_collision_rows(config, old_depth, proposed_depth, allowance):
    """Bound backward work to most threatened original C/F/V rows.

    The full nonlinear check retains every row. Unchanged or relieving seed
    intersections need not consume backward calls merely because they exist.
    Selection is deterministic; omissions never grant an acceptance exemption.
    """
    values = [np.asarray(x, dtype=np.float64) for x in (old_depth, proposed_depth, allowance)]
    old, raw, caps = values
    if (
        old.ndim != 3
        or min(old.shape) < 1
        or any(x.shape != old.shape or not np.isfinite(x).all() or (x < 0).any() for x in values)
    ):
        raise ValueError(
            "Collision proposal depths/caps must be finite nonnegative matching C/F/V arrays"
        )
    # Reuse the strict projection schema without imposing any fixture topology.
    validate_collision_guard_config(
        dict(
            schema_version=1,
            frame_count=old.shape[1],
            human_vertex_count=old.shape[2],
            reference_policy="first_stage_seeded",
            penetration_floor_m=0.0,
            numerical_margin_m=0.0,
            query_scale=1000.0,
            proposal_projection=config,
        )
    )
    growth = raw - old
    raw_excess = raw - caps
    score = np.maximum(old, raw) - caps + config["activation_margin_m"]
    active = (score >= 0) & (growth > 0)
    ids = np.argwhere(active)
    if len(ids):
        c, f, v = ids.T
        order = np.lexsort(
            (
                v,
                f,
                c,
                -growth[c, f, v],
                -raw_excess[c, f, v],
                -(raw_excess[c, f, v] > 0).astype(int),
            )
        )
        ids = ids[order]
    selected = ids[: config["max_active_constraints"]]
    omitted = ids[config["max_active_constraints"] :]
    rows = []
    for c, f, v in omitted:
        rows.append(
            dict(
                kind="collision",
                component=int(c),
                frame=int(f),
                human_vertex_id=int(v),
                raw_excess_m=float(raw_excess[c, f, v]),
                growth_m=float(growth[c, f, v]),
            )
        )
    report = dict(
        available=int(old.size),
        currently_binding=int(np.count_nonzero(old >= caps - config["activation_margin_m"])),
        positive_current_component_points=int(np.count_nonzero(old > 0)),
        candidates=len(ids),
        selected=len(selected),
        omitted=len(omitted),
        budget=config["max_active_constraints"],
        raw_proposal_violated=int(np.count_nonzero(raw_excess > 0)),
        omitted_raw_proposal_violated=(
            int(np.count_nonzero(raw_excess[tuple(omitted.T)] > 0)) if len(omitted) else 0
        ),
        unchanged_or_relieving_near_rows=int(np.count_nonzero((score >= 0) & (growth <= 0))),
        selection="raw violated rows first, descending raw excess, growth, then stable component/frame/original vertex IDs",
        omitted_candidate_constraints=rows,
    )
    return selected.astype(np.int64), report


def select_collision_working_set_rows(config, proposed_depth, allowance, existing_rows):
    """Add newly violated original IDs, including raw-relieving points.

    A shaped proposal is queried on the full field. Unlike initial selection,
    this check has no growth/activation predicate. Existing IDs are never added
    twice. Omitted rows remain subject to full nonlinear acceptance.
    """
    depth, caps = [np.asarray(value, dtype=np.float64) for value in (proposed_depth, allowance)]
    if (
        depth.ndim != 3
        or min(depth.shape) < 1
        or caps.shape != depth.shape
        or any(not np.isfinite(value).all() or (value < 0).any() for value in (depth, caps))
    ):
        raise ValueError("Working-set depths/caps require finite nonnegative matching C/F/V arrays")
    validate_collision_guard_config(
        dict(
            schema_version=1,
            frame_count=depth.shape[1],
            human_vertex_count=depth.shape[2],
            reference_policy="first_stage_seeded",
            penetration_floor_m=0.0,
            numerical_margin_m=0.0,
            query_scale=1000.0,
            proposal_projection=config,
        )
    )
    if "working_set" not in config:
        raise ValueError("Explicit collision working_set required")
    rows = np.asarray(existing_rows)
    if (
        rows.ndim != 2
        or rows.shape[1] != 3
        or not np.issubdtype(rows.dtype, np.integer)
        or np.issubdtype(rows.dtype, np.bool_)
    ):
        raise ValueError("Existing collision rows must be integer [N,3]")
    if len(rows) and ((rows < 0).any() or (rows >= np.asarray(depth.shape)).any()):
        raise ValueError("Existing collision row outside original C/F/V field")
    if len(np.unique(rows, axis=0)) != len(rows):
        raise ValueError("Duplicate existing collision row IDs")
    working = config["working_set"]
    if len(rows) > working["max_total_constraints"]:
        raise ValueError("Existing rows exceed working-set budget")
    excess = depth - caps
    violated = excess > 0
    new = violated.copy()
    if len(rows):
        new[tuple(rows.T)] = False
    ids = np.argwhere(new)
    if len(ids):
        c, f, v = ids.T
        ids = ids[np.lexsort((v, f, c, -excess[c, f, v]))]
    budget = min(
        working["max_new_constraints_per_recheck"], working["max_total_constraints"] - len(rows)
    )
    selected = ids[:budget]
    return selected.astype(np.int64), dict(
        full_component_vertex_frames=int(depth.size),
        all_violated=int(violated.sum()),
        existing_violated=int(violated[tuple(rows.T)].sum()) if len(rows) else 0,
        newly_violated=len(ids),
        selected=len(selected),
        omitted=len(ids) - len(selected),
        existing_rows=len(rows),
        available_addition_budget=budget,
        maximum_excess_m=float(excess.max()),
        selection="positive shaped-proposal excess descending, then stable original component/frame/vertex IDs; no raw-growth predicate",
    )


def detached_slack_limited_collision_values(
    signed_depth_m, allowance_m, *, safety_margin_m, fraction
):
    """Keep the old-state collision-guidance origin feasible when it really is.

    The immutable allowance remains a float64 value. Only the inward guidance
    margin is reduced; actual positive violations are never clipped away. The
    margin is detached so the derivative remains that of the signed depth.
    This selected-row calculation cannot certify the complete native field.
    """
    import torch

    _validate_guidance_margin_policy(dict(policy="detached_slack_limited", fraction=fraction))
    if (
        isinstance(safety_margin_m, bool)
        or not isinstance(safety_margin_m, (int, float))
        or not math.isfinite(safety_margin_m)
        or not 0 <= safety_margin_m <= 0.001
    ):
        raise ValueError("Collision guidance margin must be finite0..0.001metres")
    if (
        not torch.is_tensor(signed_depth_m)
        or signed_depth_m.ndim != 1
        or signed_depth_m.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(signed_depth_m).all()
    ):
        raise ValueError("Signed collision depths must be a finite float32/64 vector")
    caps_array = np.asarray(allowance_m)
    if (
        not np.issubdtype(caps_array.dtype, np.number)
        or np.iscomplexobj(caps_array)
        or caps_array.shape != tuple(signed_depth_m.shape)
        or not np.isfinite(caps_array).all()
        or (caps_array < 0).any()
    ):
        raise ValueError(
            "Collision guidance allowances must be matching finite nonnegative numeric values"
        )
    depth = signed_depth_m.to(dtype=torch.float64)
    # Copy, preserving the fixed float64 allowance and never changing the guard.
    caps = torch.tensor(
        np.array(caps_array, dtype=np.float64, copy=True), dtype=torch.float64, device=depth.device
    )
    slack = caps - depth.detach().clamp(min=0)
    effective = torch.minimum(
        depth.new_full(depth.shape, safety_margin_m), fraction * slack.clamp(min=0)
    ).detach()
    values = depth - caps + effective
    if not torch.isfinite(values).all() or not torch.isfinite(slack).all():
        raise ValueError("Nonfinite collision guidance residual or slack")
    details = {
        "signed_current_depth_m": depth.detach().cpu().numpy().copy(),
        "requested_guidance_margin_m": np.full(
            tuple(depth.shape), safety_margin_m, dtype=np.float64
        ),
        "effective_guidance_margin_m": effective.cpu().numpy().copy(),
        "old_occupancy_slack_m": slack.cpu().numpy().copy(),
    }
    return values, details


def collision_row_values(
    human_vertices,
    object_R,
    object_t,
    mesh,
    rows,
    allowance,
    *,
    safety_margin_m=0.0,
    guidance_margin_policy=None,
    return_guidance=False,
):
    """Exact old-state original closest-feature signed rows through native tensors.

    No detached human coordinates or alternate body parameterization. Nearest
    triangle and winding sign are held piecewise constant by the mesh query;
    object inverse-pose gradients and human coordinate gradients remain live.
    """
    import torch

    if (
        isinstance(safety_margin_m, bool)
        or not isinstance(safety_margin_m, (int, float))
        or not math.isfinite(safety_margin_m)
        or not 0 <= safety_margin_m <= 0.001
    ):
        raise ValueError("Collision guidance margin must be finite0..0.001metres")
    if guidance_margin_policy is not None:
        _validate_guidance_margin_policy(guidance_margin_policy)
    if type(return_guidance) is not bool:
        raise ValueError("return_guidance must be boolean")
    rows = np.asarray(rows)
    caps = np.asarray(allowance)
    if (
        rows.ndim != 2
        or rows.shape[1] != 3
        or not np.issubdtype(rows.dtype, np.integer)
        or np.issubdtype(rows.dtype, np.bool_)
    ):
        raise ValueError("Collision row IDs must be integer [N,3] component/frame/vertex")
    shape = (len(mesh.component_face_ids), *human_vertices.shape[:2])
    if caps.shape != shape or not np.isfinite(caps).all() or (caps < 0).any():
        raise ValueError("Collision proposal allowance shape mismatch")
    if len(rows) and ((rows < 0).any() or (rows >= np.asarray(shape)).any()):
        raise ValueError("Collision proposal row outside original topology/timeline/components")
    if (
        human_vertices.ndim != 3
        or human_vertices.shape[-1] != 3
        or object_R.shape != (len(human_vertices), 3, 3)
        or object_t.shape != (len(human_vertices), 3)
    ):
        raise ValueError("Collision proposal native geometry shape mismatch")
    if not all(torch.isfinite(x).all() for x in (human_vertices, object_R, object_t)):
        raise ValueError("Nonfinite collision proposal native geometry")
    if not len(rows):
        values = human_vertices.reshape(-1)[:0]
        details = {}
        if guidance_margin_policy is not None:
            values, details = detached_slack_limited_collision_values(
                values,
                np.empty(0, dtype=np.float64),
                safety_margin_m=safety_margin_m,
                fraction=guidance_margin_policy["fraction"],
            )
        return (values, details) if return_guidance else values
    c, f, v = rows.T
    ft = torch.as_tensor(f, dtype=torch.long, device=human_vertices.device)
    vt = torch.as_tensor(v, dtype=torch.long, device=human_vertices.device)
    local = torch.bmm((human_vertices[ft, vt] - object_t[ft]).unsqueeze(1), object_R[ft]).squeeze(1)
    fields = mesh.query(local, return_details=True)["component_signed_distance_m"]
    if fields.shape != (shape[0], len(rows)) or not torch.isfinite(fields).all():
        raise ValueError("Invalid differentiable original component field")
    ct = torch.as_tensor(c, dtype=torch.long, device=fields.device)
    signed = fields[ct, torch.arange(len(rows), device=fields.device)]
    if guidance_margin_policy is not None:
        values, details = detached_slack_limited_collision_values(
            -signed,
            caps[c, f, v],
            safety_margin_m=safety_margin_m,
            fraction=guidance_margin_policy["fraction"],
        )
    else:
        # Preserve legacy dtype and operation order exactly when policy absent.
        values, details = -signed - signed.new_tensor(caps[c, f, v]) + safety_margin_m, {}
    return (values, details) if return_guidance else values


def collision_row_jacobians(values, active_parameters, *, batch_size=8, deadline=None):
    """Bounded batched VJPs; retain real native-model and spline derivatives."""
    import torch

    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("Collision VJP batch_size must be integer1..32")
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("Collision row values must be finite vector")
    parameters = list(active_parameters)
    if not parameters:
        raise ValueError("Collision Jacobians require active parameters")
    rows = []
    for start in range(0, len(values), batch_size):
        _check_deadline(deadline)
        stop = min(start + batch_size, len(values))
        if values.requires_grad:
            seeds = torch.zeros(
                (stop - start, len(values)), dtype=values.dtype, device=values.device
            )
            seeds[
                torch.arange(stop - start, device=values.device),
                torch.arange(start, stop, device=values.device),
            ] = 1
            gradients = torch.autograd.grad(
                values,
                parameters,
                grad_outputs=seeds,
                retain_graph=True,
                allow_unused=True,
                is_grads_batched=True,
            )
        else:
            gradients = [None] * len(parameters)
        row = np.concatenate(
            [
                (
                    np.zeros((stop - start, parameter.numel()), dtype=np.float64)
                    if gradient is None
                    else gradient.detach()
                    .cpu()
                    .numpy()
                    .reshape(stop - start, -1)
                    .astype(np.float64)
                )
                for parameter, gradient in zip(parameters, gradients)
            ],
            axis=1,
        )
        if not np.isfinite(row).all():
            raise ValueError("Nonfinite native collision Jacobian")
        rows.append(row)
        _check_deadline(deadline)
    return (
        np.concatenate(rows, axis=0)
        if rows
        else np.empty((0, sum(p.numel() for p in parameters)), dtype=np.float64)
    )
