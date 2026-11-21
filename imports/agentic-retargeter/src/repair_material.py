"""Optional tensor proposal for fixed material contacts and mesh clearance.

This bounded optimizer never exports or accepts a motion. The caller must run
the independent original-triangle, source, temporal and native-field audit.
Torch/Warp are imported only when an explicit material solve is requested.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Integral, Real
import re
import time

import numpy as np


class MaterialRepairCancelled(RuntimeError):
    """A requested cancellation produces no candidate for adoption."""


@dataclass(frozen=True)
class MaterialOptions:
    device: str = "cpu"
    steps: int = 180
    lr: float = 0.003
    fit_substeps: int = 2
    penalty: float = 100.0
    smooth: float = 1.0
    penetration_margin_m: float = 0.0007
    object_geoms: tuple[str, ...] | None = None
    progress_every: int = 40

    def __post_init__(self):
        if not isinstance(self.device, str) or not re.fullmatch(
            r"cpu|cuda:[0-9]+", self.device
        ):
            raise ValueError("Material device must explicitly be cpu or cuda:N")
        for key, maximum in (
            ("steps", 10000),
            ("fit_substeps", 8),
            ("progress_every", 10000),
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"Invalid material {key}")
            object.__setattr__(self, key, int(value))
        for key in ("lr", "penalty", "smooth", "penetration_margin_m"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Material {key} must be finite and nonnegative")
            object.__setattr__(self, key, float(value))
        if self.lr == 0 or self.penetration_margin_m > 0.01:
            raise ValueError(
                "Material learning rate must be positive and margin at most 1cm"
            )
        if self.object_geoms is not None:
            if (
                not isinstance(self.object_geoms, (list, tuple))
                or not self.object_geoms
            ):
                raise ValueError(
                    "object_geoms must be a nonempty list/tuple of original mesh IDs"
                )
            if any(not isinstance(key, str) for key in self.object_geoms) or len(
                set(self.object_geoms)
            ) != len(self.object_geoms):
                raise ValueError("object_geoms must contain unique original mesh IDs")
            object.__setattr__(self, "object_geoms", tuple(self.object_geoms))

    def asdict(self):
        return asdict(self)

    def validate_context(self, context):
        """Validate scope/policy even when an initial audit needs no repair."""
        chosen, _, _ = _geometry_scope(context, self.object_geoms)
        clearance = context.config.get("solver", {}).get("clearance_weight", 2000.0)
        if (
            isinstance(clearance, bool)
            or not isinstance(clearance, Real)
            or not np.isfinite(clearance)
            or clearance <= 0
        ):
            raise ValueError(
                "Material objective normalization requires positive clearance_weight"
            )
        if not chosen and not context.contacts:
            raise ValueError(
                "Material repair needs fixed contacts or declared robot-object meshes"
            )
        return self


@dataclass(frozen=True)
class _MaterialContact:
    body: int
    indices: object
    local: object
    object_normal: object
    target_local: object
    pad_normal: object
    gap_gate: float
    normal_gate: float
    target_gap: float
    position_scale: float
    normal_scale: float


def _prepare_contact_terms(
    contacts, times, reference, *, contact_weight, normal_weight
):
    """Snapshot fixed targets once; ordered integer IDs avoid CUDA nonzero sync.

    Clone tensor storage even on CPU so changes to a caller's NumPy descriptor
    cannot change this solve's targets. No prepared state is shared across solves.
    """
    import torch

    times = np.asarray(times)
    tensor = lambda value: torch.as_tensor(
        value, device=reference.device, dtype=reference.dtype
    ).detach().clone()
    terms = []
    for contact in contacts:
        indices = np.flatnonzero(
            (times >= contact["active"][0]) & (times <= contact["active"][1])
        )
        if not len(indices):
            continue
        terms.append(
            _MaterialContact(
                body=int(contact["body"]),
                indices=torch.as_tensor(
                    indices, device=reference.device, dtype=torch.long
                ).clone(),
                local=tensor(contact["local"]),
                object_normal=tensor(contact["normal_local"]),
                target_local=tensor(contact["target_local"]),
                pad_normal=tensor(contact["pad_normal"]),
                gap_gate=0.95 * float(contact["max_gap_m"]),
                normal_gate=2
                * np.sin(np.radians(0.95 * float(contact["max_normal_deg"])) / 2),
                target_gap=float(contact.get("target_gap_m", 0.0)),
                position_scale=contact_weight**2 * float(contact["position_weight"]),
                normal_scale=normal_weight**2 * float(contact["normal_weight"]),
            )
        )
    return tuple(terms)


def contact_residual_squared(
    contacts,
    rotations,
    positions,
    object_R,
    object_t,
    times,
    *,
    mode,
    contact_weight,
    normal_weight,
):
    """Exact native pad/normal residual; each direct call reads its inputs anew."""
    if mode not in ("target", "within_gates"):
        raise ValueError("Unknown material contact mode")
    terms = _prepare_contact_terms(
        contacts, times, positions,
        contact_weight=contact_weight, normal_weight=normal_weight,
    )
    return _prepared_contact_residual_squared(
        terms, rotations, positions, object_R, object_t, mode=mode
    )


def _prepared_contact_residual_squared(
    terms, rotations, positions, object_R, object_t, *, mode
):
    import torch

    value = positions.sum() * 0.0

    def outside(vector, limit):
        return vector * torch.clamp(
            1
            - limit
            / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-30),
            min=0,
        )

    if mode not in ("target", "within_gates"):
        raise ValueError("Unknown material contact mode")
    for contact in terms:
        active = contact.indices
        R = rotations[active, contact.body]
        point = (R @ contact.local[:, None]).squeeze(-1) + positions[
            active, contact.body
        ]
        normal = (object_R[active] @ contact.object_normal[:, None]).squeeze(
            -1
        )
        target = (object_R[active] @ contact.target_local[:, None]).squeeze(
            -1
        ) + object_t[active]
        gap = point - target
        normal_delta = (R @ contact.pad_normal[:, None]).squeeze(-1) + normal
        if mode == "within_gates":
            gap = outside(gap, contact.gap_gate)
            normal_delta = outside(normal_delta, contact.normal_gate)
        else:
            gap = gap - contact.target_gap * normal
        value = value + contact.position_scale * gap.square().sum()
        value = value + contact.normal_scale * normal_delta.square().sum()
    return value


def _prepare_iteration_terms(context, tensor, torch):
    """Prepare the same per-hinge normalization as the native seed residual."""
    if context.iteration is None:
        return ()
    from .repair_iteration import derivative, stencil_mask

    terms = []
    for order in (2, 3):
        mask = stencil_mask(context.iteration.editable, order)
        ref = derivative(context.initial_qpos[:, context.qids], context.iteration.timestamps, order)[mask]
        normalization = np.maximum(np.sqrt(np.mean(ref ** 2, axis=0)), 1.0)
        grids = []
        grid = context.iteration.timestamps.copy()
        for _ in range(order):
            grids.append(tensor(np.diff(grid))[:, None])
            grid = (grid[1:] + grid[:-1]) / 2
        terms.append((tensor(np.flatnonzero(mask), torch.long), tensor(normalization), grids))
    return tuple(terms)


def _seed_derivative_residual_squared(selected, seed_selected, terms):
    """Unweighted SSE matching RepairIteration.derivative_terms, not its mean."""
    import torch

    residual = selected.sum() * 0.0
    for selected_rows, normalization, grids in terms:
        change = selected - seed_selected
        for spacing in grids:
            change = torch.diff(change, dim=0) / spacing
        residual = residual + (change[selected_rows] / normalization).square().sum()
    return residual


def _geometry_scope(context, selection):
    eligible = set()
    for pair in context.pairs:
        if pair["a"] == "object":
            eligible.add(pair["b"])
        elif pair["b"] == "object":
            eligible.add(pair["a"])
    eligible = {
        key
        for key in eligible
        if context.meshes[key]["body"] is not None
        and context.meshes[key].get("geom") is not None
    }
    editable_bodies = {
        int(context.model.jnt_bodyid[context.model.joint(name).id])
        for name in context.names
    }

    def influenced(body):
        while body:
            if body in editable_bodies:
                return True
            body = int(context.model.body_parentid[body])
        return False

    movable = {key for key in eligible if influenced(context.meshes[key]["body"])}
    chosen = movable if selection is None else set(selection)
    if not chosen <= eligible:
        raise ValueError(
            "object_geoms must be native meshes in declared robot-object pairs"
        )
    return chosen, movable, sorted(eligible - movable) if selection is None else []


def _geometry_samples(context, selection):
    """Original geometry only; optional selection does not narrow the audit."""
    from scipy.spatial import ConvexHull, QhullError
    from scipy.spatial.transform import Rotation

    chosen, movable, omitted = _geometry_scope(context, selection)
    points, bodies, records = [], [], []
    for key, item in context.meshes.items():
        if key not in chosen:
            continue
        v, f = item["mesh"].vertices, item["mesh"].faces
        try:
            hull = ConvexHull(v).vertices
        except QhullError:
            hull = np.arange(len(v))  # Valid planar original surfaces are supported.
        ids = np.unique(
            np.r_[hull, np.linspace(0, len(v) - 1, min(256, len(v)), dtype=int)]
        )
        ids = ids[np.linspace(0, len(ids) - 1, min(512, len(ids)), dtype=int)]
        local = np.concatenate(
            [
                v[ids],
                v[f[np.linspace(0, len(f) - 1, min(128, len(f)), dtype=int)]].mean(1),
            ]
        )
        geom = item["geom"]
        R = Rotation.from_quat(context.model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()
        points.append(local @ R.T + context.model.geom_pos[geom])
        bodies.extend([item["body"]] * len(local))
        records.append(
            dict(
                mesh_id=key,
                body=int(item["body"]),
                samples=len(local),
                original_vertices=len(v),
                original_faces=len(f),
                has_editable_ancestor=key in movable,
            )
        )
    return (
        (np.concatenate(points) if points else np.empty((0, 3))),
        np.asarray(bodies, dtype=int),
        records,
        omitted,
    )


def _bounded_cast(values, lower, upper, dtype):
    """Keep original bounds even when an output float32 rounds outward."""
    lo, hi = lower.astype(dtype), upper.astype(dtype)
    lo = np.where(
        lo.astype(float) < lower, np.nextafter(lo, np.asarray(np.inf, dtype=dtype)), lo
    )
    hi = np.where(
        hi.astype(float) > upper, np.nextafter(hi, np.asarray(-np.inf, dtype=dtype)), hi
    )
    if np.any(lo > hi):
        raise ValueError("Correction corridor contains no representable native value")
    return np.maximum(np.minimum(values.astype(dtype), hi), lo)


def solve_material(context, *, options=None, progress=None, cancelled=None):
    """Return ``(qpos, detail)``; exact audit/export/selection belong to caller.

    Source qpos prior and velocity/acceleration regularizers use full-sequence
    means, including joins to frozen frames. Image-source and temporal gates,
    self/environment collisions and whole-mesh clearance remain audit checks;
    this proposal does not replace them with its sampled winding-distance loss.
    """
    options = (
        options
        if isinstance(options, MaterialOptions)
        else MaterialOptions(**({} if options is None else options))
    )
    options.validate_context(context)
    if any(hook is not None and not callable(hook) for hook in (progress, cancelled)):
        raise TypeError("Material progress/cancel hooks must be callable")

    def check_cancel():
        if cancelled is not None and cancelled():
            raise MaterialRepairCancelled(
                "Material repair cancelled; no candidate returned"
            )

    check_cancel()
    context.check_inputs_unchanged()
    started = time.perf_counter()
    from .local_repair import WindowProblem
    from .repair_tensor import NativeTensorFK, WarpSignedDistance
    import torch

    problem = WindowProblem(context)
    if problem.weights["clearance_weight"] <= 0:
        raise ValueError(
            "Material objective normalization requires positive clearance_weight"
        )
    if progress:
        progress(
            dict(backend="material", phase="initializing", step=0, steps=options.steps)
        )
    check_cancel()
    fk = NativeTensorFK(context.model, options.device)
    t = fk.t
    q0 = context.parent["qpos"]
    base = t(q0)
    fk.validate_qpos(base)
    points, bodies, geometry, omitted_frozen = _geometry_samples(
        context, options.object_geoms
    )
    if not len(points) and not context.contacts:
        raise ValueError(
            "Material repair needs fixed contacts or declared robot-object meshes"
        )
    query = None
    if len(points):
        mesh = context.meshes["object"]["mesh"]
        query = WarpSignedDistance(mesh.vertices, mesh.faces, device=options.device)
    points, bodies = t(points), t(bodies, torch.long)
    times = (
        np.arange(
            context.start * options.fit_substeps, context.end * options.fit_substeps + 1
        )
        / options.fit_substeps
    )
    full_sample_count = (context.T - 1) * options.fit_substeps + 1
    sampled = [context.sample(q0, float(frame)) for frame in times]
    frozen_samples = t(np.asarray([row[0] for row in sampled]))
    fk.validate_qpos(frozen_samples)
    object_R, object_t = (
        t(np.asarray([row[1] for row in sampled])),
        t(np.asarray([row[2] for row in sampled])),
    )
    contact_terms = _prepare_contact_terms(
        context.contacts, times, base,
        contact_weight=problem.weights["contact_weight"],
        normal_weight=problem.weights["normal_weight"],
    )
    lo_frames = np.floor(times).astype(int)
    lower_frames, upper_frames = (
        t(lo_frames, torch.long),
        t(np.minimum(lo_frames + 1, context.T - 1), torch.long),
    )
    alpha = t(times - lo_frames)[:, None]
    indices, frames = t(context.qids, torch.long), t(context.frames, torch.long)
    original = base[:, indices]
    variable = t(context.initial_qpos[np.ix_(context.frames, context.qids)]).clone().requires_grad_(True)
    seed_selected = t(context.initial_qpos[:, context.qids])
    iteration_terms = _prepare_iteration_terms(context, t, torch)
    iteration_objective_scale = (
        options.penalty * context.iteration.derivative_weight ** 2
        / (problem.weights["clearance_weight"] ** 2 * full_sample_count)
        if context.iteration is not None else 0.0
    )
    lower, upper = t(context.lower), t(context.upper)
    with torch.no_grad():
        Rcheck, pcheck = fk(base[frames])
        position_error = rotation_error = 0.0
        for row, frame in enumerate(context.frames):
            context.forward(q0[frame])
            position_error = max(
                position_error,
                float(np.max(np.abs(pcheck[row].cpu().numpy() - context.data.xpos))),
            )
            rotation_error = max(
                rotation_error,
                float(
                    np.max(
                        np.abs(
                            Rcheck[row].cpu().numpy()
                            - context.data.xmat.reshape(-1, 3, 3)
                        )
                    )
                ),
            )
        if max(position_error, rotation_error) > 1e-8:
            raise ValueError("Tensor FK differs from original native model")
    optimizer = torch.optim.Adam([variable], lr=options.lr)
    sync = lambda: (
        torch.cuda.synchronize(fk.device) if fk.device.type == "cuda" else None
    )
    history = []
    invalid_gradient_count = 0
    sync()
    setup_seconds = time.perf_counter() - started
    tick = time.perf_counter()
    for step in range(options.steps):
        check_cancel()
        optimizer.zero_grad()
        selected = original.index_copy(0, frames, variable)
        interpolated = (
            selected[lower_frames] * (1 - alpha) + selected[upper_frames] * alpha
        )
        q = frozen_samples.index_copy(1, indices, interpolated)
        R, p = fk(q)
        penetration = p.sum() * 0.0
        if query is not None:
            surface = (R[:, bodies] @ points[None, :, :, None]).squeeze(-1) + p[
                :, bodies
            ]
            diagnostics = query.query((surface - object_t[:, None]) @ object_R)
            distances = diagnostics["distances"]
            if not bool(torch.isfinite(distances).all()) or not bool(
                diagnostics["found"].all()
            ):
                raise FloatingPointError(
                    "Original mesh distance query failed; proposal aborted"
                )
            invalid_gradient_count += int((~diagnostics["gradient_valid"]).sum())
            penetration = (
                torch.relu(options.penetration_margin_m - distances)
                .topk(min(12, len(points)), dim=1)
                .values.square()
                .mean()
            )
            penetration = penetration * len(times) / full_sample_count
        anchor = _prepared_contact_residual_squared(
            contact_terms,
            R,
            p,
            object_R,
            object_t,
            mode=problem.contact_mode,
        )
        anchor = anchor / (problem.weights["clearance_weight"] ** 2 * full_sample_count)
        source = (selected - original).square().mean()
        velocity = (
            torch.diff(selected, dim=0).square().mean() if context.T > 1 else source * 0
        )
        acceleration = (
            torch.diff(selected, n=2, dim=0).square().mean()
            if context.T > 2
            else source * 0
        )
        loss = (
            options.penalty * (penetration + anchor)
            + 0.00015 * source
            + options.smooth * (0.0003 * velocity + 0.0015 * acceleration)
        )
        derivative_preservation = source * 0.0
        if context.iteration is not None:
            derivative_preservation = _seed_derivative_residual_squared(
                selected, seed_selected, iteration_terms
            )
            # Both fixed contact and seed-derivative terms originate as native
            # residual SSE. Apply the same clearance/timeline normalization;
            # an unscaled dimensionless derivative mean overwhelms the existing
            # metre-squared Material objective even for microscopic edits.
            loss = loss + iteration_objective_scale * derivative_preservation
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite material proposal objective")
        loss.backward()
        if variable.grad is None or not bool(torch.isfinite(variable.grad).all()):
            raise FloatingPointError("Nonfinite material proposal derivative")
        check_cancel()
        optimizer.step()
        with torch.no_grad():
            variable.copy_(torch.maximum(torch.minimum(variable, upper), lower))
        if step % options.progress_every == 0 or step == options.steps - 1:
            row = dict(
                step=step,
                loss=float(loss.detach()),
                penetration=float(penetration.detach()),
                material_anchor_loss=float(anchor.detach()),
                source=float(source.detach()),
                joint_velocity=float(velocity.detach()),
                joint_acceleration=float(acceleration.detach()),
                **({"seed_derivative_preservation": float(derivative_preservation.detach())}
                   if context.iteration is not None else {}),
                **({"seed_derivative_objective": iteration_objective_scale * float(derivative_preservation.detach())}
                   if context.iteration is not None else {}),
            )
            history.append(row)
            if progress:
                progress(
                    dict(
                        backend="material",
                        phase="optimizing",
                        steps=options.steps,
                        **row,
                    )
                )
    sync()
    optimization_seconds = time.perf_counter() - tick
    check_cancel()
    result = q0.copy()
    result[np.ix_(context.frames, context.qids)] = _bounded_cast(
        variable.detach().cpu().numpy(), context.lower, context.upper, q0.dtype
    )
    if not np.array_equal(result[~context.mask], q0[~context.mask]):
        raise RuntimeError("Material proposal changed frozen native values")
    values = result[np.ix_(context.frames, context.qids)]
    if np.any(values < context.lower) or np.any(values > context.upper):
        raise RuntimeError("Material proposal exceeded original correction corridor")
    context.check_inputs_unchanged()
    detail = dict(
        backend="material",
        selected=False,
        acceptance=False,
        options=options.asdict(),
        bindings=deepcopy(context.bindings),
        config_sha256=context.config_digest,
        source_guide="frozen_original_parent",
        fit_source_frames=times.tolist(),
        setup_seconds=setup_seconds,
        optimization_seconds=optimization_seconds,
        history=history,
        selected_object_geoms=geometry,
        omitted_frozen_object_geoms=omitted_frozen,
        surface_sample_count=len(points),
        prepared_contacts=dict(
            scope="per_solve_cloned_constants",
            active_contact_count=len(contact_terms),
            active_sample_counts=[len(term.indices) for term in contact_terms],
            inclusive_active_boundaries=True,
        ),
        fk=fk.metadata(),
        query=None if query is None else query.metadata(),
        native_fk_max_position_error_m=position_error,
        native_fk_max_rotation_matrix_error=rotation_error,
        frozen_qpos_exact=True,
        original_corridor_unchanged=True,
        invalid_distance_gradient_samples=invalid_gradient_count,
        normalization="full_sequence_means; fixed contact SSE / clearance_weight^2 / full sampled timeline",
        normalization_sample_count=full_sample_count,
        native_solver_settings_used=dict(
            contact_mode=problem.contact_mode,
            contact_weight=problem.weights["contact_weight"],
            normal_weight=problem.weights["normal_weight"],
            clearance_weight=problem.weights["clearance_weight"],
        ),
        native_solver_settings_ignored=dict(
            max_rounds=problem.max_rounds,
            max_evaluations=problem.max_evaluations,
            pose_weight=problem.weights["pose_weight"],
            acceleration_weight=problem.weights["acceleration_weight"],
            source_weight=problem.weights["source_weight"],
            fit_substeps=context.config.get("solver", {}).get("fit_substeps", 2),
        ),
        objective_coefficients=dict(
            penetration=options.penalty,
            normalized_material_anchor=options.penalty,
            source_qpos_mean_square=0.00015,
            joint_velocity_mean_square=options.smooth * 0.0003,
            joint_acceleration_mean_square=options.smooth * 0.0015,
        ),
        objective_formula=(
            "penalty * (full_timeline_top12_signed_clearance_MSE + "
            "weighted_material_contact_SSE / clearance_weight^2 / full_sample_count) "
            "+ 0.00015 * full_sequence_source_qpos_MSE "
            "+ smooth * (0.0003 * full_sequence_velocity_MSE "
            "+ 0.0015 * full_sequence_acceleration_MSE)"
        ),
        audit_only_checks=[
            "source image gates",
            "temporal gates",
            "self/environment collisions",
            "original whole-mesh/subframe clearance",
        ],
        limitations=[
            "Sampled winding distances are proposal heuristics; solid signs are not certified.",
            "No source-image or explicit temporal-gate residual is optimized; original qpos and velocity/acceleration regularizers are retained.",
            "No dynamics, force closure or continuous collision guarantee; independent exact audit/export is required.",
        ],
    )
    if context.iteration is not None:
        detail["iteration"] = dict(
            seed_sha256=context.iteration.sha256,
            original_posture_prior_preserved=True,
            derivative_weight=context.iteration.derivative_weight,
            objective="penalty * weight^2 * seed_derivative_SSE / (clearance_weight^2 * full_sample_count)",
            objective_scale=iteration_objective_scale,
            raw_residual="Unweighted SSE of seed-relative hinge acceleration/jerk normalized per hinge by max(seed RMS, 1 native unit); same residuals as native solver",
            nonlinear_wrist_smoothness="Independent audit gate; not a hard optimized constraint",
        )
        detail["objective_coefficients"]["seed_derivative_SSE"] = iteration_objective_scale
        detail["objective_formula"] += " + iteration_objective_scale * seed_derivative_SSE"
    return result, detail
