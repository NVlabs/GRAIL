"""Bounded research adapters, not production repair or upstream reproductions.

HOI's nearest-surface Adam objective can be evaluated on an explicitly bound
native case, including on CPU. MR11-inspired named starts retain one immutable
posture/contact objective and are independently screened. Every output remains
unselected; passing numeric gates is not motion acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from .contact_edit import native_io
from .local_repair import WindowProblem
from .motion_audit import AuditContext


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finish(context, qpos, before, detail, started):
    if not np.array_equal(qpos[~context.mask], context.parent["qpos"][~context.mask]):
        raise RuntimeError("Research adapter changed frozen qpos")
    plan = dict(
        schema_version=1,
        channels=[
            dict(
                joint=name,
                window=[context.start, context.end],
                keys=[
                    [
                        context.start,
                        float(context.parent["qpos"][context.start, address]),
                    ],
                    [context.end, float(context.parent["qpos"][context.end, address])],
                ],
                interpolation="LINEAR",
            )
            for name, address in zip(context.names, context.qids)
        ],
        derived_fields=context.config.get("derived_fields", []),
    )
    candidate, merge = native_io.merge_curve_samples(
        context.parent, context.model_snapshot, plan, qpos
    )
    after = context.audit(candidate)
    status = "UNREVIEWED" if after["passed"] else "REJECTED_BY_AUDIT"
    candidate.update(
        status=np.asarray(status),
        selected=np.asarray(False),
        export_kind=np.asarray("native_reference_adaptation_" + status),
    )
    context.check_inputs_unchanged()
    report = dict(
        schema_version=1,
        method=detail.pop("method"),
        evidence_kind="ADAPTATION_NOT_EXACT_UPSTREAM_RUN",
        status=status,
        selected=False,
        audit_before=before,
        audit_after=after,
        native_merge=merge,
        frozen_qpos_exact=True,
        bindings=context.bindings,
        config_sha256=context.config_digest,
        code_sha256={**context.code_bindings, __file__: _sha(__file__)},
        elapsed_seconds=time.perf_counter() - started,
        **detail,
    )
    return candidate, report


def named_seeds(problem, count=4, fraction=0.35):
    """Explicit source-centered starts; bounds and guides never move with a seed."""
    if not 1 <= count <= 16 or not 0 <= fraction <= 1:
        raise ValueError("Invalid seed count or fraction")
    seeds = [("source", problem.initial.copy())]
    c = problem.context
    linear = np.array(
        [
            (
                (c.end - f) * c.parent["qpos"][c.start, c.qids]
                + (f - c.start) * c.parent["qpos"][c.end, c.qids]
            )
            / (c.end - c.start)
            for f in c.frames
        ]
    ).ravel()
    candidates = [
        ("endpoint_continuation", np.clip(linear, problem.lower, problem.upper))
    ]
    for sign in (1, -1):
        direction = np.array(
            [(-1 if "roll" in n or "elbow" in n else 1) for n in c.names]
        )
        span = (problem.upper - problem.lower).reshape(-1, problem.D)
        value = (
            problem.initial.reshape(-1, problem.D)
            + sign * fraction * 0.5 * span * direction
        )
        candidates.append(
            (
                "arm_branch_" + ("positive" if sign > 0 else "negative"),
                np.clip(value.ravel(), problem.lower, problem.upper),
            )
        )
    rng = np.random.default_rng(7106)
    while len(candidates) < count - 1:
        offset = (
            rng.uniform(-1, 1, problem.N)
            * fraction
            * 0.5
            * (problem.upper - problem.lower)
        )
        candidates.append(
            (
                f"bounded_{len(candidates)}",
                np.clip(problem.initial + offset, problem.lower, problem.upper),
            )
        )
    seeds.extend(candidates[: count - 1])
    return seeds


def multistart(
    config,
    config_dir=".",
    *,
    count=4,
    fraction=0.35,
    max_evaluations=None,
    max_rounds=None,
    seeds=None,
    progress=None,
):
    started = time.perf_counter()
    c = AuditContext(config, config_dir)
    shared = WindowProblem(c)
    before = c.audit()
    rows = []
    outputs = []
    seeds = named_seeds(shared, count, fraction) if seeds is None else seeds
    if (
        not seeds
        or seeds[0][0] != "source"
        or not np.array_equal(seeds[0][1], shared.initial)
    ):
        raise ValueError("First seed must be the unchanged source")
    for name, start in seeds:
        start = np.asarray(start, dtype=float)
        if (
            start.shape != shared.initial.shape
            or not np.isfinite(start).all()
            or np.any(start < shared.lower)
            or np.any(start > shared.upper)
        ):
            raise ValueError("Seed outside original finite corridor")
    seen = []
    for name, initial in seeds:
        duplicate = next(
            (i for i, x in enumerate(seen) if np.array_equal(x, initial)), None
        )
        if duplicate is not None:
            rows.append(dict(name=name, duplicate_of=duplicate))
            continue
        seen.append(initial.copy())
        p = WindowProblem(c)
        x = initial.copy()
        rounds = []
        tick = time.perf_counter()
        current = c.audit(
            {**c.parent, "qpos": p.qpos(x).astype(c.parent["qpos"].dtype)}
        )
        for index in range(max_rounds or p.max_rounds):
            if current["passed"] or current["affected_scope_passed"]:
                break
            added = p.add_failures(current, p.qpos(x))
            opt = least_squares(
                lambda v: p.evaluate(v)[0],
                x,
                jac=lambda v: p.evaluate(v)[1],
                bounds=(p.lower, p.upper),
                max_nfev=max_evaluations or p.max_evaluations,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
                x_scale="jac",
                tr_options={"atol": 1e-10, "btol": 1e-10},
            )
            x = opt.x
            current = c.audit(
                {**c.parent, "qpos": p.qpos(x).astype(c.parent["qpos"].dtype)}
            )
            rounds.append(
                dict(
                    round=index + 1,
                    nfev=int(opt.nfev),
                    cost=float(opt.cost),
                    added_clearance=added,
                    passed=current["passed"],
                    affected_scope_passed=current["affected_scope_passed"],
                )
            )
        # Rank all candidates using one fixed objective with no seed-specific witnesses.
        residual, _ = shared.evaluate(x)
        cost = float(residual @ residual / 2)
        row = dict(
            name=name,
            eligible=bool(current["passed"]),
            affected_scope_passed=current["affected_scope_passed"],
            common_cost=cost,
            elapsed_seconds=time.perf_counter() - tick,
            rounds=rounds,
            summary=current["summary"],
            failure_count=len(current["failures"]),
            posture_guide_equal_to_parent=np.array_equal(p.initial, shared.initial),
        )
        rows.append(row)
        outputs.append((row, p.qpos(x).astype(c.parent["qpos"].dtype)))
        if progress:
            progress(row)
    ranked = sorted(
        range(len(outputs)),
        key=lambda i: (
            not outputs[i][0]["eligible"],
            not outputs[i][0]["affected_scope_passed"],
            outputs[i][0]["common_cost"],
            i,
        ),
    )
    best = ranked[0]
    eligible = next((i for i in ranked if outputs[i][0]["eligible"]), None)
    return _finish(
        c,
        outputs[best][1],
        before,
        dict(
            method="mr11_inspired_multistart_fixed_anchors",
            upstream_commit="022ec96ec946a17fce176a74cf12d6df61f6444a",
            attempts=rows,
            diagnostic_output_name=outputs[best][0]["name"],
            eligible_proposal_name=None
            if eligible is None
            else outputs[eligible][0]["name"],
            objective_shared_across_starts=True,
            source_guide="frozen_original_parent",
            adaptation_differences=[
                "Coupled local-window objective replaces upstream static fit_contacts.",
                "Explicit deterministic bounded seeds; no new grasp/contact proposals.",
                "Independent original-triangle, material-anchor, source and temporal audit retained.",
            ],
        ),
        started,
    )


def warm_start_anchored(
    config,
    config_dir=".",
    *,
    warm_start_path,
    max_rounds=None,
    max_evaluations=None,
    retain_parent_witnesses=True,
    round_output_dir=None,
    progress=None,
):
    """One explicit warm start without moving original guides, bounds or phases."""
    started = time.perf_counter()
    c = AuditContext(config, config_dir)
    p = WindowProblem(c)
    path = Path(warm_start_path).resolve(strict=True)
    warm_digest = _sha(path)
    initial = native_io.load_npz(path)
    if _sha(path) != warm_digest:
        raise ValueError("Warm-start file changed while loading")
    before = c.audit()
    warm = c.audit(initial)
    if not np.array_equal(initial["qpos"][~c.mask], c.parent["qpos"][~c.mask]):
        raise ValueError("Warm start changed frozen coordinates")
    x = initial["qpos"][np.ix_(c.frames, c.qids)].astype(float).ravel()
    if np.any(x < p.lower) or np.any(x > p.upper):
        raise ValueError("Warm start outside original correction corridor")
    # Retain collision evidence that motivated the trial, even when the warm
    # start clears one old crossing and creates a different one elsewhere.
    if retain_parent_witnesses:
        p.add_failures(before, c.parent["qpos"])
    current = warm
    rounds = []
    tick = time.perf_counter()
    for index in range(p.max_rounds if max_rounds is None else max_rounds):
        if current["passed"] or current["affected_scope_passed"]:
            break
        added = p.add_failures(current, p.qpos(x))
        opt = least_squares(
            lambda v: p.evaluate(v)[0],
            x,
            jac=lambda v: p.evaluate(v)[1],
            bounds=(p.lower, p.upper),
            max_nfev=p.max_evaluations if max_evaluations is None else max_evaluations,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            x_scale="jac",
            tr_options={"atol": 1e-10, "btol": 1e-10},
        )
        x = opt.x
        current = c.audit(
            {**c.parent, "qpos": p.qpos(x).astype(c.parent["qpos"].dtype)}
        )
        row = dict(
            round=index + 1,
            nfev=int(opt.nfev),
            cost=float(opt.cost),
            added_clearance_witnesses=added,
            audit_passed=current["passed"],
            affected_scope_passed=current["affected_scope_passed"],
            summary=current["summary"],
        )
        rounds.append(row)
        if round_output_dir is not None:
            folder = Path(round_output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            motion, round_report = _finish(
                c,
                p.qpos(x).astype(c.parent["qpos"].dtype),
                before,
                dict(
                    method="warm_start_intermediate_diagnostic",
                    round=index + 1,
                    posture_guide_equal_to_parent=True,
                ),
                started,
            )
            destination = folder / f"round_{index + 1:02d}_UNREVIEWED.npz"
            if destination.exists():
                raise FileExistsError("Cannot overwrite round evidence")
            np.savez_compressed(destination, **motion)
            row["candidate_path"] = str(destination.resolve())
            row["candidate_sha256"] = _sha(destination)
        if progress:
            progress(row)
    solve_seconds = time.perf_counter() - tick
    if _sha(path) != warm_digest:
        raise ValueError("Warm-start file changed during repair")
    return _finish(
        c,
        p.qpos(x).astype(c.parent["qpos"].dtype),
        before,
        dict(
            method="explicit_warm_start_original_anchored_repair",
            warm_start_path=str(path),
            warm_start_sha256=warm_digest,
            audit_warm_start=warm,
            rounds=rounds,
            optimization_seconds=solve_seconds,
            clearance_witness_count=len(p.clearance),
            posture_guide_equal_to_parent=np.array_equal(
                p.initial,
                c.parent["qpos"][np.ix_(c.frames, c.qids)].astype(float).ravel(),
            ),
            original_correction_corridor_unchanged=True,
            retain_parent_witnesses=retain_parent_witnesses,
            diagnostic_rounds_saved=round_output_dir is not None,
            adaptation_differences=[
                "External proposal changes initialization only; all native constraints/source/posture priors remain bound to original parent.",
                "Warm-start collision witnesses are retained; optional original-parent witnesses are an explicit ablation; source branch is not redundantly optimized.",
            ],
        ),
        started,
    )


def _load_upstream(root):
    root = Path(root).resolve(strict=True)
    binding = json.loads((root / "bindings.json").read_text())
    if binding["commit"] != "58e618feef66c8a168efe75329c57efc3c36b34f":
        raise ValueError("Unexpected upstream revision")
    modules = {}
    for stem in ["kinematics", "geometry"]:
        path = root / (stem + ".py")
        if _sha(path) != binding["files"][stem + ".py"]["sha256"]:
            raise ValueError("Upstream source changed")
        name = "_repair_reference_" + stem + "_" + _sha(path)[:12]
        if name in sys.modules:
            module = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        modules[stem] = module
    return modules, binding


def _mesh_body_samples(c, item):
    """Upstream hull/vertex/centroid rule in the native body's local frame."""
    from scipy.spatial.transform import Rotation

    mesh = item["mesh"]
    v, f = mesh.vertices, mesh.faces
    ids = np.unique(
        np.r_[
            ConvexHull(v).vertices,
            np.linspace(0, len(v) - 1, min(256, len(v)), dtype=int),
        ]
    )
    ids = ids[np.linspace(0, len(ids) - 1, min(512, len(ids)), dtype=int)]
    vv = np.concatenate(
        [
            v[ids],
            v[f[np.linspace(0, len(f) - 1, min(128, len(f)), dtype=int)]].mean(1),
        ]
    )
    geom = item["geom"]
    rotation = Rotation.from_quat(c.model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()
    return vv @ rotation.T + c.model.geom_pos[geom]


def _hand_samples(c, side):
    points = []
    bodies = []
    names = []
    for item in c.meshes.values():
        g = item.get("geom")
        if g is None:
            continue
        name = c.model.body(item["body"]).name
        if not name.startswith(side + "_") or not ("hand" in name or "wrist" in name):
            continue
        vv = _mesh_body_samples(c, item)
        points.append(vv)
        bodies.extend([item["body"]] * len(vv))
        names.extend([name] * len(vv))
    if not points:
        raise ValueError("No original hand/wrist meshes in configured geometry scope")
    return np.concatenate(points), np.array(bodies), np.array(names)


def relative_mesh_points(
    body_R, body_p, source_body, source_local, target_body, target_geom_R, target_geom_p
):
    """Map body-local source samples into another original geometry's frame.

    All operations are differentiable through BOTH native body transforms;
    detach neither body, including when they share an editable parent joint.
    Inputs follow column-vector rotation conventions; point arrays use rows.
    """
    source_R = body_R[:, source_body]
    source_p = body_p[:, source_body]
    world_points = source_local[None] @ source_R.transpose(-1, -2) + source_p[:, None]
    target_R = body_R[:, target_body] @ target_geom_R
    target_p = (body_R[:, target_body] @ target_geom_p[:, None]).squeeze(-1) + body_p[
        :, target_body
    ]
    return (world_points - target_p[:, None]) @ target_R


def self_clearance_loss(distances, margins, *, reduction="global_topk"):
    """Top-12 signed-clearance proposal, with explicit pair normalization."""
    import torch

    if not distances or len(distances) != len(margins):
        raise ValueError(
            "Self distances and margins must have matching nonempty groups"
        )
    violations = [torch.relu(margin - d) for d, margin in zip(distances, margins)]
    if reduction == "global_topk":
        # Extra distant configured pairs must not dilute the active collision.
        merged = torch.cat(violations, dim=1)
        return merged.topk(min(12, merged.shape[1]), dim=1).values.square().mean()
    if reduction == "mean_pairs":
        return torch.stack(
            [
                v.topk(min(12, v.shape[1]), dim=1).values.square().mean()
                for v in violations
            ]
        ).mean()
    raise ValueError("Unknown self-distance reduction")


def _self_collision_scope(context):
    """Separate robot self objectives from static pairs retained by the audit."""
    robot_pairs, audit_only = [], []
    for pair in context.pairs:
        a, b = pair["a"], pair["b"]
        if any(context.meshes[key].get("static", False) for key in (a, b)):
            audit_only.append(dict(
                pair=[a, b],
                reason="Static environment clearance is not optimized by this objective",
                original_triangle_audit_retained=True,
            ))
        elif "object" not in (a, b):
            if any(context.meshes[key]["body"] is None for key in (a, b)):
                raise ValueError("Configured self pair must have two native robot bodies")
            robot_pairs.append(pair)
    return robot_pairs, audit_only


def _audit_material_samples(c, audits, side, *, pair_kind="object"):
    """Sample original robot faces at audited crossings, without contact relabeling.

    Barycentric interpolants cover narrow crossing slivers. Their body-local
    coordinates stay fixed while the curved-mesh distance is queried anew.
    Object mode samples only the requested hand; self mode samples BOTH robot
    sides, including torso surfaces, for their explicitly configured self pairs.
    """
    from scipy.spatial.transform import Rotation

    if pair_kind not in ("object", "self"):
        raise ValueError("Unknown witness pair kind")
    samples = {}
    for label, audit in audits:
        for failure in audit["failures"]:
            frame = failure.get("source_frame")
            if failure["kind"] != "collision" or frame is None:
                continue
            if not c.start < frame < c.end:
                continue
            a, b = failure["pair"]
            if pair_kind == "object":
                sides = (
                    [(a, "a")] if b == "object" else [(b, "b")] if a == "object" else []
                )
            else:
                sides = (
                    [(a, "a"), (b, "b")]
                    if "object" not in (a, b)
                    and all(c.meshes[key]["body"] is not None for key in (a, b))
                    else []
                )
            if not sides:
                continue
            for mesh_id, suffix in sides:
                item = c.meshes[mesh_id]
                geom = item.get("geom")
                body = item["body"]
                if geom is None or body is None:
                    continue
                name = c.model.body(body).name
                if pair_kind == "object" and (
                    not name.startswith(side + "_")
                    or not ("hand" in name or "wrist" in name)
                ):
                    continue
                mesh = item["mesh"]
                rotation = Rotation.from_quat(
                    c.model.geom_quat[geom][[1, 2, 3, 0]]
                ).as_matrix()
                for witness in failure["witnesses"]:
                    face = int(witness["face_" + suffix])
                    bary = np.asarray(witness["barycentric_" + suffix], dtype=float)
                    if (
                        face < 0
                        or face >= len(mesh.faces)
                        or bary.shape != (3,)
                        or not np.isfinite(bary).all()
                        or abs(float(bary.sum()) - 1) > 1e-6
                        or float(bary.min()) < -1e-6
                    ):
                        raise ValueError("Invalid original-face collision witness")
                    barycentrics = [bary]
                    for vertex in np.eye(3):
                        for fraction in (0.1, 0.25, 0.5, 1.0):
                            barycentrics.append(
                                (1 - fraction) * bary + fraction * vertex
                            )
                    for coordinate in barycentrics:
                        key = (mesh_id, face, tuple(coordinate.round(12)))
                        if key not in samples:
                            local = coordinate @ mesh.vertices[mesh.faces[face]]
                            samples[key] = dict(
                                mesh_id=mesh_id,
                                original_face=face,
                                barycentric=coordinate.tolist(),
                                body=int(body),
                                body_name=name,
                                local=(
                                    rotation @ local + c.model.geom_pos[geom]
                                ).tolist(),
                                evidence=[],
                            )
                        evidence = dict(audit=label, frame=float(frame), pair=[a, b])
                        if evidence not in samples[key]["evidence"]:
                            samples[key]["evidence"].append(evidence)
    rows = list(samples.values())
    return (
        np.asarray([x["local"] for x in rows]).reshape(-1, 3),
        np.asarray([x["body"] for x in rows], dtype=int),
        np.asarray([x["body_name"] for x in rows], dtype=str),
        rows,
    )


def fixed_anchor_loss(
    contacts,
    body_R,
    body_p,
    object_R,
    object_t,
    times,
    *,
    contact_mode="within_gates",
    contact_weight=1000.0,
    normal_weight=10.0,
):
    """Torch evaluation of the exact WindowProblem material-contact residual norm.

    This returns the summed squared weighted residual, before method-specific
    averaging. Torch/Warp remain optional imports for all other module users.
    """
    import torch

    value = body_p.sum() * 0.0
    t = lambda x: torch.as_tensor(x, dtype=body_p.dtype, device=body_p.device)
    for contact in contacts:
        mask = torch.as_tensor(
            (np.asarray(times) >= contact["active"][0])
            & (np.asarray(times) <= contact["active"][1]),
            device=body_p.device,
        )
        if not bool(mask.any()):
            continue
        rot = body_R[mask, contact["body"]]
        point = (rot @ t(contact["local"])[:, None]).squeeze(-1) + body_p[
            mask, contact["body"]
        ]
        normal = (object_R[mask] @ t(contact["normal_local"])[:, None]).squeeze(-1)
        target = (object_R[mask] @ t(contact["target_local"])[:, None]).squeeze(
            -1
        ) + object_t[mask]
        gap = point - target
        padnormal = (rot @ t(contact["pad_normal"])[:, None]).squeeze(-1)
        nd = padnormal + normal
        if contact_mode == "within_gates":
            # Equivalent to WindowProblem._outside without a zero-norm divide.
            pg = 0.95 * contact["max_gap_m"]
            ng = 2 * np.sin(np.radians(0.95 * contact["max_normal_deg"]) / 2)
            gap = gap * torch.clamp(
                1
                - pg
                / torch.linalg.vector_norm(gap, dim=-1, keepdim=True).clamp_min(1e-30),
                min=0,
            )
            nd = nd * torch.clamp(
                1
                - ng
                / torch.linalg.vector_norm(nd, dim=-1, keepdim=True).clamp_min(1e-30),
                min=0,
            )
        elif contact_mode == "target":
            gap = gap - contact.get("target_gap_m", 0.0) * normal
        else:
            raise ValueError("Unknown material contact mode")
        value = value + (
            contact_weight**2 * contact["position_weight"] * gap.square().sum()
            + normal_weight**2 * contact["normal_weight"] * nd.square().sum()
        )
    return value


def hoi_adam(
    config,
    config_dir=".",
    *,
    upstream_root,
    device="cpu",
    steps=180,
    lr=0.003,
    penalty=100.0,
    gap_weight=2.0,
    smooth=1.0,
    fit_substeps=1,
    side="left",
    loss_normalization="upstream_full_sequence",
    contact_objective="nearest_mesh",
    sample_augmentation="none",
    witness_report_paths=(),
    penetration_margin_m=0.0007,
    self_collision_weight=0.0,
    self_collision_reduction="global_topk",
    progress=None,
):
    """Same HOI contact/penetration math, adapted explicit window/native scope.

    The pinned upstream FK and Warp winding query run unchanged. This wrapper
    removes their hardcoded CUDA device and constrains optimization to the
    caller's declared window/joints. Nearest loss does not fix material contacts.
    """
    if device not in ("cpu", "cuda", "cuda:0"):
        raise ValueError(
            "Research runtime supports only CPU or the explicitly owned GPU0"
        )
    if side not in ("left", "right"):
        raise ValueError("Unknown hand side")
    if (
        any(
            not np.isfinite(v) or v < 0
            for v in (lr, penalty, gap_weight, smooth, self_collision_weight)
        )
        or lr == 0
    ):
        raise ValueError(
            "Optimizer scales must be finite nonnegative; learning rate positive"
        )
    if not 1 <= steps <= 10000 or not 1 <= fit_substeps <= 8:
        raise ValueError("Invalid solve budget")
    if loss_normalization not in ("upstream_full_sequence", "local_window"):
        raise ValueError("Unknown normalization")
    if contact_objective not in (
        "nearest_mesh",
        "fixed_anchors",
        "nearest_plus_anchors",
    ):
        raise ValueError("Unknown contact objective")
    if sample_augmentation not in ("none", "audit_material"):
        raise ValueError("Unknown sample augmentation")
    if self_collision_reduction not in ("global_topk", "mean_pairs"):
        raise ValueError("Unknown self-distance reduction")
    if witness_report_paths and sample_augmentation != "audit_material":
        raise ValueError("Witness reports require explicit audit_material mode")
    if not np.isfinite(penetration_margin_m) or not 0 <= penetration_margin_m <= 0.01:
        raise ValueError("Invalid signed-distance separation margin")
    started = time.perf_counter()
    c = AuditContext(config, config_dir)
    before = c.audit()
    import torch
    import warp as wp

    torch.set_num_threads(1)
    modules, binding = _load_upstream(upstream_root)
    fk = modules["kinematics"].TorchFK(c.model, device)
    t = fk.t
    q0 = c.parent["qpos"]
    points, bodies, names = _hand_samples(c, side)
    witness_bindings, sample_provenance = [], []
    if sample_augmentation == "audit_material":
        audits = [("original_parent", before)]
        for report_path in witness_report_paths:
            path = Path(report_path).resolve(strict=True)
            report = json.loads(path.read_text())
            audit = report["audit_after"]
            if (
                audit["bindings"] != before["bindings"]
                or audit["config_sha256"] != before["config_sha256"]
            ):
                raise ValueError(
                    "Witness audit must bind the same original input/config"
                )
            witness_bindings.append(dict(path=str(path), sha256=_sha(path)))
            audits.append((str(path), audit))
        extra, extra_bodies, extra_names, sample_provenance = _audit_material_samples(
            c, audits, side
        )
        points = np.concatenate([points, extra])
        bodies = np.concatenate([bodies, extra_bodies])
        names = np.concatenate([names, extra_names])
    pts, bi = t(points), t(bodies, torch.long)
    groups = [
        t(np.flatnonzero(np.char.startswith(names, side + "_hand_" + f)), torch.long)
        for f in ["thumb", "index", "middle"]
    ]
    if any(len(g) == 0 for g in groups):
        raise ValueError("HOI objective requires thumb/index/middle sample groups")
    times = np.arange(c.start * fit_substeps, c.end * fit_substeps + 1) / fit_substeps
    lower = np.floor(times).astype(int)
    alpha = t(times - lower)
    base = t(q0)
    indices = t(c.qids, torch.long)
    frameids = t(c.frames, torch.long)
    variable = t(q0[np.ix_(c.frames, c.qids)]).clone().requires_grad_(True)
    lo, hi = t(c.lower), t(c.upper)
    original = t(q0[np.ix_(c.frames, c.qids)])
    sampled = [c.sample(q0, float(f)) for f in times]
    object_R = t(np.array([x[1] for x in sampled]))
    object_t = t(np.array([x[2] for x in sampled]))
    active_contacts = [
        x for x in c.contacts if c.model.body(x["body"]).name.startswith(side + "_")
    ]
    hold = t(
        np.array(
            [
                any(x["active"][0] <= f <= x["active"][1] for x in active_contacts)
                for f in times
            ]
        ),
        torch.bool,
    )
    full_times = np.arange((c.T - 1) * fit_substeps + 1) / fit_substeps
    full_hold_count = sum(
        any(x["active"][0] <= f <= x["active"][1] for x in active_contacts)
        for f in full_times
    )
    if full_hold_count == 0:
        raise ValueError(
            "No active contacts for the requested hand on the source timeline"
        )
    wpdevice = "cuda:0" if device.startswith("cuda") else "cpu"
    scenes, scene_bounds = {}, {}

    def add_scene(mesh_key):
        if mesh_key in scenes:
            return
        mesh = c.meshes[mesh_key]["mesh"]
        scenes[mesh_key] = wp.Mesh(
            points=wp.array(mesh.vertices, dtype=wp.vec3, device=wpdevice),
            indices=wp.array(mesh.faces.ravel(), dtype=wp.int32, device=wpdevice),
            support_winding_number=True,
        )
        scene_bounds[mesh_key] = (
            mesh.vertices.min(0) - 0.05,
            mesh.vertices.max(0) + 0.05,
        )

    add_scene("object")
    self_terms, self_pair_metadata, self_sample_provenance = [], [], []
    robot_self_pairs, audit_only_environment_pairs = _self_collision_scope(c)
    if self_collision_weight:
        from scipy.spatial.transform import Rotation

        sample_cache, self_extra = {}, {}
        if sample_augmentation == "audit_material":
            _, _, _, self_sample_provenance = _audit_material_samples(
                c, audits, side, pair_kind="self"
            )
            for row in self_sample_provenance:
                self_extra.setdefault(row["mesh_id"], []).append(row["local"])
        for pair in robot_self_pairs:
            a, b = pair["a"], pair["b"]
            metadata = dict(pair=[a, b], margin_m=pair["clearance_m"], directions=[])
            for source, target in [(a, b), (b, a)]:
                if source not in sample_cache:
                    samples = _mesh_body_samples(c, c.meshes[source])
                    if source in self_extra:
                        samples = np.concatenate(
                            [samples, np.asarray(self_extra[source])]
                        )
                    sample_cache[source] = t(samples)
                add_scene(target)
                source_body = c.meshes[source]["body"]
                target_body = c.meshes[target]["body"]
                geom = c.meshes[target]["geom"]
                if geom is None:
                    raise ValueError(
                        "Self pair target must have an original robot geom"
                    )
                geom_R = t(
                    Rotation.from_quat(
                        c.model.geom_quat[geom][[1, 2, 3, 0]]
                    ).as_matrix()
                )
                self_terms.append(
                    (
                        source_body,
                        sample_cache[source],
                        target_body,
                        geom_R,
                        t(c.model.geom_pos[geom]),
                        target,
                        pair["clearance_m"],
                    )
                )
                metadata["directions"].append(
                    dict(
                        source=source, target=target, samples=len(sample_cache[source])
                    )
                )
            self_pair_metadata.append(metadata)

    class Distance(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p, mesh_key):
            scene = scenes[mesh_key]
            bblo, bbhi = scene_bounds[mesh_key]
            x = p.reshape(-1, 3).float().contiguous()
            out = torch.empty(len(x), device=device, dtype=torch.float32)
            grad = torch.empty_like(x)
            scope = (
                wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()))
                if device.startswith("cuda")
                else contextlib.nullcontext()
            )
            with scope:
                wp.launch(
                    modules["geometry"].query,
                    dim=len(x),
                    inputs=[
                        scene.id,
                        wp.from_torch(x, dtype=wp.vec3),
                        wp.vec3(*bblo),
                        wp.vec3(*bbhi),
                        1,
                        wp.from_torch(out),
                        wp.from_torch(grad, dtype=wp.vec3),
                    ],
                    device=wpdevice,
                )
            ctx.save_for_backward(grad.to(p.dtype).reshape(p.shape))
            return out.to(p.dtype).reshape(p.shape[:-1])

        @staticmethod
        def backward(ctx, g):
            return g[..., None] * ctx.saved_tensors[0], None

    # Validate the exact upstream native FK before using it as an optimizer.
    with torch.no_grad():
        Rcheck, pcheck = fk(base[frameids])
        fk_error = 0.0
        fk_rotation_error = 0.0
        for row, frame in enumerate(c.frames):
            c.forward(q0[frame])
            fk_error = max(
                fk_error, float(np.max(np.abs(pcheck[row].cpu().numpy() - c.data.xpos)))
            )
            fk_rotation_error = max(
                fk_rotation_error,
                float(
                    np.max(
                        np.abs(
                            Rcheck[row].cpu().numpy() - c.data.xmat.reshape(-1, 3, 3)
                        )
                    )
                ),
            )
        if fk_error > 1e-8 or fk_rotation_error > 1e-8:
            raise ValueError("Upstream FK differs from native model")
    opt = torch.optim.Adam([variable], lr=lr)
    history = []
    sync = lambda: torch.cuda.synchronize() if device.startswith("cuda") else None
    sync()
    tick = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        selected = base[:, indices].index_copy(0, frameids, variable)
        q = base.index_copy(1, indices, selected)
        qs = (
            q[t(lower, torch.long)] * (1 - alpha[:, None])
            + q[t(np.minimum(lower + 1, c.T - 1), torch.long)] * alpha[:, None]
        )
        # Preserve native frozen free-joint samples; TorchFK normalizes the
        # interpolated quaternion exactly as native mj_normalizeQuat does.
        qs = qs.clone()
        qs[:, :7] = t(np.array([x[0][:7] for x in sampled]))
        R, p = fk(qs)
        surf = (R[:, bi] @ pts[None, :, :, None]).squeeze(-1) + p[:, bi]
        d = Distance.apply((surf - object_t[:, None]) @ object_R, "object")
        self_pen = d.sum() * 0.0
        if self_terms:
            directional = []
            for (
                source_body,
                samples,
                target_body,
                geom_R,
                geom_p,
                key,
                margin,
            ) in self_terms:
                local = relative_mesh_points(
                    R, p, source_body, samples, target_body, geom_R, geom_p
                )
                distances = Distance.apply(local, key)
                directional.append(distances)
            self_pen = self_clearance_loss(
                directional,
                [entry[-1] for entry in self_terms],
                reduction=self_collision_reduction,
            )
        pen = (
            torch.relu(penetration_margin_m - d)
            .topk(min(12, len(points)), dim=1)
            .values.square()
            .mean()
        )
        fg = torch.stack([d[:, g].abs().amin(1) for g in groups], -1)
        gap = (
            torch.stack([fg[:, 0], fg[:, 1:].amin(1)], -1)[hold].square().mean()
            if bool(hold.any())
            else d.sum() * 0.0
        )
        if loss_normalization == "upstream_full_sequence":
            # Terms outside the editable interval are constant, so omit their
            # expensive geometry queries but retain their original denominators.
            pen = pen * len(times) / len(full_times)
            self_pen = self_pen * len(times) / len(full_times)
            gap = gap * int(hold.sum()) / full_hold_count
            source = (selected - base[:, indices]).square().mean()
            temporalq = selected
        else:
            source = (variable - original).square().mean()
            temporalq = q[c.start : c.end + 1, indices]
        vel = torch.diff(temporalq, dim=0).square().mean()
        acc = torch.diff(temporalq, n=2, dim=0).square().mean()
        anchor = loss_contact = gap * 0.0
        if contact_objective != "nearest_mesh":
            settings = config.get("solver", {})
            anchor = fixed_anchor_loss(
                active_contacts,
                R,
                p,
                object_R,
                object_t,
                times,
                contact_mode=settings.get("contact_mode", "target"),
                contact_weight=settings.get("contact_weight", 1000.0),
                normal_weight=settings.get("normal_weight", 1.0),
            )
            # Preserve the native solver's relative material/clearance weights.
            # Geometry is top-k signed mesh distance rather than witness SSE,
            # so this is explicitly a hybrid, not identical total objectives.
            denominator = (
                len(full_times)
                if loss_normalization == "upstream_full_sequence"
                else len(times)
            )
            anchor = anchor / (
                settings.get("clearance_weight", 2000.0) ** 2 * denominator
            )
            loss_contact = penalty * anchor
        if contact_objective != "fixed_anchors":
            loss_contact = loss_contact + gap_weight * gap
        loss = (
            penalty * pen
            + penalty * self_collision_weight * self_pen
            + loss_contact
            + 0.00015 * source
            + smooth * (0.0003 * vel + 0.0015 * acc)
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite HOI loss")
        loss.backward()
        if not bool(torch.isfinite(variable.grad).all()):
            raise FloatingPointError("Nonfinite HOI derivative")
        opt.step()
        with torch.no_grad():
            variable.copy_(torch.maximum(torch.minimum(variable, hi), lo))
        if step % 40 == 0 or step == steps - 1:
            row = dict(
                step=step,
                loss=float(loss.detach()),
                penetration=float(pen.detach()),
                self_penetration=float(self_pen.detach()),
                nearest_contact=float(gap.detach()),
                material_anchor_loss=float(anchor.detach()),
                source=float(source.detach()),
                joint_velocity=float(vel.detach()),
                joint_acceleration=float(acc.detach()),
            )
            history.append(row)
            if progress:
                progress(row)
    sync()
    solve_seconds = time.perf_counter() - tick
    result = q0.copy()
    result[np.ix_(c.frames, c.qids)] = variable.detach().cpu().numpy().astype(q0.dtype)
    for binding_row in witness_bindings:
        if _sha(binding_row["path"]) != binding_row["sha256"]:
            raise ValueError("Witness report changed during repair")
    return _finish(
        c,
        result,
        before,
        dict(
            method=(
                "hoi_inspired_nearest_adam"
                if contact_objective == "nearest_mesh"
                else "hoi_gpu_fk_material_anchor_hybrid"
            ),
            upstream=binding,
            device=device,
            optimization_seconds=solve_seconds,
            history=history,
            points=len(points),
            steps=steps,
            lr=lr,
            loss_normalization=loss_normalization,
            contact_objective=contact_objective,
            active_hold_samples_in_window=int(hold.sum()),
            total_hold_samples_in_sequence=int(full_hold_count),
            penalty=penalty,
            gap_weight=gap_weight,
            smooth=smooth,
            penetration_margin_m=penetration_margin_m,
            self_collision_weight=self_collision_weight,
            self_collision_reduction=self_collision_reduction,
            self_collision_pairs=self_pair_metadata,
            audit_only_environment_pairs=audit_only_environment_pairs,
            self_augmented_sample_provenance=self_sample_provenance,
            sample_augmentation=sample_augmentation,
            witness_report_bindings=witness_bindings,
            augmented_sample_provenance=sample_provenance,
            omitted_fixed_geometry_loss_constants=True,
            native_fk_max_error_m=fk_error,
            native_fk_max_rotation_matrix_error=fk_rotation_error,
            source_guide="frozen_original_parent",
            fit_source_frames=times.tolist(),
            held_out_source_frames=sorted(set(c.times()) - set(times)),
            adaptation_differences=[
                "Explicit native joints/window/protected poses replace upstream full-sequence arms14.",
                "Configured original hand/wrist meshes sampled with the upstream hull/vertex/centroid rule.",
                "Object scale applied from the bound native input; device selectable CPU/CUDA.",
                "Full-sequence normalization preserves upstream variable gradients while omitting constant geometry outside the window.",
                "Optional fit subframes use native frozen root interpolation; default uses integers like upstream.",
                "Fixed material-anchor/normal terms and configured bidirectional robot self-pair signed distances are optional explicit hybrid variants; independent original-triangle audit remains authoritative.",
                "Self-pair weight is zero by default; when enabled, each declared robot pair uses its own clearance margin with derivatives through both native bodies. Signed winding distance remains a sampled proposal objective, not proof of mesh separation.",
                "Static environment pairs, including packed support tables, remain original-triangle audit checks; this objective does not optimize their clearance.",
                "No GraspGenX candidate generation or physics; this tests the downstream contact repair objective.",
                "Optional audit_material sampling adds bound original-face barycentric crossing samples/interpolants, using a freshly queried curved-mesh distance at every optimizer step.",
            ],
        ),
        started,
    )


def run_reference(config_path, out_dir, method, **options):
    path = Path(config_path).resolve(strict=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    raw = path.read_bytes()
    config = json.loads(raw)
    (out / "config.json").write_bytes(raw)
    try:
        if method == "multistart_anchored":
            candidate, report = multistart(config, path.parent, **options)
        elif method == "hoi_nearest_adam":
            candidate, report = hoi_adam(config, path.parent, **options)
        elif method == "warm_start_anchored":
            candidate, report = warm_start_anchored(
                config, path.parent, round_output_dir=out / "rounds", **options
            )
        else:
            raise ValueError("Unknown reference adaptation")
        target = out / "candidate_UNREVIEWED.npz"
        np.savez_compressed(target, **candidate)
        loaded = native_io.load_npz(target)
        if set(loaded) != set(candidate) or any(
            not np.array_equal(loaded[k], v) for k, v in candidate.items()
        ):
            raise RuntimeError(
                "Serialized reference candidate differs from audited arrays"
            )
        report["candidate_sha256"] = _sha(target)
        report["serialized_exact"] = True
        (out / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        return candidate, report
    except Exception as error:
        (out / "error.json").write_text(
            json.dumps(
                dict(
                    method=method,
                    status="FAILED_RESEARCH_TRIAL",
                    error_type=type(error).__name__,
                    message=str(error),
                ),
                indent=2,
            )
            + "\n"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["multistart_anchored", "hoi_nearest_adam", "warm_start_anchored"],
    )
    parser.add_argument("--options-json", default="{}")
    args = parser.parse_args()
    _, report = run_reference(
        args.config, args.output, args.method, **json.loads(args.options_json)
    )
    print(
        json.dumps(
            dict(
                method=report["method"],
                status=report["status"],
                elapsed_seconds=report["elapsed_seconds"],
                summary=report["audit_after"]["summary"],
            )
        )
    )


if __name__ == "__main__":
    main()
