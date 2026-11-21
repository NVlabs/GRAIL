"""Bounded proposals on existing Stage 2 keys, always relative to the original.

This is a numerical proposal, not contact acceptance. Native geometry is
recomputed on affected integer source frames; the immutable dense contact map
retains its complete source timeline and independent human/object fields.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np

from .contracts import binding, validate_scene, verify_binding
from .interpolation import (
    apply_edits,
    quaternion_multiply,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
    validate_plan,
)
from .native import evaluate_motion, validate_motion

_WEIGHTS = dict(contact=1.0, posture=1e-4, temporal=1.0, source=0.01, feet=1.0)
_OBJECTIVE_KEYS = {
    "partial_mass_fraction",
    "pair_temperature_m2",
    "coverage_temperature_m2",
    "robust_delta_m",
    "gap_tolerance_m",
    "normal_distance_scale_m",
    "min_opposing_cosine",
    "activation_ramp_frames",
    "phase_blend_frames",
}


class _BudgetExhausted(Exception):
    pass


def _positive(value, name):
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _config(repair):
    allowed = {
        "contact_map",
        "objective",
        "max_nfev",
        "max_delta_rad",
        "finite_difference_rad",
        "max_time_s",
        "weights",
        "foot_joint_ids",
    }
    if not isinstance(repair, dict) or set(repair) - allowed:
        raise ValueError("Unknown local repair configuration fields")
    result = copy.deepcopy(repair)
    path = Path(result["contact_map"]).expanduser()
    if not path.is_absolute():
        raise ValueError("repair.contact_map must be an absolute path")
    result["contact_map"] = str(path.resolve(strict=True))
    for key, default in (
        ("max_delta_rad", 0.08),
        ("finite_difference_rad", 0.001),
        ("max_time_s", 120.0),
    ):
        result[key] = _positive(result.get(key, default), key)
    if result["finite_difference_rad"] > result["max_delta_rad"]:
        raise ValueError("finite_difference_rad must not exceed max_delta_rad")
    result.setdefault("max_nfev", 20)
    if type(result["max_nfev"]) is not int or result["max_nfev"] < 1:
        raise ValueError("max_nfev must be a positive integer objective-evaluation budget")
    objective = result.setdefault("objective", {})
    if not isinstance(objective, dict) or set(objective) - _OBJECTIVE_KEYS:
        raise ValueError("Unknown PhaseContactMapLoss objective settings")
    # Full evidenced first-touch activation; adjacent phase blending is opt-in.
    objective.setdefault("activation_ramp_frames", 1)
    weights = result.setdefault("weights", {})
    if not isinstance(weights, dict) or set(weights) - set(_WEIGHTS):
        raise ValueError("Unknown repair objective weights")
    weights = dict(_WEIGHTS, **weights)
    for name, value in weights.items():
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"weights.{name} must be finite and nonnegative")
    if weights["contact"] <= 0:
        raise ValueError("weights.contact must be positive for a contact repair")
    result["weights"] = weights
    ids = result.setdefault("foot_joint_ids", [])
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("foot_joint_ids must be explicit native nonnegative joint IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("foot_joint_ids must be unique")
    return result


def _geometry(value, frames, *, reference=None):
    result = {name: np.asarray(value[name]) for name in ("vertices", "joints", "faces", "frames")}
    if not np.array_equal(result["frames"], frames):
        raise ValueError("Native evaluator returned a different source-frame schedule")
    for name in ("vertices", "joints"):
        array = result[name]
        if (
            array.ndim != 3
            or array.shape[0] != len(frames)
            or array.shape[2] != 3
            or not array.shape[1]
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"Invalid native {name}")
        if reference is not None and array.shape[1:] != reference[name].shape[1:]:
            raise ValueError("Native evaluator changed geometry topology")
    faces = result["faces"]
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or faces.min() < 0
        or faces.max() >= result["vertices"].shape[1]
    ):
        raise ValueError("Native evaluator returned invalid human faces")
    if reference is not None and not np.array_equal(faces, reference["faces"]):
        raise ValueError("Native evaluator changed human topology")
    return result


def _projection(points, scene):
    camera = np.asarray(scene["camera_matrix_world"])
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    projected = local @ np.asarray(scene["camera_intrinsics"]).T
    valid = local[..., 2] > 1e-8
    pixels = np.zeros_like(projected[..., :2], dtype=np.float64)
    np.divide(projected[..., :2], projected[..., 2, None], out=pixels, where=valid[..., None])
    return pixels, valid


def _rotation_difference(candidate, source):
    left = rotvec_to_quaternion(np.asarray(source).reshape(len(source), -1, 3))
    left[..., 1:] *= -1
    right = rotvec_to_quaternion(np.asarray(candidate).reshape(len(candidate), -1, 3))
    return quaternion_to_rotvec(quaternion_multiply(left, right))


def repair_plan(base_motion, seed_plan, scene, model_config, repair, evidence, *, evaluator=None):
    """Return ``(cumulative_proposal_plan, UNREVIEWED_report)``.

    ``base_motion`` is the immutable original native motion, never the edited
    seed checkpoint. The injected evaluator has the same positional arguments
    and ``frames`` keyword as native.evaluate_motion; production uses that
    adapter with a caller-owned model cache. Evidence requires an actual
    ``source_video`` path, its ``source_video_sha256`` and ``sequence``.

    ``max_nfev`` counts every scalar objective call including finite-difference
    probes. Two setup geometry calls (original and seed) are reported separately.
    The wall-clock cap is checked between calls; the outer worker timeout is
    needed to interrupt a blocked native model invocation.
    """
    import torch
    from scipy.optimize import minimize

    from grail.optimization.fixed_contact import build_binding
    from grail.optimization.phase_contact_map_loss import PhaseContactMapLoss

    started = time.monotonic()
    metadata = validate_motion(base_motion)
    validate_plan(base_motion, seed_plan)
    validate_scene(scene)
    for field in ("obj_R", "obj_t"):
        if field in scene and not np.array_equal(scene[field], base_motion[field]):
            raise ValueError(f"Scene {field} differs from immutable original object track")
    if "object_scale" in scene and not np.array_equal(
        scene["object_scale"], base_motion["obj_scale"]
    ):
        raise ValueError("Scene object_scale differs from immutable original object scale")
    if model_config.get("model_type") != metadata["model_type"]:
        raise ValueError("Repair model_type differs from immutable original motion")
    if model_config.get("backend") != "native":
        raise ValueError("Repair requires an explicit supported native evaluator backend")
    cfg = _config(repair)
    source_file = binding(evidence["source_video"])
    if source_file["sha256"] != evidence.get("source_video_sha256"):
        raise ValueError("Actual source video sha256 differs from evidence")
    map_file = binding(cfg["contact_map"])
    # No new keys, joints, windows, timing, bounds or endpoints are authored here.
    slots = [
        (wi, ki) for wi, w in enumerate(seed_plan["windows"]) for ki in range(1, len(w["keys"]) - 1)
    ]
    if not slots:
        raise ValueError("Local repair needs at least one existing interior native key")
    seed_values = np.asarray(
        [seed_plan["windows"][w]["keys"][k]["delta_rotvec"] for w, k in slots], dtype=np.float64
    )
    limits = np.asarray([seed_plan["windows"][w]["max_angle_rad"] for w, _ in slots])
    editable = np.zeros((metadata["frame_count"], metadata["joint_count"]), dtype=bool)
    for window in seed_plan["windows"]:
        if len(window["keys"]) > 2:
            editable[int(window["start"]) + 1 : int(window["end"]), window["joint"]] = True
    active = np.flatnonzero(editable.any(axis=1))
    if not len(active):
        raise ValueError("Repair keys must affect at least one interior source-rate frame")
    cache = {}
    native_evaluations = 0

    def evaluate(motion, frames, reference=None):
        nonlocal native_evaluations
        native_evaluations += 1
        if evaluator is None:
            result = evaluate_motion(motion, model_config, frames=frames, model_cache=cache)
        else:
            result = evaluator(motion, model_config, frames=frames)
        return _geometry(result, frames, reference=reference)

    all_frames = np.arange(metadata["frame_count"], dtype=np.float64)
    original = evaluate(base_motion, all_frames)
    seed_motion = apply_edits(base_motion, seed_plan)
    seed = evaluate(seed_motion, all_frames, reference=original)
    foot_ids = cfg["foot_joint_ids"]
    if foot_ids and max(foot_ids) >= seed["joints"].shape[1]:
        raise ValueError("foot_joint_ids exceed evaluated native joint count")
    original_pixels, source_visible = _projection(original["joints"], scene)
    if cfg["weights"]["source"] and not source_visible.any():
        raise ValueError("Source projection preservation requires visible original native joints")
    obj_vertices = np.asarray(scene["object_vertices"]) * np.asarray(base_motion["obj_scale"])
    contact_binding = build_binding(
        sequence=evidence["sequence"],
        source_video_sha256=source_file["sha256"],
        frame_count=metadata["frame_count"],
        fps=metadata["fps"],
        human_model_type=metadata["model_type"],
        human_vertex_count=original["vertices"].shape[1],
        human_faces=original["faces"],
        object_vertices=obj_vertices,
        object_faces=scene["object_faces"],
    )
    contact = PhaseContactMapLoss.from_file(
        cfg["contact_map"],
        binding=contact_binding,
        object_vertices=obj_vertices,
        object_faces=scene["object_faces"],
        human_faces=original["faces"],
        device="cpu",
        dtype=torch.float64,
        **cfg["objective"],
    )
    active_contact = np.zeros(metadata["frame_count"], dtype=bool)
    for schedule in contact.provenance["runtime_schedule"].values():
        active_contact |= np.asarray(schedule["hand_activation"]) > 0
    if not active_contact[active].any():
        raise ValueError("Repair windows contain no video-authored active contact frames")
    object_R = torch.as_tensor(np.asarray(base_motion["obj_R"]), dtype=torch.float64)
    object_t = torch.as_tensor(np.asarray(base_motion["obj_t"]), dtype=torch.float64)
    diagonal2 = float(scene["image_width"]) ** 2 + float(scene["image_height"]) ** 2

    def make_plan(flat):
        values = np.asarray(flat).reshape(-1, 3).copy()
        # Project into the original per-key ball, then the seed-centered ball.
        # The latter is a convex interpolation between two points in the first.
        norms = np.linalg.norm(values, axis=1)
        values *= np.minimum(1, limits / np.maximum(norms, 1e-30))[:, None]
        delta = values - seed_values
        norms = np.linalg.norm(delta, axis=1)
        values = (
            seed_values
            + delta * np.minimum(1, cfg["max_delta_rad"] / np.maximum(norms, 1e-30))[:, None]
        )
        plan = copy.deepcopy(seed_plan)
        for (window, key), value in zip(slots, values):
            plan["windows"][window]["keys"][key]["delta_rotvec"] = value.tolist()
        validate_plan(base_motion, plan)
        return plan

    def terms(motion, geometry):
        with torch.no_grad():
            contact_cost = float(
                contact(
                    torch.as_tensor(geometry["vertices"], dtype=torch.float64), object_R, object_t
                )
            )
        posture = _rotation_difference(motion["poses"], base_motion["poses"])[editable]
        displacement = geometry["joints"] - seed["joints"]
        temporal = sum(
            float(np.mean(np.diff(displacement, n=n, axis=0) ** 2))
            for n in (2, 3)
            if len(displacement) > n
        )
        pixels, visible = _projection(geometry["joints"], scene)
        squared = np.sum((pixels - original_pixels) ** 2, axis=-1) / diagonal2
        squared[source_visible & ~visible] = 100.0
        return dict(
            contact=contact_cost,
            posture=float(np.mean(np.sum(posture**2, axis=-1))),
            temporal=temporal,
            source=float(np.mean(squared[source_visible])) if source_visible.any() else 0.0,
            feet=float(np.mean(displacement[:, foot_ids] ** 2)) if foot_ids else 0.0,
        )

    initial_terms = terms(seed_motion, seed)
    initial_cost = sum(cfg["weights"][k] * v for k, v in initial_terms.items())
    if not np.isfinite(initial_cost):
        raise ValueError("Nonfinite repair seed objective")
    best = dict(plan=copy.deepcopy(seed_plan), terms=initial_terms, cost=initial_cost)
    nfev = 0
    trace = []
    evaluated = {seed_values.tobytes(): (initial_cost, initial_terms)}

    def objective(flat):
        nonlocal nfev, best
        if nfev >= cfg["max_nfev"]:
            raise _BudgetExhausted("max_nfev")
        if time.monotonic() - started >= cfg["max_time_s"]:
            raise _BudgetExhausted("max_time_s")
        nfev += 1
        proposal = make_plan(flat)
        values = np.asarray(
            [proposal["windows"][w]["keys"][k]["delta_rotvec"] for w, k in slots], dtype=np.float64
        )
        key = values.tobytes()
        if key in evaluated:
            cost, components = evaluated[key]
        else:
            motion = apply_edits(base_motion, proposal)
            local = evaluate(motion, active, reference=original)
            geometry = {name: seed[name].copy() for name in ("vertices", "joints")}
            for name in geometry:
                geometry[name][active] = local[name]
            components = terms(motion, geometry)
            cost = sum(cfg["weights"][k] * v for k, v in components.items())
            if not np.isfinite(cost):
                raise ValueError("Nonfinite native repair objective")
            evaluated[key] = (cost, components)
        if cost < best["cost"]:
            best = dict(plan=proposal, terms=components, cost=cost)
        trace.append(dict(nfev=nfev, objective=cost, best_objective=best["cost"]))
        # Regional squared-metre costs are small. A fixed seed normalization
        # keeps L-BFGS stopping tolerances from treating real gradients as zero.
        return cost / max(abs(initial_cost), 1e-10)

    stop_reason, solver_message = "solver_terminated", ""
    converged = False
    try:
        result = minimize(
            objective,
            seed_values.ravel(),
            method="L-BFGS-B",
            bounds=[
                (v - cfg["max_delta_rad"], v + cfg["max_delta_rad"]) for v in seed_values.ravel()
            ],
            options=dict(
                eps=cfg["finite_difference_rad"],
                maxfun=cfg["max_nfev"],
                maxiter=cfg["max_nfev"],
                ftol=1e-10,
                gtol=1e-7,
                maxls=10,
            ),
        )
        converged = bool(result.success)
        solver_message = str(result.message)
        if nfev >= cfg["max_nfev"]:
            stop_reason = "max_nfev"
        elif converged:
            stop_reason = "solver_converged"
    except _BudgetExhausted as exc:
        stop_reason = str(exc)
        solver_message = "Execution budget ended; retained best evaluated feasible proposal"
    # Bind the input snapshot across the complete solve, not just at startup.
    verify_binding(source_file)
    verify_binding(map_file)
    proposal_motion = apply_edits(base_motion, best["plan"])
    for name, original_value in base_motion.items():
        if (
            name != "poses"
            and np.asarray(original_value).tobytes() != proposal_motion[name].tobytes()
        ):
            raise RuntimeError(f"Repair changed frozen field {name}")
    original_poses = np.asarray(base_motion["poses"]).reshape(editable.shape + (3,))
    proposal_poses = proposal_motion["poses"].reshape(editable.shape + (3,))
    # Other seed windows may contain corrections but have no optimizable keys.
    seed_poses = seed_motion["poses"].reshape(editable.shape + (3,))
    if proposal_poses[~editable].tobytes() != seed_poses[~editable].tobytes():
        raise RuntimeError("Repair changed protected seed native controls")
    root_and_frozen = sorted(
        set(range(metadata["joint_count"])) - set(metadata["editable_body_joints"])
    )
    if proposal_poses[:, root_and_frozen].tobytes() != original_poses[:, root_and_frozen].tobytes():
        raise RuntimeError("Repair changed frozen original native controls")
    proposed_values = np.asarray(
        [best["plan"]["windows"][w]["keys"][k]["delta_rotvec"] for w, k in slots]
    )
    report = dict(
        schema="grail.hoi_local_repair.v1",
        status="UNREVIEWED",
        review_status="UNREVIEWED",
        accepted=False,
        stop_reason=stop_reason,
        solver_converged=converged,
        solver_message=solver_message,
        solver="projected L-BFGS-B with explicit forward finite differences",
        max_nfev=cfg["max_nfev"],
        nfev=nfev,
        native_evaluations=native_evaluations,
        setup_native_evaluations=2,
        elapsed_s=time.monotonic() - started,
        parameter_count=int(seed_values.size),
        active_source_frames=active.tolist(),
        max_seed_key_delta_rad=float(np.linalg.norm(proposed_values - seed_values, axis=1).max()),
        objective_before=initial_cost,
        objective_after=best["cost"],
        objective_reduced=best["cost"] < initial_cost,
        terms_before=initial_terms,
        terms_after=best["terms"],
        trace=trace,
        config=cfg,
        bindings=dict(
            source_video=source_file, contact_map=map_file, native_contact=contact_binding
        ),
        contact_objective=contact.provenance,
        invariants=dict(
            cumulative_plan_relative_to="immutable_original",
            existing_interior_keys_only=True,
            protected_seed_controls_exact=True,
            frozen_original_native_fields_exact=True,
            endpoints_exact=True,
        ),
        term_semantics=dict(
            contact="existing dense PhaseContactMapLoss, m^2; independent regional fields",
            posture="source-relative local rotation geodesic squared, mean rad^2 over edited controls",
            temporal="seed-relative native joint displacement second and third source-frame differences; mean m^2/frame^4 plus m^2/frame^6",
            source="original native joint projection displacement divided by image diagonal squared; not measured RGB keypoint error",
            feet=(
                "seed-relative configured native foot joint positions, mean m^2"
                if foot_ids
                else "NOT_CONFIGURED"
            ),
        ),
        collision=dict(
            status="AUDIT_ONLY",
            optimized=False,
            reason="No collision objective in this proposal; whole original-mesh post-audit required",
        ),
        limitations=[
            "Objective reduction or solver termination does not accept contact or collision quality",
            "Finite differences operate on native float32 geometry with the explicit reported step",
            "Only existing keys are optimized; insufficient inner budget is not task completion",
            "Source-rate samples only; subframe contact and collision require independent review",
            "Model call cannot be interrupted by internal wall-clock checks; enforce worker timeout",
        ],
    )
    return best["plan"], report
