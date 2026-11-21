"""Couple fixed material holds and editable acquisition/release buffers.

The mounted proposal is an initialization, never an accepted grasp. This route
keeps the original native parent/prior and calls the existing transition solver
and original-mesh audit. It does not run wrist-route IK or pin a first-touch key.
"""

import argparse
import copy
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .contact_edit import native_io
from .contact_edit.pose import _keys, _sha
from .grasp_io import number
from .motion_audit import AuditContext
from .motion_diagnostics import timestamps
from . import transition_repair


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _absolute_repair(config, directory):
    """Relocate declared file inputs without changing their policy or bytes."""
    result = copy.deepcopy(config)
    for key in ("model_xml", "trajectory_npz", "object_mesh_npz"):
        result[key] = str((directory / result[key]).resolve(strict=True))
    for item in result.get("environment", []):
        item["mesh_npz"] = str((directory / item["mesh_npz"]).resolve(strict=True))
    diagnostic = result.get("diagnostics", {})
    if "baseline_npz" in diagnostic:
        diagnostic["baseline_npz"] = str((directory / diagnostic["baseline_npz"]).resolve(strict=True))
    if "human_reference" in diagnostic:
        ref = diagnostic["human_reference"]
        ref["npz"] = str((directory / ref["npz"]).resolve(strict=True))
    return result


def _routes(context, config):
    """Bind diagnostic wrists and assign existing native windows to controls.

    No path is inferred: native_input bypasses route/blend initialization and
    the route weight is zero. Shared ancestors and fingers use blend ownership.
    """
    wrists = config["wrist_bodies"]
    if (not isinstance(wrists, list) or not wrists
            or any(not isinstance(n, str) for n in wrists)
            or len(set(wrists)) != len(wrists)):
        raise ValueError("wrist_bodies requires unique native body names")
    ancestry = {}
    for name in wrists:
        body = mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body <= 0:
            raise ValueError("Unknown native wrist body: " + name)
        chain = set()
        while body:
            chain.add(body)
            body = int(context.model.body_parentid[body])
        ancestry[name] = chain
    owners = {name: [w for w in wrists if int(context.model.jnt_bodyid[j]) in ancestry[w]]
              for name, j in zip(context.names, context.jids)}
    routes, claimed = [], set()
    for wrist in wrists:
        groups = {}
        for name in context.names:
            if owners[name] == [wrist]:
                groups.setdefault(tuple(config["joint_windows"][name]), []).append(name)
        if not groups:
            raise ValueError("Each wrist needs an editable local arm ancestor: " + wrist)
        for window, names in groups.items():
            routes.append(dict(body=wrist, joints=names, window=list(window), waypoints=[]))
            claimed.update(names)
    groups = {}
    for name in context.names:
        if name not in claimed:
            groups.setdefault(tuple(config["joint_windows"][name]), []).append(name)
    blends = [dict(joints=names, window=list(window)) for window, names in groups.items()]
    return routes, blends


def prepare(config, directory=".", *, cache_dir=None):
    """Bind an explicit seed, fixed contact phases and editable boundary buffers."""
    _keys(config, {"schema_version", "repair_config", "seed_native", "seed_sha256",
                   "seed_review", "holds", "joint_windows", "wrist_bodies", "solver"},
          {"collision_scope", "contact_substeps", "posture_reference"}, "grasp sequence")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported grasp sequence schema")
    directory = Path(directory)
    path = (directory / config["repair_config"]).resolve(strict=True)
    seed_path = (directory / config["seed_native"]).resolve(strict=True)
    review = config["seed_review"]
    _keys(review, {"status", "reason"}, set(), "seed review")
    if review["status"] not in ("DIAGNOSTIC", "REVIEWED_PARTIAL", "UNREVIEWED"):
        raise ValueError("Seed review must explicitly retain diagnostic/partial status")
    if not isinstance(review["reason"], str) or not review["reason"].strip():
        raise ValueError("Seed review requires the reason for using this initialization")
    repair = _absolute_repair(json.loads(path.read_text()), path.parent)
    if "iteration" in repair:
        raise ValueError("Declare the sequence seed explicitly; repair already has an iteration seed")
    repair["iteration"] = dict(seed_npz=str(seed_path), seed_sha256=config["seed_sha256"],
                               wrist_bodies=config["wrist_bodies"], preserve_smoothness=False)
    context = AuditContext(repair, path.parent, cache_dir=cache_dir)
    if not context.joint_velocity_gate.configured:
        raise ValueError("A whole-motion max_velocity_rad_s hard gate is required")
    if not isinstance(config["joint_windows"], dict) or set(config["joint_windows"]) != set(context.names):
        raise ValueError("joint_windows must cover exactly the editable joints")
    solver = copy.deepcopy(config["solver"])
    if not isinstance(solver, dict):
        raise ValueError("solver must be an object")
    solver.setdefault("derivative_units", "seconds")
    solver.setdefault("velocity_projection", True)
    solver.setdefault("contact_mode", "target")
    solver.setdefault("projection_scope", "editable_hinges")
    solver.setdefault("weights", {}).setdefault("route", 0.)
    if solver["weights"]["route"] != 0:
        raise ValueError("Sequence repair does not preserve a raw wrist route")
    if (solver["derivative_units"] != "seconds" or solver["velocity_projection"] is not True
            or not solver.get("temporal_projection")):
        raise ValueError("Sequence repair requires seconds-based velocity/acceleration/jerk projection")
    if solver["contact_mode"] != "target" or solver["weights"].get("contact", 100.) <= 0:
        raise ValueError("Sequence repair requires a positive fixed-material target objective")
    routes, blends = _routes(context, config)
    transition = dict(repair_config="repair_config.json", joint_windows=config["joint_windows"],
                      routes=routes, joint_blends=blends, seed=dict(native_input=True), solver=solver)
    if "collision_scope" in config:
        transition["collision_scope"] = config["collision_scope"]
    if "posture_reference" in config:
        reference = copy.deepcopy(config["posture_reference"])
        reference["native_npz"] = str((directory / reference["native_npz"]).resolve(strict=True))
        transition["posture_reference"] = reference
    mask, _ = transition_repair.prepare(context, transition, directory)
    if not np.array_equal(context.initial_qpos[~mask], context.parent["qpos"][~mask]):
        raise ValueError("Sequence seed changes samples outside the declared joint buffers")
    contacts = {item["name"]: item for item in context.contacts}
    holds, named, used = config["holds"], set(), set()
    if not isinstance(holds, list) or not holds:
        raise ValueError("At least one explicit fixed-material hold is required")
    for hold in holds:
        _keys(hold, {"name", "window", "contact_names", "joint_names"}, {"max_material_slip_m"}, "hold")
        name, window = hold["name"], hold["window"]
        if not isinstance(name, str) or not name or name in named:
            raise ValueError("Hold names must be unique")
        named.add(name)
        if (not isinstance(window, list) or len(window) != 2
                or any(type(f) is not int for f in window)
                or not context.start <= window[0] < window[1] <= context.end):
            raise ValueError("Hold window must be inside the coupled repair window (inclusive)")
        names = hold["contact_names"]
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or not set(names) <= set(contacts) or used.intersection(names)):
            raise ValueError("Each fixed contact must belong to exactly one hold")
        used.update(names)
        if any(not np.array_equal(contacts[n]["active"], window) for n in names):
            raise ValueError("Hold windows must exactly match fixed contact active_frames")
        joints = hold["joint_names"]
        if (not isinstance(joints, list) or not joints or len(set(joints)) != len(joints)
                or not set(joints) <= set(context.names)):
            raise ValueError("Hold joint_names must name its editable native controls")
        for joint in joints:
            lo, hi = config["joint_windows"][joint]
            address = context.qids[context.names.index(joint)]
            if lo > window[0] or hi < window[1]:
                raise ValueError("Joint buffer does not cover the complete hold: " + joint)
            if window[0] > 0 and (lo >= window[0] or not mask[window[0], address]):
                raise ValueError("Acquisition needs an editable first hold sample and preceding buffer")
            if window[1] < context.T - 1 and (hi <= window[1] or not mask[window[1], address]):
                raise ValueError("Release needs an editable last hold sample and following buffer")
        if "max_material_slip_m" in hold:
            number(hold["max_material_slip_m"], "hold material slip", 1e-12)
    if used != set(contacts):
        raise ValueError("Every configured material contact must have an explicit hold")
    substeps = config.get("contact_substeps", max(4, context.substeps))
    if type(substeps) is not int or not context.substeps <= substeps <= 16:
        raise ValueError("contact_substeps must cover at least the audit substeps, at most 16")
    return context, repair, transition, mask, substeps


def material_review(context, qpos, holds, substeps):
    """Complete fixed-anchor review, independent of an early geometry rejection."""
    records = []
    for hold in holds:
        lo, hi = hold["window"]
        frames = (np.arange(lo * substeps, hi * substeps + 1) / substeps).tolist()
        for contact in context.contacts:
            if contact["name"] not in hold["contact_names"]:
                continue
            offsets = []
            for frame in frames:
                q, rotation, translation = context.sample(qpos, frame)
                context.forward(q)
                point = context.point(contact["body"], contact["local"])
                offsets.append(rotation.T @ (point - translation) - contact["target_local"])
            offsets = np.asarray(offsets)
            gaps = np.linalg.norm(offsets, axis=1)
            delta = offsets - offsets[0]
            drift = np.linalg.norm(delta, axis=1)
            normal = contact["normal_local"]
            tangent = delta - (delta @ normal)[:, None] * normal
            max_drift = hold.get("max_material_slip_m")
            records.append(dict(name=contact["name"], hold=hold["name"],
                                source_frames=frames, gap_m=gaps.tolist(),
                                max_gap_m=float(gaps.max()), gap_limit_m=contact["max_gap_m"],
                                gap_passed=bool(gaps.max() <= contact["max_gap_m"] + 1e-10),
                                object_relative_drift_m=drift.tolist(),
                                max_object_relative_drift_m=float(drift.max()),
                                max_material_slip_m=float(drift.max()),
                                material_slip_range_diameter_upper_bound_m=float(np.linalg.norm(np.ptp(offsets, axis=0))),
                                slip_gate_status="NOT_CONFIGURED" if max_drift is None else "CONFIGURED",
                                max_tangential_drift_m=float(np.linalg.norm(tangent, axis=1).max()),
                                drift_limit_m=max_drift, material_slip_limit_m=max_drift,
                                drift_passed=None if max_drift is None else bool(drift.max() <= max_drift + 1e-10)))
    return dict(complete=True, passed=all(r["gap_passed"] and r["drift_passed"] is not False for r in records),
                contacts=records, substeps=substeps,
                slip_gate_status=("CONFIGURED" if all(r["drift_passed"] is not None for r in records)
                                  else "PARTIALLY_CONFIGURED" if any(r["drift_passed"] is not None for r in records)
                                  else "NOT_CONFIGURED"),
                metric="Fixed robot material point relative to the same object triangle anchor; drift relative to first hold sample",
                limitations=["Finite sampled contact points; not force closure, continuous contact or full rigid wrist-pose equality.",
                             "Drift is diagnostic unless a hold declares max_material_slip_m; source/visual and whole-mesh gates remain separate."])


def run(config_path, output, *, cache_dir=None, proposal_only=False, progress=None):
    started = time.monotonic()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        path = Path(config_path).resolve(strict=True)
        config = json.loads(path.read_text())
        context, repair, transition, mask, substeps = prepare(config, path.parent, cache_dir=cache_dir)
        original_repair = (path.parent / config["repair_config"]).resolve(strict=True)
        hashes = {str(p): _sha(p) for p in (path, original_repair, Path(__file__))}
        _write(out / "config.json", config)
        _write(out / "repair_config.json", repair)
        _write(out / "transition_config.json", transition)
        seed = native_io.load_npz(repair["iteration"]["seed_npz"])
        seed_review = dict(declaration=config["seed_review"], motion_accepted=False,
                           archive_status=str(seed.get("status", seed.get("native_status", "UNKNOWN"))),
                           joint_velocity=context.joint_velocity_gate.evaluate(context.initial_qpos, context.fps,
                               timestamps(context.parent), mask),
                           posture_reference=(context.posture_reference.evaluate(context.initial_qpos)
                                              if context.posture_reference else None),
                           material=material_review(context, context.initial_qpos, config["holds"], substeps))
        _write(out / "seed_review.json", seed_review)
        report = transition_repair.run(out / "transition_config.json", out / "coupled",
            cache_dir=cache_dir, proposal_only=proposal_only, progress=progress)
        report.update(schema_version=1, tool="grasp_sequence", seed_review=seed_review,
                      holds=config["holds"], joint_windows=config["joint_windows"],
                      fitting_window=[context.start, context.end],
                      fitting_substeps=transition["solver"].get("fit_substeps", 2),
                      protected_frames=context.protected,
                      source_visual_review_status="NOT_PERFORMED_BY_THIS_TOOL",
                      sequence_input_sha256=hashes, sequence_candidate=None)
        candidate_path = out / "coupled/candidate_UNREVIEWED.npz"
        if candidate_path.is_file():
            candidate = native_io.load_npz(candidate_path)
            material = material_review(context, candidate["qpos"], config["holds"], substeps)
            _write(out / "material_review.json", material)
            report["material_review"] = material
            velocity = context.joint_velocity_gate.evaluate(candidate["qpos"], context.fps,
                                                            timestamps(context.parent), mask)
            report["full_motion_joint_velocity"] = velocity
            if not velocity["passed"] and not report["status"].startswith("REJECTED"):
                report["status"] = "REJECTED_BY_JOINT_VELOCITY"
            report["hold_gate_status"] = "PASS" if material["passed"] else "FAIL"
            if not material["passed"] and not report["status"].startswith("REJECTED"):
                report["status"] = "REJECTED_BY_HOLD_GATE"
            elif report["status"] == "UNREVIEWED_TRANSITION":
                report["status"] = "UNREVIEWED_GRASP_SEQUENCE"
            candidate["status"] = np.asarray(report["status"])
            target = out / "candidate_UNREVIEWED.npz"
            np.savez_compressed(target, **candidate)
            loaded = native_io.load_npz(target)
            if any(not np.array_equal(v, loaded[k]) for k, v in candidate.items()):
                raise RuntimeError("Sequence candidate serialization changed arrays")
            report["candidate_sha256"] = _sha(target)
            report["sequence_candidate"] = str(target)
        report["motion_accepted"] = report["selected"] = False
        report["limitations"] = report.get("limitations", []) + [
            "One coupled window with fixed material identities; source roles/anchors are authored hypotheses.",
            "A rejected or temporally invalid seed is initialization only; no seed approval transfers to the candidate.",
            "Integer/sample gates do not establish continuous-time clearance or source/visual acceptance."]
        context.check_inputs_unchanged()
        if any(_sha(p) != value for p, value in hashes.items()):
            raise ValueError("Sequence input or implementation changed during run")
        report["elapsed_seconds"] = time.monotonic() - started
        _write(out / "report.json", report)
        return report
    except Exception as exc:
        _write(out / "error.json", dict(status="FAILED", motion_accepted=False,
                                       type=type(exc).__name__, message=str(exc)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--proposal-only", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.config, args.output, cache_dir=args.cache, proposal_only=args.proposal_only,
                 progress=lambda r: print(json.dumps(r), flush=True))
    print(json.dumps({key: report.get(key) for key in ("status", "audit_status", "elapsed_seconds")}))
    return 0 if ((report.get("audit_after") or {}).get("passed") is True
                 and report.get("material_review", {}).get("passed") is True
                 and report.get("full_motion_joint_velocity", {}).get("passed") is True
                 and (report.get("derivative_gate") or {}).get("passed") is True
                 and (report.get("posture_reference_gate") is None
                      or report["posture_reference_gate"]["passed"] is True)
                 and not report.get("diagnostic_only", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
