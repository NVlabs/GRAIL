"""Coupled short-window native repair with audit-derived clearance witnesses.

Fixed original contact anchors, protected poses and source-relative limits are
never changed by this solver. Locally oriented witness halfspaces guide a repair;
original-triangle audits, including independent subframes, decide numeric scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from .contact_edit import native_io
from .contact_edit.pose import PoseResult, _integer, _keys, _weight
from .motion_audit import AuditContext, digest, skew
from .repair_geometry import mesh_intersections


class WindowProblem:
    def __init__(self, context):
        self.context = context
        config = context.config.get("solver", {})
        _keys(
            config,
            set(),
            {
                "max_evaluations",
                "max_rounds",
                "contact_weight",
                "normal_weight",
                "pose_weight",
                "acceleration_weight",
                "clearance_weight",
                "source_weight",
                "contact_mode",
                "fit_substeps",
            },
            "solver",
        )
        self.max_evaluations = _integer(
            config.get("max_evaluations", 80), "max_evaluations", 1
        )
        self.max_rounds = _integer(config.get("max_rounds", 4), "max_rounds", 1)
        if self.max_evaluations > 10000 or self.max_rounds > 30:
            raise ValueError("Solver trial budget exceeds supported range")
        self.weights = {
            key: _weight(config.get(key, value), key)
            for key, value in [
                ("contact_weight", 1000.0),
                ("normal_weight", 1.0),
                ("pose_weight", 0.1),
                ("acceleration_weight", 1.0),
                ("clearance_weight", 2000.0),
                ("source_weight", 0.1),
            ]
        }
        self.contact_mode = config.get("contact_mode", "target")
        if self.contact_mode not in ("target", "within_gates"):
            raise ValueError("Unknown contact_mode")
        substeps = _integer(config.get("fit_substeps", 2), "fit_substeps", 1)
        if substeps > 32:
            raise ValueError("fit_substeps exceeds 32")
        self.fit_times = set(
            np.arange(context.start * substeps, context.end * substeps + 1) / substeps
        )
        self.fit_times.update(
            g["frame"]
            for g in context.guides
            if context.start <= g["frame"] <= context.end
        )
        self.index = {int(f): i for i, f in enumerate(context.frames)}
        self.D = len(context.qids)
        self.N = len(context.frames) * self.D
        self.initial = (
            context.initial_qpos[np.ix_(context.frames, context.qids)]
            .astype(float)
            .ravel()
        )
        self.original = context.parent["qpos"][np.ix_(context.frames, context.qids)].astype(float).ravel()
        self.lower = context.lower.ravel()
        self.upper = context.upper.ravel()
        self.clearance = []
        self.clearance_keys = set()
        self.unresolved = []
        self.reference_checks = {}
        self._last_x = None
        self._last_eval = None

    def qpos(self, x):
        q = self.context.parent["qpos"].astype(float).copy()
        q[np.ix_(self.context.frames, self.context.qids)] = np.asarray(x).reshape(
            -1, self.D
        )
        return q

    def invalidate(self):
        self._last_x = None
        self._last_eval = None

    def add_failures(self, audit, qpos):
        """Bind witness material coordinates and safe-side choice to original data."""
        c = self.context
        added = 0
        for failure in audit["failures"]:
            frame = failure.get("source_frame")
            if frame is None or not c.start < frame < c.end:
                continue
            self.fit_times.add(frame)
            if failure["kind"] != "collision":
                continue
            a, b = failure["pair"]
            pair = next(p for p in c.pairs if [p["a"], p["b"]] == [a, b])
            # Put an editable robot side first when the caller lists object first.
            reverse = c.meshes[a]["body"] is None
            if reverse:
                a, b = b, a
            if c.meshes[a]["body"] is None:
                self.unresolved.append(
                    dict(
                        frame=frame,
                        pair=[a, b],
                        reason="Neither side has editable native kinematics",
                    )
                )
                continue
            qa, Ra, ta = c.sample(qpos, frame)
            c.forward(qa)
            current_A = c.transform(a, Ra, ta)
            current_B = c.transform(b, Ra, ta)
            qr, Rr, tr = c.sample(c.parent["qpos"], pair["reference_frame"])
            c.forward(qr)
            ref_A = c.transform(a, Rr, tr)
            ref_B = c.transform(b, Rr, tr)
            ma, mb = c.meshes[a]["mesh"], c.meshes[b]["mesh"]
            reference_key = (a, b, pair["reference_frame"])
            if reference_key not in self.reference_checks:
                self.reference_checks[reference_key] = mesh_intersections(
                    ma.transformed(ref_A), mb.transformed(ref_B)
                )
            reference_check = self.reference_checks[reference_key]
            if reference_check["intersecting"] or not reference_check["complete"]:
                self.unresolved.append(
                    dict(
                        frame=frame,
                        pair=[a, b],
                        reference_frame=pair["reference_frame"],
                        reason="Reference pair is intersecting or incomplete; no safe side can be inferred",
                    )
                )
                continue
            for witness in failure["witnesses"]:
                fa = int(witness["face_b" if reverse else "face_a"])
                fb = int(witness["face_a" if reverse else "face_b"])
                bary = np.asarray(
                    witness["barycentric_b" if reverse else "barycentric_a"]
                )
                # Surface witness plus the crossing triangle's material vertices.
                # All remain fixed within this trial; no pad-target relabeling.
                points = [bary @ ma.vertices[ma.faces[fa]]] + list(
                    ma.vertices[ma.faces[fa]]
                )
                tri = mb.vertices[mb.faces[fb]]
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                length = np.linalg.norm(normal)
                if length < 1e-14:
                    continue
                normal /= length
                target = tri.mean(axis=0)
                nr = ref_B[:3, :3] @ normal
                for k, local in enumerate(points):
                    key = (float(frame), a, b, fa, fb, k)
                    if key in self.clearance_keys:
                        continue
                    reference_delta = (ref_A[:3, :3] @ local + ref_A[:3, 3]) - (
                        ref_B[:3, :3] @ target + ref_B[:3, 3]
                    )
                    projection = float(reference_delta @ nr)
                    if abs(projection) < 1e-8:
                        self.unresolved.append(
                            dict(
                                frame=frame,
                                pair=[a, b],
                                face_a=fa,
                                face_b=fb,
                                reason="Reference side is ambiguous",
                            )
                        )
                        continue
                    sign = 1.0 if projection > 0 else -1.0
                    signed = (
                        (
                            current_A[:3, :3] @ local
                            + current_A[:3, 3]
                            - (current_B[:3, :3] @ target + current_B[:3, 3])
                        )
                        @ (current_B[:3, :3] @ normal)
                        * sign
                    )
                    # Far vertices do not define the local crossing neighborhood.
                    if abs(signed) > 0.015 and k:
                        continue
                    self.clearance.append(
                        dict(
                            frame=float(frame),
                            a=a,
                            b=b,
                            local=np.asarray(local),
                            target=target.copy(),
                            normal=normal.copy() * sign,
                            margin=pair["clearance_m"],
                            reference_frame=pair["reference_frame"],
                            face_a=fa,
                            face_b=fb,
                        )
                    )
                    self.clearance_keys.add(key)
                    added += 1
        self.invalidate()
        return added

    @staticmethod
    def _outside(vector, gate):
        length = np.linalg.norm(vector)
        if length <= gate:
            return np.zeros_like(vector), np.zeros((len(vector), len(vector)))
        factor = 1 - gate / length
        return vector * factor, factor * np.eye(len(vector)) + gate * np.outer(
            vector, vector
        ) / length**3

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if self._last_x is not None and np.array_equal(x, self._last_x):
            return self._last_eval
        c = self.context
        qseq = self.qpos(x)
        residual = []
        blocks = []

        def append(value, jac, frame):
            value = np.atleast_1d(value)
            jac = np.atleast_2d(jac)
            full = np.zeros((len(value), self.N))
            f = int(np.floor(frame))
            alpha = frame - f
            for index, weight in ((f, 1 - alpha), (f + 1, alpha)):
                if index in self.index and weight:
                    start = self.index[index] * self.D
                    full[:, start : start + self.D] += weight * jac
            residual.append(value)
            blocks.append(full)

        for frame in sorted(self.fit_times):
            q, R, t = c.sample(qseq, frame)
            c.forward(q)
            for contact in c.contacts:
                if not contact["active"][0] <= frame <= contact["active"][1]:
                    continue
                p, jp, jr = c.point(contact["body"], contact["local"], jac=True)
                n = c.data.xmat[contact["body"]].reshape(3, 3) @ contact["pad_normal"]
                normal = R @ contact["normal_local"]
                target = R @ contact["target_local"] + t
                gap = p - target
                nj = -skew(n) @ jr
                nd = n + normal
                if self.contact_mode == "within_gates":
                    value, derivative = self._outside(gap, 0.95 * contact["max_gap_m"])
                    position_jac = derivative @ jp
                    nv, nderivative = self._outside(
                        nd, 2 * np.sin(np.radians(0.95 * contact["max_normal_deg"]) / 2)
                    )
                    normal_jac = nderivative @ nj
                else:
                    value = gap - contact["target_gap_m"] * normal
                    position_jac = jp
                    nv = nd
                    normal_jac = nj
                wp = self.weights["contact_weight"] * np.sqrt(
                    contact["position_weight"]
                )
                wn = self.weights["normal_weight"] * np.sqrt(contact["normal_weight"])
                append(wp * value, wp * position_jac, frame)
                append(wn * nv, wn * normal_jac, frame)
            for plane in self.clearance:
                if plane["frame"] != frame:
                    continue
                pa, ja, _ = c.mesh_point(plane["a"], plane["local"], R, t, jac=True)
                pb, jb, jrb = c.mesh_point(plane["b"], plane["target"], R, t, jac=True)
                normal = c.transform(plane["b"], R, t)[:3, :3] @ plane["normal"]
                distance = float((pa - pb) @ normal)
                derivative = normal @ (ja - jb) + (pa - pb) @ (-skew(normal) @ jrb)
                violation = max(plane["margin"] - distance, 0.0)
                w = self.weights["clearance_weight"]
                append(
                    [w * violation],
                    (-w * derivative if violation else np.zeros(self.D))[None, :],
                    frame,
                )
            for guide in c.guides:
                if guide["frame"] != frame:
                    continue
                world, jp, _ = c.point(guide["body"], guide["local"], jac=True)
                cam = c.E[:3, :3] @ world + c.E[:3, 3]
                if cam[2] <= 1e-9:
                    raise ValueError(
                        "Source projection crossed behind camera during repair"
                    )
                projected = c.K @ cam
                uv = projected[:2] / projected[2]
                projjac = (
                    c.K[:2] * projected[2] - projected[:2, None] * c.K[2]
                ) / projected[2] ** 2
                delta, dd = self._outside(
                    uv - guide["uv"], guide["max_error_px"] * 0.95
                )
                w = self.weights["source_weight"]
                append(w * delta, w * dd @ projjac @ c.E[:3, :3] @ jp, frame)
        w = self.weights["pose_weight"]
        residual.append(w * (x - self.original))
        blocks.append(w * np.eye(self.N))
        wa = self.weights["acceleration_weight"]
        for frame in range(max(1, c.start), min(c.T - 1, c.end) + 1):
            if frame >= c.T - 1:
                continue
            value = wa * (
                qseq[frame - 1, c.qids]
                - 2 * qseq[frame, c.qids]
                + qseq[frame + 1, c.qids]
            )
            full = np.zeros((self.D, self.N))
            for f, coefficient in ((frame - 1, 1), (frame, -2), (frame + 1, 1)):
                if f in self.index:
                    start = self.index[f] * self.D
                    full[:, start : start + self.D] = wa * coefficient * np.eye(self.D)
            residual.append(value)
            blocks.append(full)
        # Explicit speed/acceleration gates can guide repair as well as audit.
        for order, key, scale in (
            (1, "max_step_rad", 1.0),
            (2, "max_acceleration_rad_s2", c.fps**2),
        ):
            if key not in c.temporal:
                continue
            coefficients = [-1, 1] if order == 1 else [1, -2, 1]
            for f in range(max(0, c.start - order), min(c.T - order, c.end + 1)):
                values = (
                    sum(
                        coef * qseq[f + k, c.qids]
                        for k, coef in enumerate(coefficients)
                    )
                    * scale
                )
                violation = np.maximum(np.abs(values) - c.temporal[key] * 0.98, 0.0)
                sign = np.sign(values) * (violation > 0) * scale
                full = np.zeros((self.D, self.N))
                for k, coef in enumerate(coefficients):
                    if f + k in self.index:
                        start = self.index[f + k] * self.D
                        full[:, start : start + self.D] = np.diag(sign * coef)
                residual.append(violation)
                blocks.append(full)
        if c.iteration is not None:
            for values, derivatives in c.iteration.derivative_terms(qseq):
                residual.append(values)
                blocks.append(derivatives)
        value = np.concatenate(residual)
        jac = csr_matrix(np.vstack(blocks))
        self._last_x = x.copy()
        self._last_eval = (value, jac)
        return value, jac


def solve_window(config, config_dir=".", *, progress=None, cache_dir=None):
    started = time.perf_counter()
    context = AuditContext(config, config_dir, cache_dir=cache_dir)
    problem = WindowProblem(context)
    initial = context.parent["qpos"]
    before = context.audit()
    x = problem.initial.copy()
    rounds = []
    current = before if context.iteration is None else context.audit(
        {**context.parent, "qpos": context.initial_qpos}
    )
    seed_audit = current
    stop_reason = "TRIAL_BUDGET_EXHAUSTED"
    backend_name = context.repair_backend["name"]
    backend_detail = dict(name=backend_name)
    if backend_name == "material":
        from .repair_material import MaterialOptions, solve_material

        options = MaterialOptions(**{
            key: value for key, value in context.repair_backend.items() if key != "name"
        })
        options.validate_context(context)
        if current["affected_scope_passed"]:
            stop_reason = ("CHECKED_SCOPE_PASS" if current["passed"]
                           else "ONLY_FAILURES_OUTSIDE_EDITABLE_SUPPORT")
            backend_detail["skipped"] = "No failures in editable support"
        else:
            qpos, detail = solve_material(context, options=options, progress=progress)
            backend_detail.update(detail)
            x = np.asarray(qpos)[np.ix_(context.frames, context.qids)].ravel()
            if not np.array_equal(np.asarray(qpos)[~context.mask], initial[~context.mask]):
                raise RuntimeError("Material repair modified frozen native coordinates")
            problem.fit_times = set(detail["fit_source_frames"])
            current = context.audit({**context.parent, "qpos": problem.qpos(x).astype(initial.dtype)})
            stop_reason = ("CHECKED_SCOPE_PASS" if current["passed"]
                           else "ONLY_FAILURES_OUTSIDE_EDITABLE_SUPPORT"
                           if current["affected_scope_passed"] else "TRIAL_BUDGET_EXHAUSTED")
            rounds.append(dict(round=1, backend=backend_name,
                               audit_passed=current["passed"],
                               failure_count=len(current["failures"]),
                               summary=current["summary"], reuse=current["reuse"]))
    for index in range(problem.max_rounds if backend_name == "native" else 0):
        if current["passed"]:
            stop_reason = "CHECKED_SCOPE_PASS"
            break
        if current["affected_scope_passed"]:
            stop_reason = "ONLY_FAILURES_OUTSIDE_EDITABLE_SUPPORT"
            break
        added = problem.add_failures(current, problem.qpos(x))
        solve_start = time.perf_counter()
        optimized = least_squares(
            lambda values: problem.evaluate(values)[0],
            x,
            jac=lambda values: problem.evaluate(values)[1],
            bounds=(problem.lower, problem.upper),
            max_nfev=problem.max_evaluations,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            x_scale="jac",
            tr_options={"atol": 1e-10, "btol": 1e-10},
        )
        x = optimized.x
        native = {k: v.copy() for k, v in context.parent.items()}
        native["qpos"] = problem.qpos(x).astype(initial.dtype)
        current = context.audit(native)
        record = dict(
            round=index + 1,
            optimizer_success=bool(optimized.success),
            evaluations=int(optimized.nfev),
            cost=float(optimized.cost),
            added_clearance_witnesses=added,
            clearance_witnesses=len(problem.clearance),
            fitting_source_frames=sorted(problem.fit_times),
            audit_passed=current["passed"],
            failure_count=len(current["failures"]),
            summary=current["summary"],
            solve_and_audit_seconds=time.perf_counter() - solve_start,
            reuse=current["reuse"],
        )
        rounds.append(record)
        if progress:
            progress(record)
        if current["passed"]:
            stop_reason = "CHECKED_SCOPE_PASS"
            break
        if current["affected_scope_passed"]:
            stop_reason = "ONLY_FAILURES_OUTSIDE_EDITABLE_SUPPORT"
            break
    qpos = problem.qpos(x).astype(initial.dtype)
    if not np.array_equal(qpos[~context.mask], initial[~context.mask]):
        raise RuntimeError("Repair modified frozen native coordinates")
    plan = dict(
        schema_version=1,
        channels=[
            dict(
                joint=name,
                window=[context.start, context.end],
                keys=[
                    [context.start, float(initial[context.start, address])],
                    [context.end, float(initial[context.end, address])],
                ],
                interpolation="LINEAR",
            )
            for name, address in zip(context.names, context.qids)
        ],
        derived_fields=config.get("derived_fields", []),
    )
    candidate, merge = native_io.merge_curve_samples(
        context.parent, context.model_snapshot, plan, qpos
    )
    # Protected interior keys are stricter than the generic curve window.
    if not np.array_equal(candidate["qpos"][~context.mask], initial[~context.mask]):
        raise RuntimeError("Stored repair changed protected samples")
    after = context.audit(candidate)
    status = "UNREVIEWED" if after["passed"] else "REJECTED_BY_AUDIT"
    candidate["status"] = np.asarray(status)
    candidate["selected"] = np.asarray(False)
    candidate["export_kind"] = np.asarray("native_local_repair_" + status)
    candidate["local_repair_parent_sha256"] = np.asarray(
        context.bindings["trajectory_npz"]["sha256"]
    )
    candidate["local_repair_config_sha256"] = np.asarray(context.config_digest)
    context.check_inputs_unchanged()
    report = dict(
        schema_version=1,
        status=status,
        selected=False,
        backend=backend_detail,
        audit_before=before,
        audit_after=after,
        rounds=rounds,
        stop_reason=stop_reason,
        checks=dict(
            frozen_qpos_exact=True,
            metadata_preserved=merge["source_metadata_preserved"],
        ),
        native_merge=merge,
        bindings=context.bindings,
        model_dependencies_sha256=digest(context.dependencies),
        config_sha256=context.config_digest,
        code_sha256=context.code_bindings,
        elapsed_seconds=time.perf_counter() - started,
        clearance_witness_count=len(problem.clearance),
        unresolved_witnesses=problem.unresolved,
        fitting_source_frames=sorted(problem.fit_times),
        audit_source_frames=context.times(),
        held_out_audit_source_frames=sorted(set(context.times()) - problem.fit_times),
        scope="Declared native hinges and strict window interior only; fixed original material contacts; finite-sample geometric audit, no motion acceptance.",
        limitations=[
            "Witness halfspaces are local repair proposals, not a global signed-distance or collision certificate.",
            "Only configured source observations, geometry pairs and temporal gates are checked; inspect the saved failure report and visual motion before selection.",
        ],
    )
    if backend_name == "material":
        report["limitations"][0] = (
            "Sampled signed mesh distances guide the proposal; open surfaces and finite "
            "samples can miss collisions. Independent original-triangle audits decide scope."
        )
    if context.iteration is not None:
        candidate["local_repair_seed_sha256"] = np.asarray(context.iteration.sha256)
        report["iteration"] = dict(
            seed_binding=context.bindings["iteration_seed_npz"].copy(),
            original_parent_binding=context.bindings["trajectory_npz"].copy(),
            audit_seed=seed_audit,
            original_posture_prior_preserved=True,
            original_corridor_preserved=True,
            derivative_weight=context.iteration.derivative_weight,
            proposal_objective="Seed-relative hinge acceleration/jerk differences normalized per hinge by max(seed RMS, 1 native unit); native wrist metrics are independent audit gates",
            adoption="No automatic acceptance or fallback selection; rejected candidates remain diagnostic and the seed is unchanged",
        )
    return PoseResult(candidate, report)


def run(config_path, out_dir, *, progress=None, cache_dir=None):
    config_path = Path(config_path).resolve(strict=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    raw = config_path.read_bytes()
    config = json.loads(raw)
    (out / "config.json").write_bytes(raw)
    try:
        result = solve_window(config, config_path.parent, progress=progress, cache_dir=cache_dir)
        candidate_path = out / "candidate_UNREVIEWED.npz"
        np.savez_compressed(candidate_path, **result.trajectory)
        reloaded = native_io.load_npz(candidate_path)
        if set(reloaded) != set(result.trajectory) or any(
            not np.array_equal(value, reloaded[key])
            for key, value in result.trajectory.items()
        ):
            raise RuntimeError(
                "Serialized candidate differs from the audited native arrays"
            )
        result.report.update(
            serialized_exact=True,
            candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            config_file_sha256=hashlib.sha256(raw).hexdigest(),
        )
        (out / "report.json").write_text(
            json.dumps(result.report, indent=2, allow_nan=False) + "\n"
        )
        return result
    except Exception as error:
        (out / "error.json").write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "type": type(error).__name__,
                    "message": str(error),
                },
                indent=2,
            )
            + "\n"
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("coverage", "audit", "repair"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if name != "coverage":
            p.add_argument("--cache", help="Persistent exact audit sample cache")
        if name == "audit":
            p.add_argument("--candidate", help="Audit this native NPZ against the fixed parent")
            p.add_argument("--diagnostic", action="store_true",
                           help="Return 0 after a completed diagnostic, retaining failed gates")
    args = parser.parse_args(argv)
    if args.action == "repair":
        result = run(
            args.config,
            args.output,
            progress=lambda row: print(json.dumps(row), flush=True),
            cache_dir=args.cache,
        )
        print(
            json.dumps(
                {
                    "status": result.report["status"],
                    "elapsed_seconds": result.report["elapsed_seconds"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0 if result.report["audit_after"]["passed"] else 1
    config_path = Path(args.config).resolve(strict=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    raw = config_path.read_bytes()
    (out / "config.json").write_bytes(raw)
    if args.action == "coverage":
        from .repair_scope import preflight_config

        report = preflight_config(json.loads(raw), config_path.parent)
        report["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    else:
        context = AuditContext(json.loads(raw), config_path.parent, cache_dir=args.cache)
        candidate_hash = hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest() if args.candidate else None
        candidate = native_io.load_npz(args.candidate) if args.candidate else None
        report = context.audit(candidate)
        if args.candidate:
            if hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest() != candidate_hash:
                raise ValueError("Candidate changed during audit")
            report["candidate_sha256"] = candidate_hash
        context.check_inputs_unchanged()
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report.get("summary", report.get("counts")),
                "output": str(out),
            },
            indent=2,
        )
    )
    if args.action == "coverage":
        # A successful inventory says nothing about geometric clearance.
        return 0
    return 0 if report["passed"] or args.diagnostic else 1


if __name__ == "__main__":
    raise SystemExit(main())
