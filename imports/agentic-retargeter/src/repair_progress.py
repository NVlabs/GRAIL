"""Preserve independently audited repair progress without selecting by loss alone.

One ledger has one fixed audit config and contact revision. New hypotheses use
another ledger; their scores are not matched improvements. Visual receipts are
recorded reviews, not machine verification that someone watched the media.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile


GATES = {
    "contacts": {"contact", "contact_coverage"},
    "collision": {"collision", "geometry_incomplete"},
    "source": {"source", "source_coverage"},
    "temporal": {"temporal"},
    "invariants": {"frozen_qpos", "limits"},
    "smoothness": {"smoothness"},
}
DEFAULT_GATES = ("contacts", "collision", "source", "temporal", "invariants")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric evidence")
    return float(value)


def _evidence(candidate, config_path, audit_path):
    candidate = Path(candidate).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    report = _read(audit_path)
    config = _read(config_path)
    if report.get("schema_version") != 1 or "audit_after" in report:
        raise ValueError("Supply an independent AuditContext report, not a solver summary")
    if report.get("candidate_sha256") != file_hash(candidate):
        raise ValueError("Audit candidate_sha256 does not match the candidate file")
    if report.get("config_sha256") != digest(config):
        raise ValueError("Audit config binding does not match")
    bindings = report.get("bindings", {})
    for key in ("trajectory_npz", "model_xml", "object_mesh_npz"):
        bound = bindings.get(key, {})
        actual = (config_path.parent / config[key]).resolve(strict=True)
        if actual != Path(bound.get("path", "")) or bound.get("sha256") != file_hash(actual):
            raise ValueError(f"Audit input binding changed: {key}")
    code = report.get("code_sha256")
    if not isinstance(code, dict) or not code:
        raise ValueError("Audit implementation bindings are required")
    audit_implementation = Path(__file__).with_name("motion_audit.py").resolve()
    if code.get(str(audit_implementation)) != file_hash(audit_implementation):
        raise ValueError("Report does not bind the current independent AuditContext implementation")
    for path, sha in code.items():
        if file_hash(path) != sha:
            raise ValueError("Audit implementation changed: " + path)
    from .contact_edit.pose import _model_dependencies
    dependencies = {str(p): file_hash(p) for p in
                    _model_dependencies(Path(bindings["model_xml"]["path"]))}
    if report.get("model_dependencies_sha256") != digest(dependencies):
        raise ValueError("Audit model assets changed")
    coverage = report.get("coverage", {})
    rows, failures = report.get("rows"), report.get("failures")
    if not isinstance(rows, list) or not rows or not isinstance(failures, list):
        raise ValueError("Audit requires sampled rows and explicit failures")
    if [r.get("source_frame") for r in rows] != coverage.get("source_frames"):
        raise ValueError("Audit sample coverage does not match its rows")
    if coverage.get("window") != config.get("window"):
        raise ValueError("Audit window differs from the fixed config")
    affected = report.get("affected_failures")
    inherited = report.get("failures_outside_editable_support")
    if not isinstance(affected, list) or not isinstance(inherited, list):
        raise ValueError("Audit must distinguish affected and inherited failures")
    if sorted(map(digest, affected + inherited)) != sorted(map(digest, failures)):
        raise ValueError("Affected/inherited failures do not partition the audit")
    # Derive gates from detailed failures; a top-level `passed` assertion is not
    # enough. NOT_CONFIGURED/NOT_IN_SCOPE and absent checks remain UNKNOWN.
    full, scoped = {}, {}
    for name, kinds in GATES.items():
        check = report.get("checks", {}).get(name, {})
        known = check.get("status") in ("PASS", "FAIL")
        failed = any(f.get("kind") in kinds for f in failures)
        if known and (check.get("status") != ("FAIL" if failed else "PASS")
                      or check.get("passed") is not (not failed)):
            raise ValueError("Audit check contradicts its failures: " + name)
        full[name] = ("FAIL" if failed else "PASS") if known else "UNKNOWN"
        scoped[name] = ("FAIL" if any(f.get("kind") in kinds for f in affected)
                        else "PASS") if known else "UNKNOWN"
    if any(f.get("kind") not in set().union(*GATES.values()) for f in failures):
        raise ValueError("Audit contains an unsupported failure category")
    fixed = dict(config_sha256=report["config_sha256"],
                 input_sha256={k: bindings[k]["sha256"] for k in
                               ("trajectory_npz", "model_xml", "object_mesh_npz")},
                 model_dependencies_sha256=report.get("model_dependencies_sha256"),
                 code_sha256=code, coverage=coverage)
    return config, report, fixed, full, scoped


def _visual(path, *, native_sha, scope_sha, revision):
    if path is None:
        return None
    path = Path(path).resolve(strict=True)
    review = _read(path)
    if (review.get("candidate_sha256") != native_sha or review.get("scope_sha256") != scope_sha
            or review.get("contact_revision") != revision):
        raise ValueError("Visual review is bound to another candidate/scope/contact revision")
    if review.get("decision") not in ("PASS", "FAIL", "UNKNOWN"):
        raise ValueError("Visual review needs an explicit PASS/FAIL/UNKNOWN decision")
    if not isinstance(review.get("reviewed_by"), str) or not review["reviewed_by"].strip():
        raise ValueError("Visual review must identify its reviewer")
    evidence = review.get("evidence", [])
    if not evidence:
        raise ValueError("Visual review requires bound media evidence")
    for item in evidence:
        media = (path.parent / item["path"]).resolve(strict=True)
        if media.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov"):
            raise ValueError("Visual evidence must be an image or video artifact")
        if file_hash(media) != item.get("sha256"):
            raise ValueError("Visual evidence changed")
    return review


def _complete_scope(config, report, candidate):
    """Match AuditContext.times(), including configured subframes and halo."""
    import numpy as np
    with np.load(candidate, allow_pickle=False) as native:
        frames = len(native["qpos"])
    audit = config.get("audit", {})
    substeps = audit.get("substeps", 4)
    halo = max(audit.get("halo", 2), 3 if "iteration" in config else 0)
    start, end = max(0, config["window"][0] - halo), min(frames - 1, config["window"][1] + halo)
    times = set((np.arange(start * substeps, end * substeps + 1) / substeps).tolist())
    times.update(g["frame"] for g in (config.get("source") or {}).get("observations", [])
                 if start <= g["frame"] <= end)
    return (report["coverage"].get("halo") == halo
            and report["coverage"]["source_frames"] == sorted(times)
            and not report.get("screening_only", False))


@contextmanager
def _locked(directory):
    with (Path(directory) / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def read_progress(directory):
    directory = Path(directory)
    ledger = _read(directory / "progress.json")
    for record in ledger["candidates"].values():
        for name, item in record["artifacts"].items():
            if file_hash(directory / item["path"]) != item["sha256"]:
                raise ValueError("Saved progress artifact changed: " + name)
    return ledger


def initialize(directory, baseline, config, audit_report, *, contact_revision,
               objective="max_contact_gap_m", required_gates=DEFAULT_GATES,
               scope_kind="local"):
    """Create a fixed comparison scope and preserve its independently audited baseline."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    if not isinstance(contact_revision, str) or not contact_revision.strip():
        raise ValueError("An explicit contact_revision is required")
    gates = list(dict.fromkeys(required_gates))
    if not gates or set(gates) - set(GATES) or "invariants" not in gates:
        raise ValueError("Required gates must name known checks and include invariants")
    if scope_kind not in ("local", "full"):
        raise ValueError("scope_kind must be local or full")
    cfg, report, fixed, _, _ = _evidence(baseline, config, audit_report)
    value = report.get("summary", {}).get(objective)
    _number(value, "Baseline objective")
    policy = dict(contact_revision=contact_revision, objective=objective,
                  required_gates=gates, scope_kind=scope_kind, fixed_evidence=fixed,
                  contact_definition_sha256=digest({k: cfg.get(k) for k in
                      ("contacts", "contact_gap_policy", "contact_normal_policy")}))
    policy["scope_sha256"] = digest(policy)
    directory.mkdir(parents=True)
    _write(directory / "progress.json", dict(schema_version=1, policy=policy,
           candidates={}, baseline=None, best_scoped_reviewed_partial=None,
           best_diagnostic=None, full_accepted=None,
           limitations=["Full acceptance is for this declared gate policy and sampled timeline; it does not certify unconfigured geometry, continuous clearance or physical stability.",
                        "Visual receipts record a review and bound media; they do not prove the reviewer watched or correctly interpreted the media."]))
    return record_candidate(directory, baseline, config, audit_report, candidate_id="baseline",
                            contact_revision=contact_revision, _baseline=True)


def record_candidate(directory, candidate, config, audit_report, *, candidate_id,
                     contact_revision, visual_review=None, checkpoint=None,
                     accept_full=False, _baseline=False):
    """Append evidence; failed branches never erase previously reviewed progress.

    `accept_full` is an explicit review action, not a solver flag. It still needs
    full declared timeline coverage, every required independent gate and a bound
    PASS visual receipt. No claim of continuous clearance or dynamics is made.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", candidate_id):
        raise ValueError("Invalid candidate_id")
    directory = Path(directory)
    with _locked(directory):
        ledger = read_progress(directory)
        policy = ledger["policy"]
        if _baseline and (ledger["candidates"] or candidate_id != "baseline"):
            raise ValueError("Baseline is immutable after initialization")
        if candidate_id in ledger["candidates"]:
            raise ValueError("Candidate ID is already saved; append a new revision")
        cfg, report, fixed, full, scoped = _evidence(candidate, config, audit_report)
        if report.get("contact_revision", contact_revision) != contact_revision:
            raise ValueError("Audit contact revision differs from the declared revision")
        if contact_revision != policy["contact_revision"] or fixed != policy["fixed_evidence"]:
            raise ValueError("Different scope/contact revision: use a separate comparison ledger")
        value = _number(report.get("summary", {}).get(policy["objective"]), "Candidate objective")
        native_sha = file_hash(candidate)
        visual = _visual(visual_review, native_sha=native_sha,
                         scope_sha=policy["scope_sha256"], revision=contact_revision)
        visual_pass = visual is not None and visual["decision"] == "PASS"
        complete_scope = _complete_scope(cfg, report, candidate)
        scope_pass = complete_scope and all(scoped[g] == "PASS" for g in policy["required_gates"])
        full_pass = all(full[g] == "PASS" for g in policy["required_gates"])
        if accept_full:
            import numpy as np
            with np.load(candidate, allow_pickle=False) as native:
                frame_count = len(native["qpos"])
                if str(native.get("status", "")) == "REJECTED_BY_POSTURE_REFERENCE":
                    raise ValueError("Full acceptance cannot override a failed posture reference gate")
            covered = set(report["coverage"]["source_frames"])
            if (policy["scope_kind"] != "full" or cfg["window"] != [0, frame_count - 1]
                    or not set(range(frame_count)).issubset(covered)
                    or not complete_scope or report["failures"] or not full_pass or not visual_pass):
                raise ValueError("Full acceptance needs full scope, required gates and visual PASS")
        classification = ("BASELINE" if _baseline else "FULL_ACCEPTED" if accept_full else
                          "SCOPED_REVIEWED_PARTIAL" if scope_pass and visual_pass else "DIAGNOSTIC")
        target = directory / "candidates" / candidate_id
        target.mkdir(parents=True, exist_ok=False)
        artifacts = {}
        try:
            sources = dict(native=Path(candidate), audit=Path(audit_report), config=Path(config))
            if checkpoint is not None:
                sources["checkpoint"] = Path(checkpoint)
            if visual_review is not None:
                sources["visual_review"] = Path(visual_review)
                for i, item in enumerate(visual["evidence"]):
                    sources[f"visual_media_{i}"] = (Path(visual_review).resolve().parent / item["path"]).resolve()
            for name, source in sources.items():
                dest = target / (name + source.suffix)
                shutil.copy2(source, dest)
                artifacts[name] = dict(path=str(dest.relative_to(directory)), sha256=file_hash(dest))
            if _read(target / ("config" + Path(config).suffix)) != cfg or _read(target / ("audit" + Path(audit_report).suffix)) != report:
                raise ValueError("Audit/config changed while preserving progress")
            if visual is not None:
                if _read(target / ("visual_review" + Path(visual_review).suffix)) != visual:
                    raise ValueError("Visual review changed while preserving progress")
                if any(artifacts[f"visual_media_{i}"]["sha256"] != item["sha256"]
                       for i, item in enumerate(visual["evidence"])):
                    raise ValueError("Visual evidence changed while preserving progress")
            if artifacts["native"]["sha256"] != native_sha:
                raise ValueError("Candidate changed while preserving progress")
            baseline_value = (value if _baseline else ledger["candidates"][ledger["baseline"]]["objective_value"])
            record = dict(candidate_id=candidate_id, candidate_sha256=native_sha,
                          classification=classification, contact_revision=contact_revision,
                          scope_sha256=policy["scope_sha256"], gates=full, scoped_gates=scoped,
                          complete_independent_audit_scope=complete_scope,
                          visual_review="UNKNOWN" if visual is None else visual["decision"],
                          objective_value=value, improved_vs_baseline=value < baseline_value,
                          full_motion_accepted=bool(accept_full), artifacts=artifacts)
            ledger["candidates"][candidate_id] = record
            if _baseline:
                ledger["baseline"] = candidate_id
            elif accept_full:
                ledger["full_accepted"] = candidate_id
            else:
                slot = ("best_scoped_reviewed_partial" if classification == "SCOPED_REVIEWED_PARTIAL"
                        else "best_diagnostic")
                previous = ledger[slot]
                if record["improved_vs_baseline"] and (previous is None or value < ledger["candidates"][previous]["objective_value"]):
                    ledger[slot] = candidate_id
            _write(directory / "progress.json", ledger)
        except Exception:
            # Do not remove a committed checkpoint if an interruption/error
            # arrives just after the atomic ledger replacement.
            if candidate_id not in _read(directory / "progress.json")["candidates"]:
                shutil.rmtree(target)
            raise
        return ledger


def next_action(directory, *, remaining_seconds, next_step_seconds, reserve_seconds=240):
    """Return a budget/quality decision only; never run, retry or extend work."""
    ledger = read_progress(directory)
    remaining = _number(remaining_seconds, "remaining_seconds")
    needed = _number(next_step_seconds, "next_step_seconds")
    reserve = _number(reserve_seconds, "reserve_seconds")
    if min(remaining, needed, reserve) < 0:
        raise ValueError("Budget durations cannot be negative")
    if ledger["full_accepted"] is not None:
        action, reason = "FINISH", "FULL_DECLARED_SCOPE_ACCEPTED"
    elif remaining >= needed + reserve and needed > 0:
        action, reason = "CONTINUE", "NEXT_STEP_FITS_WITH_DELIVERY_RESERVE"
    else:
        action, reason = "FINALIZE", "NO_DECLARED_NEXT_STEP_FITS_WITH_DELIVERY_RESERVE"
    return dict(action=action, reason=reason, task_completed=ledger["full_accepted"] is not None,
                remaining_seconds=remaining, next_step_seconds=needed, reserve_seconds=reserve,
                baseline=ledger["baseline"], best_scoped_reviewed_partial=ledger["best_scoped_reviewed_partial"],
                best_diagnostic=ledger["best_diagnostic"], full_accepted=ledger["full_accepted"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("init", "record", "status", "decision"):
        p = commands.add_parser(name)
        p.add_argument("--directory", required=True)
        if name in ("init", "record"):
            p.add_argument("--candidate", required=True)
            p.add_argument("--config", required=True)
            p.add_argument("--audit-report", required=True)
            p.add_argument("--contact-revision", required=True)
        if name == "init":
            p.add_argument("--objective", default="max_contact_gap_m")
            p.add_argument("--required-gates", nargs="+", default=list(DEFAULT_GATES))
            p.add_argument("--scope-kind", choices=("local", "full"), default="local")
        if name == "record":
            p.add_argument("--candidate-id", required=True)
            p.add_argument("--visual-review")
            p.add_argument("--checkpoint")
            p.add_argument("--accept-full", action="store_true")
        if name == "decision":
            p.add_argument("--remaining-seconds", type=float, required=True)
            p.add_argument("--next-step-seconds", type=float, required=True)
            p.add_argument("--reserve-seconds", type=float, default=240)
    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    if action == "init":
        args["baseline"] = args.pop("candidate")
        result = initialize(**args)
    elif action == "record":
        result = record_candidate(**args)
    elif action == "decision":
        result = next_action(**args)
    else:
        result = read_progress(**args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
