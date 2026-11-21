"""Compare a finite grasp bank at common mounted keys and short transitions.

This screens native proposals; it neither selects a motion nor treats sparse
geometry coverage as a full hold pass. Contact revisions remain distinct.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit import native_io
from .contact_edit.pose import _keys, _sha
from .grasp_io import digest
from .motion_audit import AuditContext


def review_frames(native, hold_windows, transition_windows, difficult_frames, substeps):
    """Include acquisition, largest relative rotation, last hold and dense joins."""
    total = len(native_io._native_qpos(native))
    if type(substeps) is not int or not 1 <= substeps <= 8:
        raise ValueError("substeps must be an integer in [1, 8]")
    if not isinstance(hold_windows, list) or not hold_windows:
        raise ValueError("Declare hold_windows")
    if not isinstance(transition_windows, list) or not transition_windows:
        raise ValueError("Declare transition_windows for acquisition/release review")
    frames, keys = set(), []
    for kind, windows in (("hold", hold_windows), ("transition", transition_windows)):
        for window in windows:
            if (not isinstance(window, list) or len(window) != 2 or
                    any(type(v) is not int for v in window) or
                    not 0 <= window[0] < window[1] < total):
                raise ValueError("Windows require increasing inclusive native frame endpoints")
            start, end = window
            if kind == "hold":
                rotations = np.asarray(native["obj_R"])[start:end+1]
                angles = Rotation.from_matrix(rotations @ rotations[0].T).magnitude()
                hardest = start + int(np.argmax(angles))
                keys.append(dict(window=window, first=start, largest_rotation=hardest, last=end))
                frames.update((float(start), float(hardest), float(end)))
            else:
                frames.update(float(f) / substeps
                              for f in range(start * substeps, end * substeps + 1))
    if (not isinstance(difficult_frames, list) or
            any(type(f) is not int or not 0 <= f < total for f in difficult_frames)):
        raise ValueError("difficult_frames must contain in-range integer frame indices")
    frames.update(float(f) for f in difficult_frames)
    return sorted(frames), keys


def _comparison_contract(context):
    config = copy.deepcopy(context.config)
    # Different material surface hypotheses may be tested under the same gates.
    # Removing these coordinates never makes their gaps a matched improvement.
    surface_fields = {"body_name", "point_local", "normal_local", "object_face_id",
                      "object_barycentric", "object_normal_sign"}
    for contact in config["contacts"]:
        for field in surface_fields:
            contact.pop(field, None)
    for field in ("model_xml", "trajectory_npz", "object_mesh_npz"):
        config.pop(field, None)
    return digest(dict(config=config,
                       inputs={k: v["sha256"] for k, v in context.bindings.items()},
                       code=context.code_bindings))


def evaluate(config_path, output, *, cache_dir=None, progress=None):
    path = Path(config_path).resolve(strict=True)
    manifest_hash = _sha(path)
    cfg = json.loads(path.read_text())
    _keys(cfg, {"schema_version", "candidates", "hold_windows", "transition_windows"},
          {"difficult_frames", "substeps", "required_gates"}, "grasp evaluation")
    if cfg["schema_version"] != 1:
        raise ValueError("Expected schema_version 1")
    entries = cfg["candidates"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Declare a finite nonempty candidate list")
    required = cfg.get("required_gates", ["contacts", "collision", "temporal", "invariants"])
    if (not isinstance(required, list) or "invariants" not in required or len(set(required)) != len(required)
            or not set(required) <= {"contacts", "collision", "temporal", "invariants", "source", "smoothness"}):
        raise ValueError("required_gates must name unique audit gates including invariants")
    contexts, seen, revisions, contract = [], set(), {}, None
    # Validate all bindings before spending time on mesh queries or creating output.
    for entry in entries:
        _keys(entry, {"id", "native_npz", "repair_config", "contact_revision"}, set(), "candidate")
        name = entry["id"]
        if (not isinstance(name, str) or not name or name in seen or
                any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)):
            raise ValueError("Candidate ids must be unique simple names")
        if not isinstance(entry["contact_revision"], str) or not entry["contact_revision"].strip():
            raise ValueError("Declare each contact_revision")
        seen.add(name)
        repair_path = (path.parent / entry["repair_config"]).resolve(strict=True)
        context = AuditContext(json.loads(repair_path.read_text()), repair_path.parent,
                               cache_dir=cache_dir)
        definition = digest(context.config["contacts"])
        revision = entry["contact_revision"]
        if revision in revisions and revisions[revision] != definition:
            raise ValueError("One contact_revision cannot identify different contact definitions")
        revisions[revision] = definition
        current = _comparison_contract(context)
        if contract is not None and current != contract:
            raise ValueError("Candidates must share parent, geometry, scope and gates")
        contract = current
        native_path = (path.parent / entry["native_npz"]).resolve(strict=True)
        candidate_hash = _sha(native_path)
        native = native_io.load_npz(native_path)
        if _sha(native_path) != candidate_hash:
            raise ValueError("Candidate changed while loading")
        contexts.append((entry, context, native_path, native, candidate_hash))
    frames, keys = review_frames(contexts[0][1].parent, cfg["hold_windows"],
                                cfg["transition_windows"], cfg.get("difficult_frames", []),
                                cfg.get("substeps", 2))
    for _, context, _, _, _ in contexts:
        lo, hi = min(context.times()), max(context.times())
        if frames[0] < lo or frames[-1] > hi:
            raise ValueError("Review frames exceed the declared audit scope")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(path.read_text())
    rows = []
    for entry, context, native_path, native, candidate_hash in contexts:
        report = context.audit(native, times=frames)
        context.check_inputs_unchanged()
        if _sha(native_path) != candidate_hash:
            raise ValueError("Candidate changed during evaluation")
        report.update(candidate_sha256=candidate_hash, screening_only=True,
                      contact_revision=entry["contact_revision"],
                      full_scope_covered=frames == context.times())
        report_path = out / (entry["id"] + ".audit.json")
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        gates = {g: report["checks"].get(g, {}).get("passed") for g in required}
        failed = [g for g in required if gates[g] is False]
        unknown = [g for g in required if gates[g] is not True and g not in failed]
        # Full-trajectory derivative gates still run even for sparse geometry keys.
        row = dict(id=entry["id"], candidate_sha256=candidate_hash,
                   contact_revision=entry["contact_revision"], audit=str(report_path.resolve()),
                   screen_passed=not failed and not unknown,
                   failed_gates=failed, unknown_gates=unknown,
                   max_contact_gap_m=report["summary"]["max_contact_gap_m"],
                   full_scope_covered=report["full_scope_covered"], selected=False)
        rows.append(row)
        if progress:
            progress(row)
        # Persist incremental evidence so interruption does not lose completed checks.
        (out / "completed.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    order = sorted(range(len(rows)), key=lambda i: (
        not rows[i]["screen_passed"], len(rows[i]["failed_gates"]), len(rows[i]["unknown_gates"]), i))
    if _sha(path) != manifest_hash:
        raise ValueError("Manifest changed during evaluation")
    result = dict(schema_version=1, status="SCREENED_UNSELECTED", motion_accepted=False,
                  comparison_contract_sha256=contract, manifest_sha256=manifest_hash,
                  source_frames=frames, keys=keys, required_gates=required,
                  candidates=rows, review_order=[rows[i]["id"] for i in order],
                  limitations=["A screen pass requires full propagation, full-scope geometry and visual review before acceptance.",
                               "Largest object rotation is one diagnostic key, not proof of the hardest arm configuration.",
                               "Different contact revisions are feasibility hypotheses, not matched slip/gap improvements."])
    (out / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    args = parser.parse_args()
    result = evaluate(args.config, args.output, cache_dir=args.cache,
                      progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"status": result["status"], "review_order": result["review_order"]}))


if __name__ == "__main__":
    main()
