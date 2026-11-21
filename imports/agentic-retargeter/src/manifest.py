"""Immutable input bindings and explicit per-case candidate selection.

JSON paths are resolved relative to the JSON that contains them. Each command writes a
new manifest; original inputs and earlier selections are never overwritten.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

CHECKS = (
    "source_alignment",
    "contact",
    "penetration",
    "arm_motion",
    "finger_motion",
    "native_binding",
)
INPUTS = ("source_video", "camera", "object_mesh", "human_model", "human_object", "robot")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Exclusive creation avoids silently overwriting a prior checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path, base=None):
    path = Path(path)
    if not path.is_absolute():
        path = Path(base or ".") / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Bind individual model/asset files, not a directory: {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _bindings(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from _bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _bindings(child)


def verify(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if not isinstance(manifest["expected_cases"], int) or manifest["expected_cases"] < 1:
        raise ValueError("Declare a positive fixed benchmark denominator")
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != manifest["expected_cases"] or len(set(ids)) != len(ids):
        raise ValueError("Benchmark denominator or unique case IDs changed")
    seen = set()
    for binding in _bindings(manifest):
        key = (binding["path"], binding["sha256"], binding["bytes"])
        if key in seen:
            continue
        path = Path(binding["path"])
        if path.stat().st_size != binding["bytes"] or sha256(path) != binding["sha256"]:
            raise ValueError(f"Changed binding: {path}")
        seen.add(key)
    for case in cases:
        human = selected(case, "human_object")
        robot = selected(case, "robot")
        if human is None:
            raise ValueError(f"A human/object fallback is required: {case['id']}")
        for stage, candidate in (("human_object", human), ("robot", robot)):
            if candidate and case["selected"][stage] != "original":
                if candidate.get("status") not in ("accepted", "partial") or not candidate.get(
                    "review"
                ):
                    raise ValueError(f"Selected candidate lacks an accepted review: {case['id']}")
        if robot and robot["parent_human_sha256"] != human["native"]["sha256"]:
            raise ValueError(f"Stale human/G1 pairing: {case['id']}")
    return {"cases": len(cases), "verified_bindings": len(seen)}


def create(spec_path):
    spec = read_json(spec_path)
    base = Path(spec_path).resolve().parent
    result = {
        "schema_version": 1,
        "benchmark": spec["benchmark"],
        "expected_cases": spec["expected_cases"],
        "cases": [],
    }
    cache = {}

    def bind_input(value):
        path = (base / value).resolve()
        if path not in cache:
            cache[path] = bind(path)
        return cache[path]

    for item in spec["cases"]:
        if not item["id"] or not isinstance(item["id"], str):
            raise ValueError("Each case needs a nonempty string ID")
        fps, count = float(item["fps"]), item["frame_count"]
        if not math.isfinite(fps) or fps <= 0 or not isinstance(count, int) or count < 4:
            raise ValueError("Positive source fps and at least four frames required")
        inputs = {key: bind_input(item[key]) for key in INPUTS}
        case = {
            "id": item["id"],
            "fps": fps,
            "frame_count": count,
            "inputs": inputs,
            "dependencies": [bind_input(p) for p in item.get("dependencies", [])],
            "candidates": [],
            "selected": {"human_object": "original", "robot": "original"},
        }
        case["original"] = {
            "human_object": {
                "id": "original",
                "native": inputs["human_object"],
                "status": "original_fallback",
                "limitations": item.get(
                    "limitations", ["Original reference; quality not certified."]
                ),
            },
            "robot": {
                "id": "original",
                "native": inputs["robot"],
                "parent_human_sha256": inputs["human_object"]["sha256"],
                "status": "original_fallback",
                "limitations": item.get(
                    "limitations", ["Original reference; quality not certified."]
                ),
            },
        }
        for stage, video in item.get("comparisons", {}).items():
            if stage not in case["original"]:
                raise ValueError(f"Unknown comparison stage {stage}")
            case["original"][stage]["comparison"] = bind_input(video)
        result["cases"].append(case)
    verify(result)
    return result


def get_case(manifest, case_id):
    return next(case for case in manifest["cases"] if case["id"] == case_id)


def selected(case, stage):
    identifier = case["selected"][stage]
    if identifier is None:
        return None
    if identifier == "original":
        return case["original"][stage]
    return next(c for c in case["candidates"] if c["id"] == identifier and c["stage"] == stage)


def register(manifest, case_id, record_path):
    """Record a trial and its dependencies; this does not select it."""
    verify(manifest)
    result = copy.deepcopy(manifest)
    case = get_case(result, case_id)
    record = read_json(record_path)
    base = Path(record_path).resolve().parent
    stage = record["stage"]
    if stage not in ("human_object", "robot"):
        raise ValueError("stage must be human_object or robot")
    if (
        not record["id"]
        or record["id"] == "original"
        or any(c["id"] == record["id"] for c in case["candidates"])
    ):
        raise ValueError("Use a new nonempty candidate ID")
    candidate = {
        "id": record["id"],
        "stage": stage,
        "status": "unreviewed",
        "native": bind(record["native"], base),
        "report": bind(record["report"], base),
        "config": bind(record["config"], base),
        "dependencies": [bind(p, base) for p in record.get("dependencies", [])],
    }
    if "comparison" in record:
        candidate["comparison"] = bind(record["comparison"], base)
    parent = record["parent_human_sha256"]
    known = [case["original"]["human_object"]["native"]["sha256"]]
    known += [c["native"]["sha256"] for c in case["candidates"] if c["stage"] == "human_object"]
    if parent not in known:
        raise ValueError("Unknown parent human/object pair")
    if stage == "robot" and Path(candidate["native"]["path"]).suffix == ".npz":
        import numpy as np

        with np.load(candidate["native"]["path"], allow_pickle=False) as native:
            if "source_sha256" in native and str(native["source_sha256"].item()) != parent:
                raise ValueError("Robot native export disagrees with declared parent human/object")
    candidate["parent_human_sha256"] = parent
    case["candidates"].append(candidate)
    return result


def select(manifest, case_id, candidate_id, review_path):
    """Selection is an explicit evidence-based decision, separate from solver exit."""
    verify(manifest)
    result = copy.deepcopy(manifest)
    case = get_case(result, case_id)
    candidate = next(c for c in case["candidates"] if c["id"] == candidate_id)
    if candidate["status"] != "unreviewed":
        raise ValueError("This candidate already has a decision; register a new version")
    review = read_json(review_path)
    if review["case_id"] != case_id or review["candidate_sha256"] != candidate["native"]["sha256"]:
        raise ValueError("Review belongs to a different case or native result")
    decision = review["decision"]
    if decision not in ("accepted", "partial", "rejected"):
        raise ValueError("decision must be accepted, partial or rejected")
    if not review.get("scope") or not isinstance(review.get("limitations"), list):
        raise ValueError("Declare scope and limitations, including remaining failures")
    evidence = []
    for name in CHECKS:
        check = review["checks"][name]
        if check["status"] not in ("pass", "partial", "fail", "unknown") or not check.get("note"):
            raise ValueError(f"Invalid or unexplained check: {name}")
        if decision == "accepted" and check["status"] != "pass":
            raise ValueError("Full selection requires all checks; use partial or rejected")
        if check["status"] != "unknown" and not check.get("evidence"):
            raise ValueError(f"Missing check evidence: {name}")
        evidence.extend(check.get("evidence", []))
    visual = review["visual_review"]
    frames = visual["observed_frames"]
    if (
        not visual.get("reviewer")
        or not visual.get("findings")
        or not frames
        or any(not isinstance(f, int) or f < 0 or f >= case["frame_count"] for f in frames)
    ):
        raise ValueError("Record actual reviewer, frame coverage and findings")
    if not {"source", "novel_left", "novel_right"} <= set(visual["views"]):
        raise ValueError("Review source camera and two diagnostic views")
    if not visual.get("artifacts"):
        raise ValueError("Visual review requires bound artifacts")
    evidence.extend(visual["artifacts"])
    if decision == "partial" and not review["limitations"]:
        raise ValueError("Partial selections must disclose remaining problems")
    if decision != "rejected":
        if review["checks"]["native_binding"]["status"] != "pass":
            raise ValueError("Native binding must pass before selection")
        if review["checks"]["source_alignment"]["status"] not in ("pass", "partial"):
            raise ValueError("Source alignment cannot be failed/unknown for selection")
        if "comparison" not in candidate:
            raise ValueError("Selection requires a playable full-sequence comparison")
        from .media import check_video

        check_video(candidate["comparison"]["path"], case["fps"], case["frame_count"])
    candidate.update(
        status=decision,
        scope=review["scope"],
        limitations=review["limitations"],
        review=bind(review_path),
        review_evidence=[bind(p, Path(review_path).resolve().parent) for p in evidence],
    )
    if decision == "rejected":
        return result
    if candidate["stage"] == "robot":
        if candidate["parent_human_sha256"] != selected(case, "human_object")["native"]["sha256"]:
            raise ValueError("G1 candidate must be rebased on the selected exact human/object pair")
    case["selected"][candidate["stage"]] = candidate_id
    if candidate["stage"] == "human_object":
        robot = selected(case, "robot")
        if robot and robot["parent_human_sha256"] != candidate["native"]["sha256"]:
            case["selected"]["robot"] = None  # original pair remains explicitly accessible
    verify(result)
    return result
