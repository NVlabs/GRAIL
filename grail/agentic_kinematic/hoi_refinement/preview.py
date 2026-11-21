"""Dependency-free guards shared by Blender's reversible repair preview."""

import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            if digest(value["path"]) != value["sha256"]:
                raise ValueError(f"Repair binding changed: {value['path']}")
        for item in value.values():
            verify_records(item)
    elif isinstance(value, list):
        for item in value:
            verify_records(item)


def validate_preview(round_directory, *, require_apply=False):
    """A tool gate permits Apply; it never selects or accepts the full motion."""
    directory = Path(round_directory).resolve(strict=True)
    receipt_path = directory / "iteration.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "EXPORTED" or "repair" not in receipt:
        raise ValueError("Preview requires an exported local repair round")
    verify_records(receipt)
    review = json.loads(Path(receipt["review"]["path"]).read_text())
    checks = review["checks"]
    reasons = []
    guard = checks.get("iteration", {})
    if guard.get("passed") is not True:
        reasons.append("Smoothness/iteration checks failed or were not performed")
    collision = checks.get("original_mesh_intersections", {})
    if collision.get("status") != "PERFORMED" or not collision.get("frames"):
        reasons.append("Original-mesh intersection checks were not performed")
    else:
        for frame in collision["frames"]:
            count = sum(
                frame["after"][direction]["intersection_locations"]
                for direction in ("human_edges_to_object", "object_edges_to_human")
            )
            if count:
                reasons.append(f"Detected original-mesh crossings at source frame {frame['frame']}")
    if require_apply and reasons:
        raise ValueError("Repair cannot be applied: " + "; ".join(reasons))
    checkpoint = receipt["blender"]["artifacts"]["checkpoint"]
    return {
        "round_directory": str(directory),
        "receipt_sha256": digest(receipt_path),
        "checkpoint": checkpoint,
        "plan": receipt["plan"],
        "bindings": receipt.get("bindings", {}),
        "can_apply": not reasons,
        "reasons": reasons,
        "motion_status": "UNREVIEWED",
        "selected": False,
        "limitations": "Finite native replay checks only. Apply does not establish source "
        "agreement, self-collision clearance, correct contact or physical validity.",
    }
