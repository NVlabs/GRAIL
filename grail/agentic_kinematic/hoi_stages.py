"""Explicit numerical Stage 1/2 from hash-bound cached source observations.

This endpoint does not claim to rerun predictors. Stage 2 takes an explicit native
placement parent while preserving the original Stage 1 motion prior and targets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from grail.core.native_pickle import load_native_pickle


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def bind(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return dict(path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size)


def placement_events(events, frame_count):
    """Keep human alignment, observed object motion, and contact independent."""
    touch, move = events["contact_frame"], events["object_first_motion_frame"]
    mode = events.get("mode", "contact_alignment")
    if mode not in ("contact_alignment", "reviewed_unchanged"):
        raise ValueError("Unknown placement mode")
    unknown = events.get("object_motion_onset_status") == "unknown"
    if "object_motion_onset_status" in events and not unknown:
        raise ValueError("Explicit object_motion_onset_status must be unknown")
    if unknown and move is not None:
        raise ValueError(
            "Unknown object motion onset requires explicit null, never a frame sentinel"
        )
    if not unknown and (type(move) is not int or not 0 <= move <= frame_count):
        raise ValueError("Require an in-range source-supported first object motion frame")
    if touch is None:
        if mode != "reviewed_unchanged" or "alignment_frame" not in events:
            raise ValueError(
                "Absent contact requires reviewed_unchanged placement and alignment_frame"
            )
    elif type(touch) is not int or not 0 <= touch <= frame_count:
        raise ValueError("Require an in-range source-supported contact key or explicit null")
    if mode == "reviewed_unchanged":
        if touch is not None or any(
            events.get(key) is not None for key in ("recovery_start_frame", "recovery_end_frame")
        ):
            raise ValueError("Unchanged placement must not invent contact or transition frames")
    alignment = events.get("alignment_frame", 0 if touch == frame_count else touch)
    if type(alignment) is not int or not 0 <= alignment < frame_count:
        raise ValueError("alignment_frame must be an actual source frame")
    result = dict(
        mode=mode, contact_frame=touch, object_first_motion_frame=move, alignment_frame=alignment
    )
    if unknown:
        result["object_motion_onset_status"] = "unknown"
    return result


def object_prefix_length(events):
    """Constraint extent only; zero does not assert an observed motion onset."""
    return (
        0
        if events.get("object_motion_onset_status") == "unknown"
        else events["object_first_motion_frame"]
    )


def prepare_object_poses(poses, prefix_length):
    """Canonicalize representation and freeze only a reviewed stationary prefix."""
    from grail.optimization.stage2_policy import canonicalize_rigid_poses

    result, _ = canonicalize_rigid_poses(poses)
    if type(prefix_length) is not int or not 0 <= prefix_length <= len(result):
        raise ValueError("Invalid object stationary prefix length")
    result[:prefix_length] = result[0].copy()
    return result


def constrain_object_prefix(rotations, translations, reference, prefix_length):
    """Keep a declared prefix exact while allowing normal fitting elsewhere."""
    import torch

    moving = torch.arange(len(reference), device=reference.device) >= prefix_length
    return (
        torch.where(moving[:, None, None], rotations, reference[:, :3, :3]),
        torch.where(moving[:, None], translations, reference[:, :3, 3]),
    )


def validate_unknown_motion_review(inventory, inputs):
    """Bind unresolved source timing to actual extracted images, not a sentinel."""
    import cv2

    if "source_motion_review" not in inputs:
        raise ValueError("Unknown object motion onset requires a bound source_motion_review")
    record = inputs["source_motion_review"]
    if bind(record["path"]) != record:
        raise ValueError("Source motion review binding changed")
    review = json.loads(Path(record["path"]).read_text())
    n, fps = inventory["frame_count"], inventory["fps"]
    if (
        review.get("schema") != "grail.object_motion_onset_review.v1"
        or review.get("object_motion_onset_status") != "unknown"
        or "object_first_motion_frame" not in review
        or review["object_first_motion_frame"] is not None
        or review.get("interval") != [0, n - 1]
        or review.get("frame_count") != n
        or review.get("fps") != fps
        or review.get("source_video_sha256") != inputs["source_video"]["sha256"]
        or review.get("object_poses_sha256") != inputs["object_poses"]["sha256"]
        or not isinstance(review.get("reviewed_by"), str)
        or not review["reviewed_by"].strip()
        or not isinstance(review.get("observation"), str)
        or len(review["observation"].strip()) < 12
    ):
        raise ValueError("Unknown onset requires an explicit source/trajectory-bound review")
    frames = review.get("inspected_frames")
    if (
        not isinstance(frames, list)
        or not frames
        or any(type(f) is not int for f in frames)
        or frames != sorted(set(frames))
        or frames[0] != 0
        or frames[-1] != n - 1
        or any(b - a > 6 for a, b in zip(frames, frames[1:]))
    ):
        raise ValueError(
            "Unknown-onset inspection must span the source with gaps at most six frames"
        )
    receipts = review.get("inspection_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("Unknown onset requires source inspection receipts")
    images = {}
    for item in receipts:
        if bind(item["path"]) != item:
            raise ValueError("Source inspection receipt binding changed")
        receipt_path = Path(item["path"])
        source = json.loads(receipt_path.read_text())
        video = source.get("source_video", {})
        if (
            source.get("schema") != "grail.source_inspection.v1"
            or source.get("frame_count") != n
            or source.get("fps") != fps
            or video.get("sha256") != inputs["source_video"]["sha256"]
            or video.get("bytes") != inputs["source_video"]["bytes"]
            or bind(video["path"]) != video
            or len(source.get("frames", [])) != len(source.get("outputs", {}).get("frames", []))
        ):
            raise ValueError("Source inspection receipt has different source/timing")
        sheet = source["outputs"]["sheet"]
        if bind(sheet["path"]) != sheet:
            raise ValueError("Source inspection sheet changed")
        for frame, image in zip(source["frames"], source["outputs"]["frames"]):
            image_path = Path(image["path"])
            if (
                type(frame) is not int
                or not 0 <= frame < n
                or image_path.parent.resolve() != receipt_path.parent.resolve()
                or image_path.suffix.lower() not in (".jpg", ".png")
                or bind(image_path) != image
                or (frame in images and images[frame]["sha256"] != image["sha256"])
            ):
                raise ValueError("Invalid or changed source inspection image")
            images[frame] = image
    if not set(frames) <= images.keys():
        raise ValueError("Claimed source inspection lacks extracted image evidence")
    cap = cv2.VideoCapture(inputs["source_video"]["path"])
    decoded = 0
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 1e-6:
            raise ValueError("Source video timing differs from the unknown-onset review")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded in frames:
                image_path = Path(images[decoded]["path"])
                ok, encoded = cv2.imencode(image_path.suffix, frame)
                if not ok or encoded.tobytes() != image_path.read_bytes():
                    raise ValueError("Inspected image is not the claimed original source frame")
            decoded += 1
    finally:
        cap.release()
    if decoded != n:
        raise ValueError("Source frame count differs from the unknown-onset review")
    return record


def load_case(path, *, require_contacts=True):
    path = Path(path).resolve(strict=True)
    case = json.loads(path.read_text())
    required = {"schema_version", "inventory", "backend", "human_model", "optimization_config"}
    if set(case) != required or case["schema_version"] != 1:
        raise ValueError(
            "Case requires schema_version, inventory, backend, human_model, optimization_config"
        )
    if case["backend"] != "soma03":
        raise ValueError("Explicit supported native backend required")
    if case["human_model"].get("body_model") != case["backend"]:
        raise ValueError("Human model and backend must agree")
    inventory_path = (path.parent / case["inventory"]).resolve(strict=True)
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema") != "grail.cached_fitting_inputs.v1":
        raise ValueError("Expected a cached-fitting source inventory")
    n, fps = inventory["frame_count"], inventory["fps"]
    if type(n) is not int or n < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source timing")
    placement_events(inventory["placement"], n)
    files = {}
    for name, record in inventory["files"].items():
        target = (inventory_path.parent / record["path"]).resolve(strict=True)
        if not target.is_relative_to(inventory_path.parent):
            raise ValueError("Input record escapes inventory directory")
        actual = bind(target)
        if actual["sha256"] != record["sha256"] or actual["bytes"] != record["bytes"]:
            raise ValueError(f"Changed input: {name}")
        files[name] = actual
    required_inputs = [
        "source_video",
        "object_mesh",
        "object_poses",
        "render_config",
        "mask",
        "depth",
        "depth_metadata",
        f"{case['backend']}/human_motion",
    ]
    if require_contacts:
        required_inputs += [f"{case['backend']}/contact_map", f"{case['backend']}/contact_temporal"]
    for key in required_inputs:
        if key not in files:
            raise ValueError(f"Missing bound input {key}")
    if inventory["placement"].get("object_motion_onset_status") == "unknown":
        validate_unknown_motion_review(inventory, files)
    elif "source_motion_review" in files:
        raise ValueError("Bound unknown-onset review cannot be relabeled as a known motion frame")
    case["optimization_config"] = str(
        (path.parent / case["optimization_config"]).resolve(strict=True)
    )
    return case, inventory, files, bind(inventory_path)


def contact_inputs(stage, backend, inputs, contact_map=None, contact_temporal=None):
    """Bind agent-authored contacts separately from unchanged Stage 1 observations."""
    if stage == "stage1":
        if contact_map is not None or contact_temporal is not None:
            raise ValueError("Stage 1 does not consume contact attraction annotations")
        return {}
    if (contact_map is None) != (contact_temporal is None):
        raise ValueError("Provide both contact-map and contact-temporal")
    if contact_map is not None:
        return {
            f"{backend}/contact_map": bind(contact_map),
            f"{backend}/contact_temporal": bind(contact_temporal),
        }
    keys = [f"{backend}/contact_map", f"{backend}/contact_temporal"]
    if any(key not in inputs for key in keys):
        raise ValueError("Stage 2 requires authored contact-map and contact-temporal")
    return {key: inputs[key] for key in keys}


def camera_depth_axis(camera_rotation):
    """Use the placement producer's float64 unit axis for depth projection."""
    rotation = np.asarray(camera_rotation, dtype=np.float64).reshape(3, 3)
    if (
        not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Camera rotation must be a proper orthonormal rotation")
    axis = rotation[:, 2].copy()
    return axis / np.linalg.norm(axis)


def validate_stage2_parent(stage1_dir, parent, inventory, contact_map):
    """Check edited placement or an explicitly reviewed byte-identical parent."""
    import joblib

    from grail.agentic_kinematic.hoi_placement import (
        correction_weights,
        inspect_unchanged_placement,
    )

    first, parent = Path(stage1_dir).resolve(strict=True), Path(parent).resolve(strict=True)
    events = placement_events(inventory["placement"], inventory["frame_count"])
    parent_data = load_native_pickle(parent)
    baseline = load_native_pickle(first / "stage1.pkl")
    if events.get("object_motion_onset_status") == "unknown":
        source_inputs = json.loads((first / "inputs.json").read_text())
        review = source_inputs["files"]["source_motion_review"]
        if bind(review["path"]) != review:
            raise ValueError("Stage 1 unknown-onset review changed")
        for value in (baseline, parent_data):
            meta = value.get("meta", {})
            if (
                meta.get("object_motion_onset_status") != "unknown"
                or "object_first_motion_frame" not in meta
                or meta["object_first_motion_frame"] is not None
                or type(meta.get("object_stationary_prefix_length")) is not int
                or meta.get("object_stationary_prefix_length") != 0
                or meta.get("source_contact_frame") != events["contact_frame"]
                or meta.get("alignment_frame") != events["alignment_frame"]
                or meta.get("source_motion_review") != review
            ):
                raise ValueError("Stage 2 parent changed unknown object-motion metadata lineage")
    if events["mode"] == "reviewed_unchanged":
        if parent.name != "stage1_object_refined.pkl":
            raise ValueError("Unchanged placement requires its verified identity-parent artifact")
        inspect_unchanged_placement(first, parent.parent, contact_map)
        if bind(parent)["sha256"] != bind(first / "stage1.pkl")["sha256"]:
            raise ValueError("Unchanged placement changed the native Stage 1 parent")
        return parent_data
    if joblib.hash(parent_data["human_data"]) != joblib.hash(baseline["human_data"]):
        raise ValueError("Stage 1 placement changed the human")
    for key in ("obj_R", "obj_scale"):
        if not np.array_equal(parent_data["obj_data"][key], baseline["obj_data"][key]):
            raise ValueError("Placement changed object rotation or scale")
    tail = inventory["placement"]["recovery_end_frame"]
    if not np.array_equal(
        parent_data["obj_data"]["obj_t"][tail:], baseline["obj_data"]["obj_t"][tail:]
    ):
        raise ValueError("Placement changed protected tail")
    placement_dir = parent.parent
    receipt = json.loads((placement_dir / "completion.json").read_text())
    if (
        receipt["status"] != "PLACEMENT_COMPLETE_UNREVIEWED"
        or receipt["outputs"]["stage1_object_refined.pkl"] != bind(parent)["sha256"]
        or not all(receipt["checks"].values())
    ):
        raise ValueError("Placement receipt is incomplete or does not bind this parent")
    plan = json.loads((placement_dir / "edit_plan.json").read_text())
    if events.get("object_motion_onset_status") == "unknown":
        if (
            plan.get("object_motion_onset_status") != "unknown"
            or "object_first_motion_frame" not in plan
            or plan["object_first_motion_frame"] is not None
            or "stationary_prefix" not in plan
            or plan.get("stationary_prefix") is not None
            or plan.get("source_motion_review") != review
        ):
            raise ValueError("Placement plan changed unknown object-motion evidence")
    start = inventory["placement"]["recovery_start_frame"]
    if (
        plan["recovery_start_frame"] != start
        or plan["recovery_end_frame"] != tail
        or plan["contact_frame"] != events["alignment_frame"]
    ):
        raise ValueError("Placement changed declared source events")
    with np.load(first / "stage1_geometry.npz") as geometry:
        axis = camera_depth_axis(geometry["camera_R"])
    delta = np.asarray(plan["delta_world"])
    if not np.allclose(delta, axis * np.dot(delta, axis), atol=1e-7, rtol=0):
        raise ValueError("Placement correction is not camera-depth translation")
    expected = (
        baseline["obj_data"]["obj_t"]
        + correction_weights(np.arange(inventory["frame_count"]), start, tail)[:, None] * delta
    )
    if not np.allclose(parent_data["obj_data"]["obj_t"], expected, atol=2e-5, rtol=0):
        raise ValueError("Placement parent does not match its declared correction")
    return parent_data


def validate_inactive_contact_scope(contact_map, contact_temporal, fps):
    from grail.optimization.contact_temporal_scope import load_temporal_scope

    options, _ = load_temporal_scope(
        str(contact_temporal), contact_map_sha256=bind(contact_map)["sha256"], defaults={}
    )
    if (
        options["hold_interval_ids"] != {"left": [], "right": []}
        or options["windows"] != []
        or type(options["fps"]) not in (int, float)
        or not np.isclose(options["fps"], fps, rtol=0, atol=1e-9)
    ):
        raise ValueError("No-evidenced-contact Stage 2 requires empty holds/windows and source FPS")


def object_motion_policy(case, inventory, inputs):
    """Assess original bound poses before any prefix freezing or placement edit."""
    import yaml

    from grail.optimization.stage2_policy import stage2_skip_decision

    config = yaml.safe_load(Path(case["optimization_config"]).read_text())
    cfg = copy.deepcopy(config["optimization"])
    cfg.setdefault("is_static_obj", config.get("pipeline", {}).get("is_static_obj", False))
    poses = np.asarray(load_native_pickle(inputs["object_poses"]["path"]))
    if poses.shape != (inventory["frame_count"], 4, 4):
        raise ValueError("Original object poses must cover the entire source timeline")
    decision = stage2_skip_decision(cfg, poses)
    static = stage2_skip_decision({**cfg, "skip_stage2_static_obj": True}, poses)
    decision["object_static"] = static["skip"]
    if static["reason"] == "invalid_object_poses":
        raise ValueError("Invalid original object pose sequence")
    if inventory["placement"]["object_first_motion_frame"] == inventory["frame_count"]:
        # The end-exclusive no-motion sentinel cannot hide actual source motion,
        # even when a separate static-contact task explicitly disables skipping.
        if not static["skip"]:
            raise ValueError("No-motion sentinel contradicts the original object trajectory")
    if inventory["placement"].get("object_motion_onset_status") == "unknown":
        # Uncertainty is not evidence for a static bypass, even if a tracker
        # happened to return a constant trajectory. Keep physical diagnostics.
        decision.update(
            skip=False,
            reason="unknown_source_motion_onset",
            object_static=False,
            tracked_poses_static=static["skip"],
        )
    return cfg, decision


def inspect_stage2_skip(case_path, parent, stage1_dir):
    """Verify a static bypass using bound source poses and a completed native parent.

    Shared with the allocation broker so a zero-iteration receipt cannot bypass
    the normal stage contract merely by declaring itself static.
    """
    import joblib

    from grail.optimization.stage2_policy import stage2_skip_decision

    case, inventory, inputs, inventory_binding = load_case(case_path, require_contacts=False)
    cfg, decision = object_motion_policy(case, inventory, inputs)
    if not decision["skip"]:
        return None
    first = Path(stage1_dir).resolve(strict=True)
    receipt = json.loads((first / "completion.json").read_text())
    checks = receipt.get("checks")
    if (
        receipt.get("status") != "OPTIMIZATION_COMPLETE_REVIEW_PENDING"
        or receipt.get("stage") != "stage1"
        or receipt.get("backend") != case["backend"]
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or type(receipt.get("iterations")) is not int
        or receipt["iterations"] < 1
    ):
        raise ValueError("Static Stage 2 requires a complete checked Stage 1 receipt")
    first_binding = bind(first / "stage1.pkl")
    if receipt["output"] != first_binding:
        raise ValueError("Stage 1 native output changed after completion")
    parent_geometry = bind(first / "stage1_geometry.npz")
    if receipt.get("geometry") != parent_geometry:
        raise ValueError("Stage 1 geometry is unbound or changed after completion")
    source_bindings = json.loads((first / "inputs.json").read_text())
    if (
        source_bindings["inventory"]["sha256"] != inventory_binding["sha256"]
        or source_bindings["case"]["sha256"] != bind(case_path)["sha256"]
        or source_bindings["optimization_config"]["sha256"]
        != bind(case["optimization_config"])["sha256"]
    ):
        raise ValueError("Stage 1 has different source inputs or configuration")
    parent_binding = bind(parent)
    baseline = load_native_pickle(first_binding["path"])
    candidate = load_native_pickle(parent_binding["path"])
    parent_kind = "stage1"
    if parent_binding["sha256"] != first_binding["sha256"]:
        # The same placement contract as numerical Stage 2; it is not permission
        # to replace the human or to supply an arbitrary native checkpoint.
        placement = Path(parent_binding["path"]).parent
        placement_receipt = json.loads((placement / "completion.json").read_text())
        placement_checks = placement_receipt.get("checks")
        if (
            placement_receipt.get("status") != "PLACEMENT_COMPLETE_UNREVIEWED"
            or placement_receipt["outputs"]["stage1_object_refined.pkl"] != parent_binding["sha256"]
            or not isinstance(placement_checks, dict)
            or not placement_checks
            or not all(value is True for value in placement_checks.values())
        ):
            raise ValueError("Placement receipt does not bind a checked parent")
        if joblib.hash(candidate["human_data"]) != joblib.hash(baseline["human_data"]):
            raise ValueError("Placement changed the human")
        if set(candidate) != set(baseline):
            raise ValueError("Placement changed native fields")
        for key in baseline:
            if key not in ("obj_data", "meta") and joblib.hash(candidate[key]) != joblib.hash(
                baseline[key]
            ):
                raise ValueError(f"Placement changed frozen native field: {key}")
        if set(candidate["obj_data"]) != set(baseline["obj_data"]):
            raise ValueError("Placement changed native object fields")
        for key in baseline["obj_data"]:
            if key != "obj_t" and joblib.hash(candidate["obj_data"][key]) != joblib.hash(
                baseline["obj_data"][key]
            ):
                raise ValueError(f"Placement changed frozen object field: {key}")
        parent_geometry = bind(placement / "refined_geometry.npz")
        if placement_receipt["outputs"].get("refined_geometry.npz") != parent_geometry["sha256"]:
            raise ValueError("Placement geometry changed after completion")
        from grail.agentic_kinematic.hoi_placement import correction_weights

        plan = json.loads((placement / "edit_plan.json").read_text())
        events = inventory["placement"]
        alignment = placement_events(events, inventory["frame_count"])["alignment_frame"]
        # The legacy placement-plan key names the alignment sample, not first touch.
        if plan["contact_frame"] != alignment:
            raise ValueError("Placement changed declared source events")
        for key in ("recovery_start_frame", "recovery_end_frame"):
            if plan[key] != events[key]:
                raise ValueError("Placement changed declared source events")
        with np.load(first / "stage1_geometry.npz") as geometry:
            axis = camera_depth_axis(geometry["camera_R"])
        delta = np.asarray(plan["delta_world"])
        if not np.allclose(delta, axis * np.dot(delta, axis), atol=1e-7, rtol=0):
            raise ValueError("Placement correction is not camera-depth translation")
        tail = events["recovery_end_frame"]
        expected = (
            baseline["obj_data"]["obj_t"]
            + correction_weights(
                np.arange(inventory["frame_count"]), events["recovery_start_frame"], tail
            )[:, None]
            * delta
        )
        if not np.allclose(
            candidate["obj_data"]["obj_t"], expected, atol=2e-5, rtol=0
        ) or not np.array_equal(
            candidate["obj_data"]["obj_t"][tail:], baseline["obj_data"]["obj_t"][tail:]
        ):
            raise ValueError("Placement parent does not match its declared correction")
        parent_kind = "placement"
    n = inventory["frame_count"]
    obj = candidate["obj_data"]
    if np.shape(obj["obj_R"]) != (n, 3, 3) or np.shape(obj["obj_t"]) != (n, 3):
        raise ValueError("Native parent object trajectory has a different timeline")
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = obj["obj_R"], obj["obj_t"]
    parent_decision = stage2_skip_decision(cfg, poses)
    if not parent_decision["skip"]:
        return None
    return dict(
        object_motion=decision,
        parent_object_motion=parent_decision,
        parent=parent_binding,
        case=bind(case_path),
        stage1_output=first_binding,
        parent_geometry=parent_geometry,
        parent_kind=parent_kind,
    )


def export_static_stage2(case_path, parent, stage1_dir, output, evidence):
    """Carry the selected native checkpoint forward byte-for-byte, without fitting."""
    started = time.monotonic()
    out = Path(output).resolve()
    first = Path(stage1_dir).resolve(strict=True)
    geometry = Path(evidence["parent_geometry"]["path"])
    geometry_binding = bind(geometry)
    if geometry_binding != evidence["parent_geometry"] or bind(parent) != evidence["parent"]:
        raise ValueError("Static parent changed after inspection")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(parent, out / "stage2.pkl")
    shutil.copy2(geometry, out / "stage2_geometry.npz")
    for name in (
        "global_reference.pkl",
        "source_observations.npz",
        "fixed_contact_binding.json",
        "fixed_contact_geometry.npz",
        "native_hand_segments.json",
    ):
        if (first / name).is_file():
            shutil.copy2(first / name, out / name)
    case, inventory, inputs, inventory_binding = load_case(case_path, require_contacts=False)
    dump(
        out / "inputs.json",
        dict(
            case=bind(case_path),
            inventory=inventory_binding,
            files=inputs,
            optimization_config=bind(case["optimization_config"]),
            parent=evidence["parent"],
            authored_contacts={},
            object_motion=evidence["object_motion"],
        ),
    )
    dump(out / "history.json", [])
    actual = bind(out / "stage2.pkl")
    checks = dict(
        parent_bytes_preserved=actual["sha256"] == evidence["parent"]["sha256"],
        source_object_static=evidence["object_motion"]["skip"],
        parent_object_static=evidence["parent_object_motion"]["skip"],
        geometry_bytes_preserved=bind(out / "stage2_geometry.npz")["sha256"]
        == geometry_binding["sha256"],
    )
    if not all(checks.values()):
        raise ValueError("Static Stage 2 export changed the selected parent")
    completion = dict(
        status="SKIPPED_STATIC_OBJECT",
        stage="stage2",
        backend=case["backend"],
        iterations=0,
        checks=checks,
        solve_seconds=0.0,
        elapsed_seconds=time.monotonic() - started,
        motion_accepted=False,
        output=actual,
        geometry=bind(out / "stage2_geometry.npz"),
        runner=bind(__file__),
        **evidence,
    )
    dump(out / "completion.json", completion)
    return completion


def run(
    case_path,
    stage,
    output,
    *,
    parent=None,
    stage1_dir=None,
    iterations=None,
    device="cuda",
    contact_map=None,
    contact_temporal=None,
):
    if stage not in ("stage1", "stage2"):
        raise ValueError("Expected stage1 or stage2")
    if stage == "stage2" and (parent is None or stage1_dir is None):
        raise ValueError("Stage 2 requires a native parent and original Stage 1 directory")
    if stage == "stage1" and (parent is not None or stage1_dir is not None):
        raise ValueError("Stage 1 starts from bound predictor observations")
    if iterations is not None and (type(iterations) is not int or iterations < 0):
        raise ValueError("iterations must be a nonnegative integer")
    if stage == "stage2":
        evidence = inspect_stage2_skip(case_path, parent, stage1_dir)
        if evidence is not None:
            return export_static_stage2(case_path, parent, stage1_dir, output, evidence)
    import cv2
    import joblib
    import torch
    import yaml

    from grail.core.io import load_mesh, load_object_pose_data
    from grail.core.torch_utils import tensor_to_numpy
    from grail.optimization.ground_config import configure_loss_support
    from grail.optimization.hoi_optimizer import HOIOptimizer
    from grail.optimization.loss_computer import LossComputer
    from grail.preprocessing.preprocess import load_depth_from_cache, load_masks_from_cache
    from grail.rendering.camera import transform_pose_c2w

    case, inventory, inputs, inventory_binding = load_case(case_path, require_contacts=False)
    _, object_motion = object_motion_policy(case, inventory, inputs)
    authored_contacts = contact_inputs(
        stage, case["backend"], inputs, contact_map, contact_temporal
    )
    inputs = {**inputs, **authored_contacts}
    out = Path(output).resolve()
    protected = [Path(v["path"]) for v in inputs.values()] + [Path(case_path).resolve()]
    if out.exists() or any(out == p or out in p.parents for p in protected):
        raise ValueError("Use a new output directory outside protected inputs")
    if stage not in ("stage1", "stage2"):
        raise ValueError("Expected stage1 or stage2")
    if stage == "stage2" and (parent is None or stage1_dir is None):
        raise ValueError("Stage 2 requires the placement parent and original Stage 1 directory")
    if stage == "stage1" and (parent is not None or stage1_dir is not None):
        raise ValueError("Stage 1 starts from bound predictor observations")
    out.mkdir(parents=True)
    start = time.monotonic()
    dump(out / "completion.json", dict(status="RUNNING", stage=stage, motion_accepted=False))
    backend = case["backend"]
    n, fps = inventory["frame_count"], inventory["fps"]
    events = placement_events(inventory["placement"], n)
    move, touch = events["object_first_motion_frame"], events["contact_frame"]
    prefix_length = object_prefix_length(events)

    def path(key):
        return inputs[key]["path"]

    if stage == "stage2" and events["mode"] == "reviewed_unchanged":
        validate_inactive_contact_scope(
            path(f"{backend}/contact_map"), path(f"{backend}/contact_temporal"), fps
        )
    cfg = copy.deepcopy(
        yaml.safe_load(Path(case["optimization_config"]).read_text())["optimization"]
    )
    configure_loss_support(cfg, fps)
    cfg["human_model"] = case["human_model"]
    name = "init_opt" if stage == "stage1" else "contact_opt"
    cfg["opt_stage_specs"] = {name: cfg["opt_stage_specs"][name]}
    spec = cfg["opt_stage_specs"][name]
    if iterations is not None:
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        spec["niter"] = iterations
    cfg.update(
        vis_cfg={"enable": False},
        eval={},
        export_contact_review_inputs=True,
        output_smoothing=False,
    )
    if stage == "stage2":
        spec["loss_cfg"].pop("depth_pointcloud", None)
        spec["loss_cfg"]["contact"]["prediction_path"] = path(f"{backend}/contact_map")
        spec["loss_cfg"]["phase_contact_temporal"]["scope_path"] = path(
            f"{backend}/contact_temporal"
        )
        # SOMA stores the wrist controls in hand slots 0/25. All individual
        # finger rotations are restored before FK by frozen_apply below.
        spec["opt_vars"]["hand_pose_res"] = {"lr": spec["opt_vars"]["human_pose_res"]["lr"]}
    predictor_status = inventory.get("predictors", "REUSED_HASH_BOUND")
    dump(
        out / "inputs.json",
        dict(
            case=bind(case_path),
            inventory=inventory_binding,
            files=inputs,
            optimization_config=bind(case["optimization_config"]),
            predictors=predictor_status,
            authored_contacts=authored_contacts,
            object_motion=object_motion,
            parent=bind(parent) if parent else None,
        ),
    )
    torch.set_num_threads(4)
    torch.manual_seed(20261008)
    np.random.seed(20261008)

    class BoundOptimizer(HOIOptimizer):
        def _load_video_frames(self, video):
            frames = out / "source_frames"
            frames.mkdir()
            cap = cv2.VideoCapture(video)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            paths = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                target = frames / f"{len(paths):06d}.png"
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError("Frame export failed")
                paths.append(str(target))
            cap.release()
            if len(paths) != n or abs(actual_fps - fps) > 1e-3:
                raise ValueError("Source video timing changed")
            return paths, fps, n

        def _load_masks(self, video_id):
            masks = load_masks_from_cache(path("mask"))
            return {i: masks[i][1] for i in range(n)}, {i: masks[i][0] for i in range(n)}

        def _load_object(self, mesh, scale, poses, cam_R, cam_t):
            if not np.allclose(
                np.asarray(tensor_to_numpy(scale)),
                inventory["original_object_scale"],
                rtol=0,
                atol=1e-7,
            ):
                raise ValueError("Original object scale changed")
            vertices, faces, _ = load_mesh(
                mesh, mesh_scale=scale, target_num_verts=None, device=self.device
            )
            incam = load_object_pose_data(poses, to_tensor=True, device=self.device)
            canonical = prepare_object_poses(tensor_to_numpy(incam), prefix_length)
            incam = torch.as_tensor(canonical, dtype=incam.dtype, device=incam.device)
            return vertices, faces, incam, transform_pose_c2w(incam, cam_R, cam_t)

        def _detect_interaction(self, *args):
            # This legacy return slot chooses the native human alignment frame;
            # regional contact activation comes only from the authored map.
            return events["alignment_frame"], n, object_motion["object_static"]

        def _load_depth(self, video_id):
            # Original sidecar bytes stay unchanged; inventory binds relocation.
            return (
                load_depth_from_cache(
                    path("depth"), device=self.device, metadata_binding=inputs["depth_metadata"]
                )
                if stage == "stage1"
                else None
            )

        def forward(self, data, params, **kwargs):
            prediction = super().forward(data, params, **kwargs)
            prediction.obj.R, prediction.obj.trans = constrain_object_prefix(
                prediction.obj.R, prediction.obj.trans, data.obj.poses, prefix_length
            )
            prediction.obj.verts_seq = (
                data.obj.verts[None] @ prediction.obj.R.transpose(1, 2)
                + prediction.obj.trans[:, None]
            )
            return prediction

    opt = BoundOptimizer(
        inventory["sequence"], cfg, str(out / "cache"), str(out / "logs"), device=device
    )
    original_apply = opt.human_model.apply_pose_residuals
    editable = torch.zeros(77, dtype=torch.bool, device=device)
    editable[[i + 1 for i in opt.human_model.BODY_POSE_INDICES] + [14, 42]] = True
    editable[0] = False

    def frozen_apply(motion, body, hands, count):
        original = motion["poses"].clone()
        result = original_apply(motion, body, hands, count)
        result["poses"] = torch.where(editable[None, :, None], result["poses"], original)
        return result

    opt.human_model.apply_pose_residuals = frozen_apply
    data = opt.init_data(
        path("source_video"),
        path(f"{backend}/human_motion"),
        path("object_mesh"),
        path("object_poses"),
        path("render_config"),
    )
    # Legacy regularization consumes a prefix extent, not the observed onset.
    # Unknown onset has no stationary-prefix penalty; alignment stays independent.
    data.inter_start_idx = prefix_length
    original_reference = tensor_to_numpy(data.human.motion_data_global_init)

    def tensorize(value):
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value.copy(), device=device)
        if isinstance(value, dict):
            return {k: tensorize(v) for k, v in value.items()}
        return copy.deepcopy(value)

    observations = dict(
        gt_body_2d=tensor_to_numpy(data.human.body_keypoints_seq),
        gt_hand_2d=tensor_to_numpy(data.human.hand_keypoints_seq),
        foot_contact_probs=tensor_to_numpy(data.human.foot_contact_probs),
        object_tracking_2d=tensor_to_numpy(data.obj.verts_tracking_seq),
    )
    if stage == "stage2":
        first = Path(stage1_dir).resolve(strict=True)
        receipt = json.loads((first / "completion.json").read_text())
        if (
            receipt["status"] != "OPTIMIZATION_COMPLETE_REVIEW_PENDING"
            or receipt["stage"] != "stage1"
            or receipt["backend"] != backend
        ):
            raise ValueError("Stage 2 requires a complete Stage 1 receipt")
        stage1_inputs = json.loads((first / "inputs.json").read_text())
        if stage1_inputs["inventory"]["sha256"] != inventory_binding["sha256"]:
            raise ValueError("Stage 1 has different source inputs")
        if stage1_inputs["case"]["sha256"] != bind(case_path)["sha256"]:
            raise ValueError("Stage 1 has a different model/configuration")
        if bind(first / "stage1.pkl")["sha256"] != receipt["output"]["sha256"]:
            raise ValueError("Stage 1 native output changed after completion")
        saved_reference = load_native_pickle(first / "global_reference.pkl")
        for key in ("poses", "trans"):
            if not np.allclose(saved_reference[key], original_reference[key], rtol=0, atol=1e-5):
                raise ValueError("Original global-motion reference changed")
        with np.load(first / "source_observations.npz") as old:
            for key, value in observations.items():
                if not np.array_equal(old[key], value, equal_nan=True):
                    raise ValueError(f"Source observations changed: {key}")
        parent_data = validate_stage2_parent(
            first, parent, inventory, path(f"{backend}/contact_map")
        )
        data.human.motion_data = tensorize(parent_data["human_data"])
        data.human.motion_data_global_init = tensorize(saved_reference)
        # Export the exact effective prior, not a fresh floating-point
        # re-evaluation of the predictor's alignment performed during init_data.
        original_reference = saved_reference
        data.obj.poses[:, :3, :3] = tensorize(parent_data["obj_data"]["obj_R"])
        data.obj.poses[:, :3, 3] = tensorize(parent_data["obj_data"]["obj_t"])
        data.obj.verts_seq = (
            data.obj.verts[None] @ data.obj.poses[:, :3, :3].transpose(1, 2)
            + data.obj.poses[:, None, :3, 3]
        )
    joblib.dump(original_reference, out / "global_reference.pkl")
    np.savez_compressed(out / "source_observations.npz", **observations)
    dump(out / "fixed_contact_binding.json", data.fixed_contact_context["binding"])
    np.savez_compressed(
        out / "fixed_contact_geometry.npz",
        human_faces=tensor_to_numpy(data.human.faces),
        object_vertices=tensor_to_numpy(data.obj.verts),
        object_faces=tensor_to_numpy(data.obj.faces),
    )
    # Native anatomical masks are review aids, not an authored contact map.
    # Export once so the in-job agent can inspect saved geometry on CPU.
    dump(
        out / "native_hand_segments.json",
        {
            "left_hand": np.unique(opt.human_model.get_segment_indices(["L_Hand"])).tolist(),
            "right_hand": np.unique(opt.human_model.get_segment_indices(["R_Hand"])).tolist(),
            "semantics": "Anatomical hand vertex masks only; no contact regions or phases selected",
        },
    )
    opt.init_params(data)
    opt.loss_computer = LossComputer(
        opt.cameras, opt.human_model, device, opt.num_body_joints, opt.logger
    )
    loss_cfg = opt.opt_stage_specs[name]["loss_cfg"]
    if stage == "stage2":
        opt.loss_computer.prepare_phase_contact_map(data, loss_cfg["contact"])
        opt.loss_computer.prepare_phase_contact_temporal(data, loss_cfg["phase_contact_temporal"])
    dump(out / "executed_config.json", opt.cfg)

    def save(label):
        with torch.no_grad():
            pred = opt.forward(data, opt.params)
            old, new = data.human.motion_data, pred.human.motion_data
            keys = ("identity_coeffs", "scale_params", "scale", "left_hand_pose", "right_hand_pose")
            checks = {
                key: bool(np.array_equal(tensor_to_numpy(old[key]), tensor_to_numpy(new[key])))
                for key in keys
            }
            a, b = old["poses"].reshape(n, -1, 3), new["poses"].reshape(n, -1, 3)
            frozen = [i for i in range(77) if not bool(editable[i])]
            checks.update(
                frozen_native_joints_exact=bool(torch.equal(a[:, frozen], b[:, frozen])),
                object_prefix_R_exact=bool(
                    torch.equal(pred.obj.R[:prefix_length], data.obj.poses[:prefix_length, :3, :3])
                ),
                object_prefix_t_exact=bool(
                    torch.equal(
                        pred.obj.trans[:prefix_length], data.obj.poses[:prefix_length, :3, 3]
                    )
                ),
                source_timing=len(b) == n and float(new["mocap_frame_rate"]) == fps,
                finite_geometry=bool(
                    torch.isfinite(pred.human.verts_seq).all()
                    and torch.isfinite(pred.obj.verts_seq).all()
                ),
            )
            if not all(checks.values()):
                raise ValueError(checks)
            native = opt.get_optimized_data(data, pred, smooth=False)
            native["meta"].update(
                stage=name,
                model=backend,
                status="UNREVIEWED",
                source_fps=fps,
                predictors=predictor_status,
                object_first_motion_frame=move,
                source_contact_frame=touch,
                alignment_frame=events["alignment_frame"],
                placement_mode=events["mode"],
                stage2_static_policy=object_motion,
            )
            if events.get("object_motion_onset_status") == "unknown":
                native["meta"].update(
                    object_motion_onset_status="unknown",
                    object_stationary_prefix_length=0,
                    source_motion_review=inputs["source_motion_review"],
                )
            joblib.dump(native, out / f"{label}.pkl")
            np.savez_compressed(
                out / f"{label}_geometry.npz",
                human_vertices=tensor_to_numpy(pred.human.verts_seq),
                human_faces=tensor_to_numpy(data.human.faces),
                object_vertices=tensor_to_numpy(pred.obj.verts_seq),
                object_faces=tensor_to_numpy(data.obj.faces),
                joints=tensor_to_numpy(opt.human_model.get_native_joints(new)),
                body_2d=tensor_to_numpy(pred.human.body_keypoints_seq),
                hand_2d=tensor_to_numpy(pred.human.hand_keypoints_seq),
                camera_R=tensor_to_numpy(opt.cameras.R),
                camera_T=tensor_to_numpy(opt.cameras.T),
                focal=float(data.camera.focal_length),
                image_size=[data.camera.frame_height, data.camera.frame_width],
            )
            return checks

    save("initial")
    optimizer, active = opt.init_opt(data, opt.params, opt.opt_stage_specs[name])
    history = []
    solve_start = time.monotonic()
    for step in range(spec["niter"]):
        optimizer.zero_grad(set_to_none=True)
        pred = opt.forward(data, opt.params)
        total, parts = opt.loss_computer.compute_loss(data, pred, loss_cfg)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite loss")
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in active.values()):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        item = dict(
            iteration=step + 1,
            seconds=time.monotonic() - solve_start,
            total=float(total.detach()),
            losses=parts,
        )
        history.append(item)
        if step == 0 or (step + 1) % 20 == 0:
            dump(out / "progress.json", item)
            print(json.dumps(item), flush=True)
        if (step + 1) % 100 == 0:
            save(f"checkpoint_{step+1:04d}")
    checks = save(stage)
    dump(out / "history.json", history)
    completion = dict(
        status="OPTIMIZATION_COMPLETE_REVIEW_PENDING" if spec["niter"] else "PREFLIGHT_ONLY",
        stage=stage,
        backend=backend,
        iterations=spec["niter"],
        checks=checks,
        solve_seconds=time.monotonic() - solve_start,
        elapsed_seconds=time.monotonic() - start,
        motion_accepted=False,
        object_motion=object_motion,
        output=bind(out / f"{stage}.pkl"),
        geometry=bind(out / f"{stage}_geometry.npz"),
        runner=bind(__file__),
    )
    dump(out / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stage1", "stage2"])
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent")
    parser.add_argument("--stage1-dir")
    parser.add_argument("--contact-map", help="New source-reviewed map authored after Stage 1")
    parser.add_argument("--contact-temporal", help="Temporal scope paired with the authored map")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.case,
                args.stage,
                args.output,
                parent=args.parent,
                stage1_dir=args.stage1_dir,
                iterations=args.iterations,
                device=args.device,
                contact_map=args.contact_map,
                contact_temporal=args.contact_temporal,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
