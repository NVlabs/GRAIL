"""Object-only Stage 1 placement through an authored Blender translation control.

Regional centers determine gross camera-depth alignment, not exact surface contact.
The input human and protected object tail remain byte-for-byte equal as arrays.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np

from grail.core.native_pickle import load_native_pickle
from grail.optimization.contact_maps import PhaseContactMap
from grail.optimization.fixed_contact import file_sha256


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def correction_weights(frames, start, end):
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("Transition bounds must be increasing nonnegative integer frames")
    u = np.clip((np.asarray(frames, dtype=float) - start) / (end - start), 0, 1)
    return 1 - (10 * u**3 - 15 * u**4 + 6 * u**5)


def exact(first, second):
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(exact(value, second[key]) for key, value in first.items())
        )
    if isinstance(first, np.ndarray):
        return (
            isinstance(second, np.ndarray)
            and first.dtype == second.dtype
            and first.shape == second.shape
            and np.array_equal(first, second, equal_nan=True)
        )
    if isinstance(first, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(exact(a, b) for a, b in zip(first, second))
        )
    return first == second


def prepare_plan(native, geometry, local, binding, contact_spec, contact_frame, start, end):
    """Validate native caches and compute a single translation without fitting."""
    obj = native["obj_data"]
    rotations, translations = np.asarray(obj["obj_R"]), np.asarray(obj["obj_t"])
    count = len(translations)
    if (
        any(type(value) is not int for value in (contact_frame, start, end))
        or not 0 <= contact_frame <= start < end < count
    ):
        raise ValueError("Require 0 <= contact frame <= transition start < end < frame count")
    if rotations.shape != (count, 3, 3) or translations.shape != (count, 3):
        raise ValueError("Native object poses must have [T,3,3] / [T,3] shape")
    for value in [
        rotations,
        translations,
        geometry["human_vertices"],
        geometry["object_vertices"],
        geometry["camera_R"],
        geometry["camera_T"],
        local["object_vertices"],
    ]:
        if not np.isfinite(value).all():
            raise ValueError("Native geometry and camera must be finite")
    human = np.asarray(geometry["human_vertices"])
    vertices = np.asarray(local["object_vertices"])
    if (
        human.ndim != 3
        or human.shape[0] != count
        or human.shape[-1] != 3
        or binding["frame_count"] != count
        or binding["human_vertex_count"] != human.shape[1]
    ):
        raise ValueError("Native human geometry and contact timeline differ")
    if not np.array_equal(geometry["human_faces"], local["human_faces"]) or not np.array_equal(
        geometry["object_faces"], local["object_faces"]
    ):
        raise ValueError("Original mesh topology differs between native caches")
    expected = vertices[None] @ rotations.transpose(0, 2, 1) + translations[:, None]
    if np.asarray(geometry["object_vertices"]).shape != expected.shape or not np.allclose(
        geometry["object_vertices"], expected, rtol=0, atol=1e-5
    ):
        raise ValueError("Object geometry differs from original local mesh and native transforms")
    # A sampled first touch can coincide with the first moving frame. Preserve
    # that motion: stationarity ends before the declared motion onset, while
    # the placement correction can remain constant through the contact key.
    meta = native.get("meta", {})
    move = meta.get("object_first_motion_frame", start + 1)
    unknown = meta.get("object_motion_onset_status") == "unknown"
    if unknown:
        from grail.agentic_kinematic.hoi_stages import bind

        review = meta.get("source_motion_review", {})
        if (
            move is not None
            or type(meta.get("object_stationary_prefix_length")) is not int
            or meta.get("object_stationary_prefix_length") != 0
            or meta.get("source_contact_frame") != contact_frame
            or not review
            or bind(review["path"]) != review
        ):
            raise ValueError("Unknown onset placement requires unchanged native review metadata")
        decision = json.loads(Path(review["path"]).read_text())
        if (
            decision.get("schema") != "grail.object_motion_onset_review.v1"
            or decision.get("object_motion_onset_status") != "unknown"
            or "object_first_motion_frame" not in decision
            or decision["object_first_motion_frame"] is not None
            or decision.get("source_video_sha256") != binding["source_video_sha256"]
            or decision.get("frame_count") != count
            or decision.get("fps") != binding["fps"]
        ):
            raise ValueError("Unknown onset placement review differs from native source/timing")
    else:
        if type(move) is not int or not 0 <= move < count or start < move - 1:
            raise ValueError("Transition must not recover during the declared stationary prefix")
        if not np.array_equal(
            translations[:move], np.broadcast_to(translations[0], (move, 3))
        ) or not np.array_equal(rotations[:move], np.broadcast_to(rotations[0], (move, 3, 3))):
            raise ValueError("Input object prefix must already be stationary before first motion")
    camera_R = np.asarray(geometry["camera_R"]).reshape(3, 3)
    if not np.allclose(camera_R.T @ camera_R, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(
        np.linalg.det(camera_R), 1, atol=1e-5
    ):
        raise ValueError("Camera rotation must be a proper orthonormal rotation")
    fps = float(binding["fps"])
    native_fps = native["human_data"].get("mocap_frame_rate")
    if native_fps is not None and not np.isclose(float(native_fps), fps, atol=1e-8, rtol=0):
        raise ValueError("Native FPS differs from contact binding")
    PhaseContactMap.from_spec(
        contact_spec,
        binding=binding,
        human_faces=local["human_faces"],
        object_vertices=vertices,
        object_faces=local["object_faces"],
    )
    regions, hand_errors = [], []
    for hand, phases in contact_spec["hands"].items():
        phase = next(p for p in phases if p["frames"][0] <= contact_frame < p["frames"][1])
        if phase["state"] != "contact":
            continue
        errors = []
        for region in phase["regions"]:
            hc = np.average(
                human[contact_frame, region["human_vertex_ids"]],
                axis=0,
                weights=region["human_strength"],
            )
            oc = np.average(
                expected[contact_frame, region["object_vertex_ids"]],
                axis=0,
                weights=region["object_strength"],
            )
            errors.append(hc - oc)
            regions.append(
                dict(
                    hand=hand,
                    phase=phase["id"],
                    region=region["id"],
                    human_center=hc.tolist(),
                    object_center=oc.tolist(),
                )
            )
        hand_errors.append(np.mean(errors, axis=0))
    if not hand_errors:
        raise ValueError("Contact frame has no source-authored active contact region")
    axis = camera_R[:, 2].astype(float)
    axis /= np.linalg.norm(axis)
    offset = float(np.mean(hand_errors, axis=0) @ axis)
    delta = offset * axis
    plan = dict(
        schema="grail.stage1_object_placement.v1",
        contact_frame=contact_frame,
        recovery_start_frame=start,
        recovery_end_frame=end,
        fps=fps,
        stationary_prefix=None if unknown else [0, move - 1],
        object_first_motion_frame=move,
        tail_exact_from_frame=end,
        delta_world=delta.tolist(),
        camera_depth_offset_m=offset,
        human_frozen=True,
        object_rotation_and_scale_frozen=True,
        translation_only=True,
        frame_mapping="Blender frame = source frame + 1",
        curve="Constant correction before transition; quintic fade to zero; exact tail",
        status="UNREVIEWED_GROSS_PLACEMENT",
        caption_language="English",
    )
    if unknown:
        plan.update(object_motion_onset_status="unknown", source_motion_review=review)
    diagnostic = dict(
        regions=regions,
        depth_axis_world=axis.tolist(),
        aggregation="Equal hand weight; equal authored-region weight per active hand",
        limitation="Regional centers are gross placement hypotheses, not matched surface points or verified contact",
    )
    return plan, diagnostic


def export_native(native, geometry, plan, exported):
    """Consume Blender transforms, preserving protected native arrays exactly."""
    before = np.asarray(native["obj_data"]["obj_t"])
    rotations = np.asarray(native["obj_data"]["obj_R"])
    start, end = plan["recovery_start_frame"], plan["recovery_end_frame"]
    unknown = plan.get("object_motion_onset_status") == "unknown"
    prefix_count = plan.get("object_first_motion_frame", start + 1)
    if unknown and (
        prefix_count is not None
        or "stationary_prefix" not in plan
        or plan.get("stationary_prefix") is not None
    ):
        raise ValueError("Unknown onset placement must not invent a stationary prefix")
    expected = (
        before
        + correction_weights(np.arange(len(before)), start, end)[:, None] * plan["delta_world"]
    )
    if (
        np.asarray(exported["obj_t"]).shape != before.shape
        or not np.allclose(exported["obj_t"], expected, atol=2e-5, rtol=0)
        or np.asarray(exported["obj_R"]).shape != rotations.shape
        or not np.allclose(exported["obj_R"], rotations, atol=2e-5, rtol=0)
    ):
        raise ValueError("Blender export differs from the declared translation-only edit")
    result = copy.deepcopy(native)
    result["obj_data"]["obj_t"][:end] = np.asarray(exported["obj_t"])[:end].astype(before.dtype)
    result.setdefault("meta", {}).update(
        stage="stage1_blender_object_placement", status="UNREVIEWED", agentic_placement=plan
    )
    after = {key: value.copy() for key, value in geometry.items()}
    delta = result["obj_data"]["obj_t"] - before
    after["object_vertices"][:end] += delta[:end, None]
    checks = dict(
        human_native_exact=exact(native["human_data"], result["human_data"]),
        object_rotation_exact=exact(rotations, result["obj_data"]["obj_R"]),
        object_scale_exact=exact(native["obj_data"]["obj_scale"], result["obj_data"]["obj_scale"]),
        protected_tail_exact=exact(before[end:], result["obj_data"]["obj_t"][end:]),
        other_geometry_exact=all(
            exact(value, after[key]) for key, value in geometry.items() if key != "object_vertices"
        ),
    )
    if unknown:
        checks["complete_declared_translation_exact"] = bool(
            np.allclose(result["obj_data"]["obj_t"], expected, atol=2e-5, rtol=0)
        )
    else:
        checks["stationary_prefix_exact"] = np.array_equal(
            result["obj_data"]["obj_t"][:prefix_count],
            np.broadcast_to(result["obj_data"]["obj_t"][0], (prefix_count, 3)),
        )
    if not all(checks.values()):
        raise ValueError(f"Placement invariant failed: {checks}")
    return result, after, checks


def temporal_report(translations, fps):
    velocity = np.diff(translations, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    return {
        "max_speed_mps": float(np.linalg.norm(velocity, axis=-1).max(initial=0)),
        "max_acceleration_mps2": float(np.linalg.norm(acceleration, axis=-1).max(initial=0)),
    }


def run_placement(
    stage1_dir, contact_map, contact_frame, transition_start, transition_end, blender, output
):
    start_time = time.monotonic()
    stage1_dir, contact_map = Path(stage1_dir).resolve(), Path(contact_map).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if stage1_dir == output or stage1_dir in output.parents:
        raise ValueError("Placement output must be separate from the Stage 1 input directory")
    paths = {
        "native": stage1_dir / "stage1.pkl",
        "geometry": stage1_dir / "stage1_geometry.npz",
        "local": stage1_dir / "fixed_contact_geometry.npz",
        "binding": stage1_dir / "fixed_contact_binding.json",
        "contact_map": contact_map,
    }
    bindings = {
        key: {"path": str(path), "sha256": file_sha256(path)} for key, path in paths.items()
    }
    native = load_native_pickle(paths["native"])
    with np.load(paths["geometry"], allow_pickle=False) as values:
        geometry = {key: values[key] for key in values.files}
    with np.load(paths["local"], allow_pickle=False) as values:
        local = {key: values[key] for key in values.files}
    plan, diagnostic = prepare_plan(
        native,
        geometry,
        local,
        json.loads(paths["binding"].read_text()),
        json.loads(contact_map.read_text()),
        contact_frame,
        transition_start,
        transition_end,
    )
    output.mkdir(parents=True)
    write_json(output / "edit_plan.json", plan)
    write_json(output / "placement_diagnostic.json", diagnostic)
    write_json(output / "inputs.json", bindings)
    np.savez_compressed(
        output / "edit_inputs.npz",
        human_vertices=geometry["human_vertices"],
        human_faces=geometry["human_faces"],
        object_local_scaled=local["object_vertices"],
        object_faces=local["object_faces"],
        obj_R=native["obj_data"]["obj_R"],
        obj_t=native["obj_data"]["obj_t"],
        camera_R=geometry["camera_R"],
        camera_T=geometry["camera_T"],
        focal=geometry["focal"],
        image_size=geometry["image_size"],
    )
    worker = Path(__file__).with_name("hoi_placement_blender.py")
    command = [
        str(blender),
        "-b",
        "--enable-autoexec",
        "--python",
        str(worker),
        "--",
        "--root",
        str(output),
        "--render-stills",
    ]
    with (output / "blender.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not (output / "blender_export_receipt.json").is_file():
        raise RuntimeError(f"Blender placement failed; inspect {output / 'blender.log'}")
    receipt = json.loads((output / "blender_export_receipt.json").read_text())
    if receipt["input_sha256"] != file_sha256(output / "edit_inputs.npz") or receipt[
        "plan_sha256"
    ] != file_sha256(output / "edit_plan.json"):
        raise ValueError("Blender input/plan binding changed")
    with np.load(output / "blender_reopened_integer.npz", allow_pickle=False) as values:
        candidate, after, checks = export_native(native, geometry, plan, dict(values))
    for key, path in paths.items():
        if file_sha256(path) != bindings[key]["sha256"]:
            raise ValueError(f"Input changed during Blender edit: {path}")
    joblib.dump(candidate, output / "stage1_object_refined.pkl")
    np.savez_compressed(output / "refined_geometry.npz", **after)
    temporal = {
        key: temporal_report(value, plan["fps"])
        for key, value in [
            ("before", native["obj_data"]["obj_t"]),
            ("after", candidate["obj_data"]["obj_t"]),
        ]
    }
    report = dict(
        status="PLACEMENT_COMPLETE_UNREVIEWED",
        checks=checks,
        seconds=time.monotonic() - start_time,
        temporal=temporal,
        frame_count=len(native["obj_data"]["obj_t"]),
        fps=plan["fps"],
        grasp_accepted=False,
        physical_validation=False,
        worker_sha256=file_sha256(worker),
        outputs={
            name: file_sha256(output / name)
            for name in [
                "stage1_object_refined.pkl",
                "refined_geometry.npz",
                "stage1_object_placement.blend",
            ]
        },
        images=receipt.get("blender_stills", []),
        limitation="Translation-only gross alignment. Actual trajectory and surface contact still require review; a smooth correction does not guarantee smooth replay.",
    )
    write_json(output / "completion.json", report)
    return report


def _unchanged_inputs(stage1_dir, contact_map, review):
    """Validate a reviewed identity placement without fabricating contact keys."""
    from grail.agentic_kinematic.hoi_stages import bind

    first = Path(stage1_dir).resolve(strict=True)
    workspace = first.parent
    contact_map, review = Path(contact_map).resolve(strict=True), Path(review).resolve(strict=True)
    paths = dict(
        stage1_native=first / "stage1.pkl",
        stage1_geometry=first / "stage1_geometry.npz",
        stage1_completion=first / "completion.json",
        stage1_inputs=first / "inputs.json",
        fixed_contact_binding=first / "fixed_contact_binding.json",
        fixed_contact_geometry=first / "fixed_contact_geometry.npz",
        contact_map=contact_map,
        review=review,
    )
    if any(not path.resolve(strict=True).is_relative_to(workspace) for path in paths.values()):
        raise ValueError("Unchanged placement inputs must stay inside the case workspace")
    inputs = {key: bind(path) for key, path in paths.items()}
    native = json.loads(paths["stage1_completion"].read_text())
    checks = native.get("checks")
    if (
        native.get("stage") != "stage1"
        or native.get("status") != "OPTIMIZATION_COMPLETE_REVIEW_PENDING"
        or type(native.get("iterations")) is not int
        or native["iterations"] < 1
        or not isinstance(checks, dict)
        or not checks
        or not all(v is True for v in checks.values())
        or native.get("output") != inputs["stage1_native"]
        or native.get("geometry") != inputs["stage1_geometry"]
    ):
        raise ValueError(
            "Unchanged placement requires checked, hash-bound Stage 1 native and geometry"
        )
    binding = json.loads(paths["fixed_contact_binding"].read_text())
    spec = json.loads(contact_map.read_text())
    with np.load(paths["fixed_contact_geometry"], allow_pickle=False) as geometry:
        PhaseContactMap.from_spec(
            spec,
            binding=binding,
            human_faces=geometry["human_faces"],
            object_vertices=geometry["object_vertices"],
            object_faces=geometry["object_faces"],
        )
    if any(
        interval["state"] == "contact" for phases in spec["hands"].values() for interval in phases
    ):
        raise ValueError(
            "Unchanged no-evidenced-contact placement cannot discard active contact intervals"
        )
    decision = json.loads(review.read_text())
    if (
        decision.get("schema") != "grail.unchanged_placement_review.v1"
        or decision.get("decision") != "preserve_stage1_no_evidenced_contact"
        or not isinstance(decision.get("reviewed_by"), str)
        or not decision["reviewed_by"].strip()
        or not isinstance(decision.get("observation"), str)
        or not decision["observation"].strip()
        or decision.get("source_video_sha256") != binding["source_video_sha256"]
        or decision.get("stage1_native_sha256") != inputs["stage1_native"]["sha256"]
        or decision.get("stage1_geometry_sha256") != inputs["stage1_geometry"]["sha256"]
        or decision.get("contact_map_sha256") != inputs["contact_map"]["sha256"]
    ):
        raise ValueError("Unchanged placement requires a source/model-bound review decision")
    count = binding["frame_count"]
    frames = decision.get("inspected_frames")
    if (
        not isinstance(frames, list)
        or len(set(frames)) != len(frames)
        or any(type(frame) is not int or not 0 <= frame < count for frame in frames)
        or not {0, count - 1} <= set(frames)
    ):
        raise ValueError(
            "Unchanged placement review must inspect valid source frames including both ends"
        )
    for role in ("source_evidence", "oblique_evidence"):
        evidence = decision.get(role)
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Unchanged placement requires source and oblique review evidence")
        inputs[role] = []
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
                raise ValueError("Review evidence requires path/SHA256/byte bindings")
            path = (review.parent / item["path"]).resolve(strict=True)
            if not path.is_relative_to(workspace) or not path.is_file():
                raise ValueError("Review evidence escapes the case workspace")
            actual = bind(path)
            if (
                actual["sha256"] != item["sha256"]
                or actual["bytes"] != item["bytes"]
                or not actual["bytes"]
            ):
                raise ValueError("Review evidence changed or is empty")
            inputs[role].append(actual)
    return inputs, binding, decision


def inspect_contact_placement(
    stage1_dir, placement_dir, contact_map, contact_frame, transition_start, transition_end
):
    """Recheck an existing Blender placement without rerunning or relabeling it."""
    from grail.agentic_kinematic.hoi_stages import bind

    first, output = Path(stage1_dir).resolve(strict=True), Path(placement_dir).resolve(strict=True)
    paths = dict(
        native=first / "stage1.pkl",
        geometry=first / "stage1_geometry.npz",
        local=first / "fixed_contact_geometry.npz",
        binding=first / "fixed_contact_binding.json",
        contact_map=Path(contact_map).resolve(strict=True),
    )
    expected_inputs = {
        key: {"path": str(path), "sha256": file_sha256(path)} for key, path in paths.items()
    }
    if json.loads((output / "inputs.json").read_text()) != expected_inputs:
        raise ValueError("Placement source inputs changed")
    stage1 = json.loads((first / "completion.json").read_text())
    stage1_checks = stage1.get("checks")
    if (
        stage1.get("status") != "OPTIMIZATION_COMPLETE_REVIEW_PENDING"
        or stage1.get("stage") != "stage1"
        or not isinstance(stage1_checks, dict)
        or not stage1_checks
        or not all(value is True for value in stage1_checks.values())
        or stage1.get("output") != bind(paths["native"])
        or stage1.get("geometry") != bind(paths["geometry"])
    ):
        raise ValueError("Placement requires checked, unchanged Stage 1 native and geometry")
    native = load_native_pickle(paths["native"])
    with np.load(paths["geometry"], allow_pickle=False) as values:
        geometry = dict(values)
    with np.load(paths["local"], allow_pickle=False) as values:
        local = dict(values)
    plan, _ = prepare_plan(
        native,
        geometry,
        local,
        json.loads(paths["binding"].read_text()),
        json.loads(paths["contact_map"].read_text()),
        contact_frame,
        transition_start,
        transition_end,
    )
    if json.loads((output / "edit_plan.json").read_text()) != plan:
        raise ValueError("Placement plan changed from its source-authored contact/transition")
    blender = json.loads((output / "blender_export_receipt.json").read_text())
    if blender.get("input_sha256") != file_sha256(output / "edit_inputs.npz") or blender.get(
        "plan_sha256"
    ) != file_sha256(output / "edit_plan.json"):
        raise ValueError("Placement Blender receipt input/plan binding changed")
    expected_blender = dict(
        human_vertices=geometry["human_vertices"],
        human_faces=geometry["human_faces"],
        object_local_scaled=local["object_vertices"],
        object_faces=local["object_faces"],
        obj_R=native["obj_data"]["obj_R"],
        obj_t=native["obj_data"]["obj_t"],
        camera_R=geometry["camera_R"],
        camera_T=geometry["camera_T"],
        focal=geometry["focal"],
        image_size=geometry["image_size"],
    )
    with np.load(output / "edit_inputs.npz", allow_pickle=False) as values:
        if not exact(dict(values), expected_blender):
            raise ValueError("Placement Blender inputs differ from original native geometry")
    with np.load(output / "blender_reopened_integer.npz", allow_pickle=False) as values:
        candidate, after, checks = export_native(native, geometry, plan, dict(values))
    if not exact(load_native_pickle(output / "stage1_object_refined.pkl"), candidate):
        raise ValueError("Placement native candidate differs from verified Blender export")
    with np.load(output / "refined_geometry.npz", allow_pickle=False) as values:
        if not exact(dict(values), after):
            raise ValueError("Placement geometry differs from verified native export")
    receipt = json.loads((output / "completion.json").read_text())
    outputs = {
        name: file_sha256(output / name)
        for name in (
            "stage1_object_refined.pkl",
            "refined_geometry.npz",
            "stage1_object_placement.blend",
        )
    }
    if (
        receipt.get("status") != "PLACEMENT_COMPLETE_UNREVIEWED"
        or receipt.get("checks") != checks
        or receipt.get("outputs") != outputs
    ):
        raise ValueError("Placement completion receipt does not bind the verified candidate")
    return receipt


def inspect_unchanged_placement(stage1_dir, placement_dir, contact_map=None):
    """Independently verify the identity-parent route before Stage 2/import."""
    from grail.agentic_kinematic.hoi_stages import bind

    output = Path(placement_dir).resolve(strict=True)
    recorded = json.loads((output / "inputs.json").read_text())
    inputs, binding, decision = _unchanged_inputs(
        stage1_dir, recorded["contact_map"]["path"], recorded["review"]["path"]
    )
    if inputs != recorded:
        raise ValueError("Unchanged placement inputs or review evidence changed")
    if contact_map is not None and bind(contact_map)["sha256"] != inputs["contact_map"]["sha256"]:
        raise ValueError("Stage 2 contact map differs from the unchanged placement review")
    native, geometry = bind(output / "stage1_object_refined.pkl"), bind(
        output / "refined_geometry.npz"
    )
    if (
        native["sha256"] != inputs["stage1_native"]["sha256"]
        or native["bytes"] != inputs["stage1_native"]["bytes"]
        or geometry["sha256"] != inputs["stage1_geometry"]["sha256"]
        or geometry["bytes"] != inputs["stage1_geometry"]["bytes"]
    ):
        raise ValueError(
            "Reviewed unchanged placement must preserve native and geometry byte-for-byte"
        )
    receipt = json.loads((output / "completion.json").read_text())
    expected = dict(
        native_bytes_exact=True,
        geometry_bytes_exact=True,
        no_evidenced_contact=True,
        source_and_oblique_review_bound=True,
    )
    if (
        receipt.get("status") != "PLACEMENT_REVIEWED_UNCHANGED"
        or receipt.get("mode") != "reviewed_unchanged"
        or receipt.get("checks") != expected
        or receipt.get("inputs_sha256") != file_sha256(output / "inputs.json")
        or receipt.get("outputs")
        != {
            "stage1_object_refined.pkl": native["sha256"],
            "refined_geometry.npz": geometry["sha256"],
        }
    ):
        raise ValueError("Unchanged placement receipt does not bind the verified result")
    return receipt


def run_unchanged_placement(stage1_dir, contact_map, review, output):
    """Preserve reviewed Stage 1 when the source supports no contact attraction."""
    inputs, binding, decision = _unchanged_inputs(stage1_dir, contact_map, review)
    output, first = Path(output).resolve(), Path(stage1_dir).resolve()
    if output.exists() or output == first or first in output.parents or output in first.parents:
        raise ValueError("Use a new unchanged placement directory outside the Stage 1 input")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "inputs.json", inputs)
    shutil.copyfile(inputs["stage1_native"]["path"], output / "stage1_object_refined.pkl")
    shutil.copyfile(inputs["stage1_geometry"]["path"], output / "refined_geometry.npz")
    receipt = dict(
        status="PLACEMENT_REVIEWED_UNCHANGED",
        mode="reviewed_unchanged",
        frame_count=binding["frame_count"],
        fps=binding["fps"],
        contact_frame=None,
        translation_applied=False,
        checks=dict(
            native_bytes_exact=True,
            geometry_bytes_exact=True,
            no_evidenced_contact=True,
            source_and_oblique_review_bound=True,
        ),
        inputs_sha256=file_sha256(output / "inputs.json"),
        outputs={
            name: file_sha256(output / name)
            for name in ("stage1_object_refined.pkl", "refined_geometry.npz")
        },
        grasp_accepted=False,
        motion_accepted=False,
        physical_validation=False,
        observation=decision["observation"],
        limitation="No source-supported contact attraction; preserve no_contact and unknown. "
        "Identity placement is not proof of correct interaction or tracking.",
    )
    write_json(output / "completion.json", receipt)
    return inspect_unchanged_placement(first, output, contact_map)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--contact-map", required=True)
    parser.add_argument(
        "--mode", choices=["contact_alignment", "reviewed_unchanged"], default="contact_alignment"
    )
    parser.add_argument("--review")
    parser.add_argument("--contact-frame", type=int)
    parser.add_argument("--transition-start", type=int)
    parser.add_argument("--transition-end", type=int)
    parser.add_argument("--blender")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.mode == "reviewed_unchanged":
        if not args.review or any(
            getattr(args, key) is not None
            for key in ("contact_frame", "transition_start", "transition_end", "blender")
        ):
            parser.error(
                "reviewed_unchanged requires --review and forbids edit/transition parameters"
            )
        result = run_unchanged_placement(
            args.stage1_dir, args.contact_map, args.review, args.output
        )
    else:
        if args.review is not None or any(
            getattr(args, key) is None
            for key in ("contact_frame", "transition_start", "transition_end", "blender")
        ):
            parser.error("contact_alignment requires contact/transition/blender parameters")
        values = vars(args).copy()
        values.pop("mode")
        values.pop("review")
        result = run_placement(**values)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
