"""One explicit native edit, Blender refresh, export and review round."""

from __future__ import annotations

import os
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np

from .contracts import (
    binding,
    load_npz,
    read_case,
    read_json,
    save_npz,
    validate_scene,
    verify_binding,
    write_json,
)
from .interpolation import apply_edits, sample_motion, validate_plan
from .native import DERIVED_FIELDS, export_native, load_native, validate_motion

REPO = Path(__file__).resolve().parents[3]
EMPTY_PLAN = {"schema_version": 1, "windows": []}


def _without_caches(motion):
    return {key: value for key, value in motion.items() if key not in DERIVED_FIELDS}


def _check_parent(base, parent):
    """A persisted scene cannot redefine frozen fields or model identity."""
    base, parent = _without_caches(base), _without_caches(parent)
    if base.keys() != parent.keys():
        raise ValueError("Saved scene parent has different canonical fields")
    for key in base:
        a, b = np.asarray(base[key]), np.asarray(parent[key])
        if a.dtype != b.dtype or a.shape != b.shape:
            raise ValueError(f"Saved scene changed field shape/dtype: {key}")
        if key != "poses" and a.tobytes() != b.tobytes():
            raise ValueError(f"Saved scene changed frozen field: {key}")
    meta = validate_motion(base)
    frozen = sorted(set(range(meta["joint_count"])) - set(meta["editable_body_joints"]))
    a = base["poses"].reshape(meta["frame_count"], -1, 3)[:, frozen]
    b = parent["poses"].reshape(meta["frame_count"], -1, 3)[:, frozen]
    if a.tobytes() != b.tobytes():
        raise ValueError("Saved scene changed frozen root, finger or face rotations")


def _model_bindings(model):
    """Bind consumed model assets and native implementations, excluding caches."""
    paths = set()
    for key, value in model.items():
        if key.endswith(("_path", "_dir")):
            path = Path(value)
            if path.is_file():
                paths.add(path)
                if key == "soma_identity_params_path":
                    paths.add(path.with_suffix(".json"))
            elif path.is_dir():
                paths.update(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix in {".npz", ".npy", ".pkl", ".obj", ".json"}
                )
    if model["model_type"] == "soma03":
        from grail.models.soma_x_runtime import official_runtime_inputs

        if model.get("soma_runtime", "gem_x") != "gem_x" or "gep_gem_bundle" in model:
            raise ValueError(
                "Legacy SOMA cases require their archived code; migrate model bindings explicitly"
            )
        native_paths, _ = official_runtime_inputs(model["gem_x_soma_assets"])
        paths.update(native_paths)
    paths.update((REPO / "grail/models").glob("*.py"))
    return [binding(path) for path in sorted(paths)]


def _execute(command, log, timeout):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    for key in ("NVIDIA_API_KEY", "NVIDIA_API_KEY_FILE", "OPENAI_API_KEY"):
        env.pop(key, None)
    with Path(log).open("xb") as stream:
        try:
            subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Refinement worker failed; see {log}") from error


def _geometry(motion_path, model_path, frames_path, out, case):
    result = out.with_suffix(".npz")
    _execute(
        [
            case["runtime"]["native_python"],
            "-m",
            "grail.agentic_kinematic.hoi_refinement.native_worker",
            "--motion",
            str(motion_path),
            "--model-config",
            str(model_path),
            "--frames",
            str(frames_path),
            "--output",
            str(result),
        ],
        out.with_suffix(".log"),
        case["runtime"]["timeout_s"],
    )
    data = load_npz(result)
    expected = np.asarray(read_json(frames_path), dtype=float)
    if not np.array_equal(data["frames"], expected):
        raise ValueError("Native worker returned a different frame batch")
    if not np.isfinite(data["vertices"]).all() or not np.isfinite(data["joints"]).all():
        raise ValueError("Native worker returned nonfinite geometry")
    return result, data


def edit_plan(plan, operations):
    """Set/delete explicit delta-rotation keys; interpolation validates scope."""
    plan = deepcopy(plan)
    if not isinstance(operations, list):
        raise ValueError("Operations must be a JSON array")
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("op") not in {"set_key", "delete_key"}:
            raise ValueError("Use explicit set_key/delete_key operations")
        required = {"op", "window", "frame"} | (
            {"delta_rotvec"} if operation["op"] == "set_key" else set()
        )
        if set(operation) != required:
            raise ValueError("Unexpected operation fields")
        index = operation["window"]
        if type(index) is not int or not 0 <= index < len(plan["windows"]):
            raise ValueError("Operation window is not in the declared plan")
        window, frame = plan["windows"][index], operation["frame"]
        if (
            isinstance(frame, bool)
            or not isinstance(frame, (int, float))
            or not window["start"] < frame < window["end"]
        ):
            raise ValueError("Protected window endpoints cannot be edited")
        keys = [key for key in window["keys"] if key["frame"] != frame]
        if operation["op"] == "delete_key" and len(keys) == len(window["keys"]):
            raise ValueError("Cannot delete a missing key")
        if operation["op"] == "set_key":
            keys.append({"frame": frame, "delta_rotvec": operation["delta_rotvec"]})
        window["keys"] = sorted(keys, key=lambda key: key["frame"])
    return plan


def _replay_comparison(base, candidate, plan, frames):
    """Compare authored rotation corrections with exported source-rate replay."""
    from .interpolation import rotvec_to_quaternion

    authored = sample_motion(base, plan, frames)["poses"].reshape(len(frames), -1, 3)
    replay = sample_motion(candidate, EMPTY_PLAN, frames)["poses"].reshape(len(frames), -1, 3)
    left, right = rotvec_to_quaternion(authored), rotvec_to_quaternion(replay)
    angles = 2 * np.arccos(np.clip(np.abs(np.sum(left * right, axis=-1)), 0, 1))
    return {
        "frames": list(frames),
        "max_rotation_difference_rad": float(angles.max(initial=0)),
        "preview_semantics": "Native source-rate export replay; quaternion interpolation",
        "scope": "Samples only; no continuous-time guarantee",
    }


def run(config_path, output, *, input_blend=None, operations_path=None, repair=False):
    case_record = binding(config_path)
    case = read_case(config_path)
    if repair and (not case.get("repair") or not case.get("source_video")):
        raise ValueError("Local repair requires repair configuration and the actual source_video")
    if repair and operations_path:
        raise ValueError("Save authored edits in a run round before requesting local repair")
    base = load_native(case["input_native"], case["model"]["model_type"])
    meta = validate_motion(base)
    frames = case["review_frames"]
    if frames[-1] > meta["frame_count"] - 1:
        raise ValueError("Review frame lies outside the native source timeline")
    scene = validate_scene(load_npz(case["scene"]))
    for field in ("obj_R", "obj_t"):
        if field in scene and not np.array_equal(scene[field], base[field]):
            raise ValueError(f"Scene changes frozen object motion: {field}")
        scene[field] = base[field].copy()
    if "object_scale" in scene and not np.array_equal(scene["object_scale"], base["obj_scale"]):
        raise ValueError("Scene changes original object scale")
    scene["object_scale"] = base["obj_scale"].copy()
    records = {name: binding(case[name]) for name in ("input_native", "scene", "plan")}
    if case.get("source_video"):
        records["source_video"] = binding(case["source_video"])
    if case.get("contact_review"):
        records["contact_review"] = binding(case["contact_review"])
    if repair:
        records["contact_map"] = binding(case["repair"]["contact_map"])
    records["source_images"] = [binding(p) for p in case["source_images"]]
    model_records = _model_bindings(case["model"])
    code_paths = set(Path(__file__).parent.glob("*.py"))
    code_paths.update(
        REPO / relative
        for relative in (
            "grail/core/contact.py",
            "grail/optimization/hoi_contact_normals.py",
            "grail/optimization/contact_review.py",
            "grail/optimization/fixed_contact.py",
            "grail/optimization/contact_maps.py",
            "grail/optimization/phase_contact_map_loss.py",
        )
    )
    code_records = [binding(p) for p in sorted(code_paths)]
    bound = {
        "source_native": records["input_native"],
        "scene": records["scene"],
        "model": model_records,
        "model_config": case["model"],
    }
    input_record = binding(input_blend) if input_blend else None
    operation_record = binding(operations_path) if operations_path else None
    evidence = deepcopy(case["evidence"])
    evidence.setdefault("sequence", case["name"])
    evidence["scene_sha256"] = records["scene"]["sha256"]
    if "source_video" in records:
        source_hash = records["source_video"]["sha256"]
        if evidence.get("source_video_sha256", source_hash) != source_hash:
            raise ValueError("Source evidence hash differs from the source video")
        evidence["source_video_sha256"] = source_hash
    elif case.get("contact_review") or any(
        evidence.get(key) is not None
        for key in ("source_video_sha256", "fixed_depth", "source_keypoints")
    ):
        raise ValueError("Contact/source/depth evidence requires an actual bound source_video")
    if evidence.get("source_keypoints") is not None:
        raise ValueError(
            "Observed source_keypoints review is not implemented; use source views and projection displacement"
        )
    fixed_records = []
    for key in ("fixed_depth", "source_keypoints"):
        if isinstance(evidence.get(key), dict) and evidence[key].get("path"):
            fixed_records.append(binding(evidence[key]["path"]))
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 1,
        "status": "RUNNING",
        "motion_status": "UNREVIEWED",
        "selected": False,
        "api_calls": 0,
        "configuration": case_record,
        "case": case,
        "inputs": records,
        "bindings": bound,
        "runtime_code": code_records,
        "input_blend": input_record,
        "operations": operation_record,
    }
    state_path = out / "iteration.json"
    write_json(state_path, receipt)
    try:
        from .blender_bridge import run_blender

        runtime = case["runtime"]
        parent = deepcopy(base)
        plan = read_json(case["plan"])
        if input_record:
            inspection = run_blender(
                {
                    "schema_version": 1,
                    "mode": "inspect",
                    "input_blend": input_record,
                    "bindings": bound,
                },
                out / "inspect",
                runtime["blender"],
                runtime["timeout_s"],
            )
            if inspection.get("bindings") != bound:
                raise ValueError(
                    "Saved scene belongs to a different native model, source or camera"
                )
            plan = inspection["plan"]
            parent = load_native(inspection["parent_motion"]["path"], case["model"]["model_type"])
            _check_parent(base, parent)
            receipt["parent_checkpoint"] = input_record
        if operation_record:
            plan = edit_plan(plan, read_json(operation_record["path"]))
        validate_plan(base, plan)
        if repair:
            # A saved scene's edited keys are the working seed. Bounds and all
            # cumulative corrections still refer to the immutable case input.
            parent = apply_edits(base, plan)
            save_npz(out / "original.npz", _without_caches(base))
            write_json(out / "seed_plan.json", plan)
            write_json(out / "repair_model.json", case["model"])
            save_npz(out / "repair_scene.npz", scene)
            repair_evidence = dict(evidence, source_video=case["source_video"])
            repair_request = {
                "base_motion": str(out / "original.npz"),
                "seed_plan": str(out / "seed_plan.json"),
                "scene": str(out / "repair_scene.npz"),
                "model": str(out / "repair_model.json"),
                "repair": case["repair"],
                "evidence": repair_evidence,
                "output_dir": str(out / "repair"),
            }
            write_json(out / "repair_request.json", repair_request)
            _execute(
                [
                    runtime["native_python"],
                    "-m",
                    "grail.agentic_kinematic.hoi_refinement.repair_worker",
                    "--request",
                    str(out / "repair_request.json"),
                ],
                out / "repair.log",
                runtime["timeout_s"],
            )
            plan = read_json(out / "repair" / "proposal_plan.json")
            validate_plan(base, plan)
            receipt["repair"] = binding(out / "repair" / "report.json")
            receipt["seed_plan"] = binding(out / "seed_plan.json")
            # Integer replay and a three-frame halo support derivative checks.
            # Half-frame checks inspect the actual exported replay, not merely
            # the continuous authored correction curve.
            intervals = set()
            for window in plan["windows"]:
                start = max(0, window["start"] - 3)
                end = min(meta["frame_count"] - 1, window["end"] + 3)
                intervals.update(np.arange(start, end + 0.25, 0.5).tolist())
            frames = sorted(set(frames) | intervals)
            evidence["collision_frames"] = sorted(
                set(evidence.get("collision_frames", [])) | intervals
            )
        frames = sorted(
            set(frames) | {key["frame"] for window in plan["windows"] for key in window["keys"]}
        )
        candidate = apply_edits(base, plan)
        save_npz(out / "parent.npz", _without_caches(parent))
        save_npz(out / "candidate.npz", _without_caches(candidate))
        write_json(out / "plan.json", plan)
        write_json(out / "model.json", case["model"])
        # GPU skinning reductions can differ by several micrometres between
        # processes. Compare all three review meshes on the same CPU evaluator.
        geometry_model = out / "geometry_model.json"
        write_json(geometry_model, {**case["model"], "device": "cpu"})
        write_json(out / "frames.json", frames)
        before_path, before = _geometry(
            out / "parent.npz",
            geometry_model,
            out / "frames.json",
            out / "before_geometry",
            case,
        )
        after_path, after = _geometry(
            out / "candidate.npz",
            geometry_model,
            out / "frames.json",
            out / "after_geometry",
            case,
        )
        if not np.array_equal(before["faces"], after["faces"]):
            raise ValueError("Native model topology changed after editing")
        # Re-open the actual exported checkpoint, then independently reconstruct it.
        suffix = (
            ".pkl"
            if Path(case["input_native"]).suffix in (".pkl", ".pickle", ".joblib")
            else ".npz"
        )
        native_path = out / ("candidate_native" + suffix)
        receipt["native_export"] = export_native(
            case["input_native"], candidate, native_path, case["model"]["model_type"]
        )
        reloaded = load_native(native_path, case["model"]["model_type"])
        for key in _without_caches(candidate):
            if key not in reloaded or not np.array_equal(candidate[key], reloaded[key]):
                raise ValueError(f"Native export changed canonical field: {key}")
        save_npz(out / "reopened.npz", reloaded)
        reopened_path, reopened = _geometry(
            out / "reopened.npz",
            geometry_model,
            out / "frames.json",
            out / "reopened_geometry",
            case,
        )
        if not np.array_equal(after["faces"], reopened["faces"]):
            raise ValueError("Reopened native topology differs from the preview")
        max_vertex = float(np.max(np.abs(after["vertices"] - reopened["vertices"]), initial=0))
        max_joint = float(np.max(np.abs(after["joints"] - reopened["joints"]), initial=0))
        receipt["roundtrip"] = {
            "geometry_device": "cpu",
            "canonical_arrays_equal": True,
            "max_vertex_difference_m": max_vertex,
            "max_joint_difference_m": max_joint,
        }
        if max(max_vertex, max_joint) > 2e-6:
            raise ValueError("Reopened native geometry differs from the preview")
        sampled = sample_motion(candidate, EMPTY_PLAN, frames)
        scene["observation_obj_R"] = sampled["obj_R"]
        scene["observation_obj_t"] = sampled["obj_t"]
        save_npz(out / "scene.npz", scene)
        review_request = {
            "parent": str(out / "parent.npz"),
            "candidate": str(out / "candidate.npz"),
            "before": str(before_path),
            "after": str(after_path),
            "scene": str(out / "scene.npz"),
            "scene_sha256": records["scene"]["sha256"],
            "contact_review": case.get("contact_review"),
            "evidence": evidence,
            "output": str(out / "review.json"),
        }
        write_json(out / "review_request.json", review_request)
        _execute(
            [
                runtime["native_python"],
                "-m",
                "grail.agentic_kinematic.hoi_refinement.review_worker",
                "--request",
                str(out / "review_request.json"),
            ],
            out / "review.log",
            runtime["timeout_s"],
        )
        report = read_json(out / "review.json")
        if repair:
            from .iteration import review_iteration

            iteration_review = review_iteration(
                base, parent, candidate, before, after, plan, case.get("iteration")
            )
            report["checks"]["iteration"] = iteration_review
            proposal_report = read_json(out / "repair" / "report.json")
            report["checks"]["dense_contact_map_objective"] = {
                "status": "PERFORMED",
                "before": proposal_report["terms_before"]["contact"],
                "after": proposal_report["terms_after"]["contact"],
                "contact_map": records["contact_map"],
                "limitations": "Regional optimization cost from the bounded proposal, not "
                "measured material-pair gaps or acceptance of the grasp.",
            }
            write_json(out / "review.json", report)
            write_json(out / "iteration_review.json", iteration_review)
            receipt["iteration_review"] = binding(out / "iteration_review.json")
        newton_reports = _newton(case, scene, after, out) if case.get("newton") else []
        request = {
            "schema_version": 1,
            "mode": "build",
            "bindings": bound,
            "parent_motion": str(out / "parent.npz"),
            "candidate_motion": str(out / "candidate.npz"),
            "before_geometry": str(before_path),
            "after_geometry": str(after_path),
            "scene": str(out / "scene.npz"),
            "plan": plan,
            "fps": meta["fps"],
            "model_type": meta["model_type"],
            "camera_convention": "opencv_camera_to_world",
            "observation_frames": case["observation_frames"],
            "render": case.get("render", True),
            "review": report,
            "newton_reports": newton_reports,
        }
        request["contact_overlay"] = _contact_overlay(report, before, after, scene)
        if input_record:
            request["input_blend"] = input_record
        visual = run_blender(request, out / "blender", runtime["blender"], runtime["timeout_s"])
        receipt.update(
            status="EXPORTED",
            checkpoint=visual["checkpoint"],
            native=binding(native_path),
            candidate=binding(out / "candidate.npz"),
            plan=binding(out / "plan.json"),
            review=binding(out / "review.json"),
            blender=visual,
            newton_reports=newton_reports,
            authored_vs_replay=_replay_comparison(base, candidate, plan, frames),
        )
    except BaseException as error:
        receipt["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        checks = [
            case_record,
            *model_records,
            *code_records,
            *fixed_records,
            *records["source_images"],
            *[v for v in records.values() if isinstance(v, dict)],
            *([input_record] if input_record else []),
            *([operation_record] if operation_record else []),
        ]
        try:
            for record in checks:
                verify_binding(record)
        except (OSError, ValueError) as error:
            receipt["status"] = "INPUT_CHANGED"
            receipt["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            write_json(state_path, receipt)
    return {
        "iteration": str(state_path),
        "status": receipt["status"],
        "checkpoint": receipt["checkpoint"],
        "native": receipt["native"]["path"],
        "motion_status": "UNREVIEWED",
        "selected": False,
        "api_calls": 0,
    }


def _contact_overlay(report, before, after, scene):
    """Show reviewed fixed material pairs on the actual native evaluated surfaces."""
    samples = []
    pairs = report["checks"]["fixed_material_contacts"].get("pairs", [])
    for pair in pairs:
        vertex_id = pair["human_vertex_id"]
        object_faces = np.asarray(scene["object_faces"])
        object_vertex = pair.get("object_vertex_id")
        if object_vertex is not None:
            object_faces = object_faces[np.any(object_faces == object_vertex, axis=1)]
        else:
            object_faces = object_faces[[pair["object_face_id"]]]
        for phase, geometry in (("before", before), ("after", after)):
            human_faces = geometry["faces"]
            adjacent_faces = human_faces[np.any(human_faces == vertex_id, axis=1)]
            for index, frame in enumerate(geometry["frames"]):
                if frame not in pair["evaluated_active_frames"]:
                    continue
                triangles = (
                    scene["object_vertices"][object_faces] * scene["object_scale"]
                ) @ geometry["obj_R"][index].T + geometry["obj_t"][index]
                if object_vertex is not None:
                    object_point = (
                        scene["object_vertices"][object_vertex] * scene["object_scale"]
                    ) @ geometry["obj_R"][index].T + geometry["obj_t"][index]
                else:
                    object_point = np.asarray(pair["object_barycentric"]) @ triangles[0]
                human = geometry["vertices"][index]
                adjacent = human[adjacent_faces]
                human_normal = np.cross(
                    adjacent[:, 1] - adjacent[:, 0], adjacent[:, 2] - adjacent[:, 0]
                ).sum(axis=0)
                object_normal = np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ).sum(axis=0)
                sample = {
                    "source_frame": float(frame),
                    "phase": phase,
                    "label": str(pair["id"]),
                    "human_point": human[vertex_id].tolist(),
                    "object_point": object_point.tolist(),
                }
                for name, normal in (
                    ("human_normal", human_normal),
                    ("object_normal", object_normal),
                ):
                    length = np.linalg.norm(normal)
                    if length > 1e-12:
                        sample[name] = (normal / length).tolist()
                samples.append(sample)
    return {
        "samples": samples,
        "semantics": "Fixed material points; original winding normals, not solid occupancy",
    }


def _newton(case, scene, geometry, out):
    from .newton import run_query

    requested = case["newton"].get("frames", case["observation_frames"])
    reports = []
    for frame in requested:
        matches = np.flatnonzero(geometry["frames"] == frame)
        if len(matches) != 1:
            raise ValueError("Newton frame must belong to the native evaluated batch")
        index = int(matches[0])
        matrix = np.eye(4)
        matrix[:3, :3] = scene["observation_obj_R"][index]
        matrix[:3, 3] = scene["observation_obj_t"][index]
        snapshot = {
            "schema_version": 1,
            "source_frame": frame,
            "units": "m",
            "meshes": [
                {
                    "name": "human",
                    "vertices": geometry["vertices"][index].tolist(),
                    "triangles": geometry["faces"].tolist(),
                    "matrix_world": np.eye(4).tolist(),
                },
                {
                    "name": "object",
                    "vertices": (scene["object_vertices"] * scene["object_scale"]).tolist(),
                    "triangles": scene["object_faces"].tolist(),
                    "matrix_world": matrix.tolist(),
                },
            ],
            "pairs": [["human", "object"]],
        }
        report = run_query(snapshot, case["newton"], out / f"newton_{index:04d}")
        reports.append(report)
    return reports
