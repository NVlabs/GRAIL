"""One numerical recipe; source-video annotations supply hand phase semantics.

This is the routine hand-only entrypoint. Low-level research solvers remain
available for explicit ablations. Outputs are unreviewed targets, never accepted
grasps or mounted robot motion.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HOLD = {"hold", "grasp", "stable_hold"}
PATH = {"support", "slide", "sliding"}
ROLES = HOLD | PATH | {"unknown", "no_contact", "approach", "contact_acquisition", "release", "free"}
PROFILE_ID = "source_phase_shared_hand_v004"


def shared_profile():
    """Return the fixed base solve plus the tested SG/contact/smooth refinement."""
    from .hand_shared_seed_projection import projection_policy
    return dict(
        profile_id=PROFILE_ID, topo_steps=200, pad_steps=0, paired_steps=300,
        samples_per_geom=128, transition_seconds=.25,
        paired_learning_rate=.002, translation_limit_m=.15, rotation_limit_rad=1.5,
        clearance_m=.001, cross_hand_clearance_m=.002,
        global_worst_object_weight=400., global_worst_cross_hand_weight=400.,
        paired_loss_weights=dict(object_collision=1., self_collision=1., cross_hand=1., scene_collision=1., contact=4., fixed_anchor=4.),
        source_prior=dict(position=.02, rotation=.0001, finger=.00002),
        temporal_weights=dict(wrist_acceleration=2e-5, wrist_jerk=2e-8,
            angular_acceleration=5e-7, angular_jerk=5e-10,
            finger_acceleration=2e-7, finger_jerk=2e-10,
            hold_translation_velocity=.01, hold_angular_velocity=.0001, hold_finger_velocity=.00002),
        collision_collocation_fractions=[.5], collision_collocation_weight=1.,
        object_relative_temporal_roles=["hold", "grasp", "stable_hold", "support", "slide", "sliding"],
        multistart=dict(count_per_frame=16, seed=17, translation_radius_m=.015,
                       rotation_radius_rad=.2, finger_radius_rad=.3),
        topo_objective=dict(graph_weight=500., direction_weight=.01, graph_decay=30.,
                            object_sample_count=50, fallback_neighbors=4, node_set="all_mapped"),
        object_sampling_rule="min(configured_count, original_face_count)",
        bank_learning_rate=.008, bank_self_clearance_weight=1., bank_scene_clearance_weight=1.,
        contact_patch=dict(points_per_member=4, max_gap_m=.005, min_spacing_m=.003),
        path_alternatives=8, normal_alignment=False, material_anchors=False,
        boundary_blend_policy="phase_clipped", protected_frames=[0],
        allowed_blend_roles=["no_contact", "approach", "contact_acquisition", "release", "free"],
        candidate_selection="coverage_missing_penetration_source_lexicographic_unreviewed",
        shared_seed_projection=projection_policy(),
        shared_feasibility_backtracking=dict(kind="backtracking", max_backtracks=12, shrink_factor=.5),
        shared_base_control_projection="original per-frame boxes before full-decoder backtracking; no shared-output clamp",
        bank_chunking=dict(max_candidates_per_hand=128, rule="ordered_original_keys_contiguous_chunks; no keys or starts dropped"),
        post_refinement=dict(
            recipe_id="sg9_contact256_temporal100_v001",
            filter=dict(filter_kind="savgol", window_length=9, polyorder=3,
                        translation_limit_m=.15, rotation_limit_rad=1.5),
            paired_steps=300, contact_weight=256.,
            spaced_contact_selection=dict(points_per_digit=4, minimum_hand_spacing_m=.003),
            temporal_multiplier=100.),
    )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _keys(value, required, optional, label):
    if not isinstance(value, dict) or required-set(value) or set(value)-required-optional:
        raise ValueError(f"Invalid {label} fields; numerical case overrides are not allowed")


def compile_refinement(base_config, recipe):
    """Keep the initial paired objective intact; derive its fixed later stages.

    The recipe is part of the global profile, never a per-case argument. The SG
    output only changes the later warm start; original source priors, bounds,
    contact intervals, static policy and collision terms remain unchanged.
    """
    if base_config.get("anchor_report") or base_config.get("refinement_reference_prior"):
        raise ValueError("Default post-refinement does not add anchor or warm-reference objectives")
    filtered = deepcopy(recipe["filter"])
    for key in ("translation_limit_m", "rotation_limit_rad"):
        if filtered[key] != base_config[key]:
            raise ValueError("SG and paired stages must use the same original source bounds")
    filtered["strict_hold_intervals"] = deepcopy(
        base_config.get("strict_hold_parameterization", {}).get("intervals", {}))
    refined = deepcopy(base_config)
    refined["paired_steps"] = recipe["paired_steps"]
    refined["paired_loss_weights"]["contact"] = recipe["contact_weight"]
    refined["spaced_contact_selection"] = deepcopy(recipe["spaced_contact_selection"])
    refined["temporal_weights"] = {
        key: value * recipe["temporal_multiplier"]
        for key, value in base_config["temporal_weights"].items()
    }
    refined["workflow_policy"] = {
        **deepcopy(base_config["workflow_policy"]), "stage":"sg_contact_smooth",
        "post_refinement_recipe_id":recipe["recipe_id"],
        "base_paired_semantic_config_sha256":digest(base_config),
    }
    return filtered, refined


def compile_policy(case, annotations, *, frame_count, fps, is_static_obj, device="cuda:0"):
    """Compile semantics to fixed solver configs, without changing input objects.

    All annotation intervals use half-open source-frame indices. An observed
    same-grasp interval shares object-relative wrist and finger pose except the
    protected first frame. Missing evidence remains unknown and per-frame.
    """
    _keys(case, {"inputs", "source_arrays", "contact_mask"},
          {"contact_region_faces", "contact_patch_members", "is_static_obj"}, "case")
    _keys(annotations, {"schema_version", "source_video", "frame_count", "fps", "hands"}, set(), "annotations")
    if annotations["schema_version"] != 1 or type(annotations["schema_version"]) is not int:
        raise ValueError("Unsupported annotation schema")
    if type(frame_count) is not int or frame_count < 4 or type(annotations["frame_count"]) is not int or annotations["frame_count"] != frame_count:
        raise ValueError("Annotation frame count must match original source")
    if not np.isfinite(fps) or fps <= 0 or annotations["fps"] != fps:
        raise ValueError("Annotations must retain original FPS")
    if type(is_static_obj) is not bool or ("is_static_obj" in case and case["is_static_obj"] is not is_static_obj):
        raise ValueError("Static policy must agree with the entire original trajectory")
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("Explicit cpu or cuda:N required")
    video = annotations["source_video"]
    _keys(video, {"path", "sha256"}, set(), "source video")
    if not isinstance(video["path"], str) or not video["path"] or not isinstance(video["sha256"], str) or len(video["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in video["sha256"]):
        raise ValueError("Source video needs an exact SHA256 binding")
    sides = set(case["inputs"]["hands"])
    if not sides or not sides <= {"left", "right"} or set(annotations["hands"]) != sides or set(case["contact_mask"]) != sides:
        raise ValueError("Annotations and contact masks must bind every configured hand")
    profile = shared_profile()
    labels, intervals, seed_phases, seed_frames = {}, {}, {}, {}
    protected = []
    for side in sorted(sides):
        active = np.asarray(case["contact_mask"][side])
        if active.dtype.kind != "b" or active.shape != (frame_count,):
            raise ValueError("Exact original Boolean contact mask required")
        labels[side] = ["unknown"]*frame_count
        intervals[side], seed_phases[side], seed_frames[side] = [], [], []
        rows = annotations["hands"][side]
        if not isinstance(rows, list):
            raise ValueError("Hand annotations must be an ordered interval list")
        previous_end = 0
        for row in rows:
            _keys(row, {"start", "end_exclusive", "role", "evidence"}, {"grasp_id"}, "phase")
            first, end, role = row["start"], row["end_exclusive"], row["role"]
            if type(first) is not int or type(end) is not int or not previous_end <= first < end <= frame_count:
                raise ValueError("Phase intervals must be ordered, nonoverlapping original frames")
            if role not in ROLES:
                raise ValueError("Unknown phase role")
            evidence = row["evidence"]
            _keys(evidence, {"frames", "note", "certainty"}, set(), "phase evidence")
            frames = evidence["frames"]
            if not isinstance(frames, list) or not frames or frames != sorted(set(frames)) or any(type(f) is not int or not first <= f < end for f in frames):
                raise ValueError("Evidence must identify observed frames inside its phase")
            if not isinstance(evidence["note"], str) or not evidence["note"].strip() or evidence["certainty"] not in ("observed", "uncertain"):
                raise ValueError("Phase evidence needs an explanation and explicit certainty")
            if role in HOLD and (evidence["certainty"] != "observed" or not isinstance(row.get("grasp_id"), str) or not row["grasp_id"].strip()):
                raise ValueError("A shared grasp needs observed evidence and a grasp_id; uncertainty stays per-frame")
            if role not in HOLD and "grasp_id" in row:
                raise ValueError("Only strict grasps have a shared grasp_id")
            if role in HOLD | PATH and not active[first:end].all():
                raise ValueError("Phase conflicts with original contact mask; revise evidence explicitly upstream")
            if role in {"no_contact", "free"} and active[first:end].any():
                raise ValueError("Noncontact phase conflicts with original contact mask; revise evidence explicitly upstream")
            labels[side][first:end] = [role]*(end-first)
            previous_end = end
            if is_static_obj or role not in HOLD | PATH:
                continue
            start = max(1, first)
            if first == 0:
                protected.append(dict(side=side, original_start=0, effective_start=start, reason="original_frame_zero_is_protected"))
            if start >= end:
                continue
            if role in HOLD:
                intervals[side].append(dict(start=start, end_exclusive=end, grasp_id=row["grasp_id"]))
                keys = [(start+end-1)//2]
            else:
                keys = sorted({start, (start+end-1)//2, end-1})
            seed_phases[side].append(dict(start=start, end_exclusive=end, role=role, keys=keys))
            seed_frames[side].extend(keys)
        seed_frames[side] = sorted(set(seed_frames[side]))
    has_holds = any(intervals.values())
    transition = int(np.floor(fps*profile["transition_seconds"]+.5))
    if not 0 <= transition <= 120:
        raise ValueError("Source FPS exceeds fixed transition implementation range")
    common = dict(inputs=deepcopy(case["inputs"]), fps=fps, contact_mask=deepcopy(case["contact_mask"]),
                  phase_labels=labels, is_static_obj=is_static_obj, device=device)
    if "contact_region_faces" in case:
        common["contact_region_faces"] = deepcopy(case["contact_region_faces"])
    numerical = ("paired_steps", "paired_learning_rate", "samples_per_geom", "translation_limit_m", "rotation_limit_rad",
                 "clearance_m", "cross_hand_clearance_m", "global_worst_object_weight", "global_worst_cross_hand_weight",
                 "paired_loss_weights", "source_prior", "temporal_weights", "collision_collocation_fractions",
                 "collision_collocation_weight", "object_relative_temporal_roles")
    paired = {**deepcopy(common), **{key:deepcopy(profile[key]) for key in numerical},
              "temporal_coordinate_space":"phase", "rotation_backend":"rodrigues",
              "execution_backend":"eager" if has_holds or device == "cpu" else "cuda_graph",
              "refinement_editable_frames":{side:list(range(1,frame_count)) for side in sides}}
    if has_holds:
        paired["strict_hold_parameterization"] = dict(warm_policy="require_constant", intervals=intervals,
            protected_frames=[0], boundary_blend_frames=transition,
            boundary_blend_policy=profile["boundary_blend_policy"], allowed_blend_roles=profile["allowed_blend_roles"])
        paired["feasible_shared_hold_step"] = deepcopy(profile["shared_feasibility_backtracking"])
    paired["workflow_policy"] = dict(profile_id=profile["profile_id"], profile_sha256=digest(profile),
                                    annotation_sha256=digest(annotations))
    filtered, refined = compile_refinement(paired, profile["post_refinement"])
    source = {**deepcopy(common), "source_arrays":case["source_arrays"], "topo_steps":0, "pad_steps":0,
              "temporal_steps":0, "samples_per_geom":profile["samples_per_geom"],
              "loss_weights":{"scene_collision":profile["paired_loss_weights"]["scene_collision"]}}
    topo = None
    if any(seed_frames.values()):
        if "contact_patch_members" not in case or set(case["contact_patch_members"]) != sides:
            raise ValueError("Contact search needs explicit anatomical patch members, not case-specific thresholds")
        searched = {side for side,frames in seed_frames.items() if frames}
        bank_inputs = deepcopy(case["inputs"])
        bank_inputs["hands"] = {side:hand for side,hand in bank_inputs["hands"].items() if side in searched}
        bank = dict(inputs=bank_inputs, seed_frames={side:seed_frames[side] for side in sorted(searched)}, device=device, screen_backend="cpu",
            samples_per_geom=profile["samples_per_geom"], translation_limit_m=profile["translation_limit_m"],
            rotation_limit_rad=profile["rotation_limit_rad"], clearance_m=profile["clearance_m"],
            learning_rate=profile["bank_learning_rate"], self_clearance_weight=profile["bank_self_clearance_weight"],
            scene_clearance_weight=profile["bank_scene_clearance_weight"],
            contact_patch={**deepcopy(profile["contact_patch"]), "members":{side:deepcopy(case["contact_patch_members"][side]) for side in sorted(searched)}},
            multistart=deepcopy(profile["multistart"]))
        if "contact_region_faces" in case:
            bank["contact_region_faces"] = {side:deepcopy(regions) for side,regions in case["contact_region_faces"].items() if side in searched}
        topo = dict(bank=bank, source_arrays=case["source_arrays"], topo_steps=profile["topo_steps"],
                    pad_steps=0, topo_objective=deepcopy(profile["topo_objective"]), is_static_obj=False)
    return dict(schema_version=1, profile=profile, profile_sha256=digest(profile),
        annotation_sha256=digest(annotations), source_video=deepcopy(video), source_config=source,
        topo_config=topo, paired_config=paired, filter_config=filtered,
        refined_config=refined, seed_phases=seed_phases,
        execution_order=["source", "topo", "phase_seed", "seed_feasibility", "base_paired",
                         "sg_filter", "sg_contact_smooth"],
        selected_stage="sg_contact_smooth",
        protected_first_frame_exceptions=protected, motion_accepted=False,
        limits=["Fixed SG/contact/smooth continuation tested on two cases; this is not a quality guarantee for every case.",
                "Source video annotations are authored evidence, not verified physical contact.",
                "Hand-only results require full-body SOMA and independent geometry review."])


def automatic_seed(source, plan, banks, joint_limits):
    """Use the same transparent ranking/DP for every case, retaining failures."""
    from .hand_candidate_selection import shortlist_candidates, build_seed_plans
    from .hand_sequence_seed import build_seed
    from .hand_sequence_path_seed import build_path_seed
    selection, rankings = {}, {}
    for side, phases in plan["seed_phases"].items():
        if not phases:
            continue
        total = sum(len(b["data"]["candidates"].get(side, [])) for b in banks.values())
        ranking = shortlist_candidates(banks, side, limit=max(1,total), source_arrays=source)
        rankings[side] = ranking
        selection[side] = []
        for phase in phases:
            layers = []
            for frame in phase["keys"]:
                rows = [row for row in ranking["candidates"] if row["source_frame"] == frame]
                if not rows:
                    raise ValueError(f"Missing Topo candidates for {side} source frame {frame}")
                choices = [dict(stage=row["stage"], bank_id=row["bank_id"], candidate_id=row["candidate_id"],
                    diagnostic_reason="Fixed-profile numerical warm seed only; all rejection evidence remains unresolved")
                    for row in rows[:plan["profile"]["path_alternatives"]]]
                layers.append(dict(frame=frame, candidates=choices))
            item = dict(start=phase["start"], end=phase["end_exclusive"]-1, role=phase["role"])
            if phase["role"] in HOLD:
                item["candidate"] = layers[0]["candidates"][0]
            else:
                item["key_layers"] = layers
            selection[side].append(item)
    config = plan["paired_config"]
    holds, paths, provenance = build_seed_plans(banks, config, selection, joint_limits)
    # Transitions are optimized once inside the live shared-hold decoder.
    warm, hold_receipt = build_seed(source, config, holds, transition_frames=0)
    warm, path_receipt = build_path_seed(warm, config, paths, joint_limits)
    for side in config["inputs"]["hands"]:
        for suffix in ("wrist_position_object", "wrist_rotation_object", "finger_qpos"):
            key = f"{side}_{suffix}"
            warm[key][0] = source[key][0]
        warm[f"{side}_phase_labels"] = np.asarray(config["phase_labels"][side])
    return warm, dict(selection=selection, rankings=rankings, provenance=provenance,
        hold_seed=hold_receipt, path_seed=path_receipt, frame_zero_restored_from_original=True,
        material_anchors_created=False, motion_accepted=False)


def topo_chunks(config):
    """Preserve all starts and source keys within the existing per-bank cap.

    Multistart randomness is already keyed by side/frame/start, so partitioning
    a long key list does not change the generated initial perturbations.
    Every resulting bank retains its own exact config/input/SHA bindings.
    """
    from .grasp_multistart import MAX_CANDIDATES_PER_HAND
    count=config["bank"]["multistart"]["count_per_frame"]
    if type(count) is not int or not 1 <= count <= MAX_CANDIDATES_PER_HAND:
        raise ValueError("Invalid globally configured starts per key")
    capacity=MAX_CANDIDATES_PER_HAND//count
    frames=config["bank"]["seed_frames"]
    if not frames or any(not row or row != sorted(set(row)) for row in frames.values()):
        raise ValueError("Topo chunks require ordered unique original keyframes")
    chunks=[]
    for offset in range(0,max(map(len,frames.values())),capacity):
        chunk=deepcopy(config);bank=chunk["bank"]
        bank["seed_frames"]={side:row[offset:offset+capacity] for side,row in frames.items() if offset<len(row)}
        selected=set(bank["seed_frames"])
        bank["inputs"]["hands"]={side:hand for side,hand in bank["inputs"]["hands"].items() if side in selected}
        bank["contact_patch"]["members"]={side:hand for side,hand in bank["contact_patch"]["members"].items() if side in selected}
        if "contact_region_faces" in bank:
            bank["contact_region_faces"]={side:regions for side,regions in bank["contact_region_faces"].items() if side in selected}
        chunks.append(chunk)
    return chunks


def _file_binding(path):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def _paired_artifacts(config_path, report_path, sequence_path):
    """Verify the actual solver export before exposing it as a downstream input."""
    artifacts = {key:_file_binding(path) for key,path in
                 dict(config=config_path, report=report_path, sequence=sequence_path).items()}
    config = json.loads(Path(config_path).read_bytes())
    report = json.loads(Path(report_path).read_bytes())
    if (report.get("motion_accepted") is not False or report.get("input_config") != config
            or report.get("steps") != config["paired_steps"]
            or report.get("outputs", {}).get(Path(sequence_path).name, {}).get("sha256")
                != artifacts["sequence"]["sha256"]):
        raise ValueError("Paired output/config/iteration binding mismatch")
    warm = _file_binding(config["warm_start_npz"])
    if report.get("source_warm_start", {}).get("sha256") != warm["sha256"]:
        raise ValueError("Paired solver warm-start binding mismatch")
    return artifacts


def _final_preservation(source_path, sequence_path, config):
    """Cheap exact payload checks; these are not geometry or motion-quality tests."""
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key:archive[key].copy() for key in archive.files}
    with np.load(sequence_path, allow_pickle=False) as archive:
        final = {key:archive[key].copy() for key in archive.files}
    exact = lambda a,b:a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    checks = {key:exact(source[key], final[key]) for key in
              ("source_qpos", "object_R", "object_trans", "object_scale", "timestamps", "fps")}
    for side in config["inputs"]["hands"]:
        for suffix in ("wrist_position_object", "wrist_rotation_object", "finger_qpos"):
            key = side+"_"+suffix
            checks[key+"_frame0"] = exact(source[key][0], final[key][0])
        for suffix in ("finger_addresses", "finger_joint_names"):
            key = side+"_"+suffix
            checks[key] = exact(source[key], final[key])
        expected_mask = np.asarray(config["contact_mask"][side], dtype=bool)
        if config["is_static_obj"]:
            expected_mask[:] = False
        checks[side+"_contact_mask"] = np.array_equal(final[side+"_contact_active"], expected_mask)
        checks[side+"_phase_labels"] = final[side+"_phase_labels"].tolist() == config["phase_labels"][side]
    if not all(checks.values()):
        raise ValueError("Final hand trajectory changed protected data: "+str([key for key,value in checks.items() if not value]))
    return dict(checks={key:bool(value) for key,value in checks.items()}, all_pass=True,
                geometry_audited=False, motion_accepted=False)


def run(case_path, annotation_path, output, *, device="cuda:0", plan_only=False):
    """Execute Topo → original paired solve → SG → contact/smooth paired solve.

    ``handoff_inputs`` in the final report binds the final config, solver report,
    sequence and explicit unaccepted selection. Intermediate ``paired`` outputs
    remain available for comparison and are never the implicit final target.
    """
    from .grasp_io import GraspInputs
    from .hand_sequence import resolve_static_object_policy, solve as source_solve
    from .topo_hand_initialization import run as topo_run
    from .hand_candidate_selection import load_bound_bank
    from .hand_sequence_pair import solve as pair_solve
    from .hand_sequence_filter import run as filter_run
    tick = time.monotonic()
    case_path, annotation_path = Path(case_path).resolve(strict=True), Path(annotation_path).resolve(strict=True)
    raw_case, raw_annotations = case_path.read_bytes(), annotation_path.read_bytes()
    case, annotations = json.loads(raw_case), json.loads(raw_annotations)
    for key in ("native_npz", "model_xml", "object_mesh_npz"):
        case["inputs"][key] = str((case_path.parent/case["inputs"][key]).resolve(strict=True))
    case["source_arrays"] = str((case_path.parent/case["source_arrays"]).resolve(strict=True))
    video = (annotation_path.parent/annotations["source_video"]["path"]).resolve(strict=True)
    if hashlib.sha256(video.read_bytes()).hexdigest() != annotations["source_video"]["sha256"]:
        raise ValueError("Source video hash mismatch")
    ctx = GraspInputs(case["inputs"], case_path.parent)
    static = resolve_static_object_policy(case, ctx.R, ctx.t)["is_static_obj"]
    if "fps" not in ctx.native:
        raise ValueError("Original native reference must declare FPS")
    fps = float(ctx.native["fps"])
    if "timestamps" in ctx.native and not np.array_equal(ctx.native["timestamps"], np.arange(len(ctx.q))/fps):
        raise ValueError("Original clock is not uniform; no quiet resampling")
    plan = compile_policy(case, annotations, frame_count=len(ctx.q), fps=fps, is_static_obj=static, device=device)
    if plan["topo_config"] is not None:
        # Same rule for every object: never request more original faces than exist.
        plan["topo_config"]["topo_objective"]["object_sample_count"] = min(
            plan["profile"]["topo_objective"]["object_sample_count"], len(ctx.object.faces))
    plan["bindings"] = dict(original=ctx.binding, source_arrays_sha256=hashlib.sha256(Path(case["source_arrays"]).read_bytes()).hexdigest(),
        case_sha256=hashlib.sha256(raw_case).hexdigest(), annotations_sha256=hashlib.sha256(raw_annotations).hexdigest(),
        source_video_path=str(video), implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        path = output/name
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
        return path
    stages = []
    def timed(name, operation):
        started = time.monotonic()
        row = dict(stage=name, status="RUNNING")
        stages.append(row)
        save("stage_progress.json", stages)
        try:
            value = operation()
            row["status"] = "COMPLETED_UNREVIEWED"
            return value
        except Exception as error:
            row.update(status="FAILED", error=type(error).__name__+": "+str(error))
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic()-started
            save("stage_progress.json", stages)
    save("plan.json", plan)
    if plan_only:
        return dict(status="COMPILED_NOT_EXECUTED", profile_sha256=plan["profile_sha256"], output=str(output))
    timed("source", lambda:source_solve(plan["source_config"], case_path.parent, output/"source"))
    with np.load(output/"source/source.npz", allow_pickle=False) as data:
        source = {key:data[key].copy() for key in data.files}
    banks = {}
    if plan["topo_config"] is not None:
        chunks=topo_chunks(plan["topo_config"])
        bank_index=[]
        for index,chunk in enumerate(chunks):
            bank_id="topo" if len(chunks)==1 else f"topo_chunk_{index:03d}"
            config_name="topo_config.json" if len(chunks)==1 else f"{bank_id}_config.json"
            topo = timed(bank_id, lambda:topo_run(save(config_name,chunk), output/bank_id))
            bank_ctx = GraspInputs(chunk["bank"]["inputs"],case_path.parent)
            banks[bank_id]=load_bound_bank(topo["final_bank"],sha256=topo["final_bank_sha256"],
                expected_binding=bank_ctx.binding,stage="topo",bank_id=bank_id)
            bank_index.append(dict(bank_id=bank_id,config=config_name,seed_frames=chunk["bank"]["seed_frames"],
                bank=topo["final_bank"],bank_sha256=topo["final_bank_sha256"],starts_per_key=chunk["bank"]["multistart"]["count_per_frame"],
                motion_accepted=False))
            save("topo_banks.json",dict(chunks=bank_index,planned_chunks=len(chunks),
                original_seed_frames=plan["topo_config"]["bank"]["seed_frames"],all_chunks_complete=len(bank_index)==len(chunks)))
    limits = {side:ctx.model.jnt_range[hand["joints"]].copy() for side,hand in ctx.hands.items()}
    warm, seed_receipt = timed("phase_seed", lambda:automatic_seed(source, plan, banks, limits))
    save("seed_receipt.json", seed_receipt)
    from .hand_shared_seed_projection import project_seed, SeedProjectionError
    np.savez_compressed(output/"warm_before_feasibility.npz", **warm)
    try:
        warm, projection_receipt = timed("seed_feasibility", lambda:project_seed(source, warm, plan["paired_config"], limits))
    except SeedProjectionError as error:
        save("seed_feasibility.json", error.receipt)
        raise
    save("seed_feasibility.json", projection_receipt)
    np.savez_compressed(output/"warm.npz", **warm)
    config = {**plan["paired_config"], "warm_start_npz":str(output/"warm.npz")}
    base_config_path = save("paired_config.json", config)
    timed("base_paired", lambda:pair_solve(config, case_path.parent, output/"paired"))
    initial = _paired_artifacts(base_config_path, output/"paired/report.json", output/"paired/hand_sequence.npz")
    save("filter_config.json", plan["filter_config"])
    filtered_report = timed("sg_filter", lambda:filter_run(
        initial["sequence"]["path"], output/"source/source.npz", output/"filtered",
        joint_limits=limits, config=plan["filter_config"]))
    filtered_path = output/"filtered/hand_sequence_UNREVIEWED.npz"
    refined_config = {**plan["refined_config"], "warm_start_npz":str(filtered_path)}
    refined_config_path = save("refined_config.json", refined_config)
    timed("sg_contact_smooth", lambda:pair_solve(refined_config, case_path.parent, output/"refined"))
    handoff = _paired_artifacts(refined_config_path, output/"refined/report.json", output/"refined/hand_sequence.npz")
    preservation = _final_preservation(output/"source/source.npz", handoff["sequence"]["path"], refined_config)
    selection = dict(schema="hand_sequence_selection.v1", motion_accepted=False,
        **{key+"_sha256":handoff[key]["sha256"] for key in ("config", "sequence", "report")},
        reason="Fixed default recipe: Topo then original paired solve, SG9/poly3, and contact256/spaced4/3mm/temporal100 paired continuation. Selected only as unreviewed soft targets for fresh SOMA; no grasp, collision or motion acceptance.")
    handoff["selection"] = _file_binding(save("selection.json", selection))
    report = dict(status="UNREVIEWED_HAND_SEQUENCE", profile_id=plan["profile"]["profile_id"], profile_sha256=plan["profile_sha256"],
        output=str(output), elapsed_seconds=time.monotonic()-tick, motion_accepted=False,
        frame_count=len(ctx.q), fps=fps, preservation=preservation,
        selected_stage="sg_contact_smooth", handoff_inputs=handoff,
        selected_sequence=handoff["sequence"]["path"], selected_report=handoff["report"]["path"],
        selected_config=handoff["config"]["path"], selection=handoff["selection"]["path"],
        paired_report=handoff["report"]["path"], initial_paired=initial,
        filtered=dict(sequence=_file_binding(filtered_path), report=_file_binding(output/"filtered/report.json"),
                      source=filtered_report["source"], parent=filtered_report["candidate"]),
        stages=stages, timings_seconds={row["stage"]:row["elapsed_seconds"] for row in stages},
        paired_iterations=dict(initial=config["paired_steps"], refinement=refined_config["paired_steps"]),
        standalone_pad_executed=False, material_anchors_created=False, whole_body_soma_executed=False,
        review_required="Audit final selected targets and fresh whole-body SOMA outputs; fixed-stage selection is not motion acceptance.")
    save("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.case, args.phases, args.out, device=args.device, plan_only=args.plan_only)))


if __name__ == "__main__":
    main()
