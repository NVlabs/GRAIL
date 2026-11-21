"""External-runtime bridge adding material contacts to the SAME SOMA IK solve.

Run by soma.py only when contact_constraints is explicit. Adapt the upstream
pipeline classes inside this single subprocess; no external checkout is edited.
Newton/Warp imports stay out of the ordinary CPU adapter/test environment.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import warp as wp
import newton
import newton.ik as ik


@wp.kernel
def _residual(body_q: wp.array2d(dtype=wp.transform), target: wp.array(dtype=wp.vec3),
              link: int, local: wp.vec3, direction: int, weight: wp.array(dtype=float),
              row: int, problem_idx: wp.array(dtype=int), out: wp.array2d(dtype=float)):
    i = wp.tid()
    current = wp.transform_point(body_q[i, link], local)
    if direction == 1:
        current = wp.transform_vector(body_q[i, link], local)
    error = (target[problem_idx[i]] - current) * weight[0]
    for k in range(3):
        out[i, row+k] = error[k]


@wp.kernel
def _jacobian(body_q: wp.array2d(dtype=wp.transform), motion: wp.array2d(dtype=wp.spatial_vector),
              affects: wp.array(dtype=wp.uint8), link: int, local: wp.vec3, direction: int,
              weight: wp.array(dtype=float), row: int, out: wp.array3d(dtype=float)):
    i, dof = wp.tid()
    value = wp.vec3(0.0)
    if affects[dof] != wp.uint8(0):
        s = motion[i, dof]
        omega = wp.vec3(s[3], s[4], s[5])
        if direction == 1:
            value = wp.cross(omega, wp.transform_vector(body_q[i, link], local))
        else:
            value = wp.vec3(s[0], s[1], s[2]) + wp.cross(omega, wp.transform_point(body_q[i, link], local))
    for k in range(3):
        out[i, row+k, dof] = -weight[0]*value[k]


class MaterialObjective(ik.IKObjectivePosition):
    """Position or unit-vector error with a graph-safe device-side phase gain.

The inherited initialization supplies the exact articulation ancestor mask.
Normal alignment constrains two directions, leaving twist about the normal free.
"""
    def __init__(self, link, local, *, direction=False):
        super().__init__(link, wp.vec3(*local), wp.zeros(1, dtype=wp.vec3), weight=1.0)
        self.direction = int(direction)
        self.dynamic_weight = wp.zeros(1, dtype=wp.float32)

    def init_buffers(self, model, jacobian_mode):
        if jacobian_mode != ik.IKJacobianType.ANALYTIC:
            raise ValueError("Material contact bridge requires Newton analytic Jacobians")
        super().init_buffers(model, jacobian_mode)

    def update(self, target, weight):
        self.target_positions.assign(np.asarray(target, dtype=np.float32).reshape(1, 3))
        self.dynamic_weight.assign(np.asarray([weight], dtype=np.float32))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_residual, dim=body_q.shape[0], inputs=[body_q, self.target_positions,
                  self.link_index, self.link_offset, self.direction, self.dynamic_weight,
                  start_idx, problem_idx, residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_jacobian, dim=(body_q.shape[0], model.joint_dof_count), inputs=[
            body_q, joint_S_s, self.affects_dof, self.link_index, self.link_offset,
            self.direction, self.dynamic_weight, start_idx, jacobian], device=self.device)


def material_objectives(link, point, normal, normal_weight):
    """Omit identically zero normal rows without changing the contact objective.

    Dense Newton LM allocates GPU tiles from the residual count. Zero-weight
    normals needlessly enlarge these tiles and can exceed device shared memory.
    Positive normal weights retain the original objective and diagnostics.
    """
    return (MaterialObjective(link, point),
            MaterialObjective(link, normal, direction=True) if normal_weight > 0 else None)


def material_kinematics(pipeline, q, schedule):
    """Actual mounted native material points, normals and rigid body poses."""
    from soma_retargeter.utils.newton_utils import get_name_from_label
    state = pipeline.ik_model.state()
    pipeline.ik_model.joint_q.assign(q)
    newton.eval_fk(pipeline.ik_model, pipeline.ik_model.joint_q,
                   pipeline.ik_model.joint_qd, state)
    transforms = state.body_q.numpy()
    labels = [get_name_from_label(x) for x in pipeline.ik_model.body_label]
    points, normals = [], []
    for body, point, normal in zip(schedule["body_names"], schedule["points"], schedule["normals"]):
        tx = wp.transform(*transforms[labels.index(str(body))])
        points.append(np.asarray(wp.transform_point(tx, wp.vec3(*point))))
        normals.append(np.asarray(wp.transform_vector(tx, wp.vec3(*normal))))
    return (np.asarray(points).reshape(-1, 3), np.asarray(normals).reshape(-1, 3),
            wp.array(transforms.reshape(1, -1, 7), dtype=wp.transform))


def object_contact_gate(pipeline, q, controller, schedule):
    """Report contact quality without controlling object updates or export.

    The historical key is retained for readers of earlier reports. Object motion
    is selected deterministically from the source trajectory and contact frame;
    a measured gap is diagnostic, including when no contact row is active.
    """
    config = controller.config
    rows = []
    frames = {config["contact_frame"]}
    if config["first_edit_frame"] > 0:
        frames.add(config["first_edit_frame"] - 1)
    for frame in sorted(frames):
        active = schedule["gains"][frame] > 0
        points, _, _ = material_kinematics(pipeline, q[frame], schedule)
        gaps = np.linalg.norm(points-schedule["targets"][frame], axis=1)[active]
        passed = bool(len(gaps) and np.max(gaps) <= config["contact_tolerance_m"])
        rows.append(dict(frame=frame, active_contacts=int(active.sum()),
            maximum_gap_m=float(gaps.max()) if len(gaps) else None, passed=passed))
    return dict(passed=all(row["passed"] for row in rows), rows=rows, diagnostic_only=True,
        tolerance_m=config["contact_tolerance_m"],
        scope="Mounted material gaps at declared contact and last precontact frame; not force closure or exact mesh touch")


def refine_temporal_sequence(pipeline, pass_, buffer, config, report_path,
                             object_controller=None, schedule=None):
    """Revisit source frames with the original IK objectives and temporal rows.

    Each visit minimizes the conditional sequence objective with neighboring
    hinges fixed. This is an alternating sequence solve, not a post-hoc filter.
    Only complete sweeps are published if the cooperative deadline is reached.
    """
    if config.get("method") == "synchronous_sparse":
        from soma_sequence_runtime import refine_synchronous_sequence
        return refine_synchronous_sequence(pipeline, pass_, buffer, config, report_path,
            object_controller=object_controller, schedule=schedule)
    from soma_temporal import conditional_target, temporal_metrics
    started = time.monotonic()
    deadline = started + config["max_seconds"]
    deadline_source = "configured_temporal_duration"
    finalization_reserve_seconds = 30.
    inherited_deadlines = {}
    # The owning batch process terminates repair stages at its absolute cutoff.
    # Its deadline may precede max_seconds, which starts only after initial IK.
    # Finish the normal hash checks, post-clamp measurements and CSV export
    # before that cutoff instead of requiring a manually reconstructed export.
    for name in ("RETARGET_STOP_REPAIRS_UNIX", "RETARGET_AGENT_DEADLINE_UNIX",
                 "RETARGET_ALLOCATION_DEADLINE_UNIX"):
        if name in os.environ:
            value = float(os.environ[name])
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite absolute deadline")
            inherited_deadlines[name] = value
    if inherited_deadlines:
        name = min(inherited_deadlines, key=inherited_deadlines.get)
        cutoff = started + inherited_deadlines[name] - time.time() - finalization_reserve_seconds
        if cutoff < deadline:
            deadline, deadline_source = cutoff, name
    solver = pass_._ik_solver
    objective = solver.temporal_objective
    coordinates = np.asarray(objective.coord_indices, dtype=int)
    q = np.asarray(buffer.data).copy()
    initial = q.copy()
    fps = float(buffer.sample_rate)
    source_targets = np.ascontiguousarray(pipeline.input_targets[0])
    target_hash = hashlib.sha256(source_targets.tobytes()).hexdigest()
    before = temporal_metrics(q[:, coordinates], fps, config, joint_names=objective.names)
    checkpoint = Path(report_path).with_name("temporal_sequence_states.npz")
    history = []
    object_history = []
    robot_only_sweeps = 0
    object_updates_enabled = False
    initial_contact_gate = None
    object_fields = ("positions", "rotations", "translation_offsets", "rotation_offsets")
    if object_controller is not None:
        initial_objects = {key: getattr(object_controller, key).copy() for key in object_fields}
        gate = object_contact_gate(pipeline, q, object_controller, schedule)
        initial_contact_gate = gate
        object_updates_enabled = object_controller.optimization_enabled
        print("[TEMPORAL] object policy=" + object_controller.motion_policy["mode"]
            + "; contact gaps are diagnostic only", flush=True)
    reason = "SWEEPS_COMPLETE"
    warm_start_report = None
    warm_started = None
    if "sequence_warm_start" in config:
        from soma_sequence_warm_start import sequence_warm_start
        model = pipeline.ik_model
        dofs = objective.dof_indices
        lower = model.joint_limit_lower.numpy()[dofs]
        upper = model.joint_limit_upper.numpy()[dofs]
        seeded, warm_start_report, seed_arrays = sequence_warm_start(
            q[:, coordinates], fps, config, objective.names, lower, upper)
        q[:, coordinates] = seeded
        warm_started = q.copy()
        seed_path = Path(report_path).with_name("hand_sequence_warm_start.npz")
        np.savez_compressed(seed_path, **seed_arrays)
        warm_start_report["evidence_path"] = str(seed_path)
        warm_start_report["evidence_sha256"] = hashlib.sha256(seed_path.read_bytes()).hexdigest()
        warm_start_report["after_seed_temporal_metrics"] = temporal_metrics(
            q[:, coordinates], fps, config, joint_names=objective.names)
        warm_start_report["source_targets_unchanged"] = (
            hashlib.sha256(np.ascontiguousarray(pipeline.input_targets[0]).tobytes()).hexdigest()
            == target_hash)
        if not warm_start_report["source_targets_unchanged"]:
            raise ValueError("Sequence warm start changed original human targets")
        warm_start_report["contact_diagnostic_after_seed"] = (
            object_contact_gate(pipeline, q, object_controller, schedule)
            if object_controller is not None else None)
        print("[TEMPORAL] declared global hand warm start; original IK objectives retained", flush=True)
    try:
        for sweep in range(config["sweeps"]):
            work = q.copy()
            joint_sweep = object_updates_enabled
            if object_controller is not None:
                saved_objects = {key: getattr(object_controller, key).copy() for key in object_fields}
            object_updates = []
            direction = "forward" if sweep % 2 == 0 else "backward"
            order = range(len(work)) if direction == "forward" else range(len(work)-1, -1, -1)
            completed = True
            sweep_start = time.monotonic()
            precontact_update = None
            if object_controller is not None and joint_sweep:
                prefix_end = object_controller.config["contact_frame"]
                if prefix_end and time.monotonic() < deadline:
                    # The resting interval is one shared SE(3) variable. Freeze
                    # every current mounted robot pose for this conditional
                    # block, then synchronize the whole prefix before robot IK.
                    # Never reuse these contexts in the next sweep.
                    prefix_points, prefix_normals, collision_contexts = [], [], []
                    collision = solver.collision_objective
                    for frame in range(prefix_end):
                        if time.monotonic() >= deadline:
                            completed = False
                            reason = "TIME_LIMIT_INCOMPLETE_SWEEP_DISCARDED"
                            break
                        points, normals, poses = material_kinematics(pipeline, work[frame], schedule)
                        prefix_points.append(points)
                        prefix_normals.append(normals)
                        collision_contexts.append(collision.object_context(poses,
                            object_controller.rotations[frame], object_controller.positions[frame]))
                    if time.monotonic() >= deadline:
                        completed = False
                        reason = "TIME_LIMIT_INCOMPLETE_SWEEP_DISCARDED"
                    if completed:
                        precontact_update = object_controller.update_precontact(
                            np.asarray(prefix_points), np.asarray(prefix_normals), collision_contexts)
            for count, frame in enumerate(order):
                if time.monotonic() >= deadline:
                    completed = False
                    reason = "TIME_LIMIT_INCOMPLETE_SWEEP_DISCARDED"
                    break
                if (object_controller is not None and joint_sweep
                        and frame >= object_controller.config["first_edit_frame"]):
                    points, normals, poses = material_kinematics(pipeline, work[frame], schedule)
                    collision = solver.collision_objective
                    # This conditional update freezes the current mounted robot
                    # state. Never reuse a context after the following IK step.
                    collision_context = collision.object_context(poses,
                        object_controller.rotations[frame], object_controller.positions[frame])
                    row = object_controller.update(frame, points, normals, collision_context)
                    object_controller.apply_schedule(frame)
                    object_updates.append({**row, "frame": frame})
                target, gain = conditional_target(work[:, coordinates], frame, fps, config, joint_names=objective.names)
                objective.update(target, gain)
                solver.joint_q.assign(work[frame:frame+1])
                solver.solver.reset()
                pass_.execute(frame + pass_._num_frames_to_remove)
                # Keep the official native clamp inside each coordinate update.
                pipeline.joint_limit_clamper.apply(solver.joint_q)
                work[frame] = solver.joint_q.numpy()[0]
                if not np.isfinite(work[frame]).all():
                    raise ValueError("Nonfinite SOMA temporal coordinate update")
                if count % 30 == 0:
                    print(f"[TEMPORAL] sweep={sweep+1} direction={direction} frame={frame}", flush=True)
            if object_controller is not None and completed:
                gate = object_contact_gate(pipeline, work, object_controller, schedule)
                if not object_controller.report()["passed"]:
                    completed = False
                    reason = "OBJECT_HARD_AUDIT_INCOMPLETE_SWEEP_DISCARDED"
            if not completed:
                if object_controller is not None:
                    for key in object_fields:
                        getattr(object_controller, key)[...] = saved_objects[key]
                    object_controller.apply_schedule()
                break
            if hashlib.sha256(np.ascontiguousarray(pipeline.input_targets[0]).tobytes()).hexdigest() != target_hash:
                raise ValueError("SOMA human reference targets changed during temporal optimization")
            q = work
            history.append(dict(sweep=sweep+1, direction=direction,
                stage=("joint_robot_object" if joint_sweep else "robot_static_object")
                    if object_controller is not None else "robot_temporal",
                seconds=time.monotonic()-sweep_start,
                metrics=temporal_metrics(q[:, coordinates], fps, config, joint_names=objective.names)))
            extra = {}
            if object_controller is not None:
                if joint_sweep and object_updates:
                    object_history.append(dict(sweep=sweep+1, updates=object_updates,
                        precontact_update=precontact_update,
                        audit=object_controller.report(), contact_gate=gate))
                elif not joint_sweep:
                    robot_only_sweeps += 1
                extra = dict(initial_obj_R=initial_objects["rotations"], initial_obj_t=initial_objects["positions"],
                    refined_obj_R=object_controller.rotations, refined_obj_t=object_controller.positions)
            checkpoint_metadata = dict(status="COMPLETE_SWEEP_UNREVIEWED",
                completed_sweeps=len(history), same_soma_objectives=True,
                completed_joint_sweeps=len(object_history), completed_robot_only_sweeps=robot_only_sweeps,
                object_motion_policy=(object_controller.motion_policy if object_controller is not None else None),
                source_targets_unchanged=True, source_target_sha256=target_hash,
                frame_count=len(q), fps=fps, config=config,
                scope="Complete clamped sequence sweep; final converter measurements/export and motion acceptance remain separate")
            temporary_checkpoint = checkpoint.with_suffix(".npz.partial")
            with temporary_checkpoint.open("wb") as stream:
                if warm_started is not None:
                    extra["warm_started_joint_q"] = warm_started
                np.savez_compressed(stream, initial_joint_q=initial, refined_joint_q=q,
                    fps=fps, coordinate_indices=coordinates, joint_names=np.asarray(objective.names),
                    checkpoint_metadata_json=np.asarray(json.dumps(checkpoint_metadata, sort_keys=True)), **extra)
            temporary_checkpoint.replace(checkpoint)
            print(f"[TEMPORAL] completed sweep={sweep+1} metrics={history[-1]['metrics']}", flush=True)
    finally:
        objective.disable()
    if not history:
        raise RuntimeError("SOMA temporal run ended before any complete sequence sweep: " + reason)
    if object_updates_enabled and not object_history:
        raise RuntimeError("SOMA object optimization ended before any complete object-enabled sweep: " + reason)
    if hashlib.sha256(np.ascontiguousarray(pipeline.input_targets[0]).tobytes()).hexdigest() != target_hash:
        raise ValueError("SOMA human reference targets changed during temporal optimization")
    buffer.data = q
    return dict(status="TEMPORAL_INITIALIZATION_UNREVIEWED", config=config,
        same_soma_objectives=True, source_targets_unchanged=True,
        source_target_sha256=target_hash, frame_count=len(q), fps=fps,
        method="alternating forward/backward conditional SOMA IK sequence sweeps",
        joint_names=list(objective.names), coordinate_indices=coordinates.tolist(),
        completed_sweeps=len(history), stop_reason=reason, history=history,
        completed_joint_sweeps=len(object_history), completed_robot_only_sweeps=robot_only_sweeps,
        initial_contact_gate=initial_contact_gate,
        object_motion_policy=(object_controller.motion_policy if object_controller is not None else None),
        deadline=dict(source=deadline_source, inherited_unix=inherited_deadlines,
            finalization_reserve_seconds=finalization_reserve_seconds if inherited_deadlines else 0.,
            effective_temporal_seconds=max(0., deadline-started)),
        before=before, after=temporal_metrics(q[:, coordinates], fps, config, joint_names=objective.names),
        checkpoint=str(checkpoint), seconds=time.monotonic()-started, object_history=object_history,
        **({"sequence_warm_start": warm_start_report} if warm_start_report is not None else {}),
        limitations=["Soft hinge derivative costs, not hard velocity/acceleration/jerk acceptance.",
            "Free root translation and quaternion are not temporal residual coordinates.",
            "Full-body source, material contact, joint-limit and configured collision objectives remain active.",
            "Finite alternating sweeps do not establish global convergence or continuous collision clearance."])


def export_object_result(pipeline, q, controller, schedule, report_path, reference_sha256):
    """Export complete paired results with contact quality recorded, not gated."""
    gate = object_contact_gate(pipeline, q, controller, schedule)
    audit = controller.report()
    if not audit["passed"]:
        raise RuntimeError("Final object hard audit failed")
    object_path = Path(report_path).with_name("optimized_object.npz")
    np.savez_compressed(object_path, obj_R=controller.rotations,
        obj_t=controller.positions, reference_obj_R=schedule["object_reference_rotations"],
        reference_obj_t=schedule["object_reference_positions"], object_scale=schedule["object_scale"],
        fps=schedule["fps"], timestamps=schedule["object_timestamps"],
        world_frame=np.asarray("soma_retargeter_z_up_m"), reference_sha256=np.asarray(reference_sha256),
        config_json=np.asarray(json.dumps(controller.config, sort_keys=True)))
    return dict(config=controller.config, audit=audit, contact_gate=gate,
        path=str(object_path), motion_policy=controller.motion_policy,
        method=("Alternating constrained object SE(3) and original SOMA robot IK blocks"
            if controller.optimization_enabled else
            "Original static object trajectory retained during original SOMA robot IK blocks"))


def install_bridge(schedule, report_path, temporal_config=None, object_config=None, object_reference_sha256=None, hand_target_config=None, robot_hand_target_config=None, pose_keyframe_config=None, wrist_keyframe_config=None):
    from soma_object_policy import contact_loss_policy
    import soma_retargeter.pipelines.soma_retargeting_pipeline as pipeline
    from soma_retargeter.utils.newton_utils import get_name_from_label
    original_pipeline = pipeline.SomaRetargetingPipeline
    original_solver = pipeline.BaseIKSolver
    original_pass = pipeline.BasePass
    frames = len(schedule["gains"])
    contact_policy = contact_loss_policy(schedule)
    collision_enabled = "collision_metadata" in schedule
    collision_holder = []
    pass_holder = []
    object_controller = None
    if object_config is not None:
        from soma_object import ObjectTrajectory
        if temporal_config is None or not collision_enabled:
            raise ValueError("Object optimization requires temporal and collision objectives")
        object_controller = ObjectTrajectory(schedule, object_config,
            allow_empty_material=robot_hand_target_config is not None)
        object_controller.apply_schedule()
    if temporal_config is not None:
        from soma_temporal import validate_temporal_config
        temporal_config = validate_temporal_config(temporal_config)
        if frames < 4:
            raise ValueError("SOMA temporal optimization requires at least four source frames")

    class ContactSolver(original_solver):
        def __init__(self, ik_model, num_envs, model, **kwargs):
            if num_envs != 1:
                raise ValueError("Contact bridge requires exactly one bound source motion")
            labels = [get_name_from_label(x) for x in ik_model.body_label]
            self.material_objectives = []
            contact_rows = (zip(schedule["body_names"], schedule["points"],
                                schedule["normals"], schedule["weights"])
                            if contact_policy["enabled"] else ())
            for body, point, normal, weights in contact_rows:
                link = labels.index(str(body))
                self.material_objectives.append(material_objectives(link, point, normal, weights[1]))
            kwargs["extra_objectives"] = [*(kwargs.get("extra_objectives") or []),
                *(o for pair in self.material_objectives for o in pair if o is not None)]
            self.robot_hand_targets = None
            if robot_hand_target_config is not None:
                from soma_robot_hand_targets import load_robot_hand_targets
                from soma_robot_hand_runtime import RobotHandTargetGroup
                joint_labels = [get_name_from_label(x) for x in ik_model.joint_label]
                loaded = load_robot_hand_targets(robot_hand_target_config,
                    fps=float(schedule["fps"]), frame_count=frames,
                    object_positions=schedule["object_reference_positions"],
                    object_rotations=schedule["object_reference_rotations"],
                    available_joint_names=joint_labels, available_body_names=labels,
                    object_scale=schedule["object_scale"])
                self.robot_hand_targets = RobotHandTargetGroup(ik_model, loaded,
                    body_labels=labels, joint_labels=joint_labels)
                kwargs["extra_objectives"].extend(self.robot_hand_targets.objectives)
            self.wrist_keyframes = None
            if wrist_keyframe_config is not None:
                if self.robot_hand_targets is None:
                    raise ValueError("Wrist keys require unchanged original robot hand targets")
                from soma_wrist_keyframes import load
                from soma_wrist_keyframe_runtime import WristKeyframeGroup
                wrist_loaded = load(wrist_keyframe_config["config"],
                    model_xml=wrist_keyframe_config["model_xml"], source_bvh=wrist_keyframe_config["source_bvh"],
                    frame_count=frames, fps=float(schedule["fps"]), hand_targets=self.robot_hand_targets.targets)
                self.wrist_keyframes = WristKeyframeGroup(ik_model, wrist_loaded, labels)
                kwargs["extra_objectives"].extend(self.wrist_keyframes.objectives)
            self.pose_keyframes = None
            if pose_keyframe_config is not None:
                from soma_pose_keyframe_runtime import PoseKeyframeObjective
                self.pose_keyframes = PoseKeyframeObjective(ik_model,
                    pose_keyframe_config["config"], model_xml=pose_keyframe_config["model_xml"],
                    source_bvh=pose_keyframe_config["source_bvh"], frame_count=frames,
                    fps=float(schedule["fps"]))
                kwargs["extra_objectives"].append(self.pose_keyframes)
            self.collision_objective = None
            if collision_enabled:
                from soma_collision_runtime import CollisionObjective
                self.collision_objective = CollisionObjective(schedule, labels)
                collision_holder[:] = [self.collision_objective]
                kwargs["extra_objectives"].append(self.collision_objective)
            self.temporal_objective = None
            if temporal_config is not None:
                from soma_temporal_runtime import TemporalObjective
                self.temporal_objective = TemporalObjective(ik_model)
                kwargs["extra_objectives"].append(self.temporal_objective)
            super().__init__(ik_model, num_envs, model, **kwargs)
            self.forward_cuda_solver = None
            if (self.solver._impl.device.is_cuda and temporal_config is not None
                    and temporal_config.get("method") == "synchronous_sparse"):
                from soma_forward_cuda import install_forward_cuda_solver
                self.forward_cuda_solver = install_forward_cuda_solver(self.solver._impl)

    class ContactPass(original_pass):
        def __init__(self, *args, **kwargs):
            started = time.monotonic()
            super().__init__(*args, **kwargs)
            self.forward_pass_setup_seconds = time.monotonic()-started
            self.forward_dispatch_seconds = 0.
            self.forward_target_update_seconds = 0.
            self.forward_dispatched_frames = 0
            pass_holder[:] = [self]

        def update_frame_targets(self, frame, *, update_source=True):
            """Update bound original targets without solving or changing poses."""
            source_frame = frame - self._num_frames_to_remove
            if self._ik_solver.pose_keyframes is not None:
                self._ik_solver.pose_keyframes.update(source_frame)
            hand_group = self._ik_solver.robot_hand_targets
            if hand_group is not None:
                target_frame = min(max(source_frame, 0), frames-1)
                hand_group.update(target_frame,
                    (object_controller.rotations if object_controller is not None
                     else schedule["object_reference_rotations"])[target_frame],
                    (object_controller.positions if object_controller is not None
                     else schedule["object_reference_positions"])[target_frame],
                    enabled=0 <= source_frame < frames)
            wrist_group = self._ik_solver.wrist_keyframes
            if wrist_group is not None:
                target_frame = min(max(source_frame, 0), frames-1)
                wrist_group.update(target_frame,
                    (object_controller.rotations if object_controller is not None
                     else schedule["object_reference_rotations"])[target_frame],
                    (object_controller.positions if object_controller is not None
                     else schedule["object_reference_positions"])[target_frame],
                    enabled=0 <= source_frame < frames)
            if self._ik_solver.collision_objective is not None:
                self._ik_solver.collision_objective.update(
                    schedule["collision_object_poses"][min(max(source_frame, 0), frames-1)],
                    enabled=0 <= source_frame < frames)
            for index, (position, normal) in enumerate(self._ik_solver.material_objectives):
                gain = schedule["gains"][source_frame, index] if 0 <= source_frame < frames else 0.
                weights = np.sqrt(gain * schedule["weights"][index])
                target_frame = min(max(source_frame, 0), frames-1)
                position.update(schedule["targets"][target_frame, index], weights[0])
                if normal is not None:
                    normal.update(-schedule["target_normals"][target_frame, index], weights[1])
            if update_source:
                for index in range(len(self._pos_objs)):
                    self._pos_objs[index].set_target_positions(self._input_positions[frame][index])
                    self._rot_objs[index].set_target_rotations(self._input_rotations[frame][index])

        def execute(self, frame):
            started = time.monotonic()
            # BasePass.execute sets the source targets once before its solve.
            self.update_frame_targets(frame, update_source=False)
            self.forward_target_update_seconds += time.monotonic()-started
            super().execute(frame)
            self.forward_dispatch_seconds += time.monotonic()-started
            self.forward_dispatched_frames += 1

    class ContactPipeline(original_pipeline):
        def measure_robot_hands(self, group, sequence):
            """Native mounted errors against the same live object-relative targets."""
            from scipy.spatial.transform import Rotation
            state = self.ik_model.state()
            rotations = (object_controller.rotations if object_controller is not None
                         else schedule["object_reference_rotations"])
            positions = (object_controller.positions if object_controller is not None
                         else schedule["object_reference_positions"])
            targets = {item["side"]: group._world(item["data"], rotations, positions)
                       for item in group.hands}
            rows = []
            for frame, q in enumerate(sequence):
                self.ik_model.joint_q.assign(q)
                newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
                transforms = state.body_q.numpy()
                for item in group.hands:
                    side, data = item["side"], item["data"]
                    tx = transforms[item["position"].link_index]
                    position, rotation = targets[side]
                    angle = Rotation.from_matrix(Rotation.from_quat(tx[3:]).as_matrix()
                                                @ rotation[frame].T).magnitude()
                    fingers = np.asarray(q)[item["finger"].coord_indices] - data["finger_qpos"][frame]
                    rows.append(dict(frame=frame, side=side,
                        object_relative=bool(data["contact_mask"][frame]),
                        wrist_translation_m=float(np.linalg.norm(tx[:3]-position[frame])),
                        wrist_rotation_rad=float(angle),
                        finger_rms_rad=float(np.sqrt(np.mean(fingers*fingers))),
                        finger_max_rad=float(np.max(np.abs(fingers)))))
            return rows

        def measure_wrist_keys(self, group, sequence):
            """Mounted FK expressed in the current object frame, all source times."""
            from scipy.spatial.transform import Rotation
            state = self.ik_model.state()
            rotations = object_controller.rotations if object_controller is not None else schedule["object_reference_rotations"]
            positions = object_controller.positions if object_controller is not None else schedule["object_reference_positions"]
            p, r = [], []
            for frame, q in enumerate(sequence):
                self.ik_model.joint_q.assign(q)
                newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
                tx = state.body_q.numpy()[[item["position"].link_index for item in group.hands]]
                p.append((tx[:, :3]-positions[frame]) @ rotations[frame])
                r.append(rotations[frame].T @ Rotation.from_quat(tx[:, 3:]).as_matrix())
            return np.asarray(p), np.asarray(r)

        def execute(self):
            execute_started = time.monotonic()
            diagnostic_seconds = 0.
            removed = self.num_initialization_frames + self.num_stabilization_frames
            if (len(self.input_targets) != 1 or len(self.input_targets[0]) - removed != frames
                    or not np.isclose(self.input_sample_rates[0], float(schedule["fps"]))):
                raise ValueError("Contact schedule does not match actual source frames/FPS")
            hand_target_report = None
            if hand_target_config is not None:
                from soma_hand_targets import smooth_hand_targets, hand_target_weight_multipliers
                original_targets = np.asarray(self.input_targets[0]).copy()
                derived_targets, hand_target_report = smooth_hand_targets(
                    original_targets, self.mapped_joints, self.input_sample_rates[0],
                    hand_target_config, first_source_frame=removed)
                pos_mult, rot_mult = hand_target_weight_multipliers(self.mapped_joints, hand_target_config)
                for attribute, multipliers in (("mapped_body_link_pos_data", pos_mult),
                                               ("mapped_body_link_rot_data", rot_mult)):
                    entries = getattr(self, attribute)
                    if len(entries) != len(multipliers):
                        raise ValueError("Hand source weights do not match mapped target rows")
                    setattr(self, attribute, [(link, weight * float(multiplier))
                        for (link, weight), multiplier in zip(entries, multipliers)])
                self.input_targets[0] = derived_targets
                priors_path = Path(report_path).with_name("hand_target_priors.npz")
                np.savez_compressed(priors_path, original_targets=original_targets,
                    derived_targets=derived_targets, mapped_joints=np.asarray(self.mapped_joints),
                    first_source_frame=np.asarray(removed), fps=np.asarray(self.input_sample_rates[0]))
                hand_target_report.update(path=str(priors_path),
                    applied_weight_multipliers={"position": pos_mult.tolist(), "rotation": rot_mult.tolist()},
                    original_source_priors_unchanged=not hand_target_report["actual_targets_changed"],
                    original_source_objective_weights_unchanged=bool(np.all(pos_mult == 1) and np.all(rot_mult == 1)),
                    solver_immutability_hash_basis="Executed derived targets; original target hashes recorded separately")
            robot_hand_report = None
            if robot_hand_target_config is not None:
                from soma_robot_hand_targets import validate_robot_hand_target_config
                cfg = validate_robot_hand_target_config(robot_hand_target_config)
                metadata = json.loads(Path(cfg["handoff_json"]).read_text())
                sides = cfg.get("enabled_hands", list(metadata["hands"]))
                multiplier = cfg["source_hand_weight_multiplier"]
                selected = [any(str(name).startswith(side.capitalize()+"Hand") for side in sides)
                            for name in self.mapped_joints]
                for side in sides:
                    if not any(str(name).startswith(side.capitalize()+"Hand") for name in self.mapped_joints):
                        raise ValueError(f"No original source hand mapping found for {side}")
                weight_records = {}
                for attribute in ("mapped_body_link_pos_data", "mapped_body_link_rot_data"):
                    entries = getattr(self, attribute)
                    if len(entries) != len(selected):
                        raise ValueError("Native hand source weight mapping has inconsistent rows")
                    weight_records[attribute] = [dict(source=str(name), original=float(weight),
                        effective=float(weight*(multiplier if replace else 1.0)))
                        for name, (_, weight), replace in zip(self.mapped_joints, entries, selected)]
                    setattr(self, attribute, [(link, weight*(multiplier if replace else 1.0))
                        for (link, weight), replace in zip(entries, selected)])
                robot_hand_report = dict(config=cfg, active_in_forward=True,
                    active_in_global=temporal_config is not None,
                    source_hand_weight_multiplier=multiplier,
                    source_mapping_weights=weight_records, original_body_weights_unchanged=True,
                    target_role="Unaccepted soft native hand trajectory; not contact truth")
            # The default limb stabilizer writes human wrist targets after IK.
            # This explicit opt-in mode keeps the original body IK objectives,
            # but disables the entire postpass (including foot plant corrections).
            self.post_processing_enabled = False
            forward_started = time.monotonic()
            buffers = super().execute()
            if self.ik_model.device.is_cuda:
                wp.synchronize_device(self.ik_model.device)
            forward_seconds = time.monotonic()-forward_started
            forward_pass = pass_holder[0]
            forward_execution = dict(device=str(self.ik_model.device),
                backend=("cuda_normal_equations_then_small_cholesky"
                         if forward_pass._ik_solver.forward_cuda_solver is not None
                         else "upstream_newton_lm"),
                pipeline_wall_seconds=forward_seconds,
                pass_setup_capture_seconds=forward_pass.forward_pass_setup_seconds,
                python_dispatch_seconds=forward_pass.forward_dispatch_seconds,
                python_extra_target_update_seconds=forward_pass.forward_target_update_seconds,
                dispatched_frames_including_warmup=forward_pass.forward_dispatched_frames,
                solver=forward_pass._ik_solver.forward_cuda_solver,
                timing_scope="Pipeline wall includes setup/capture, warmup and all source frames; dispatch counters are host wall time, not isolated GPU kernel timing")
            if robot_hand_report is not None:
                group = pass_holder[0]._ik_solver.robot_hand_targets
                robot_hand_report["binding"] = group.targets["report"]
                robot_hand_report["forward_seconds"] = forward_seconds
                forward_path = Path(report_path).with_name("forward_stage_states.npz")
                np.savez_compressed(forward_path, joint_q=np.asarray(buffers[0].data),
                    object_rotations=(object_controller.rotations if object_controller is not None
                                      else schedule["object_reference_rotations"]),
                    object_positions=(object_controller.positions if object_controller is not None
                                      else schedule["object_reference_positions"]),
                    fps=schedule["fps"], timestamps=schedule["object_timestamps"])
                robot_hand_report["forward_checkpoint"] = str(forward_path)
                robot_hand_report["forward_checkpoint_sha256"] = hashlib.sha256(forward_path.read_bytes()).hexdigest()
                diagnostic_started = time.monotonic()
                robot_hand_report["forward_observations"] = self.measure_robot_hands(group, buffers[0].data)
                diagnostic_seconds += time.monotonic()-diagnostic_started
            pose_key_report = None
            pose_keys = pass_holder[0]._ik_solver.pose_keyframes
            if pose_keys is not None:
                from soma_pose_keyframes import measure
                pose_key_forward = np.asarray(buffers[0].data)[:, pose_keys.coord_indices].copy()
                pose_key_report = dict(binding=pose_keys.loaded["report"], active_in_forward=True,
                    active_in_global=temporal_config is not None,
                    forward=measure(pose_keys.loaded, pose_key_forward))
            temporal_report = None
            wrist_key_report = None
            wrist_keys = pass_holder[0]._ik_solver.wrist_keyframes
            if wrist_keys is not None:
                from soma_wrist_keyframes import measure as measure_wrist
                wrist_forward_p, wrist_forward_r = self.measure_wrist_keys(wrist_keys, buffers[0].data)
                wrist_forward_q = np.asarray(buffers[0].data).copy()
                wrist_forward_obj_r = np.asarray(object_controller.rotations if object_controller is not None else schedule["object_reference_rotations"]).copy()
                wrist_forward_obj_p = np.asarray(object_controller.positions if object_controller is not None else schedule["object_reference_positions"]).copy()
                wrist_key_report = dict(binding=wrist_keys.loaded["report"], active_in_forward=True,
                    active_in_global=temporal_config is not None,
                    forward=measure_wrist(wrist_keys.loaded, wrist_forward_p, wrist_forward_r))
            if temporal_config is not None:
                temporal_report = refine_temporal_sequence(
                    self, pass_holder[0], buffers[0], temporal_config, report_path,
                    object_controller=object_controller, schedule=schedule)
            if wrist_key_report is not None:
                wrist_final_p, wrist_final_r = self.measure_wrist_keys(wrist_keys, buffers[0].data)
                wrist_key_report["final"] = measure_wrist(wrist_keys.loaded, wrist_final_p, wrist_final_r)
                wrist_path = Path(report_path).with_name("wrist_keyframe_states.npz")
                np.savez_compressed(wrist_path, forward_positions=wrist_forward_p, forward_rotations=wrist_forward_r,
                    final_positions=wrist_final_p, final_rotations=wrist_final_r,
                    forward_joint_q=wrist_forward_q, final_joint_q=np.asarray(buffers[0].data),
                    forward_obj_R=wrist_forward_obj_r, forward_obj_t=wrist_forward_obj_p,
                    final_obj_R=(object_controller.rotations if object_controller is not None else schedule["object_reference_rotations"]),
                    final_obj_t=(object_controller.positions if object_controller is not None else schedule["object_reference_positions"]),
                    hands=np.asarray(wrist_keys.loaded["report"]["enabled_hands"]), timestamps=np.arange(frames)/float(schedule["fps"]))
                wrist_key_report.update(states_path=str(wrist_path), states_sha256=hashlib.sha256(wrist_path.read_bytes()).hexdigest())
            if pose_key_report is not None:
                from soma_pose_keyframes import measure
                pose_key_final = np.asarray(buffers[0].data)[:, pose_keys.coord_indices].copy()
                pose_key_report["final"] = measure(pose_keys.loaded, pose_key_final)
                key_path = Path(report_path).with_name("pose_keyframe_states.npz")
                np.savez_compressed(key_path, forward_hinges=pose_key_forward, final_hinges=pose_key_final,
                    joint_names=np.asarray(pose_keys.names), timestamps=np.arange(frames)/float(schedule["fps"]))
                pose_key_report.update(states_path=str(key_path), states_sha256=hashlib.sha256(key_path.read_bytes()).hexdigest())
            if robot_hand_report is not None:
                diagnostic_started = time.monotonic()
                robot_hand_report["final_observations"] = self.measure_robot_hands(group, buffers[0].data)
                diagnostic_seconds += time.monotonic()-diagnostic_started
            # Measure AFTER the upstream final joint-limit clamp, not just before.
            diagnostic_started = time.monotonic()
            state = self.ik_model.state()
            labels = [get_name_from_label(x) for x in self.ik_model.body_label]
            links = [labels.index(str(x)) for x in schedule["body_names"]]
            observations = []
            collision_report = None
            if collision_enabled:
                collision = collision_holder[0]
                collision.check_queries()
                collision_report = dict(collision.metadata, active_in_same_solve=True, failed_queries=0, post_clamp=[])
            for frame, q in enumerate(buffers[0].data):
                self.ik_model.joint_q.assign(q)
                newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
                transforms = state.body_q.numpy()
                if collision_enabled:
                    collision.update(schedule["collision_object_poses"][frame])
                    collision.query(wp.array(transforms.reshape(1, -1, 7), dtype=wp.transform))
                    penalties = collision.penalties.numpy()[0]
                    collision_report["post_clamp"].append(dict(frame=frame,
                        violating_samples=int(np.count_nonzero(penalties > 0)),
                        max_clearance_deficit_m=float(penalties.max()),
                        rms_clearance_deficit_m=float(np.sqrt(np.mean(penalties**2)))))
                for i, link in enumerate(links):
                    if schedule["gains"][frame, i] <= 0:
                        continue
                    tx = wp.transform(*transforms[link])
                    point = np.asarray(wp.transform_point(tx, wp.vec3(*schedule["points"][i])))
                    normal = np.asarray(wp.transform_vector(tx, wp.vec3(*schedule["normals"][i])))
                    observations.append({"frame": frame, "name": str(schedule["names"][i]),
                        "gain": float(schedule["gains"][frame, i]),
                        "gap_m": float(np.linalg.norm(point-schedule["targets"][frame, i])),
                        "normal_error_deg": float(np.degrees(np.arccos(np.clip(
                            -normal @ schedule["target_normals"][frame, i], -1, 1))))})
            if collision_enabled:
                collision.check_queries()
            object_report = (export_object_result(self, buffers[0].data, object_controller,
                schedule, report_path, object_reference_sha256) if object_controller is not None else None)
            if (object_report is not None and object_controller.optimization_enabled
                    and temporal_config is not None and temporal_config.get("method") == "synchronous_sparse"):
                object_report["method"] = "Simultaneous full-trajectory robot and object sparse optimization"
            diagnostic_seconds += time.monotonic()-diagnostic_started
            Path(report_path).write_text(json.dumps({"status": "CONTACT_INITIALIZATION_UNREVIEWED",
                "same_soma_ik_solve": True, "device": str(wp.get_device()),
                "forward_execution": forward_execution,
                "diagnostic_wall_seconds": diagnostic_seconds,
                "pipeline_execution_wall_seconds": time.monotonic()-execute_started,
                "post_processing": "disabled", "frame_count": frames,
                "contact_loss_policy": contact_policy,
                "authored_contact_count": len(schedule["names"]),
                "position_objective_count": len(pass_holder[0]._ik_solver.material_objectives),
                "normal_objective_count": sum(normal is not None for _, normal in pass_holder[0]._ik_solver.material_objectives),
                "omitted_zero_normal_residual_rows": (int(3*np.count_nonzero(schedule["weights"][:, 1] == 0))
                                                      if contact_policy["enabled"] else 0),
                "observations_role": "diagnostic_only" if not contact_policy["enabled"] else "active_contact_objectives",
                "warmup_frames_without_contacts": removed,
                "measurements_after_joint_limit_clamp": True, "observations": observations,
                "collision_avoidance": collision_report,
                "temporal_optimization": temporal_report,
                "hand_target_optimization": hand_target_report,
                "robot_hand_targets": robot_hand_report,
                "pose_keyframes": pose_key_report,
                "wrist_keyframes": wrist_key_report,
                "object_optimization": object_report,
                "limitations": ["Authored targets, not predicted contact truth.",
                    "Soft objectives, not hard equality constraints or grasp acceptance.",
                    ("Sampled collision avoidance is enabled; original-triangle and subframe audit remain separate."
                     if collision_enabled else "No object/self collision objective; no original-mesh or subframe audit."),
                    "Full body IK may change root/legs; postpass foot stabilization is disabled."]}, indent=2)+"\n")
            return buffers

    pipeline.BaseIKSolver = ContactSolver
    pipeline.BasePass = ContactPass
    pipeline.SomaRetargetingPipeline = ContactPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--viewer", choices=["null"], default="null")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    from soma_temporal import default_forward_device
    device = (args.device if args.device is not None else
              default_forward_device(config.get("agentic_temporal_optimization")))
    root = Path(config["agentic_upstream_root"])
    sys.path.insert(0, str(root))
    from soma_runtime import install_scale_override
    records = install_scale_override(config.get("human_scale_ratio", 1.0),
                                     config["agentic_scale_report"], config["retarget_target"])
    with np.load(config["agentic_contacts"], allow_pickle=False) as data:
        schedule = {k: data[k].copy() for k in data.files}
    install_bridge(schedule, config["agentic_contact_report"], config.get("agentic_temporal_optimization"),
        config.get("agentic_object_optimization"), config.get("agentic_object_reference_sha256"),
        config.get("agentic_hand_target_optimization"), config.get("agentic_robot_hand_targets"),
        config.get("agentic_pose_keyframes"), config.get("agentic_wrist_keyframes"))
    script = root / "app/bvh_to_csv_converter.py"
    sys.argv = [str(script), "--config", args.config, "--viewer", args.viewer, "--device", device]
    runpy.run_path(str(script), run_name="__main__")
    if not records:
        raise RuntimeError("Official converter did not apply the configured SOMA scale")
    if not Path(config["agentic_contact_report"]).is_file():
        raise RuntimeError("Official converter did not execute the material contact bridge")


if __name__ == "__main__":
    main()
