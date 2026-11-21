"""All-frame sparse SOMA solve with original Newton frame objectives.

Frame residual/Jacobian evaluation may be serial. Every accepted nonlinear step
updates one full trajectory tangent vector, coupling time and robot/object DOFs.
No upstream solver, source target, original mesh or exported trajectory is edited.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
import warp as wp

from soma_sparse_sequence import solve_sparse_sequence
from soma_sequence_object import (build_object_layout, retract_object_poses,
                                 object_sequence_residual_jacobian)
from soma_temporal import (joint_weight_vector, temporal_metrics,
                           synchronous_evaluation_settings)


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def robot_temporal_rows(q, fps, config, coordinates, dofs, n_dofs, names):
    """Exact sparse global derivative residuals in native robot tangent DOFs."""
    q = np.asarray(q)
    coordinates, dofs = np.asarray(coordinates), np.asarray(dofs)
    weights = np.sqrt(joint_weight_vector(config, names))
    residuals, jacobians = [], []
    selector = sparse.coo_matrix((weights, (np.arange(len(dofs)), dofs)),
                                 shape=(len(dofs), n_dofs)).tocsr()
    for order, name in enumerate(("velocity", "acceleration", "jerk"), 1):
        gain = np.sqrt(config[name + "_weight"]) * fps ** order
        if not gain or len(q) <= order:
            continue
        coefficients = np.diff(np.eye(order + 1), n=order, axis=0).ravel()
        difference = sparse.diags([np.full(len(q)-order, c) for c in coefficients],
            np.arange(order+1), shape=(len(q)-order, len(q)), format="csr")
        residuals.append((gain * np.diff(q[:, coordinates].astype(float), n=order, axis=0)
                          * weights).ravel())
        jacobians.append(gain * sparse.kron(difference, selector, format="csr"))
    if not residuals:
        return np.zeros(0), sparse.csr_matrix((0, len(q)*n_dofs))
    return np.concatenate(residuals), sparse.vstack(jacobians, format="csr")


def grouped_object_collision_jacobian(collision, translation):
    """Derivative of the same grouped RMS clearance residual w.r.t. object SE3.

    Uses the query's signed-distance normal and frozen nearest topology, exactly
    as the original robot analytic Jacobian. No samples or mesh pairs omitted.
    """
    penalty = collision.penalties.numpy()[0].astype(float)
    normal = collision.gradients.numpy()[0].astype(float)
    point = collision.world_points.numpy()[0].astype(float)
    offsets = collision.offsets.numpy()
    weights = collision.query_weights_cpu
    mask = collision.object_query_mask
    derivative = -mask[:, None] * np.c_[normal, np.cross(point-translation, normal)]
    result = np.zeros((len(offsets)-1, 6))
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        total = np.dot(weights[start:end], penalty[start:end]**2)
        if total > 1e-20:
            result[i] = collision.weight * np.sum(
                (weights[start:end]*penalty[start:end])[:, None]*derivative[start:end], axis=0)/np.sqrt(total)
    return result


def native_limit_filter_data(objective):
    """Copy the immutable, final-forward limit settings once per sequence."""
    return (objective.active_dof_indices.numpy(), objective.dof_to_coord.numpy(),
            objective.joint_limit_lower.numpy(), objective.joint_limit_upper.numpy(),
            objective.coord_masks.numpy(), objective.offset_limit_lower.numpy(),
            objective.offset_limit_upper.numpy(), float(objective._weight.numpy()[0]))


def exact_native_limit_filter_jacobian(objective, q, jacobian, offset, data=None):
    """Differentiate upstream residual itself, replacing its constant approximation.

    This preserves every original joint-limit residual and its weight/mask.
    The upstream analytic derivative is constant even where this residual is 0.
    """
    active, mapping, lower, upper, mask, offset_lo, offset_hi, weight = (
        native_limit_filter_data(objective) if data is None else data)
    for row, dof in enumerate(active):
        coord = mapping[dof]
        center = (float(lower[dof])+float(upper[dof]))*.5
        lo = float(lower[dof])+float(offset_lo[coord])-center
        hi = float(upper[dof])-float(offset_hi[coord])-center
        error = float(q[coord])-center
        derivative = 0.
        if not lo < error <= hi:
            distance, sign = (lo-error, -1.) if error <= lo else (error-hi, 1.)
            power = distance**6.5
            exponential = np.exp(-power)
            derivative = weight*mask[coord]*(1.-exponential
                + error*exponential*6.5*distance**5.5*sign)
        jacobian[offset+row] = 0.
        jacobian[offset+row, dof] = derivative


class SynchronousProblem:
    def __init__(self, pipeline, pass_, initial, fps, config, controller, schedule, deadline):
        self.forward_device = str(pipeline.ik_model.device)
        self.evaluation_backend, evaluation_device = synchronous_evaluation_settings(config)
        self.device_setup_seconds = 0.
        if self.evaluation_backend == "cuda_graph":
            from soma_sequence_device import clone_evaluation_context
            tick = time.monotonic()
            pipeline, pass_ = clone_evaluation_context(
                pipeline, pass_, evaluation_device)
            self.device_setup_seconds = time.monotonic()-tick
        self.pipeline, self.pass_ = pipeline, pass_
        self.solver = pass_._ik_solver
        self.impl = self.solver.solver._impl
        self.initial = initial.copy()
        self.fps, self.config = float(fps), config
        self.controller, self.schedule = controller, schedule
        self.deadline = deadline
        self.frames, self.n_dofs = len(initial), self.impl.n_dofs
        self.robot_variables = self.frames*self.n_dofs
        self.object_variables = 0
        self.layout = None
        if controller is not None:
            self.layout = build_object_layout(controller.reference_rotations,
                controller.reference_positions, controller.config)
            self.object_variables = self.layout.variable_count
        self.variables = self.robot_variables+self.object_variables
        self.offsets = {id(o): int(off) for o, off in zip(self.impl.objectives, self.impl.residual_offsets)}
        self.relative_hand_temporal = None
        if "relative_hand_temporal" in config:
            from soma_relative_temporal import RelativeHandTemporal
            self.relative_hand_temporal = RelativeHandTemporal(
                getattr(self.solver, "robot_hand_targets", None), self.offsets, self.fps,
                config["relative_hand_temporal"], self.layout)
        self.world_hand_temporal = None
        if "world_hand_temporal" in config:
            from soma_world_temporal import WorldHandTemporal
            self.world_hand_temporal = WorldHandTemporal(
                getattr(self.solver, "robot_hand_targets", None), self.offsets, self.fps,
                config["world_hand_temporal"], self.layout)
        self.temporal = self.solver.temporal_objective
        self.temporal.disable()
        self.hinge_speed_before = None
        self.hinge_speed_report = None
        if "hinge_speed" in config:
            from soma_hinge_speed import hinge_speed_rows
            _, _, self.hinge_speed_before = hinge_speed_rows(self.initial,
                np.arange(self.frames)/self.fps, config["hinge_speed"], self.temporal.coord_indices,
                self.temporal.dof_indices, self.n_dofs, self.temporal.names,
                self.object_variables, jacobian=False)

        self.limit_filters = [(objective, int(offset), native_limit_filter_data(objective))
            for objective, offset in zip(self.impl.objectives, self.impl.residual_offsets)
            if type(objective).__name__ == "IKSmoothJointFilter"]
        self.cuda_evaluator = None
        self.evaluation_report = {}
        if self.evaluation_backend == "cuda_graph":
            from soma_sequence_cuda import CudaFrameEvaluator
            self.cuda_evaluator = CudaFrameEvaluator(self)
        self.problem_report = {}
        self.full_evaluations = 0
        self.full_linearizations = 0
        self.forward_target_hash = hashlib.sha256(
            np.ascontiguousarray(pipeline.input_targets[0]).tobytes()).hexdigest()
        self.linearization_path = None
        self.linearization_evidence = None
        self.refinement_tangent_mask = None
        self.refinement_coordinate_mask = None
        self.refinement_evidence = None

    def initial_state(self):
        state = {"q": self.initial.copy()}
        if self.controller is not None:
            state.update(obj_R=self.controller.rotations.copy(), obj_t=self.controller.positions.copy())
        return state

    def synchronize(self, state):
        if self.controller is not None:
            self.controller.rotations[...] = state["obj_R"]
            self.controller.positions[...] = state["obj_t"]
            self.controller.translation_offsets[...] = state["obj_t"]-self.controller.reference_positions
            self.controller.rotation_offsets[...] = Rotation.from_matrix(
                state["obj_R"]@self.controller.reference_rotations.transpose(0, 2, 1)).as_rotvec()
            self.controller.apply_schedule()

    def frame(self, state, frame, jacobian):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Global sequence evaluation reached cooperative deadline")
        self.pass_.update_frame_targets(frame+self.pass_._num_frames_to_remove)
        self.solver.joint_q.assign(state["q"][frame:frame+1])
        ctx = self.impl._ctx_solver(self.solver.joint_q)
        self.impl._residuals_analytic(ctx)
        residual = ctx.residuals.numpy()[0].astype(float)
        collision = self.solver.collision_objective
        if collision is not None:
            collision.check_queries()
        if not jacobian:
            return residual, None, None
        matrix = self.impl._jacobian_at(ctx).numpy()[0].astype(float)
        if collision is not None:
            collision.check_queries()
        self.fix_limit_jacobian(state, frame, matrix)
        object_matrix = self.object_frame_jacobian(state, frame, residual)
        return residual, matrix, object_matrix

    def fix_limit_jacobian(self, state, frame, matrix):
        for objective, offset, data in self.limit_filters:
            exact_native_limit_filter_jacobian(objective, state["q"][frame], matrix, offset, data)

    def object_frame_jacobian(self, state, frame, residual, collision_jacobian=None):
        collision = self.solver.collision_objective
        object_matrix = np.zeros((len(residual), 6))
        if self.object_variables:
            rotation = state["obj_R"][frame]
            for index, (position, normal) in enumerate(self.solver.material_objectives):
                gain = np.sqrt(self.schedule["gains"][frame, index]*self.schedule["weights"][index])
                point = rotation@self.controller.local_points[index]
                row = self.offsets[id(position)]
                object_matrix[row:row+3, :3] = gain[0]*np.eye(3)
                object_matrix[row:row+3, 3:] = -gain[0]*_skew(point)
                if normal is not None:
                    row = self.offsets[id(normal)]
                    object_matrix[row:row+3, 3:] = gain[1]*_skew(rotation@self.controller.local_normals[index])
            if collision is not None:
                row = self.offsets[id(collision)]
                object_matrix[row:row+collision.residual_dim()] = (
                    grouped_object_collision_jacobian(collision, state["obj_t"][frame])
                    if collision_jacobian is None else collision_jacobian)
            hand_targets = getattr(self.solver, "robot_hand_targets", None)
            if hand_targets is not None:
                # Use the same evaluated SO(3) log rows for the exact target
                # differential, including after a bulk CUDA transfer. No FK
                # round trip or frozen-object approximation is introduced.
                object_matrix += hand_targets.object_jacobian(
                    frame, rotation, residual, self.offsets)
            wrist_keys = getattr(self.solver, "wrist_keyframes", None)
            if wrist_keys is not None:
                object_matrix += wrist_keys.object_jacobian(frame, rotation, residual, self.offsets)
        return object_matrix

    def residual_jacobian(self, state, jacobian):
        self.synchronize(state)
        residuals, robot_blocks = [], []
        relative_samples = []
        world_samples = []
        object_row, object_col, object_data = [], [], []
        offset, coupling_nnz = 0, 0
        gpu_frames = None
        if self.cuda_evaluator is not None:
            gpu_frames = self.cuda_evaluator.evaluate(state, jacobian)
            self.evaluation_report = gpu_frames[3]
        for frame in range(self.frames):
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Global sequence evaluation reached cooperative deadline")
            if gpu_frames is None:
                residual, matrix, object_matrix = self.frame(state, frame, jacobian)
            else:
                residual = gpu_frames[0][frame].astype(float)
                matrix = gpu_frames[1][frame].astype(float) if jacobian else None
                object_matrix = None
                if jacobian:
                    self.fix_limit_jacobian(state, frame, matrix)
                    collision_jac = gpu_frames[2][frame] if gpu_frames[2] is not None else None
                    object_matrix = self.object_frame_jacobian(state, frame, residual, collision_jac)
            residuals.append(residual)
            if self.relative_hand_temporal is not None:
                reference = self.relative_hand_temporal.group.targets
                object_rotation = (state["obj_R"][frame] if self.controller is not None
                                   else reference["object_reference_rotations"][frame])
                object_position = (state["obj_t"][frame] if self.controller is not None
                                   else reference["object_reference_positions"][frame])
                relative_samples.append(self.relative_hand_temporal.sample(
                    frame, residual, matrix, object_rotation, object_position))
            if self.world_hand_temporal is not None:
                reference = self.world_hand_temporal.group.targets
                object_rotation = (state["obj_R"][frame] if self.controller is not None
                                   else reference["object_reference_rotations"][frame])
                object_position = (state["obj_t"][frame] if self.controller is not None
                                   else reference["object_reference_positions"][frame])
                world_samples.append(self.world_hand_temporal.sample(
                    frame, residual, matrix, object_rotation, object_position))
            if jacobian:
                robot_blocks.append(sparse.csr_matrix(matrix))
                if self.object_variables:
                    rr, cc = np.nonzero(object_matrix)
                    object_row.extend((rr+offset).tolist())
                    object_col.extend((cc+6*int(self.layout.frame_to_block[frame])).tolist())
                    object_data.extend(object_matrix[rr, cc].tolist())
                    # Count actual residual rows with both robot and object entries.
                    mixed = np.any(matrix != 0, axis=1)&np.any(object_matrix != 0, axis=1)
                    coupling_nnz += int(np.count_nonzero(object_matrix[mixed]))
            offset += len(residual)
        temporal_r, temporal_j = robot_temporal_rows(state["q"], self.fps, self.config,
            self.temporal.coord_indices, self.temporal.dof_indices, self.n_dofs, self.temporal.names)
        residuals.append(temporal_r)
        if self.controller is not None:
            object_r, object_j = object_sequence_residual_jacobian(state["obj_R"], state["obj_t"],
                self.controller.reference_rotations, self.controller.reference_positions,
                self.fps, self.controller.config, self.layout)
        else:
            object_r, object_j = np.zeros(0), sparse.csr_matrix((0, 0))
        residuals.append(object_r)
        relative_r, relative_j = np.zeros(0), None
        if self.relative_hand_temporal is not None:
            relative_r, relative_j = self.relative_hand_temporal.assemble(relative_samples, jacobian)
            residuals.append(relative_r)
        world_r, world_j = np.zeros(0), None
        if self.world_hand_temporal is not None:
            world_r, world_j = self.world_hand_temporal.assemble(world_samples, jacobian)
            residuals.append(world_r)
        speed_r, speed_j = np.zeros(0), None
        if "hinge_speed" in self.config:
            from soma_hinge_speed import hinge_speed_rows
            speed_r, speed_j, self.hinge_speed_report = hinge_speed_rows(state["q"],
                np.arange(self.frames)/self.fps, self.config["hinge_speed"],
                self.temporal.coord_indices, self.temporal.dof_indices, self.n_dofs,
                self.temporal.names, self.object_variables, jacobian=jacobian)
            residuals.append(speed_r)
        residual = np.concatenate(residuals)
        if not np.isfinite(residual).all():
            raise ValueError("Nonfinite complete trajectory residual")
        self.full_evaluations += 1
        if not jacobian:
            return residual
        frame_robot = sparse.block_diag(robot_blocks, format="csr")
        frame_object = sparse.coo_matrix((object_data, (object_row, object_col)),
            shape=(offset, self.object_variables)).tocsr()
        matrix = sparse.vstack([
            sparse.hstack([frame_robot, frame_object], format="csr"),
            sparse.hstack([temporal_j, sparse.csr_matrix((len(temporal_r), self.object_variables))], format="csr"),
            sparse.hstack([sparse.csr_matrix((len(object_r), self.robot_variables)), object_j], format="csr")], format="csr")
        if relative_j is not None:
            matrix = sparse.vstack([matrix, relative_j], format="csr")
        if world_j is not None:
            matrix = sparse.vstack([matrix, world_j], format="csr")
        if speed_j is not None:
            matrix = sparse.vstack([matrix, speed_j], format="csr")
        self.full_linearizations += 1
        self.problem_report = dict(variable_count=self.variables, robot_variable_count=self.robot_variables,
            object_variable_count=self.object_variables, residual_count=len(residual),
            jacobian_shape=list(matrix.shape), jacobian_nnz=int(matrix.nnz),
            cross_frame_temporal_nnz=int(temporal_j.nnz), robot_object_cross_nnz=coupling_nnz,
            frame_count=self.frames, robot_dofs_per_frame=self.n_dofs,
            robot_coordinate_count=state["q"].shape[1],
            source_frame_residuals=self.frames*self.impl.n_residuals,
            robot_temporal_residuals=len(temporal_r), object_reference_temporal_residuals=len(object_r),
            same_frame_objective_types=[type(x).__name__ for x in self.impl.objectives],
            native_robot_hand_targets_active=getattr(self.solver, "robot_hand_targets", None) is not None,
            root_policy=("Full Jacobian retains original native root tangent columns; scoped refinement fixes their increments to zero and keeps root coordinates byte exact." if self.refinement_tangent_mask is not None else "All original native root tangent DOFs remain variables; original temporal penalty covers43 scalar hinges."),
            joint_limit_derivative="Differentiate same upstream filter residual exactly; replace constant analytic approximation.",
            residual_evaluation=("CUDA graph original Newton/Warp frame residuals with bulk transfers; simultaneous CPU sparse tangent step."
                if self.cuda_evaluator is not None else
                "Serial original Newton/Warp frame residuals; simultaneous all-frame sparse tangent step."),
            execution_devices=dict(forward=self.forward_device, residual_jacobian=str(self.impl.device),
                global_sparse_linear_solve="cpu", evaluation_backend=self.evaluation_backend),
            device_setup_seconds=self.device_setup_seconds,
            evaluation_telemetry=self.evaluation_report)
        if self.relative_hand_temporal is not None:
            self.problem_report["relative_hand_temporal"] = dict(
                self.relative_hand_temporal.report, jacobian_nnz=int(relative_j.nnz),
                robot_jacobian_nnz=int(relative_j[:, :self.robot_variables].nnz),
                object_jacobian_nnz=int(relative_j[:, self.robot_variables:].nnz),
                row_start=offset+len(temporal_r)+len(object_r), row_stop=len(residual)-len(world_r)-len(speed_r))
        if self.world_hand_temporal is not None:
            self.problem_report["world_hand_temporal"] = dict(
                self.world_hand_temporal.report, jacobian_nnz=int(world_j.nnz),
                robot_jacobian_nnz=int(world_j[:, :self.robot_variables].nnz),
                object_jacobian_nnz=int(world_j[:, self.robot_variables:].nnz),
                row_start=len(residual)-len(world_r)-len(speed_r), row_stop=len(residual)-len(speed_r))
        if "hinge_speed" in self.config:
            self.problem_report["hinge_speed"] = dict(self.hinge_speed_report,
                row_start=len(residual)-len(speed_r), row_stop=len(residual),
                jacobian_nnz=int(speed_j.nnz), object_jacobian_nnz=0,
                scalar_coordinate_indices=self.temporal.coord_indices.tolist(),
                scalar_dof_indices=self.temporal.dof_indices.tolist())
        if self.linearization_path is not None and self.linearization_evidence is None:
            np.savez_compressed(self.linearization_path, residual=residual, data=matrix.data,
                indices=matrix.indices, indptr=matrix.indptr, shape=np.asarray(matrix.shape),
                joint_q=state["q"], obj_R=state.get("obj_R", np.zeros((0, 3, 3))),
                obj_t=state.get("obj_t", np.zeros((0, 3))),
                robot_dofs_per_frame=np.asarray(self.n_dofs),
                frame_residual_count=np.asarray(self.impl.n_residuals),
                frame_count=np.asarray(self.frames), robot_variable_count=np.asarray(self.robot_variables),
                object_variable_count=np.asarray(self.object_variables),
                temporal_row_start=np.asarray(offset), temporal_row_stop=np.asarray(offset+len(temporal_r)),
                object_frame_to_block=(self.layout.frame_to_block if self.layout is not None
                                       else np.full(self.frames, -1)),
                scalar_coordinate_indices=self.temporal.coord_indices,
                scalar_dof_indices=self.temporal.dof_indices,
                **(self.relative_hand_temporal.evidence() if self.relative_hand_temporal is not None else {}),
                **(self.world_hand_temporal.evidence() if self.world_hand_temporal is not None else {}))
            self.linearization_evidence = dict(path=str(self.linearization_path),
                sha256=hashlib.sha256(self.linearization_path.read_bytes()).hexdigest(),
                state="initial_trajectory_before_first_global_step")
        return residual, matrix

    def evaluate(self, state):
        return self.residual_jacobian(state, False)

    def linearize(self, state):
        return self.residual_jacobian(state, True)

    def retract(self, state, increment):
        increment = np.asarray(increment, dtype=float)
        if increment.shape != (self.variables,) or not np.isfinite(increment).all():
            raise ValueError("Invalid complete trajectory tangent increment")
        if self.refinement_tangent_mask is not None and np.any(increment[~self.refinement_tangent_mask] != 0):
            raise ValueError("Sparse solve attempted to change a fixed refinement variable")
        q = wp.array(state["q"], dtype=float, device=self.impl.device)
        dq = wp.array(increment[:self.robot_variables].reshape(self.frames, self.n_dofs).astype(np.float32),
                      dtype=float, device=self.impl.device)
        proposed = wp.empty_like(q)
        qd = wp.zeros((self.frames, self.n_dofs), dtype=float, device=self.impl.device)
        with wp.ScopedDevice(self.impl.device):
            self.impl._integrate_dq(q, dq_in=dq, joint_q_out=proposed, joint_qd_out=qd)
            self.pipeline.joint_limit_clamper.apply(proposed)
        candidate = {"q": proposed.numpy()}
        if self.refinement_coordinate_mask is not None:
            # Native integration/clamping may normalize an unchanged free root.
            # Restore every fixed coordinate exactly, not approximately.
            fixed = ~self.refinement_coordinate_mask
            candidate["q"][fixed] = state["q"][fixed]
        if self.controller is not None:
            if self.refinement_tangent_mask is not None:
                rotation, translation = state["obj_R"].copy(), state["obj_t"].copy()
            else:
                rotation, translation = retract_object_poses(state["obj_R"], state["obj_t"],
                    increment[self.robot_variables:], self.layout)
            candidate.update(obj_R=rotation, obj_t=translation)
        return candidate


def refine_synchronous_sequence(pipeline, pass_, buffer, config, report_path,
                                object_controller=None, schedule=None):
    started = time.monotonic()
    deadline = started+config["max_seconds"]
    for name in ("RETARGET_STOP_REPAIRS_UNIX", "RETARGET_AGENT_DEADLINE_UNIX", "RETARGET_ALLOCATION_DEADLINE_UNIX"):
        if name in os.environ:
            deadline = min(deadline, started+float(os.environ[name])-time.time()-30.)
    problem = SynchronousProblem(pipeline, pass_, np.asarray(buffer.data), buffer.sample_rate,
        config, object_controller, schedule, deadline)
    problem.linearization_path = Path(report_path).with_name("synchronous_linearization.npz")
    initial = problem.initial_state()
    seed_report = None
    if "prior_global_seed" in config:
        from soma_prior_global_seed import load as load_prior_seed, checked_state
        if object_controller is None:
            raise ValueError("Prior-global seed requires explicit object controller")
        loaded = load_prior_seed(config["prior_global_seed"], frame_count=problem.frames,
            fps=problem.fps, reference_rotations=object_controller.reference_rotations,
            reference_positions=object_controller.reference_positions,
            object_scale=schedule["object_scale"], object_config=object_controller.config)
        fresh = initial
        initial = checked_state(loaded, fresh, coordinates=problem.temporal.coord_indices,
            dofs=problem.temporal.dof_indices, names=problem.temporal.names,
            source_target_sha256=problem.forward_target_hash)
        problem.initial = initial["q"].copy()
        if "hinge_speed" in config:
            from soma_hinge_speed import hinge_speed_rows
            _, _, problem.hinge_speed_before = hinge_speed_rows(initial["q"],
                np.arange(problem.frames)/problem.fps, config["hinge_speed"],
                problem.temporal.coord_indices, problem.temporal.dof_indices,
                problem.n_dofs, problem.temporal.names, problem.object_variables, jacobian=False)
        seed_path = Path(report_path).with_name("prior_global_seed_states.npz")
        np.savez_compressed(seed_path, **{'fresh_'+k:v for k,v in fresh.items()},
            **{'seed_'+k:v for k,v in initial.items()}, timestamps=np.arange(problem.frames)/problem.fps)
        seed_report = dict(binding=loaded["config"], input_files=loaded["files"],
            states_path=str(seed_path), states_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
            fresh_forward_executed=True, global_initial_state="prior_global_checkpoint",
            original_targets_unchanged=True, motion_accepted=False)
    coords = problem.temporal.coord_indices
    if "refinement_scope" in config:
        from soma_refinement_scope import resolve, check_fixed
        mask, coordinate_mask, evidence = resolve(config["refinement_scope"],
            frames=problem.frames, dofs_per_frame=problem.n_dofs,
            coordinate_count=initial["q"].shape[1], coordinates=coords,
            dofs=problem.temporal.dof_indices, names=problem.temporal.names,
            object_variables=problem.object_variables)
        problem.refinement_tangent_mask = mask
        problem.refinement_coordinate_mask = coordinate_mask
        problem.refinement_evidence = evidence
    before = temporal_metrics(initial["q"][:, coords], problem.fps, config, joint_names=problem.temporal.names)
    checkpoint = Path(report_path).with_name("temporal_sequence_states.npz")
    history_path = Path(report_path).with_name("synchronous_optimization.json")

    accepted_count = 0

    def checkpoint_state(state, record):
        nonlocal accepted_count
        accepted_count += 1
        extra = {}
        if object_controller is not None:
            extra = dict(initial_obj_R=initial["obj_R"], initial_obj_t=initial["obj_t"],
                         refined_obj_R=state["obj_R"], refined_obj_t=state["obj_t"])
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, initial_joint_q=initial["q"], refined_joint_q=state["q"],
                fps=problem.fps, coordinate_indices=coords, joint_names=np.asarray(problem.temporal.names),
                completed_global_iterations=np.asarray(accepted_count), **extra)
        temporary.replace(checkpoint)
        print("[SYNCHRONOUS] accepted full-trajectory iteration="+str(accepted_count)
              +" energy="+str(record.get("energy_after", record.get("energy"))), flush=True)

    solver_config = dict(config["synchronous_solver"])
    solver_config["max_seconds"] = min(solver_config["max_seconds"], max(.001, deadline-started))
    scope_args = ({} if problem.refinement_tangent_mask is None else
                  dict(active_columns=np.flatnonzero(problem.refinement_tangent_mask)))
    state, solver_report = solve_sparse_sequence(initial, problem.linearize, problem.evaluate,
        problem.retract, solver_config, deadline=deadline, callback=checkpoint_state, **scope_args)
    accepted = solver_report.get("completed_global_iterations", solver_report.get("accepted_steps", 0))
    if accepted < 1:
        raise RuntimeError("Synchronous trajectory solver completed no accepted global update: "
                           + str(solver_report.get("stop_reason")))
    problem.synchronize(state)
    if problem.relative_hand_temporal is not None or problem.world_hand_temporal is not None or "hinge_speed" in config:
        # Explicit final-state metrics, rather than a rejected line-search trial.
        problem.evaluate(state)
    if object_controller is not None and not object_controller.report()["passed"]:
        raise RuntimeError("Synchronous object trajectory violates original static/shared-pose/corridor policy")
    current_hash = hashlib.sha256(np.ascontiguousarray(pipeline.input_targets[0]).tobytes()).hexdigest()
    if current_hash != problem.forward_target_hash:
        raise RuntimeError("Synchronous solver changed original human reference targets")
    buffer.data[...] = state["q"]
    accepted_count = accepted-1
    checkpoint_state(state, {"energy": solver_report.get("final_energy")})
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = dict(schema="soma_synchronous_trajectory.v1", method="synchronous_sparse",
        same_soma_objectives=True, source_targets_unchanged=True, frame_count=problem.frames,
        fps=problem.fps, config=config, completed_sweeps=0, completed_joint_sweeps=0,
        completed_global_iterations=accepted, completed_robot_only_sweeps=0,
        history=solver_report["history"], solver_report=solver_report, global_problem=problem.problem_report,
        object_history=[], object_motion_policy=object_controller.motion_policy if object_controller is not None else None,
        before=before, after=temporal_metrics(state["q"][:, coords], problem.fps, config, joint_names=problem.temporal.names),
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, seconds=time.monotonic()-started,
        full_objective_evaluations=problem.full_evaluations, full_jacobian_evaluations=problem.full_linearizations,
        linearization_evidence=problem.linearization_evidence,
        original_target_sha256=current_hash,
        limitations=["Finite simultaneous sparse iterations do not establish a global nonlinear minimum.",
            "Original collision query coverage/estimated-contact limitations remain; no motion acceptance.",
            "Native scalar hinges retain original temporal objective; free-root smoothness is not newly added."])
    if seed_report is not None:
        report["prior_global_seed"] = seed_report
    if problem.refinement_evidence is not None:
        report["refinement_scope"] = dict(problem.refinement_evidence,
            fixed_state_byte_exact=check_fixed(initial, state, problem.refinement_coordinate_mask))
        report["limitations"].append("This refinement solves only the declared native hinge/frame subset while retaining every original residual row and boundary stencil. Root/object/unselected coordinates remain exact; stationarity is constrained to that subset. Mesh collision review remains necessary.")
    if problem.relative_hand_temporal is not None:
        report["relative_hand_temporal"] = dict(before=problem.relative_hand_temporal.before_report,
                                               after=problem.relative_hand_temporal.report)
    if problem.world_hand_temporal is not None:
        report["world_hand_temporal"] = dict(before=problem.world_hand_temporal.before_report,
                                            after=problem.world_hand_temporal.report)
        report["limitations"][2] = "World wrist derivative rows include native free-root tangents; they do not directly regularize every free-root DOF."
    if "hinge_speed" in config:
        report["hinge_speed"] = dict(before=problem.hinge_speed_before,
                                     after=problem.hinge_speed_report)
    history_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    report["solver_report_path"] = str(history_path)
    report["solver_report_sha256"] = hashlib.sha256(history_path.read_bytes()).hexdigest()
    print("[SYNCHRONOUS] completed global iterations="+str(accepted)+" variables="+str(problem.variables), flush=True)
    return report
