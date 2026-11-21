"""CUDA graph evaluator for the original single-frame SOMA objectives.

This module evaluates the existing Newton residuals and analytic Jacobians. It
does not run Newton's dense LM kernel or choose/accept any optimizer step. The
global trajectory, temporal rows, object reference and sparse LM/LSMR remain
owned by :mod:`soma_sequence_runtime`.
"""
import time

import numpy as np
import warp as wp


@wp.kernel
def _gather_q(sequence: wp.array2d(dtype=float), frame: wp.array(dtype=int),
              output: wp.array2d(dtype=float)):
    column = wp.tid()
    output[0, column] = sequence[frame[0], column]


@wp.kernel
def _gather_position(sequence: wp.array2d(dtype=wp.vec3), frame: wp.array(dtype=int),
                     objective: int, output: wp.array(dtype=wp.vec3)):
    output[0] = sequence[frame[0], objective]


@wp.kernel
def _gather_rotation(sequence: wp.array2d(dtype=wp.vec4), frame: wp.array(dtype=int),
                     objective: int, output: wp.array(dtype=wp.vec4)):
    output[0] = sequence[frame[0], objective]


@wp.kernel
def _gather_material(targets: wp.array2d(dtype=wp.vec3),
                     gains: wp.array2d(dtype=float), frame: wp.array(dtype=int),
                     objective: int, output: wp.array(dtype=wp.vec3),
                     weight: wp.array(dtype=float)):
    output[0] = targets[frame[0], objective]
    weight[0] = gains[frame[0], objective]


@wp.kernel
def _gather_collision(poses: wp.array(dtype=wp.transform), frame: wp.array(dtype=int),
                      output: wp.array(dtype=wp.transform), enabled: wp.array(dtype=float)):
    output[0] = poses[frame[0]]
    enabled[0] = 1.0


@wp.kernel
def _store_residual(source: wp.array2d(dtype=float), frame: wp.array(dtype=int),
                    output: wp.array2d(dtype=float)):
    row = wp.tid()
    output[frame[0], row] = source[0, row]


@wp.kernel
def _store_jacobian(source: wp.array3d(dtype=float), frame: wp.array(dtype=int),
                    output: wp.array3d(dtype=float)):
    row, column = wp.tid()
    output[frame[0], row, column] = source[0, row, column]


@wp.kernel
def _advance_frame(frame: wp.array(dtype=int)):
    frame[0] = frame[0] + 1


@wp.kernel
def _grouped_object_jacobian(penalty: wp.array2d(dtype=float),
                             normals: wp.array2d(dtype=wp.vec3),
                             points: wp.array2d(dtype=wp.vec3),
                             offsets: wp.array(dtype=int), weights: wp.array(dtype=float),
                             masks: wp.array(dtype=int), translations: wp.array(dtype=wp.vec3d),
                             frame: wp.array(dtype=int), weight: wp.float64,
                             output: wp.array3d(dtype=wp.float64)):
    """Same grouped RMS derivative as the CPU path, without query-array copies.

    Accumulate in float64 to match the CPU helper's float32-query -> float64
    arithmetic. Both object query directions use the original signed mask.
    """
    group = wp.tid()
    index = frame[0]
    translation = translations[index]
    total = wp.float64(0.0)
    x = wp.float64(0.0)
    y = wp.float64(0.0)
    z = wp.float64(0.0)
    rx = wp.float64(0.0)
    ry = wp.float64(0.0)
    rz = wp.float64(0.0)
    for query in range(offsets[group], offsets[group+1]):
        p = wp.float64(penalty[0, query])
        query_weight = wp.float64(weights[query])
        total += query_weight*p*p
        coefficient = -wp.float64(masks[query])*query_weight*p
        normal = normals[0, query]
        point = points[0, query]
        nx = wp.float64(normal[0])
        ny = wp.float64(normal[1])
        nz = wp.float64(normal[2])
        px = wp.float64(point[0])-translation[0]
        py = wp.float64(point[1])-translation[1]
        pz = wp.float64(point[2])-translation[2]
        x += coefficient*nx
        y += coefficient*ny
        z += coefficient*nz
        rx += coefficient*(py*nz-pz*ny)
        ry += coefficient*(pz*nx-px*nz)
        rz += coefficient*(px*ny-py*nx)
    factor = wp.float64(0.0)
    if total > wp.float64(1.0e-20):
        factor = weight/wp.sqrt(total)
    output[index, group, 0] = factor*x
    output[index, group, 1] = factor*y
    output[index, group, 2] = factor*z
    output[index, group, 3] = factor*rx
    output[index, group, 4] = factor*ry
    output[index, group, 5] = factor*rz


class CudaFrameEvaluator:
    """Replay one bound single-frame graph and transfer whole trajectory arrays.

    ``problem`` must already own a CUDA single-environment analytic Newton
    implementation with the exact same objective layout as its CPU control.
    Creating that implementation, and keeping fresh forward initialization on
    CPU, is deliberately outside this evaluator. ``evaluate`` returns
    ``(residuals[T,R], robot_jacobian[T,R,D] | None,
    object_collision_jacobian[T,G,6] | None, telemetry)``. The robot Jacobian is
    Newton's raw analytic result: the caller retains its exact native joint
    limit derivative correction, as well as the material object Jacobians.
    """

    def __init__(self, problem):
        self.problem = problem
        self.impl, self.solver = problem.impl, problem.solver
        self.device = wp.get_device(self.impl.device)
        if not self.device.is_cuda or self.impl.n_batch != 1:
            raise ValueError("CUDA frame evaluator requires one CUDA environment")
        if str(getattr(self.impl.jacobian_mode, "name", self.impl.jacobian_mode)).upper() != "ANALYTIC":
            raise ValueError("CUDA frame evaluator requires analytic Jacobians")
        self.frames, self.rows, self.dofs = problem.frames, self.impl.n_residuals, self.impl.n_dofs
        self.collision = self.solver.collision_objective
        self.groups = self.collision.residual_dim() if self.collision is not None else 0
        self.materials = self.solver.material_objectives
        self.robot_hand_targets = getattr(self.solver, "robot_hand_targets", None)
        self.pose_keyframes = getattr(self.solver, "pose_keyframes", None)
        self.wrist_keyframes = getattr(self.solver, "wrist_keyframes", None)
        self.graphs = {}
        self.captures = {}
        self.calls = 0
        self.warmup_seconds = 0.0
        self.capture_seconds = 0.0
        self.setup_started = time.monotonic()
        with wp.ScopedDevice(self.device):
            self.frame_index = wp.zeros(1, dtype=int)
            self.q = wp.empty((self.frames, self.impl.n_coords), dtype=float)
            self.residuals = wp.empty((self.frames, self.rows), dtype=float)
            self.jacobian = wp.empty((self.frames, self.rows, self.dofs), dtype=float)
            self.context = self.impl._ctx_solver(self.solver.joint_q)
            removed = problem.pass_._num_frames_to_remove
            positions = problem.pass_._input_positions.numpy()[removed:removed+self.frames]
            rotations = problem.pass_._input_rotations.numpy()[removed:removed+self.frames]
            if positions.ndim != 4 or positions.shape[2:] != (1, 3):
                raise ValueError("Unexpected original SOMA position target layout")
            if rotations.ndim != 4 or rotations.shape[2:] != (1, 4):
                raise ValueError("Unexpected original SOMA rotation target layout")
            if len(positions) != self.frames or len(rotations) != self.frames:
                raise ValueError("Original source target timeline is incomplete")
            self.source_positions = wp.array(np.ascontiguousarray(positions[:, :, 0]), dtype=wp.vec3)
            self.source_rotations = wp.array(np.ascontiguousarray(rotations[:, :, 0]), dtype=wp.vec4)
            count = len(self.materials)
            self.material_targets = wp.empty((self.frames, count), dtype=wp.vec3)
            self.material_normals = wp.empty((self.frames, count), dtype=wp.vec3)
            self.material_position_gains = wp.empty((self.frames, count), dtype=float)
            self.material_normal_gains = wp.empty((self.frames, count), dtype=float)
            self.object_poses = wp.empty(self.frames, dtype=wp.transform)
            self.object_translations = wp.empty(self.frames, dtype=wp.vec3d)
            self.object_collision_jacobian = None
            self.object_query_masks = None
            if self.collision is not None and problem.object_variables:
                self.object_query_masks = wp.array(self.collision.object_query_mask.astype(np.int32), dtype=int)
                self.object_collision_jacobian = wp.empty((self.frames, self.groups, 6), dtype=wp.float64)
        self.setup_seconds = time.monotonic()-self.setup_started

    def _stage(self, state):
        self.problem.synchronize(state)
        schedule = self.problem.schedule
        self.q.assign(np.asarray(state["q"], dtype=np.float32))
        if self.materials:
            gains = np.sqrt(np.asarray(schedule["gains"])[:, :, None]
                            * np.asarray(schedule["weights"])[None, :, :])
            self.material_targets.assign(np.asarray(schedule["targets"], dtype=np.float32))
            self.material_normals.assign(np.asarray(-schedule["target_normals"], dtype=np.float32))
            self.material_position_gains.assign(np.asarray(gains[:, :, 0], dtype=np.float32))
            self.material_normal_gains.assign(np.asarray(gains[:, :, 1], dtype=np.float32))
        if self.robot_hand_targets is not None:
            # Only contact-frame wrists follow the live object. The group
            # retains original world targets outside their declared phases.
            self.robot_hand_targets.stage(
                state.get("obj_R", schedule["object_reference_rotations"]),
                state.get("obj_t", schedule["object_reference_positions"]))
        if self.wrist_keyframes is not None:
            self.wrist_keyframes.stage(
                state.get("obj_R", schedule["object_reference_rotations"]),
                state.get("obj_t", schedule["object_reference_positions"]))
        if self.collision is not None:
            poses = np.asarray(schedule["collision_object_poses"])
            if poses.shape != (self.frames, 7):
                raise ValueError("Collision object poses do not match the original source timeline")
            self.object_poses.assign(poses.astype(np.float32))
            if self.object_collision_jacobian is not None:
                self.object_translations.assign(np.asarray(state["obj_t"], dtype=np.float64))

    def _frame_operations(self, jacobian):
        wp.launch(_gather_q, dim=self.impl.n_coords,
                  inputs=[self.q, self.frame_index, self.solver.joint_q], device=self.device)
        for index, objective in enumerate(self.problem.pass_._pos_objs):
            wp.launch(_gather_position, dim=1,
                      inputs=[self.source_positions, self.frame_index, index, objective.target_positions],
                      device=self.device)
        for index, objective in enumerate(self.problem.pass_._rot_objs):
            wp.launch(_gather_rotation, dim=1,
                      inputs=[self.source_rotations, self.frame_index, index, objective.target_rotations],
                      device=self.device)
        for index, (position, normal) in enumerate(self.materials):
            wp.launch(_gather_material, dim=1, inputs=[self.material_targets,
                      self.material_position_gains, self.frame_index, index,
                      position.target_positions, position.dynamic_weight], device=self.device)
            if normal is not None:
                wp.launch(_gather_material, dim=1, inputs=[self.material_normals,
                          self.material_normal_gains, self.frame_index, index,
                          normal.target_positions, normal.dynamic_weight], device=self.device)
        if self.collision is not None:
            wp.launch(_gather_collision, dim=1, inputs=[self.object_poses, self.frame_index,
                      self.collision.object_pose, self.collision.enabled], device=self.device)
        if self.robot_hand_targets is not None:
            self.robot_hand_targets.gather(self.frame_index)
        if self.pose_keyframes is not None:
            self.pose_keyframes.gather(self.frame_index)
        if self.wrist_keyframes is not None:
            self.wrist_keyframes.gather(self.frame_index)
        self.impl._residuals_analytic(self.context)
        if jacobian:
            self.impl._jacobian_at(self.context)
            wp.launch(_store_jacobian, dim=(self.rows, self.dofs),
                      inputs=[self.context.jacobian_out, self.frame_index, self.jacobian], device=self.device)
            if self.object_collision_jacobian is not None:
                collision = self.collision
                wp.launch(_grouped_object_jacobian, dim=self.groups,
                          inputs=[collision.penalties, collision.gradients, collision.world_points,
                                  collision.offsets, collision.query_weights, self.object_query_masks,
                                  self.object_translations, self.frame_index, wp.float64(collision.weight),
                                  self.object_collision_jacobian], device=self.device)
        wp.launch(_store_residual, dim=self.rows,
                  inputs=[self.context.residuals, self.frame_index, self.residuals], device=self.device)
        wp.launch(_advance_frame, dim=1, inputs=[self.frame_index], device=self.device)

    def _graph(self, jacobian):
        if jacobian not in self.graphs:
            started = time.monotonic()
            self.frame_index.zero_()
            self._frame_operations(jacobian)
            wp.synchronize_device(self.device)
            self.warmup_seconds += time.monotonic()-started
            started = time.monotonic()
            self.frame_index.zero_()
            with wp.ScopedCapture(device=self.device) as capture:
                self._frame_operations(jacobian)
            self.graphs[jacobian] = capture.graph
            self.capture_seconds += time.monotonic()-started
            self.captures[jacobian] = True
        return self.graphs[jacobian]

    def evaluate(self, state, jacobian):
        """Evaluate every original source frame, then perform deferred bulk reads."""
        started = time.monotonic()
        with wp.ScopedDevice(self.device):
            stage_started = time.monotonic()
            self._stage(state)
            stage_seconds = time.monotonic()-stage_started
            graph = self._graph(bool(jacobian))
            self.frame_index.zero_()
            if self.collision is not None:
                self.collision.failures.zero_()
            replay_started = time.monotonic()
            for _ in range(self.frames):
                if time.monotonic() >= self.problem.deadline:
                    wp.synchronize_device(self.device)
                    raise TimeoutError("CUDA sequence evaluation reached cooperative deadline")
                wp.capture_launch(graph)
            wp.synchronize_device(self.device)
            replay_seconds = time.monotonic()-replay_started
            transfer_started = time.monotonic()
            # Query failures are sticky across all source frames of this exact
            # evaluation. Never drop failures because a subsequent frame works.
            failures = int(self.collision.failures.numpy()[0]) if self.collision is not None else 0
            if failures:
                raise RuntimeError(f"CUDA collision winding query failed ({failures} queries)")
            residuals = self.residuals.numpy().astype(float)
            robot_jacobian = self.jacobian.numpy().astype(float) if jacobian else None
            object_jacobian = (self.object_collision_jacobian.numpy() if jacobian
                               and self.object_collision_jacobian is not None else None)
            transfer_seconds = time.monotonic()-transfer_started
        arrays = [residuals, robot_jacobian, object_jacobian]
        if not all(array is None or np.isfinite(array).all() for array in arrays):
            raise RuntimeError("Nonfinite CUDA trajectory residual/Jacobian")
        self.calls += 1
        metadata = dict(backend="cuda_graph_single_context", device=str(self.device),
                        frame_count=self.frames, residual_count_per_frame=self.rows,
                        robot_dofs_per_frame=self.dofs, collision_residual_groups=self.groups,
                        collision_query_failures=failures, graph_launches=self.frames,
                        per_frame_host_copies=0, full_batch_evaluations=self.calls,
                        stage_seconds=stage_seconds, replay_seconds=replay_seconds,
                        transfer_seconds=transfer_seconds, elapsed_seconds=time.monotonic()-started,
                        setup_seconds=self.setup_seconds, warmup_seconds=self.warmup_seconds,
                        capture_seconds=self.capture_seconds,
                        dense_newton_lm_launched=False, global_linear_solver="cpu_scipy_lsmr",
                        native_limit_derivative_patch="caller_after_bulk_transfer",
                        material_object_jacobian="caller_after_bulk_transfer",
                        native_robot_hand_targets_active=self.robot_hand_targets is not None,
                        native_hand_object_jacobian="exact_SO3_log_from_bulk_residuals")
        return residuals, robot_jacobian, object_jacobian, metadata
