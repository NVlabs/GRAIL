"""Device-local SOMA objective evaluation without capturing a forward IK solve.

The original CPU forward pass remains untouched.  This module rebuilds its same
model and objectives on another Warp device, but deliberately never constructs a
BasePass: that constructor eagerly captures Newton's dense tiled LM kernel.
"""
from contextlib import contextmanager
from copy import copy
import hashlib
import json

import numpy as np
import warp as wp


def _array_hash(value):
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _model_binding(model):
    """Topology and limits must agree; the current pose is evaluation state."""
    names = ("joint_type", "joint_parent", "joint_child", "joint_q_start",
             "joint_qd_start", "joint_dof_dim", "joint_axis", "joint_X_p",
             "joint_X_c", "joint_limit_lower", "joint_limit_upper")
    return dict(counts=[model.body_count, model.joint_count, model.joint_coord_count,
                        model.joint_dof_count],
                body_labels=list(map(str, model.body_label)),
                joint_labels=list(map(str, model.joint_label)),
                arrays={name: _array_hash(getattr(model, name)) for name in names})


def _source_binding(pipeline, pass_):
    return dict(targets=[_array_hash(value) for value in pipeline.input_targets],
                sample_rates=list(map(float, pipeline.input_sample_rates)),
                mapped_joints=list(map(str, pipeline.mapped_joints)),
                position_mapping=[(int(link), float(weight))
                                  for link, weight in pipeline.mapped_body_link_pos_data],
                rotation_mapping=[(int(link), float(weight))
                                  for link, weight in pipeline.mapped_body_link_rot_data],
                masks=_array_hash(pipeline.smooth_joint_filter_coord_masks),
                offsets=[None if value is None else _array_hash(value)
                         for value in pipeline.smooth_joint_limit_offsets],
                positions=_array_hash(pass_._input_positions),
                rotations=_array_hash(pass_._input_rotations),
                reference=_array_hash(pipeline.human_robot_scaler.reference_joint_q),
                model=_model_binding(pipeline.ik_model))


def _device_value(value, device):
    if value is None:
        return None
    if hasattr(value, "device") and hasattr(value, "numpy"):
        return wp.clone(value, device=device)
    return np.asarray(value).copy()


@contextmanager
def _preserve_collision_holder(solver_type):
    """The bridge constructor registers its collision object in a closure list.

    Restore the original registration so the owning CPU pipeline still performs
    its own post-clamp audit using its original CPU objective and state.
    """
    constructor = solver_type.__init__
    closure = dict(zip(getattr(constructor.__code__, "co_freevars", ()),
                       constructor.__closure__ or ()))
    cell = closure.get("collision_holder")
    holder = cell.cell_contents if cell is not None else None
    if cell is not None and not isinstance(holder, list):
        raise RuntimeError("Unexpected SOMA collision registration holder")
    previous = list(holder) if isinstance(holder, list) else None
    try:
        yield
    finally:
        if previous is not None:
            holder[:] = previous


def _forbid_forward_solve():
    raise RuntimeError("This SOMA device context is for residual/Jacobian evaluation only")


def clone_evaluation_context(pipeline, pass_, device="cuda:0"):
    """Return a device-local pipeline/pass pair with exactly the same objectives.

    Call only after the original single-motion forward initialization.  The
    returned pass supports ``update_frame_targets`` and the solver's evaluation
    primitives, not another forward solve.  Temporal conditional rows are zeroed
    because the synchronous solver supplies their full-sequence counterparts.
    Native source targets, weights, limits and the owning CPU context are checked
    for unchanged bindings.  No LM step or CUDA graph capture is performed.
    """
    from soma_retargeter.pipelines.joint_limit_clamper import JointLimitClamper

    if len(pipeline.input_targets) != 1 or pass_._num_envs != 1:
        raise ValueError("A SOMA evaluation clone requires exactly one original motion")
    original_solver = pass_._ik_solver
    if original_solver.temporal_objective is None:
        raise ValueError("A synchronous SOMA evaluation clone requires temporal objectives")
    original_impl = original_solver.solver._impl
    target_device = wp.get_device(device)
    if original_impl.device == target_device:
        # An already device-local evaluator needs no rebuild.  Its owner retains
        # responsibility for construction and for disabling conditional rows.
        return pipeline, pass_
    before = _source_binding(pipeline, pass_)
    original_layout = [(type(obj).__name__, int(obj.residual_dim()))
                       for obj in original_impl.objectives]
    shadow, shadow_pass = copy(pipeline), copy(pass_)

    with wp.ScopedDevice(target_device):
        shadow.smooth_joint_filter_coord_masks = _device_value(
            pipeline.smooth_joint_filter_coord_masks, target_device)
        shadow.smooth_joint_limit_offsets = [_device_value(value, target_device)
                                            for value in pipeline.smooth_joint_limit_offsets]
        shadow.ik_model = shadow._build_model(1)
        model = shadow._build_model(1)
        state = model.state()
        if _model_binding(shadow.ik_model) != before["model"]:
            raise RuntimeError("Device-local SOMA model differs from original topology/limits")
        positions, rotations, limit, smooth = shadow._create_ik_objectives(1, model, state)
        extras = []
        if shadow.joint_limit_weight > 0:
            extras.append(limit)
        if shadow.smooth_joint_filter_weight > 0:
            extras.append(smooth)
        reference = wp.clone(pipeline.human_robot_scaler.reference_joint_q, device=target_device)
        with _preserve_collision_holder(type(original_solver)):
            solver = type(original_solver)(shadow.ik_model, 1, model,
                position_objectives=positions, rotation_objectives=rotations,
                extra_objectives=extras, reference_joint_q=reference)
        layout = [(type(obj).__name__, int(obj.residual_dim()))
                  for obj in solver.solver._impl.objectives]
        if layout != original_layout:
            raise RuntimeError("Device-local SOMA objective order/dimensions changed")
        # Keep the actual warmed filter weight, rather than resetting it to the
        # zero weight used when upstream originally constructs this objective.
        smooth._weight.assign(pass_._smooth_filter_obj._weight.numpy())
        for previous, current in zip(original_impl.objectives,
                                     solver.solver._impl.objectives, strict=True):
            if hasattr(previous, "weight") and previous.weight != current.weight:
                raise RuntimeError("Device-local SOMA objective weight changed")
            for name in ("target_positions", "target_rotations", "dynamic_weight"):
                if hasattr(previous, name):
                    getattr(current, name).assign(getattr(previous, name).numpy())
        solver.temporal_objective.disable()
        solver.joint_q.assign(original_solver.joint_q.numpy())
        shadow.joint_limit_clamper = JointLimitClamper(shadow.ik_model)
        shadow_pass._ik_solver, shadow_pass._state = solver, state
        shadow_pass._pos_objs, shadow_pass._rot_objs = positions, rotations
        shadow_pass._smooth_filter_obj = smooth
        shadow_pass._input_positions = wp.clone(pass_._input_positions, device=target_device)
        shadow_pass._input_rotations = wp.clone(pass_._input_rotations, device=target_device)
        shadow_pass._joint_q = wp.clone(pass_._joint_q, device=target_device)
        shadow_pass._graph = None
        shadow_pass._gpu_step = _forbid_forward_solve

    if _source_binding(pipeline, pass_) != before:
        raise RuntimeError("Constructing SOMA evaluation context changed original source bindings")
    provenance = dict(schema="soma_evaluation_device_clone.v1",
        original_device=str(original_impl.device), evaluation_device=str(target_device),
        original_source_binding_sha256=hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()).hexdigest(),
        objective_layout=layout, source_bindings_unchanged=True,
        original_context_preserved=True, forward_solve_performed=False,
        cuda_graph_captured=False, temporal_rows="Original conditional rows disabled; full-sequence rows retained by synchronous solver")
    shadow._evaluation_device_provenance = provenance
    shadow_pass._evaluation_device_provenance = provenance
    return shadow, shadow_pass
