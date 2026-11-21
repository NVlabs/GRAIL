"""Validate actual sparse coupling, native checkpoints and global descent."""
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy import sparse

try:
    from .soma_temporal import joint_weight_vector, validate_temporal_config
except ImportError:
    from soma_temporal import joint_weight_vector, validate_temporal_config


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value, label, minimum=0):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu" or int(array) < minimum:
        raise ValueError(f"Invalid integer {label}")
    return int(array)


def _number(value, label):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value)):
        raise ValueError(f"Invalid finite number {label}")
    return float(value)


def _array(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"Incomplete/nonfinite {label}")
    return array


def _same_array(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and actual.tobytes() == expected.tobytes())


def _same(actual, expected, label, *, rtol=1e-10, atol=1e-10):
    if not np.isclose(_number(actual, label), expected, rtol=rtol, atol=atol):
        raise ValueError(f"Inconsistent {label}")


def _same_telemetry(actual, expected, label):
    """Exact report structure/phase indices with float recomputation tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Inconsistent {label} fields")
        for key, value in expected.items():
            _same_telemetry(actual[key], value, label+"/"+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Inconsistent {label} entries")
        for i, value in enumerate(expected):
            _same_telemetry(actual[i], value, label+"/"+str(i))
    elif isinstance(expected, float):
        _same(actual, expected, label, rtol=1e-11, atol=1e-12)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Inconsistent {label}")


def _solver_history(report):
    solver = report.get("solver_report", {})
    accepted = _integer(report.get("completed_global_iterations"), "accepted global count", 1)
    if (report.get("method") != "synchronous_sparse"
            or report.get("schema") != "soma_synchronous_trajectory.v1"
            or solver.get("method") != "whole_sequence_sparse_levenberg_marquardt_lsmr"
            or solver.get("framewise_block_updates") is not False
            or solver.get("technical_execution_complete") is not True
            or solver.get("objective") != "0.5 * sum(residual**2)"
            or _integer(solver.get("accepted_steps"), "solver accepted count") != accepted
            or _integer(solver.get("completed_global_iterations"), "solver global count") != accepted
            or _integer(report.get("completed_sweeps"), "legacy sweep count") != 0
            or _integer(report.get("completed_joint_sweeps"), "legacy joint sweep count") != 0
            or _integer(report.get("completed_robot_only_sweeps"), "legacy robot sweep count") != 0):
        raise ValueError("Missing genuine simultaneous full-trajectory solver execution")
    solves = _integer(solver.get("linear_solves"), "whole-system solve count", 1)
    history = solver.get("history")
    if not isinstance(history, list) or not history or not all(isinstance(x, dict) for x in history):
        raise ValueError("Missing accepted whole-system iteration history")
    current = initial = _number(solver.get("initial_energy"), "initial complete energy")
    energies, count, actual_solves = [current], 0, 0
    for index, row in enumerate(history):
        if _integer(row.get("iteration"), "history iteration") != index or type(row.get("accepted")) is not bool:
            raise ValueError("Invalid global history iteration/acceptance flag")
        _same(row.get("energy_before"), current, "accepted energy chain")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or not all(isinstance(x, dict) for x in attempts):
            raise ValueError("Missing whole-system linear-solve attempts")
        actual_solves += len(attempts)
        accepted_trials = [trial for attempt in attempts for trial in attempt.get("candidates", [])
                           if isinstance(trial, dict) and trial.get("accepted") is True]
        if row["accepted"]:
            after = _number(row.get("energy_after"), "accepted complete energy")
            if not after < current or len(accepted_trials) != 1:
                raise ValueError("Global step did not lower the complete nonlinear objective")
            trial = accepted_trials[0]
            _same(trial.get("energy"), after, "accepted candidate energy")
            _same(trial.get("actual_reduction"), current-after, "actual global energy reduction")
            prediction = _number(trial.get("predicted_reduction"), "predicted reduction")
            if prediction <= 0:
                raise ValueError("Accepted global step has no positive model reduction")
            _same(trial.get("gain_ratio"), (current-after)/prediction, "global acceptance ratio")
            current = after
            energies.append(current)
            count += 1
        elif accepted_trials:
            raise ValueError("Unaccepted iteration contains a falsely accepted candidate")
    saved = solver.get("accepted_energy_history")
    if (count != accepted or solves != actual_solves or not isinstance(saved, list)
            or len(saved) != len(energies)):
        raise ValueError("Inconsistent accepted whole-system iteration/solve/energy counts")
    for actual, expected in zip(saved, energies):
        _same(actual, expected, "saved accepted energy history")
    _same(solver.get("final_energy"), current, "final accepted complete energy")
    _same(solver.get("energy_reduction"), initial-current, "total global energy reduction")
    if _integer(solver.get("evaluations"), "complete evaluation count", 1) < accepted+1:
        raise ValueError("Accepted global steps lack complete objective evaluations")
    return solver, accepted, initial


def _sparse_matrix(data):
    shape = np.asarray(data["shape"])
    if shape.shape != (2,) or shape.dtype.kind not in "iu" or np.any(shape <= 0):
        raise ValueError("Invalid saved global Jacobian shape")
    shape = tuple(int(x) for x in shape)
    indices, indptr, values = data["indices"], data["indptr"], data["data"]
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or indptr.shape != (shape[0]+1,) or indptr.dtype.kind not in "iu"
            or values.shape != indices.shape or values.dtype.kind not in "fiu"
            or indptr[0] != 0 or indptr[-1] != len(values) or np.any(np.diff(indptr) < 0)
            or np.any(indices < 0) or np.any(indices >= shape[1]) or not np.isfinite(values).all()):
        raise ValueError("Malformed saved coupled CSR Jacobian")
    matrix = sparse.csr_matrix((values, indices, indptr), shape=shape)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _rotations(values, frames, label):
    values = _array(values, (frames, 3, 3), label)
    # Preserve the same source-matrix rounding allowance as soma_object._poses.
    if not (np.allclose(values @ values.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
            and np.allclose(np.linalg.det(values), 1., atol=1e-5, rtol=0)):
        raise ValueError(f"Invalid SO3 matrices in {label}")
    return values


def _temporal(matrix, first, stop, initial, coords, dofs, names, fps, config, native_dofs):
    frames = len(initial)
    weights = np.sqrt(joint_weight_vector(config, names))
    selector = sparse.csr_matrix((weights, (np.arange(len(dofs)), dofs)), shape=(len(dofs), native_dofs))
    expected_r, expected_j = [], []
    for order, label in enumerate(("velocity", "acceleration", "jerk"), 1):
        gain = np.sqrt(config[label+"_weight"])*fps**order
        if not gain or frames <= order:
            continue
        coefficients = [(-1.)**(order-k)*comb(order, k) for k in range(order+1)]
        difference = sparse.diags([np.full(frames-order, c) for c in coefficients],
                                  np.arange(order+1), shape=(frames-order, frames), format="csr")
        expected_r.append((gain*np.diff(initial[:, coords].astype(float), n=order, axis=0)*weights).ravel())
        expected_j.append(gain*sparse.kron(difference, selector, format="csr"))
    expected_r = np.concatenate(expected_r)
    expected_j = sparse.vstack(expected_j, format="csr")
    if stop-first != len(expected_r):
        raise ValueError("Saved temporal rows omit declared whole-sequence derivative terms")
    robot = frames*native_dofs
    actual = matrix[first:stop, :robot]
    difference = actual-expected_j
    difference.eliminate_zeros()
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-10:
        raise ValueError("Saved temporal Jacobian differs from original named derivative terms")
    if matrix[first:stop, robot:].nnz:
        raise ValueError("Robot temporal rows contain undeclared object variables")
    row_ids = np.repeat(np.arange(actual.shape[0]), np.diff(actual.indptr))
    frame_ids = actual.indices//native_dofs
    minimum, maximum = np.full(actual.shape[0], frames), np.full(actual.shape[0], -1)
    np.minimum.at(minimum, row_ids, frame_ids)
    np.maximum.at(maximum, row_ids, frame_ids)
    cross_rows = int(np.count_nonzero(maximum > minimum))
    if cross_rows < 1:
        raise ValueError("No actual nonzero cross-frame Jacobian rows in saved global problem")
    return expected_r, cross_rows


def _validate(output, report):
    output = Path(output)
    solver, accepted, initial_energy = _solver_history(report)
    config = validate_temporal_config(report["config"])
    if config.get("method") != "synchronous_sparse" or "sequence_warm_start" in config:
        raise ValueError("Legacy sweeps/G20 warm start cannot establish synchronous execution")
    problem = report["global_problem"]
    evidence = report.get("linearization_evidence", {})
    path = output/"synchronous_linearization.npz"
    if (not path.is_file() or evidence.get("sha256") != _sha(path)
            or evidence.get("state") != "initial_trajectory_before_first_global_step"):
        raise ValueError("Missing/hash-inconsistent saved coupled sparse linearization")
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key].copy() for key in loaded.files}
    matrix = _sparse_matrix(data)
    shape = matrix.shape
    frames = _integer(data["frame_count"], "source frame count", 4)
    dof_count = _integer(data["robot_dofs_per_frame"], "native tangent DOFs", 1)
    coordinate_count = _integer(problem["robot_coordinate_count"], "native coordinates", 1)
    robot = _integer(data["robot_variable_count"], "robot variable count", 1)
    obj = _integer(data["object_variable_count"], "object variable count")
    per_frame = _integer(data["frame_residual_count"], "same-frame residual count", 1)
    first = _integer(data["temporal_row_start"], "temporal row start")
    stop = _integer(data["temporal_row_stop"], "temporal row stop")
    speed = problem.get("hinge_speed")
    speed_count = 0
    if "hinge_speed" in config:
        if not isinstance(speed,dict):
            raise ValueError("Missing hinge speed execution evidence")
        speed_count = _integer(speed.get("residual_count"), "hinge speed residual count")
        if (speed.get("row_start") != shape[0]-speed_count
                or speed.get("row_stop") != shape[0]
                or speed.get("config") != config["hinge_speed"]
                or speed.get("object_jacobian_nnz") != 0):
            raise ValueError("Hinge speed rows/config/object independence differ")
    elif speed is not None or "hinge_speed" in report:
        raise ValueError("Undeclared hinge speed rows")
    world = problem.get("world_hand_temporal")
    world_count = 0
    if "world_hand_temporal" in config:
        if not isinstance(world, dict):
            raise ValueError("Missing world wrist temporal execution evidence")
        world_count = _integer(world.get("residual_count"), "world wrist residual count")
        if (world.get("row_start") != shape[0]-speed_count-world_count
                or world.get("row_stop") != shape[0]-speed_count
                or world.get("config") != config["world_hand_temporal"]
                or world.get("object_jacobian_nnz") != 0):
            raise ValueError("World wrist temporal rows/config/object independence differ")
    elif world is not None or "world_hand_sides" in data:
        raise ValueError("Undeclared world wrist temporal rows")
    relative = problem.get("relative_hand_temporal")
    relative_count = 0
    if "relative_hand_temporal" in config:
        if not isinstance(relative, dict):
            raise ValueError("Missing relative wrist temporal execution evidence")
        relative_count = _integer(relative.get("residual_count"), "relative wrist residual count")
        if (relative.get("row_start") != shape[0]-speed_count-world_count-relative_count
                or relative.get("row_stop") != shape[0]-speed_count-world_count
                or relative.get("config") != config["relative_hand_temporal"]):
            raise ValueError("Relative wrist temporal rows/config differ")
    elif relative is not None or "relative_hand_sides" in data:
        raise ValueError("Undeclared relative wrist temporal rows")
    residual = _array(data["residual"], (shape[0],), "initial complete residual")
    initial = _array(data["joint_q"], (frames, coordinate_count), "whole native initial trajectory")
    if (robot != frames*dof_count or shape[1] != robot+obj or first != frames*per_frame
            or not first < stop <= shape[0] or report.get("frame_count") != frames
            or problem.get("variable_count") != shape[1] or problem.get("residual_count") != shape[0]
            or problem.get("robot_variable_count") != robot or problem.get("object_variable_count") != obj
            or problem.get("frame_count") != frames or problem.get("robot_dofs_per_frame") != dof_count
            or problem.get("source_frame_residuals") != first
            or problem.get("robot_temporal_residuals") != stop-first
            or problem.get("object_reference_temporal_residuals") != shape[0]-stop-relative_count-world_count-speed_count
            or solver.get("variable_count") != shape[1] or solver.get("residual_count") != shape[0]):
        raise ValueError("Saved global problem variable/residual dimensions differ")
    _same(initial_energy, .5*float(np.dot(residual, residual)), "saved initial residual energy", rtol=1e-6, atol=1e-8)
    checkpoint = output/"temporal_sequence_states.npz"
    if not checkpoint.is_file() or report.get("checkpoint_sha256") != _sha(checkpoint):
        raise ValueError("Missing/hash-inconsistent final synchronous checkpoint")
    with np.load(checkpoint, allow_pickle=False) as loaded:
        final_data = {key: loaded[key].copy() for key in loaded.files}
    if not _same_array(final_data["initial_joint_q"], initial):
        raise ValueError("Checkpoint initial native state differs from saved global linearization")
    if "prior_global_seed" in config or "prior_global_seed" in report:
        try:
            from .soma_prior_global_seed import validate_result as validate_prior_seed_result
        except ImportError:
            from soma_prior_global_seed import validate_result as validate_prior_seed_result
        validate_prior_seed_result(output, report,
            dict(q=initial, obj_R=data["obj_R"], obj_t=data["obj_t"]))
    final = _array(final_data["refined_joint_q"], initial.shape, "final synchronous native trajectory")
    if _integer(final_data["completed_global_iterations"], "checkpoint accepted count", 1) != accepted:
        raise ValueError("Checkpoint does not bind accepted global iteration count")
    fps = _number(report["fps"], "source FPS")
    if fps <= 0:
        raise ValueError("Source FPS must be positive")
    _same(final_data["fps"].item(), fps, "checkpoint FPS")
    coords, dofs = data["scalar_coordinate_indices"], data["scalar_dof_indices"]
    if (coords.ndim != 1 or coords.dtype.kind not in "iu" or dofs.shape != coords.shape
            or dofs.dtype.kind not in "iu" or not len(coords)
            or np.any(coords < 0) or np.any(coords >= coordinate_count)
            or np.any(dofs < 0) or np.any(dofs >= dof_count)
            or len(np.unique(coords)) != len(coords) or len(np.unique(dofs)) != len(dofs)
            or not np.array_equal(final_data["coordinate_indices"], coords)):
        raise ValueError("Saved scalar native coordinate layout differs")
    names = final_data["joint_names"].tolist()
    if not isinstance(names, list) or len(names) != len(coords):
        raise ValueError("Checkpoint does not bind one original name per scalar hinge")
    if "refinement_scope" in config or "refinement_scope" in report or "active_variable_indices" in solver:
        try:
            from .soma_refinement_scope import validate_result as validate_refinement_result
        except ImportError:
            from soma_refinement_scope import validate_result as validate_refinement_result
        validate_refinement_result(report,
            dict(q=initial, obj_R=final_data["initial_obj_R"], obj_t=final_data["initial_obj_t"]),
            dict(q=final, obj_R=final_data["refined_obj_R"], obj_t=final_data["refined_obj_t"]),
            frames=frames, dofs_per_frame=dof_count, coordinate_count=coordinate_count,
            coordinates=coords, dofs=dofs, names=names, object_variables=obj)
    expected_r, cross_rows = _temporal(matrix, first, stop, initial, coords, dofs, names, fps, config, dof_count)
    if not np.allclose(residual[first:stop], expected_r, atol=1e-9, rtol=1e-10):
        raise ValueError("Saved temporal residual differs from original initial trajectory")
    source_rows = matrix[:first]
    row_ids = np.repeat(np.arange(first), np.diff(source_rows.indptr))
    columns = source_rows.indices
    robot_mask = columns < robot
    if np.any(columns[robot_mask]//dof_count != row_ids[robot_mask]//per_frame):
        raise ValueError("Frame objectives reference another robot frame's variables")
    mapping = data["object_frame_to_block"]
    policy = report.get("object_motion_policy")
    if policy is None:
        expected_blocks = np.full(frames, -1)
        if obj != 0:
            raise ValueError("Undeclared object optimization variables")
    else:
        if (not isinstance(policy, dict) or type(policy.get("optimization_enabled")) is not bool
                or type(policy.get("source_is_static")) is not bool
                or policy["optimization_enabled"] == policy["source_is_static"]):
            raise ValueError("Invalid deterministic object-motion policy")
        contact = _integer(policy["contact_frame"], "source contact frame")
        if contact >= frames:
            raise ValueError("Object contact boundary outside source timeline")
        if policy["optimization_enabled"]:
            expected_blocks = (np.r_[np.zeros(contact, dtype=int), np.arange(1, frames-contact+1)]
                               if contact else np.arange(frames))
            if obj != 6*(frames-contact+int(contact > 0)):
                raise ValueError("Moving object lacks all shared-prefix/postcontact variables")
        else:
            expected_blocks = np.full(frames, -1)
            if obj != 0:
                raise ValueError("Static object has undeclared optimized variables")
    if mapping.dtype.kind not in "iu" or not np.array_equal(mapping, expected_blocks):
        raise ValueError("Saved object variable incidence violates exact shared-prefix/static policy")
    if relative is not None:
        try:
            from .soma_relative_temporal import relative_temporal_rows
        except ImportError:
            from soma_relative_temporal import relative_temporal_rows
        sides = data["relative_hand_sides"].tolist()
        if not isinstance(sides, list) or not sides or len(set(sides)) != len(sides) or set(sides)-{"left", "right"}:
            raise ValueError("Invalid saved relative wrist hand identities")
        poses = {}
        for side in sides:
            prefix = "relative_"+side+"_"
            poses[side] = {
                "position": _array(data[prefix+"position"], (frames, 3), "relative wrist position"),
                "rotation": _rotations(data[prefix+"rotation"], frames, "relative wrist rotation"),
                "position_robot": _array(data[prefix+"position_robot"], (frames, 3, dof_count), "relative wrist position robot map"),
                "rotation_robot": _array(data[prefix+"rotation_robot"], (frames, 3, dof_count), "relative wrist rotation robot map"),
                "position_object": _array(data[prefix+"position_object"], (frames, 3, 6), "relative wrist position object map"),
                "rotation_object": _array(data[prefix+"rotation_object"], (frames, 3, 6), "relative wrist rotation object map"),
                "phase_labels": data[prefix+"phase_labels"], "contact_mask": data[prefix+"contact_mask"]}
        expected_relative_r, expected_relative_j, expected_relative_report = relative_temporal_rows(
            poses, fps, config["relative_hand_temporal"], mapping, obj)
        row_start, row_end = shape[0]-speed_count-world_count-relative_count, shape[0]-speed_count-world_count
        if (len(expected_relative_r) != relative_count
                or not np.allclose(residual[row_start:row_end], expected_relative_r, atol=1e-9, rtol=1e-10)):
            raise ValueError("Saved relative wrist residuals differ from physical poses/phases")
        difference = matrix[row_start:row_end]-expected_relative_j
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-8:
            raise ValueError("Saved relative wrist Jacobian differs from joint robot/object maps")
        before = report.get("relative_hand_temporal", {}).get("before")
        _same_telemetry(before, expected_relative_report, "initial relative wrist telemetry")
        after = report.get("relative_hand_temporal", {}).get("after", {})
        if after.get("config") != config["relative_hand_temporal"] or after.get("residual_count") != relative_count:
            raise ValueError("Missing final relative wrist temporal telemetry")
    if world is not None:
        try:
            from .soma_world_temporal import world_temporal_rows
        except ImportError:
            from soma_world_temporal import world_temporal_rows
        sides = data["world_hand_sides"].tolist()
        if not isinstance(sides,list) or not sides or len(set(sides))!=len(sides) or set(sides)-{"left","right"}:
            raise ValueError("Invalid saved world wrist hand identities")
        poses = {}
        for side in sides:
            prefix = "world_"+side+"_"
            poses[side] = {
                "position": _array(data[prefix+"position"],(frames,3),"world wrist position"),
                "rotation": _rotations(data[prefix+"rotation"],frames,"world wrist rotation"),
                "position_robot": _array(data[prefix+"position_robot"],(frames,3,dof_count),"world wrist position robot map"),
                "rotation_robot": _array(data[prefix+"rotation_robot"],(frames,3,dof_count),"world wrist rotation robot map"),
                "phase_labels": data[prefix+"phase_labels"], "contact_mask": data[prefix+"contact_mask"]}
        expected_world_r,expected_world_j,expected_world_report = world_temporal_rows(
            poses,fps,config["world_hand_temporal"],mapping,obj)
        row_start = shape[0]-speed_count-world_count
        if len(expected_world_r)!=world_count or not np.allclose(residual[row_start:shape[0]-speed_count],expected_world_r,atol=1e-9,rtol=1e-10):
            raise ValueError("Saved world wrist residuals differ from physical poses/phases")
        difference = matrix[row_start:shape[0]-speed_count]-expected_world_j
        if difference.nnz and np.max(np.abs(difference.data))>1e-8:
            raise ValueError("Saved world wrist Jacobian differs from physical robot maps")
        if matrix[row_start:shape[0]-speed_count,robot:].nnz:
            raise ValueError("World wrist derivatives must not couple free hands to object variables")
        _same_telemetry(report.get("world_hand_temporal",{}).get("before"),expected_world_report,"initial world wrist telemetry")
        after = report.get("world_hand_temporal",{}).get("after",{})
        if after.get("config")!=config["world_hand_temporal"] or after.get("residual_count")!=world_count:
            raise ValueError("Missing final world wrist temporal telemetry")
    if speed is not None:
        try:
            from .soma_hinge_speed import hinge_speed_rows
        except ImportError:
            from soma_hinge_speed import hinge_speed_rows
        clock = np.arange(frames)/fps
        er, ej, before_speed = hinge_speed_rows(initial,clock,config["hinge_speed"],
            coords,dofs,dof_count,names,obj)
        _, _, after_speed = hinge_speed_rows(final,clock,config["hinge_speed"],
            coords,dofs,dof_count,names,obj,jacobian=False)
        start = shape[0]-speed_count
        if (len(er)!=speed_count or not np.allclose(residual[start:],er,atol=1e-9,rtol=1e-10)
                or speed.get("scalar_coordinate_indices")!=coords.tolist()
                or speed.get("scalar_dof_indices")!=dofs.tolist()
                or speed.get("joint_names")!=names
                or speed.get("limits_rad_s")!=before_speed["limits_rad_s"]):
            raise ValueError("Saved hinge speed residual/native layout/limits differ")
        delta = matrix[start:]-ej
        if delta.nnz and np.max(np.abs(delta.data))>1e-8:
            raise ValueError("Saved hinge speed Jacobian differs from native scalar derivatives")
        if matrix[start:,robot:].nnz:
            raise ValueError("Hinge speed rows have object derivatives")
        _same_telemetry(report.get("hinge_speed",{}).get("before"),before_speed,"initial hinge speed telemetry")
        _same_telemetry(report.get("hinge_speed",{}).get("after"),after_speed,"final hinge speed telemetry")
    object_mask = ~robot_mask
    if np.any((columns[object_mask]-robot)//6 != expected_blocks[row_ids[object_mask]//per_frame]):
        raise ValueError("Frame objective uses the wrong shared/postcontact object variable block")
    robot_rows = np.bincount(row_ids[robot_mask], minlength=first) > 0
    object_rows = np.bincount(row_ids[object_mask], minlength=first) > 0
    mixed_rows = int(np.count_nonzero(robot_rows & object_rows))
    if policy is not None:
        old_R = _rotations(data["obj_R"], frames, "initial object rotations")
        old_t = _array(data["obj_t"], (frames, 3), "initial object positions")
        new_R = _rotations(final_data["refined_obj_R"], frames, "final object rotations")
        new_t = _array(final_data["refined_obj_t"], (frames, 3), "final object positions")
        if not (_same_array(final_data["initial_obj_R"], old_R) and _same_array(final_data["initial_obj_t"], old_t)):
            raise ValueError("Checkpoint initial object differs from saved global linearization")
        if policy["optimization_enabled"]:
            if mixed_rows < 1 or np.any(np.diff(matrix[:, robot:].tocsc().indptr) == 0):
                raise ValueError("Moving object lacks actual nonzero robot-object coupling/all editable columns")
            if contact and not all(np.all(values[:contact] == values[0]) for values in (old_R, old_t, new_R, new_t)):
                raise ValueError("Object precontact interval is not exactly one shared static pose")
        elif not (np.array_equal(old_R, new_R) and np.array_equal(old_t, new_t)):
            raise ValueError("Static object was changed by synchronous optimization")
    elif data["obj_R"].shape != (0, 3, 3) or data["obj_t"].shape != (0, 3):
        raise ValueError("Unbound initial object arrays in saved global problem")
    if np.array_equal(initial, final) and (policy is None or (np.array_equal(old_R, new_R) and np.array_equal(old_t, new_t))):
        raise ValueError("Claimed accepted global iterations did not change any native state")
    history_path = output/"synchronous_optimization.json"
    if not history_path.is_file() or report.get("solver_report_sha256") != _sha(history_path):
        raise ValueError("Missing/hash-inconsistent saved synchronous objective history")
    if json.loads(history_path.read_text()) != {key: value for key, value in report.items() if key not in ("solver_report_path", "solver_report_sha256")}:
        raise ValueError("Runtime synchronous report differs from saved solver history")
    return dict(passed=True, scope="Saved sparse all-frame/mixed-variable coupling, native checkpoint and global descent; not motion acceptance",
        variable_count=shape[1], residual_count=shape[0], initial_jacobian_nnz=int(matrix.nnz),
        cross_frame_rows=cross_rows, mixed_robot_object_rows=mixed_rows,
        completed_global_iterations=accepted, saved_linearization_sha256=evidence["sha256"],
        checkpoint_sha256=report["checkpoint_sha256"], solver_report_sha256=report["solver_report_sha256"])


def validate_synchronous_result(output, report):
    try:
        return _validate(output, report)
    except (KeyError, TypeError, IndexError, OSError, OverflowError) as exc:
        raise ValueError(f"Malformed or incomplete synchronous execution evidence: {exc}") from exc
