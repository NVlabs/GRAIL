"""Optional physical object-relative wrist derivatives for synchronous SOMA.

Recover actual wrist poses and their analytic tangent maps from the existing
Newton hand-target rows. This adds no GPU FK query or device transfer. Position
uses metres; rotation uses SO(3) angular increments in the object's coordinates.
The object translation/left-rotation tangent is differentiated jointly with the
robot, including a shared precontact object block. Source phases and clock are
immutable; support/slide never receive a zero-relative-velocity objective.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation


_WEIGHTS = {
    "translation_acceleration_weight", "translation_jerk_weight",
    "rotation_acceleration_weight", "rotation_jerk_weight",
    "hold_translation_velocity_weight", "hold_rotation_velocity_weight",
}
_ROLES = {"hold", "support", "slide"}


def validate_relative_temporal_config(value):
    """Weights multiply SUMS of squared SI derivatives; omission is disabled."""
    if not isinstance(value, dict) or set(value) - _WEIGHTS - {"roles", "role_boundary_policy"}:
        raise ValueError("Invalid relative_hand_temporal configuration fields")
    result = {name: 0. for name in sorted(_WEIGHTS)}
    for name in _WEIGHTS:
        number = value.get(name, 0.)
        if (isinstance(number, bool) or not isinstance(number, Real)
                or not math.isfinite(number) or number < 0):
            raise ValueError(f"relative_hand_temporal {name} must be finite and nonnegative")
        result[name] = float(number)
    if not any(result.values()):
        raise ValueError("relative_hand_temporal requires a positive derivative weight")
    roles = value.get("roles", ["hold", "support", "slide"])
    if (not isinstance(roles, (list, tuple)) or not roles
            or any(not isinstance(role, str) for role in roles)
            or len(set(roles)) != len(roles) or set(roles) - _ROLES):
        raise ValueError("relative_hand_temporal roles must be unique hold/support/slide names")
    result["roles"] = list(roles)
    policy = value.get("role_boundary_policy", "same_role")
    if not isinstance(policy, str) or policy not in {"same_role", "all_active"}:
        raise ValueError("relative_hand_temporal role_boundary_policy must be same_role or all_active")
    # Preserve the legacy normalized config and saved telemetry byte-for-byte.
    if policy != "same_role":
        result["role_boundary_policy"] = policy
    if "hold" not in roles and any(result[name] for name in _WEIGHTS if name.startswith("hold_")):
        raise ValueError("Relative hold velocity requires the hold role")
    return result


def skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def left_jacobian(phi):
    """SO(3) exponential differential mapping log derivatives to left tangents."""
    phi = np.asarray(phi, dtype=float)
    theta2 = float(phi @ phi)
    matrix = skew(phi)
    if theta2 < 1e-8:
        a, b = .5-theta2/24+theta2*theta2/720, 1/6-theta2/120+theta2*theta2/5040
    else:
        theta = np.sqrt(theta2)
        a, b = (1-np.cos(theta))/theta2, (theta-np.sin(theta))/(theta2*theta)
    return np.eye(3)+a*matrix+b*matrix@matrix


def left_log_jacobian(phi):
    phi = np.asarray(phi, dtype=float)
    theta2 = float(phi @ phi)
    matrix = skew(phi)
    if theta2 < 1e-8:
        coefficient = 1/12+theta2/720+theta2*theta2/30240
    else:
        theta = np.sqrt(theta2)
        coefficient = (1-.5*theta/np.tan(.5*theta))/theta2
    return np.eye(3)-.5*matrix+coefficient*matrix@matrix


def recover_relative_pose(position_residual, rotation_residual, position_jacobian,
                          rotation_jacobian, gains, desired_position, desired_rotation,
                          object_position, object_rotation):
    """Recover actual relative pose and exact robot/object tangent maps.

    Newton position rows are gain*(desired-actual). Hand rotation rows are
    gain*Log(actual*desired.T), with a left-tangent analytic derivative. Object
    columns here differentiate the physical relative pose directly, not merely
    the moving target error. Jacobians may both be None for residual evaluation.
    """
    gain_p, gain_r = np.asarray(gains, dtype=float)[:2]
    if not np.isfinite([gain_p, gain_r]).all() or min(gain_p, gain_r) <= 0:
        raise ValueError("Relative wrist reconstruction requires positive finite hand pose gains")
    inverse = np.asarray(object_rotation, dtype=float).T
    world_position = np.asarray(desired_position)-np.asarray(position_residual)/gain_p
    phi = np.asarray(rotation_residual)/gain_r
    world_rotation = Rotation.from_rotvec(phi).as_matrix() @ desired_rotation
    position = inverse @ (world_position-object_position)
    rotation = inverse @ world_rotation
    if position_jacobian is None and rotation_jacobian is None:
        return dict(position=position, rotation=rotation)
    if position_jacobian is None or rotation_jacobian is None:
        raise ValueError("Both relative wrist Jacobian blocks must be supplied together")
    position_robot = -inverse @ position_jacobian/gain_p
    rotation_robot = inverse @ left_jacobian(phi) @ rotation_jacobian/gain_r
    position_object = np.c_[-inverse, inverse @ skew(world_position-object_position)]
    rotation_object = np.c_[np.zeros((3, 3)), -inverse]
    return dict(position=position, rotation=rotation, position_robot=position_robot,
                rotation_robot=rotation_robot, position_object=position_object,
                rotation_object=rotation_object)


def phase_starts(labels, contact_mask, order, roles, *, role_boundary_policy="same_role"):
    """Keep complete allowed active stencils; optionally cross contact roles.

    The default is also used by the independent WORLD block and remains its
    original partition. The optional policy changes only RELATIVE derivatives.
    """
    if role_boundary_policy not in {"same_role", "all_active"}:
        raise ValueError("Invalid relative wrist role boundary policy")
    labels, contact_mask = np.asarray(labels), np.asarray(contact_mask, dtype=bool)
    if role_boundary_policy == "all_active":
        allowed = np.isin(labels, list(roles)) & contact_mask
        return np.asarray([i for i in range(len(labels)-order)
            if np.all(allowed[i:i+order+1])], dtype=int)
    return np.asarray([i for i in range(len(labels)-order)
        if labels[i] in roles and np.all(labels[i:i+order+1] == labels[i])
        and np.all(contact_mask[i:i+order+1])], dtype=int)


def _difference_matrix(length, order):
    if length <= order:
        return sparse.csr_matrix((0, length))
    coefficients = np.diff(np.eye(order+1), n=order, axis=0).ravel()
    return sparse.diags([np.full(length-order, c) for c in coefficients],
                        np.arange(order+1), shape=(length-order, length), format="csr")


def pose_tangent_matrix(robot, objects, frame_to_block, object_variables):
    """Assemble 3 x tangent maps; repeated object columns sum shared blocks."""
    robot, objects = np.asarray(robot), np.asarray(objects)
    frames = len(robot)
    robot_matrix = sparse.block_diag(list(robot), format="csr")
    if not object_variables:
        return robot_matrix
    rows, cols, values = [], [], []
    for frame, block in enumerate(frame_to_block):
        if block < 0 or 6*int(block)+6 > object_variables:
            raise ValueError("Relative temporal object layout index is invalid")
        rr, cc = np.nonzero(objects[frame])
        rows.extend((3*frame+rr).tolist())
        cols.extend((6*int(block)+cc).tolist())
        values.extend(objects[frame][rr, cc].tolist())
    object_matrix = sparse.coo_matrix((values, (rows, cols)),
                                     shape=(3*frames, object_variables)).tocsr()
    return sparse.hstack([robot_matrix, object_matrix], format="csr")


def angular_velocity_jacobian(rotations, fps):
    """Physical object-frame SO(3) increments and analytic left-pose maps."""
    rotations = np.asarray(rotations, dtype=float)
    count = len(rotations)
    phi = Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).as_rotvec()
    rows, cols, values = [], [], []
    for frame, vector in enumerate(phi):
        for index, matrix in ((frame, -left_log_jacobian(-vector)),
                              (frame+1, left_log_jacobian(vector))):
            rr, cc = np.nonzero(matrix)
            rows.extend((3*frame+rr).tolist())
            cols.extend((3*index+cc).tolist())
            values.extend((fps*matrix[rr, cc]).tolist())
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(3*max(0, count-1), 3*count)).tocsr()
    return fps*phi, matrix


def relative_temporal_rows(poses, fps, config, frame_to_block=None, object_variables=0,
                           *, jacobian=True):
    """Append summed SI derivative costs for one or more bound hands.

    Each pose dict contains position[T,3], rotation[T,3,3], phase_labels,
    contact_mask and, when linearizing, robot/object left tangent maps.
    """
    config = validate_relative_temporal_config(config)
    if not poses or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Relative wrist temporal rows need hands and a finite positive FPS")
    residuals, matrices, reports = [], [], {}
    columns = None
    for side, pose in poses.items():
        position, rotation = np.asarray(pose["position"]), np.asarray(pose["rotation"])
        count = len(position)
        if (position.shape != (count, 3) or rotation.shape != (count, 3, 3)
                or count < 2 or not np.isfinite(position).all() or not np.isfinite(rotation).all()):
            raise ValueError("Relative wrist poses must be finite complete trajectories")
        labels, mask = np.asarray(pose["phase_labels"]), np.asarray(pose["contact_mask"])
        if labels.shape != (count,) or mask.shape != (count,):
            raise ValueError("Relative wrist phase timeline differs")
        omega, omega_pose_j = angular_velocity_jacobian(rotation, fps)
        if jacobian:
            p_j = pose_tangent_matrix(pose["position_robot"], pose["position_object"], frame_to_block, object_variables)
            r_j = pose_tangent_matrix(pose["rotation_robot"], pose["rotation_object"], frame_to_block, object_variables)
            omega_j = omega_pose_j @ r_j
            columns = p_j.shape[1]
        terms = {}
        for kind, values in (("translation", position), ("rotation", omega)):
            for order, name in enumerate(("velocity", "acceleration", "jerk"), 1):
                key = ("hold_" if order == 1 else "")+kind+"_"+name+"_weight"
                weight = config[key]
                roles = ["hold"] if order == 1 else config["roles"]
                # Zero velocity is strict hold-only even when acceleration and
                # jerk explicitly connect distinct active contact roles.
                policy = "same_role" if order == 1 else config.get("role_boundary_policy", "same_role")
                starts = phase_starts(labels, mask, order, roles, role_boundary_policy=policy)
                derivative_order = order if kind == "translation" else order-1
                difference = _difference_matrix(len(values), derivative_order)[starts]
                physical = np.asarray(difference @ values)*fps**derivative_order
                if weight and len(starts):
                    gain = np.sqrt(weight)*fps**derivative_order
                    residuals.append((np.sqrt(weight)*physical).ravel())
                    if jacobian:
                        matrix = p_j if kind == "translation" else omega_j
                        matrices.append(gain*sparse.kron(difference, sparse.eye(3), format="csr") @ matrix)
                norms = np.linalg.norm(physical, axis=1)
                terms[key] = dict(weight=weight, stencil_count=len(starts),
                    active_residual_count=3*len(starts) if weight else 0,
                    stencil_first_frames=starts.tolist(),
                    component_rms=float(np.sqrt(np.mean(physical**2))) if physical.size else None,
                    vector_max=float(norms.max()) if norms.size else None,
                    energy=float(weight*np.sum(physical**2)),
                    unit=("m" if kind == "translation" else "rad")+"/s"+(str(order) if order > 1 else ""))
                if config.get("role_boundary_policy") == "all_active":
                    mixed = [int(i) for i in starts if np.any(labels[i:i+order+1] != labels[i])]
                    terms[key]["cross_role_stencil_first_frames"] = mixed
                    terms[key]["cross_role_stencil_count"] = len(mixed)
        reports[side] = terms
    residual = np.concatenate(residuals) if residuals else np.zeros(0)
    matrix = ((sparse.vstack(matrices, format="csr") if matrices else sparse.csr_matrix((0, columns)))
              if jacobian else None)
    report = dict(enabled=True, config=config, fps=float(fps), frame_count=count,
        residual_count=len(residual), weighted_energy=float(residual @ residual), hands=reports,
        derivative="Object-frame position differences; SO(3) log angular velocity followed by angular differences.",
        phase_policy="Every pose in a stencil has the same declared allowed contact role; phase boundaries are excluded.",
        hold_velocity_policy="Zero-relative-velocity guidance only in strict hold; support and slide retain motion.",
        jacobian="Analytic original Newton hand rows and physical relative pose maps; robot and object left tangents jointly differentiated.",
        energy_normalization="Sum of squared residuals, no frame or component mean.",
        extra_gpu_fk_queries=0, extra_device_transfers=0,
        limitations=["Sampled derivatives are not continuous-time collision or dynamic feasibility certificates.",
                     "SO(3) log derivatives are locally defined away from the pi cut locus."])
    if config.get("role_boundary_policy") == "all_active":
        report["phase_policy"] = (
            "Acceleration and jerk require every pose to be active and in the declared allowed roles; "
            "mixed contact-role stencils are included. Hold velocity remains same-role hold-only. "
            "Independent world-wrist costs are unchanged, including at contact-role joins.")
    return residual, matrix, report


class RelativeHandTemporal:
    """Bridge adapter with no dependence on Warp/Newton imports in this module."""
    def __init__(self, group, offsets, fps, config, object_layout):
        if group is None:
            raise ValueError("relative_hand_temporal requires bound robot_hand_targets")
        self.group, self.offsets = group, offsets
        self.fps, self.config, self.layout = float(fps), validate_relative_temporal_config(config), object_layout
        times = np.asarray(group.targets["timestamps"])
        if len(times) < 2 or not np.allclose(np.diff(times), 1/self.fps, atol=1e-9, rtol=0):
            raise ValueError("Relative temporal FPS differs from bound original source timestamps")
        self.before_report = self.report = None
        self.last_poses = None

    def sample(self, frame, residual, matrix, object_rotation, object_position):
        result = {}
        for item in self.group.hands:
            data = item["data"]
            if data["contact_mask"][frame]:
                desired_p = object_rotation @ data["wrist_position_object"][frame]+object_position
                desired_r = object_rotation @ data["wrist_rotation_object"][frame]
            else:
                desired_p, desired_r = data["wrist_position_world"][frame], data["wrist_rotation_world"][frame]
            p, r = self.offsets[id(item["position"])], self.offsets[id(item["rotation"])]
            result[item["side"]] = recover_relative_pose(residual[p:p+3], residual[r:r+3],
                None if matrix is None else matrix[p:p+3], None if matrix is None else matrix[r:r+3],
                self.group.gains, desired_p, desired_r, object_position, object_rotation)
        return result

    def assemble(self, samples, jacobian):
        poses = {}
        for item in self.group.hands:
            side, data = item["side"], item["data"]
            poses[side] = {key: np.asarray([sample[side][key] for sample in samples]) for key in samples[0][side]}
            poses[side].update(phase_labels=data["phase_labels"], contact_mask=data["contact_mask"])
        residual, matrix, self.report = relative_temporal_rows(poses, self.fps, self.config,
            None if self.layout is None else self.layout.frame_to_block,
            0 if self.layout is None else self.layout.variable_count, jacobian=jacobian)
        self.last_poses = poses
        if self.before_report is None:
            self.before_report = self.report
        return residual, matrix

    def evidence(self):
        """Original first-linearization physical poses, phase masks and maps."""
        result = {"relative_hand_sides": np.asarray(list(self.last_poses))}
        for side, pose in self.last_poses.items():
            result.update({"relative_"+side+"_"+key: np.asarray(value) for key, value in pose.items()})
        return result
