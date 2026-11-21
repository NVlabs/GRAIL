"""Native hand trajectory soft targets for the existing SOMA Newton solve.

Each hand contributes wrist position (metres), wrist SO(3) log error (radians)
and seven named native hinge errors (radians). During declared contact the wrist
target follows the *current* optimized object pose. Outside contact it is fixed
in the original world frame. No hand/object material anchors are introduced.

The robot Jacobian is analytic, including the SO(3) log differential. Object
columns use that same residual, so the CUDA evaluator needs no extra FK copies.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import newton
import newton.ik as ik
import warp as wp

from soma_contact_runtime import MaterialObjective
from soma_temporal_runtime import _temporal_residual, _temporal_jacobian


@wp.func
def _rotation_error(body: wp.transform, target: wp.vec4):
    actual = wp.quat(body[3], body[4], body[5], body[6])
    desired = wp.quat(target[0], target[1], target[2], target[3])
    error = actual * wp.quat_inverse(desired)
    if error[3] < 0.0:
        error = -error
    v = wp.vec3(error[0], error[1], error[2])
    length = wp.length(v)
    phi = 2.0 * v
    if length > 1.0e-8:
        phi = (2.0 * wp.atan2(length, error[3]) / length) * v
    return phi


@wp.kernel
def _rotation_residual(body_q: wp.array2d(dtype=wp.transform),
                       target: wp.array(dtype=wp.vec4), link: int,
                       gain: wp.array(dtype=float), row: int,
                       problem_idx: wp.array(dtype=int), out: wp.array2d(dtype=float)):
    i = wp.tid()
    value = gain[0] * _rotation_error(body_q[i, link], target[problem_idx[i]])
    for k in range(3):
        out[i, row+k] = value[k]


@wp.kernel
def _rotation_jacobian(body_q: wp.array2d(dtype=wp.transform),
                       target: wp.array(dtype=wp.vec4),
                       motion: wp.array2d(dtype=wp.spatial_vector),
                       affects: wp.array(dtype=wp.uint8), link: int,
                       gain: wp.array(dtype=float), row: int,
                       out: wp.array3d(dtype=float)):
    i, dof = wp.tid()
    value = wp.vec3(0.0)
    if affects[dof] != wp.uint8(0):
        # The bridge binds a single source motion; analytic rows share target 0.
        phi = _rotation_error(body_q[i, link], target[0])
        theta2 = wp.dot(phi, phi)
        coefficient = 1.0 / 12.0 + theta2 / 720.0
        if theta2 > 1.0e-4:
            theta = wp.sqrt(theta2)
            coefficient = (1.0 - 0.5 * theta / wp.tan(0.5 * theta)) / theta2
        s = motion[i, dof]
        omega = wp.vec3(s[3], s[4], s[5])
        cross = wp.cross(phi, omega)
        value = gain[0] * (omega - 0.5 * cross + coefficient * wp.cross(phi, cross))
    for k in range(3):
        out[i, row+k, dof] = value[k]


class HandRotationObjective(ik.IKObjectiveRotation):
    """Exact geodesic rotation rows with a graph-safe dynamic gain."""
    def __init__(self, link):
        super().__init__(link, wp.quat_identity(),
                         wp.array([[0., 0., 0., 1.]], dtype=wp.vec4), weight=1.0)
        self.dynamic_weight = wp.zeros(1, dtype=float)

    def init_buffers(self, model, jacobian_mode):
        if jacobian_mode != ik.IKJacobianType.ANALYTIC:
            raise ValueError("Native hand targets require analytic Newton Jacobians")
        super().init_buffers(model, jacobian_mode)

    def update(self, quaternion, gain):
        quaternion = np.asarray(quaternion, dtype=np.float32)
        if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-5)
                or not np.isfinite(gain) or gain < 0):
            raise ValueError("Hand rotation target requires a unit XYZW quaternion and nonnegative gain")
        self.target_rotations.assign(quaternion.reshape(1, 4))
        self.dynamic_weight.assign(np.asarray([gain], dtype=np.float32))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_rotation_residual, dim=body_q.shape[0], inputs=[body_q,
            self.target_rotations, self.link_index, self.dynamic_weight, start_idx,
            problem_idx, residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_rotation_jacobian, dim=(body_q.shape[0], model.joint_dof_count),
            inputs=[body_q, self.target_rotations, joint_S_s, self.affects_dof,
                    self.link_index, self.dynamic_weight, start_idx, jacobian], device=self.device)


class HandFingerObjective(ik.IKObjective):
    """Seven scalar hinges indexed independently in q and tangent DOF space."""
    def __init__(self, model, names, joint_labels=None):
        super().__init__()
        names = list(names)
        labels = (list(joint_labels) if joint_labels is not None
                  else [str(label).rsplit("/", 1)[-1] for label in model.joint_label])
        if len(names) != 7 or len(set(names)) != 7:
            raise ValueError("A native Dex3 target must name seven distinct finger hinges")
        q_start, d_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        types = model.joint_type.numpy()
        coordinates, dofs = [], []
        for name in names:
            if labels.count(name) != 1:
                raise ValueError(f"Native finger joint is missing or ambiguous: {name}")
            index = labels.index(name)
            if (types[index] != int(newton.JointType.REVOLUTE)
                    or q_start[index+1]-q_start[index] != 1
                    or d_start[index+1]-d_start[index] != 1):
                raise ValueError(f"Native finger joint must be a scalar revolute hinge: {name}")
            coordinates.append(int(q_start[index]))
            dofs.append(int(d_start[index]))
        self.coord_indices = np.asarray(coordinates, dtype=np.int32)
        self.dof_indices = np.asarray(dofs, dtype=np.int32)
        self.names, self._model = names, model
        self._coords = wp.array(self.coord_indices, dtype=int, device=model.device)
        self._dofs = wp.array(self.dof_indices, dtype=int, device=model.device)
        self._target = wp.zeros((1, 7), dtype=float, device=model.device)
        self._gain = wp.zeros(7, dtype=float, device=model.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return 7

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if model is not self._model or jacobian_mode != ik.IKJacobianType.ANALYTIC:
            raise ValueError("Native finger target requires its bound model and analytic Jacobians")

    def update(self, target, gain):
        target, gain = np.asarray(target, dtype=np.float32), float(gain)
        if target.shape != (7,) or not np.isfinite(target).all() or not np.isfinite(gain) or gain < 0:
            raise ValueError("Native finger target requires seven finite radians and a nonnegative gain")
        self._target.assign(target.reshape(1, 7))
        self._gain.assign(np.full(7, gain, dtype=np.float32))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_temporal_residual, dim=(joint_q.shape[0], 7), inputs=[joint_q,
            self._coords, self._target, self._gain, start_idx, problem_idx, residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_temporal_jacobian, dim=(joint_q.shape[0], 7), inputs=[self._dofs,
            self._gain, start_idx, jacobian], device=self.device)


@wp.kernel
def _gather_hand_targets(positions: wp.array(dtype=wp.vec3),
                         rotations: wp.array(dtype=wp.vec4),
                         fingers: wp.array2d(dtype=float), frame: wp.array(dtype=int),
                         gains: wp.vec3, position: wp.array(dtype=wp.vec3),
                         rotation: wp.array(dtype=wp.vec4), finger: wp.array2d(dtype=float),
                         position_gain: wp.array(dtype=float), rotation_gain: wp.array(dtype=float),
                         finger_gain: wp.array(dtype=float)):
    index = frame[0]
    position[0] = positions[index]
    rotation[0] = rotations[index]
    position_gain[0] = gains[0]
    rotation_gain[0] = gains[1]
    for j in range(7):
        finger[0, j] = fingers[index, j]
        finger_gain[j] = gains[2]


def _skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _right_log_inverse(phi):
    theta2 = float(np.dot(phi, phi))
    if theta2 < 1e-4:
        coefficient = 1/12 + theta2/720 + theta2*theta2/30240
    else:
        theta = np.sqrt(theta2)
        coefficient = (1 - .5*theta/np.tan(.5*theta))/theta2
    skew = _skew(phi)
    return np.eye(3) + .5*skew + coefficient*skew@skew


class RobotHandTargetGroup:
    """Own all per-hand Newton objectives and per-frame CUDA gather buffers.

    ``targets`` is the immutable output of ``load_robot_hand_targets``. Hook
    ``objectives`` into ContactSolver extras, call ``update`` during forward/CPU
    evaluation, or ``stage`` once and ``gather`` inside each CUDA frame graph.
    Add ``object_jacobian`` to the existing object-frame matrix using the same
    evaluation's residual vector and objective-id-to-offset dictionary.
    """
    def __init__(self, model, targets, body_labels=None, joint_labels=None):
        self.targets, self.device = targets, model.device
        self.frames = len(targets["timestamps"])
        labels = (list(body_labels) if body_labels is not None
                  else [str(label).rsplit("/", 1)[-1] for label in model.body_label])
        cfg = targets["config"]
        self.gains = np.array([np.sqrt(cfg[name+"_weight"])/cfg[sigma]
            for name, sigma in (("translation", "translation_sigma_m"),
                                ("rotation", "rotation_sigma_rad"),
                                ("finger", "finger_sigma_rad"))], dtype=float)
        if not np.isfinite(self.gains).all() or np.any(self.gains < 0):
            raise ValueError("Native hand target gains must be finite and nonnegative")
        self.hands, self.objectives = [], []
        with wp.ScopedDevice(self.device):
            for side, data in targets["hands"].items():
                if labels.count(data["wrist_body"]) != 1:
                    raise ValueError(f"Native wrist body is missing or ambiguous: {data['wrist_body']}")
                link = labels.index(data["wrist_body"])
                position = MaterialObjective(link, [0., 0., 0.])
                rotation = HandRotationObjective(link)
                finger = HandFingerObjective(model, data["finger_joints"], joint_labels)
                record = dict(side=side, data=data, position=position, rotation=rotation, finger=finger,
                    positions=wp.empty(self.frames, dtype=wp.vec3),
                    rotations=wp.empty(self.frames, dtype=wp.vec4),
                    fingers=wp.array(np.asarray(data["finger_qpos"], dtype=np.float32), dtype=float))
                self.hands.append(record)
                self.objectives.extend([position, rotation, finger])

    def _world(self, data, rotations, positions):
        positions = np.einsum("tij,tj->ti", rotations, data["wrist_position_object"]) + positions
        rotations = rotations @ data["wrist_rotation_object"]
        mask = data["contact_mask"]
        return (np.where(mask[:, None], positions, data["wrist_position_world"]),
                np.where(mask[:, None, None], rotations, data["wrist_rotation_world"]))

    def update(self, frame, object_rotation, object_position, enabled=True):
        if not 0 <= frame < self.frames:
            raise ValueError("Native hand target frame is outside its bound source timeline")
        gain = self.gains if enabled else np.zeros(3)
        for item in self.hands:
            data = item["data"]
            if data["contact_mask"][frame]:
                position = object_rotation @ data["wrist_position_object"][frame] + object_position
                rotation = object_rotation @ data["wrist_rotation_object"][frame]
            else:
                position, rotation = data["wrist_position_world"][frame], data["wrist_rotation_world"][frame]
            item["position"].update(position, gain[0])
            item["rotation"].update(Rotation.from_matrix(rotation).as_quat(), gain[1])
            item["finger"].update(data["finger_qpos"][frame], gain[2])

    def stage(self, object_rotations, object_positions):
        rotations, positions = np.asarray(object_rotations), np.asarray(object_positions)
        if (rotations.shape != (self.frames, 3, 3) or positions.shape != (self.frames, 3)
                or not np.isfinite(rotations).all() or not np.isfinite(positions).all()):
            raise ValueError("Live object trajectory must match the native hand target timeline")
        for item in self.hands:
            position, rotation = self._world(item["data"], rotations, positions)
            item["positions"].assign(np.asarray(position, dtype=np.float32))
            item["rotations"].assign(np.asarray(Rotation.from_matrix(rotation).as_quat(), dtype=np.float32))

    def gather(self, frame_index):
        """Graph-safe frame selection; stage() may update targets between replays."""
        for item in self.hands:
            position, rotation, finger = item["position"], item["rotation"], item["finger"]
            wp.launch(_gather_hand_targets, dim=1, inputs=[item["positions"], item["rotations"],
                item["fingers"], frame_index, wp.vec3(*self.gains), position.target_positions,
                rotation.target_rotations, finger._target, position.dynamic_weight,
                rotation.dynamic_weight, finger._gain], device=self.device)

    def object_jacobian(self, frame, object_rotation, residual, offsets):
        """World translation / left-rotation object columns of the SAME rows.

        Canonical SO(3) log is differentiable away from angle pi; no extra
        approximation, frozen object target, or float32 FK finite difference.
        """
        result = np.zeros((len(residual), 6), dtype=float)
        for item in self.hands:
            data = item["data"]
            if not data["contact_mask"][frame]:
                continue
            row = offsets[id(item["position"])]
            point = object_rotation @ data["wrist_position_object"][frame]
            result[row:row+3, :3] = self.gains[0]*np.eye(3)
            result[row:row+3, 3:] = -self.gains[0]*_skew(point)
            if self.gains[1]:
                row = offsets[id(item["rotation"])]
                phi = np.asarray(residual[row:row+3])/self.gains[1]
                result[row:row+3, 3:] = -self.gains[1]*_right_log_inverse(phi)
        return result
