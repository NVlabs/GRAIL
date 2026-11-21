"""Analytic Newton objective for the scalar hinges in a SOMA temporal sweep."""
import numpy as np
import newton
import newton.ik as ik
import warp as wp


@wp.kernel
def _temporal_residual(joint_q: wp.array2d(dtype=float), coords: wp.array(dtype=int),
                       targets: wp.array2d(dtype=float), gain: wp.array(dtype=float),
                       row: int, problem_idx: wp.array(dtype=int),
                       residuals: wp.array2d(dtype=float)):
    problem, k = wp.tid()
    residuals[problem, row+k] = gain[k] * (joint_q[problem, coords[k]]
                                         - targets[problem_idx[problem], k])


@wp.kernel
def _temporal_jacobian(dofs: wp.array(dtype=int), gain: wp.array(dtype=float),
                       row: int, jacobian: wp.array3d(dtype=float)):
    problem, k = wp.tid()
    jacobian[problem, row+k, dofs[k]] = gain[k]


class TemporalObjective(ik.IKObjective):
    """One residual per native revolute hinge; free-root quaternion is excluded.

    Coordinates and update DOFs are separately indexed. In particular, a free
    root contributes seven coordinates and six DOFs; assuming they share an
    offset would constrain the wrong joint and produce an incorrect Jacobian.
    """
    def __init__(self, model):
        super().__init__()
        types = model.joint_type.numpy()
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        coords, dofs, names = [], [], []
        for joint in range(model.joint_count):
            if types[joint] != int(newton.JointType.REVOLUTE):
                continue
            next_q = q_start[joint+1] if joint+1 < len(q_start) else model.joint_coord_count
            next_dof = qd_start[joint+1] if joint+1 < len(qd_start) else model.joint_dof_count
            if next_q-q_start[joint] != 1 or next_dof-qd_start[joint] != 1:
                raise ValueError("temporal objective requires scalar revolute joints")
            coords.append(int(q_start[joint]))
            dofs.append(int(qd_start[joint]))
            names.append(str(model.joint_label[joint]))
        if not coords:
            raise ValueError("temporal objective requires at least one revolute hinge")
        self.coord_indices = np.asarray(coords, dtype=np.int32)
        self.dof_indices = np.asarray(dofs, dtype=np.int32)
        self.names = names
        self._model = model
        self._coords = wp.array(self.coord_indices, dtype=int, device=model.device)
        self._dofs = wp.array(self.dof_indices, dtype=int, device=model.device)
        self._target = wp.zeros((1, len(coords)), dtype=float, device=model.device)
        self._gain = wp.zeros(len(coords), dtype=float, device=model.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return len(self.coord_indices)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if model is not self._model or jacobian_mode != ik.IKJacobianType.ANALYTIC:
            raise ValueError("temporal objective requires its bound model and analytic Jacobians")

    def update(self, target, gain):
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (self.residual_dim(),) or not np.isfinite(target).all():
            raise ValueError("temporal target must contain one finite value per hinge")
        gains = np.asarray(gain)
        if gains.dtype.kind == "b" or gains.dtype.kind not in "fiu":
            raise ValueError("temporal gain must be numeric and nonnegative")
        if gains.ndim == 0:
            gains = np.full(self.residual_dim(), gains.item(), dtype=np.float32)
        else:
            gains = gains.astype(np.float32)
        if (gains.shape != (self.residual_dim(),)
                or not np.isfinite(gains).all() or np.any(gains < 0)):
            raise ValueError("temporal gain must be scalar or one finite nonnegative value per hinge")
        self._target.assign(target.reshape(1, -1))
        self._gain.assign(gains)

    def disable(self):
        self._gain.zero_()

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_temporal_residual, dim=(joint_q.shape[0], self.residual_dim()),
                  inputs=[joint_q, self._coords, self._target, self._gain, start_idx,
                          problem_idx, residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_temporal_jacobian, dim=(joint_q.shape[0], self.residual_dim()),
                  inputs=[self._dofs, self._gain, start_idx, jacobian], device=self.device)
