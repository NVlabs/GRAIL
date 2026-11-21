"""Graph-safe sparse authored hinge objectives, independent of temporal priors."""
import numpy as np
import newton
import newton.ik as ik
import warp as wp
from soma_temporal_runtime import _temporal_residual, _temporal_jacobian
from soma_pose_keyframes import load


@wp.kernel
def _gather(targets: wp.array2d(dtype=float), gains: wp.array2d(dtype=float),
            frame: wp.array(dtype=int), target: wp.array2d(dtype=float), gain: wp.array(dtype=float)):
    j = wp.tid()
    target[0, j] = targets[frame[0], j]
    gain[j] = gains[frame[0], j]


class PoseKeyframeObjective(ik.IKObjective):
    def __init__(self, model, config, *, model_xml, source_bvh, frame_count, fps):
        super().__init__()
        labels = [str(x).rsplit('/', 1)[-1] for x in model.joint_label]
        if len(labels) != len(set(labels)):
            raise ValueError('Ambiguous native hinge names in pose keyframe model')
        qstart, dstart, types = (model.joint_q_start.numpy(), model.joint_qd_start.numpy(), model.joint_type.numpy())
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        indices = {name: i for i,name in enumerate(labels)
            if types[i] == int(newton.JointType.REVOLUTE)
            and qstart[i+1]-qstart[i] == 1 and dstart[i+1]-dstart[i] == 1}
        limits = {name:(float(lower[dstart[i]]),float(upper[dstart[i]])) for name,i in indices.items()}
        self.loaded = load(config, model_xml=model_xml, source_bvh=source_bvh,
            frame_count=frame_count, fps=fps, joint_limits=limits)
        self.names = self.loaded['joint_names']; self._model = model
        self.coord_indices = np.asarray([qstart[indices[name]] for name in self.names], dtype=np.int32)
        self.dof_indices = np.asarray([dstart[indices[name]] for name in self.names], dtype=np.int32)
        self._coords = wp.array(self.coord_indices, dtype=int, device=model.device)
        self._dofs = wp.array(self.dof_indices, dtype=int, device=model.device)
        self._target = wp.zeros((1,len(self.names)), dtype=float, device=model.device)
        self._gain = wp.zeros(len(self.names), dtype=float, device=model.device)
        self._targets = wp.array(self.loaded['targets'], dtype=float, device=model.device)
        self._gains = wp.array(self.loaded['gains'], dtype=float, device=model.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return len(self.names)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if model is not self._model or jacobian_mode != ik.IKJacobianType.ANALYTIC:
            raise ValueError('Pose keyframes require bound model and analytic Jacobians')

    def update(self, frame):
        if 0 <= frame < len(self.loaded['targets']):
            self._target.assign(self.loaded['targets'][frame:frame+1])
            self._gain.assign(self.loaded['gains'][frame])
        else:
            self._gain.zero_()

    def gather(self, frame):
        wp.launch(_gather, dim=len(self.names), inputs=[self._targets,self._gains,frame,self._target,self._gain], device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_temporal_residual, dim=(joint_q.shape[0],len(self.names)), inputs=[joint_q,
            self._coords,self._target,self._gain,start_idx,problem_idx,residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_temporal_jacobian, dim=(joint_q.shape[0],len(self.names)), inputs=[self._dofs,
            self._gain,start_idx,jacobian], device=self.device)
