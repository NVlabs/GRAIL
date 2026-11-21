"""Sparse additive relative wrist costs, with analytic robot/object derivatives."""
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from soma_contact_runtime import MaterialObjective
from soma_robot_hand_runtime import HandRotationObjective, _skew, _right_log_inverse


@wp.kernel
def _gather(positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.vec4),
            gains: wp.array2d(dtype=float), frame: wp.array(dtype=int),
            p: wp.array(dtype=wp.vec3), r: wp.array(dtype=wp.vec4),
            pg: wp.array(dtype=float), rg: wp.array(dtype=float)):
    f = frame[0]
    p[0] = positions[f]; r[0] = rotations[f]
    pg[0] = gains[f, 0]; rg[0] = gains[f, 1]


class WristKeyframeGroup:
    def __init__(self, model, loaded, body_labels=None):
        self.loaded, self.device = loaded, model.device
        self.frames = len(loaded['timestamps'])
        labels = list(body_labels) if body_labels is not None else [str(x).rsplit('/', 1)[-1] for x in model.body_label]
        self.hands, self.objectives = [], []
        with wp.ScopedDevice(self.device):
            for side, data in loaded['hands'].items():
                if labels.count(data['wrist_body']) != 1:
                    raise ValueError('Missing/ambiguous authored wrist body')
                link = labels.index(data['wrist_body'])
                position, rotation = MaterialObjective(link, [0., 0., 0.]), HandRotationObjective(link)
                self.hands.append(dict(side=side, data=data, position=position, rotation=rotation,
                    positions=wp.empty(self.frames, dtype=wp.vec3), rotations=wp.empty(self.frames, dtype=wp.vec4),
                    gains=wp.array(data['gains'], dtype=float)))
                self.objectives.extend([position, rotation])

    def update(self, frame, object_rotation, object_position, enabled=True):
        if not 0 <= frame < self.frames:
            raise ValueError('Wrist key frame outside source clock')
        for item in self.hands:
            data = item['data']; gains = data['gains'][frame] if enabled else np.zeros(2)
            item['position'].update(object_rotation @ data['wrist_position_object'][frame]+object_position, gains[0])
            item['rotation'].update(Rotation.from_matrix(object_rotation @ data['wrist_rotation_object'][frame]).as_quat(), gains[1])

    def stage(self, object_rotations, object_positions):
        r, p = np.asarray(object_rotations), np.asarray(object_positions)
        if r.shape != (self.frames, 3, 3) or p.shape != (self.frames, 3) or not np.isfinite(r).all() or not np.isfinite(p).all():
            raise ValueError('Wrist key live object trajectory differs from source clock')
        for item in self.hands:
            data = item['data']
            item['positions'].assign(np.einsum('tij,tj->ti', r, data['wrist_position_object'])+p)
            item['rotations'].assign(Rotation.from_matrix(r @ data['wrist_rotation_object']).as_quat())

    def gather(self, frame_index):
        for item in self.hands:
            p, r = item['position'], item['rotation']
            wp.launch(_gather, dim=1, inputs=[item['positions'], item['rotations'], item['gains'], frame_index,
                p.target_positions, r.target_rotations, p.dynamic_weight, r.dynamic_weight], device=self.device)

    def object_jacobian(self, frame, object_rotation, residual, offsets):
        result = np.zeros((len(residual), 6))
        for item in self.hands:
            data = item['data']; gp, gr = map(float, data['gains'][frame])
            if gp:
                row = offsets[id(item['position'])]
                point = object_rotation @ data['wrist_position_object'][frame]
                result[row:row+3, :3] = gp*np.eye(3)
                result[row:row+3, 3:] = -gp*_skew(point)
            if gr:
                row = offsets[id(item['rotation'])]
                phi = np.asarray(residual[row:row+3])/gr
                result[row:row+3, 3:] = -gr*_right_log_inverse(phi)
        return result
