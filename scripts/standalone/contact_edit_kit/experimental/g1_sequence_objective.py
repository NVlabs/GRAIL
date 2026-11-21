"""Configured native, object-relative interpolation experiment (not a kit API).

Use --config config.json --out new_directory to preflight; append --run only
after reading the declaration and preflight. --continue-from previous_output
starts another bounded execution of exactly the same objective, preserving the
original native parent, corridor, target definition and all protected keys.
No collision or source acceptance is performed by this CPU-only experiment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import numpy as np
import mujoco as mj
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.sparse import block_diag, lil_matrix, vstack
from scipy.spatial.transform import Rotation, Slerp


JOINT_SUFFIXES = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                 'wrist_roll', 'wrist_pitch', 'wrist_yaw', 'hand_thumb_0',
                 'hand_thumb_1', 'hand_thumb_2', 'hand_middle_0',
                 'hand_middle_1', 'hand_index_0', 'hand_index_1']
PAD_FIELDS = ['active', 'body_name', 'point_local', 'normal_local',
              'object_face_id', 'object_barycentric', 'object_point_local_m',
              'object_outward_normal_local']
DERIVED_FIELDS = {'qvel', 'qacc', 'qacc_warmstart', 'body_R', 'body_rot',
                  'body_rotmat', 'body_quat', 'body_positions', 'body_world_pos',
                  'body_world_R', 'body_world_quat', 'body_world_transforms',
                  'joint_world_pos', 'joint_world_R', 'joint_world_quat',
                  'joint_world_transforms', 'xpos', 'xmat', 'xquat', 'xipos',
                  'ximat', 'xanchor', 'xaxis', 'geom_xpos', 'geom_xmat',
                  'site_xpos', 'site_xmat', 'cam_xpos', 'cam_xmat', 'light_xpos',
                  'light_xdir', 'subtree_com', 'cvel', 'cacc', 'solver_diagnostics'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def phase_at(desc, side, frame):
    exact = [p for p in desc['phases'] if p['side'] == side and
             p.get('reference_frame') == frame]
    if exact:
        choices = exact
    else:
        choices = [p for p in desc['phases'] if p['side'] == side and
                   p.get('active_frames', p.get('frames', [1, 0]))[0] <= frame <=
                   p.get('active_frames', p.get('frames', [1, 0]))[1]]
    if not choices:
        raise ValueError(f'No explicit phase for {side} at frame {frame}')
    identities = [{p['finger']: {k: p[k] for k in PAD_FIELDS}
                   for p in c['pairs']} for c in choices]
    if any(canonical(z) != canonical(identities[0]) for z in identities[1:]):
        raise ValueError(f'Ambiguous overlapping phase identities: {side}, {frame}')
    return choices[0]


class Trial:
    def __init__(self, config_path, continuation=None):
        self.config_path = Path(config_path).resolve()
        self.spec = json.loads(self.config_path.read_text())
        c = self.spec
        assert c.get('schema_version', 1) == 1
        def resolve(value):
            p = Path(value)
            return p.resolve() if p.is_absolute() else (self.config_path.parent / p).resolve()
        self.native_path = resolve(c['native'])
        self.desc_path = resolve(c['descriptor'])
        self.native = dict(np.load(self.native_path, allow_pickle=False))
        self.parent = self.native['qpos']
        assert self.parent.ndim == 2 and np.isfinite(self.parent).all()
        self.desc = json.loads(self.desc_path.read_text())
        self.xml_path = resolve(c.get('model_xml', self.desc['model_xml']))
        self.object_mesh_path = resolve(c['object_mesh']) if 'object_mesh' in c else None
        if self.object_mesh_path:
            obj = dict(np.load(self.object_mesh_path, allow_pickle=False))
            self.object_vertices = obj['vertices'] * np.asarray(self.native['object_scale']).reshape(-1)
            self.object_faces = obj['faces']
            if 'object_vertices_scaled' in self.native:
                np.testing.assert_allclose(self.object_vertices, self.native['object_vertices_scaled'], atol=1e-9, rtol=0)
                np.testing.assert_array_equal(self.object_faces, self.native['object_faces'])
        else:
            self.object_vertices = self.native['object_vertices_scaled']
            self.object_faces = self.native['object_faces']
        self.m = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.m)
        assert self.m.neq == 0, 'Equality-constrained models are unsupported'
        assert self.parent.shape[1] == self.m.nq
        self.fps = float(self.native['fps'])
        assert np.isfinite(self.fps) and self.fps > 0
        self.start, self.end = c['window']
        assert all(type(x) is int for x in [self.start, self.end])
        assert 2 <= self.start < self.end <= len(self.parent) - 3
        self.keys = np.asarray(c['protected_keys'], dtype=int)
        assert all(type(x) is int for x in c['protected_keys'])
        assert np.array_equal(self.keys, np.unique(self.keys))
        assert self.keys[0] == self.start and self.keys[-1] == self.end
        assert self.start + 1 in self.keys and self.end - 1 in self.keys, 'Protect adjacent endpoint keys to preserve external joins'
        self.frames = np.arange(self.start, self.end + 1)
        self.variable = np.asarray([f for f in self.frames if f not in self.keys])
        assert len(self.variable) > 0
        self.sides = c.get('sides', ['left', 'right'])
        assert self.sides and len(set(self.sides)) == len(self.sides)
        assert set(self.sides).issubset({'left', 'right'})
        self.names = [side + '_' + suffix + '_joint' for side in self.sides for suffix in JOINT_SUFFIXES]
        self.js = np.asarray([self.m.joint(n).id for n in self.names])
        assert np.all(self.m.jnt_type[self.js] == mj.mjtJoint.mjJNT_HINGE)
        assert np.all(self.m.jnt_limited[self.js])
        self.ads = self.m.jnt_qposadr[self.js]
        self.dofs = self.m.jnt_dofadr[self.js]
        self.D = len(self.ads)
        assert len(np.unique(self.ads)) == self.D
        for field, values in [('hinge_qposadr', self.m.jnt_qposadr[self.m.jnt_type == mj.mjtJoint.mjJNT_HINGE]),
                              ('joint_qposadr', self.m.jnt_qposadr), ('joint_dofadr', self.m.jnt_dofadr)]:
            if field in self.native:
                np.testing.assert_array_equal(self.native[field], values)
        if 'joint_names' in self.native:
            np.testing.assert_array_equal(self.native['joint_names'], [self.m.joint(i).name for i in range(self.m.njnt)])
        self.model_snapshot_path = resolve(c['model_snapshot']) if 'model_snapshot' in c else None
        if self.model_snapshot_path:
            snap = dict(np.load(self.model_snapshot_path, allow_pickle=False))
            for name in ['jnt_type', 'jnt_limited', 'jnt_qposadr', 'jnt_dofadr', 'jnt_range']:
                if name in snap:
                    np.testing.assert_array_equal(snap[name], getattr(self.m, name))
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.wrists = [self.m.body(s + '_wrist_yaw_link').id for s in self.sides]
        self.pads = []
        for side in self.sides:
            phases = [phase_at(self.desc, side, int(f)) for f in self.frames]
            identity = {p['finger']: {k: p[k] for k in PAD_FIELDS} for p in phases[0]['pairs']}
            for phase in phases[1:]:
                other = {p['finger']: {k: p[k] for k in PAD_FIELDS} for p in phase['pairs']}
                assert canonical(identity) == canonical(other), f'Contact identity or active phase changes inside window: {side}'
            for p in phases[0]['pairs']:
                if not p['active']:
                    continue
                triangles = self.object_vertices[self.object_faces[p['object_face_id']]]
                bary = np.asarray(p['object_barycentric'])
                assert abs(bary.sum()-1) < 1e-8 and bary.min() > -1e-8
                np.testing.assert_allclose(bary @ triangles, p['object_point_local_m'], atol=1e-8, rtol=0)
                for normal in ['normal_local', 'object_outward_normal_local']:
                    assert abs(np.linalg.norm(p[normal]) - 1) < 1e-5
                self.pads.append((side, p, self.m.body(p['body_name']).id))
        assert self.pads, 'No active material contacts in chosen window'
        self.target_mode = c.get('target_mode', 'rigid_wrist')
        assert self.target_mode in ['rigid_wrist', 'wrist_finger_fk', 'independent_pad']
        self.task_multiplier = float(c.get('task_multiplier', 5))
        assert self.task_multiplier > 0 and np.isfinite(self.task_multiplier)
        self.radius = np.deg2rad(float(c.get('radius_deg', 5)))
        assert 0 < self.radius <= np.deg2rad(5)
        self.make_targets()
        center = self.parent[self.variable[:, None], self.ads].copy()
        self.lo = np.maximum(self.m.jnt_range[self.js, 0], center-self.radius)
        self.hi = np.minimum(self.m.jnt_range[self.js, 1], center+self.radius)
        assert np.all(self.lo < self.hi)
        self.seed = center.copy()
        self.continuation = Path(continuation).resolve() if continuation else None
        if self.continuation:
            old = json.loads((self.continuation/'declaration.json').read_text())
            assert old['input_config'] == binding(self.config_path), 'Continuation must retain exact original config'
            assert old['script_sha256'] == sha(__file__), 'Continuation requires unchanged objective code'
            assert old['parent_native'] == binding(self.native_path)
            assert old['descriptor'] == binding(self.desc_path)
            seed_q = np.load(self.continuation/'candidate_UNREVIEWED.npz', allow_pickle=False)['qpos']
            self.seed = seed_q[self.variable[:, None], self.ads].copy()
        self.seed = np.minimum(np.maximum(self.seed, self.lo+1e-10), self.hi-1e-10)
        self.temp_frames = np.arange(self.start-2, self.end+3)
        self.temporal_weight = .4/.05
        self.T = lil_matrix(((len(self.temp_frames)-2)*self.D, len(self.variable)*self.D))
        lookup = {int(f): i for i, f in enumerate(self.variable)}
        for i in range(len(self.temp_frames)-2):
            for offset, coefficient in [(0, 1), (1, -2), (2, 1)]:
                f = int(self.temp_frames[i+offset])
                if f in lookup:
                    for j in range(self.D):
                        self.T[i*self.D+j, lookup[f]*self.D+j] = self.temporal_weight*coefficient
        self.T = self.T.tocsr()
        self.cached = None
        self.max_nfev = int(c.get('max_nfev', 300))
        assert 1 <= self.max_nfev <= 1000

    def fk(self, q):
        self.data.qpos[:] = q
        mj.mj_forward(self.m, self.data)

    def pad(self, p, b):
        R = self.data.xmat[b].reshape(3, 3)
        return self.data.xpos[b]+R@p['point_local'], R@p['normal_local']

    def make_targets(self):
        wrist_p = [[] for _ in self.wrists]
        wrist_R = [[] for _ in self.wrists]
        pad_p = [[] for _ in self.pads]
        pad_n = [[] for _ in self.pads]
        rigid_p = [[] for _ in self.pads]
        rigid_n = [[] for _ in self.pads]
        for f in self.keys:
            self.fk(self.parent[f])
            R, t = self.native['obj_R'][f], self.native['obj_t'][f]
            for j, b in enumerate(self.wrists):
                wrist_p[j].append(R.T@(self.data.xpos[b]-t))
                wrist_R[j].append(R.T@self.data.xmat[b].reshape(3, 3))
            for j, (side, p, b) in enumerate(self.pads):
                point, normal = self.pad(p, b)
                pad_p[j].append(R.T@(point-t))
                pad_n[j].append(R.T@normal)
                wb = self.wrists[self.sides.index(side)]
                wR = self.data.xmat[wb].reshape(3, 3)
                rigid_p[j].append(wR.T@(point-self.data.xpos[wb]))
                rigid_n[j].append(wR.T@normal)
        self.rigidity = []
        for j, (side, p, _) in enumerate(self.pads):
            ps, ns = np.asarray(rigid_p[j]), np.asarray(rigid_n[j])
            ep = float(np.max(np.linalg.norm(ps-ps[0], axis=1)))
            en = float(np.max(np.linalg.norm(ns-ns[0], axis=1)))
            self.rigidity.append({'side': side, 'finger': p['finger'], 'position_key_variation_m': ep, 'normal_key_variation': en})
            if self.target_mode == 'rigid_wrist':
                assert ep < 1e-9 and en < 1e-9, f'Rigid pad assumption failed: {side} {p["finger"]}, {ep} m. Choose another hold window or explicitly declare wrist_finger_fk.'
        pos_curves = [PchipInterpolator(self.keys, np.asarray(p), axis=0) for p in wrist_p]
        rot_curves = [Slerp(self.keys, Rotation.from_matrix(r)) for r in wrist_R]
        pp_curves = [PchipInterpolator(self.keys, np.asarray(p), axis=0) for p in pad_p]
        pn_curves = [PchipInterpolator(self.keys, np.asarray(n), axis=0) for n in pad_n]
        finger_ids = np.asarray([j for j, n in enumerate(self.names) if '_hand_' in n])
        finger_curve = PchipInterpolator(self.keys, self.parent[self.keys[:, None], self.ads[finger_ids]], axis=0)
        finger_check_frames = np.linspace(self.start, self.end, (self.end-self.start)*16+1)
        finger_check_values = finger_curve(finger_check_frames)
        finger_ranges = self.m.jnt_range[self.js[finger_ids]]
        assert np.all(finger_check_values >= finger_ranges[:, 0]-1e-9)
        assert np.all(finger_check_values <= finger_ranges[:, 1]+1e-9)
        self.finger_target_limit_check = {'sample_count': len(finger_check_frames),
                                          'substeps_per_frame': 16, 'minimum_limit_margin_rad': float(np.min(
                                              np.minimum(finger_check_values-finger_ranges[:, 0],
                                                         finger_ranges[:, 1]-finger_check_values))),
                                          'sampled_limits_passed': True,
                                          'construction': 'Component-wise shape-preserving PCHIP of bounded native hinge keys; FK is evaluated at each source frame.'}
        self.targets = {}
        for f in self.frames:
            R, t = self.native['obj_R'][f], self.native['obj_t'][f]
            wp = [R@p(f)+t for p in pos_curves]
            wr = [R@s([f]).as_matrix()[0] for s in rot_curves]
            if self.target_mode == 'independent_pad':
                pp = [R@p(f)+t for p in pp_curves]
                pn = [n(f) for n in pn_curves]
                pn = [R@(n/np.linalg.norm(n)) for n in pn]
            else:
                rp, rn = [p[0] for p in rigid_p], [n[0] for n in rigid_n]
                if self.target_mode == 'wrist_finger_fk':
                    q = self.parent[f].copy()
                    q[self.ads[finger_ids]] = finger_curve(f)
                    self.fk(q)
                    rp, rn = [], []
                    for side, p, b in self.pads:
                        point, normal = self.pad(p, b)
                        wb = self.wrists[self.sides.index(side)]
                        wR = self.data.xmat[wb].reshape(3, 3)
                        rp.append(wR.T@(point-self.data.xpos[wb]))
                        rn.append(wR.T@normal)
                pp = [wp[self.sides.index(side)]+wr[self.sides.index(side)]@rp[j] for j, (side, _, _) in enumerate(self.pads)]
                pn = [wr[self.sides.index(side)]@rn[j] for j, (side, _, _) in enumerate(self.pads)]
            self.targets[int(f)] = (wp, wr, pp, pn)

    def point_jac(self, point, body):
        mj.mj_jac(self.m, self.data, self.jacp, self.jacr, point, body)
        return self.jacp[:, self.dofs].copy(), self.jacr[:, self.dofs].copy()

    def local(self, x, f):
        q = self.parent[f].copy()
        q[self.ads] = x
        self.fk(q)
        wp, wr, pp, pn = self.targets[f]
        residual, jac = [], []
        for j, b in enumerate(self.wrists):
            pos, rot = self.data.xpos[b], self.data.xmat[b].reshape(3, 3)
            jp, jr = self.point_jac(pos, b)
            residual.extend([(pos-wp[j])/.005, (rot-wr[j]).ravel()/.08])
            jac.extend([jp/.005, np.stack([-skew(rot[:, k])@jr for k in range(3)], axis=1).reshape(9, self.D)/.08])
        for j, (_, p, b) in enumerate(self.pads):
            pos, normal = self.pad(p, b)
            jp, jr = self.point_jac(pos, b)
            residual.extend([(pos-pp[j])/.003, (normal-pn[j])/.12])
            jac.extend([jp/.003, -skew(normal)@jr/.12])
        residual = [self.task_multiplier*r for r in residual]
        jac = [self.task_multiplier*j for j in jac]
        prior = .25/np.deg2rad(5.)
        residual.append(prior*(x-self.parent[f, self.ads]))
        jac.append(prior*np.eye(self.D))
        return np.concatenate(residual), np.vstack(jac)

    def evaluate(self, x):
        if self.cached is not None and np.array_equal(x, self.cached[0]):
            return self.cached[1:]
        X = x.reshape(len(self.variable), self.D)
        parts = [self.local(xx, int(f)) for xx, f in zip(X, self.variable)]
        ext = self.parent[self.temp_frames[:, None], self.ads].copy()
        ext[self.variable-self.temp_frames[0]] = X
        r = np.r_[np.concatenate([p[0] for p in parts]), (self.temporal_weight*np.diff(ext, n=2, axis=0)).ravel()]
        J = vstack([block_diag([p[1] for p in parts], format='csr'), self.T], format='csr')
        self.cached = (x.copy(), r, J)
        return r, J

    def declaration(self):
        value = {'status': 'PREDECLARED UNREVIEWED native interpolation experiment',
                 'input_config': binding(self.config_path), 'parent_native': binding(self.native_path),
                 'descriptor': binding(self.desc_path), 'model_xml': binding(self.xml_path),
                 'script_sha256': sha(__file__), 'source_frames': len(self.parent), 'fps': self.fps,
                 'editable_joint_names': self.names, 'editable_qpos_addresses': self.ads.tolist(),
                 'window': [self.start, self.end], 'protected_keyframes': self.keys.tolist(),
                 'variable_frames': self.variable.tolist(), 'target_mode': self.target_mode,
                 'key_rigidity_evidence': self.rigidity, 'task_residual_multiplier': self.task_multiplier,
                 'finger_target_limit_check': self.finger_target_limit_check,
                 'corridor_deg': float(np.rad2deg(self.radius)), 'native_limits_enforced': True,
                 'wrist_targets': 'Object-relative PCHIP position and SLERP orientation using the same protected native key poses as curve trials.',
                 'pad_targets': {'rigid_wrist': 'Apply the same interpolated wrist rigid transform to the verified fixed wrist-relative key pad pose.',
                                 'wrist_finger_fk': 'PCHIP native finger hinge values at protected keys, reconstruct exact relative pad FK, then apply the same interpolated wrist transform.',
                                 'independent_pad': 'Diagnostic legacy control: independently PCHIP achieved object-local pad positions and normals.'}[self.target_mode],
                 'material_targets': 'Original triangle/barycentric anchors and normals remain unchanged; fitting preserves achieved key offsets and is not an inherited-gap repair.',
                 'residual_scales': {'wrist_position_m': .005, 'wrist_rotation_matrix': .08, 'pad_position_m': .003, 'pad_normal': .12,
                                     'joint_prior': '.25/radians(5)', 'second_difference': '.4/.05'},
                 'temporal_context': [int(self.temp_frames[0]), int(self.temp_frames[-1])],
                 'solver': {'max_nfev': self.max_nfev, 'method': 'trf/lsmr analytic sparse native Jacobian', 'ftol': 1e-8, 'xtol': 1e-9, 'gtol': 1e-7},
                 'unchecked': ['original-mesh intersections', 'fixed source rays/depth', 'continuous collision clearance', 'dynamics'],
                 'comparison_limit': 'Coupled task-space tracking and second-difference regularization form one method; compare all contact, source and hand-motion metrics independently.'}
        if self.model_snapshot_path:
            value['model_snapshot'] = binding(self.model_snapshot_path)
        if self.object_mesh_path:
            value['original_object_mesh'] = binding(self.object_mesh_path)
        if self.continuation:
            value['continuation_seed'] = binding(self.continuation/'candidate_UNREVIEWED.npz')
        return value

    def check(self):
        x = self.seed.ravel()
        r, J = self.evaluate(x)
        rng = np.random.default_rng(20261003)
        rows = []
        for i in range(8):
            direction = rng.normal(size=len(x))
            direction /= np.linalg.norm(direction)
            eps = 1e-6
            fd = (self.evaluate(x+eps*direction)[0]-self.evaluate(x-eps*direction)[0])/(2*eps)
            analytic = J@direction
            rows.append({'direction': i, 'max_abs_error': float(np.max(np.abs(fd-analytic))),
                         'relative_l2': float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(fd), 1e-12))})
        assert max(x['max_abs_error'] for x in rows) < 1e-5
        assert max(x['relative_l2'] for x in rows) < 1e-7
        return {'status': 'PASS', 'directional_jacobian_checks': rows, 'initial_cost': float(r@r/2),
                'residual_shape': list(r.shape), 'jacobian_shape': list(J.shape), 'targets_fixed_before_solve': True}

    def metrics(self, q):
        rows = []
        for f in self.frames:
            self.fk(q[f])
            R, t = self.native['obj_R'][f], self.native['obj_t'][f]
            for side, p, b in self.pads:
                pos, normal = self.pad(p, b)
                target = R@p['object_point_local_m']+t
                on = R@p['object_outward_normal_local']
                rows.append({'frame': int(f), 'side': side, 'finger': p['finger'], 'gap_mm': float(np.linalg.norm(pos-target)*1000),
                             'normal_error_deg': float(np.rad2deg(np.arccos(np.clip(normal@(-on), -1, 1))))})
        v = np.diff(q[self.temp_frames[:, None], self.ads], axis=0)*self.fps
        acc = np.diff(v, axis=0)*self.fps
        jerk = np.diff(acc, axis=0)*self.fps
        return {'fixed_material_rows': rows, 'window_including_join_derivatives': {
            name: {'rms': float(np.sqrt(np.mean(a*a))), 'max_abs': float(np.max(np.abs(a)))}
            for name, a in [('velocity_rad_s', v), ('acceleration_rad_s2', acc), ('jerk_rad_s3', jerk)]}}

    def run(self, out):
        assert json.loads((out/'declaration.json').read_text()) == self.declaration(), 'Preflight declaration or code changed'
        assert json.loads((out/'preflight.json').read_text())['status'] == 'PASS'
        assert not (out/'candidate_UNREVIEWED.npz').exists()
        tic = time.monotonic()
        fit = least_squares(lambda x: self.evaluate(x)[0], self.seed.ravel(), jac=lambda x: self.evaluate(x)[1],
                            bounds=(self.lo.ravel(), self.hi.ravel()), method='trf', tr_solver='lsmr', max_nfev=self.max_nfev,
                            ftol=1e-8, xtol=1e-9, gtol=1e-7, x_scale='jac', tr_options={'atol': 1e-10, 'btol': 1e-10, 'maxiter': 300})
        q = self.parent.copy()
        q[self.variable[:, None], self.ads] = fit.x.reshape(len(self.variable), self.D)
        frozen = np.ones_like(q, dtype=bool)
        frozen[self.variable[:, None], self.ads] = False
        assert np.array_equal(q[frozen], self.parent[frozen])
        assert np.array_equal(q[self.keys], self.parent[self.keys])
        assert np.max(np.abs(q-self.parent)) <= self.radius+1e-10
        assert np.all(q[self.variable[:, None], self.ads] >= self.lo-1e-10)
        assert np.all(q[self.variable[:, None], self.ads] <= self.hi+1e-10)
        output = {k: v.copy() for k, v in self.native.items() if k not in DERIVED_FIELDS}
        for k in list(output):
            if any(word in k.lower() for word in ['status', 'selected', 'accepted', 'review']) or k == 'native_task_errors':
                if not k.startswith('parent_'):
                    output['parent_'+k] = output.pop(k)
        bp, br, gp, gr = [], [], [], []
        for row in q:
            self.fk(row)
            bp.append(self.data.xpos.copy())
            br.append(self.data.xquat.copy())
            gp.append(self.data.geom_xpos.copy())
            gr.append(self.data.geom_xmat.reshape(-1, 3, 3).copy())
        output.update(qpos=q, body_pos=np.asarray(bp), body_quat_wxyz=np.asarray(br), geom_pos=np.asarray(gp), geom_R=np.asarray(gr),
                      native_status=np.array('UNREVIEWED interpolation experiment'), selected=np.array(False),
                      parent_native_sha256=np.array(sha(self.native_path)), interpolation_descriptor_sha256=np.array(sha(self.desc_path)),
                      interpolation_script_sha256=np.array(sha(__file__)), fk_cache_valid=np.array(True))
        path = out/'candidate_UNREVIEWED.npz'
        np.savez_compressed(path, **output)
        (out/'descriptor.json').write_bytes(self.desc_path.read_bytes())
        result = {'status': 'UNREVIEWED: independent geometry/source/temporal review required', 'solver_status': int(fit.status),
                  'message': fit.message, 'nfev': fit.nfev, 'njev': fit.njev, 'cost': float(fit.cost), 'optimality': float(fit.optimality),
                  'seconds': time.monotonic()-tic, 'native_sha256': sha(path), 'all_noneditable_and_keys_exact': True,
                  'max_native_change_deg': float(np.rad2deg(np.abs(q-self.parent).max())),
                  'bounded_execution_complete': True, 'underfit_evaluation_cap': bool(fit.status == 0),
                  'parent': self.metrics(self.parent), 'candidate': self.metrics(q)}
        write(out/'result.json', result)
        print(json.dumps({k: result[k] for k in ['status', 'solver_status', 'nfev', 'seconds', 'optimality', 'max_native_change_deg', 'native_sha256']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--continue-from')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trial = Trial(args.config, args.continue_from)
    if args.run:
        trial.run(out)
    else:
        assert not (out/'declaration.json').exists(), 'Keep each preflight and execution in a new directory'
        write(out/'declaration.json', trial.declaration())
        try:
            write(out/'preflight.json', trial.check())
        except Exception as exc:
            write(out/'preflight_failure.json', {'error': repr(exc), 'script_sha256': sha(__file__)})
            raise
        (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps({'out': str(out), 'preflight': 'PASS', 'variables': len(trial.variable)*trial.D}), flush=True)


if __name__ == '__main__':
    main()
