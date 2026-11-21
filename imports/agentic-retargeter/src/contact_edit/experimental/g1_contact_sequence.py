"""Quarter-frame contact-preserving native interpolation experiment.

The unchanged old residual is the objective. Material gap and opposing-normal
limits are nonlinear inequalities, checked by actual FK before accepting each
step. A linearized SQP constraint or a low objective alone cannot accept a step.
This is an experimental CPU solver, not collision/physics validation.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')

import numpy as np
import mujoco as mj
import osqp
from scipy.sparse import csr_matrix, eye, vstack, triu
from scipy.spatial.transform import Rotation, Slerp

OLD_SCRIPT = Path(__file__).resolve().with_name('g1_sequence_objective.py')
_spec = importlib.util.spec_from_file_location('prior_native_sequence_trial', OLD_SCRIPT)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)


class ConstrainedTrial(old.Trial):
    def __init__(self, config_path, continuation=None):
        # Original targets, residual weights, variable set and bounds are reused.
        super().__init__(config_path)
        constraints = self.spec['contact_constraints']
        assert constraints['enabled'] is True
        assert constraints['reference'] == 'selected_parent_native_linear_at_matching_time'
        for key in ['active_pads_only', 'fixed_original_material_anchors', 'retain_inherited_contact_errors']:
            assert constraints[key] is True
        self.review_window = constraints['review_window']
        assert self.review_window == [self.start-1, self.end+1]
        self.sample_step = float(constraints['sample_step_frames'])
        assert self.sample_step in [.25, .125]
        self.extra_times = np.asarray(constraints.get('extra_constraint_frames', []), dtype=float)
        assert np.isfinite(self.extra_times).all()
        assert np.all(self.extra_times >= self.review_window[0]) and np.all(self.extra_times <= self.review_window[1])
        self.times = np.unique(np.r_[np.arange(self.review_window[0], self.review_window[1]+self.sample_step/2, self.sample_step), self.extra_times])
        self.external_gap_margin = float(constraints['gap_allowance_m'])
        self.external_normal_margin = np.deg2rad(float(constraints['normal_allowance_deg']))
        self.gap_margin = float(constraints['interior_gap_allowance_m'])
        self.normal_margin = np.deg2rad(float(constraints['interior_normal_allowance_deg']))
        assert 0 < self.gap_margin < self.external_gap_margin <= .0005
        assert 0 < self.normal_margin < self.external_normal_margin <= np.deg2rad(2)
        self.max_iterations = int(self.spec.get('max_sqp_iterations', 80))
        assert 1 <= self.max_iterations <= 500
        self.trust_radius = np.deg2rad(float(self.spec.get('trust_radius_deg', 1)))
        assert 0 < self.trust_radius <= self.radius
        self.parent_x = self.parent[self.variable[:, None], self.ads].ravel().copy()
        # Do not nudge parent away from hinge bounds: it is the exact feasible start.
        self.seed = self.parent_x.reshape(len(self.variable), self.D).copy()
        assert np.all(self.parent_x >= self.lo.ravel()) and np.all(self.parent_x <= self.hi.ravel())
        self.lookup = {int(f): i for i, f in enumerate(self.variable)}
        for side in self.sides:
            phases = [old.phase_at(self.desc, side, f) for f in range(*[self.review_window[0], self.review_window[1]+1])]
            def identity(p):
                return {'role': p['role'], 'pairs': sorted(p['pairs'], key=lambda pair: pair['finger'])}
            assert all(old.canonical(identity(p)) == old.canonical(identity(phases[0])) for p in phases), 'Contact definition changed at reviewed join'
        self.samples = []
        self.constraint_labels = []
        for frame in self.times:
            low = int(np.floor(frame)); high = min(low+1, len(self.parent)-1)
            alpha = float(frame-low)
            q = self.parent[low].copy() if alpha == 0 else (1-alpha)*self.parent[low]+alpha*self.parent[high]
            q[3:7] /= np.linalg.norm(q[3:7])
            R = self.native['obj_R'][low].astype(float) if alpha == 0 else Slerp(
                [0, 1], Rotation.from_matrix(self.native['obj_R'][[low, high]]))([alpha]).as_matrix()[0]
            t = (1-alpha)*self.native['obj_t'][low]+alpha*self.native['obj_t'][high]
            weights = [(self.lookup[f], w) for f, w in [(low, 1-alpha), (high, alpha)] if f in self.lookup and w > 0]
            self.samples.append((float(frame), q, R, t, weights))
            for side, p, b in self.pads:
                self.constraint_labels.extend([{'frame': float(frame), 'contact': side+':'+p['finger'], 'quantity': 'gap'},
                                               {'frame': float(frame), 'contact': side+':'+p['finger'], 'quantity': 'normal_cosine'}])
        self.contact_cache = None
        _, _, baseline_rows = self.contact_evaluate(self.parent_x, initialize=True)
        self.baseline_rows = baseline_rows
        self.gap_bounds = np.asarray([r['gap_m']+self.gap_margin for r in baseline_rows])
        self.angle_bounds = np.minimum(np.pi, np.asarray([r['normal_error_rad'] for r in baseline_rows])+self.normal_margin)
        self.cos_bounds = np.cos(self.angle_bounds)
        self.vacuous_normal = self.angle_bounds >= np.pi
        self.contact_cache = None
        self.continuation = Path(continuation).resolve() if continuation else None
        if self.continuation:
            prev = json.loads((self.continuation/'declaration.json').read_text())
            assert prev['input_config'] == old.binding(self.config_path)
            assert prev['script_sha256'] == old.sha(__file__)
            assert prev['objective_script'] == old.binding(OLD_SCRIPT)
            q = np.load(self.continuation/'candidate_UNREVIEWED.npz', allow_pickle=False)['qpos']
            frozen = np.ones_like(q, dtype=bool); frozen[self.variable[:, None], self.ads] = False
            assert np.array_equal(q[frozen], self.parent[frozen])
            self.seed = q[self.variable[:, None], self.ads].copy()
            assert self.feasibility(self.seed.ravel())['passed']

    def contact_evaluate(self, x, initialize=False):
        if not initialize and self.contact_cache is not None and np.array_equal(x, self.contact_cache[0]):
            return self.contact_cache[1:]
        X = x.reshape(len(self.variable), self.D)
        delta_x = X-self.parent_x.reshape(X.shape)
        vals, jac_rows, jac_cols, jac_data, rows = [], [], [], [], []
        contact_index = 0
        for frame, parent_q, R, t, weights in self.samples:
            q = parent_q.copy()
            for f, w in weights:
                q[self.ads] += w*delta_x[f]
            self.fk(q)
            for side, p, body in self.pads:
                point, raw_normal = self.pad(p, body)
                normal_length = np.linalg.norm(raw_normal)
                normal = raw_normal/normal_length
                anchor = np.asarray(p['object_barycentric']) @ self.object_vertices[self.object_faces[p['object_face_id']]]
                on = np.asarray(p['object_outward_normal_local'], dtype=float)
                on = on/np.linalg.norm(on)
                delta = R.T@(point-t)-anchor
                gap = float(np.linalg.norm(delta))
                opposing = -R@on
                dot = float(normal@opposing)
                angle = float(np.arccos(np.clip(dot, -1, 1)))
                rows.append({'frame': frame, 'contact': side+':'+p['finger'], 'gap_m': gap,
                             'normal_error_rad': angle, 'raw_opposing_dot': dot})
                if not initialize:
                    bound = self.gap_bounds[contact_index]
                    # Scale into roughly millimetres, retaining exact squared-distance feasible set.
                    vals.append((bound*bound-gap*gap)/(2*bound*.001))
                    vals.append(1. if self.vacuous_normal[contact_index] else (dot-self.cos_bounds[contact_index])/.02)
                    jp, jr = self.point_jac(point, body)
                    gap_j = -(delta @ R.T @ jp)/(bound*.001)
                    normal_j = (np.eye(3)-np.outer(normal, normal))@(-old.skew(raw_normal)@jr)/normal_length
                    angle_j = np.zeros(self.D) if self.vacuous_normal[contact_index] else opposing@normal_j/.02
                    local_j = np.vstack([gap_j, angle_j])
                    for f, w in weights:
                        rr, cc = np.nonzero(local_j)
                        jac_rows.extend((contact_index*2+rr).tolist())
                        jac_cols.extend((f*self.D+cc).tolist())
                        jac_data.extend((w*local_j[rr, cc]).tolist())
                contact_index += 1
        c = np.asarray(vals)
        A = csr_matrix((jac_data, (jac_rows, jac_cols)), shape=(len(vals), len(x)))
        if not initialize:
            self.contact_cache = (x.copy(), c, A, rows)
        return c, A, rows

    def feasibility(self, x):
        c, _, rows = self.contact_evaluate(x)
        gaps = np.asarray([r['gap_m'] for r in rows])
        angles = np.asarray([r['normal_error_rad'] for r in rows])
        gap_violation = float(np.max(gaps-self.gap_bounds))
        angle_violation = float(np.max(angles-self.angle_bounds))
        bound_violation = float(max(np.max(self.lo.ravel()-x), np.max(x-self.hi.ravel())))
        # Numeric tolerance is much smaller than the predeclared interior margin.
        passed = gap_violation <= 1e-11 and angle_violation <= 1e-10 and bound_violation <= 1e-11
        return {'passed': passed, 'minimum_scaled_slack': float(np.min(c)),
                'max_gap_bound_violation_m': gap_violation, 'max_normal_bound_violation_rad': angle_violation,
                'max_joint_bound_violation_rad': bound_violation,
                'max_gap_regression_mm': float(np.max(gaps-np.asarray([r['gap_m'] for r in self.baseline_rows]))*1000),
                'max_normal_regression_deg': float(np.rad2deg(np.max(angles-np.asarray([r['normal_error_rad'] for r in self.baseline_rows]))))}

    def declaration(self):
        d = super().declaration()
        d['objective_script'] = old.binding(OLD_SCRIPT)
        d['script_sha256'] = old.sha(__file__)
        d['status'] = 'PREDECLARED UNREVIEWED constrained native interpolation experiment'
        d['solver'] = {'method': 'Feasible sparse Gauss-Newton SQP / OSQP; exact nonlinear feasible Armijo line search',
                       'max_iterations': self.max_iterations, 'trust_radius_deg': float(np.rad2deg(self.trust_radius)),
                       'backtracking_factors': [2.**-i for i in range(15)],
                       'linearized_fraction_to_boundary': .95, 'linearized_scaled_slack_reserve': 1e-5,
                       'qp_eps_abs': 1e-8, 'qp_eps_rel': 1e-8, 'qp_max_iterations': 100000,
                       'second_order_restoration': {'max_corrections_per_trial': 4,
                            'objective': 'Minimum Euclidean native hinge correction',
                            'each_correction_box_rad': '0.5 * alpha * max_abs(original_SQP_step)',
                            'combined_step_bound': 'Original trust box around current accepted iterate and original native/correction limits',
                            'acceptance': 'Original SQP directional Armijo condition plus all exact nonlinear constraints; no acceptance of restoration cost alone'},
                       'normal_equivalence': 'Raw normalized-pad opposing dot >= cos(min(pi, baseline clipped-acos error + margin)); pi limit is vacuous'}
        d['contact_constraints'] = {'times': self.times.tolist(), 'source_rate_substeps': int(1/self.sample_step),
                                    'extra_constraint_frames': self.extra_times.tolist(),
                                    'active_contacts': [s+':'+p['finger'] for s,p,b in self.pads],
                                    'gap_regression_limit_mm': self.gap_margin*1000,
                                    'normal_regression_limit_deg': float(np.rad2deg(self.normal_margin)),
                                    'external_gap_regression_limit_mm': self.external_gap_margin*1000,
                                    'external_normal_regression_limit_deg': float(np.rad2deg(self.external_normal_margin)),
                                    'gap_definition': '||R.T @ (world_pad - object_translation) - original_triangle_barycentric_anchor||; integer R is unchanged native matrix',
                                    'enforcement': 'Every accepted step and export must pass actual FK at every listed time, not linearized prediction or penalty loss',
                                    'no_continuous_time_claim': True}
        if self.continuation:
            d['continuation_seed'] = old.binding(self.continuation/'candidate_UNREVIEWED.npz')
        return d

    def check(self):
        result = super().check()
        x = self.seed.ravel()
        c, A, _ = self.contact_evaluate(x)
        rng = np.random.default_rng(20261004)
        checks = []
        for i in range(8):
            u = rng.normal(size=len(x)); u /= np.linalg.norm(u)
            # Squared gaps subtract nearly equal values. A 1e-5-radian
            # directional probe avoids the cancellation observed at 1e-6
            # in Drill without changing acceptance thresholds or Jacobians.
            eps = 1e-5
            fd = (self.contact_evaluate(x+eps*u)[0]-self.contact_evaluate(x-eps*u)[0])/(2*eps)
            pred = A@u
            checks.append({'direction': i, 'epsilon_rad': eps, 'max_abs_error': float(np.abs(fd-pred).max()),
                           'relative_l2': float(np.linalg.norm(fd-pred)/max(np.linalg.norm(fd), 1e-12))})
        assert max(r['max_abs_error'] for r in checks) < 1e-5
        assert max(r['relative_l2'] for r in checks) < 1e-7
        legacy = old.Trial(self.config_path)
        parity = []
        for x_test in [self.parent_x, self.parent_x+rng.normal(size=len(x))*.001]:
            r0, J0 = legacy.evaluate(x_test); r1, J1 = self.evaluate(x_test)
            assert np.array_equal(r0, r1)
            assert (J0-J1).nnz == 0
            parity.append({'residual_exact': True, 'jacobian_exact': True})
        parent_feasibility = self.feasibility(self.parent_x)
        assert parent_feasibility['passed']
        result.update(constraint_directional_jacobian_checks=checks,
                      unchanged_objective_parity=parity, parent_feasibility=parent_feasibility,
                      constraint_shape=list(A.shape), active_contact_samples=len(self.baseline_rows))
        return result

    def export(self, x, out):
        q = self.parent.copy(); q[self.variable[:, None], self.ads] = x.reshape(len(self.variable), self.D)
        frozen = np.ones_like(q, dtype=bool); frozen[self.variable[:, None], self.ads] = False
        assert np.array_equal(q[frozen], self.parent[frozen])
        assert np.array_equal(q[self.keys], self.parent[self.keys])
        assert np.max(np.abs(q-self.parent)) <= self.radius+1e-10
        assert self.feasibility(x)['passed']
        output = {k: v.copy() for k, v in self.native.items() if k not in old.DERIVED_FIELDS}
        for k in list(output):
            if any(word in k.lower() for word in ['status','selected','accepted','review']) or k == 'native_task_errors':
                if not k.startswith('parent_'):
                    output['parent_'+k] = output.pop(k)
        bp, br, gp, gr = [], [], [], []
        for row in q:
            self.fk(row)
            bp.append(self.data.xpos.copy()); br.append(self.data.xquat.copy())
            gp.append(self.data.geom_xpos.copy()); gr.append(self.data.geom_xmat.reshape(-1,3,3).copy())
        output.update(qpos=q, body_pos=np.asarray(bp), body_quat_wxyz=np.asarray(br),
                      geom_pos=np.asarray(gp), geom_R=np.asarray(gr),
                      native_status=np.array('UNREVIEWED contact-constrained interpolation experiment'), selected=np.array(False),
                      parent_native_sha256=np.array(old.sha(self.native_path)),
                      interpolation_descriptor_sha256=np.array(old.sha(self.desc_path)),
                      interpolation_script_sha256=np.array(old.sha(__file__)), fk_cache_valid=np.array(True))
        path = out/'candidate_UNREVIEWED.npz'; np.savez_compressed(path, **output)
        (out/'descriptor.json').write_bytes(self.desc_path.read_bytes())
        return path, q

    def run(self, out):
        assert json.loads((out/'declaration.json').read_text()) == self.declaration(), 'Preflight declaration changed'
        assert json.loads((out/'preflight.json').read_text())['status'] == 'PASS'
        assert not (out/'candidate_UNREVIEWED.npz').exists()
        states = out/'trial_states'; states.mkdir()
        tic = time.monotonic(); x = self.seed.ravel().copy()
        assert self.feasibility(x)['passed']
        history = []; attempts = 0; accepted_count = 0
        initial_r = self.evaluate(x)[0]; initial_cost = float(initial_r@initial_r/2)
        termination = 'bounded_iteration_cap'
        for iteration in range(self.max_iterations):
            r, J = self.evaluate(x); cost = float(r@r/2)
            c, A, _ = self.contact_evaluate(x)
            H = (J.T@J).tocsc(); g = np.asarray(J.T@r)
            scale = max(float(np.max(np.abs(H.diagonal()))), 1.)
            H = H/scale + eye(len(x), format='csc')*1e-8
            matrix = vstack([A, eye(len(x))], format='csc')
            # Keep a tiny strict feasible-side reserve, and restore any negative
            # roundoff slack rather than clipping it to zero. The old clipping
            # allowed a tangent step to repeatedly exceed the nonlinear gate.
            # This numerical reserve is ~1e-5 mm for ordinary gap rows, far below
            # the 0.05 mm interior/external difference. Actual bounds are unchanged.
            lower = np.r_[-.95*c+1e-5, np.maximum(self.lo.ravel()-x, -self.trust_radius)]
            upper = np.r_[np.full(len(c), np.inf), np.minimum(self.hi.ravel()-x, self.trust_radius)]
            qp = osqp.OSQP()
            qp.setup(P=triu(H, format='csc'), q=g/scale, A=matrix, l=lower, u=upper,
                     verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, polish=True)
            sol = qp.solve()
            step = sol.x
            qp_row = {'iteration': iteration, 'cost_before': cost, 'qp_status': sol.info.status,
                      'qp_iterations': int(sol.info.iter), 'qp_primal_residual': float(sol.info.pri_res),
                      'qp_dual_residual': float(sol.info.dua_res), 'trials': []}
            if step is None or not np.isfinite(step).all() or sol.info.status_val not in [1,2]:
                qp_row['reason'] = 'QP failed; feasible checkpoint retained'; history.append(qp_row)
                termination = 'qp_failure'; break
            slope = float(g@step)
            qp_row.update(step_max_rad=float(np.abs(step).max()), objective_directional_derivative=slope)
            if np.abs(step).max() < 1e-8 or slope >= -1e-10:
                qp_row['reason'] = 'No meaningful feasible descent in local model'; history.append(qp_row)
                termination = 'local_step_tolerance'; break
            accepted = False
            for backtrack in range(15):
                alpha = 2.**-backtrack
                candidate = x+alpha*step
                correction_info = None
                for correction in range(5):
                    feasibility = self.feasibility(candidate)
                    cr = self.evaluate(candidate)[0]; new_cost = float(cr@cr/2)
                    descent = new_cost <= cost+1e-4*alpha*slope
                    combined_step_max = float(np.abs(candidate-x).max())
                    combined_slope = float(g@(candidate-x))
                    trust_passed = bool(combined_step_max <= self.trust_radius+1e-11)
                    path = states/f'trial_{attempts:04d}.npz'
                    np.savez_compressed(path, editable_qpos=candidate.reshape(len(self.variable), self.D),
                                        variable_frames=self.variable, editable_addresses=self.ads,
                                        parent_native_sha256=np.array(old.sha(self.native_path)))
                    trial = {'attempt': attempts, 'alpha': alpha, 'correction_index': correction,
                             'restoration_qp': correction_info, 'cost': new_cost, 'exact_feasibility': feasibility,
                             'armijo_descent': descent,
                             'accepted': bool(feasibility['passed'] and descent and trust_passed and combined_slope < 0),
                             'combined_step_max_rad': combined_step_max,
                             'combined_directional_derivative': combined_slope, 'original_trust_region_passed': trust_passed,
                             'checkpoint': old.binding(path)}
                    attempts += 1; qp_row['trials'].append(trial)
                    if trial['accepted']:
                        assert np.abs(candidate-x).max() <= self.trust_radius+1e-11
                        x = candidate; accepted = True; accepted_count += 1
                        break
                    if feasibility['passed'] or correction == 4:
                        break
                    # Curved contact constraints can reject every useful tangent
                    # step although the objective decreases (Maratos effect).
                    # Restore the full trial with minimum native correction,
                    # checking actual FK again before any acceptance.
                    cc, AA, _ = self.contact_evaluate(candidate)
                    correction_radius = .5*alpha*np.abs(step).max()
                    correction_low = np.maximum.reduce([self.lo.ravel()-candidate,
                        x-self.trust_radius-candidate, np.full(len(x), -correction_radius)])
                    correction_high = np.minimum.reduce([self.hi.ravel()-candidate,
                        x+self.trust_radius-candidate, np.full(len(x), correction_radius)])
                    restore = osqp.OSQP()
                    restore.setup(P=eye(len(x), format='csc'), q=np.zeros(len(x)),
                        A=vstack([AA, eye(len(x))], format='csc'),
                        l=np.r_[-cc+1e-5, correction_low], u=np.r_[np.full(len(cc), np.inf), correction_high],
                        verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, polish=True)
                    fixed = restore.solve()
                    correction_info = {'status': fixed.info.status, 'iterations': int(fixed.info.iter),
                        'primal_residual': float(fixed.info.pri_res), 'dual_residual': float(fixed.info.dua_res)}
                    trial['next_restoration_qp'] = correction_info
                    if fixed.x is None or not np.isfinite(fixed.x).all() or fixed.info.status_val not in [1,2]:
                        break
                    correction_info['correction_max_rad'] = float(np.abs(fixed.x).max())
                    candidate = candidate+fixed.x
                if accepted:
                    break
            history.append(qp_row)
            old.write(out/'iteration_history.json', history)
            print(json.dumps({'iteration': iteration, 'accepted': accepted, 'cost': new_cost,
                              'gap_regression_mm': feasibility['max_gap_regression_mm'],
                              'normal_regression_deg': feasibility['max_normal_regression_deg'],
                              'elapsed_s': time.monotonic()-tic}), flush=True)
            if not accepted:
                termination = 'nonlinear_line_search_blocked'; break
            if cost-new_cost <= 1e-8*max(1., cost):
                termination = 'relative_objective_tolerance'; break
        old.write(out/'iteration_history.json', history)
        path, q = self.export(x, out)
        final_r = self.evaluate(x)[0]
        _, _, contact_rows = self.contact_evaluate(x)
        old.write(out/'contact_constraints.json', {'parent': self.baseline_rows, 'candidate': contact_rows,
                   'gap_upper_m': self.gap_bounds.tolist(), 'angle_upper_rad': self.angle_bounds.tolist(),
                   'final_feasibility': self.feasibility(x)})
        result = {'status': 'UNREVIEWED: independent original-mesh/source/temporal checks required',
                  'termination': termination, 'bounded_execution_complete': True,
                  'cap_is_not_task_completion': True, 'iterations': len(history), 'accepted_steps': accepted_count,
                  'candidate_trials_saved': attempts, 'seconds': time.monotonic()-tic,
                  'initial_cost': initial_cost, 'final_cost': float(final_r@final_r/2),
                  'native_sha256': old.sha(path), 'all_noneditable_and_keys_exact': True,
                  'max_native_change_deg': float(np.rad2deg(np.abs(q-self.parent).max())),
                  'final_feasibility': self.feasibility(x),
                  'parent': self.metrics(self.parent), 'candidate': self.metrics(q)}
        old.write(out/'result.json', result)
        print(json.dumps({k: result[k] for k in ['status','termination','iterations','accepted_steps','seconds','native_sha256']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--continue-from')
    args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    trial = ConstrainedTrial(args.config, args.continue_from)
    if args.run:
        trial.run(out)
    else:
        assert not (out/'declaration.json').exists(), 'Use a fresh preflight directory'
        old.write(out/'declaration.json', trial.declaration())
        try:
            old.write(out/'preflight.json', trial.check())
        except Exception as exc:
            old.write(out/'preflight_failure.json', {'error': repr(exc), 'script_sha256': old.sha(__file__)})
            raise
        (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
        (out/'objective_script.py').write_bytes(OLD_SCRIPT.read_bytes())
        print(json.dumps({'out': str(out), 'preflight': 'PASS', 'variables': len(trial.parent_x),
                          'quarter_frame_contact_samples': len(trial.baseline_rows)}), flush=True)


if __name__ == '__main__':
    main()
