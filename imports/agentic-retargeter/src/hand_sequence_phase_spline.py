"""Phase-specific wrist coordinates on the unmodified source clock.

Free motion is fitted in world coordinates. Active hold/support/slide motion is
fitted in object coordinates. Quintic exterior transitions select between two
live C2 spline poses without changing any role or contact mask. C2 describes the
coordinate transition and spline curves: it does not smooth the supplied object
trajectory, or guarantee C2 native subframe interpolation after export.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

from .hand_sequence_spline import cubic_basis, _bounded_finger_fit
from .hand_sequence_rotation import so3_log


CONTACT_ROLES = ('hold', 'grasp', 'stable_hold', 'support', 'slide')


def blend_coordinates(world_position, world_rotation, relative_position, relative_rotation,
                      object_rotation, object_translation, weight):
    """Blend live physical poses in the original object frame, with SO(3) log.

    The principal logarithm has its usual cut locus at pi. This function adds
    no hard pose switch, projection or detach; both branch gradients remain
    active throughout a coordinate transition.
    """
    from .hand_sequence import _rotation
    inverse = object_rotation.transpose(-1,-2)
    world_in_object = inverse@world_rotation
    delta = so3_log(relative_rotation@world_in_object.transpose(-1,-2))
    rotation = _rotation(weight[...,None]*delta)@world_in_object
    world_p_object = (inverse@(world_position-object_translation)[...,None]).squeeze(-1)
    position = (1-weight[...,None])*world_p_object+weight[...,None]*relative_position
    return position, rotation


def phase_weights(timestamps, active, roles, transition_frames, allowed_transition_roles):
    """Return a fixed coordinate weight; labels and contact activation are inputs.

    The transition occupies only exterior frames. Onsets at the first frame or
    offsets at the last frame need no transition. Overlapping transitions and
    transitions through undeclared roles fail explicitly.
    """
    times = np.asarray(timestamps, dtype=float)
    active = np.asarray(active, dtype=bool)
    roles = np.asarray(roles, dtype=str)
    if active.shape != times.shape or roles.shape != times.shape:
        raise ValueError('Phase labels and active mask must match original timestamps')
    if type(transition_frames) is not int or transition_frames < 1:
        raise ValueError('phase_cubic_bspline requires positive transition_frames')
    if not isinstance(allowed_transition_roles, (list, tuple)) or not allowed_transition_roles:
        raise ValueError('Explicit allowed_transition_roles are required')
    if any(not isinstance(x, str) for x in allowed_transition_roles):
        raise ValueError('Transition roles must be strings')
    contact = active & np.isin(roles, CONTACT_ROLES)
    w = contact.astype(float)
    edges = np.diff(np.r_[False, contact, False].astype(int))
    records = []
    occupied = np.zeros(len(times), dtype=bool)
    for first, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        for direction in ('entry', 'exit'):
            if (direction == 'entry' and first == 0) or (direction == 'exit' and end == len(times)):
                continue
            if direction == 'entry':
                a, b = max(0, first-transition_frames-1), first
                ids = np.arange(a+1, b)
            else:
                a, b = end-1, min(len(times)-1, end+transition_frames)
                ids = np.arange(a+1, b)
            if np.any(contact[ids]) or np.any(occupied[ids]):
                raise ValueError('Overlapping phase coordinate transition windows')
            if any(roles[i] not in allowed_transition_roles for i in ids):
                raise ValueError('Phase transition crosses an undeclared exterior role')
            if b <= a:
                raise ValueError('No duration for phase coordinate transition')
            u = (times[ids]-times[a])/(times[b]-times[a])
            blend = u**3*(10-15*u+6*u*u)
            w[ids] = blend if direction == 'entry' else 1-blend
            occupied[ids] = True
            records.append(dict(direction=direction, endpoint_frames=[int(a), int(b)],
                                exterior_frames=ids.tolist(), endpoint_seconds=times[[a,b]].tolist()))
    return w, contact, records


def _space_fit(basis, target, weight):
    """Fit a coordinate branch only where it contributes; stable nullspace fill."""
    visible = np.flatnonzero(weight > 1e-8)
    if not len(visible):
        # Unused branch has no gradient, but remains a finite replayable state.
        return np.linalg.lstsq(basis, target, rcond=None)[0]
    filled = np.stack([np.interp(np.arange(len(target)), visible, target[visible, j])
                       for j in range(3)], axis=1)
    initial = np.linalg.lstsq(basis, filled, rcond=None)[0]
    weighted = np.sqrt(weight)[:, None]*basis
    ridge = 1e-5
    return np.linalg.lstsq(np.r_[weighted, ridge*np.eye(basis.shape[1])],
                          np.r_[np.sqrt(weight)[:,None]*target, ridge*initial], rcond=None)[0]


def _position_fit(basis, world, relative, obj_r, obj_t, weight, source, bound,
                  contact_fit_weight=None):
    wc = _space_fit(basis, world, 1-weight)
    # Legacy initialization extrapolates the strong-contact branch into its
    # exterior window. The explicit alternative fits the original targets
    # throughout that branch's nonzero coordinate support.
    fit_weight = (weight == 1).astype(float) if contact_fit_weight is None else contact_fit_weight
    rc = _space_fit(basis, relative, fit_weight)
    n, k = basis.shape
    inverse = obj_r.transpose(0,2,1)
    # Flatten controls branch, knot, axis; coupled source boxes stay object local.
    aw = np.einsum('t,tab,tk->takb', 1-weight, inverse, basis).reshape(3*n, 3*k)
    ar = np.einsum('t,ab,tk->takb', weight, np.eye(3), basis).reshape(3*n, 3*k)
    design = np.concatenate((aw, ar), axis=1)
    offset = -(1-weight)[:,None]*np.einsum('tij,tj->ti', inverse, obj_t)
    coeff = np.r_[wc.ravel(), rc.ravel()]
    lower, upper = (source-bound-offset).ravel(), (source+bound-offset).ravel()
    violation = float(np.maximum(lower-design@coeff, design@coeff-upper).max())
    record = dict(unconstrained_max_box_violation_m=max(0., violation), projection_needed=violation>1e-8)
    if violation > 1e-8:
        target = (relative-offset).ravel()
        start = coeff.copy()
        ridge = 1e-8
        def fun(c):
            residual = design@c-target
            return .5*(residual@residual+ridge*np.square(c-start).sum())
        def jac(c):
            return design.T@(design@c-target)+ridge*(c-start)
        fit = minimize(fun, coeff, jac=jac, constraints=[LinearConstraint(design, lower, upper)],
                       method='SLSQP', options=dict(maxiter=500, ftol=1e-11))
        coeff = fit.x
        error = float(np.maximum(lower-design@coeff, design@coeff-upper).max())
        record.update(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),
                      projected_max_box_violation_m=max(0.,error))
        if not np.isfinite(coeff).all() or error > 1e-6:
            raise ValueError('No feasible phase spline translation warm fit within original source box')
    return coeff[:3*k].reshape(k,3), coeff[3*k:].reshape(k,3), record


def _rotation_reference(times, rotations, greville, weight):
    visible = np.flatnonzero(weight > 1e-8)
    if not len(visible):
        visible = np.arange(len(times))
    if len(visible) == 1:
        return np.repeat(rotations[visible],len(times),axis=0)
    key_times = np.clip(greville, times[visible[0]], times[visible[-1]])
    keys = Slerp(times[visible], Rotation.from_matrix(rotations[visible]))(key_times)
    return RotationSpline(greville, keys)(times).as_matrix()


class PhaseHandSplineParameters:
    """K x hands x 19 controls: world pose6, object pose6, native finger7."""
    def __init__(self, timestamps, warm, hands, *, device='cpu', dtype=None, control_count=17,
                 position_bound=.15, finger_space='bounded_joint_controls', transition_frames=6,
                 allowed_transition_roles=None, warm_fit_scope='strong_contact'):
        import torch
        from .hand_sequence import _rotation
        self._rotation = _rotation
        dtype = dtype or torch.float32
        if finger_space != 'bounded_joint_controls':
            raise ValueError('phase_cubic_bspline requires bounded_joint_controls')
        if warm_fit_scope not in ('strong_contact', 'coordinate_support'):
            raise ValueError('warm_fit_scope must be strong_contact or coordinate_support')
        self.finger_space = finger_space
        times = np.asarray(timestamps,dtype=float)
        basis, greville, knots = cubic_basis(times,control_count)
        obj_r = np.asarray(warm['object_R'],dtype=float)
        obj_t = np.asarray(warm['object_trans'],dtype=float)
        if obj_r.shape != (len(times),3,3) or obj_t.shape != (len(times),3):
            raise ValueError('Original object poses must match all original source frames')
        if not np.isfinite(obj_r).all() or not np.isfinite(obj_t).all():
            raise ValueError('Original object poses must be finite')
        if not np.allclose(obj_r@obj_r.transpose(0,2,1),np.eye(3),atol=1e-5) or np.any(np.linalg.det(obj_r)<.99999):
            raise ValueError('Object rotations must be proper physical rotations')
        self.basis = torch.tensor(basis,device=device,dtype=dtype)
        self.object_rotation = torch.tensor(obj_r,device=device,dtype=dtype)
        self.object_translation = torch.tensor(obj_t,device=device,dtype=dtype)
        self.hands = hands
        self.references = {}; self.world_references = {}; self.weights = {}
        controls = []; records = {}; phases = {}
        for side,h in hands.items():
            active = h['active'].detach().cpu().numpy()
            w,contact,transitions = phase_weights(times,active,h['roles'],transition_frames,allowed_transition_roles)
            p = np.asarray(warm[f'{side}_wrist_position_object'],dtype=float)
            r = np.asarray(warm[f'{side}_wrist_rotation_object'],dtype=float)
            q = np.asarray(warm[f'{side}_finger_qpos'],dtype=float)
            world_p = np.einsum('tij,tj->ti',obj_r,p)+obj_t
            world_r = obj_r@r
            fit_weight = w if warm_fit_scope == 'coordinate_support' else None
            wc,rc,fit = _position_fit(basis,world_p,p,obj_r,obj_t,w,h['source_position'].detach().cpu().numpy(),position_bound,
                                     contact_fit_weight=fit_weight)
            wr = _rotation_reference(times,world_r,greville,1-w)
            rr = _rotation_reference(times,r,greville,w if warm_fit_scope == 'coordinate_support' else contact.astype(float))
            self.references[side] = torch.tensor(rr,device=device,dtype=dtype)
            self.world_references[side] = torch.tensor(wr,device=device,dtype=dtype)
            self.weights[side] = torch.tensor(w,device=device,dtype=dtype)
            low = h['low'].detach().cpu().numpy(); high = h['high'].detach().cpu().numpy()
            qc,qfit,qrecord = _bounded_finger_fit(basis,q,low,high)
            controls.append(np.concatenate((wc,np.zeros((control_count,3)),rc,np.zeros((control_count,3)),qc),1))
            records[side] = dict(position_box_constrained_fit=fit,bounded_finger_control_fit=qrecord,
                                 finger_mean_abs_rad=float(abs(qfit-q).mean()),finger_max_abs_rad=float(abs(qfit-q).max()),
                                 original_finger_limits=dict(lower=low.tolist(),upper=high.tolist()))
            phases[side] = dict(object_relative_frames=np.flatnonzero(contact).tolist(),
                               world_only_frames=np.flatnonzero(w==0).tolist(),coordinate_weights=w.tolist(),transitions=transitions,
                               original_phase_labels=list(h['roles']),original_contact_active=active.tolist())
        self.controls = torch.nn.Parameter(torch.tensor(np.stack(controls,1),device=device,dtype=dtype))
        self.metadata = dict(kind='phase_cubic_bspline',control_count=control_count,degree=3,knots_s=knots.tolist(),
            rotation_reference_times_s=greville.tolist(),frame_count=len(times),trainable_parameter_count=int(self.controls.numel()),
            dense_equivalent_parameter_count=len(times)*len(hands)*13,warm_fit=records,phases=phases,
            control_layout=['world_position:0:3','world_rotation_tangent:3:6','object_position:6:9',
                            'object_rotation_tangent:9:12','bounded_finger_logits:12:19'],
            phase_freezing=False,finger_space=finger_space,transition_frames=transition_frames,
            allowed_transition_roles=list(allowed_transition_roles),contact_roles=list(CONTACT_ROLES),
            coordinates='World outside contact; original-object-relative hold/support/slide. No constant-grip assumption.',
            transitions='Quintic C2 coordinate weights, exterior only; contact labels/masks unchanged.',
            continuity_limit='C2 spline and blend reference curves; original object trajectory and exported native interpolation are unchanged.',
            source_prior='Matching-time raw SOMA; original source boxes and native limits remain authoritative.',
            subframes='Collision collocation uses interpolated exported integer-frame states, not spline curves.')
        initial = self.dense().detach()
        if warm_fit_scope != 'strong_contact':
            self.metadata['warm_fit_scope'] = warm_fit_scope
            self.metadata['warm_fit_coordinate_support'] = (
                'Object position and rotation references include original approach/release targets '
                'where the unchanged coordinate weight is positive. No objective or mask change.')
        for i,side in enumerate(hands):
            pp = (initial[:,i,:3]+hands[side]['source_position']).cpu().numpy()
            self.metadata['warm_fit'][side].update(position_mean_m=float(np.linalg.norm(pp-warm[f'{side}_wrist_position_object'],axis=1).mean()),
                position_max_m=float(np.linalg.norm(pp-warm[f'{side}_wrist_position_object'],axis=1).max()),
                initial_rotation_vector_box_max_rad=float(initial[:,i,3:6].abs().max()))

    def dense(self):
        import torch
        values = torch.einsum('tk,khd->thd',self.basis,self.controls)
        result = []
        for index,(side,h) in enumerate(self.hands.items()):
            v = values[:,index]
            w = self.weights[side]
            world_r = self._rotation(v[:,3:6])@self.world_references[side]
            relative_r = self._rotation(v[:,9:12])@self.references[side]
            position,rotation = blend_coordinates(v[:,:3],world_r,v[:,6:9],relative_r,
                self.object_rotation,self.object_translation,w)
            native_controls = h['low']+(h['high']-h['low'])*self.controls[:,index,12:].sigmoid()
            finger = self.basis@native_controls
            result.append(torch.cat((position-h['source_position'],
                so3_log(rotation@h['source_rotation'].transpose(-1,-2)),finger-h['source_finger']),1))
        return torch.stack(result,1)

    def export_arrays(self):
        """Additional arrays needed to replay controls after native export."""
        arrays = {}
        for side in self.hands:
            arrays[f'parameterization_{side}_world_rotation_reference'] = self.world_references[side].detach().cpu().numpy().copy()
            arrays[f'parameterization_{side}_coordinate_weight'] = self.weights[side].detach().cpu().numpy().copy()
        return arrays
