"""Low-dimensional coherent hand trajectories without time resampling.

Absolute object-local wrist translation uses an open uniform cubic B-spline.
Finger hinges default to sigmoid-squashed cubic spline logits. An optional
bounded_joint_controls mode instead interpolates bounded native-angle controls,
so the nonlinear sigmoid acts on controls rather than on the dense trajectory.
Both modes preserve original native limits. Wrist rotation is a smooth SO(3) reference fit composed with a cubic
spline tangent correction. The original SOMA trajectory remains the source prior;
the warm trajectory only supplies initialization. No phase is frozen here.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint,minimize,lsq_linear
from scipy.spatial.transform import Rotation,RotationSpline,Slerp
from .hand_sequence_rotation import so3_log


def cubic_basis(timestamps,control_count=17):
    t=np.asarray(timestamps,dtype=np.float64)
    if t.ndim!=1 or len(t)<4 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Need at least four increasing original timestamps')
    if type(control_count) is not int or not 4 <= control_count <= len(t):
        raise ValueError('control_count must be 4..T')
    degree=3
    interior=np.linspace(t[0],t[-1],control_count-degree+1)[1:-1]
    knots=np.r_[np.repeat(t[0],4),interior,np.repeat(t[-1],4)]
    basis=BSpline.design_matrix(t,knots,degree).toarray()
    greville=np.asarray([knots[i+1:i+4].mean() for i in range(control_count)])
    if not np.allclose(basis.sum(1),1) or np.min(basis)<0:
        raise AssertionError('Expected a nonnegative partition-of-unity spline basis')
    return basis,greville,knots


def _position_fit(basis,target,source,bound):
    coeff=np.linalg.lstsq(basis,target,rcond=None)[0]
    results=[]
    for axis in range(3):
        initial=coeff[:,axis]
        lower=source[:,axis]-bound;upper=source[:,axis]+bound
        violation=np.maximum(lower-basis@initial,basis@initial-upper).max()
        record=dict(axis=axis,unconstrained_max_box_violation_m=float(max(0,violation)),projection_needed=bool(violation>1e-8))
        if violation>1e-8:
            result=minimize(lambda c:.5*np.square(basis@c-target[:,axis]).sum(),initial,
                            jac=lambda c:basis.T@(basis@c-target[:,axis]),
                            constraints=[LinearConstraint(basis,lower,upper)],method='SLSQP',
                            options=dict(ftol=1e-11,maxiter=300))
            projected=np.maximum(lower-basis@result.x,basis@result.x-upper).max()
            record.update(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                          projected_max_box_violation_m=float(max(0,projected)))
            if projected > 1e-6:
                raise ValueError('No feasible cubic translation warm fit within original source box')
            coeff[:,axis]=result.x
        results.append(record)
    return coeff,results


def _bounded_finger_fit(basis, target, low, high):
    """Least-squares native-angle fit with convex-hull joint-limit bounds.

    The fitted controls are encoded as logits for unconstrained optimization.
    A small interior offset makes finite logits possible; its actual fit error
    is reported rather than replacing the supplied warm/source trajectory.
    """
    target=np.asarray(target,dtype=float);low=np.asarray(low,dtype=float);high=np.asarray(high,dtype=float)
    if target.ndim!=2 or target.shape[0]!=len(basis) or low.shape!=(target.shape[1],) or high.shape!=low.shape:
        raise ValueError('Bounded finger fit needs frame-by-joint poses and matching native limits')
    if not all(np.isfinite(v).all() for v in (target,low,high)) or np.any(high<=low):
        raise ValueError('Bounded finger fit needs finite poses and increasing native limits')
    if np.any(target<low-1e-6) or np.any(target>high+1e-6):
        raise ValueError('Bounded finger warm pose violates native joint limits')
    normalized=(target-low)/(high-low)
    fractions=[];records=[]
    for joint in range(target.shape[1]):
        fit=lsq_linear(basis,normalized[:,joint],bounds=(0.,1.),tol=1e-11,max_iter=300)
        if not fit.success or not np.isfinite(fit.x).all():
            raise ValueError('Bounded native finger control fit failed: '+str(fit.message))
        fraction=np.clip(fit.x,1e-6,1-1e-6)
        fractions.append(fraction)
        records.append(dict(joint_index=joint,success=bool(fit.success),iterations=int(fit.nit),
            optimality=float(fit.optimality),active_bound_controls=int(np.count_nonzero(fit.active_mask)),
            finite_logit_interior_adjustment_max_rad=float(np.max(np.abs(fraction-fit.x))*(high[joint]-low[joint]))))
    fractions=np.stack(fractions,axis=1)
    logits=np.log(fractions/(1-fractions))
    angles=low+(high-low)*fractions
    return logits,basis@angles,records


class HandSplineParameters:
    """Trainable K x hands x 13 controls; produces source-relative all-frame x."""
    def __init__(self,timestamps,warm,hands,*,device='cpu',dtype=None,control_count=17,position_bound=.15,
                 finger_space='logit_spline'):
        import torch
        from .hand_sequence import _rotation
        self.torch=torch;self._rotation=_rotation
        dtype=dtype or torch.float32
        if finger_space not in ('logit_spline','bounded_joint_controls'):
            raise ValueError('finger_space must be logit_spline or bounded_joint_controls')
        self.finger_space=finger_space
        basis,greville,knots=cubic_basis(timestamps,control_count)
        self.basis=torch.tensor(basis,device=device,dtype=dtype)
        self.hands=hands;self.references={};controls=[];records={}
        times=np.asarray(timestamps,dtype=float)
        for side,h in hands.items():
            wp=np.asarray(warm[f'{side}_wrist_position_object'],dtype=float)
            wr=np.asarray(warm[f'{side}_wrist_rotation_object'],dtype=float)
            q=np.asarray(warm[f'{side}_finger_qpos'],dtype=float)
            sourcep=h['source_position'].detach().cpu().numpy()
            pc,fit_record=_position_fit(basis,wp,sourcep,position_bound)
            low=h['low'].detach().cpu().numpy();high=h['high'].detach().cpu().numpy()
            finger_control_fit=None
            if finger_space=='bounded_joint_controls':
                qc,finger_fit,finger_control_fit=_bounded_finger_fit(basis,q,low,high)
            else:
                fraction=np.clip((q-low)/(high-low),1e-6,1-1e-6)
                logits=np.log(fraction/(1-fraction));qc=np.linalg.lstsq(basis,logits,rcond=None)[0]
                finger_fit=low+(high-low)/(1+np.exp(-np.clip(basis@qc,-700,700)))
            # A C2 SO(3) reference, not per-frame noisy source rotations.
            key_rotations=Slerp(times,Rotation.from_matrix(wr))(greville)
            reference=RotationSpline(greville,key_rotations)(times).as_matrix()
            self.references[side]=torch.tensor(reference,device=device,dtype=dtype)
            controls.append(np.concatenate((pc,np.zeros((control_count,3)),qc),axis=1))
            rot_error=Rotation.from_matrix(reference@wr.transpose(0,2,1)).magnitude()
            pos_error=np.linalg.norm(basis@pc-wp,axis=1)
            joint_error=np.abs(finger_fit-q)
            records[side]=dict(position_mean_m=float(pos_error.mean()),position_max_m=float(pos_error.max()),
                              rotation_mean_rad=float(rot_error.mean()),rotation_max_rad=float(rot_error.max()),
                              finger_mean_abs_rad=float(joint_error.mean()),finger_max_abs_rad=float(joint_error.max()),
                              position_box_constrained_fit=fit_record,
                              original_finger_limits=dict(lower=low.tolist(),upper=high.tolist()))
            if finger_control_fit is not None:
                records[side]['bounded_finger_control_fit']=finger_control_fit
        self.controls=torch.nn.Parameter(torch.tensor(np.stack(controls,1),device=device,dtype=dtype))
        self.metadata=dict(kind='cubic_bspline',control_count=control_count,degree=3,knots_s=knots.tolist(),
                           rotation_reference_times_s=greville.tolist(),frame_count=len(times),
                           trainable_parameter_count=int(self.controls.numel()),
                           dense_equivalent_parameter_count=len(times)*len(hands)*13,
                           warm_fit=records,phase_freezing=False,
                           finger_parameterization='Original limited hinges: low+(high-low)*sigmoid(Bc).',
                           rotation_parameterization='Exp(Bc) composed with a fixed C2 SO(3) warm fit.',
                           position_parameterization='Absolute object-local Bc; no source-offset jitter inheritance.',
                           source_prior='Matching-time raw SOMA; warm fit is initialization only.',
                           subframes='Collision collocation uses interpolated exported integer-frame states, not the spline curve.')
        if finger_space=='bounded_joint_controls':
            self.metadata.update(finger_space=finger_space,
                finger_parameterization='q=B@(low+(high-low)*sigmoid(control_logits)); nonnegative partition-of-unity basis preserves original native limits.',
                finger_initial_fit='Bounded native-angle least squares, with measured finite-logit interior adjustment; original warm/source arrays unchanged.')

    def dense(self):
        import torch
        values=torch.einsum('tk,khd->thd',self.basis,self.controls)
        result=[]
        for index,(side,h) in enumerate(self.hands.items()):
            position=values[:,index,:3]
            rotation=self._rotation(values[:,index,3:6])@self.references[side]
            if self.finger_space=='bounded_joint_controls':
                native_controls=h['low']+(h['high']-h['low'])*self.controls[:,index,6:].sigmoid()
                finger=self.basis@native_controls
            else:
                finger=h['low']+(h['high']-h['low'])*values[:,index,6:].sigmoid()
            relative=so3_log(rotation@h['source_rotation'].transpose(-1,-2))
            result.append(torch.cat((position-h['source_position'],relative,finger-h['source_finger']),1))
        return torch.stack(result,1)
