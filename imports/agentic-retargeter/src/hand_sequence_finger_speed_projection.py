"""Explicit initialization-only QP on bounded native finger spline controls."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from .grasp_io import number


def projection_policy(option):
    if option is None or option is False:
        return None
    if not isinstance(option, dict) or set(option) != {'limit_rad_s', 'numerical_margin_rad_s'}:
        raise ValueError('finger_speed_initialization requires limit_rad_s and numerical_margin_rad_s')
    value = {k: number(v, 'finger_speed_initialization.'+k, 1e-12) for k,v in option.items()}
    if value['numerical_margin_rad_s'] >= value['limit_rad_s']:
        raise ValueError('Finger speed numerical margin must be below the gate')
    return value


def project_controls(basis, controls, low, high, timestamps, policy):
    """Minimum trajectory displacement QP, independent per native scalar joint.

    ``q = B u`` and every ``|(B[t+1]-B[t])u/dt| <= limit-margin`` are
    affine. Positive partition-of-unity B and bounded u retain native limits.
    No source/frame/phase arrays are modified.
    """
    B,u=np.asarray(basis,float),np.asarray(controls,float)
    lo,hi=np.asarray(low,float),np.asarray(high,float);t=np.asarray(timestamps,float)
    if (B.ndim!=2 or u.ndim!=2 or B.shape[1]!=u.shape[0] or len(B)<2
            or lo.shape!=(u.shape[1],) or hi.shape!=lo.shape or t.shape!=(len(B),)
            or not all(np.isfinite(a).all()for a in [B,u,lo,hi,t])
            or np.any(np.diff(t)<=0) or np.any(hi<=lo) or np.min(B)<-1e-7
            or not np.allclose(B.sum(1),1.,atol=2e-7,rtol=0)
            or np.any(u<lo-1e-6) or np.any(u>hi+1e-6)):
        raise ValueError('QP requires original finite clock, bounded controls and partition-of-unity basis')
    A=np.diff(B,axis=0)/np.diff(t)[:,None];cap=policy['limit_rad_s']-policy['numerical_margin_rad_s']
    target=B@u;result=u.copy();records=[];ridge=1e-8
    for j in range(u.shape[1]):
        lower=lo[j]+(hi[j]-lo[j])*1e-6;upper=hi[j]-(hi[j]-lo[j])*1e-6
        start=np.clip(u[:,j],lower,upper);before=float(np.max(abs(A@start)))
        if before<=cap:
            row=dict(success=True,iterations=0,message='Already feasible')
        else:
            def fun(x):
                d=B@x-target[:,j];e=x-u[:,j]
                return .5*(d@d+ridge*(e@e))
            def jac(x):return B.T@(B@x-target[:,j])+ridge*(x-u[:,j])
            fit=minimize(fun,start,jac=jac,method='SLSQP',bounds=Bounds(lower,upper),
                constraints=[LinearConstraint(A,-cap,cap)],options=dict(maxiter=500,ftol=1e-12))
            start=fit.x;row=dict(success=bool(fit.success),iterations=int(fit.nit),message=str(fit.message))
        after=float(np.max(abs(A@start)));error=max(after-cap,float(np.max(lower-start)),float(np.max(start-upper)),0.)
        if not row['success'] or not np.isfinite(start).all() or error>1e-7:
            raise ValueError('Finger control QP failed original constraints: '+str(row))
        result[:,j]=start;records.append(dict(row,joint_index=j,before_max_speed_rad_s=before,
            after_max_speed_rad_s=after,trajectory_rms_delta_rad=float(np.sqrt(np.mean((B@start-target[:,j])**2)))))
    return result,records


def initialize_finger_speed(spline,timestamps,policy):
    import torch
    if spline is None or spline.finger_space!='bounded_joint_controls':
        raise ValueError('finger_speed_initialization requires bounded native finger controls')
    started=time.monotonic();old=spline.controls.detach().clone();updated=old.clone();B=spline.basis.detach().cpu().double().numpy();rows={}
    for i,(side,h)in enumerate(spline.hands.items()):
        n=h['low'].numel();lo=h['low'].detach().cpu().double().numpy();hi=h['high'].detach().cpu().double().numpy()
        u=(h['low']+(h['high']-h['low'])*old[:,i,-n:].sigmoid()).detach().cpu().double().numpy()
        result,records=project_controls(B,u,lo,hi,timestamps,policy)
        fraction=(result-lo)/(hi-lo);logits=np.log(fraction/(1-fraction))
        updated[:,i,-n:]=torch.as_tensor(logits,device=old.device,dtype=old.dtype)
        cast_u=h['low']+(h['high']-h['low'])*updated[:,i,-n:].sigmoid()
        cast_q=(spline.basis@cast_u).detach().cpu().double().numpy()
        speed=float(np.max(abs(np.diff(cast_q,axis=0)/np.diff(timestamps)[:,None])))
        if speed>policy['limit_rad_s']+1e-6 or np.any(cast_q<lo-1e-6) or np.any(cast_q>hi+1e-6):
            raise ValueError('Cast finger initialization exceeds original native/speed bounds')
        rows[side]=dict(joints=records,actual_cast_maximum_rad_s=speed,
            original_native_lower=lo.tolist(),original_native_upper=hi.tolist())
    with torch.no_grad():spline.controls.copy_(updated)
    return dict(config=policy,hands=rows,elapsed_seconds=time.monotonic()-started,
        original_frame_count=len(timestamps),source_clock_unchanged=True,wrist_controls_unchanged=True,
        scope='Initialization only. Every original interval/native finger; no phase mask, retiming, source-prior replacement or final optimizer feasibility guarantee.',
        objective='Half summed trajectory displacement squared plus1e-8 control-displacement ridge per joint.',
        motion_accepted=False)
