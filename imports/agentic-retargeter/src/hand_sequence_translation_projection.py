"""Closest-control projection onto immutable affine wrist translation boxes.

Projection operates on spline position coefficients only. Rotation/finger
coefficients and Adam state are untouched. The caller must still decode and
backtrack against all original constraints before accepting the update.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import LinearConstraint, minimize


def affine_translation_blocks(spline, position_bound):
    """Return exact original-frame affine boxes for the two supported decoders."""
    kind=spline.metadata['kind']
    if kind not in ('cubic_bspline','phase_cubic_bspline'):
        raise ValueError('Affine translation projection requires a supported spline decoder')
    basis=spline.basis.detach().cpu().numpy().astype(float)
    n,k=basis.shape
    blocks=[]
    for index,(side,h) in enumerate(spline.hands.items()):
        source=h['source_position'].detach().cpu().numpy().astype(float)
        if kind=='cubic_bspline':
            for axis in range(3):
                blocks.append(dict(side=side,hand_index=index,kind=kind,axis=axis,control_count=k,
                    design=basis.copy(),offset=np.zeros(n),source=source[:,axis],
                    lower=source[:,axis]-position_bound,upper=source[:,axis]+position_bound))
        else:
            w=spline.weights[side].detach().cpu().numpy().astype(float)
            rotation=spline.object_rotation.detach().cpu().numpy().astype(float)
            translation=spline.object_translation.detach().cpu().numpy().astype(float)
            inverse=rotation.transpose(0,2,1)
            aw=np.einsum('t,tab,tk->takb',1-w,inverse,basis).reshape(3*n,3*k)
            ar=np.einsum('t,ab,tk->takb',w,np.eye(3),basis).reshape(3*n,3*k)
            offset=(-(1-w)[:,None]*np.einsum('tij,tj->ti',inverse,translation)).ravel()
            source=source.ravel()
            blocks.append(dict(side=side,hand_index=index,kind=kind,axis=None,control_count=k,
                design=np.concatenate((aw,ar),axis=1),offset=offset,source=source,
                lower=source-position_bound-offset,upper=source+position_bound-offset))
    return blocks


def gather_translation(controls, block):
    i=block['hand_index']
    if block['kind']=='cubic_bspline':
        return controls[:,i,block['axis']].copy()
    return np.r_[controls[:,i,:3].ravel(),controls[:,i,6:9].ravel()]


def scatter_translation(controls, block, value):
    i=block['hand_index'];k=block['control_count']
    if block['kind']=='cubic_bspline':
        controls[:,i,block['axis']]=value
    else:
        controls[:,i,:3]=value[:3*k].reshape(k,3)
        controls[:,i,6:9]=value[3*k:].reshape(k,3)


def box_excess(design, value, lower, upper):
    dense=design@value
    return float(max(0,np.max(lower-dense),np.max(dense-upper)))


def nearest_box_projection(proposal, design, lower, upper, *, max_iterations=100):
    """Convex Euclidean control projection with an explicitly checked solve."""
    proposal=np.asarray(proposal,dtype=float)
    if not all(np.isfinite(v).all()for v in (proposal,design,lower,upper)) or np.any(lower>upper):
        raise ValueError('Projection inputs must be finite with a nonempty ordered box')
    excess=box_excess(design,proposal,lower,upper)
    if excess==0:
        return proposal.copy(),dict(success=True,iterations=0,proposal_excess_m=0.,projected_excess_m=0.,control_l2_correction=0.)
    fit=minimize(lambda x:.5*np.square(x-proposal).sum(),proposal,
                 jac=lambda x:x-proposal,constraints=[LinearConstraint(design,lower,upper)],
                 method='SLSQP',options=dict(maxiter=max_iterations,ftol=1e-12))
    residual=box_excess(design,fit.x,lower,upper)if np.isfinite(fit.x).all()else None
    success=bool(fit.success and residual is not None and residual<=1e-8)
    return fit.x,dict(success=success,iterations=int(fit.nit),message=str(fit.message),
        proposal_excess_m=excess,projected_excess_m=residual,
        control_l2_correction=float(np.linalg.norm(fit.x-proposal))if np.isfinite(fit.x).all()else None)


class AffineTranslationProjector:
    def __init__(self,spline,position_bound):
        self.spline=spline
        self.blocks=affine_translation_blocks(spline,position_bound)

    def project(self):
        """Project the current Adam proposal atomically; failures leave it intact."""
        import torch
        started=time.monotonic()
        proposal=self.spline.controls.detach().cpu().numpy().copy()
        trial=proposal.copy();records=[]
        result=dict(success=False,applied=False,blocks=records,
                    objective='Minimum squared Euclidean change of position controls to the Adam proposal.',
                    fallback='Caller applies original feasible backtracking if projection fails.',
                    geometry_and_source_bounds_unchanged=True)
        try:
            if not np.isfinite(proposal).all():
                raise ValueError('Nonfinite Adam control proposal')
            for block in self.blocks:
                original=gather_translation(proposal,block)
                record=dict(side=block['side'],axis=block['axis'],coupled_world_object=block['kind']=='phase_cubic_bspline')
                value,solve=nearest_box_projection(original,block['design'],block['lower'],block['upper'])
                record.update(solve);record['interior_margin_m']=0.;records.append(record)
                if not solve['success']:
                    raise ValueError('Affine position projection failed: '+solve.get('message','unknown'))
                scatter_translation(trial,block,value)
                # Verify the actual coefficient dtype, not only the float64 QP.
                cast_value=gather_translation(trial,block)
                cast_excess=box_excess(block['design'],cast_value,block['lower'],block['upper'])
                if cast_excess>5e-7:
                    margin=2e-6
                    value,retry=nearest_box_projection(original,block['design'],block['lower']+margin,block['upper']-margin)
                    record['interior_retry']=retry;record['interior_margin_m']=margin
                    if not retry['success']:
                        raise ValueError('Interior roundoff retry failed')
                    scatter_translation(trial,block,value)
                    cast_excess=box_excess(block['design'],gather_translation(trial,block),block['lower'],block['upper'])
                record['cast_coefficient_box_excess_m']=cast_excess
                if cast_excess>1e-6:
                    raise ValueError('Cast projection remains outside original tolerance')
            with torch.no_grad():
                self.spline.controls.copy_(torch.as_tensor(trial,device=self.spline.controls.device,dtype=self.spline.controls.dtype))
            result.update(success=True,applied=not np.array_equal(proposal,trial),
                          projected_blocks=sum(r['proposal_excess_m']>0 for r in records))
        except (ValueError,RuntimeError,FloatingPointError)as error:
            # All work above is private until every block passes validation.
            result['error']=str(error)
        result['elapsed_seconds']=time.monotonic()-started
        return result
