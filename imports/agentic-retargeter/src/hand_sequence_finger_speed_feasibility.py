"""Explicit original3rad/s decode-only gate for existing spline backtracking."""
from __future__ import annotations
import numpy as np


def validate_speed_feasibility(option, *, initialization, soft_policy, backtracking):
    if option is None or option is False:
        return False
    if option is not True:
        raise ValueError('enforce_finger_speed_feasibility must be Boolean')
    if (initialization is None or soft_policy is None or backtracking is None
            or initialization['config']['limit_rad_s']!=3.
            or soft_policy['limit_rad_s']!=3.):
        raise ValueError('Hard finger gate requires3rad/s QP initialization, matching soft policy and feasible spline backtracking')
    return True


def decoded_finger_speed_status(value,hands,timestamps):
    """Check actual decoded native float states on the unchanged source clock.

    The comparison is <=3.0rad/s with no positive tolerance. Controls are never
    clipped or projected here. Existing backtracking decides reduction/restore.
    """
    time=np.asarray(timestamps,dtype=float)
    if time.shape!=(len(value),)or len(time)<2 or not np.isfinite(time).all()or np.any(np.diff(time)<=0):
        raise ValueError('Native finger speed gate needs original finite source timestamps')
    rows={}
    for i,(side,h)in enumerate(hands.items()):
        q=(h['source_finger']+value[:,i,6:]).detach().cpu().double().numpy()
        if not np.isfinite(q).all():
            return dict(feasible=False,finite=False,limit_rad_s=3.,maximum_rad_s=None,hands={})
        speed=np.abs(np.diff(q,axis=0)/np.diff(time)[:,None]);maximum=float(speed.max())
        rows[side]=dict(maximum_rad_s=maximum,violating_samples=int(np.sum(speed>3.)),feasible=maximum<=3.)
    return dict(feasible=all(x['feasible']for x in rows.values()),finite=True,limit_rad_s=3.,
        maximum_rad_s=max(x['maximum_rad_s']for x in rows.values()),hands=rows,
        interval_count=len(time)-1,positive_tolerance_rad_s=0.)
