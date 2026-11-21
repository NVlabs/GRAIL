"""Bounded cubic smoothing-spline proposals for sampled native hinges.

This is a smoothing approximation, not interpolation of every noisy input key.
The internal natural cubic spline is C2; downstream LINEAR replay of its baked
samples is not C2. Contact/FK, collision and source checks remain external.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.linalg import solve
from scipy.optimize import minimize


def smooth(qpos, mask, lower, upper, timestamps, *, smoothing_lambda=1e-5,
           weights=None, max_iterations=300):
    """Return a bounded, unreviewed native-sample smoothing proposal and report.

    Only entries in ``mask`` may change; use false entries for phase milestones,
    protected holds and every root/quaternion coordinate. Bounds should combine
    model hinge limits with the caller's declared corridor. No angle wrapping,
    retiming, contact attraction or post-fit clipping is performed. The caller
    remains responsible for ensuring editable columns are native hinges.

    Objective: mean(w * (s(t)-q)^2) + lambda * mean_time(s''(t)^2),
    with native time in seconds and lambda in seconds**4. Weights may be a
    positive frame vector or full sample matrix. Increasing contact-phase
    fidelity weights encourages retention; it does not enforce material contact.
    A shared velocity projection may be applied afterwards and must be audited
    for changes to acceleration, contact and source fidelity.
    """
    q = np.asarray(qpos, dtype=float)
    editable = np.asarray(mask, dtype=bool)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    if q.ndim != 2 or len(q) < 4 or not np.isfinite(q).all():
        raise ValueError("qpos must contain at least four finite native samples")
    if editable.shape != q.shape or lo.shape != q.shape or hi.shape != q.shape:
        raise ValueError("mask and bounds must match qpos")
    if (np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi)
            or ts.shape != (len(q),) or not np.isfinite(ts).all()
            or np.any(np.diff(ts) <= 0)):
        raise ValueError("Invalid bounds or native timestamps")
    if not np.isfinite(smoothing_lambda) or smoothing_lambda < 0:
        raise ValueError("smoothing_lambda must be finite and nonnegative")
    if np.any(q[~editable] < lo[~editable]) or np.any(q[~editable] > hi[~editable]):
        raise ValueError("Frozen native samples violate supplied bounds")
    if weights is None:
        w = np.ones_like(q)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape == (len(q),):
            w = np.broadcast_to(w[:, None], q.shape)
        if w.shape != q.shape or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("weights must be finite positive frame/sample weights")
    # Cardinal natural cubic B-spline basis: each coordinate is its value at one
    # original timestamp. Two-point Gauss quadrature integrates squared second
    # derivatives exactly on each polynomial span.
    basis = make_interp_spline(ts, np.eye(len(ts)), k=3, bc_type="natural")
    step = np.diff(ts)
    center = (ts[1:] + ts[:-1]) * .5
    samples = (center[:, None] + step[:, None] * np.array([-1., 1.])
               / (2 * np.sqrt(3))).ravel()
    quadrature = np.repeat(step * .5, 2)
    second = basis(samples, nu=2)
    roughness = second.T @ (quadrature[:, None] * second) / (ts[-1] - ts[0])
    out = q.copy()
    reports = []
    for a in np.flatnonzero(editable.any(axis=0)):
        free = np.flatnonzero(editable[:, a] & (hi[:, a] > lo[:, a]))
        fixed = np.setdiff1d(np.arange(len(q)), free)
        out[editable[:, a] & (hi[:, a] == lo[:, a]), a] = lo[
            editable[:, a] & (hi[:, a] == lo[:, a]), a]
        hessian = np.diag(w[:, a] / len(q)) + smoothing_lambda * roughness
        rhs = w[:, a] * q[:, a] / len(q)
        status = "NO_FREE_SAMPLES"
        if len(free):
            H = hessian[np.ix_(free, free)]
            b = rhs[free] - hessian[np.ix_(free, fixed)] @ out[fixed, a]
            optimum = solve(H, b, assume_a="pos")
            if np.all(optimum >= lo[free, a]) and np.all(optimum <= hi[free, a]):
                status = "UNCONSTRAINED_OPTIMUM_WITHIN_BOUNDS"
            else:
                scale = max(float(np.max(np.diag(H))), 1e-12)
                fit = minimize(lambda x: (.5 * x @ H @ x - b @ x) / scale,
                               np.clip(optimum, lo[free, a], hi[free, a]),
                               jac=lambda x: (H @ x - b) / scale,
                               bounds=list(zip(lo[free, a], hi[free, a])),
                               method="L-BFGS-B",
                               options=dict(maxiter=int(max_iterations), ftol=1e-14,
                                            gtol=1e-10, maxls=40))
                if not fit.success:
                    raise RuntimeError(f"Spline bound solve failed at address {a}: {fit.message}")
                optimum = fit.x
                status = "BOUNDED_OPTIMUM"
            out[free, a] = optimum
        before = q[:, a]
        after = out[:, a]
        reports.append(dict(address=int(a), status=status,
                            max_change_rad=float(np.max(np.abs(after-before))),
                            weighted_fidelity=float(np.mean(w[:, a]*(after-before)**2)),
                            parent_spline_acceleration_mean_square=float(before@roughness@before),
                            proposal_spline_acceleration_mean_square=float(after@roughness@after)))
    if not np.array_equal(out[~editable], q[~editable]):
        raise RuntimeError("Spline fit changed frozen native samples")
    if np.any(out < lo-1e-10) or np.any(out > hi+1e-10):
        raise RuntimeError("Spline proposal violated native sample bounds")
    return out, dict(status="SMOOTHING_SPLINE_UNREVIEWED", motion_accepted=False,
                     method="weighted_natural_cubic_smoothing_spline",
                     smoothing_lambda_seconds4=float(smoothing_lambda),
                     authored_continuity="C2 natural cubic spline before baking",
                     exported_interpolation="LINEAR; no C2 claim",
                     protected_native_samples_exact=True,
                     retimed=False, per_joint=reports,
                     limitation="Sample bounds do not bound spline overshoot. No contact, "
                     "collision, source, speed or motion acceptance guarantee.")
