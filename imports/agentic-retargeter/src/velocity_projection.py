"""Separable native hinge-speed projection; never a contact/collision certificate."""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint


def project(qpos, mask, lower, upper, gate, timestamps, *, deadline=None):
    """Find nearby bounded samples under native speed limits and frozen samples.

    The convex per-hinge objective is squared displacement from the proposal.
    An exactly feasible interval-propagation seed is retained if the optional
    least-squares refinement reaches its budget. No wrapping of hinge angles.
    """
    q = np.asarray(qpos, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    if not gate.configured:
        raise ValueError("Velocity projection requires a configured native speed gate")
    if mask.shape != q.shape or lo.shape != q.shape or hi.shape != q.shape:
        raise ValueError("Projection bounds and mask must match qpos")
    if not np.isfinite(q).all() or np.any(lo > hi) or ts.shape != (len(q),) or np.any(np.diff(ts) <= 0):
        raise ValueError("Invalid projection values, bounds or timestamps")
    out = q.copy()
    rows = []
    for joint in gate.hinges:
        a, name = joint["address"], joint["joint"]
        limit = gate.overrides.get(name, gate.default_limit)
        step = np.diff(ts) * limit * 0.9999
        low = np.where(mask[:, a], lo[:, a], q[:, a]).copy()
        high = np.where(mask[:, a], hi[:, a], q[:, a]).copy()
        # Frozen samples are equalities; propagate reachability in both directions.
        for i in range(1, len(q)):
            low[i] = max(low[i], low[i-1] - step[i-1])
            high[i] = min(high[i], high[i-1] + step[i-1])
        for i in range(len(q)-2, -1, -1):
            low[i] = max(low[i], low[i+1] - step[i])
            high[i] = min(high[i], high[i+1] + step[i])
        if np.any(low > high + 1e-10):
            raise ValueError("Infeasible frozen samples/corridor under velocity limit: " + name)
        feasible = np.clip(q[:, a], low, high)
        for i in range(1, len(q)):
            feasible[i] = np.clip(feasible[i], max(low[i], feasible[i-1]-step[i-1]),
                                 min(high[i], feasible[i-1]+step[i-1]))
        ids = np.flatnonzero(mask[:, a] & (high-low > 1e-12))
        status = "FEASIBLE_INTERVAL_SEED"
        if len(ids) and (deadline is None or time.monotonic() < deadline):
            fixed = feasible.copy()
            fixed[ids] = 0
            D = np.diff(np.eye(len(q)), axis=0)[:, ids]
            offset = np.diff(fixed)
            constraint = LinearConstraint(D, -step-offset, step-offset)
            target = q[ids, a]
            class BudgetStop(Exception):
                pass
            def callback(_):
                if deadline is not None and time.monotonic() >= deadline:
                    raise BudgetStop()
            try:
                fit = minimize(lambda x: 0.5*np.square(x-target).sum(), feasible[ids],
                               jac=lambda x: x-target, bounds=list(zip(low[ids], high[ids])),
                               constraints=[constraint], method="SLSQP", callback=callback,
                               options={"maxiter": 100, "ftol": 1e-10})
                proposed = fixed.copy()
                proposed[ids] = fit.x
                if (fit.success and np.all(np.abs(np.diff(proposed)) <= step+1e-9)
                        and np.all(proposed >= low-1e-10) and np.all(proposed <= high+1e-10)):
                    feasible, status = proposed, "QUADRATIC_PROJECTION"
            except BudgetStop:
                status = "FEASIBLE_INTERVAL_SEED_TIME_LIMIT"
        out[mask[:, a], a] = feasible[mask[:, a]]
        rows.append(dict(joint=name, status=status,
                         max_change_rad=float(np.max(np.abs(out[:, a]-q[:, a])))))
    result = gate.evaluate(out, 1.0, ts, mask)
    if not result["passed"]:
        raise ValueError("Projection failed independent native velocity gate")
    if not np.array_equal(out[~mask], q[~mask]):
        raise RuntimeError("Projection changed frozen samples")
    return out, dict(status="VELOCITY_FEASIBLE_UNREVIEWED", motion_accepted=False,
                     per_joint=rows, velocity=result,
                     limitation="Contact, source, collision and acceleration may regress; independent audit required.")
