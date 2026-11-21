"""Sparse sampled-hinge smoothing with hard kinematic derivative constraints.

This CPU QP is a proposal primitive. It does not check contacts, geometry,
continuous interpolation, actuator ratings, or dynamic feasibility.
"""
from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def derivative_operators(timestamps):
    """Factorial-scaled divided differences on actual sample timestamps.

    The order-k recurrence is k * diff(previous) / (t[k:] - t[:-k]).
    This matches the independent temporal gate and is exact on degree-k
    polynomials, including jerk on irregular source timestamps.
    """
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or len(times) < 4 or not np.isfinite(times).all():
        raise ValueError("Temporal projection requires at least four finite timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Temporal timestamps must be strictly increasing")
    operator = sparse.eye(len(times), format="csc")
    result = []
    for order in range(1, 4):
        scale = order / (times[order:] - times[:-order])
        difference = sparse.diags((-scale, scale), (0, 1),
                                  shape=(len(scale), len(scale) + 1))
        operator = (difference @ operator).tocsc()
        if not np.isfinite(operator.data).all():
            raise ValueError("Nonfinite temporal derivative coefficients")
        result.append(operator)
    return tuple(result)


def _positive(value, label, *, allow_zero=False):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(label + " must be a finite scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(label + " must be finite and " + ("nonnegative" if allow_zero else "positive"))
    return value


def _options(options, names):
    if not isinstance(options, Mapping):
        raise ValueError("Temporal projection options must be a mapping")
    required = {"acceleration_limit_rad_s2", "jerk_limit_rad_s3"}
    defaults = dict(position_weight=1., acceleration_weight=1., jerk_weight=1.,
                    position_scale_rad=1., max_seconds=10., per_joint_seconds=1.,
                    max_iterations=20000, preserve_endpoints=True)
    override_keys = ("joint_acceleration_limits_rad_s2", "joint_jerk_limits_rad_s3")
    unknown = set(options) - required - set(defaults) - set(override_keys)
    if unknown or not required <= set(options):
        raise ValueError("Temporal projection requires explicit acceleration/jerk limits; "
                         "unknown options: " + str(sorted(unknown)))
    result = dict(defaults, **options)
    for key in required | (set(defaults) - {"preserve_endpoints", "max_iterations"}):
        result[key] = _positive(result[key], key, allow_zero=key in {"acceleration_weight", "jerk_weight"})
    if type(result["preserve_endpoints"]) is not bool:
        raise ValueError("preserve_endpoints must be boolean")
    iterations = result["max_iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or not 1 <= iterations <= 100000:
        raise ValueError("max_iterations must be an integer in [1, 100000]")
    result["max_iterations"] = int(iterations)
    for key in override_keys:
        overrides = result.get(key, {})
        if not isinstance(overrides, Mapping) or set(overrides) - names:
            raise ValueError(key + " must map native hinge names to positive limits")
        result[key] = {name: _positive(value, key + ": " + name) for name, value in overrides.items()}
    return result


def _metrics(operators, values):
    result = {}
    for label, operator in zip(("velocity_rad_s", "acceleration_rad_s2", "jerk_rad_s3"), operators):
        derivative = np.asarray(operator @ values)
        result["max_" + label] = float(np.max(np.abs(derivative)))
        result["rms_" + label] = float(np.sqrt(np.mean(derivative ** 2)))
    return result


def project(qpos, mask, lower, upper, gate, timestamps, options, *, deadline=None):
    """Smooth native hinges while preserving the declared immutable samples.

    Required options are ``acceleration_limit_rad_s2`` and ``jerk_limit_rad_s3``;
    per-hinge replacements use ``joint_acceleration_limits_rad_s2`` and
    ``joint_jerk_limits_rad_s3``. No acceleration/jerk policy is inferred.
    The objective is position deviation divided by ``position_scale_rad``
    plus acceleration and jerk divided by their respective acceptance limits,
    with positive ``position_weight`` and nonnegative derivative weights.
    First/last samples stay exact by default (``preserve_endpoints=True``).

    ``deadline`` is an absolute ``time.monotonic()`` value. The total call and
    each QP also have bounded ``max_seconds``/``per_joint_seconds`` budgets.
    Infeasibility or an unverified/time-limited solve raises; no partially
    projected trajectory is returned. Requires the optional OSQP dependency.
    """
    started = time.monotonic()
    q = np.asarray(qpos, dtype=float)
    editable = np.asarray(mask)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if not gate.configured:
        raise ValueError("Temporal projection requires a configured native speed gate")
    if q.ndim != 2 or q.shape[1] != gate.nq or not np.isfinite(q).all():
        raise ValueError("Temporal qpos must be finite [frames, model nq]")
    if editable.shape != q.shape or editable.dtype.kind != "b" or lo.shape != q.shape or hi.shape != q.shape:
        raise ValueError("Projection bounds and boolean mask must match qpos")
    if np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi):
        raise ValueError("Invalid projection bounds")
    ts = np.asarray(timestamps, dtype=float)
    if ts.shape != (len(q),):
        raise ValueError("Temporal timestamps must match qpos frames")
    operators = derivative_operators(ts)
    settings = _options(options, {joint["joint"] for joint in gate.hinges})
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("deadline must be finite monotonic time")
    stop = min(started + settings["max_seconds"], deadline if deadline is not None else np.inf)
    editable = editable.copy()
    if settings["preserve_endpoints"]:
        editable[[0, -1]] = False
    hinge_addresses = {joint["address"] for joint in gate.hinges}
    if any(editable[:, a].any() for a in range(q.shape[1]) if a not in hinge_addresses):
        raise ValueError("Temporal projection only edits native hinge coordinates")
    try:
        import osqp
    except ImportError as exc:
        raise ImportError("Temporal projection requires OSQP; install the experimental extra") from exc
    out = q.copy()
    rows = []
    identity = sparse.eye(len(q), format="csc")
    for joint in gate.hinges:
        address, name = joint["address"], joint["joint"]
        movable = editable[:, address]
        low, high = lo[:, address].copy(), hi[:, address].copy()
        if joint["limited"]:
            low = np.maximum(low, joint["limits"][0])
            high = np.minimum(high, joint["limits"][1])
        # Native bounds apply even to frozen samples; do not conceal inherited failures.
        if (np.any(low > high) or np.any(q[~movable, address] < low[~movable])
                or np.any(q[~movable, address] > high[~movable])):
            raise ValueError("Infeasible frozen samples/native corridor: " + name)
        low[~movable] = high[~movable] = q[~movable, address]
        limits = (gate.overrides.get(name, gate.default_limit),
                  settings["joint_acceleration_limits_rad_s2"].get(name, settings["acceleration_limit_rad_s2"]),
                  settings["joint_jerk_limits_rad_s3"].get(name, settings["jerk_limit_rad_s3"]))
        normalized = [operator / limit for operator, limit in zip(operators, limits)]
        before = _metrics(operators, q[:, address])
        candidate = q[:, address].copy()
        status, iterations = "FROZEN_VERIFIED", 0
        if movable.any():
            remaining = stop - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Temporal projection deadline reached before joint: " + name)
            position = settings["position_weight"] / settings["position_scale_rad"] ** 2
            # Eliminate frozen samples exactly, avoiding derivative amplification
            # when numerical equality errors are reset after a solve.
            ids = np.flatnonzero(movable)
            fixed = candidate.copy()
            fixed[ids] = 0.
            hessian = position * identity
            for weight, operator in zip((settings["acceleration_weight"], settings["jerk_weight"]), normalized[1:]):
                hessian = hessian + weight * (operator.T @ operator)
            reduced = hessian[ids][:, ids].tocsc()
            linear = np.asarray(hessian[ids] @ fixed).ravel() - position * q[ids, address]
            derivatives = sparse.vstack(normalized, format="csc")
            offset = derivatives @ fixed
            margin = np.concatenate([
                1. - 1e-5 * (np.asarray(abs(operator) @ movable.astype(float)).ravel() > 0)
                for operator in normalized
            ])
            unconstrained = np.asarray(spsolve(reduced, -linear))
            candidate = fixed.copy()
            candidate[ids] = unconstrained
            unconstrained_ok = (
                np.isfinite(candidate).all() and np.all(candidate >= low)
                and np.all(candidate <= high)
                and np.all(np.abs(derivatives @ candidate) <= margin)
            )
            if unconstrained_ok:
                status = "SPARSE_UNCONSTRAINED_OPTIMUM_VERIFIED"
            else:
                matrix = sparse.vstack([sparse.eye(len(ids)), derivatives[:, ids]], format="csc")
                solver = osqp.OSQP()
                modern = int(osqp.__version__.split(".")[0]) >= 1
                remaining = stop - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Temporal projection deadline reached: " + name)
                solver.setup(P=sparse.triu(2 * reduced, format="csc"), q=2 * linear, A=matrix,
                             l=np.r_[low[ids], -margin - offset],
                             u=np.r_[high[ids], margin - offset], verbose=False,
                             eps_abs=1e-8, eps_rel=1e-8, max_iter=settings["max_iterations"],
                             time_limit=max(1e-6, min(remaining, settings["per_joint_seconds"])),
                             **({"polishing": True} if modern else {"polish": True}))
                solver.warm_start(x=np.clip(unconstrained, low[ids], high[ids]))
                solution = solver.solve(**({"raise_error": False} if modern else {}))
                status, iterations = str(solution.info.status), int(solution.info.iter)
                if solution.info.status_val not in (1, 2):
                    if "time" in status.lower():
                        raise TimeoutError("Temporal projection time limit for " + name)
                    raise ValueError("Infeasible or unverified temporal projection for " + name + ": " + status)
                candidate = fixed.copy()
                candidate[ids] = solution.x
        if not np.isfinite(candidate).all():
            raise ValueError("Nonfinite temporal projection: " + name)
        if np.any(candidate < low - 1e-8) or np.any(candidate > high + 1e-8):
            raise ValueError("Temporal projection violates native corridor: " + name)
        # Only tiny numeric bound errors may be snapped; independently recheck all stencils.
        candidate = np.clip(candidate, low, high)
        after = _metrics(operators, candidate)
        for operator, limit in zip(operators, limits):
            if np.any(np.abs(operator @ candidate) > limit + 1e-9):
                raise ValueError("Infeasible frozen samples or temporal derivative verification failed: " + name)
        out[:, address] = candidate
        rows.append(dict(joint=name, status=status, iterations=iterations,
                         velocity_limit_rad_s=float(limits[0]), acceleration_limit_rad_s2=float(limits[1]),
                         jerk_limit_rad_s3=float(limits[2]), before=before, after=after,
                         max_change_rad=float(np.max(np.abs(candidate - q[:, address])))))
    velocity = gate.evaluate(out, 1., ts, editable)
    if not velocity["passed"] or not np.array_equal(out[~editable], q[~editable]):
        raise ValueError("Temporal projection failed independent velocity/frozen verification")
    return out, dict(status="TEMPORAL_FEASIBLE_UNREVIEWED", motion_accepted=False,
                     selected=False, velocity=velocity, per_joint=rows, options=settings,
                     temporal=dict(passed=True, per_joint=rows,
                                   difference="factorial-scaled divided differences on actual timestamps",
                                   scope="all native hinges, including frozen samples and joins"),
                     elapsed_seconds=time.monotonic() - started,
                     limitation="Sampled kinematic policy, not actuator ratings. Contact, source, geometry, "
                                "continuous interpolation and dynamics require independent review.")
