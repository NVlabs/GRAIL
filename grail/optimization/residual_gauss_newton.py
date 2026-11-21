"""Reusable residual-Jacobian capture and bounded Gauss--Newton subproblem.

No model, geometry, source observation, video ID, or acceptance policy lives here.
The caller owns current branch selection, scalar identity and finite evaluation.
Importing this module does not enable an optimizer or alter LossComputer. Callers
must evaluate proposed poses afresh; a ready direction is not an accepted fit.
"""

import numpy as np


def _reverse(outputs, inputs, *, counts, max_reverse_calls, **kwargs):
    import torch

    counts["autograd_reverse_invocations_started"] += 1
    if counts["autograd_reverse_invocations_started"] > max_reverse_calls:
        raise RuntimeError("Autograd reverse cap exhausted")
    value = torch.autograd.grad(outputs, inputs, **kwargs)[0]
    counts["autograd_reverse_invocations_returned"] += 1
    return value


def capture_vector_jacobian(
    vector,
    parameters,
    *,
    counts,
    checkpoint=None,
    max_outputs,
    max_reverse_calls,
    column_counter="vector_jacobian_columns_returned",
):
    """One pullback plus one reverse per parameter column, with no loss meaning.

    Values may include residuals, signed guards, or arbitrary live diagnostics.
    A disconnected graph is unavailable, never silently replaced by zero.
    A connected but identically zero parameter column naturally returns zeros.
    Unsupported double backward raises with completed columns preserved.
    No row is squared, normalized, weighted or added to an objective here.
    """
    import torch

    if (
        vector.ndim != 1
        or parameters.ndim != 1
        or not len(vector)
        or not len(parameters)
        or len(vector) > max_outputs
    ):
        raise ValueError("Vector/parameter shape or output cap differs")
    if not vector.requires_grad or not parameters.requires_grad:
        raise ValueError("Live connected vector and parameter tensors required")
    if not bool(torch.isfinite(vector).all()) or not bool(torch.isfinite(parameters).all()):
        raise ValueError("Nonfinite vector or parameter")
    if len(parameters) + 1 > max_reverse_calls:
        raise ValueError("Pullback and columns exceed reverse cap")
    cotangent = torch.zeros_like(vector, requires_grad=True)
    counts["differentiable_pullbacks_started"] += 1
    pullback = _reverse(
        vector,
        parameters,
        grad_outputs=cotangent,
        create_graph=True,
        retain_graph=True,
        counts=counts,
        max_reverse_calls=max_reverse_calls,
    )
    counts["differentiable_pullbacks_returned"] += 1
    columns = []
    for j in range(len(parameters)):
        counts["cotangent_column_reverses_started"] += 1
        column = _reverse(
            pullback[j],
            cotangent,
            retain_graph=True,
            counts=counts,
            max_reverse_calls=max_reverse_calls,
        )
        counts["cotangent_column_reverses_returned"] += 1
        if not bool(torch.isfinite(column).all()):
            raise ValueError("Nonfinite vector Jacobian column")
        columns.append(column.detach().cpu().double().numpy())
        counts[column_counter] += 1
        if checkpoint is not None:
            checkpoint(j, np.stack(columns, axis=1))
    return vector.detach().cpu().double().numpy(), np.stack(columns, axis=1)


def capture_scalar_gradient(scalar, parameters, *, counts, max_reverse_calls):
    """One separately counted live scalar reverse; no implied vector-row policy."""
    import torch

    if scalar.ndim != 0 or not scalar.requires_grad or not bool(torch.isfinite(scalar)):
        raise ValueError("Finite live scalar merit required")
    counts["independent_scalar_merit_reverses_started"] += 1
    gradient = _reverse(
        scalar, parameters, retain_graph=True, counts=counts, max_reverse_calls=max_reverse_calls
    )
    counts["independent_scalar_merit_reverses_returned"] += 1
    return gradient.detach().cpu().double().numpy()


def check_objective_gradient(
    objective_residual, objective_jacobian, scalar_gradient, *, identity_atol, identity_rtol
):
    """Pure array identity: caller explicitly supplies only objective rows.

    Constraint outputs must not be passed as residuals. This function performs
    no autograd call, solver, scalar reconstruction or constraint weighting.
    """
    r, J, g = (
        np.asarray(x, dtype=np.float64)
        for x in (objective_residual, objective_jacobian, scalar_gradient)
    )
    if r.ndim != 1 or g.ndim != 1 or not len(r) or not len(g) or J.shape != (len(r), len(g)):
        raise ValueError("Objective residual/Jacobian/gradient shapes differ")
    if (
        not np.isfinite([identity_atol, identity_rtol]).all()
        or identity_atol <= 0
        or identity_rtol < 0
    ):
        raise ValueError("Positive absolute and nonnegative relative identity tolerances required")
    if not np.isfinite(np.r_[r, J.ravel()]).all():
        raise ValueError("Nonfinite objective residual/Jacobian")
    expected = 2 * J.T @ r
    bound = identity_atol + identity_rtol * np.maximum(np.abs(expected), np.abs(g))
    error = np.abs(expected - g)
    report = dict(
        independent_scalar_gradient=g.tolist(),
        residual_gradient=expected.tolist(),
        absolute_error=error.tolist(),
        allowed_error=bound.tolist(),
        maximum_absolute_error=float(error.max(initial=0)),
        maximum_error_coordinate=int(error.argmax()),
        error_quantiles=np.quantile(error, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1]).tolist(),
        error_to_bound_ratios=(error / bound).tolist(),
        maximum_error_to_bound_ratio=float((error / bound).max(initial=0)),
        atol=identity_atol,
        rtol=identity_rtol,
        passed=bool(np.isfinite(g).all() and np.all(error <= bound)),
        numerical_scope="Differentiable pullback may include native float32 accumulation; output J promoted to float64",
    )
    if not report["passed"]:
        failure = ValueError("Residual versus scalar gradient identity unavailable")
        failure.identity_report = report
        raise failure
    return report


def residual_jacobian(
    residual,
    parameters,
    scalar_merit,
    *,
    counts,
    checkpoint=None,
    max_residuals,
    max_reverse_calls,
    identity_atol,
    identity_rtol,
):
    """Compatibility wrapper: every supplied row remains an objective residual."""
    import torch

    # Preserve old pre-capture validation and the numerical reverse order.
    if (
        residual.ndim != 1
        or parameters.ndim != 1
        or not len(residual)
        or not len(parameters)
        or len(residual) > max_residuals
    ):
        raise ValueError("Residual/parameter vector shape or residual cap differs")
    if (
        scalar_merit.ndim != 0
        or not scalar_merit.requires_grad
        or not bool(torch.isfinite(scalar_merit))
    ):
        raise ValueError("Finite live scalar merit required")
    if (
        not np.isfinite([identity_atol, identity_rtol]).all()
        or identity_atol <= 0
        or identity_rtol < 0
    ):
        raise ValueError("Positive absolute and nonnegative relative identity tolerances required")
    if not residual.requires_grad or not parameters.requires_grad:
        raise ValueError("Live connected residual and parameter tensors required")
    if not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(parameters).all()):
        raise ValueError("Nonfinite residual or parameter")
    if len(parameters) + 2 > max_reverse_calls:
        raise ValueError("Pullback, all columns and scalar check exceed reverse cap")
    r, J = capture_vector_jacobian(
        residual,
        parameters,
        counts=counts,
        checkpoint=checkpoint,
        max_outputs=max_residuals,
        max_reverse_calls=max_reverse_calls,
        column_counter="residual_jacobian_columns_returned",
    )
    g = capture_scalar_gradient(
        scalar_merit, parameters, counts=counts, max_reverse_calls=max_reverse_calls
    )
    return (
        r,
        J,
        check_objective_gradient(r, J, g, identity_atol=identity_atol, identity_rtol=identity_rtol),
    )


def solve_quadratic(
    residual,
    jacobian,
    linear_A,
    linear_b,
    *,
    radius,
    blocks,
    trace,
    maxiter=100,
    max_unique=1000,
    tolerance=1e-7,
):
    """One convex quadratic with affine, block-ball and global-ball constraints.

    Each block is {'indices': [...], 'radius': positive}. Linear slack is checked
    in original caller units. SLSQP feasibility is not an optimality certificate.
    The supplied zero step must be feasible; this is not a restoration solver.
    """
    from scipy.optimize import minimize

    r, J, A, b = (np.asarray(x, dtype=np.float64) for x in (residual, jacobian, linear_A, linear_b))
    if r.ndim != 1 or not len(r) or J.ndim != 2 or J.shape[0] != len(r) or J.shape[1] == 0:
        raise ValueError("Residual Jacobian shape differs")
    if (
        int(maxiter) != maxiter
        or int(max_unique) != max_unique
        or min(maxiter, max_unique) <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Positive integer solver budgets and finite positive tolerance required")
    n = J.shape[1]
    if A.ndim != 2 or A.shape[1] != n or b.shape != (len(A),):
        raise ValueError("Affine row shape differs")
    if not np.isfinite(np.r_[r, J.ravel(), A.ravel(), b, radius]).all() or radius <= 0:
        raise ValueError("Finite data and positive trust radius required")
    parsed = []
    for block in blocks:
        raw_ix = np.asarray(block["indices"])
        ix = raw_ix.astype(int)
        rad = float(block["radius"])
        if (
            ix.ndim != 1
            or not np.array_equal(raw_ix, ix)
            or len(set(ix.tolist())) != len(ix)
            or not len(ix)
            or np.any((ix < 0) | (ix >= n))
            or not np.isfinite(rad)
            or rad <= 0
        ):
            raise ValueError("Invalid parameter block ball")
        parsed.append((ix, rad))
    parsed.append((np.arange(n), float(radius)))
    scale = np.maximum(np.linalg.norm(A, axis=1), 1e-12)
    allowed = tolerance * np.maximum(scale, 1)
    if np.any(b < -allowed):
        return dict(status="initial_zero_infeasible", original_unit_affine_slacks=b.tolist())
    H = J.T @ J
    linear = J.T @ r
    ridge = 1e-6 * max(float(np.trace(H)) / n, 1.0)
    M, rhs = A / scale[:, None], b / scale
    trace.update(solver_calls_started=0, solver_calls_returned=0, unique_points=0)
    seen = set()

    def observe(x):
        seen.add(np.asarray(x, dtype=np.float64).tobytes())
        trace["unique_points"] = len(seen)
        if len(seen) > max_unique:
            raise RuntimeError("Quadratic subproblem point budget exhausted")

    def objective(x):
        observe(x)
        # Omit the constant ||r||^2 only from solver evaluation.
        return float(x @ H @ x + 2 * linear @ x + ridge * (x @ x))

    def jac(x):
        observe(x)
        return 2 * (H @ x + linear + ridge * x)

    def ball_rows(x):
        values = []
        rows = []
        for ix, rad in parsed:
            row = np.zeros(n)
            row[ix] = -2 * x[ix] / rad**2
            values.append(1 - float(x[ix] @ x[ix]) / rad**2)
            rows.append(row)
        return np.asarray(values), np.asarray(rows)

    def con(x):
        observe(x)
        return np.r_[rhs - M @ x, ball_rows(x)[0]]

    def cjac(x):
        observe(x)
        return np.r_[-M, ball_rows(x)[1]]

    trace["solver_calls_started"] += 1
    opt = minimize(
        objective,
        np.zeros(n),
        jac=jac,
        method="SLSQP",
        constraints=[dict(type="ineq", fun=con, jac=cjac)],
        options=dict(maxiter=maxiter, ftol=1e-12, disp=False),
    )
    trace["solver_calls_returned"] += 1
    x = np.asarray(opt.x)
    slack = b - A @ x
    balls = ball_rows(x)[0]
    feasible = bool(
        np.isfinite(x).all() and np.all(slack >= -allowed) and np.all(balls >= -tolerance)
    )
    predicted = float(r @ r - (r + J @ x) @ (r + J @ x))
    return dict(
        status=(
            "direction_ready"
            if opt.success and feasible and predicted > 1e-12
            else "no_qualified_direction"
        ),
        direction=x.tolist(),
        radius=radius,
        ridge=ridge,
        ridge_subproblem_only=True,
        baseline_model=float(r @ r),
        candidate_model=float((r + J @ x) @ (r + J @ x)),
        predicted_decrease=predicted,
        directional_derivative=float(2 * linear @ x),
        original_unit_affine_slacks=slack.tolist(),
        original_unit_allowed_slack_error=allowed.tolist(),
        normalized_ball_slacks=balls.tolist(),
        feasible=feasible,
        solver_success=bool(opt.success),
        solver_message=str(opt.message),
        iterations=int(opt.nit),
        counts=dict(trace),
        separate_optimality_certificate=False,
    )


def agreement(residual, jacobian, realized_step, initial_merit, actual_merit):
    """Assess the measured step against the unregularized residual model.

    Native float32 parameter updates should supply their realized coordinates.
    A passing ratio is only a model check, independent of source/physics review.
    """
    r, J, x = (np.asarray(a, dtype=np.float64) for a in (residual, jacobian, realized_step))
    if (
        r.ndim != 1
        or x.ndim != 1
        or not len(r)
        or not len(x)
        or J.shape != (len(r), len(x))
        or not np.isfinite(np.r_[r, J.ravel(), x, initial_merit]).all()
    ):
        raise ValueError(
            "Finite residual model and realized step with compatible dimensions required"
        )
    model = float((r + J @ x) @ (r + J @ x))
    predicted = float(r @ r - model)
    actual = float(initial_merit - actual_merit)
    ratio = actual / predicted if predicted > 1e-12 and np.isfinite(actual) else None
    return dict(
        predicted_current_merit=model,
        predicted_decrease=predicted,
        actual_decrease=actual,
        rho=ratio,
        ridge_excluded=True,
        pass_model_agreement=bool(ratio is not None and ratio >= 0.25 and actual > 0),
    )
