"""Sparse, whole-sequence nonlinear least squares without framewise updates.

The caller owns the state manifold, native limits, residuals, and their Jacobian.
A linear solve uses all tangent variables by default, or an explicitly declared
subset for constrained refinement. Retraction receives a complete zero-padded
increment, and acceptance uses the complete nonlinear objective.
This module has no dependency on Newton, Warp, or the retargeting runtime.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr


@dataclass(frozen=True)
class SparseSequenceConfig:
    """Trust-region-like LM controls; times use monotonic ``perf_counter``."""

    max_iterations: int = 20
    max_seconds: float = 600.0
    initial_damping: float = 1e-3
    min_damping: float = 1e-9
    max_damping: float = 1e12
    max_linear_attempts: int = 6
    max_backtracks: int = 6
    backtrack_factor: float = 0.5
    min_acceptance_ratio: float = 1e-4
    gradient_tolerance: float = 1e-7
    step_tolerance: float = 1e-8
    function_tolerance: float = 1e-6
    column_scale_floor: float = 1e-8
    lsmr_atol: float = 1e-6
    lsmr_btol: float = 1e-6
    lsmr_max_iterations: int = 400
    linearization_residual_atol: float = 1e-8
    linearization_residual_rtol: float = 1e-6


def validate_sparse_sequence_config(
    config: Mapping[str, Any] | SparseSequenceConfig | None,
) -> dict[str, Any]:
    """Return explicit validated controls, rejecting misspelled/ignored keys."""
    if config is None:
        values = {}
    elif isinstance(config, SparseSequenceConfig):
        values = asdict(config)
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        raise ValueError("sparse sequence config must be a mapping or SparseSequenceConfig")
    defaults = asdict(SparseSequenceConfig())
    unknown = sorted(set(values) - {field.name for field in fields(SparseSequenceConfig)})
    if unknown:
        raise ValueError(f"unknown sparse sequence config keys: {unknown}")
    settings = {**defaults, **values}
    integer_keys = (
        "max_iterations", "max_linear_attempts", "max_backtracks", "lsmr_max_iterations",
    )
    for key in integer_keys:
        value = settings[key]
        minimum = 0 if key == "max_backtracks" else 1
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        settings[key] = int(value)
    nonnegative = {
        "gradient_tolerance", "step_tolerance", "function_tolerance", "lsmr_atol",
        "lsmr_btol", "linearization_residual_atol", "linearization_residual_rtol",
    }
    for key in set(settings) - set(integer_keys):
        value = settings[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite and numeric")
        if value < 0 or (key not in nonnegative and value == 0):
            raise ValueError(f"{key} must be {'nonnegative' if key in nonnegative else 'positive'}")
        settings[key] = float(value)
    if not settings["min_damping"] <= settings["initial_damping"] <= settings["max_damping"]:
        raise ValueError("damping must satisfy min_damping <= initial_damping <= max_damping")
    if not 0 < settings["backtrack_factor"] < 1:
        raise ValueError("backtrack_factor must be in (0, 1)")
    if not 0 < settings["min_acceptance_ratio"] < 1:
        raise ValueError("min_acceptance_ratio must be in (0, 1)")
    return settings


def _residual(value: Any, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("residual must be a nonempty one-dimensional array")
    if size is not None and result.size != size:
        raise ValueError("residual dimension changed during nonlinear solve")
    if not np.all(np.isfinite(result)):
        raise ValueError("residual contains nonfinite values")
    # Copy because a runtime may reuse its evaluation buffer on its next call.
    return result.copy()


def _energy(residual: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(0.5 * np.dot(residual, residual))
    if not np.isfinite(value):
        raise ValueError("squared residual energy is nonfinite")
    return value


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:500]}"


def solve_sparse_sequence(
    initial_state: Any,
    linearize: Callable[[Any], tuple[Any, sparse.spmatrix]],
    evaluate: Callable[[Any], Any],
    retract: Callable[[Any, np.ndarray], Any],
    config: Mapping[str, Any] | SparseSequenceConfig | None,
    deadline: float | None = None,
    callback: Callable[[Any, dict[str, Any]], None] | None = None,
    *,
    active_columns: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Minimize ``0.5 * ||evaluate(state)||²`` with simultaneous sparse LM.

    ``linearize(state)`` must return the same complete residual as ``evaluate``
    and a sparse Jacobian whose columns cover the whole-sequence tangent state.
    ``retract(state, step)`` applies the *complete* step, enforcing manifold and
    native limits in the caller. Hooks must not mutate their state argument.
    State is copied before retraction and callbacks, so a rejected candidate
    cannot replace the last accepted state. State must support ``deepcopy``.

    ``deadline`` is an optional absolute ``time.perf_counter()`` timestamp;
    the earlier of it and ``max_seconds`` wins. Hooks/LSMR cannot be preempted,
    so checks occur before and after each bounded operation. Callback runs only
    after accepted steps and receives a snapshot and its iteration record.

    Initial evaluation/config errors raise ValueError. Later failures/deadlines
    return the last accepted finite state with an explicit nonconverged status.
    Technical execution (a real accepted whole-system update, or verified
    first-order stationarity) is reported separately from convergence/quality.
    """
    settings = validate_sparse_sequence_config(config)
    active = None
    if active_columns is not None:
        active = np.asarray(active_columns)
        if (active.ndim != 1 or active.dtype.kind not in "iu" or not len(active)
                or np.any(active < 0) or np.any(active > np.iinfo(np.int64).max)
                or np.any(active[1:] <= active[:-1])):
            raise ValueError("active_columns must be nonempty sorted unique tangent indices")
        active = active.astype(np.int64, copy=True)
    start = perf_counter()
    if deadline is not None and (
        isinstance(deadline, (bool, np.bool_)) or not isinstance(deadline, Real)
        or not np.isfinite(deadline)
    ):
        raise ValueError("deadline must be a finite absolute perf_counter timestamp")
    end = min(start + settings["max_seconds"], float(deadline)) if deadline is not None else start + settings["max_seconds"]
    state = deepcopy(initial_state)
    report: dict[str, Any] = {
        "method": "whole_sequence_sparse_levenberg_marquardt_lsmr",
        "update_scope": "all tangent variables in one simultaneous linear system",
        "framewise_block_updates": False,
        "objective": "0.5 * sum(residual**2)",
        "global_optimum_claimed": False,
        "config": settings,
        "status": "running",
        "converged": False,
        "accepted_steps": 0,
        "rejected_candidates": 0,
        "linear_solves": 0,
        "linearizations": 0,
        "evaluations": 0,
        "callback_count": 0,
        "variable_count": None,
        "residual_count": None,
        "history": [],
        "accepted_energy_history": [],
        "timing_seconds": {"linearize": 0.0, "linear_solve": 0.0, "evaluate": 0.0, "retract": 0.0, "callback": 0.0},
    }
    if active is not None:
        report.update(active_variable_indices=active.tolist(), active_variable_count=len(active),
                      update_scope="declared tangent subset; complete trajectory residual and boundary coupling retained",
                      stationarity_scope="declared active tangent columns only")

    def expired() -> bool:
        return perf_counter() >= end

    def evaluated(candidate: Any, size: int | None = None) -> tuple[np.ndarray, float]:
        tick = perf_counter()
        report["evaluations"] += 1
        try:
            result = _residual(evaluate(candidate), size)
            return result, _energy(result)
        finally:
            report["timing_seconds"]["evaluate"] += perf_counter() - tick

    try:
        residual, energy = evaluated(state)
    except Exception as exc:
        raise ValueError(f"initial state must have a valid finite residual: {_error(exc)}") from exc
    report["residual_count"] = residual.size
    report["initial_energy"] = energy
    report["accepted_energy_history"].append(energy)
    damping = settings["initial_damping"]
    status = "max_iterations"
    failure = None

    for iteration in range(settings["max_iterations"]):
        if expired():
            status = "deadline"
            break
        record: dict[str, Any] = {
            "iteration": iteration,
            "energy_before": energy,
            "accepted": False,
            "attempts": [],
        }
        report["history"].append(record)
        tick = perf_counter()
        report["linearizations"] += 1
        try:
            raw_residual, raw_jacobian = linearize(state)
            model_residual = _residual(raw_residual, residual.size)
            if not sparse.issparse(raw_jacobian):
                raise ValueError("whole-sequence Jacobian must be scipy sparse")
            jacobian = sparse.csr_matrix(raw_jacobian, dtype=np.float64, copy=True)
            jacobian.sum_duplicates()
            jacobian.eliminate_zeros()
            if jacobian.shape[0] != residual.size or jacobian.shape[1] == 0:
                raise ValueError("Jacobian dimensions do not match nonempty tangent/residual vectors")
            if report["variable_count"] not in (None, jacobian.shape[1]):
                raise ValueError("tangent variable dimension changed during nonlinear solve")
            if not np.all(np.isfinite(jacobian.data)):
                raise ValueError("Jacobian contains nonfinite values")
            if not np.allclose(model_residual, residual, atol=settings["linearization_residual_atol"], rtol=settings["linearization_residual_rtol"]):
                raise ValueError("linearize residual differs from complete evaluated residual")
            report["variable_count"] = jacobian.shape[1]
            if active is not None and active[-1] >= jacobian.shape[1]:
                raise ValueError("active tangent index is outside complete Jacobian")
        except Exception as exc:
            status, failure = "linearization_failed", _error(exc)
            break
        finally:
            report["timing_seconds"]["linearize"] += perf_counter() - tick
        row_width = np.diff(jacobian.indptr)
        record.update({
            "jacobian_shape": list(jacobian.shape),
            "jacobian_nnz": int(jacobian.nnz),
            "rows_with_multiple_columns": int(np.count_nonzero(row_width > 1)),
            "maximum_row_columns": int(np.max(row_width, initial=0)),
        })
        if expired():
            status = "deadline"
            break
        with np.errstate(over="ignore", invalid="ignore"):
            column_norm = np.sqrt(np.asarray(jacobian.power(2).sum(axis=0)).ravel())
            column_scale = np.maximum(column_norm, settings["column_scale_floor"])
            gradient = np.asarray(jacobian.T @ residual).ravel()
            scaled_gradient = gradient / column_scale
        if not np.all(np.isfinite(column_scale)) or not np.all(np.isfinite(scaled_gradient)):
            status, failure = "linearization_failed", "nonfinite column scales or gradient"
            break
        gradient_scope = gradient if active is None else gradient[active]
        scaled_scope = scaled_gradient if active is None else scaled_gradient[active]
        record["gradient_linf"] = float(np.max(np.abs(gradient_scope), initial=0.0))
        record["scaled_gradient_linf"] = float(np.max(np.abs(scaled_scope), initial=0.0))
        if record["scaled_gradient_linf"] <= settings["gradient_tolerance"]:
            status = "stationary_initial" if report["accepted_steps"] == 0 else "gradient_tolerance"
            report["converged"] = True
            break
        solve_jacobian = jacobian if active is None else jacobian[:, active]
        solve_scale = column_scale if active is None else column_scale[active]
        scaled_jacobian = solve_jacobian @ sparse.diags(1.0 / solve_scale, format="csr")
        accepted = False
        accepted_step_norm = None
        before = energy
        for linear_attempt in range(settings["max_linear_attempts"]):
            if expired():
                status = "deadline"
                break
            attempt: dict[str, Any] = {
                "linear_attempt": linear_attempt, "damping": damping, "candidates": [],
            }
            record["attempts"].append(attempt)
            tick = perf_counter()
            report["linear_solves"] += 1
            try:
                solved = lsmr(scaled_jacobian, -residual, damp=np.sqrt(damping), atol=settings["lsmr_atol"], btol=settings["lsmr_btol"], maxiter=settings["lsmr_max_iterations"])
                step = solved[0] / solve_scale
                if active is not None:
                    complete_step = np.zeros(jacobian.shape[1], dtype=np.float64)
                    complete_step[active] = step
                    step = complete_step
                if not np.all(np.isfinite(step)):
                    raise ValueError("LSMR returned a nonfinite tangent increment")
                attempt["lsmr"] = {
                    "stop_code": int(solved[1]), "iterations": int(solved[2]),
                    "residual_norm": float(solved[3]), "normal_residual_norm": float(solved[4]),
                    "condition_estimate": float(solved[6]),
                }
            except Exception as exc:
                status, failure = "linear_solve_failed", _error(exc)
                break
            finally:
                report["timing_seconds"]["linear_solve"] += perf_counter() - tick
            if expired():
                status = "deadline"
                break
            for backtrack in range(settings["max_backtracks"] + 1):
                if expired():
                    status = "deadline"
                    break
                alpha = settings["backtrack_factor"] ** backtrack
                trial_step = alpha * step
                trial: dict[str, Any] = {"alpha": alpha, "accepted": False}
                attempt["candidates"].append(trial)
                try:
                    predicted = before - _energy(residual + jacobian @ trial_step)
                    trial["predicted_reduction"] = predicted
                    if predicted <= 0.0:
                        trial["reason"] = "nonpositive_model_reduction"
                        report["rejected_candidates"] += 1
                        continue
                    tick = perf_counter()
                    try:
                        candidate = retract(deepcopy(state), trial_step.copy())
                    finally:
                        report["timing_seconds"]["retract"] += perf_counter() - tick
                    if expired():
                        status = "deadline"
                        break
                    candidate_residual, candidate_energy = evaluated(candidate, residual.size)
                    actual = before - candidate_energy
                    ratio = actual / predicted
                    trial.update({"energy": candidate_energy, "actual_reduction": actual, "gain_ratio": ratio})
                    if actual > 0.0 and ratio >= settings["min_acceptance_ratio"]:
                        state, residual, energy = candidate, candidate_residual, candidate_energy
                        accepted_step_norm = float(np.linalg.norm(trial_step))
                        record.update({"accepted": True, "energy_after": energy, "step_norm": accepted_step_norm, "alpha": alpha, "gain_ratio": ratio})
                        trial["accepted"] = True
                        report["accepted_steps"] += 1
                        report["accepted_energy_history"].append(energy)
                        # Strong agreement lowers damping; poor agreement raises it.
                        damping *= 1.0 / 3.0 if ratio > 0.75 else (2.0 if ratio < 0.25 else 1.0)
                        damping = float(np.clip(damping, settings["min_damping"], settings["max_damping"]))
                        accepted = True
                        break
                    trial["reason"] = "nonlinear_objective_did_not_decrease_sufficiently"
                except Exception as exc:
                    # A bad trial cannot poison the last finite accepted checkpoint.
                    trial["reason"] = "invalid_candidate"
                    trial["error"] = _error(exc)
                report["rejected_candidates"] += 1
            if accepted or status in ("deadline", "linear_solve_failed"):
                break
            damping = min(damping * 10.0, settings["max_damping"])
        if accepted and callback is not None and not expired():
            tick = perf_counter()
            report["callback_count"] += 1
            try:
                callback(deepcopy(state), deepcopy(record))
            except Exception as exc:
                status, failure = "callback_failed", _error(exc)
            finally:
                report["timing_seconds"]["callback"] += perf_counter() - tick
        if status in ("linear_solve_failed", "callback_failed"):
            break
        if expired() or status == "deadline":
            status = "deadline"
            break
        if not accepted:
            status = "no_acceptable_step"
            break
        if accepted_step_norm <= settings["step_tolerance"]:
            status = "step_tolerance"
            report["converged"] = True
            break
        if before - energy <= settings["function_tolerance"] * max(1.0, abs(before)):
            status = "function_tolerance"
            report["converged"] = True
            break
    finish = perf_counter()
    report.update({
        "status": status,
        "completed_global_iterations": report["accepted_steps"],
        "final_energy": energy,
        "energy_reduction": report["initial_energy"] - energy,
        "iterations": len(report["history"]),
        "final_damping": damping,
        "usable_finite_state": True,
        "technical_execution_complete": report["accepted_steps"] > 0 or status == "stationary_initial",
        "motion_quality_accepted": False,
        "elapsed_seconds": finish - start,
        "deadline_overrun_seconds": max(0.0, finish - end),
    })
    if failure is not None:
        report["failure"] = failure
    return state, report
