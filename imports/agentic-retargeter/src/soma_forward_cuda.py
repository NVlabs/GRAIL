"""Bounded-memory CUDA normal equations for the pinned Newton LM forward pass.

Only one optimizer instance's linear solve is replaced. Newton still owns
residuals, Jacobians, damping, acceptance and native-coordinate integration.
No upstream files or specialized-class caches are modified.
"""
from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from pathlib import Path
from types import MethodType

import warp as wp


@wp.kernel
def _assemble_normal_equations(
    jac: wp.array3d[wp.float32],
    res: wp.array3d[wp.float32],
    rows: int,
    cols: int,
    hessian: wp.array3d[wp.float32],
    gradient: wp.array2d[wp.float32],
):
    """One thread per entry; deterministic ascending residual accumulation."""
    batch, i, j = wp.tid()
    value = float(0.0)
    if j < cols:
        for k in range(rows):
            value = value + jac[batch, k, i] * jac[batch, k, j]
        hessian[batch, i, j] = value
    else:
        for k in range(rows):
            value = value + jac[batch, k, i] * res[batch, k, 0]
        gradient[batch, i] = value


@lru_cache(maxsize=None)
def _small_cholesky_kernel(n_dofs: int):
    dof = wp.constant(n_dofs)

    def solve(
        hessian: wp.array3d[wp.float32],
        gradient: wp.array2d[wp.float32],
        damping: wp.array[wp.float32],
        delta_out: wp.array2d[wp.float32],
        prediction_out: wp.array[wp.float32],
    ):
        batch = wp.tid()
        h = wp.tile_load(hessian[batch], shape=(dof, dof))
        g = wp.tile_load(gradient[batch], shape=(dof,))
        lam = damping[batch]
        diag = wp.tile_zeros(shape=(dof,), dtype=wp.float32)
        for i in range(dof):
            diag[i] = lam
        a = wp.tile_diag_add(h, diag)
        rhs = wp.tile_map(wp.neg, g)
        lower = wp.tile_cholesky(a)
        delta = wp.tile_cholesky_solve(lower, rhs)
        wp.tile_store(delta_out[batch], delta)
        lambda_delta = wp.tile_zeros(shape=(dof,), dtype=wp.float32)
        for i in range(dof):
            lambda_delta[i] = lam * delta[i]
        diff = wp.tile_map(wp.sub, lambda_delta, g)
        prod = wp.tile_map(wp.mul, delta, diff)
        prediction_out[batch] = 0.5 * wp.tile_sum(prod)[0]

    solve.__name__ = f"soma_forward_normal_cholesky_{n_dofs}"
    solve.__qualname__ = solve.__name__
    return wp.kernel(enable_backward=False, module="unique")(solve)


class _NormalEquationSolver:
    """GPU/CPU testable kernel pair; buffers are allocated before capture."""

    def __init__(self, n_batch: int, n_dofs: int, n_residuals: int, device):
        self.n_batch = int(n_batch)
        self.n_dofs = int(n_dofs)
        self.n_residuals = int(n_residuals)
        self.device = wp.get_device(device)
        if min(self.n_batch, self.n_dofs, self.n_residuals) <= 0:
            raise ValueError("Positive batch, DOF and residual counts are required")
        # This adapter is deliberately bounded to our G1-size systems. A larger
        # Cholesky tile would need an independent shared-memory qualification.
        if self.n_dofs > 64:
            raise ValueError("Forward CUDA normal solver supports at most 64 DOFs")
        self.hessian = wp.empty((self.n_batch, self.n_dofs, self.n_dofs),
                                dtype=wp.float32, device=self.device)
        self.gradient = wp.empty((self.n_batch, self.n_dofs),
                                 dtype=wp.float32, device=self.device)
        self.kernel = _small_cholesky_kernel(self.n_dofs)

    def solve(self, jac, res, lam, dq, pred):
        wp.launch(_assemble_normal_equations,
            dim=(self.n_batch, self.n_dofs, self.n_dofs + 1),
            inputs=[jac, res, self.n_residuals, self.n_dofs],
            outputs=[self.hessian, self.gradient], device=self.device)
        wp.launch_tiled(self.kernel, dim=(self.n_batch,),
            inputs=[self.hessian, self.gradient, lam], outputs=[dq, pred],
            block_dim=32, device=self.device)


def _validate_pinned_impl(impl):
    from newton._src.sim.ik.ik_lm_optimizer import IKOptimizerLM

    if not isinstance(impl, IKOptimizerLM):
        raise TypeError("Expected a pinned Newton IKOptimizerLM instance")
    if not impl.device.is_cuda:
        raise ValueError("Forward CUDA override requires a CUDA optimizer")
    if getattr(type(impl), "_step", None) is not IKOptimizerLM._step:
        raise ValueError("Unknown Newton LM step override; refusing replacement")
    if getattr(type(impl), "step", None) is not IKOptimizerLM.step:
        raise ValueError("Unknown Newton LM outer step; refusing replacement")
    names = tuple(inspect.signature(impl._solve_tiled).parameters)
    if names != ("jac", "res", "lam", "dq", "pred"):
        raise ValueError(f"Unsupported pinned Newton linear-solve signature: {names}")
    # Detect integration/API changes explicitly without tying a shared runtime
    # directory to one local filesystem path.
    step_source = inspect.getsource(IKOptimizerLM._step)
    expected = ("ctx_curr.jacobian_out, self.residuals_3d, self.lambda_values, self.dq_dof, self.pred_reduction",
                "_accept_reject", "_update_lm_state", "step_size=step_size")
    if not all(value in step_source for value in expected):
        raise ValueError("Unsupported Newton LM step contract")
    expected_shapes = {
        "jacobian": (impl.n_batch, impl.n_residuals, impl.n_dofs),
        "residuals_3d": (impl.n_batch, impl.n_residuals, 1),
        "dq_dof": (impl.n_batch, impl.n_dofs),
        "lambda_values": (impl.n_batch,),
        "pred_reduction": (impl.n_batch,),
    }
    for name, shape in expected_shapes.items():
        value = getattr(impl, name)
        if tuple(value.shape) != shape or value.dtype != wp.float32 or value.device != impl.device:
            raise ValueError(f"Unsupported Newton LM buffer: {name}")
    return IKOptimizerLM


def install_forward_cuda_solver(impl) -> dict:
    """Install on one CUDA LM instance before its first BasePass graph capture.

    Return JSON-compatible provenance. Calling this twice is idempotent. The
    original bound method and the helper buffers remain on the instance for
    inspection; no fallback silently changes objectives or solver type.
    """
    installed = getattr(impl, "_soma_forward_cuda", None)
    if installed is not None:
        return dict(installed.provenance)
    upstream = _validate_pinned_impl(impl)
    helper = _NormalEquationSolver(impl.n_batch, impl.n_dofs, impl.n_residuals, impl.device)
    original = impl._solve_tiled

    def replacement(self, jac, res, lam, dq, pred):
        helper.solve(jac, res, lam, dq, pred)

    upstream_file = Path(inspect.getfile(upstream))
    helper.provenance = dict(
        schema="soma_forward_cuda_normal_equations.v1",
        backend="cuda_normal_equations_then_small_cholesky",
        device=str(impl.device), n_batch=impl.n_batch, n_dofs=impl.n_dofs,
        n_residuals=impl.n_residuals, warp_version=wp.__version__,
        upstream_file=str(upstream_file),
        upstream_sha256=hashlib.sha256(upstream_file.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        accumulation="float32 ascending residual loop, one writer per matrix entry, no atomics",
        equation="(J^T J + lambda I) delta = -J^T r",
        predicted_reduction="0.5 delta^T (lambda delta - J^T r)",
        extra_global_buffer_bytes=4 * impl.n_batch * (impl.n_dofs**2 + impl.n_dofs),
        upstream_residuals_jacobian_damping_acceptance_integration_unchanged=True,
        numerical_equivalence="Same equations; changed float32 summation order requires measured validation",
    )
    impl._soma_forward_cuda_original_solve = original
    impl._soma_forward_cuda = helper
    impl._solve_tiled = MethodType(replacement, impl)
    return dict(helper.provenance)
