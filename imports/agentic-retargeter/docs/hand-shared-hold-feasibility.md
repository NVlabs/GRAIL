# Feasible shared-hold updates

The optional `feasible_shared_hold_step` policy prevents Adam from repeatedly
evaluating shared-hold trajectories outside the original source wrist bounds.
It is separate from `feasible_spline_step`; existing configurations retain their
previous behavior when the option is absent or `false`.

```json
{
  "execution_backend": "eager",
  "strict_hold_parameterization": {"warm_policy": "require_constant"},
  "feasible_shared_hold_step": {
    "kind": "backtracking",
    "max_backtracks": 12,
    "shrink_factor": 0.5
  }
}
```

Use per-frame base parameters with at least one declared strict hold. The
initial decoded trajectory must already satisfy every original-frame source
translation/rotation-vector box and native finger limit, including frozen
frames. An infeasible seed is rejected, not silently projected.

After Adam proposes an update, the filter checks the actual full-sequence
decoder, including shared holds, optional boundary blending and existing edit
masks. If necessary it reduces **all base and shared controls by the same step
fraction**. If no tested fraction is feasible, it restores the exact previous
controls. Shared poses remain shared; support/slide variables stay live and
frozen frame zero retains its existing mask. The filter does not clamp decoded
poses, relax bounds, change timestamps or add a new objective. The established
1e-6 feasibility tolerance remains unchanged.

Adam moments and the step counter retain the full proposal history. This is a
feasibility filter, not an objective line search: accepted objective values can
increase, and the existing feasible best-checkpoint selection still applies.
The JSON report records each proposed/accepted status, step fraction, restoration,
backtrack count and decode-only time under `feasible_shared_hold_step`. These
checks synchronize device results in eager execution, so their time is included
in the optimization time. Original-frame feasibility does not certify subframe
clearance, stable contact, full-body reachability or physical validity.
