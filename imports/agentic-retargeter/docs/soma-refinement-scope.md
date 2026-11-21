# Scoped trajectory refinement

An optional `temporal_optimization.refinement_scope` restricts synchronous refinement to explicitly named native scalar hinges and half-open source-frame windows. It requires a hash-bound `prior_global_seed` and the synchronous sparse solver. Example:

```json
{"windows": [{"frames": [13, 34], "joints": ["left_elbow_joint"]}]}
```

All original residual rows, source targets, collision terms, keyframe terms, temporal derivatives and cross-boundary stencils remain in the objective. The linear solve uses only the declared tangent columns. The full sequence is evaluated for line-search acceptance. Unselected native coordinates, free roots and all object poses are copied exactly from the prior state; this is checked byte for byte again by the independent result validator. Frame intervals use the original clock, without resampling.

This is constrained refinement, so a small active gradient does not imply stationarity of the unrestricted full-body problem. The report records the complete and active variable counts, exact active indices, resolved native names, fixed-state evidence and the scope of stationarity. No collision or motion acceptance follows from solver completion. Author and audit native keyframes separately, then audit the actual exported trajectory and held-out fractional times on the original geometry.

The feature is opt-in. An absent scope preserves the previous unrestricted solve. It does not change geometry, phase boundaries, source tracking weights, native joint limits, or speed limits. It can fail to fix a collision when the declared subset lacks enough freedom; extend the scope only as an explicitly reviewed new experiment.
