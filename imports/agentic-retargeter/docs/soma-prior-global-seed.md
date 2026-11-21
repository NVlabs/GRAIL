# Prior-global checkpoint initialization

`temporal_optimization.prior_global_seed` is an optional initialization for an
iterative synchronous refinement. It keeps the fresh forward pass and its
validation, then starts the global solve from a specific previous global solver
checkpoint. It does not change the source targets, objective weights, object
reference, contact phases, geometry, clock, or optimization variables. It is
disabled when absent.

The purpose is to compare continuing from an actual previous solution against
restarting from the forward pass. A warm initial state can retain a useful local
solution, but it is not a trust region or a prohibition on changes elsewhere in
the sequence. Final source fidelity, speed, contacts, and original-geometry
collisions still require independent review.

## Inputs

The normal SOMA config requires `method: synchronous_sparse`, explicit
`robot_hand_targets`, and `object_optimization`. Add:

```json
"prior_global_seed": {"path": "seed.json", "sha256": "<descriptor SHA256>"}
```

The descriptor schema is `soma_prior_global_seed.v1`. It contains exact
`model_xml_sha256`, `source_bvh_sha256`, `frame_count`, `fps`,
`object_reference_sha256`, and
`coordinate_convention: newton_free_xyz_xyzw_then_scalar_hinges.v1`. Its `files`
mapping must bind each of these through `{path, sha256}`:

- `run`: the previous successfully validated `run.json`;
- `checkpoint`: its `temporal_sequence_states.npz`;
- `linearization`: its `synchronous_linearization.npz`;
- `synchronous_report`: its `synchronous_optimization.json`;
- `object_reference`: its unchanged original object trajectory sidecar.

File paths are relative to the descriptor, or absolute. `provenance` is optional.
The top-level descriptor path is relative to the SOMA config. Use
`soma_prior_global_seed.reference_hash(R, t, scale)` on the original object
reference arrays, not an arbitrary file hash. The loader cross-checks all file
hashes against the previous validated run and its report. The original object
configuration and reference must match the new solve exactly.

The state comes from `refined_joint_q`, `refined_obj_R`, and `refined_obj_t` in the
previous solver checkpoint. A MuJoCo/native-export `qpos` array is not accepted:
its root quaternion convention and joint ordering can differ. This initial
implementation supports one Newton free root followed by scalar hinges only.
It checks coordinate and tangent dimensions, ordered scalar names and indices,
root quaternion validity, frame count/FPS, and actual original human-target
array identity against the new runtime. Static object trajectories must remain
exactly equal to the source, and precontact object poses must share one exact
pose under the original variable layout. Nothing is silently projected or
resampled to pass those checks.

## Execution evidence

The fresh forward pass still runs and is included in elapsed compute cost, even
though the enabled global solve uses the prior checkpoint instead. The earlier
solve that produced that checkpoint also remains part of cumulative compute;
a shorter refinement is not a from-scratch or GPU speedup claim.

Its existing `forward_stage_states.npz`, hand
measurements, hinge-key measurements, and wrist-key measurements retain their
original meaning. An enabled global solve additionally writes
`prior_global_seed_states.npz`, containing `fresh_q`, `fresh_obj_R`,
`fresh_obj_t`, `seed_q`, `seed_obj_R`, `seed_obj_t`, and the original timestamps.

The synchronous report declares `global_initial_state: prior_global_checkpoint`
and binds this evidence and every source file. Its normal initial linearization
and `initial_*` checkpoint arrays now describe the explicitly seeded global
state. Validation verifies them byte-for-byte against the prior checkpoint,
and separately verifies the retained fresh arrays against the actual forward
checkpoint. Final states still use the existing global and native-export checks.

When the option is absent, no seed file is read or written and the original
forward-to-global validation path remains active. The existing solver contract
still requires at least one accepted global update; an already stationary seed
that causes zero updates fails that technical check rather than receiving an
implicit quality approval.

## Review scope

Use a matched fresh-versus-seeded experiment with identical added keyframes and
all other objective settings. Review the complete native sequence as well as the
edited window and held-out fractional times. A successful solver call, lower
energy, or smaller authored-key error does not establish improved collision
quality, and remaining or new collisions must be recorded before selection.
