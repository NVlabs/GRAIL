# Experimental G1 contact-preserving sequence solve

[g1_contact_sequence.py](g1_contact_sequence.py) packages the tested version-3
contact-constrained interpolation prototype. It imports the adjacent
[g1_sequence_objective.py](g1_sequence_objective.py), the unchanged preceding
experiment's coupled objective. Only that import path changed during packaging;
the numerical algorithm, defaults and limits remain unchanged. This is a
schema-specific G1 experiment, not a generic robot API or a default interpolator.
Use the [contact edit kit](../README.md) for native pose and Blender timing edits.

The solver preserves each active pad's original-material gap and opposing-normal
angle relative to the selected parent at matching times. It does not close
inherited gaps or repair a wrong grasp. There is no built-in collision,
auxiliary-digit clearance, fixed source/depth or physical-validity gate. Every
export is **UNREVIEWED** until those checks and the requested hand-motion review
are complete. Lower objective or joint jerk alone does not establish acceptance.

## Inputs and supported scope

CPU dependencies are NumPy, SciPy, MuJoCo and OSQP. The prototype was tested with
OSQP **0.6.7.post3**; newer OSQP APIs require compatibility testing. No Blender or
GPU is required for this solver. Original models, mesh assets, native motion,
material descriptors and independent review inputs must be supplied by the
caller; they are not included in this source package.

[example_config.json](example_config.json) illustrates the schema. It is **not a
ready-made case**: its frame window, protected keys and numerical settings are
examples inherited from one experiment. Copy it into your case directory, bind
your selected original assets, and author the appropriate interval and contacts.
All configured relative paths resolve against the config's directory. Absolute
paths to existing original assets also work without copying or modifying them.

Required input contracts include:

- Native NPZ: finite `qpos[T,nq]`, positive scalar `fps`, `obj_R[T,3,3]` and
  `obj_t[T,3]`. The tested G1 layout has a frozen free root at qpos 0–6, with its
  quaternion at 3–6. Editable controls are the model's limited hinge joints named
  `left_`/`right_` plus the 14 arm/hand suffixes in `JOINT_SUFFIXES` in the objective
  module. Wrist bodies are `left_wrist_yaw_link` and `right_wrist_yaw_link`.
  Equality-constrained models are unsupported. Preserve the original native
  metadata, geometry, coordinate convention, timing and unit conventions.
- Original MuJoCo XML and its referenced assets. An optional `model_snapshot`
  NPZ checks compiled `jnt_type`, `jnt_limited`, `jnt_qposadr`, `jnt_dofadr` and
  `jnt_range` when present. Existing native name/address bindings are checked too.
- Original object mesh: with `object_mesh`, an NPZ containing `vertices` and
  `faces`, plus `object_scale` in the native archive. Vertices are scaled before
  material comparisons. Without a separate mesh, the native archive must contain
  `object_vertices_scaled` and `object_faces`. Embedded and separate geometry must
  agree when both are present.
- Material descriptor JSON: `model_xml` and `phases`. Each phase has `side`,
  `role`, `pairs`, and either `reference_frame` or an inclusive `active_frames`
  (`frames` is also accepted) interval. Every pair includes `finger`, `active`,
  `body_name`, `point_local`, unit `normal_local`, `object_face_id`,
  `object_barycentric`, `object_point_local_m` and unit
  `object_outward_normal_local`. Material points must match their original scaled
  triangles; indices are case-specific. Author these from the actual original
  models and selected grasp, never by copying another object's contact indices.

The selected sides must have stable contact identities and roles throughout the
configured window **and its reviewed neighboring frames**. Choose a compatible
hold/support interval; contact activation, sliding patches or changing roles
need an explicit phase-model extension. The review window is exactly
`[window_start - 1, window_end + 1]`. The motion must also supply two surrounding
frames for the temporal objective. Sorted unique protected keys must include
both interval endpoints and their immediate interior neighbors; at least one
interior source frame must remain editable. Other native coordinates, protected
keys and object transforms remain frozen.

## Numerical behavior and limits

The objective couples object-relative wrist/pad tracking with native joint
regularization. The example uses `wrist_finger_fk`: interpolate finger hinges,
reconstruct their wrist-relative pad transforms through actual FK, then compose
them with the interpolated wrist target. Contact constraints separately compare
fixed original-material gaps and opposing-normal angles with the original parent
at the same times. Native replay uses linear hinge interpolation, normalized
frozen root quaternions, SLERP object rotation and linear object translation;
stored integer-frame object matrices are retained exactly.

Each sparse Gauss–Newton SQP proposal solves an OSQP problem with linearized
contact inequalities, native limits and the parent-relative correction corridor.
Actual nonlinear FK and objective decrease govern acceptance. Version 3 allows
up to four bounded minimum-norm second-order feasibility corrections when a
tangent step crosses a curved constraint boundary. Corrections remain within
the original trust region and native/corridor bounds. Failed attempts are saved;
linearized feasibility alone cannot accept a step.

The example uses **0.45 mm / 1.8° interior** gap/normal allowances against
**0.5 mm / 2° external** preservation criteria, with a **5°** joint correction
corridor. These are experimental settings, not universal contact tolerances.
This unchanged prototype caps the external limits at 0.5 mm / 2°, requires
strictly smaller positive interior allowances, and caps the corridor at 5°.
The interval's original parent remains the reference throughout; restarting from
a candidate must not accumulate new allowances. Settings outside the supported
ranges require an explicit implementation change and new preflight, not a
claim that editing the config alone enables them.

Constraint sampling supports `sample_step_frames` of 0.25 or 0.125 and optional
`extra_constraint_frames` within the review interval. Independently inspect
off-grid times, changed joins, original meshes, source alignment and hand motion.
A time added to the fit is no longer held out for that candidate. Finite samples
do not guarantee continuous contact or collision clearance.

## CLI and review

Run from the repository root with a suitable CPU environment. Prepare your own
case config and choose a fresh output directory for preflight:

```bash
sequence_solver=scripts/standalone/contact_edit_kit/experimental/g1_contact_sequence.py

python "$sequence_solver" --config path/to/case/config.json \
  --out tmp/g1_contact_trial
```

Preflight binds input/config/code hashes and checks parent feasibility, exact
prior-objective/Jacobian parity and directional finite differences of the
objective and constraints. Review that declaration and independently verify the
model/material/interpolation conventions before fitting. Then use the **same
preflight directory** and unchanged config/code with `--run`:

```bash
python "$sequence_solver" --config path/to/case/config.json \
  --out tmp/g1_contact_trial --run
```

This second command performs a fit; preflight alone does not. Preserve output
directories and use a new directory for each new configuration or trial. The fit
retains attempted native vectors, rejection/correction diagnostics, iteration
history, contact measurements and `candidate_UNREVIEWED.npz`. Review exports
independently before selecting or propagating them. An iteration cap bounds a
numerical execution; it does not complete an unresolved correction or prove
infeasibility.

For a saved feasible checkpoint, `--continue-from previous_output` requires exact
config and packaged implementation hashes. Perform preflight into a new output
directory with that argument, then repeat with `--run`. The relocated solver has
a new implementation hash, so archived outputs from the old path are not valid
continuations; the original selected parent remains a valid fresh start.

The historical Frypan, Drill and Kettlebell case configs and their assets are
not shipped; use [`example_config.json`](example_config.json) as the schema
example. Historical renderers, frozen reviewers, case caches, videos and motions are not
dependencies of these two packaged source modules. Supply independent review
appropriate to your current case rather than assuming that old evidence
validates a new export.
