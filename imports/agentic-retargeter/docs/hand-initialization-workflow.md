# Hand-sequence initialization and whole-body SOMA

This baseline first optimizes free wrists and native Dex3 fingers across the
source sequence, then supplies those poses to a fresh whole-body SOMA conversion.
SOMA executes sequential forward IK followed by synchronous trajectory
optimization. The hand candidate is a target, not an already mounted robot
trajectory. Every stage produces an unaccepted candidate and preserves the
previous stage for comparison.

The packaged sequence is:

1. Bind the original human, robot, object, scene, source clock and phase evidence.
2. Run multistart Topo-inspired hand initialization with the shared policy.
3. Solve a 300-step base paired hand sequence, filter its initial guess with
   phase-aware SG9/poly3, and run 300 contact-aware temporal refinement steps.
   Observed grasp intervals share poses, other frames retain live variables,
   and cross-hand collision guidance remains. Select the final refined
   checkpoint explicitly, retaining intermediate evidence.
4. Package wrist/finger targets and separately selected material anchors.
5. Run fresh CUDA forward whole-body IK, then synchronous CUDA residual/Jacobian
   evaluation with CPU sparse LM/LSMR solving.
6. Export the solver-bound native robot and object trajectories and review the
   complete result, including original meshes, temporal joins and source posture.

The Topo objective is an implementation inspired by interaction-graph ideas;
it is not a reproduction of the complete paper or a learned grasp generator.
The [packaged multistart bank](topo-hand-initialization.md) now exposes the
structure key search without importing experiment directories. A standalone pad
pass is now opt-in. The [shared policy](hand-shared-policy.md) fixes numerical
settings across cases and takes source-video phase annotations as input. The
modules below run the all-frame route and the selected-hand SOMA handoff.
Historical experiment bundles remain immutable; these tools do not claim
bitwise replay of those experiments. See the
[current workflow](current-retargeting-workflow.md) for scene preparation,
first-frame protection, refinement and independent one-GPU jobs.

## Inputs and execution

Install the package with its GPU extras in a compatible PyTorch/Warp environment:

```bash
python -m pip install -e '.[gpu-repair,test]'
```

The external SOMA/Newton runtime, original robot assets and reconstructed data
remain separate dependencies. Use the runtime version bound by the selected
experiment; installation of this package does not install or modify SOMA.

A reviewed hand configuration must contain `inputs` for `GraspInputs`: native
reference NPZ, original MJCF, original object mesh NPZ, and explicit per-hand
wrist names, finger-joint order, mesh IDs, contact groups and eligible original
object faces. It also binds `source_arrays`, containing same-time human joint
positions/rotations and the identical object trajectory/scale. These identifiers
are case-specific; copying another model's mesh IDs is not a valid configuration.

The native reference supplies body/arm context and initial fingers. During the
free-hand solve, the source body, arm, root and object arrays stay unchanged;
the optimized quantities are the free wrist pose and seven native finger angles
per enabled hand, at every original frame. For a from-scratch experiment, use a
fresh raw conversion as this reference, not an earlier refined robot result.

Run the first hand stage with a reviewed `hand.json`:

```bash
python -m agentic_retargeter.hand_sequence hand.json --output runs/hand_v001
```

Relevant controls include `device: "cuda:0"`, `mode` (`independent`, `temporal`
or `anchor_temporal`), `topo_steps`, `pad_steps`, `temporal_steps`,
`source_prior`, `temporal_weights`, explicit contact intervals and phase labels.
Low-level all-frame defaults are 200 Topo, **zero pad**, and 200 temporal steps;
the default mode is `temporal`, while explicit `independent` skips that stage.
The routine shared-policy route uses a multistart key bank followed by 300 base
paired steps, SG filtering and 300 paired refinement steps. Its final refinement
uses contact weight 256, four samples per digit at 3 mm spacing and 100 times
the base temporal weights; all cases use the same profile. These are optimizer
steps, not agent review rounds. Supplying
`warm_start_npz` skips Topo/pad optimization and continues from that candidate,
while the original source remains the reference.

Outputs include `source.npz`, snapshots for stages actually executed,
`hand_sequence.npz` and `report.json`. `pad.npz` exists only when pad refinement
was requested and executed. The report records executed/skipped steps,
source/geometry bindings, implementation hashes, timings and sampled metrics.
Use new output directories: these research entrypoints do not themselves reject
every pre-existing output filename.

For simultaneous hand refinement, prepare `paired.json` with the same original
`inputs`, the selected `warm_start_npz`, original `fps`, per-hand Boolean
`contact_mask` and phase labels. Its GPU settings can include:

```json
{
  "device": "cuda:0",
  "execution_backend": "cuda_graph",
  "rotation_backend": "rodrigues",
  "paired_steps": 200,
  "temporal_coordinate_space": "phase"
}
```

```bash
python -m agentic_retargeter.hand_sequence_pair paired.json --output runs/paired_v001
```

This is a configuration fragment, not a replacement for the original bindings
and phase data. Both active hands share one autograd graph. An unoptimized
opposite hand remains bound to its original native state for the configured
collision guidance. The solver retains sampled object, self and cross-hand
terms, source priors, native limits and source-relative wrist bounds.
By default, `phase` temporal coordinates use object coordinates only for
acceleration/jerk stencils wholly inside a strict hold; other stencils use world coordinates.
The shared recipe also uses object-relative derivatives in declared support
and slide intervals, without making them rigid or imposing zero slip.
The separate `world` and `object` settings use uniform coordinates.

CUDA Graph currently supports per-frame parameters. Spline or shared-hold
parameterizations require `execution_backend: "eager"`; they do not silently
fall back. The paired solver saves initial, periodic, last and best-objective
checkpoints. Best objective means neither best geometry nor accepted motion.
Review the full sequence before naming a selected checkpoint.

`refinement_editable_frames` can preserve selected original frames exactly while
using per-frame or shared-hold parameters. A shared hold must be wholly editable
or wholly frozen; a mask that cuts through it is rejected. Explicit
`strict_hold_parameterization.intervals` preserve separate grasp identities even
when intervals are adjacent. The shared policy explicitly excludes protected
frame zero from sharing without changing its source role. Use an explicitly
constructed warm sequence whose frame zero is the intended original pose before
freezing that frame. Spline parameters remain incompatible with this frame mask.

For a multistart bank, use the hash-bound candidate-to-phase bridge described in
[`hand-candidate-selection.md`](hand-candidate-selection.md). Running the ordinary
all-frame initializer alone does not sample multiple starts. Independently audit
relative motion with `hand_contact_stability_audit.audit`; its kinematic metrics
do not replace original-surface contact and collision checks.

## Phases and material anchors

Keep original acquisition, hold, support, slide and release timing. A fixed
material correspondence may span an explicitly declared strict hold; it must
not silently span a slide or changing support relation. A moving object's
stationary prefix is not a static-object case.

The implemented graph uses an explicit human/native node mapping. This mapping
is distinct from the contact-region policy: the human thumb region guides the
Dex3 thumb, while the remaining human fingers supply the long-finger region.
Check the two native long fingers separately so one digit cannot stand in for
both. Region evidence, finite surface samples and fixed anchors are different
representations and retain separate provenance.

`hand_sequence_anchors.py` selects eligible original-surface point pairs, taking
the nearest first and then spreading points on both surfaces. The baseline
defaults are a 5 mm maximum gap and 3 mm spacing. Keep explicit coverage
shortfalls; never fill missing anchors by raising the gap threshold silently.
Each anchor stores a native body-local point plus an original object face ID
and barycentric coordinates. Geometry queries do not establish force closure.

Bind each anchor to its intended anatomical surface as well as its digit.
A proximal support point does not establish distal-pad gripping contact. Keep
the exact source geom/body IDs and material coordinates when selecting a subset;
do not relabel a proximal point as a distal point. Report the anchors actually
optimized separately from additional anchors evaluated only during auditing.
Inspect acquisition/release and support/slide joins alongside within-hold
smoothness: a constant hold with a discontinuous entry is not a stable sequence.

The previous SOMA comparison tested both pose-only targets and pose targets with
sparse anchors. For the latter, it kept at most one eligible anchor per digit
per contiguous strict hold, requiring both the recorded and selected-sequence
reference-frame gaps to be at most 5 mm. This is an experiment selection policy,
not an automatic global acceptance rule. Re-evaluate retained anchors after
changing the selected hand sequence. No eligible anchor means an explicit
pose-only case with a recorded shortfall, not a zero measured contact error.

For an exactly constant full-source object trajectory, disable free-hand contact
terms via the static case policy and omit SOMA material-position/normal losses.
Both hand-sequence solvers derive this policy from the full source trajectory;
an optional explicit `is_static_obj` flag must agree with it.
The SOMA converter independently checks exact original translation and rotation
and preserves that object trajectory. Object/scene/self collision and source
tracking remain; static objects do not require authored material anchors.

## Hand target handoff

The public exporter is `tools/retarget/hand_targets.py`; the packaged consumer is
`soma_robot_hand_targets.py`. Neither selects a checkpoint or infers phases.
First author `selection.json` for the exact candidate being sent downstream:

```json
{
  "schema": "hand_sequence_selection.v1",
  "config_sha256": "<SHA256 of the executed hand configuration>",
  "sequence_sha256": "<SHA256 of the explicitly chosen sequence NPZ>",
  "report_sha256": "<SHA256 of its solver report>",
  "reason": "Candidate chosen for whole-body initialization; remaining defects recorded separately.",
  "motion_accepted": false
}
```

```bash
python tools/retarget/hand_targets.py \
  --config paired.json --sequence runs/paired_v001/hand_sequence.npz \
  --report runs/paired_v001/report.json --selection selection.json \
  --output runs/hand_targets_v001
```

The installed equivalent is `python -m agentic_retargeter.hand_sequence_handoff`
with the same arguments. The exporter requires the executed report's exact
configuration, geometry binding and selected-checkpoint hash. It checks original
source/object/time arrays, native finger order and limits, phase/mask agreement,
proper rotations and world/object consistency, then loads the bundle through the
actual SOMA consumer. An existing output directory is refused; invalid targets
are not published. No GPU or SOMA execution is required for this packaging step.

Outputs are `hand_targets.npz`, `handoff.json`,
`robot_hand_targets_config.json`, `anchor_evidence.json`, copied selection/config/
solver-report evidence, and `export_receipt.json` with hashes. Copy the generated
configuration object into SOMA's `robot_hand_targets` field. Its paths are
absolute; staging to another machine requires an explicit path update while
retaining the bound handoff/target hashes.

The exact full-source object rotation and translation determine whether this is
a static case. If the executed configuration provides `is_static_obj`, it must
agree; otherwise export fails. An absent flag is derived from the original
motion. A static bundle retains the authored phase labels as
evidence but supplies an empty target contact mask and `no_contact` consumer
labels. Collision and source objectives are not disabled by this conversion.
The exported bundle remains `UNREVIEWED_SOFT_TARGETS` and records that whole-body
SOMA has not run. `anchor_evidence.json` preserves available solver anchors and
rejections only; it never installs them as SOMA contacts or certifies their gaps.

`hand_targets.npz` contains exactly:

| Array | Shape/content |
| --- | --- |
| `timestamps` | Original `T` timestamps |
| `object_R`, `object_trans`, `object_scale` | Original `T×3×3`, `T×3`, and metric scale `(3,)` |
| `source_qpos_reference_only` | Original `T×nq` reference, not a solved full-body result |
| `<side>_wrist_position_object`, `<side>_wrist_rotation_object` | Selected `T×3`, `T×3×3` object-local wrist poses |
| `<side>_wrist_position_world`, `<side>_wrist_rotation_world` | Corresponding world poses |
| `<side>_finger_qpos`, `<side>_finger_joint_names` | Selected `T×7` angles and exact seven-joint order |

For each enabled side, compute world targets as
`p_world = object_R @ p_object + object_trans` and
`R_world = object_R @ R_object`. Object-local positions are already metric and
include original scale; do not multiply by `object_scale` again.

The JSON metadata has schema `unaccepted_hand_sequence_soma_handoff.v1`, original
`frames`/`fps`, `hand_only: true`, `motion_accepted: false`, the original
`GraspInputs.binding`, and explicit `hands`. Each hand declares `wrist_body`,
`finger_joints`, per-frame `phase_labels` and a Boolean `contact_mask`. Supported
contact labels are `contact_acquisition`, `hold`, `support` and `slide`; other
supported labels are `no_contact`, `unknown`, `release` and `approach`. The mask
must exactly agree with these labels. Preserve original anchor/rejection records
as additional metadata; the pose adapter does not consume anchors from it.

The exporter verifies that the selected report binds the checkpoint hash and
the original `GraspInputs` binding. It checks byte equality against the canonical
source qpos, object, scale and timestamp references used by the hand solver.
`GraspInputs` converts object rotation/translation arrays to float64; this is not
a claim that exported arrays retain a raw float32 native dtype. The receipt names
canonical-reference equality and records native/canonical dtypes. Original native,
model and object files remain independently hash-bound and unchanged. The exporter
also records the selection and source/config/code hashes.
A scalar object scale is broadcast to the consumer's three-axis representation
without rescaling wrist translations. A single
rigid `world_to_solver` transform must align every source frame; do not use
per-frame registration or retiming to hide a mismatch.

The `robot_hand_targets` section of the SOMA configuration is:

```json
{
  "handoff_json": "handoff.json",
  "handoff_sha256": "<actual SHA256 of handoff.json>",
  "targets_npz": "hand_targets.npz",
  "targets_sha256": "<actual SHA256 of hand_targets.npz>",
  "enabled_hands": ["left", "right"],
  "translation_weight": 1.0,
  "rotation_weight": 1.0,
  "finger_weight": 1.0,
  "translation_sigma_m": 0.02,
  "rotation_sigma_rad": 0.2,
  "finger_sigma_rad": 0.2,
  "source_hand_weight_multiplier": 0.0
}
```

Set `world_to_solver` explicitly for a nonidentity frame conversion. Targets are
object-relative to the live optimized object on the contact mask, and fixed in
solver-world coordinates otherwise. The shown multiplier replaces the enabled
hands' original human wrist/finger tracking contribution with the new pose
targets; original body targets remain. Other multipliers are explicit trials.
Pose targets are soft objectives, not hard no-slip or reachability constraints.

Supply `contact_constraints` separately, retaining the bound original object and
configured `collision_avoidance`, including the bound support scene (original,
or an explicitly authorized replacement). With
robot hand targets enabled, a moving-object pose-only comparison may use
`contacts: []`. To add sparse material targets, author the validated retained
anchors using the [contact-spec format](../tools/soma_retargeter/README.md#optional-hand-object-constraints-inside-soma-ik).

## Fresh whole-body conversion and export

Use the original BVH, calibrated original model, original object/scene inputs
and selected hand target contract in `soma.json`. Do not substitute the hand
checkpoint's unchanged reference qpos for the new whole-body result.

Configure `temporal_optimization.method: "synchronous_sparse"` and an explicit
iteration/time budget. Omitted device settings select CUDA forward IK and
CUDA Graph global residual/Jacobian evaluation; the global sparse solve remains
on CPU. An explicit `device: "cpu"` still overrides the forward default.
See [synchronous configuration](soma-synchronous-sequence.md) for full settings,
object-reference costs, stationary-prefix policy and execution telemetry.

```bash
python tools/soma_retargeter/run.py plan --config soma.json --output runs/soma_v001
python tools/soma_retargeter/run.py run --config soma.json --output runs/soma_v001
python tools/retarget/native.py --report runs/soma_v001/run.json \
  --object-sidecar original_object_trajectory.npz --output runs/native_v001.npz
```

`plan` is read-only with respect to the output directory. The adapter and native
exporter reject reused output paths and mismatched input hashes. Keep the
original object sidecar for export: the exporter validates it and inserts the
exact solver-bound optimized object trajectory where enabled, preserving the
original reference and static scene.

Preserve `run.json`, `conversion.log`, `contact_initialization.json`,
`forward_stage_states.npz`, `temporal_sequence_states.npz`, the CSV, optional
`optimized_object.npz`, native NPZ and export receipt. Report actual accepted
global iterations and stopping reason. Repeated checkpoint callbacks are not
additional iterations, and reaching the configured iteration cap is not
convergence.
