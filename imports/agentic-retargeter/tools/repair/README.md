# Shared motion audit and local repair

Use an explicit audit scope to diagnose a native trajectory, then repair a short
window with the same contacts and geometry. Implementation lives in
`src/motion_audit.py`, `src/local_repair.py` and `src/repair_geometry.py`.
The optional material backend uses `src/repair_material.py` and
`src/repair_tensor.py`; it does not require an HOI repair checkout.

## Batched scene and self clearance

`src/clearance_objective.py` exposes `BatchedClearance` for native optimization
workers. Supply original `TriangleMesh` instances, explicit mesh-name pairs,
device and sample count. `loss(transforms)` accepts each mesh's batched rigid
rotation/translation and returns a top-k squared clearance loss per pose.
Autograd includes both moving transforms; world-static geometry uses identity
rotation and fixed translation. Query meshes persist across optimization steps.

This is a proposal primitive, not an automatic local-repair backend or Blender
acceptance check. The caller still owns phase/variable masks, source/contact
objectives, smoothness, bounds and native export. Verify original triangles
afterwards, including shoulders and any moving object against the scene.
`allow_degenerate_subset=[name]` is an explicit exception for incomplete static
geometry: original valid-face IDs and omissions are recorded; no mesh is repaired
or substituted. Without this exception, degenerate queries fail. Negative
sampled results never certify clearance, particularly on open meshes.

The [Blender add-on](../../extensions/blender/README.md#local-window-repair)
also exposes this implementation under **GRAIL G1 → Local Window Repair**.
It snapshots current curves, runs the bundled solver in an external Python,
and provides reversible preview, actual-subframe re-audit and explicit Apply.

For screening that needs only a collision flag, the Python API
`repair_geometry.mesh_intersection_probe` stops after a confirmed original-triangle
intersection. A negative result still exhausts candidate pairs; degenerate
geometry leaves a negative decision incomplete. The probe returns no total
face-pair count or canonical repair witnesses. Continue using
`mesh_intersections` for detailed audit counts and the local solver's witnesses.
`pair_enumeration_complete` distinguishes a positive early return from an
exhaustive search; neither operation tests solid containment or continuous time.

```bash
python tools/repair/run.py coverage --config repair.json --output runs/coverage_v001
python tools/repair/run.py audit --config repair.json --output runs/audit_v001
python tools/repair/run.py repair --config repair.json --output runs/repair_v001
# Installed equivalent:
agentic-retargeter local-repair repair --config repair.json --output runs/repair_v002
```

Each output directory must be new. Repair writes the exact submitted config,
`report.json` and `candidate_UNREVIEWED.npz`. A failing candidate carries
`REJECTED_BY_AUDIT` inside the NPZ and report; the filename never implies
acceptance. CLI exit status is 0 for the checked numerical scope, 1 for an audit
failure. When only unchanged failures outside the editable support remain,
the solver stops with `ONLY_FAILURES_OUTSIDE_EDITABLE_SUPPORT` and reports
`affected_scope_passed: true`; the full audit and candidate retain their failing
status. This avoids repeated edits while keeping inherited problems visible.
Invalid inputs raise an error. No operation selects a motion or edits a
Blender scene automatically.

## Check the mesh-pair scope before fitting

The `coverage` command compiles only the native model and enumerates its topology.
It reads the model's referenced assets, but does not load motion or external mesh
NPZ files, construct triangle trees, run a solver, or start a GPU runtime. Its
`ENUMERATION_ONLY` result and exit code 0 mean the inventory succeeded; geometric
status remains `NOT_RUN`.

The report lists missing nonadjacent pairs among the loaded meshes, omitted
model mesh geoms, absent environment geometry, and the editable joints that can
change each pair's relative pose. Same-body/direct-parent exclusions are listed
as a comparison policy; they do not remove configured audit pairs or establish
clearance. A shared world parent does not make scene geometry an articulated
neighbor. Empty relative edit ancestry means the candidate preserves the parent's
relative geometry at matching times, not that the source is constant or clear.

Optional `coverage.omissions` entries use `a`, `b` and a nonempty `reason` for an
unconfigured pair. A reason remains a declaration, and the pair remains missing.
Review the proposed scope and run a baseline audit after any explicit expansion.
Neither pair expansion nor motion selection happens automatically. The normal
audit/repair reports include the same inventory as `geometry_scope`, with all
original numerical gates unchanged. Missing-pair inventories are computed once
per context and require no extra sampled-frame audit.

## Configuration

Paths resolve relative to the submitted config. Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `model_xml` | Original MuJoCo robot XML, with its referenced assets |
| `trajectory_npz` | Native parent motion with named joint binding, `qpos`, `fps`, object transforms and scale |
| `object_mesh_npz` | Original `vertices` and triangle `faces` |
| `window` | Inclusive, zero-based source-frame endpoints; only strict interior frames may change |
| `editable_joints` | Named limited hinges, each with `delta_bounds_rad: [lower, upper]` relative to the parent |
| `contacts` | List of fixed material contact descriptors; may be empty for a collision-only scope |

`protected_frames` additionally freezes interior poses. Each correction corridor
must include zero and is intersected with the original native joint limits. Root,
other joints, object trajectory/scale, timing, and all other frames remain exact.
Display placement must already be baked to native data. Input hashes, model
dependencies and compiled-model binding are checked; changing a bound file
invalidates the audit context. Stale derived FK fields are invalidated on export.

A material contact contains:

```json
{
  "name": "left_thumb",
  "body_name": "left_hand_thumb_2_link",
  "point_local": [0.01, -0.03, 0.0],
  "normal_local": [1.0, 0.0, 0.0],
  "object_face_id": 141,
  "object_barycentric": [0.85, 0.15, 0.0],
  "object_normal_sign": 1,
  "active_frames": [25, 120],
  "max_gap_m": 0.006,
  "max_normal_deg": 180,
  "normal_weight": 0.0,
  "target_gap_m": 0.002
}
```

These are schema examples, not portable pan contacts. Bind pad points/normals to
the actual hand model and original object faces. Normals are transformed with the
object and compared with the opposing pad normal for diagnostics. Per-contact
`normal_weight` defaults to zero and `max_normal_deg` defaults to 180 (no angular
rejection). The current workflow also sets `solver.normal_weight: 0` and omits
`contact_normal_policy`. Existing explicit weights/gates still work; revise them
in a new config instead of assuming new defaults override a saved experiment.
Keep collision/clearance normals and object-relative palm orientation constraints.
Targets remain fixed during a trial. Nonuniform object scale is applied before computing facet normals.

### Brief contact excursions

For visually selected motion where occasional gaps are acceptable, the current
review profile is:

```json
{
  "contact_gap_policy": {
    "nominal_gap_m": 0.015,
    "hard_gap_m": 0.040,
    "max_outlier_fraction": 0.05,
    "max_outlier_duration_s": 0.20
  }
}
```

`contact_gap_policy` replaces the corresponding **acceptance** gate. The legacy
`contact_normal_policy` remains available only for explicitly requested angular
review (fields `nominal_normal_deg`, `hard_normal_deg`, `max_outlier_fraction`,
`max_outlier_duration_s`); it is omitted in the current workflow. An omitted
policy retains the configured per-sample gate, including an explicit legacy
`max_normal_deg`, so set that gate to 180 when disabling angular rejection.
Review policies do not change solver targets, weights or fitting tolerances.
This is an explicit user-directed review-policy revision, not an execution-only
speed comparison or a claim of physically optimal tolerances.

Each named contact is judged independently over its declared active interval
within the audit scope. Values above nominal may occupy at most 5% of that
sampled duration, with no continuous run longer than 0.2 seconds, and must remain
below the hard gap cap. Normal-angle excursions do not trigger repair in the
current workflow. The unrestricted 180-degree range does not establish contact
quality; use the declared gap, slip, source and original-geometry checks. Existing
collision, source, temporal, joint-limit and frozen-state checks retain their
own scope. A selected visual result can retain unresolved metric diagnostics.

Duration uses midpoint cells between chronological samples, clipped to the first
and last active sample. This makes seconds explicit across FPS/subframe densities;
it does not certify unsampled continuous time. A single static sample must meet
the nominal limit. The complete configured grid is required, including subframes;
merge audit shards before classifying the full interval. A short local-window
pass applies only to that window. Missing samples cannot dilute the outlier rate.

Reports retain all measured values, strict per-sample exceedances, maximums,
outlier fraction, consecutive intervals and any tolerated exceptions in
`contact_gap_review` / `contact_normal_review`. Cached measurements remain strict;
aggregate classification is recomputed against the current bound policy. Once
the declared contact gate passes, do not keep optimizing merely to reduce a peak.

For existing bound measurements, avoid rerunning geometry just to change a
threshold:

```bash
python tools/repair/contact_tolerance.py --input contact_observations.json --output new_contact_review.json
```

The input JSON contains `fps`, the complete `required_frames` list, `contacts`
(`name`, inclusive `active_frames`), `rows` (`source_frame`, `contacts` with `name`,
`gap_m`, `normal_deg`), and `policies` with `gap` and/or `normal` set to the policy
fields described above. Use only `gap` under the current policy. Include
motion/target/config hashes and measurement paths under
`provenance`; the output binds the observation file and evaluator code hashes.
The tool checks measurement coverage but cannot independently verify that supplied
measurements came from the claimed native motion. Its pass concerns only the
configured contact metrics and does not overwrite prior motion decisions.

Optional geometry, sampling and solver settings:

```json
{
  "robot_geoms": [65, 72],
  "collision_pairs": [
    {"a": "geom:65", "b": "object", "reference_frame": 48, "clearance_m": 0.0003},
    {"a": "geom:65", "b": "geom:72", "reference_frame": 48, "clearance_m": 0.0003}
  ],
  "audit": {"substeps": 4, "halo": 2, "witness_limit": 16},
  "solver": {
    "contact_mode": "within_gates",
    "fit_substeps": 2,
    "max_rounds": 4,
    "max_evaluations": 80,
    "contact_weight": 1000,
    "normal_weight": 0,
    "pose_weight": 1,
    "acceleration_weight": 1,
    "clearance_weight": 3000,
    "source_weight": 0.1
  }
}
```

Geom IDs must refer to the loaded model's original mesh geoms. List all relevant
finger, palm, wrist, arm, body and environment pairs; adjacency is not silently
excluded. An optional `environment` list supplies named static original meshes
using `name`, `mesh_npz` and `matrix_world`; names can appear in collision pairs.
The reference pose establishes a side for a local collision correction. Its pair
must be complete and free of surface intersections. `clearance_m` is a local
repair offset, not an audited minimum separation distance. Contact at any triangle
surface, including touching and coplanar overlap, counts as an intersection in
this conservative audit. Degenerate geometry leaves the check incomplete.

`contact_mode: "target"` fits the fixed anchor plus its normal times
`target_gap_m` (default zero). `"within_gates"` avoids forcing an already compliant
contact to a new preferred gap. Both use the same independent acceptance gates;
switching modes is an explicit objective revision. Per-contact
`position_weight` defaults to 1 and `normal_weight` to 0; both enter under a square root,
multiplied by their global solver weights. The solver uses coupled analytic
Jacobians over all editable frames and interpolated subframes. Pose and second
difference penalties discourage unnecessary changes and rough joins.

The agentic sequence workflow requires an explicit `temporal.max_velocity_rad_s`
hard gate. Optional `temporal.joint_velocity_limits_rad_s` maps native hinge
names to positive per-joint overrides. This speed gate checks **all native hinge
joints across the full exported sequence**, including frozen holds and every
join, using `abs(q[t+1]-q[t])/(timestamps[t+1]-timestamps[t])` (or `1/fps`
when timestamps are absent). An inherited or frozen-joint violation still fails
the full candidate; reports distinguish editable from inherited support. No
angle unwrapping, clipping or automatic threshold relaxation is applied.

Copy the `temporal` section from
[the G1 + Dex3 preset](examples/g1_dex3_temporal.json) into the case repair config:
3 rad/s default (fingers/legs), 2 rad/s for shoulders/elbows/wrists, 1.5 rad/s
for waist hinges. These are initial animation acceptance settings, **not
hardware velocity ratings**; declare a different policy before a comparison.
Unknown/non-hinge override names and nonpositive limits are errors.

For a cheap speed-only precheck without geometry queries:

```bash
python tools/repair/joint_velocity.py --config case/repair.json --trajectory case/candidate.npz --out case/joint_velocity.json
```

The command writes a hash-bound report and returns nonzero on a failed gate.
Passing leaves the motion unreviewed: acceleration, closure timing, collisions,
contacts and source fidelity still need their separate checks. The speed gate
measures adjacent native samples; it bounds slopes of their LINEAR replay, not
unexported Blender curve extrema.

Legacy `temporal.max_step_rad` and `max_acceleration_rad_s2` remain optional
local-window/halo gates over the editable hinge list. Without an explicit speed
gate, older configs report it as not configured; they do not inherit the preset
silently. Optional `source` contains `camera_matrix`
(3×3), `world_to_camera` (proper rigid 4×4), and `observations`. Each observation
has `frame`, `body_name`, `point_local`, `uv`, and `max_error_px`. Supply guides in
the audited window/halo; in-scope but unsampled observations fail coverage.
Whole-sequence phases and guides outside that interval are listed as out of
scope and never reported as passed.
Keep video detections and upstream reconstructed guides distinguishable in the
case provenance. Missing source checks report `NOT_CONFIGURED`.

## Continue from a selected smoother seed

Use an optional `iteration` block when the chosen working motion differs from
the original parent. Keep `trajectory_npz` and its expected hash pointing to the
original parent; do not recenter the correction corridor or posture prior:

```json
"iteration": {
  "seed_npz": "selected_seed.npz",
  "seed_sha256": "<SHA256 of the selected native archive>",
  "wrist_bodies": ["left_wrist_yaw_link"],
  "preserve_smoothness": true,
  "relative_tolerance": 0.0,
  "absolute_tolerance": 0.0,
  "derivative_weight": 1.0
}
```

The seed must match the original model, timing, immutable metadata and frozen
coordinates, and fit the original native correction corridor. Both Native and
Material proposals initialize editable values from this seed. Reports retain
separate original-parent, seed and candidate audits. The seed remains unchanged;
a rejected proposal is saved for diagnosis and never selected automatically.

With `preserve_smoothness`, each configured metric must stay within the stated
seed-relative tolerance. The default allows only numerical roundoff. Checks use
native timestamp finite differences for each edited hinge's velocity,
acceleration and jerk, actual wrist world-space SO(3) angular motion, wrist
translation, and velocity jumps at edited/frozen joins. Stencils touching an
editable pose include its frozen neighbors; the audit halo is at least three
frames. These discrete metrics do not certify continuous-time jerk of LINEAR
curves. The strict all-metric rule can reject a contact improvement; report the
tradeoff instead of silently loosening it.

`derivative_weight` adds normalized seed-relative hinge acceleration/jerk
residuals to either proposal objective. Native adds their weighted squared sum.
Material uses the same residual sum scaled by
`penalty / (clearance_weight^2 * full_sample_count)`, matching its fixed-contact
normalization; reports record the resolved coefficient and each iteration's
contribution. The original posture prior and existing regularizers remain fixed.
Nonlinear wrist measurements remain
independent rejection gates. Relative tolerance is dimensionless; absolute
tolerance applies in each metric's reported units. Declare any tolerance change
as a new policy before a trial, preserving previous evidence. Omitting
`iteration` retains the existing behavior.

In Blender, load this repair configuration while the live curves represent the
bound seed. The add-on accepts only float32 curve round-trip differences on
editable values; it does not redefine the fixed seed or widen the original
bounds. Preview rechecks actual evaluated poses and keeps Apply blocked after a
smoothness failure. Restore returns to the previous action. Export/reopen checks
still apply to the checkpoint chosen for delivery.

## Optional GPU proposals and audit acceleration

The default `backend: {"name": "native"}` retains the coupled CPU solver.
To request material-contact proposals in an external Torch/Warp runtime, add:

```json
{
  "backend": {
    "name": "material",
    "device": "cuda:0",
    "steps": 180,
    "lr": 0.003,
    "fit_substeps": 2,
    "smooth": 1.0
  }
}
```

This experimental backend optimizes the same declared native joint samples and
fixed material pad/normal targets, using differentiable native FK, projected Adam
and sampled signed mesh distances. It does not change a contact target to the
nearest surface. It preserves the original parent as its pose reference and keeps
protected endpoints and other coordinates exact. The CPU backend's witness
objective and this sampled-distance objective differ; switching backends is a
recorded method change. Optimizer fitting substeps are independent of audit
substeps. Source-image and configured temporal limits remain audit gates; the
proposal's pose, velocity and acceleration losses do not guarantee those gates.
Material repair uses `solver.contact_mode`, `contact_weight`, `normal_weight`
and `clearance_weight` for anchor weighting. Its step count, learning rate,
fitting times and smoothness come from `backend`. Native CPU `max_rounds`,
`max_evaluations`, `pose_weight`, `acceleration_weight` and `source_weight` do
not configure this different proposal objective; resolved scales are recorded
in its report.

The report records proposal meshes, sample coverage, options and runtime
versions. By default the proposal samples configured robot–object mesh pairs with an editable native ancestor;
`object_geoms` may explicitly select mesh IDs such as `["geom:65", "geom:72"]`.
The report lists omitted frozen proposal meshes. An explicit selection can retain
frozen meshes for diagnostics. This only changes proposal sampling: every
configured collision pair is still audited. Static environment and robot self-collision checks remain in that
independent audit. Open, thin or inconsistently oriented surfaces can make signed
distance misleading; an optimizer loss is never a collision certificate.

Install the optional `gpu-repair` extra in the **external solver Python**, with a
Torch build appropriate to its CUDA driver. The validated research environment
used Python 3.11.15, Torch 2.7.0+cu128, Warp 1.12.1, MuJoCo 3.7.0, NumPy 1.26.0
and SciPy 1.15.3. `device: "cpu"` is available for reference comparisons; CUDA
selection must be explicit (`cuda:N`). Missing runtimes or unsupported models
fail explicitly. The Blender Python environment does not need Torch or Warp.
Keep this environment separate from the pinned Newton diagnostics environment,
which requires Warp 1.9.1. Set **Solver Python** to the material runtime and the
Newton panel's Python path to its own runtime; installing this extra into that
Newton environment would replace its required Warp version.

Independently, `audit: {"geometry_backend": "warp", "device": "cuda:0", ...}`
uses conservative GPU AABB candidates with the same float64 CPU triangle tests.
The ordinary `geometry_backend: "cpu"` remains the default and now reuses
original-mesh BVH partitions. Both paths retain every original triangle; they do
not substitute convex hulls for final checks. Original degenerate triangles
report incomplete coverage on CPU; the optional Warp backend rejects them
before initialization. Neither path treats them as passed geometry.

An optional CPU candidate enumerator compiles the same float64 BVH traversal:

```json
"audit": {"geometry_backend": "numba", "device": "cpu", "substeps": 4, "halo": 2}
```

Install `agentic-retargeter[compiled-audit]` in the external solver Python.
The tested runtime is Python 3.11 with Numba 0.59.1 and NumPy 1.26; that pinned
Numba release requires NumPy below 1.27. The Blender Python environment does
not need Numba or LLVM. Audit and repair backends are independent, so this CPU
audit can accompany the material GPU proposal. The original triangle tests,
witnesses and gates remain unchanged; initialization/query errors are explicit
and never trigger an unreported fallback. Compilation/cache loading and extra
memory are part of its cost. The ordinary CPU backend remains the default.

## Iteration and review

1. Inspect the initial report. Fix a bad grasp hypothesis explicitly before
   spending iterations on it. Declare full relevant geometry from the outset.
2. Repair the difficult short transition. Native CPU rounds audit original
   triangles and add local clearance witnesses and fitting times at failures.
   The material backend makes one bounded proposal on its declared fitting
   schedule, then runs the same independent audit. Reports list optimizer
   outcomes, actual sample coverage, failures and reuse.
3. Review the resulting mounted hand, canonical contacts and source view. Check
   the reported temporal metrics and surrounding joins. A numerical pass alone
   leaves the candidate **UNREVIEWED**.
4. Import the chosen candidate into a new editable Blender checkpoint. Preserve
   the current scene. Reuse tool regression evidence when code/environment are
   unchanged; run the normal save/export/reopen checks for the delivered scene.
5. Propagate only after the grasp and short transition are reviewed, then perform
   the required sequence-wide checks. Reuse exact unchanged sampled states
   within one audit context; changed joints/inputs/configurations are recomputed.

Native repair runs locally on CPU with existing NumPy/SciPy/MuJoCo dependencies;
GPU proposal and broadphase runtimes are optional. The Newton Blender diagnostics
remain a separate path. The material backend uses Warp queries and native tensor
FK, not a Newton dynamics simulation. Local witness halfspaces are
optimization proposals, not penetration depths. Finite triangle samples do not
certify volume containment, continuous-time collision freedom, force closure,
dynamics, unlisted geometry, or unconfigured source constraints.

## Durable sample cache

Supply `--cache /durable/cache/audit` to `audit` and `repair` to share exact sample
rows between processes and attempts. The cache binds policy, parent hashes,
model/assets, compiled geometry, contact/source definitions, implementation and
numerical runtime versions. Relocating identical files alone does not invalidate
it. Each sample binds its frame, interpolated qpos and object transform. Local
edits automatically recompute changed samples and affected interpolation joins.

Every completed sample is atomically saved with a checksum, including failures.
An interrupted audit reuses these rows even if its final report was never written.
Missing/corrupt entries are recomputed. Coverage, frozen variables, native limits,
temporal and smoothness metrics are assembled again for the current trajectory.
Sampling density and full triangle count/witness policy remain unchanged.
`reuse` reports hit/evaluation counts and the persistent namespace. Use a trusted
writable cache; checksums detect corruption, not malicious fabrication.

```bash
# Diagnostic raw baseline; collision failures remain in the report.
python tools/repair/run.py audit --config repair.json --output runs/raw_audit \
  --cache /durable/cache/audit --diagnostic
# SAME fixed parent/config, different candidate; unchanged rows are reused.
python tools/repair/run.py audit --config repair.json --candidate candidate.npz \
  --output runs/candidate_audit --cache /durable/cache/audit
python tools/repair/run.py repair --config repair.json --output runs/repair_new \
  --cache /durable/cache/audit
```

`--diagnostic` only returns exit code 0 after a completed audit; failed `passed`
and `checks` fields remain unchanged. Use it for a baseline or rejected diagnostic,
never as a propagation gate. `--candidate` preserves the original parent prior
and all frozen/edit constraints. These options are also available through the
installed `local-repair` command.

Legacy case-script reports are not imported as cache hits: the first run through
this shared tool establishes its own bound rows. Later runs use those rows only
when their inputs/policy/runtime/code remain identical. This cache does not
replace dense final coverage with sparse screening.

## Authored approach routes and motion diagnostics

Use [the transition tools](TRANSITIONS.md) to turn explicit world-space clearance
waypoints and per-joint phase windows into a native seed, then batch scene/self
clearance and wrist-acceleration refinement. `tools/repair/transition.py` exports
an unselected candidate and runs one cached common audit by default. It preserves
reviewed holds and does not infer safe routes automatically.

`tools/repair/diagnostics.py` reports native wrist deviation, local derivative
peaks and optional human RMS excess without rerunning geometry. The same optional
`diagnostics` configuration can be included in normal audit/repair reports and
Blender worker snapshots. Review warnings do not change any acceptance gate.

## Preserve reviewed progress

`progress.py` stores immutable copies of independently audited candidates and
keeps four identities separate: the raw `baseline`,
`best_scoped_reviewed_partial`, `best_diagnostic`, and explicitly reviewed
`full_accepted`. A rejected later branch cannot overwrite a previously reviewed
partial. This is a comparison/evidence ledger, separate from the case time-budget
ledger; it does not launch, resume or select an agent or scheduler execution.

Create a fresh ledger from an independent audit. Pass `--candidate` even for the
baseline so the audit binds the exact archive hash:

```bash
python tools/repair/run.py audit --config repair.json --candidate baseline.npz \
  --output runs/baseline_audit --cache /durable/cache/audit --diagnostic
python tools/repair/progress.py init --directory runs/progress \
  --candidate baseline.npz --config repair.json \
  --audit-report runs/baseline_audit/report.json --contact-revision grip-v1 \
  --scope-kind local
python tools/repair/run.py audit --config repair.json --candidate candidate.npz \
  --output runs/candidate_audit --cache /durable/cache/audit --diagnostic
python tools/repair/progress.py record --directory runs/progress \
  --candidate-id trial_001 --candidate candidate.npz --config repair.json \
  --audit-report runs/candidate_audit/report.json --contact-revision grip-v1 \
  --visual-review visual_review.json --checkpoint candidate.blend
python tools/repair/progress.py status --directory runs/progress
python tools/repair/progress.py decision --directory runs/progress \
  --remaining-seconds 600 --next-step-seconds 180 --reserve-seconds 240
```

`--diagnostic` lets an audit finish with known failures; it does not turn them
into passes. Use new output directories and candidate IDs. Omit `--visual-review`
when no actual review occurred: that record stays `DIAGNOSTIC`. `--checkpoint`
is optional and preserves a file with its hash; supplying it does not establish
Blender round-trip verification.

Each record verifies the independent `AuditContext` report's candidate/config
hashes, parent/model/object bindings, model dependencies, current audit code,
sampled rows and affected/inherited failure partition. Solver summaries are
rejected. `screening_only` or incomplete-scope reports cannot promote a scoped
partial. One ledger fixes its config, coverage, contact revision and metric;
changed hypotheses or coverage require another comparison ledger and a matching
baseline audit. Keep old evidence rather than comparing different targets as
improvement. Frozen/inherited failures remain in `gates`; `scoped_gates` reports
only failures attributable to editable support. Missing, `NOT_CONFIGURED` and
`NOT_IN_SCOPE` checks remain `UNKNOWN` in both views.

The default required gates are contacts, collision, source, temporal and
invariants. An explicit `--required-gates` list must include invariants; declare
any narrower local review scope before comparing, and retain the omitted or
unknown full-motion checks. Do not remove a missing source check to claim full
acceptance. For authorized generated-hand correction, revise the source policy
within that scope while preserving task intent and source evidence; unknown
source validation is still unknown. The separate sequence material-slip review
is not automatically imported into this ledger and remains required when
assessing a hold.

The default objective is the independent audit summary's `max_contact_gap_m`.
`--objective` may name another finite numeric summary field whose improvement
means **smaller**. The helper updates a best slot only for strict improvement
over its baseline and previous slot under the same policy. It does not infer
metric tolerances, certify a tradeoff, or compare unrelated contact revisions.

A visual receipt has this schema (paths are relative to the receipt):

```json
{
  "candidate_sha256": "<native archive SHA256>",
  "scope_sha256": "<progress.json policy.scope_sha256>",
  "contact_revision": "grip-v1",
  "reviewed_by": "<actual reviewer>",
  "decision": "PASS",
  "evidence": [{"path": "review.mp4", "sha256": "<media SHA256>"}]
}
```

Use `PASS`, `FAIL` or `UNKNOWN` to record the review performed. A receipt is a
hash-bound human/agent assertion, not proof that anyone watched or correctly
interpreted the media. Required scoped gates, complete configured scope and a
visual `PASS` produce `SCOPED_REVIEWED_PARTIAL`; otherwise the candidate remains
`DIAGNOSTIC`. Both are retained without accepting the complete motion.

For full acceptance, initialize a separate `--scope-kind full` ledger with a
full-timeline audit config and required policy, then explicitly use
`record --accept-full` after the other required reviews. This requires complete
configured integer/subframe coverage, no audit failures, all required gates and
a bound visual `PASS`. It is acceptance only at the declared sampled scope,
not continuous collision clearance, force closure or dynamics.

`decision` returns `FINISH` only after that explicit full acceptance, `CONTINUE`
when the declared next step fits before the delivery reserve, otherwise
`FINALIZE`. Supply actual remaining time from the persistent case-budget record;
this helper does not measure or reset that budget. Its `task_completed` concerns
the ledger's full scope; the owning agent must separately assess whether a
requested narrower correction is complete. No automatic retry, agent continuation
or checkpoint adoption is implemented by this command.
