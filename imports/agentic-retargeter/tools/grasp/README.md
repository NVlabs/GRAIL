# Surface grasps and shared holds

These foreground tools replace per-case whole-hand search and mounted hold
scripts. Implementation lives in `src/grasp_io.py`, `src/surface_grasp.py` and
`src/shared_grasp.py`; no historical run directory is imported. They propose
native motions for the existing agent/Blender review loop. They do not infer
source phases, automatically choose a candidate, or certify a full animation.

For new hand-target SOMA initializations, use the
[shared initialization policy](../../docs/hand-shared-policy.md) to orchestrate the
[packaged Topo-inspired candidate bank](../../docs/topo-hand-initialization.md)
(`tools/grasp/topo.py`) and [all-frame hand sequence/handoff](../../docs/hand-initialization-workflow.md).
The default retains a base paired solve, applies a phase-aware SG initial guess,
then performs contact-aware temporal refinement before selecting hand targets.
The standalone pad correction pass is disabled by default (`pad_steps: 0`);
enable a positive step count only for an explicit ablation. Contact/collision
terms and original contact-surface sampling remain active. Read the bank
runner's `final_bank` and `final_bank_sha256` from `report.json` rather than
assuming a `pad/bank.json` result. Existing explicit pad configs and historical
outputs retain their original meaning; this default change does not rerun them.
The generic bank and fixed-material mounted tools below remain available for
explicit grasp/contact experiments and scoped repair.

```bash
python tools/grasp/bank.py --config bank_config.json --out runs/bank_v001
python tools/grasp/hold.py --config hold_keys.json --out runs/keys_v001
python tools/grasp/evaluate.py --config evaluation.json --output runs/screen_v001
# Fit a fixed-material hold and its entry/exit buffers when mounting leaves slip/joins:
python tools/grasp/sequence.py --config sequence.json --output runs/sequence_v001
```

Installed wheel equivalents are `python -m agentic_retargeter.surface_grasp`
and `python -m agentic_retargeter.shared_grasp`, with identical arguments.
Outputs require a new directory. Stage these commands in the existing
[batch runner](../batch/README.md); the agent that owns each case reviews
gates, calls local repair for approach/closure and protects reviewed holds.
The cumulative 30-minute case deadline includes these commands. Failed gates
block acceptance; a rejected mount can initialize an explicitly diagnostic
`sequence.py` repair with `seed_review`, retaining its rejection. This does not
promote the seed or replace independent review of the new export.

## Bound inputs

Both configs contain the identical `inputs` object, illustrated below. Paths
resolve relative to the config file. Indices and names are examples: enumerate
the actual native model and author source-supported object regions for each
case. An object NPZ stores **original unscaled** `vertices` and triangular
`faces`; native `object_scale` is applied once. Do not use a remeshed/hull proxy.

```json
{
  "native_npz": "input/native.npz",
  "model_xml": "input/model.xml",
  "object_mesh_npz": "input/object_original.npz",
  "hands": {
    "right": {
      "wrist_body": "right_wrist_yaw_link",
      "finger_joints": ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_index_0_joint", "right_hand_index_1_joint"],
      "robot_geoms": [101, 102, 103, 104],
      "contact_groups": {"thumb": [102], "index": [103], "middle": [104]},
      "region_faces": [10, 11, 12]
    }
  }
}
```

Native inputs require `qpos`, scalar `fps`, `obj_R`, `obj_t`, `object_scale`;
optional joint/timestamp metadata is checked. Only limited named hinges are
editable. All configured hand geoms must be original mesh geoms downstream of
the named wrist. Multiple geoms on one body are retained separately. Include
the palm, wrist and **all** finger links in `robot_geoms` for clearance; group
only surfaces intended to make contact. For example, include proximal and
distal thumb geoms together in `thumb`; add a palm contact group only when that
grasp calls for palm contact. No fingertip or finger-pad convention is imposed.

Original model XML/includes/assets, native parent, object and hand definitions
bind the bank. The mounted solve rejects a bank when these inputs change.
Changing phases/targets is a new documented hypothesis; do not silently relabel
nearest points after propagation. Relocate inputs/configs together before
creating the bank, rather than moving only a bank file between environments.

### Reuse upstream object contact regions

A bound HOI contact map can supply the object-side region and phase prior.
Verify original mesh topology/vertex ordering, scale, source video and frame
mapping before transferring IDs. For `grail.contact_map.v1`, human and object
vertex lists are independent regions, not paired correspondences; authored
strengths are not learned contact probabilities or a claim that all vertices
touch. Keep its review status and uncertainty.

Preserve thumb and other-finger roles. Union the relevant human non-thumb object
regions for the robot long-finger role, combining overlapping per-vertex
strengths by maximum rather than addition. Retain the original region records
and missing-finger evidence. If those object regions are identical, their union
does not enlarge the region; robot digit participation still needs explicit
`contact_patch.members`. Human vertex IDs cannot be reused as robot vertex IDs.

Translate positive object vertex support to original triangle IDs using a
declared boundary rule; keeping only triangles whose three vertices belong to
the region is a conservative option. Set `contact_region_faces` and the hand's
enclosing `region_faces` consistently. Version any intentional region change
instead of silently replacing the upstream prior with a geometric bounding box.

The bank consumes binary face masks, not upstream strengths or temporal fields.
Carry strengths as provenance unless an explicit weighted objective consumes
them. Keep phase intervals in the case plan and configure downstream contact
activation in its actual interval convention. Preserve `unknown` separately
from `no_contact`, and do not turn a sliding-support hypothesis into a rigid
hold. Generate robot/object material pairs within these priors and review the
mounted native grasp; inherited human contacts do not establish robot contact.

## Batched free-wrist bank

The bank config has `inputs` as above plus:

```json
{
  "seed_frames": {"right": [26, 60, 96, 120]},
  "device": "cuda:0",
  "steps": 420,
  "screen_backend": "numba",
  "samples_per_geom": 800,
  "translation_limit_m": 0.08,
  "rotation_limit_rad": 0.8,
  "clearance_m": 0.0015,
  "learning_rate": 0.008
}
```

`device`, `steps`, `screen_backend`, and `seed_frames` are required. The other
values shown are defaults. Install the `gpu-repair` extra for Torch/Warp and
`compiled-audit` for the optional `numba` screen; `cpu` works for either device
or screen. There is no silent CPU fallback. The configured seeds are optimized
as a batch per hand and share mesh queries/FK. Seed diversity does not prove
distinct grasp families. Translation/rotation search bounds are **componentwise**
corrections to the seed's free wrist, in object coordinates; fingers retain
their original native limits. Contact-normal alignment is absent.

To search several initial poses around each source frame in the same GPU batch,
add an explicit `multistart` block:

```json
{
  "multistart": {
    "count_per_frame": 8,
    "seed": 20261024,
    "translation_radius_m": 0.04,
    "rotation_radius_rad": 0.45,
    "finger_radius_rad": 0.35
  }
}
```

The count includes the unchanged initial pose at `start_index: 0`. Three source
frames and eight starts produce 24 candidates per hand. Other starts perturb
wrist translation, wrist rotation and native finger angles. Translation and
rotation-vector radii bound vector norms; the finger radius applies per joint.
All starts retain the same
original source pose prior, search bounds, contact regions and collision rules.
Initial finger angles are clipped to native limits. The random seed makes the
bank reproducible, and candidate records retain their frame, start index and
initial delta. A hand batch is limited to 128 candidates; candidates in a batch
remain independent pose hypotheses, with no temporal coupling between them.

Omitting `multistart` retains one start per source frame. Original-triangle
screening and material-anchor extraction still run per candidate after the
batched optimization, so total time need not scale like GPU optimization time.
Phase timings distinguish these costs. A larger bank does not change rejection
criteria, select a grasp automatically, or certify mounted reach.

Optional `contact_region_faces` maps hand names to contact-group face lists,
e.g. `{"right": {"thumb": [10, 11], "index": [12, 13]}}`. Each list must be a
subset of that hand's bound `region_faces`. Use it for source-supported opposing
surfaces (such as a thumb above a rim and long fingers below); unspecified groups
retain the common region. Search and frozen anchors use the same group region.
These remain authored hypotheses; distinct regions alone do not certify wrapping
or force closure.

The objective combines sampled winding-distance clearance, closest sampled
surfaces in each declared contact group to the object region, and a seed prior.
Finite surface samples and winding on open meshes are heuristics. Each proposal
then gets original triangle screening of all listed hand/object pairs and
nonadjacent hand pairs, stopping at the first forbidden crossing. Same-body and
adjacent pairs are listed as **omitted**, not cleared; whole robot, environment,
containment and dynamics are outside this free-wrist screen.

Original object faces are strict by default. For a source asset containing exact
zero-area faces outside every authored contact region, both bank and mounted
configs may explicitly set
`inputs.object_degenerate_policy: "exclude_zero_area_queries"`. This preserves
the original vertices, faces and material face IDs. Only the normal-dependent
proposal query uses the ascending complement of the zero-area face IDs; its
provenance records those IDs, the face-map hash, original/query hashes and
`query_geometry_complete: false`. Nonzero near-degenerate faces and float32
collapse remain errors. Contact regions containing an excluded face remain
errors. CPU/Numba original-triangle screening still sees every original face,
reports incomplete/unknown coverage and rejects the proposal; this option does
not make it a cleared grasp or authorize mounting a rejected bank entry.
Generic Warp query/audit APIs remain strict. Fixed-material diagnostics can use
the original nondegenerate regions, with the geometry limitation retained.

`bank.json` contains the object-relative wrist transform, named finger values,
loss/clearance diagnostics, coverage and rejection state for each proposal.
Without `contact_patch`, it freezes one material-point hypothesis per contact group: robot body-local
point and original robot/object triangle/barycentric coordinates. Robot face IDs
refer to the hash-bound compiled triangle arrays. Winding normals are retained
as diagnostic metadata with `normal_weight: 0`. These are proposed
correspondences, not forces or learned contact probabilities. Evaluate them on
the mounted trajectory using independent fixed-anchor audits; do not replace
them with framewise nearest contacts to hide slip.

### Multi-point contact patches

For a grasp requiring contact coverage across multiple digits, opt into a
distributed material descriptor with `contact_patch`. This example keeps the
source-derived long-finger group while declaring its two participating robot
members separately. Geom IDs are illustrative; bind the actual model.

```json
{
  "contact_patch": {
    "points_per_member": 8,
    "max_gap_m": 0.015,
    "min_spacing_m": 0.003,
    "members": {
      "right": {
        "thumb": {"thumb": [102]},
        "long_fingers": {"index": [103], "middle": [104]}
      }
    }
  }
}
```

Members must partition the configured contact groups' geoms without overlap.
Only declare surfaces intended to participate; whole hand clearance still uses
`robot_geoms`, including auxiliary fingers. The settings above are a starting
example, not universal counts or a requirement that all three fingers touch.
Group-specific object regions remain in `contact_region_faces`.

The search attracts samples separately for each member and averages their
contributions within a group. Export selects up to the requested number of
spatially distributed pairs, each within `max_gap_m`; both hand and object
points must satisfy `min_spacing_m`. Duplicate or distant points are not added
to fill the quota. Coverage shortfalls are reported per member and reject that
patch proposal, even if its separate triangle screen passes. Increasing
`samples_per_geom` can improve sampling coverage but cannot make an unreachable
digit touch or prove that a stable grasp exists.

Each exported pair retains the original triangle/barycentric identities and
its group/member labels. Its `position_weight` normalizes the selected points
within each member and the members within their group, so a completely covered
group has total weight one. When converting pairs to pose/repair contacts,
preserve these weights, use unique names containing member and point index,
and apply any desired role weight to the whole group. The mounted `hold.py`
solve uses these pairs as fixed residuals when `finger_refinement` is supplied;
without it, the legacy mode fits the stored wrist pose with the finger shape
fixed. Independent material audits and downstream repair still need their bound
fixed-contact configuration.

These are geometric contact hypotheses, not learned contact probabilities.
Plot dense measured distance fields with metric units and declared sampling;
keep them distinct from eligible regions and the frozen solver targets.
Unsigned proximity is not penetration depth, and a populated patch does not
certify opposition, force closure, mounted reach or trajectory acceptance.
Changing a descriptor creates a new comparison contract; report old and new
coverage without treating losses under different targets as a matched score.

Optional `self_clearance_weight` (default `0`) enables bidirectional sampled
nonadjacent hand-mesh clearance during search. Set it explicitly in a new
configuration; the original-triangle rejection screen remains unchanged.
An optional `retained_self_pairs` maps hand names to entries
`{"pair": [geom_a, geom_b], "reason": "documented assembly evidence"}`. The
tool requires every such pair to intersect in the original neutral model,
excludes it from the proposal penalty, and retains its actual crossings and
neutral evidence in the report. This is an explicit scoped exclusion, not a
clearance certificate or permission to deepen an assembly overlap. There are
no default retained pairs. Compare retained geometry before/after separately.

## Mounted keys and full stable holds

Include the same `temporal` speed policy as the downstream repair config when
exporting full hold sequences; see the [hard speed gate](../repair/README.md).
`hold.py` checks all native hinges and all adjacent frames after propagation,
including acquisition/release joins and unchanged channels. An exceeded limit
writes `REJECTED_BY_JOINT_VELOCITY` in both candidate and report. A pass leaves the
candidate unreviewed. `max_joint_step_rad` is only a solver step bound and does
not replace this FPS-aware final check. Sparse mounted keys may fail continuity
before transition repair; retain them as pose proposals, never as an accepted
animation. A missing `temporal` section explicitly reports `NOT_CONFIGURED`.

The hold config has identical `inputs` plus:

```json
{
  "bank": "runs/bank_v001/bank.json",
  "holds": {"right": {"candidate": 0, "window": [26, 121]}},
  "frames": [26, 60, 96, 120],
  "editable_joints": {
    "right_shoulder_pitch_joint": [-1.5, 1.5],
    "right_elbow_joint": [-1.5, 1.5],
    "waist_yaw_joint": [-0.20943951, 0.20943951]
  },
  "collision_pairs": [],
  "body_guides": {"right_elbow_link": 0.04},
  "finger_refinement": {
    "max_delta_rad": 0.35,
    "contact_weight": 4.0,
    "wrist_weight": 0.2,
    "posture_weight": 0.02,
    "max_gap_m": 0.015
  },
  "max_joint_step_rad": 0.13,
  "max_nfev": 90
}
```

Declare every editable arm/waist hinge and its correction interval against the
**original** native parent; the list above is abbreviated. `collision_pairs`
must explicitly list robot geom ID pairs for native geom-distance separation.
An empty list supplies no robot self-collision objective. Those MuJoCo proxies do **not**
replace original triangle checks; moving-object proximity uses the separate term below. The bank only
screens free hands; mounting can introduce new wrist/arm/body/object collisions.

Joint finger refinement enables `object_clearance` by default. It samples
original robot surfaces downstream of the declared arm/finger joints and queries
the original scaled object triangles at the same adjusted object pose used by
the contact residuals. Both native joints and authorized object offsets receive
its derivatives. Without `finger_refinement`, the legacy rigid mode enables it
only when explicitly configured. Settings are:

```json
"object_clearance": {
  "enabled": true,
  "weight": 20.0,
  "margin_m": 0.003,
  "samples_per_geom": 32
}
```

This is a finite **unsigned** near-surface barrier. It does not resolve deep
containment, infer the safe side of a crossing or certify nonpenetration. Its
margin can compete with intended contact; review both objectives and the
declared tolerances together. The report records disabled guidance, omitted
geometry and ambiguous zero-distance samples. Keep the independent original-mesh
robot/object, robot/table and object/table audits, including frozen geometry.

When the native input carries packed static scene geometry, shared holds also
enable original-surface scene proximity guidance by default. It covers robot
mesh geoms downstream of the editable joints, including arm, wrist, palm and
finger meshes beyond the bank's hand list. The report lists frozen meshes and
unsupported geometry separately; their absence from the proposal is not a
clearance pass. The original scene arrays and world placement remain exact.

Optional hold settings (shown with defaults):

```json
"scene_clearance": {
  "enabled": true,
  "weight": 20.0,
  "margin_m": 0.003,
  "samples_per_geom": 32
}
```

This CPU term repels fixed robot surface samples within the margin of the
original scene triangles. It uses unsigned nearest-triangle distances and native
joint Jacobians, with cached scene geometry; it adds no Torch/Warp dependency.
It does not infer an exterior from open-mesh winding or replace the table with
a hull or infinite plane. It is a near-surface barrier, **not a penetration-depth
loss**: samples deep inside a closed object, intersections between samples and
exactly-on-surface ambiguous directions still require independent geometry
checks and a changed pose/path. Inspect reported degeneracies and zero-distance
queries. An explicit disable is recorded; missing scene data remains unknown.

Review mounted keys and short transitions with the original-mesh audit before
propagation. Scene guidance cannot turn a previously colliding seed into an
accepted grasp by itself. Keep collision frames outside the solved hold in the
case's approach-repair plan and full-motion review.

Windows are zero-based half-open. `frames` is an increasing list of actual
integer frames to solve. Sparse keys do not synthesize interpolation. After the
required gates, supply every hold frame for propagation, using the same bank
and candidate choices. One shared grasp descriptor per hand per invocation is supported;
use distinct, explicitly reviewed configurations for regrasp/sliding phases.
Unsolved frames and all unnamed robot channels stay exact. Without
`finger_refinement`, finger values are fixed to the candidate within solved hold
samples; the joint mode below permits bounded adaptation. Approach and hold boundaries
need the existing [local repair](../repair/README.md), with passed hold samples
protected. Do not propagate rejected free-grasp candidates.

At a hand's first active frame, its unique arm ancestry is fitted without a
step clamp to the previous unfitted pose. Joints shared with an already active
hand retain the step clamp. The acquisition transition still needs repair and
review. Sparse object-edit segments reserve their return to the next unsolved
sample throughout the segment, preserving the declared offset-speed limit.

Four wrist-local guide points fit the shared transform; no contact-normal loss
is added. Optional `body_guides` retain original native body positions, e.g.
elbows. The solver uses analytic joint Jacobians, warm starts, posture and
temporal priors. `max_joint_step_rad` bounds arm/waist changes between consecutive
solved samples. Sparse keys and unsolved approach joins are not constrained to
the old raw pose; they require transition repair/review. This is not a
finger-speed or derivative audit. Optional
`posture_weight`, `temporal_weight`, `collision_weight` default to 0.0002, 0.012,
20. An empty correction corridor fails explicitly.

### Joint arm and finger fitting

For a new or poor grasp, supply the top-level `finger_refinement` block shown
above when comparing mounted candidates. All five fields are required; the
example values are trial settings, not defaults. Omitting the block retains
the legacy rigid-finger mode for already reviewed hand shapes. Both modes remain
unreviewed proposals.

| Field | Meaning |
|---|---|
| `max_delta_rad` | Positive per-finger-joint correction bound around the chosen bank shape, also limited by original joint limits; at most pi radians. |
| `contact_weight` | Positive scale for fixed material-contact residuals, normalized within each declared group/member. |
| `wrist_weight` | Nonnegative scale for the bank's wrist-pose guide; the wrist is a soft target rather than a rigid attachment. |
| `posture_weight` | Nonnegative scale for retaining the candidate finger shape while fitting contact. |
| `max_gap_m` | Positive final material-gap threshold, at most 0.1 m, checked at the solved active hold frames. |

The solver adds the active hands' finger joints to the named arm/waist variables
and fits them together. Do not duplicate those finger joints in
`editable_joints`. The chosen bank entry must pass its free-hand screen and
contain valid `material_anchor_hypotheses` on the bound original hand/object
meshes; declared multi-point coverage must be complete. Face/barycentric and
body-local material identities stay fixed throughout mounting. The solver does
not relocate contacts to nearby surfaces. Both active hands share any configured
`object_motion` correction and its existing bounds.

A failed solved-frame gap gate writes `REJECTED_BY_MATERIAL_CONTACT`; configured
temporal failures remain separately reported. A passing fit stays
`UNREVIEWED_SHARED_GRASP` with `motion_accepted: false`. This gate checks fixed
contact gaps, not force closure, support, forbidden penetration, subframes or
full-hold slip. Inspect the mounted hand, arm and original table/object meshes
before evaluating short transitions or propagating the full hold. If mounting
fails, revise the palm seed, grasp layout or arm branch rather than treating the
free-hand fit as accepted.

`body_guides` are position priors from the supplied native robot motion. To keep
mapped human targets or a reviewed native posture reference independent of the
changing grasp seed, use `body_reference` below. Joint finger fitting does not
by itself add the full SOMA retargeting objective, infer source contact maps, or
generate multiple palm initializations.

### Fixed body targets during mounted fitting

Optional top-level `body_reference` supplies world-space position and rotation
targets for named native robot bodies. Its required config fields are `npz`,
`sha256`, `evidence_kind` (`mapped_human` or `native_reference`) and
`source: {path, sha256}`. Paths resolve relative to the config; hashes are actual
lowercase SHA256 digests of the target NPZ and original source file. Keep these
bindings unchanged while comparing grasp seeds. Use `mapped_human` only for
explicitly calibrated human-to-robot targets; the loader does not infer a
correspondence or perform retargeting/calibration.

The target NPZ contains the following arrays, with `N` native frames and `B`
unique, existing non-world robot body names:

| NPZ fields | Shape and contract |
|---|---|
| `body_names` | `(B,)` string array. |
| `positions`, `rotations` | `(N, B, 3)` metre positions and `(N, B, 3, 3)` proper rotation matrices, in native world coordinates. |
| `position_weights`, `rotation_weights` | `(B,)` nonnegative residual scales, with at least one active objective. |
| `fps`, `timestamps` | Scalar FPS and `(N,)` timestamps exactly matching the native input, including source frame alignment. |
| `units`, `world_frame` | Scalar strings `m`, `native_world`. |
| `source_sha256` | Scalar string matching the bound source file's SHA256. |
| `model_binding_sha256` | Scalar string equal to `grasp_io.digest(context.binding["model_dependencies"])` for the bound native model. |

Position and rotation-matrix residuals participate in the mounted solve alongside
contacts and the selected editable arm/waist/finger controls. The report retains
per-body position/angle errors and target provenance. These are soft objectives;
source fidelity, support and collision acceptance still need independent checks.
The input native `body_guides` can remain active too, so choose their weights
explicitly if they conflict with mapped human targets. This change supplies the
target contract and fitting path, not an automatic target exporter or the full
SOMA objective. See [`BodyReference`](../../src/grasp_body_reference.py).

### Bounded shared object motion

Object motion stays frozen unless `object_motion` is explicitly provided with
case-authorized bounds, for example:

```json
{
  "window": [27, 121],
  "translation_limit_m": 0.05,
  "rotation_limit_rad": 0.1745329252,
  "max_translation_step_m": 0.003,
  "max_rotation_step_rad": 0.0130899694,
  "ramp_frames": 12
}
```

These are trial values, **not** universal defaults. Translation and rotation-vector
offsets are jointly solved relative to the original object pose, with conservative
component boxes divided by $\sqrt{3}$ to enforce norm caps. Rotation is applied
in the world frame. Offset changes are bounded per source sample (not absolute
object speed); ramping and protected joins constrain both window boundaries.
All object samples outside the window stay exact. Both hands share the same
object path. Offset step limits do not certify smoothness relative to human
motion; retain the configured motion audit.

The outputs are `candidate.npz` and `report.json`. Old FK/velocity/acceptance
fields are invalidated, timing and frozen channels retained. A converged solve,
small guide residual or a passing free-hand screen is never full-sequence
acceptance. Bind independent fixed-anchor contact, full geometry, temporal,
source and Blender save/export/reopen evidence to the exact output. The tool
does not automatically insert these new commands into the Blender sidebar.

## Compare mounted candidates on common review frames

`evaluate.py` audits a finite list of already mounted native proposals under
one parent, original geometry, edit scope and gate policy. It does not mount
free hands or solve a new trajectory. Supply the first/last hold windows,
acquisition/release windows and additional difficult arm poses:

```bash
python tools/grasp/evaluate.py --config evaluation.json --output runs/screen_v001 \
  --cache /durable/cache/audit
```

```json
{
  "schema_version": 1,
  "candidates": [
    {"id": "grip_a", "native_npz": "mounted_a.npz", "repair_config": "audit_a.json", "contact_revision": "grip-a-v1"},
    {"id": "grip_b", "native_npz": "mounted_b.npz", "repair_config": "audit_b.json", "contact_revision": "grip-b-v1"}
  ],
  "hold_windows": [[26, 120]],
  "transition_windows": [[18, 30]],
  "difficult_frames": [60, 90],
  "substeps": 4,
  "required_gates": ["contacts", "collision", "temporal", "invariants", "source"]
}
```

Paths resolve relative to this manifest. Windows have **inclusive** increasing
integer endpoints. The sample set includes each hold's first/last frame and
largest object-rotation frame, every configured transition substep, and the
additional difficult frames. Largest object rotation is a heuristic; add known
difficult arm configurations explicitly. All requested frames must fit the
repair audit scope. Configured full-trajectory derivative gates still run.

Candidates must share the same parent, geometry, scope and gates. Material
coordinates may differ as explicit contact hypotheses; distinct definitions
need distinct `contact_revision` values. Their gaps are feasibility measurements,
not matched improvement scores. Use a new comparison when the other constraints
change. `required_gates` defaults to contacts/collision/temporal/invariants;
add source and smoothness when required by the comparison. Missing or out-of-scope
checks are unknown, not passes; omitting a gate here never supplies that evidence.

Outputs are per-candidate `*.audit.json`, incremental `completed.json`, and
`report.json` with a `review_order`. Ordering prefers complete screen passes,
then fewer failed/unknown gates, preserving input order on ties; it does not rank
hypotheses by smaller gaps. `SCREENED_UNSELECTED`, `selected=false` and
`screening_only=true` remain even when `full_scope_covered` is true. These reports
cannot promote a progress checkpoint. The CLI's successful exit means the
screen completed, including when every candidate failed.

Review the results and run the bounded coupled repair below when useful; a
rejected mount may initialize a declared diagnostic branch. Before promotion,
run an independent full-scope audit of its exact exported file and complete
source/visual and full-hold gap/slip review. See
[progress retention](../repair/README.md#preserve-reviewed-progress).

## Coupled hold and acquisition/release repair

`sequence.py` uses the existing transition optimizer to fit fixed material
contacts over a complete hold together with editable acquisition and release
buffers. A mounted grasp, including a temporally rejected mount, can initialize
the solve. Its rejection status is retained in `seed_review.json`; it supplies
no acceptance evidence. The first hold sample must remain editable and have a
preceding buffer, so an unreachable first-touch key is not frozen in place.

```bash
python tools/grasp/sequence.py --config sequence.json --output new_sequence_run
```

The output directory must not already exist. The configuration below shows the
interface; replace the abbreviated joint and contact lists with the actual
native controls and every configured fixed contact. Numeric tolerances are
case-specific examples, not robot hardware limits.

```json
{
  "schema_version": 1,
  "repair_config": "repair.json",
  "seed_native": "mounted_seed.npz",
  "seed_sha256": "<sha256 of mounted_seed.npz>",
  "seed_review": {
    "status": "DIAGNOSTIC",
    "reason": "Mounted hold has an unreviewed acquisition transition"
  },
  "wrist_bodies": ["right_wrist_yaw_link"],
  "joint_windows": {"right_elbow_joint": [0, 120]},
  "holds": [{
    "name": "right_hold",
    "window": [26, 120],
    "contact_names": ["right_thumb", "right_long_1", "right_long_2"],
    "joint_names": ["right_elbow_joint"],
    "max_material_slip_m": 0.01
  }],
  "contact_substeps": 4,
  "solver": {
    "device": "cuda",
    "steps": 240,
    "max_seconds": 240,
    "derivative_units": "seconds",
    "contact_mode": "target",
    "projection_scope": "editable_hinges",
    "velocity_projection": true,
    "weights": {"route": 0, "contact": 1000},
    "temporal_projection": {
      "acceleration_limit_rad_s2": 20,
      "jerk_limit_rad_s3": 300
    }
  }
}
```

`repair_config` is the existing [local repair configuration](../repair/README.md):
it binds the native parent, robot/object/original scene meshes, editable joint
corridors, protected samples, fixed material contacts, and the full-motion
velocity gate. Each hold window is **inclusive**, unlike `shared.py`'s
half-open hold windows. It must exactly match its contacts' `active_frames`.
Every repair contact must belong to one hold. `joint_windows` must list exactly
the repair's editable joints, and each participating hold joint must cover the
whole hold and its editable entry/exit buffers. Seed changes outside these
buffers are rejected. No wrist IK or raw wrist path constraint is added.

By default the native parent remains the posture prior and correction reference. Seed
and parent must retain the same root/object/scene/timeline and all protected
samples. The existing transition contract protects the ends of the repair
window, including timeline frames 0 and 120 in the example. A hold extending to
the last frame therefore needs a usable **explicit parent endpoint**: if that
endpoint needs editing, author a separate native parent with only the declared
endpoint robot controls changed, record those changes and the raw parent hash,
and keep the untouched raw motion as `diagnostics.baseline_npz`. Bind the seed
to that authored endpoint. This tool neither adopts the seed endpoint
automatically nor pads or retimes the source. An authored endpoint still needs
independent contact, collision, temporal and source review.

### Keep the original native posture as a separate reference

When a mounted seed or authored endpoint has already changed the body pose,
bind the original human-retargeted native motion separately. The optional
top-level `posture_reference` works in both sequence and transition configs:

```json
{
  "posture_reference": {
    "native_npz": "original_native.npz",
    "sha256": "<sha256 of original_native.npz>",
    "window": [0, 120],
    "joint_deviation_limits_rad": {
      "waist_yaw_joint": 0.1745329252,
      "left_elbow_joint": 0.3490658504
    }
  }
}
```

These limits illustrate the schema; choose them explicitly for the case. There
are no default posture limits. The reference must contain matching
`compiled_model_sha256`, joint-order metadata (`dof_names` or `joint_names`),
native dimensions, FPS and timestamps. Its path and file hash are bound and
rechecked before completion. Object transforms may differ: the reference
supplies posture only. In particular, an authorized `hold.py` object-motion
proposal can become a newly bound parent while this reference remains the
original native motion. A seed still cannot silently change the current
parent's object, scene, timing or protected robot samples.

The **parent** defines the edit corridor and frozen samples; the **seed** is the
initial trajectory; the **posture reference** supplies the `weights.posture`
target for the listed joints inside its inclusive window. Unlisted joints and
frames keep their existing parent prior. Existing wrist-route behavior is
unchanged. The final `posture_reference_gate` checks every listed hinge and
integer frame, including frozen samples and frames outside the repair window.
It reports peak/rms deviation, worst frame and violating frames per joint. An
exceeded limit produces `REJECTED_BY_POSTURE_REFERENCE` and a nonzero CLI exit
even if contact and geometry checks pass. Check this gate alongside the other
gates; a geometry-only audit does not evaluate this extra policy.

This is a native-posture proxy, not a human-similarity score or a source-image
acceptance test. Keep torso/elbow and source-video review separate. Merely
adding `diagnostics.human_reference` supplies advisory motion statistics; it
does not create this objective or gate. With no `posture_reference`, the prior
and acceptance behavior remain unchanged.

`contact_mode: "target"` makes the existing contact objective attract each
robot material point to its fixed object anchor even inside the allowed gap.
It is a **soft objective**, not a hard contact constraint; `max_gap_m` and the
independent slip gate still decide whether the result meets those criteria.
The ordinary transition tool retains its previous `within_gates` default.
`normal_weight: 0` remains valid for video-derived regions without a normal
correspondence. Contacts are fixed hypotheses supplied by the caller; the tool
does not infer them from video.

`projection_scope: "editable_hinges"` lets the proposal solver enforce temporal
limits for movable hinges without aborting on an inherited frozen-leg failure.
It still constrains their full trajectories and joins. This changes only the
proposal projection: the final whole-motion velocity and derivative checks
retain every configured hinge, and an inherited frozen failure remains a
rejection. The tool does not move those frozen channels or relax their limits.

`material_review.json` evaluates every configured hold, even when the mesh
audit rejects early. It reports sampled fixed-anchor gaps and material slip
separately. Slip is the maximum displacement of the same robot material point
in object coordinates relative to its first active sample. The additional
range-diameter field is explicitly an upper bound. Omit `max_material_slip_m`
only when a diagnostic is intended: missing thresholds are reported as
`NOT_CONFIGURED`, and a passing gap check does not establish a slip pass.

Outputs include the complete `candidate_UNREVIEWED.npz`, `report.json`, bound
repair/transition configs, `seed_review.json`, `material_review.json`, and the
underlying optimizer outputs under `coupled/`. Reports distinguish fitting
window and projection scope from full-motion gates. A numerical pass stays
`UNREVIEWED_GRASP_SEQUENCE` with `motion_accepted=false`; source/visual and
Blender save/export/reopen evidence are still required. The CLI returns 2 for
failed or incomplete configured gates, including `--proposal-only`, while
preserving the diagnostic candidate and reports.
