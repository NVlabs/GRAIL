# Video-reviewed contact intervals before Stage 2

The reconstruction agent inspects the source video **after Stage 1 placement
review and before Stage 2**. It authors a timeline independently for each hand:
which frames show contact, what part of the hand touches which object surface,
and where that relation changes. There is no automatic coarse VLM classifier,
cached `L_Hand`/`R_Hand` detection, default-right-hand fallback, or per-iteration
nearest-point assignment. This step is agent authoring, not a new trained predictor.

Stage 1's numerical fit has no hand–object contact term. Its Blender refinement
keeps the human frozen and adjusts precontact object placement with a declared short
transition back to the protected later trajectory. Detailed contact definitions and
pose/path repairs belong to Stage 2. Object-motion onset remains a separate
geometric placement/stationarity proxy: a stationary object can still be touched.

## Contact changes over time

Choose temporal granularity from each video's observed interaction, independently
for each hand. Stable contact can retain material vertex IDs while the associated
world positions move with the hand/object. Sliding, rolling contact, progressive
closure, regrasp or support transfer can change the human region, object region,
strengths or activation; the two surfaces need not change together. Inspect the
source to decide which changes, rather than reusing another case's hold schedule.

Refine intervals around observed changes, down to individual source frames when
needed and supported. The current map and pair formats are piecewise constant
within an interval; they do not implement smooth material paths or temporal
strength interpolation. Do not interpolate integer vertex IDs or blend contact
across release/no-contact gaps. Keep occluded transitions unresolved. “Fixed”
describes a schedule frozen during one optimization run, not contact fixed for
the entire video; changes to that schedule between rounds require renewed review.

## Dense contact maps

When the user requests a dense contact map, author **independent native hand and
object vertex fields**, with their own cardinalities and spatial strengths. A
region can contain many hand vertices and fewer original object vertices, or the
reverse. These are plausible material regions, including uncertain alternatives;
they do not assert that every selected vertex touches simultaneously. Do not
turn a map into pairs by zipping IDs, matching list order, creating every possible
pair, or selecting nearest points from the current fitted pose.

`grail.optimization.contact_maps.PhaseContactMap` validates the
`grail.contact_map.v1` schema against source, timeline, native human topology and
original object geometry. The top-level fields are `schema`, `binding`,
`reviewed_by` and `hands`. Each hand covers the whole timeline with zero-based
`[start, end_exclusive)` contact/no-contact/unknown intervals, descriptions and
inspected `evidence_frames`. A contact interval has `regions`; other states have
none. Each region contains:

- `id` and `description`;
- `human_vertex_ids` and matching `human_strength` values;
- `object_vertex_ids` and matching `object_strength` values.

Only each surface's own IDs and strengths have equal lengths. Strength lies in
`[0,1]` and describes authored regional evidence, not calibrated visibility or
per-pair loss weight. Overlapping regions combine by maximum. The current format
holds fields constant within each interval; it does not infer temporal ramps.
Acquisition and stable wrap can use different fields when supported by the
video. Keep unsupported and known-separated intervals distinct, and do not
invent a late region switch because the fitted hand drifts.

The helper's `compact_fields`, `dense_fields` and `export_npz` produce descriptive
per-frame arrays. It does not itself change a pose or generate correspondences:
its export provenance records `optimizer_ready=False` and
`contains_point_correspondences=False`. To optimize these maps, explicitly use
`phase_contact_map` below; its separate objective provenance records the consumed
map and solver settings. The fixed-pair compiler/loss and historical experimental
region adapter remain separate paths. Exporting a map alone does not run Stage 2
or establish a successful grasp.

Choose dense coverage from visible hand roles and model-native anatomy. A curled
cluster does not resolve every hidden digit or justify filling the entire palm.
Inspect original normals and distinguish finger phalanges from metacarpal/palm
influences in skinning-based masks. Keep weaker or unresolved regions identified
as hypotheses. Native topology determines available vertex density; do not
transfer IDs or target counts across different topologies.

Report **native vertex IDs, unique spatial positions, and auxiliary samples
separately**. Seam-split IDs may color different original faces at the same
position. A coarse object region therefore need not contain as many spatial
points as its hand region. The reviewed Drill reference additionally generated
auxiliary points by subdividing selected triangles for sampling, then bound those
points to original face IDs and barycentric coordinates. That procedure did not
replace the original displayed mesh. Its thousands of samples are not thousands
of native object vertices. This native-vertex map helper creates no such samples;
any separate sampler must preserve original geometry and label its output.

## Stage 2 regional objective

`grail.optimization.phase_contact_map_loss.PhaseContactMapLoss` consumes the exact
reviewed map through `LossComputer` and validates the source, timeline and original
geometry binding before optimization. It transforms the bound original object
vertices with the live rigid pose; a simplified tracking mesh never supplies
contact targets. Unknown/no-contact intervals have exactly zero attraction.

Within each region, normalize each surface's authored strengths independently.
Exact coincident object seam IDs share spatial mass. Compute robust gap costs
plus an optional opposing-normal penalty, then soft distances in both directions.
Each direction selects a marginal distribution `q` by minimizing expected cost
plus temperature-scaled `KL(q || prior)`, subject to `sum(q)=1` and
`q_i <= prior_i / partial_mass_fraction`. This cap prevents an arbitrarily small
subset from explaining a broad region while allowing some colored candidates to
remain out of contact. It is a soft regional objective, not authored point pairing.

The fraction is a trial control, not a measured contact fraction. `1` requires
all prior mass; smaller fractions allow more selective support. The two
directions are averaged, then active regions are averaged per hand/frame;
all source frames are averaged and hands summed. Strengths set the within-region
distribution, not each phase's total force. No temporal ramp is inferred.
Gradients reach live body/wrist and object variables allowed by the run's controls.

Example explicit Stage 2 configuration (values require case review):

```yaml
contact:
  mode: phase_contact_map
  prediction_path: /path/to/reviewed_contact_maps.json
  weight: 30000.0
  partial_mass_fraction: 0.25
  pair_temperature_m2: 0.000016
  coverage_temperature_m2: 0.000016
  robust_delta_m: 0.01
  gap_tolerance_m: 0.001
  normal_distance_scale_m: 0.005
  min_opposing_cosine: 0.5
```

Native outputs preserve the map and objective under `meta.phase_contact_map`;
they do not invent `obj_contact_points` correspondences. Diagnostics include
regional gaps, opposing normals, selected marginal mass/effective counts and
concentration. These must be considered with source alignment, whole-original-mesh
intersections and temporal behavior. Normal attraction is not a collision barrier,
and regional loss reduction is not contact or physical validation.

## Existing fixed-pair optimization mode

The following review/compiler/loss contract remains the implemented Stage 2
fixed-pair mode. It requires explicitly authored correspondences and must not be
presented as consuming the independent dense maps above.

## Author the review

1. Inspect source frames/video for each hand. Separate first touch, stable hold,
   support changes, sliding, release and recontact. Refine uncertain transition
   brackets by midpoint inspection to at most six frames, then inspect the
   remaining window. Keep occlusion uncertain when the source cannot resolve it.
2. Partition each hand's complete timeline into `contact`, `no_contact` and
   `unknown` intervals. Intervals must be ordered, contiguous and nonoverlapping
   within a hand; the two hands may have different boundaries. Record actual
   inspected evidence frames and a description of the observed relation.
3. For this pair mode, map the described hand/object surfaces onto the native
   human and original object meshes. Each pair identifies a human vertex and an
   original object vertex (`object_vertex_id`). Review source and oblique views
   before selecting them. One interval may contain several pairs. Choose the actual palmar/finger surfaces
   supported by that phase instead of using a thumb-base proxy for every grasp.
   Inspect representative frames throughout the interval, including its ends.
   Split the interval when the contacting skin patch or object vertices change,
   even if the same handle is still grasped. A stable grasp may retain its pairs;
   do not invent a new pair every frame without evidence. Never select fresh nearest
   points from the current fitted geometry to maintain an apparent contact.
4. Compile the plan below and run Stage 2 against this frozen snapshot. Review
   actual gaps/normals, original-mesh intersections, source/depth evidence and
   motion. If evidence fails, revise contact definitions, palm/body pose or path,
   save a new version and solve again. An inner solver cap does not end the edit loop.

`unknown` and `no_contact` have empty pair lists and no attraction. They remain
distinct in saved provenance. Unknown does not establish absence of contact or a
successful reconstruction. The schema validates records; it cannot prove that
the source was inspected or that an authored point is visually correct.

## Review format

`grail.contact_review.v2` JSON has exactly `schema`, `binding`, `reviewed_by`, and
`hands`. Copy the entire `fixed_contact_binding.json` object into `binding`.
The binding identifies the source video SHA256, sequence, fps/frame count, native
human model/topology and original scaled object geometry. Do not substitute the
remeshed object used by tracking.

Each hand has intervals like this **illustrative** left-hand example. IDs and
timing must be authored for the actual case; this is not a case prediction.

```json
{
  "id": "left_support",
  "frames": [33, 70],
  "state": "contact",
  "description": "Left palm supports the lower side of the object.",
  "evidence_frames": [33, 48, 69],
  "contacts": [{
    "id": "palm_support",
    "human_vertex_id": 123,
    "object_vertex_id": 456,
    "confidence": 0.9,
    "hand_surface": "Left palm below the finger bases",
    "object_surface": "Original mesh, lower side support patch"
  }]
}
```

All intervals use **zero-based `[start, end_exclusive)` source frames**. The
example covers frames 33 through 69. Add the earlier/later intervals to cover
the full video and supply `hands.left_hand` and `hands.right_hand`. Every interval,
including unknown/no-contact, needs a description and inspected evidence frames.
For unknown/no-contact set `contacts: []`. For contact, at least one pair must
have positive confidence; this field is a fixed relative loss weight in `[0,1]`.
When patch normalization is used, preserve source confidence separately from the
derived pair weight as described below.
Pair `body_part` and `active_frames` are derived from the containing hand/interval,
so there is no second independently editable activation schedule.

## Contact patches and sampling density

In fixed-pair mode, the single-pair example above illustrates the record format, not sufficient
coverage for a wrapping grasp. Author small patches on each source-supported
finger pad, opposing thumb and, when evidenced, palm support. Each patch contains
multiple explicit native human / original object vertex pairs. The patch set can
grow from acquisition to full wrap and change during observed regrasp or sliding.
Do not force all fingers or the whole palm into contact by default.

Inspect the selected points on the actual surfaces, including anatomical
membership, normal direction and spread across each patch. Keep exact hidden
correspondences labeled as authored hypotheses. Count distinct physical object
positions as well as vertex IDs: split OBJ seam vertices can have different IDs
and normals at the same position. A patch with many pairs collapsing to one target
does not cover a surface. Preserve original topology; report coarse target
resolution instead of adding a remesh, triangle centroids or fresh nearest-point
assignments inside the solve.

The loss is a confidence-weighted mean of Huber distances over all active
pair-frame records. Giving every
new pair the same weight would increase that patch's influence relative to other
patches, hands and phases. `grail.optimization.contact_patches.weight_contact_patches`
flattens already-authored patches into the existing v2 pair format, with a declared
per-hand/phase total distributed first across patches, then across their samples.
This budget applies to each active frame; an interval with twice as many active
frames still contributes twice the weight mass. It is not duration-normalized.
Save the source uncertainty and the weighting recipe alongside the review; these
effective pair weights are not calibrated visibility probabilities. The helper
does not choose correspondences, detect contact or change the loss definition.

For a patch `p` and sample `k`, the effective pair weight is
`hand_confidence * patch_weight[p]/sum(patch_weights) *
surface_weight[k]/sum(surface_weights_in_p)`.
Uniform within-patch sampling can use the default surface weight of one; when
samples represent unequal surface areas, author those areas explicitly. The
helper accepts optional native vertex arrays to report exact coincident-position
aliases. Save its returned diagnostics with the compiled review. For example,
an opposition grasp can give the thumb family half of the total and distribute
the other half across its supported long-finger patches; changing that split is
an explicit contact-model change, independent of increasing sample density.

Compare dense and sparse proposals using the same per-hand/phase weight totals,
and inspect physical patch coverage and worst local gaps in addition to the
weighted mean. Check pairwise separation compatibility with frozen fingers and
review mounted poses and original-mesh intersections. More samples alone do not
establish a feasible grasp or prevent a hand from intersecting the object.

## Compile and configure

When `optimization.export_contact_review_inputs: true` (the manipulation default)
or fixed contact is configured, `HOIOptimizer.init_data` exports
`fixed_contact_binding.json` and `fixed_contact_geometry.npz` under its log
directory. These contain provenance and geometry, **no invented points**.
Human IDs use native human topology (including SOMA); object IDs use the original
ordered vertices already scaled into metres. Original faces are retained for
geometry binding and vertex-normal review, not substituted for selected vertices.

```bash
python -m scripts.standalone.pack_fixed_contacts \
  --binding RUN/fixed_contact_binding.json \
  --geometry RUN/fixed_contact_geometry.npz \
  --review CASE/contact_review.json \
  --output CACHE/fixed_contact_points/DATASET/OBJECT/VIDEO.json
```

The compiler validates the review, expands interval pairs, checks original
geometry and refuses to overwrite an existing version. Its output is a
`grail.fixed_contact.v2` manifest with an embedded `video_review`. The low-level
pair reader preserves explicitly versioned historical v1 face/barycentric files
for reproducing earlier runs; it never rounds their anchors to mesh vertices.
New reviews use v2 vertex pairs. Mixed vertex/face fields and mismatched
review/manifest versions are rejected. Stage 2 and evaluation
**require** the embedded review and verify that compiled pairs match it exactly.
A binding mismatch, malformed interval, missing review or missing file fails
before any Stage 2 optimizer update. There is no detection/proximity fallback.

Explicit fixed-pair configs use:

```yaml
contact:
  weight: 30000.0
  mode: fixed_prediction
  prediction_path: "{cache_dir}/fixed_contact_points/{sequence}.json"
  robust_delta_m: 0.01
```

Stage 2 and evaluation must use the same manifest. To run Stage 1 alone, select
only `init_opt` and remove `eval.contact`, while keeping
`optimization.export_contact_review_inputs: true` to produce the binding/geometry
for subsequent authoring. A fixed-contact evaluation also needs the review. The old VLM cache configuration, label-dependent nearest/depth/center
losses and contact-distribution smoothing are removed. Old custom configs must
migrate explicitly. Static terrain configs retain their foot-contact objectives
and no longer include the unused VLM hand-contact evaluation. Separate optional
dense-region experiments are retained; they do not consume the independent
`grail.contact_map.v1` maps and are not this fixed-pair solver path.

## Loss and provenance

Each active frame selects its hand-specific reviewed interval and that interval's
list of native human vertex / original object vertex pairs. Different intervals
can use different hand IDs, object IDs and pair counts. `fixed` means the authored
per-frame schedule is immutable during optimization; it does not mean one pair
is reused across the whole video.
The target follows the current object rotation/translation, so it is fixed in
material coordinates, not world space. The confidence-weighted isotropic Huber
distance is `0.5*d*d` below `robust_delta_m`, otherwise
`delta*(d - 0.5*delta)`. Gradients reach live human and object variables; references
are detached. A fully reviewed no-contact/unknown sequence has differentiable
zero contact loss, with its phase counts preserved rather than fabricated pairs.

Source/model/timeline/geometry hashes bind the consumed snapshot; array hashes
include shape and ordered canonical payloads (little-endian int64 faces,
float32 vertices). Review content and its canonical JSON hash are saved with the
pair-file hash, per-hand phase counts and exported fixed points. Postprocessing
preserves these points and intervals in `obj_contact_points` and
`meta.fixed_contact_prediction`; they are intended targets, not measured touches.
Historical results without that metadata retain their old proximity export.
Before solving, render the selected vertices on both meshes at representative
frames of every interval, alongside the source view and an oblique view. Label
exact hidden correspondences as hypotheses. A vertex-only object mesh can be
coarse: retain its real vertex IDs and report poor spatial resolution rather
than silently replacing a vertex with a triangle centroid. Pair selection does
not authorize unlocking frozen human fingers. If a phase's chosen multi-point
contacts cannot be achieved with the permitted native controls, record that
limitation and revise the supported contact/palm assumptions.

Neither the point loss nor the compiler validates normals, collision-free motion
or physical grasp stability; those still require actual result review.

### Optional temporal stabilization for regional contact

The **preferred agentic Stage 2 workflow** is the regional objective plus the
temporal controls below, as configured in
[`manip_soma.yaml`](../configs/recon_4dhoi/manip_soma.yaml).
The low-level losses remain opt-in for explicit fixed-pair or other callers.
Bind every map to the original SOMA topology.

For each video, save the reviewed map at
`{cache_dir}/contact_maps/{sequence}.json` and its temporal scope at
`{cache_dir}/contact_maps/{sequence}.temporal.json`. The scope is loaded before
Stage 2 updates; Stage 1 does not require it. Missing or stale scope fails with
an actionable error rather than inventing stable holds from object motion.
The standard temporal config inherits the same group's enabled `contact`
configuration. Explicit nested contact settings must agree, apart from weight
and evaluation threshold. Paths are resolved by the normal optimizer.

The sidecar has exactly these fields:

```json
{
  "schema": "grail.contact_temporal_scope.v1",
  "contact_map_sha256": "<SHA-256 of the exact reviewed map JSON bytes>",
  "reviewed_by": "<source inspection and reviewer record>",
  "options": {
    "fps": 24.0,
    "hold_interval_ids": {"left": ["reviewed_hold_id"], "right": []},
    "windows": [
      {"id": "reviewed_join", "frames": [18, 46], "ramp_frames": 6,
       "regions": ["left/reviewed_hold_id/reviewed_region_id"], "object": true,
       "velocity_weight": 0.25, "acceleration_weight": 1.0},
      {"id": "weak_background_acceleration", "frames": [0, 121], "ramp_frames": 0,
       "regions": ["left/reviewed_hold_id/reviewed_region_id"], "object": true,
       "velocity_weight": 0.0, "acceleration_weight": 0.1}
    ]
  }
}
```

Replace all shown FPS, frame ranges and IDs with the inspected source timeline;
these numbers illustrate the fire-extinguisher trial and are not universal
annotations. `sha256sum <map.json>` provides the byte hash. Only FPS, explicit
hold IDs and windows live in this sidecar. Scalar tolerances/weights stay in the
run YAML. To disable no-slip for sliding or unresolved contact, use empty hold
lists; motion windows can still apply to explicitly chosen regions. Do not
declare a stable hold simply to satisfy a configuration field.

This profile selects contact and temporal **objectives**, not a native edit mask
or a Stage 1 checkpoint loader. Keep the case's declared native controls and
stationary prefix in its runner/adapter. In particular, the reference trial
enabled only SOMA wrist slots 0/25 within the hand residual block and hard-froze
the object's frames 0–27 to its selected Stage 1 parent. Those case controls are
not implied by the reusable YAML and must not be copied to a different model or
video. Do not enable the entire hand residual tensor to emulate wrist editing.

The optimizer preserves the native source-rate result (`output_smoothing: false`)
and refuses post-fit filtering or timeline truncation while these bound temporal
controls are enabled. Native output metadata includes the original map,
effective phase weights, exact scope and hash, resolved options, and component
weights. The sidecar is frozen once loaded for a solve. Version it between
review/edit rounds when source evidence changes.

The regional objective supports `activation_ramp_frames` and `phase_blend_frames`,
nonnegative integer controls defaulting to zero. Onset activation rises with a
quintic curve inside a contiguous known-contact run. Adjacent contact phases may
crossfade their region-mean costs over a short window, using each phase's own
native vertices and strengths. Unknown/release frames never attract contact.
This blends costs, not integer vertex IDs. The authored JSON remains unchanged;
`runtime_schedule` records effective frames and weights in the native metadata.
An onset ramp of one gives full attraction at the authored first contact. Inspect
acquisition gaps when weakening first-touch attraction.

`phase_contact_temporal` is a separate term in Stage 2 `loss_cfg`. Normal callers
use `scope_path` as above. Direct programmatic callers may instead provide the
complete inline options and matching contact configuration below. Do not mix
inline FPS/hold IDs/windows with a sidecar. Example settings below reproduce the
fire-extinguisher temporal trial; its frame IDs are not defaults for other videos:

```yaml
contact: &regional_contact
  mode: phase_contact_map
  prediction_path: /absolute/path/to/reviewed_contact_maps.json
  weight: 30000.0
  partial_mass_fraction: 0.25
  activation_ramp_frames: 1
  phase_blend_frames: 4
phase_contact_temporal:
  weight: 1.0
  contact: *regional_contact
  component_weights:
    hold_relative_velocity: 0.1
    window_velocity: 0.4
    window_acceleration: 0.4
  options:
    fps: 24.0
    hold_interval_ids:
      left: [left_lower_bar_hold]
      right: [right_connector_hold]
    hold_ramp_frames: 4
    hold_speed_tolerance_mps: 0.02
    hold_speed_scale_mps: 0.05
    window_speed_tolerance_mps: 0.35
    window_speed_scale_mps: 0.35
    window_acceleration_tolerance_mps2: 3.0
    window_acceleration_scale_mps2: 3.0
    windows:
      - id: contact_join
        frames: [18, 46]  # Half-open original source-frame interval.
        ramp_frames: 6
        velocity_weight: 0.25
        acceleration_weight: 1.0
        regions:
          - left/left_lower_bar_hold/left_hold
          - right/right_connector_hold/right_hold
        object: true
      - id: weak_background_acceleration
        frames: [0, 121]
        ramp_frames: 0
        velocity_weight: 0.0
        acceleration_weight: 0.1
        regions:
          - left/left_lower_bar_hold/left_hold
          - right/right_connector_hold/right_hold
        object: true
```

Windows optionally accept `ramp_frames`, `velocity_weight` and
`acceleration_weight`. Defaults are 0, 1 and 1. A quintic frame envelope is zero
at each end of a tapered window; edge/triple weights use the minimum of their
constituent frames. Reduction remains the mean over all source differences and
configured windows, without dividing by nonzero envelope mass or coefficient
sum. The example's two-window mean gives effective coefficients 0.05 for local
speed, 0.2 for local acceleration and 0.02 for background acceleration. The weak
background term discourages spikes just outside the local window. It does not
add contact attraction or a global speed cap.

Stable holds are selected explicitly from source evidence. Fixed authored hand
material points are transformed into the live object frame, `(H - t) @ R` in
row-vector convention, before computing velocity. Rigid co-motion cancels.
The soft squared-hinge penalty allows a declared speed tolerance; it does not
pin every colored candidate vertex to an object vertex. It has no inferred
correspondences or per-frame posterior selection. Only differences wholly inside
each named hold interval participate. Changing support/sliding should use a
revised supported scope, rather than globally enforcing a rigid hold.

Explicit local windows penalize excess speed and acceleration of fixed native
hand points and original object surface vertices, with triangle-area lumped
weights for the object. These motion terms can span approach/contact boundaries
without introducing precontact attraction. Derivatives use source FPS; each
squared hinge is divided by its physical-unit scale squared before the separate
component weight. Native metadata records scopes and settings; the executed
configuration records component weights. Human fingers remain frozen unless
separately authorized; this may create a tradeoff with a strict material hold.

Review both edges of every window, early contact gaps, late hand spikes, object
rotation/surface motion, source pose, feet and original-mesh intersections. A
world-motion cap can suppress valid source motion or move a spike just outside a
window. Lower average temporal loss alone cannot establish an improvement. Use
the preferred temporal configuration with each case's reviewed scope, and tune
its tolerances and fade lengths against source motion and contact evidence.
These terms provide no collision barrier or physical validation and do not
replace native Stage 2 key editing. A solve cap ends that solve: failed contact,
palm pose or approach review returns to the relevant edit, within the requested
scope. Preserve rejected candidates and report temporal and grasp quality
separately.
