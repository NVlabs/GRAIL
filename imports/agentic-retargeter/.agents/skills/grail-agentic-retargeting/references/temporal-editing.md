# Native smoothing and hand-closure timing

Use when a selected robot motion jitters, snaps at a join or closes too quickly, and when reviewing closure in a newly propagated sequence. Read the selected checkpoint and its decision first; preserve the fallback and inherited limitations. This is a robot edit, not a reason to rerun human reconstruction or depth calibration.

## Identify the temporal defect

| Observation | Useful measurements | Initial edit scope |
|---|---|---|
| Fingers close too quickly, even after smoothing | Per-hand onset/completion, closure excursion duration, peak finger angular speed | Finger timing during preshape/closure; protect the selected grasp and hold |
| Arm or hand jitters during motion | Joint and FK wrist/pad acceleration and jerk, with worst intervals | Local native controls around the affected interval, with contact/source constraints |
| A pop appears at a phase boundary | Native joint steps, speed changes and actual interpolated geometry on both sides | Join controls and adjacent intervals, preserving phase/contact milestones |

Keep original frame count, fps and object-event timing unless the user requests retiming them. Slowing video playback does not repair native motion. Do not infer that lower jerk RMS means closure is slow enough, or that lower finger speed improves every derivative.

Before editing, record the affected hands/joints, intervals, open/preshape and grasp keys, clearance reversals, active contact/support phases, frozen fields and comparison criteria. Derive native joint addresses from the actual model. Use source observations and the user's requested timing; mark authored choices when small or occluded source fingers cannot establish exact timing. Do not require another approval for an already authorized local correction.

## Smooth during SOMA initialization

For a requested SOMA initialization with jitter control, configure the adapter's
explicit [temporal optimization option](../../../../tools/soma_retargeter/README.md#optional-temporal-optimization-inside-the-soma-converter).
It uses a fresh framewise warm start, then alternates sequence sweeps through
the same Newton human-reference, material-contact and collision objectives.
Velocity, acceleration and jerk costs use the source timestep; they do not
replace human targets with a prior robot pose. Keep its sweep budget within
the case budget and retain time for export and review.

Use the saved pre-sweep state from that same execution for an exact A/B
comparison. Repeated local IK runs can choose different finger branches even
with nearly identical inputs. Preserve stage provenance when converting the
Newton-coordinate snapshot through the official CSV/native export; do not
label the snapshot as an independently completed run.

The current option regularizes scalar hinges, not the free root. Measure
world-space wrists, root motion, feet and exact mapped human-target error in
addition to joint derivatives. It is a soft finite-sweep solve: contact-onset
spikes may still need phase/preshape edits and another reviewed solve. Retain
the hard gates below and independent original-mesh review.

For authorized object motion in this initialization, pair `object_optimization`
with the temporal option and follow its
[object-motion contract](../../../../tools/soma_retargeter/README.md#optional-object-motion-inside-the-soma-converter).
This experimental route alternates shared object updates and robot updates
inside the converter while preserving original human targets. It does not add
an object articulation to the Newton robot model. Hold one supported object
pose constant through `first_edit_frame - 1`, verify both that last frozen frame
and the earlier declared `contact_frame` with actual material gaps before
allowing movement and after every sweep, and preserve pickup timing. A soft
contact ramp or a frozen copy of a drifting reference is insufficient.

Bind total translation/rotation corrections to the immutable original path
across every sweep using positive position/orientation reference costs. Hard
displacement and correction-rate limits are optional; audit any explicitly
configured ones. Without them, report actual deviations for quality review,
since soft costs do not guarantee a bound. Check actual object
translation and orientation motion; smoothing only the offsets can preserve a
noisy original trajectory. Rotation-matrix derivative costs are a smoothness
surrogate, not SO(3) angular acceleration/jerk bounds. Review those physical
derivatives independently along with world-space wrists, shared-hand material
gaps, object/table support and original-mesh crossings. Consume the bound
optimized object output during native export, preserving its original sidecar
identity and scene; exporting the old path would invalidate the solved contacts.

## Require a native joint-speed hard gate

Every new full-sequence refinement config must explicitly set
`temporal.max_velocity_rad_s`, with optional per-joint overrides in
`temporal.joint_velocity_limits_rad_s`. Use the
[G1 + Dex3 animation preset](../../../../tools/repair/examples/g1_dex3_temporal.json)
as the initial policy: 3 rad/s by default (including fingers and legs), 2 rad/s
for named shoulder/elbow/wrist hinges, and 1.5 rad/s for named waist hinges.
These are authored animation acceptance thresholds, not manufacturer velocity
ratings. Declare any revision and its reason before comparing candidates; do
not raise a threshold merely to pass an existing jump.

Evaluate `abs(q[t+1] - q[t]) / (time[t+1] - time[t])` on every native hinge and
every consecutive native frame, using the actual timestamps or FPS. At 24 fps,
the three initial thresholds correspond to 7.16, 4.77 and 3.58 degrees per frame,
respectively. Native joint identities come from the bound model. Include both
hands, arms, waist and legs, even when they are frozen or outside the current
repair window. An untouched hold or inherited branch jump can still fail the
full-motion gate. Record the failing joint, frame pair, speed and applicable
limit; a low wrist displacement or sequence-average derivative cannot hide it.

A failed gate rejects full-sequence acceptance, even when the solve converges
or contact and collision metrics improve. Preserve a useful local result as a
clearly scoped partial; repair the indicated interval and both boundary stencils
within the declared edit scope. Do not splice a protected pose back afterward
without rerunning the gate. If the necessary join edit conflicts with a protected
hold, revise the scope explicitly or retain the failure.

This gate measures consecutive native samples; it does not establish smooth
acceleration, jerk, collision-free interpolation or actuator feasibility. Keep
the actual Blender/native subframe checks and acceleration/jerk diagnostics.
Changing to Bézier playback cannot repair a large jump already present between
the exported integer poses.

## Compare sampled acceleration and jerk explicitly

For the transition solver's physical-time objective, set
`solver.derivative_units: "seconds"` and declare weights in those units; legacy
`"frames"` weights are not interchangeable. Joint velocity, acceleration and
jerk use rad/s, rad/s² and rad/s³; FK wrist derivatives use metres and seconds.
The optional `joint_jerk` and `wrist_jerk` weights add objectives, not acceptance
gates. Bind the parent, phase keys, editable mask, source targets and correction
corridors before comparing methods. See the [solver options](../../../../tools/repair/TRANSITIONS.md#sampled-derivative-and-spline-options).

`solver.spline_initializer` optionally fits a weighted natural cubic smoothing
approximation before coupled refinement. It can reduce noisy sample values
while keeping frozen milestones exact; phase fidelity weights do not enforce
material contact. Auto Clamped or Continuous Acceleration handles through every
original key cannot change those same sampled key values. The internal spline
is C2, but the final native **LINEAR replay is not C2**. Review its exported
samples and actual interpolation, not only the authored curve.

Optional `solver.temporal_projection` constrains sampled speed, acceleration
and jerk within the declared native bounds and frozen mask. It requires an
explicit speed policy plus explicit acceleration/jerk limits; choose these for
the case, with documented per-hinge overrides where appropriate. Do not infer
universal caps or a pass from lower loss. Evaluate all native hinges over the
full timeline, including frozen legs, held phases and every changed join.
Inherited frozen violations remain full-motion failures even if editable hand
motion improves. Report peak and RMS joint/FK derivatives and the failing
stencils separately from contact, original-mesh collision and fixed source/depth
checks. Projection or spline success does not replace those independent checks
or source-speed visual review.

## Choose wrist guidance per hand and phase

Distinguish three sources of end-effector objectives before changing weights:

- **Human-reference tracking:** SOMA's original body/end-effector objectives, or a declared soft posture prior from its native output. These remain source guidance; using native joint coordinates as a prior is not a new SOMA solve or exact human wrist tracking.
- **Hand/object contact:** material relations and shared object-relative palm/wrist pose, active only in source-supported contact phases. Normal alignment remains disabled under the current policy.
- **Previous robot wrist preservation:** an extra world-position/orientation target from an earlier robot candidate. Do not enable this by default in free approach merely because that candidate initializes the solve.

| Phase | Default guidance |
|---|---|
| Free approach, before that hand's contact | Soft human-derived posture guidance, native limits, scene/self clearance and coupled joint/FK smoothness. No future-contact attraction or extra old wrist pose preservation. |
| Acquisition / closure | Approach the reviewed grasp endpoint through a coupled short window. Activate each contact at its declared onset; an endpoint does not constrain the whole preceding path. |
| Supported hold | Preserve the declared source-supported or authorized authored object-relative grasp and material contacts. Jointly review shared torso/object changes and both hands. |
| Release | Deactivate only the relations that actually end, then return to free-motion guidance. The other hand may still hold. |

Author an interior wrist waypoint or orientation target only for a specific
source-supported pose or necessary clearance route, with its frame/window and
reason recorded. Uncertain contact timing remains a hypothesis. Do not infer
release from a missing detection or treat both hands as having the same phase.

For a free-approach trial, remove old wrist preservation from both seed fitting
and refinement. `seed.native_input: true` skips waypoint/wrist IK seeding;
`solver.weights.route: 0` removes the later wrist-position penalty. These are
separate settings. See the
[transition implementation boundary](../../../../tools/repair/TRANSITIONS.md#free-approach-and-wrist-preservation).
Bind the source posture prior separately from the selected iteration seed;
revise any prior explicitly rather than silently replacing the fixed reference.

For authorized correction of inconsistent generated hands, apply the
[generated-source policy](grasp-editing.md#correct-inconsistent-generated-video-grasps).
The held robot hand need not track the source's framewise finger/wrist changes.
Revise conflicting source or old-wrist guidance for the affected hand/arm and
fit its acquisition/hold joins together; retain the common grasp target and
independent contact, geometry and derivative checks. Record actual solver
settings, since a softer source prior alone does not enforce no-slip holding.

Optimize neighboring native states together, including the derivative stencils
across the frozen hold boundary. Preserve unrelated variables and reviewed holds
at the declared scope. If the boundary remains infeasible, report it or declare a
local scope revision; do not silently move a protected key, widen object motion
or replace source guidance with an old wrist lock. Recheck changed interpolation
support and joins, reusing only unchanged evidence under the existing budget.

## Choose the tool for the affected phase

For a new hold with slip or acquisition/release jumps, use
[`tools/grasp/sequence.py`](../../../../tools/grasp/README.md#coupled-hold-and-acquisitionrelease-repair)
after common mounted-candidate evaluation. Fit neighboring arm/finger states
through the complete hold and its buffers, keeping material targets fixed and
the first hold sample editable. Declare the seed review even when it failed
mounting or temporal checks; a seed is initialization, not an accepted motion.
The original parent remains the correction/posture reference. Endpoint and
lineage requirements are documented with the tool; do not silently restore a
seed key after solving. Inspect `material_review.json` for full-hold gap/slip,
then independently re-audit the exact exported file and changed joins.

`projection_scope: "editable_hinges"` can keep an inherited frozen-leg failure
from aborting the proposal. It does not remove that failure from full-motion
acceptance. Preserve a reviewed local improvement through the
[progress ledger](../../../../tools/repair/README.md#preserve-reviewed-progress),
with inherited failures and unknown checks still visible.

For a localized scene collision, follow the
[intermediate clearance-key loop](scene-aware-repair.md#repair-intersections-with-intermediate-pose-keys).
It checks temporal reachability before fixing the key, then returns residual
intersections between keys to pose/path editing. Use original-mesh subframe
checks and wrist smoothness comparisons; a clear key or a passing joint gate
alone does not establish a clear, natural transition.

Use Blender Bézier curves with Auto Clamped handles to author timing where appropriate, while preserving meaningful keys and reviewing the native export. Keep the kit's `auto_smoothing='NONE'` default; compare optional `CONT_ACCEL` on identical keys when useful. Neither setting understands contact or original-mesh clearance. Do not add sparse keys merely to make the curve editor's smoothing setting have an effect: removing source keys is itself a motion edit.

For a stable contact/support interval whose coordinated hand/arm motion needs adjustment, use the [contact-preserving interpolation route](contact-preserving-interpolation.md). It separates explicit contact inequalities from temporal objectives and external geometry/source review. Keep closure-only edits local when the selected grasp and hold already satisfy the request; do not open arm variables through a broader solver incidentally. A wrong grasp needs descriptor, palm or auxiliary-path editing before preserving it more precisely. Source-supported sliding or changing contact roles requires its own phase model, not an unchanged rigid-hold descriptor.

## Preserve the useful grasp while editing timing

For a closure-only request, normally keep arms/body/root, object motion, camera and all unrelated native fields exact. Keep the reviewed grasp key and subsequent motion exact unless changing them is part of the request. Preserve the per-hand anatomical roles, fixed material anchors and normals; evaluate contact attraction only in its active phase. A previously selected partial remains partial.

For broader smoothing, preserve material relations and object-relative wrist orientation during their active hold phases, with source pose guidance and declared tolerances. During free motion, follow the phase guidance above rather than preserving the old robot wrist pose. Protect support changes and important grasp keys. If a whole-sequence trial damages a later supporting thumb or changes a contact role, narrow the interval or native variables; do not simply filter harder. Joint-space smoothness alone cannot establish world-space hand smoothness when the object path is noisy.

To lengthen closure while finishing at the same grasp key, consider earlier preshape while the hand is clear. Inspect each hand separately. The opposing thumb may need a turn or return while long digits curl; preserve such nonmonotonic clearance motion rather than forcing every joint onto one progress curve.

A bounded monotone progress curve between open and closed joint values can initialize a trial. A quintic ease or shape-preserving piecewise curve is an option, not a default clearance guarantee. If earlier curl intersects the object, split preshape from final closure and adjust progress per hand/digit group. Do not move the arm/object path implicitly to rescue a fingers-only edit. If no schedule works within the declared bounds, retain the failure and identify the additional scope needed instead of weakening the checks.

Author actual native joints and Blender controls. Verify the exported interpolation: sampling a smooth curve into integer hinge keys and replaying them linearly does not preserve the curve's continuous derivatives. Avoid unintended overshoot or a switch of joint branch; check original limits on the exported motion and its evaluated subframes.

If individually reviewed grasp keys produce branch switches or missed contact between them, solve neighboring native states together with their boundary positions and velocities. Copying a key back after fitting can recreate the jump. When a reviewed key is allowed to move, declare that change, retain its contact/source constraints and re-review the resulting pose. An obstructed approach may be diagnosed separately from the hold, but their final joined motion still needs review. Starting backward from a clear mounted grasp can help identify which finger opening and wrist withdrawal are actually feasible.

If an inactive digit cannot follow a continuous clear path, inspect the actual wrist and active-finger motion across the same interval before widening that digit's search. A wrist branch jump can make isolated finger repair ineffective. Distinguish a search heuristic that pruned a path from a native limit or a measured collision. Removing a heuristic permits a diagnostic trial; it does not establish acceptable finger speed. A small declared wrist corridor or coupled contact-preserving transition may avoid a fast finger flick, but recheck every digit so clearance is not merely transferred into a new active-pad collision.

## Review joins, clearance and actual closure speed

Evaluate the changed interval plus adjacent interpolation intervals. Any pinned/restored pose or splice requires another join check: restoring a clear integer key can introduce a subframe crossing or speed jump next to it. Refine sampling near contact, clearance reversals, joins and detected worst cases, using actual native FK and the original meshes. There is no universal frame spacing.

Use the fixed baseline at matching times. In addition to the [ordinary robot review](review-delivery.md):

- Report each hand's peak finger speed, the responsible joint/interval, and closure duration in seconds. For a meaningful monotonic closing excursion, measure 10–90% progress using the actual exported interpolation and identify the representative joint. Split reversals into phases; a tiny or nonmonotonic excursion is not a reliable scalar closure clock. Keep metric definitions and comparison windows fixed across candidates, and do not generalize one joint's duration to every digit.
- Report RMS and peak acceleration/jerk separately, with their timestep, differentiation method and worst intervals. A derivative of sampled joint values is not proof of smooth continuous-time derivatives. Check FK wrist/pad motion where relevant and disclose any local regression hidden by an aggregate gain.
- Recheck complete relevant original geometry, including thumb/palm self-intersections and all object components. Compare collision pairs at the same time, not only total counts or the set of pairs across the sequence. Even without new pairs, inherited intersections can worsen; investigate changed overlap geometry and label what any local severity measure actually represents. No undefined solid occupancy for an open pan.
- Recheck fixed source/depth targets for changed states, including common rays, missing coverage and actual hit links. Earlier preshape can change source hand silhouettes or replace a hand-ray hit with torso/elbow. Report that tradeoff separately from the intended timing benefit.

Reuse prior checks only for exactly unchanged native/interpolated states with matching object transforms, model/mesh and target bindings and the same evaluation definitions. Exact array equality at a held suffix does not cover its changed incoming join. Recompute derivatives whose stencils touch changed frames. Record reused versus recomputed coverage.

Provide a full-duration source-speed comparison and useful diagnostic views. Add a same-speed hand close-up for a closure edit, using matched cameras/crops so motion is comparable. Slow-motion supplements must be labeled. Inspect source-speed playback when available; if review only used ordered frames, say so. Keep native numerical evidence and visual preference separate, and disclose remaining contact, source and collision issues when selecting a scoped partial. Deliver editable Blender with the existing native round-trip checks when requested.
