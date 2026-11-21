# Grasp representation and animation editing

## Group source keys by grasp continuity

Make this decision during source-key selection, before fitting independent robot
poses. For each hand, compare the contacted object part, palm/opposition direction,
finger arrangement and evidence of release, sliding or regrasp. Inspect the
intervening video too: similar endpoints can hide a release and reacquisition.
Object rotation or a changed camera projection alone does not imply a changed
grasp. Occluded fingers and detector gaps remain uncertain; do not turn missing
evidence into either a confirmed rigid hold or a new grasp.

Record a per-hand grasp identity, source interval/key indices, evidence frames,
relation type (rigid hold, articulation, slide, regrasp or unknown), uncertainty,
and the chosen robot descriptor version. Separate left/right identities even
when both hands hold the same object. These are host-authored planning records,
not new fields automatically consumed by the video or solver CLI.

For a confirmed rigid hold, all of that group's keys share one robot grasp:

`G_hand = inverse(T_world_object(t)) @ T_world_palm(t)`

Bind the actual native palm/wrist link and transform convention. Keep `G_hand`,
the native finger shape and original material-contact identities common across
the hold. The robot grasp must suit its own anatomy; it need not copy the human
finger angles. If the video supports finger articulation while the palm remains
fixed, declare the allowed digit motion and preserve the relevant material
contacts instead of freezing the entire hand.

Choose `G_hand` by testing a small bank of source-compatible grasps against all
representative difficult keys and short transitions in the group. Do not assume
the first key's robot pose is the best shared grasp. A shared grasp can be revised
as one group-level variable; arm joints and elbow guidance vary per frame to
realize it. Retain original geometry and joint limits, and apply the declared
source policy when comparing candidates, including the generated-video correction
scope below where authorized. Reject a candidate that requires forbidden intersections;
do not let collision correction silently move the hand to a new object-relative
pose at one difficult key. Revise the shared grasp/arm branch, or use an already
authorized bounded body/object adjustment. A failed bounded fit does not prove
the entire group infeasible or establish a source regrasp.

When object-trajectory edits are authorized, use the same adjusted object pose
for both hands: `T_world_palm(t) = T_world_object_adjusted(t) @ G_hand`. Keep the
grasp relation fixed while adjusting the shared object path, penalize deviation
from the original trajectory, and check offset velocity/acceleration and
phase joins. A small displacement cap alone does not prevent abrupt motion.

Declare the editable interval, protected stationary/onset frames, positive
position/orientation reference costs and actual-trajectory smoothness terms
before the trial. Hard displacement and correction-rate limits are optional in
the SOMA object mode; use them only when explicitly configured. Use the original
unadjusted path as the reference across revisions, not
the last offset result. Express increment limits with the actual timestep;
copying a per-frame limit to another FPS changes the allowed motion. Any ramp
into/out of object adjustment belongs in the join checks. Existing permission
for object edits remains valid, but a new case without that scope keeps
the object fixed. Historical Drill corridors and per-frame increments were
trial settings, not universal defaults or a requirement for the current
soft-reference policy.

Object fidelity is a separate acceptance condition: keep changes small relative
to the reference spatial path, movement direction and key orientations, not just
the action label. Original position/orientation costs must remain active at
matching source timestamps; they do not guarantee bounded errors. Review peak/RMS errors
and worst frames over the full changed interval and its interpolated joins;
average error cannot hide an excessive excursion. Preserve pickup, resting,
release and task-orientation milestones; do not retime the path to make errors
look smaller. If contact, body posture and object fidelity cannot all be
retained, revise the grasp/arm branch or retain a partial result. Keep any
explicit hard limits in force; do not silently widen them after a failure.

The existing hold tool bounds offsets relative to its supplied native input.
After an authorized reparenting, those local bounds alone do not constrain total
drift from the original reference. Retain the original path/hash and separately
check the final composed path against it on every round, including any global
robot/object placement change. Report placement and time-varying corrections
separately as well as their total; neither rebasing nor display alignment changes
the fidelity reference or any configured deviation budget.

If the object must be stationary before effective contact, specify one supported
reference pose and a protected interval with constant translation and rotation.
Do not equate activation of a soft grasp-attraction ramp with an established
grasp, or merely retain time-varying upstream poses outside an edit window.
Verify both the declared effective-contact frame and the last frozen frame using
the actual mounted material gaps before allowing movement, and recheck after
each sweep. Keep a later source-supported rest interval protected
when contact occurs before pickup. The [SOMA converter option](../../../../tools/soma_retargeter/README.md#optional-object-motion-inside-the-soma-converter)
supports an explicit stationary interval and shared object updates
within its temporal sweeps; it remains experimental and requires independent
support, geometry, source and temporal review.

Carry the relation through every active hold sample and native interpolation,
not just the selected endpoints. Check actual FK palm translation/rotation in
object coordinates against the group's shared transform, plus physical pad
gaps, original-mesh collisions and finger/arm motion. Record units and
explicit allowed residuals; neither a soft loss nor equal endpoint poses proves
the relation was maintained. Source-supported relation changes, or explicitly
authored changes under the correction policy below, create a new phase;
release/reacquisition is a boundary even if the later grasp looks the
same. Reuse the earlier pose as a template when useful, without constraining the
free interval as a hold.

## Correct inconsistent generated-video grasps

When correcting generated footage with the user's authorization, use the source
to identify the intended action, object region, hand roles and event timing.
Finger morphing, apparent sliding and changing opposition may be generation
artifacts. Do not automatically reproduce them or interpret them as a required
regrasp. Record the observed uncertainty and the authored robot interpretation
separately; a chosen stable hold is not proof that the source contained one.
This exception does not change source-fidelity defaults for real human recordings
or make every generated-video observation unreliable.

For an intended rigid hold, choose one feasible robot grasp across the difficult
object orientations: a common object-relative palm/wrist transform, finger shape
and material contacts. The arm/elbow trajectory and acquisition/release joins
may change within the declared robot edit scope to realize it. Test the whole
group when revising the grasp rather than choosing a different fit at each key.
Retain explicit articulation, sliding or regrasp when required by the action;
an authored change of grasp needs its own checked transition.

Declare which hand/arm source targets become soft guidance or diagnostics before
solving. Reduce or omit conflicting wrist, finger or elbow priors in that scope;
keep unrelated body/support guidance. Distinguish the source prior from the
current iteration seed and the shared object-relative grasp target. Review and
report intentional source deviations, but do not reject an otherwise valid
scoped correction solely for disagreeing with the unreliable source hand pose.
Revising a hand/arm prior does not waive separately declared torso, shoulder or
elbow deviation corridors; resolve those conflicts through an authorized shared
object-path or grasp revision.
Keep the source observations, hashes and calibration unchanged for comparison.

This policy does not relax native joint limits, material-contact/slip checks,
collision checks or temporal gates. Object trajectory, root/legs, scene, scale
and source timing retain their existing edit permissions; a hand correction
does not implicitly open them. Separately authorized robot/object or table
translations follow the [placement and support procedure](scene-aware-repair.md#adjust-placement-while-retaining-table-support).
A noisy fixed object path can drive a rigidly
attached hand too fast, and a fixed grasp may reach a joint limit. Diagnose
those causes separately from source mismatch and revise the shared grasp/arm
branch or use an already authorized trajectory adjustment.

Record the affected hands/phases, intended action, authored grasp interpretation,
revised priors and still-protected fields in the case plan and remote prompt.
These are planning decisions, not a new source-policy solver flag. `hold.py`
uses soft wrist objectives and sequential fitting; `sequence.py` couples native
samples over the declared hold and buffers using fixed material targets. Neither
turns a revised source prior into hard contact or acceptance evidence.

## Adjust object motion while preserving reference body posture

Use this route when the user permits object-trajectory correction and a stable
grasp otherwise pulls the robot away from the reference human body pose. Treat
the reconstructed object path as a revisable prior within that scope. Preserve
the intended manipulation, hand roles and event timing; generated finger or
wrist inconsistency does not make the torso, pelvis or upper arms expendable.

Before fitting, bind the original human checkpoint/video, the initial robot
posture reference, original object path and current candidate separately. Set
case-specific corridors for torso lean/twist, pelvis orientation, shoulder/elbow
arrangement and foot support, plus object position/orientation reference costs
and trajectory smoothness. Object displacement/rate caps are optional in the
SOMA object mode. Use native waist/arm correction corridors and source posture objectives
where supported; compare body-axis directions or calibrated landmarks against
the human, allowing for different limb lengths. Native joint deviation from the
initial conversion is only a proxy, not a human-pose similarity score. Do not
reset the reference to an already leaning candidate on the next iteration.
Check that the chosen robot reference itself preserves the source body action.
If conversion already distorted it, establish and version a source-guided body
reference first rather than making the bad conversion a hard target.

1. Keep the source-derived object regions and fixed material identities of each
   current grasp hypothesis. Use normalized multi-point patches per participating
   digit; more points must not increase a group's total contact weight or hide
   missing member coverage.
2. Try the smallest useful smooth object translation/rotation shared by both
   hands, before increasing torso/arm deviation bounds. Keep one world object
   pose at each time. Preserve task-relevant orientation and stationary/support
   phases; explicitly author and review any changed pickup or placement relation.
3. `tools/grasp/hold.py` supports bounded `object_motion` alongside native arm
   fitting and `body_guides`; see its [actual schema](../../../../tools/grasp/README.md).
   It mounts only eligible screened bank candidates and does not repair phase
   transitions. If that prerequisite fails, revise the grasp hypothesis or
   explicitly author a versioned native object-path candidate with a validated
   export; do not mark the bank accepted to get through the tool. The standard
   robot-only Blender export freezes object transforms and is not an object-curve
   editing route. The hold tool's per-frame offset-step limits are not
   acceleration/jerk gates, and its scene term does not audit moving-object/table
   collisions.
4. Before `sequence.py`, prepare a versioned native parent combining the revised
   object transforms with the original reference robot qpos, and a seed with
   **exactly the same** object/root/scene/timeline fields. Record any separately
   authored endpoint exceptions and preserve the untouched original as
   `diagnostics.baseline_npz`. Bind the source-reviewed original robot pose through
   [`posture_reference`](../../../../tools/grasp/README.md#keep-the-original-native-posture-as-a-separate-reference)
   with explicit body joint/window limits. It supplies a separate posture target
   and final deviation gate while the current parent still defines editable
   corridors and protected samples. Ensure those corridors can reach a permitted
   body pose; a narrow corridor around a distorted parent can be infeasible.
   Rebind native/contact/audit inputs to the new path.
   Preserve the original bank as descriptor lineage; `sequence.py` does not
   consume it. Before calling `hold.py` again, generate and screen a bank bound
   to the new parent, rather than editing an old bank's binding/hash. The sequence solver refines
   joints against that fixed path; this is an alternating edit procedure, not
   an implemented simultaneous object-and-robot trajectory optimizer.
5. Compare the exported result against both the human reference and the original
   robot/object baseline at matching frames. Check torso/pelvis orientation,
   shoulder/elbow arrangement, support, fixed-anchor gaps/slip and all affected
   geometry, including object/table. Audit absolute object and wrist derivatives
   as well as correction derivatives and joins. Keep on-table contact within
   the finite tabletop with an appropriate support gap; clearance alone is not
   support. Carry the exact revised path into native export and Blender replay.

Body fidelity is an independent selection dimension. A trial outside its
declared body corridor returns to shared object-path/grasp/arm-branch editing,
even if contact improved. Retain it only as a labeled diagnostic or explicit
scoped partial with the body regression stated. The native `posture_reference`
gate applies only to its declared joints and integer frames; it is not a human
similarity or continuous-time check. Existing source diagnostics and visual
review remain separate: record actual metrics, limits and coverage, and mark
absent evidence unknown.

## Batch surface grasps before trajectory repair

Use this recipe for an unreviewed or poor grasp; a selected grasp undergoing a
timing-only edit does not need another candidate search.

For a sequence with a support scene, interleave the
[initial-run scene repair loop](scene-aware-repair.md#own-scene-repair-during-a-from-scratch-run)
with this grasp work. Inventory approach/phase-change crossings early and author
needed clearance keys alongside mounted grasp keys. Check their connected paths
before full propagation, and recheck joins when a diagnostic endpoint changes.
The stage order below is not a reason to defer known scene crossings until the
grasp search has consumed the repair budget.

| Stage | Output and decision |
|---|---|
| Source planning | From the video and reference human motion, declare per-hand object regions, eligible original hand surfaces, active phases, grasp groups, evidence frames and uncertainty. A stable support hypothesis is labeled as such when source evidence does not establish no-slip holding. |
| Small candidate batch | Declare a finite seed bank and common constraints; a few source-compatible seeds per hand is a starting point, not a required count or evidence of distinct grasp types. Reuse original mesh queries and native FK. Optional GraspGenX supplies proposals only. |
| Surface fit and screening | Use `tools/grasp/bank.py` to fit declared whole-hand surfaces and clearance without contact-normal alignment, before discrete anchors are fixed. Sampled distance losses guide proposals; original triangle intersections and source views decide whether they can advance. |
| Joint mounted fitting and evaluation | Enable `hold.py`'s `finger_refinement` to fit native fingers and arms together against the hypothesis's fixed material contacts, with declared waist/object freedom where needed. Test both active hands together. Then run `tools/grasp/evaluate.py` against one parent/geometry/scope/gate policy at difficult hold keys and short transitions. Review geometry, gaps and source views; `screening_only` reports never promote a candidate. A failed mount returns to a revised grasp/arm branch or an explicitly diagnostic coupled repair. |
| Descriptor handoff | Save the chosen native pose, per-hand shared object-relative transform, finger configuration, active interval, original mesh/link/face/barycentric identities, evaluation tolerances and review evidence. Material targets stop moving once this candidate enters propagation. |
| Coupled sequence repair | Use `tools/grasp/sequence.py` to fit the fixed-material hold with editable acquisition/release buffers when mounting leaves slip or bad joins. Preserve the original posture prior/corridor and explicit seed status. If authorized, prepare a shared object-path revision to avoid body-pose regression, then bind it consistently before the joint solve. For an already reviewed hold, keep an approach-only repair local. |
| Independent review and retention | Audit complete declared original-mesh coverage and temporal joins, check full-hold gap/slip, and review the actual export visually. Save bound evidence with `tools/repair/progress.py`; keep scoped partial, diagnostic and full-motion outcomes separate. |

Keep eligible surfaces distinct from selected contacts. Proximal segments,
finger sides, finger backs and palm surfaces can participate when the observed
arrangement and native geometry support them; contact is not restricted to
finger pads. A result that chooses distal surfaces is valid if the
geometry/source review supports it. Do not force
a palm contact simply to diversify the descriptor. Proximity alone does not
establish opposition, a useful grasp or force closure. Inspect a canonical hand
map together with the mounted hand/object pose.

When a whole-hand grasp needs a denser descriptor, use the bank's
[`contact_patch` configuration](../../../../tools/grasp/README.md#multi-point-contact-patches).
Keep the source-level thumb/other-finger mapping, but declare separate robot
members for the digits expected to participate. For example, a pooled long-finger
region can contain separate index and middle members; one nearest point from
their union is insufficient evidence that both participate. The point count is
a case setting, not a requirement that every grasp uses three digits or pads.
Select separated original-surface pairs within the declared gap threshold,
report unavailable coverage, and preserve normalized weights when converting
them into solver contacts. Do not increase a contact's total influence merely
by adding sample points. Keep each selected material identity fixed through its
phase; a refreshed dense proximity map is diagnostic and cannot replace it.

During candidate construction, surface correspondences may change within the
declared regions. After selection, bind physical material points and measure
their gap/slip independently of wrist-pose guide points. If the descriptor is
revised later, give it a new identity, retain old-target evidence, and evaluate
parent and candidate against the same revised targets. Do not report a smaller
distance to a different surface point as preservation of the old contact.

The [public grasp tools](../../../../tools/grasp/README.md) extract the region
search and shared-grasp solve from the drill/fire experiments. `bank.py` batches
free-wrist candidates with sampled clearance, then screens declared original
hand triangles. `hold.py` mounts a selected bank entry through explicit native
arm/waist variables, opt-in finger refinement and optional authorized object
offsets, at specified keys or every hold frame. `evaluate.py` compares mounted alternatives; `sequence.py`
repairs their fixed-material holds and buffers jointly. None accepts a motion.
Independent coverage, slip, original-mesh, temporal and source/visual checks
remain necessary. These commands are not
automatically triggered by Blender sequence loading. Stage configs/inputs and
the installed package once; do not recreate a solver inside each case.

## Make the grasp inspectable

Create a small per-hand descriptor before propagation. It should identify:

| Field | Meaning |
|---|---|
| Role and evidence | Grip, support, slide, approach or release; observed source intervals; uncertain/occluded parts. |
| Grasp identity | Per-hand continuous phase, member source keys, shared object-relative palm/wrist transform and robot hand-shape version, with any permitted articulation. |
| Anatomical mapping | Human thumb to robot thumb; the human four-finger group to both robot long fingers collectively. Record actual robot links/surface regions, contact sides and local axes. |
| Surface targets | Original object mesh hash, face plus barycentric coordinates/material point, or explicitly bounded support patch. Surface normals may be retained as diagnostic geometry. |
| Hand targets | Original hand-surface material points/patches, wrist/palm orientation and optional soft elbow guide. Contact-normal alignment is disabled by default. |
| Active relation | Contact intervals, per-hand closure onset/completion and digit clearance keys, object-relative wrist/pad relation or permitted articulation/sliding. |
| Edit scope and tolerances | Editable native variables, frozen fields, contact/pose/geometry criteria declared for this trial. |

Robot link names alone do not prove anatomical correspondence. Inspect the original meshes, source video and reference human motion. For G1 + Dex3, map the robot thumb to the human thumb and the two robot long fingers collectively to the human index/middle/ring/little-finger group. This is a functional many-to-two correspondence, not a requirement to copy the human index and middle fingers or split the four fingers into fixed pairs. Choose the robot distribution from the observed grasp/support phase and its native reach: both long digits may support the same surface, or one may contribute curl/clearance without a fixed contact. Do not carry a previous clip's choice into a new grasp.

Contact may occur on distal or proximal finger surfaces, finger sides, finger backs or the palm. Choose the whole grasp and contact regions before fixing material anchors; a three-pad layout is not a universal default. Select regions on both the original robot hand mesh and the original object mesh. Record their mesh identities and source evidence, then bind any solver anchors as original face/barycentric material points. A region can contain several useful contacts; an exported sparse anchor set does not describe the entire predicted contact region. Current tools do not automatically infer dense contacts from video. Agent-authored predictions use the `authored_hypothesis` label, with uncertainty for hidden surfaces; do not turn measured proximity or a penetration patch into a target without source/geometry review.

Inspect phase changes before reusing a region through the sequence. A rigid hold
can preserve its material anchors, while a visible slide, support transfer or
regrasp needs a separately declared relation and transition. Video supports
contact topology more directly than exact hidden 3D coordinates; reference
human motion guides the mapping but is not ground-truth robot contact. A change
of camera view alone does not justify a new contact phase.

Before fitting a new object's targets, check their compatibility with the native hand: reachable contact spacing, fixed bending-plane offsets, palm pose and joint limits. Human contact points do not automatically form a feasible robot grasp. In the tested G1 model, the two long pads retain a 57 mm component along the wrist row axis; the transferred Frypan targets were only 30–34 mm apart. Derive such bounds from the actual model and pad definitions instead of treating these numbers as defaults. When a descriptor changes, declare it before the next solve, compare baseline and candidate on the same new targets, and retain the previous target measurements.

Verify each hand's actual hinge axes and signs. A palm frame copied between mirrored hands can make the digits curl away from the object, even when the named links and pad targets look symmetric. Inspect inactive digits as part of the whole hand: a former gripping thumb may need retraction during a cradle phase, and an unused long digit still needs a plausible curl with clearance. A single corrugated mesh facet can also misrepresent a pad's contact direction; inspect the nearby original surface and distinguish facet normals from an authored patch normal without changing the mesh or silently replacing the old metric.

Contact topology (which digit is above, below or opposite) and exact material location are separate decisions. Label artist-authored and occluded 3D contacts as hypotheses, even when a source image supports the general topology. A descriptor borrowed from another phase can put the fingers on the wrong part of the same object; inspect its source projection before solving. Hidden contacts are hypotheses. If a surface patch is allowed to slide, bound it in object coordinates and report material slip separately. Do not reselect an arbitrary nearest point each frame and call the resulting small distance stable contact. If an artist chooses new targets from a candidate, retain results against the old targets and identify the new correspondence explicitly.

Only when contact-normal constraints are explicitly requested: for contact at a genuine convex original-mesh edge, a declared geometric normal cone can use positive combinations of the incident original face normals. Bind the shared edge, material coordinate and incident faces; retain each raw facet normal/error separately. This represents an edge-contact hypothesis, not a smooth replacement surface or a friction/force model. Its native transition still needs review: continuous target positions do not prevent jumps when contact terms, editable joints or arm branches switch at a phase boundary.

## Diagnose and edit before solving again

Use this outer editing loop whenever review leaves an in-scope contact defect. The solver only fits the current hypothesis; more iterations cannot establish that its contact definition, palm pose or auxiliary path is appropriate. Inspect the failing hand/phase in source and diagnostic views alongside the native measurements, then route the next edit:

| Observed failure | Inspect and change | What the trial must distinguish |
|---|---|---|
| Wrong contact side, incompatible pad spacing, or small gap with wrong opposition | Check anatomical roles, native pad/hinge signs and original-surface target layout. Version the supported digit-to-patch mapping, material anchors or allowed contact regions; test hand feasibility before mounting. | A feasible contact definition versus forcing the same incompatible targets harder. Keep observed topology separate from uncertain hidden coordinates. |
| Isolated grasp fits, but the mounted palm/wrist intersects or the arm departs from the source | Inspect mounted FK and wrist/arm branches. Author a different native palm position/orientation or arm branch within the declared source corridor and original limits, initially retaining contact targets. | Whether mounting/palm orientation causes the failure. Review every digit, palm and arm; do not infer success from free-wrist IK or silently drop source constraints. |
| Active pads fit, but an auxiliary digit crosses the object, self-collides or snaps | Inspect that digit through the whole affected interval. Author its own native retraction/curl/clearance keys and timing; an inactive contact term does not specify a clear path. If the wrist or active digits force the obstruction, edit the coupled transition within scope. | A continuous plausible auxiliary path with active contacts preserved, rather than clearance transferred to another digit. Check original-mesh FK at keys and subframes, plus finger speed. |
| Clear grasp keys produce collisions, contact drift or jumps between them | Inspect the actual exported interpolation, contact activation, binding and both sides of each join. Edit phase timing, preshape/outside waypoints or neighboring native states together. | A usable transition versus merely clear endpoints. Follow [temporal editing](temporal-editing.md) for coupled paths, joins and closure speed. |

Do not mechanically change all three assumptions at once. Choose the smallest supported change that can explain the blocking defect; couple controls only when the evidence shows they interact. If the cause is unclear, perform a discriminating diagnostic (for example, fixed-contact alternate palm poses or auxiliary clearance with the wrist held) before another fit.

For each outer round:

1. **Declare the experiment.** Record the parent checkpoint/descriptor hashes, failing hand/phase, hypothesis, exact descriptor or native-control change, frozen controls and expected distinguishing evidence. Include a bounded solver budget; it ends this trial, not the editing loop.
2. **Make the edit.** Save the revised descriptor and actual native pose/curve controls, then run any required fit from that version. Verify the loaded targets, editable joints and effective constraints. A proposed waypoint, unchanged render or new solver seed is not an executed path edit.
3. **Review the exported state.** Compare per-digit material gaps, object-relative slip, whole original geometry, source/fixed-depth evidence and affected temporal intervals together; contact-normal angles remain optional diagnostics. If contacts changed, evaluate parent and candidate against the same new definition and retain old-target results; never present target relocation as hand-motion improvement. Preserve the original baseline and frozen source/depth calibration.
4. **Decide and continue.** Save the checkpoint, actual delta and review outcome. Retain useful scoped improvements; reject regressions. If the same defect survives, use its failure evidence to revise the relevant assumption or run the next diagnostic, then repeat within task authorization. Do not count repeated solver runs as exploration of unchanged contact/palm/path choices.

A numerical retry is appropriate when a diagnosed gradient, coordinate, initialization, weight-routing or solver-limit problem explains the failure. State that diagnosis and verify the narrow fix; changing weights or random seeds alone does not test a contact topology or author an auxiliary path. No fixed number of rounds establishes success or infeasibility. Use the task-level stopping rules in [the skill](../SKILL.md#continue-the-requested-correction), not a single trial's budget.

## Establish the mounted native grasp before propagation

An isolated hand with a free wrist is an optional initialization diagnostic.
Use source-supported alternatives in palm orientation, hand shape or contact
layout when the current hypothesis fails; the current bank does not automatically
invent distinct palm seeds or infer contact maps from video. Record what actually
varies between initializations. Several source frames with the same poor layout
do not necessarily explore different grasp hypotheses.

For a new grasp, enable the hold tool's
[`finger_refinement`](../../../../tools/grasp/README.md#joint-arm-and-finger-fitting)
before choosing a mounted candidate. It fits fingers and actual arm joints
together against fixed original-material contacts, within native limits and
declared correction bounds. Include waist variables when needed and authorized,
and solve both active hands against the same bounded object pose. Keep the wrist
pose a guide so the hand can adapt to mounted reach; material targets cannot be
replaced with new nearest points to hide drift. Recheck source pose, whole-hand
and arm geometry, self/table/object intersections and per-member coverage. The
tool's `body_guides` preserve positions from the input native robot motion;
they are not automatically the original SOMA human targets. When using explicit
mapped human or separately reviewed native body targets, bind them through
[`body_reference`](../../../../tools/grasp/README.md#fixed-body-targets-during-mounted-fitting)
so position/orientation guidance stays independent of the evolving grasp seed.
Prepare their calibration and frame alignment explicitly; the tool does not
infer them. Retain source-body evidence and review separately from this soft
tracking objective.

Rigid-finger mounting remains available for a reviewed hold whose shape should
stay fixed, or as a declared comparison. It should not be the sole feasibility
test for a new grasp followed by an assumption that trajectory repair will fix
its hand geometry. If joint mounting fails, inspect contact layout, palm
orientation and arm branch before retrying; a bounded failure does not establish
that the source region is wrong. Show the mounted pose separately and do not
inherit an isolated hand's acceptance. Save the native whole-pose initial grasp
and unchanged material descriptor as an explicit handoff.

For a sequence request, before propagation check the initial grasp at representative later object orientations, especially difficult shoulder/wrist configurations. Record joint-limit margin and source-body deviations. A failed bounded IK search is not proof of global infeasibility.

Distinguish original joint limits from an artificial search window around the baseline. If the latter binds, a declared trial can widen it while retaining native limits and source review. If rigid object-relative binding still fails during visible object roll, inspect whether the source supports circumferential contact redistribution. Declare the phase, original-surface path and uncertainty; report circumferential travel and longitudinal slip separately, retain fixed old-anchor measurements, and review the transition rather than switching anchors independently per frame.

Make initial grasp calibration a concrete handoff: save the native pose, fixed material descriptor, source/novel-view review and geometry decision. A sequence with roughly hand-sized gaps needs wrist/finger initialization, not only small joint corrections or smoothing. Completing the retarget and Blender blocks does not establish this handoff. A partial static contact anchor can justify a bounded reach diagnostic, but it does not authorize labeling the propagated motion accepted. Check actual opposition and palm orientation visually even though contact-normal alignment is disabled; a small gap alone does not establish the intended grasp.

For a rigid hold, define the convention explicitly. With transforms mapping local coordinates into world coordinates:

`T_object_wrist = inverse(T_world_object) @ T_world_wrist`

Propagate `T_world_wrist(t) = T_world_object(t) @ T_object_wrist` and solve native arm joints with source-pose/elbow guidance. Also check actual material-contact gaps and whole-hand geometry: exact wrist binding can still leave a gap or cause penetration. Reconsider rigid binding when the video supports articulation or a support hand rather than a fixed grip.

Treat approach, orient/preshape, closure, hold and release separately. Turn and preshape while clear; test the swept finger path, not only wrist endpoints. If interpolation crosses the object, first adjust orientation/closure timing and, if needed, add a small source-compatible outside waypoint. Declare its range and review 2D pose effects. Keep contact inactive until the observed phase; preserve precontact object stationarity.

Use continuous native joint branches and appropriate rotation interpolation (e.g. quaternion sign continuity/SLERP); reconstruct articulated geometry via FK at subframes. Avoid interpolating independently exported rigid link transforms as if they defined a valid robot. Smooth local controls and joins, then re-evaluate the actual exported interpolation. Low joint jerk and stable object-relative grip can coexist with high world-space wrist jerk when the object path is noisy. Consider coordinated hand/object smoothing only within authorized trajectory scope.

For smoothing or slower closure of an already selected motion, follow [temporal editing](temporal-editing.md). Preserve meaningful thumb reversals and support keys; one common closure curve across both hands can change a previously clear swept path. A curve's analytic smoothness does not establish the smoothness of its sampled native export.

## Blender controls and round trip

Use a scene with a locked calibrated source camera, diagnostic cameras, source reference, original geometry, contact markers and phase markers. Useful controls include wrist position/orientation, soft elbow guidance, robot finger closure, object-local grasp binding, local correction curves and version/undo snapshots. Build only the controls needed for the requested edit.

Use available MCP or Blender Python/CLI; do not invent a tool dependency. Drive native variables and FK rather than display-only deformation. Reopen the saved scene and check:

1. Untouched export preserves native motion, metadata and frozen fields to the format's declared precision.
2. A real control edit changes the intended native variables; undo restores them; frozen controls remain enforced.
3. Integer and subframe preview matches the native exporter/FK. Clearly distinguish baked preview samples from exact native subframe evaluation and document how to activate required helpers.
4. Edited exports get a new path and **UNREVIEWED** status; prior review cannot be inherited after an edit.

Deliver the requested scope. A static pose can be shown or selected as a scoped partial without first solving the full animation; label isolated versus mounted views and remaining penetration explicitly. Do not present a single-frame preview as a new sequence.

Physics/collision preview is optional feedback. Kinematic parenting/IK does not automatically solve grasp dynamics, and simplified physics colliders can miss original mesh penetration. Validate any physics-generated motion after baking/export. Do not advertise force closure, balance, environment collision clearance or physical stability unless those were actually tested.
