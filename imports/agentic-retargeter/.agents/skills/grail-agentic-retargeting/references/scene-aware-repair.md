# Scene-aware grasp and transition repair

Use this route when the handoff carries original scene geometry or the user
requests scene/self clearance. Preserve its world binding unless a placement
correction is authorized, and record any degenerate original faces as incomplete
geometry coverage. A correction retains the original geometry and provenance.

- Audit robot/scene **and object/scene** before fitting hands. A bounded object
  correction can create table crossings even when robot/scene checks improve.
  Prefer a fixed object path when it is reachable. If the user permits a support
  placement correction, declare it separately, preserve stationary intervals,
  and check the complete object path against the original scene geometry at the
  declared placement, including its support checks below.
- Include shoulders, arms, palms and auxiliary fingers. A wrist path over the
  table can still put an upper arm through the torso. Sampled winding losses on
  open robot meshes can miss these crossings; native distance proposals and
  independent original-triangle audits provide complementary evidence.
- Establish a feasible first contact independently for each hand. Do not clamp
  a newly acquired hand to its previous *unfitted* pose. The shared hold tool
  releases only that new hand's unique arm ancestry; shared torso variables
  remain coupled to an ongoing hold. Repair and review the acquisition join
  afterwards; this exception is not a smoothness pass.
- For a fixed grasp, leave clearance for mounted FK/interpolation errors rather
  than chasing zero unsigned gap. Declare the proposal margin independently of
  the accepted contact tolerance. Reject the failing key/short transition
  before propagating; preserve a passed hold while repairing its approach.
- For a local object edit, reserve its return to the next protected/unsolved
  sample throughout the window. Checking only the last step can produce an
  empty corridor after otherwise successful solves.
- Check actual wrist acceleration as well as joint acceleration over the joins.
  Preserve the configured absolute RMS-excess policy against the same human
  reference; a reduced joint loss alone does not establish smoother hand motion.

The optional bank `self_clearance_weight` places nonadjacent hand clearance
inside the search objective. Explicit `retained_self_pairs` require confirmed
neutral-model crossings and documented reasons. They remain unresolved assembly
overlaps in reports, not cleared pairs; compare their geometry before/after and
do not claim total self-collision freedom. Do not declare a new collision an
assembly seam merely because the pair already crossed somewhere else.

`clearance_objective.BatchedClearance` is a reusable GPU/CPU proposal primitive
for explicitly paired moving and static original meshes. It differentiates both
moving transforms and batches poses, with persistent query meshes. It does not
select contact phases, plan paths or replace the full audit. An explicit
degenerate-face query subset retains original-face mapping and UNKNOWN coverage;
the independent audit still receives the complete original mesh. See the
[repair tools](../../../../tools/repair/README.md#batched-scene-and-self-clearance).

Use phase-appropriate local windows and cached unchanged evidence. A failed
collision branch should trigger a different feasible route or grasp, not many
unconstrained repeats of the same loss. Keep the case deadline and retain the
best scoped result when source, geometry or smoothness remains unresolved.

## Own scene repair during a from-scratch run

A fresh retarget includes approach, acquisition, hold and release, not only a
grasp fit. When a support scene is present, make its clearance an early editing
decision in the same run. Source inspection or a confirmed mesh crossing is
enough to trigger investigation; do not wait for a user to identify the interval.

- Inspect the newly converted full timeline and record scene-crossing intervals
  per limb, including motion before first touch and between grasp keys. Reuse
  bound geometry caches and refine sampling near hits; a few passing hold keys
  do not establish a clear approach. Distinguish positive triangle intersections
  from incomplete geometry and unknown samples.
- Put each confirmed localized crossing into the first feasible editing round
  alongside the grasp keys. Author a native interior clearance pose and check
  its reachable connection to the applicable contact endpoint before committing
  to expensive full-hold propagation. While the grasp endpoint is unresolved,
  a route can remain diagnostic; when that endpoint changes, recheck its join.
  Do not defer all scene repair until the delivery reserve.
- Use the intermediate-key procedure below to change the actual pose and path.
  A larger collision weight, a smooth-looking curve, a plan to add a key, or a
  clear single frame is not a completed clearance repair. Save the authored
  native angles and window, confirm that they survive export, then check both
  sides of the key and incoming/outgoing temporal stencils.
- If the crossing moves between keys, revise the route or add a targeted key
  there. Keep local arm/finger controls available throughout the real interval.
  Preserve reviewed hold relations and other frozen fields; if a join conflicts
  with a diagnostic grasp endpoint, revise that hypothesis within the existing
  authorization instead of freezing its failure. User-protected endpoints stay
  protected. Do not change source timing, body/reference limits or placement
  merely to make the audit pass.
- After hold propagation, object/placement edits or temporal projection, check
  changed scene pairs and joins again. A new crossing returns to the same agent's
  editing loop. Compare contact, original-mesh clearance, source fidelity and
  joint/wrist derivatives separately so fixing the table does not hide a new
  object/body intersection or jitter.

Use the existing case plan and checkpoint record for each interval: offending
pairs and times, actual authored key/checkpoint, editable/protected scope,
matching before/after sample grid, residual/new collisions, and temporal/join
review. Record whether it was repaired, remains a tested diagnostic, is blocked
by a concrete constraint, or was not attempted before the deadline. Missing
checks remain unknown. This is evidence for the final decision, not a new file
schema or a supervisor acceptance mechanism.

Continue evidence-supported edits without another user prompt while the case
budget permits. Reserve delivery time and retain cumulative spending. At a
budget or prerequisite stop, deliver the best clearly labeled partial and the
unresolved intervals; do not silently fall back and call the from-scratch
workflow successful, or submit a follow-up job on the user's behalf. A skill
improvement request changes future behavior; it is not itself a case rerun.

## Adjust placement while retaining table support

Use this option when the user permits placement changes for table clearance.
Record that permission in the case plan and remote prompt; do not ask again for
an already authorized adjustment. Prefer a small, constant horizontal translation
over a time-varying offset. Derive forward/backward from the robot/table relation
in the bound ground plane, not image axes, and declare the direction and bounded
trial offsets in meters before fitting. Choose the smallest useful feasible
offset; a historical case's displacement is not a default.

- **Move robot and object together:** apply the same world translation to every
  native robot-root and object position for the complete timeline. Preserve
  orientations, joint values and timing at this placement step. This preserves
  hand/object relative motion, including any existing slip; it does not repair
  the grasp itself. Keep the table fixed and recheck feet against the ground and
  other scene geometry.
- **Move the table:** apply one constant translation to the complete original
  table, keeping robot and object motion fixed. An opposite table displacement
  produces the same robot/object relation to that table as the paired shift,
  but differs relative to the camera, ground and other scene elements. Moving
  all three together does not change their mutual clearance.

For every declared on-table interval and placement/release boundary, check actual
object support geometry against the finite tabletop: supported contact must
remain on the tabletop with a declared edge margin and vertical-gap tolerance,
without penetrating it. A center above the table or a low collision count alone
does not establish support. Preserve stationary intervals and lift/place timing;
do not require table contact while the object is intentionally carried. Retain
the hand/object contact and slip gates and check robot/table, object/table and
other affected scene pairs over the full path, including subframes near crossings
and support transitions. Report missing support evidence as unresolved; geometric
support alone is not a dynamics/stability certificate.

Save the offset as an explicit candidate relative to the unchanged upstream
baseline. Bake it into native root/object data or the bound static scene as
appropriate; solver, audit, Blender and exported motion must use the same
placement. Respect the root/object/scene coordinate conversions and apply each
offset once; a scene with world-baked vertices must not receive a second copy of
the same transform. Verify the saved native/Blender round trip. Preserve original
mesh identities, scale, source observations, camera calibration and input hashes.
Rebind derived world-space grasp/route targets and invalidate dependent geometry
caches/checks; reuse only evidence invariant under the chosen translation. Report
the authored displacement and source-overlay deviation separately from physical
acceptance. Existing local tools may freeze root/scene placement: prepare and bind
the translated candidate explicitly before local repair, rather than assuming an
object-only solver option or a Blender display offset implements this operation.

## Match collision frames to editable controls

Use the bound native baseline audit to locate scene intersections over the
requested motion scope, including the period before either hand acquires the
object. Contact onset determines when grasp constraints turn on; it does not
determine when an arm first needs collision repair. Preserve a full-duration
baseline inventory for a whole-motion request. A transition's own preflight only
covers its configured audit window and halo.

The transition command checks the actual per-joint masks at those hit samples
before fitting. Inspect which local joints can affect the colliding mesh,
which are frozen, and whether only a shared ancestor such as the waist can
move it. At interpolation samples, both neighboring native keys matter;
protected endpoints remain protected. A collision outside usable local control
requires a revised explicit window or a scoped unresolved result, not another
trial with a larger clearance weight. Do not expand root or object freedom
implicitly. Reuse cached original-mesh rows when their bindings are unchanged.

When a sampled clearance solve still leaves a table crossing, inspect the
failing frame and the mounted hand/forearm path. If the seed cuts through the
table edge, author a source-compatible clearance/preshape key or waypoint before
repeating the solve. Keep fingers open or orient them while clear when the
observed motion permits it; do not use future grasp attraction during free
approach. Check the resulting route against the whole hand, arm and torso.
Empty waypoint lists and higher weights alone are not evidence of a new route.

Mounted grasp propagation also needs scene guidance; a robot self-collision
proxy list alone omits the table. Inspect the shared-hold report's scene coverage,
frozen geometry and omissions. Its finite sampled objective remains a proposal,
and all original robot/table triangles remain in the independent audit. Preserve
the original scene, including geometry warnings, rather than repairing an audit
failure by dropping faces or hiding the table.

For delivery, use the complete requested scope's audit result. An
`affected_scope_passed` local result does not erase inherited scene collisions.
Keep failed/incomplete candidates unselected and explicitly label diagnostic
clips; do not present the clean hold segment as a complete repaired animation.
These checks replace ineffective retries, not add a second routine review loop.

## Repair intersections with intermediate pose keys

Use an actual pose edit when a localized crossing remains after smoothing or
clearance optimization. Choose frames from the current motion; historical case
frame numbers, margins and joint angles are not defaults.

1. **Locate the interval.** Identify the offending original mesh pair, entry/exit
   times and deepest or most representative crossing. Inspect source and mounted
   side views. Choose an interior key near that state, with neighboring phase
   boundaries and the editable native joints recorded.
2. **Make a reachable clearance pose.** Adjust wrist position/orientation and
   finger preshape as needed. Include arm ancestry when finger flexion alone
   cannot move the offending finger base. Check native limits and reachability
   from protected endpoints under the configured speed, acceleration and jerk
   policy before a costly trajectory solve. Endpoint speed bounds alone are not
   sufficient; a constrained temporal projection can test the full policy.
3. **Reconnect the motion.** Use native pose keys and a bounded smooth
   initialization, then couple neighboring states under the temporal constraints.
   The transition tool's `pose_keyframes` fixes authored hinge values during
   refinement/export. Keep source timing, the declared scene/object placement,
   unrelated body motion and reviewed holds unchanged within the declared scope.
   Verify the exported samples and actual Blender interpolation; an internally
   smooth curve does not make baked LINEAR playback continuously smooth.
4. **Return to editing after failed path review.** Check both sides of the key,
   subframes and incoming/outgoing joins against original hand/arm/scene meshes.
   If the crossing moves earlier or later, place a targeted approach/preshape key
   in that residual interval or revise the route. Do not keep adding keys or
   repeating the same solve when the frozen path remains infeasible. Distinguish
   newly authored diagnostic keys from user-protected or reviewed grasp keys:
   revise a diagnostic key explicitly within the authorized scope when it blocks
   a feasible path, and recheck its pose and affected joins. Never silently move
   a protected hold or relax temporal limits to make the edit pass.
5. **Compare and decide.** Use matching sample times and geometry for before/after
   collision checks, plus full native speed/acceleration/jerk gates and local
   wrist RMS/peaks. A lower collision count can worsen wrist smoothness even when
   hard gates pass; report that tradeoff. Review source and novel views, retain
   unknown geometry, and keep residual collisions marked PARTIAL. Continue an
   evidence-supported local edit within the case budget; a single failed solve
   is not task completion. Reuse unchanged checks and render affected views
   during iteration, then deliver the complete requested timeline.

The reusable key schema and overlapping-window restrictions are documented in
the [transition tool](../../../../tools/repair/TRANSITIONS.md#optional-native-velocity-constrained-refinement).
Keep saved native exports, authored keys and versioned Blender scenes bound to
the same candidate; a visual edit must survive the save/export/reopen check.

## Reuse the transition and diagnostic commands

Use [the public transition tool](../../../../tools/repair/TRANSITIONS.md) when an
authored clearance route is needed, instead of copying a case's waypoint/IK/GPU
script. Free approach follows [phase-aware wrist guidance](temporal-editing.md#choose-wrist-guidance-per-hand-and-phase);
do not add an old wrist lock just to fit a route-based interface. For a route
trial, supply source-supported phase windows, actual native joints, a feasible
held endpoint and explicit world-space route points derived from the bound scene.
Freeze each completed hold separately; shared torso edits need an explicit joint
blend or coupled follow-up. Keep the route short and review source deviation:
a long clearance detour is an authored adaptation even if contact and collision
checks improve.

The tool checks the bound baseline's scene-collision edit scope, batches seed
fitting and original-surface scene/self refinement, then audits the candidate.
Both checks reuse exact unchanged rows. Its explicit proposal-only mode is unreviewed;
a confirmed crossing rejects the candidate early with untested samples listed;
a failed or unknown geometry result remains unresolved. Preserve the selected
checkpoint until its replacement is reviewed. Respect the existing case deadline;
the solver's cooperative search timeout excludes final audit/export overhead.

Include native wrist position deviation, local acceleration peaks and their
support frames in the normal report. Use the standalone diagnostics command when
only these measurements are missing; it needs no geometry rescan. The optional
human comparison requires bound world positions and matching timestamps. Continue
using the configured **absolute RMS-excess** policy. Peak/deviation review warnings
are diagnostic, not new hard thresholds or automatic reasons for another repair.
Compare those frames visually when RMS improves but the route appears less natural.
