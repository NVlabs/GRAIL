---
name: grail-agentic-retargeting
description: Retarget SMPL-X through GMR or SOMA through NVIDIA soma-retargeter, then refine native robot motion through grasp, support, approach, closure timing and trajectory edits in Blender. Use for dual-backend conversion, G1 contact repair or source-faithful robot animation. Human/object reconstruction remains upstream.
---

# Agentic Retargeter

For routine free-hand initialization, use the
[shared hand policy](../../../docs/hand-shared-policy.md): one versioned numerical
profile across cases, source-video same-grasp/role intervals authored by the
agent, multistart Topo seeds, a 300-step base paired sequence, phase-aware SG9/poly3
initial-guess filtering and 300-step contact-aware temporal refinement. The final
refinement uses contact weight 256, four separated samples per digit at 3 mm
spacing, and 100 times the base temporal weights. Preserve the tested base solve;
do not silently skip it or export its intermediate result as the final target.
The package emits the final configuration/report/trajectory selection with
hashes. Do not tune loss weights,
transition lengths or collision collocation per case in this route. Standalone
pad refinement is disabled by default; sequence contact/collision terms remain.
Unknown/support/slide intervals stay live. Preserve raw first-hand-frame and
strict-hold filter contracts; joint refinement may adjust an entire shared hold.
When an exported warm sequence re-enters the shared-hold decoder, inspect
`shared_hold_initialization.json`: only initially invalid transition base
controls may be re-encoded. Valid starts, shared grips, frame zero, original
bounds and phase intervals are preserved; an infeasible strict grip still
fails. This numerical initialization correction is not a contact repair.
For initialization-only work, run fresh CUDA forward SOMA and synchronous SOMA,
then export and review. Post-SOMA collision-key repair is a separate requested
scope. Residual contact/collision failures remain in the report and do not
silently trigger extra repair rounds or imply motion acceptance.

For a reported burst immediately before acquisition or regrasp, use the
[approach-only hand refinement](../../../docs/hand-approach-refinement.md) as an
explicit candidate repair. Freeze the selected first frame and existing stable
holds, retain the original clock/contact assignments, and measure actual wrist
and finger speeds at the joins as well as acceleration/jerk. Review matched
contact and collision evidence before choosing the result; lower peak speed
alone does not select it or certify a mounted full-body motion.

Preserve the upstream static support scene in robot exports and Blender review unless the user explicitly authorizes a scene replacement. For the current authorized floating-block setting, use the [packaged support recipe](../../../docs/support-block.md), archive the original support and bind the replacement consistently in optimization, export, audit and rendering. Otherwise, when the object starts on a table, retain the original table geometry and dimensions, and its world placement unless the user authorizes a [placement correction](references/scene-aware-repair.md#adjust-placement-while-retaining-table-support). Use the same coordinate conversion as the moving object and bound static scene data; missing metadata is unknown, and adding visible geometry alone is not a collision check. See [upstream handoff](../../../docs/upstream-handoff.md).

Before calling a replay current, compare its exact human checkpoint and source-video hashes with the requested reconstruction. A newer completed but unselected reconstruction is a new candidate, not an accepted replacement. If the legs look wrong, compare native human hip/knee/ankle motion, initial robot motion and final edited motion to locate the change before editing the hands or translating the scene.

When a table/support scene is present, include it in SOMA initialization through `collision_avoidance.static_scene` with an explicit world-to-solver transform. Native local repair automatically includes packed scene geometry and all original robot meshes in its audit; keep its inherited failures and incomplete-query warnings. Check approach as well as hold frames. Bind the original or explicitly authorized replacement support at the declared placement, and keep foot support/source-pose checks separate from hand contact improvement. A scene added only for rendering is not a scene-aware repaired motion.

Choose scene-repair windows from observed collision frames as well as contact
phases. Before solving a transition, inspect its bound collision-scope preflight:
a hand that reaches the object late may already intersect the table early.
Shared waist freedom alone does not replace editable local arm/finger controls.
Revise an inadequate joint/window declaration within the authorized scope before
another solve; do not silently move the root, table or protected hold. Carry scene
clearance guidance into mounted shared holds, and keep inherited scene failures
in the full-motion gate. See [scene-aware repair](references/scene-aware-repair.md)
for the route-change decision and tool limitations.

When authorized, test a small constant horizontal translation of either the
robot and object together or the table alone. Choose the smallest useful offset
in scene coordinates and preserve object support during on-table phases; a clear
robot/table path with an unsupported object is not a repair. Follow the
[placement procedure](references/scene-aware-repair.md#adjust-placement-while-retaining-table-support)
and carry the permission and support constraints into remote case prompts.

For a localized table/scene crossing, execute the
[intermediate clearance-key loop](references/scene-aware-repair.md#repair-intersections-with-intermediate-pose-keys):
locate the collision interval, edit a reachable native hand/arm pose inside it,
reconnect the neighboring motion, then inspect the whole changed path and its
joins. If the collision shifts between keys, revise the approach or add a
targeted key; a clear midpoint alone is not a repaired transition. This is part
of the owning agent's work, not a suggestion to wait for the user to request.

Start from the selected native robot checkpoint and its upstream human/object handoff. For a new or poor grasp, use **source grasp groups → optional free-hand hypotheses → joint arm-and-finger fitting → common mounted evaluation → coupled hold and acquisition/release repair → independent native and visual review → preserved scoped progress**. Reuse reviewed stages for a local correction. The skill keeps the name `grail-agentic-retargeting` for existing callers; this repository is canonical for future robot-refinement changes.

An explicit from-scratch request starts a new conversion and robot contact search.
Reuse the requested human reconstruction, bound HOI object-region priors and
runtime assets as inputs, but not earlier robot trajectories, grasp poses or
material anchors as initialization. Keep earlier results for labeled comparison.

For a **from-scratch sequence with a support scene**, follow
[scene repair in the initial run](references/scene-aware-repair.md#own-scene-repair-during-a-from-scratch-run).
Inspect the fresh approach and later phase changes early. Include reachable
clearance keys alongside grasp keys in the first editing plan when crossings
are present; do not spend the repair budget only on holding contact and leave
known approach penetrations for a later user turn. Test the affected route and
its acquisition join before expensive full-hold propagation. After any grasp,
object or smoothing change, return new crossings to this same editing loop.
Keep checked gains and unresolved evidence in the case record. The workflow
owns these revisions within the existing budget; it does not guarantee that
every source is feasible or authorize a new allocation after a budget stop.

Infer Dex3 contact regions from the source video and reference human motion:
map the robot thumb to the human thumb, and the two robot long fingers
collectively to the other four human fingers. Do not impose a one-to-one
long-finger mapping or a finger-pad-only restriction. Choose eligible original
hand and object surface regions per grasp/support phase, including finger sides
or backs when supported by the observed arrangement. Record occluded 3D choices
as hypotheses and distinguish proposed regions from the discrete material
anchors used by a solver. See [grasp editing](references/grasp-editing.md).

When the HOI handoff includes a contact map, first reuse its bound object-side
regions and phase evidence as priors. Verify original topology, vertex order,
scale and source timing; preserve unknown phases and sliding/support roles.
Remap the human-side anatomy to native robot surfaces rather than copying human
vertex IDs. Record revisions to inherited regions explicitly. See
[upstream object-region reuse](../../../tools/grasp/README.md#reuse-upstream-object-contact-regions).

For a grasp that needs contact across multiple digits, use the bank's explicit
[multi-point contact patches](../../../tools/grasp/README.md#multi-point-contact-patches)
and declare the participating members within each source-derived group. Pooling
the two long fingers must not let one nearest point silently stand in for both.
Review spatial coverage and gap shortfalls per member; dense measured proximity,
eligible regions and frozen material targets remain different evidence.

When the user authorizes correcting inconsistent grasps in generated video, use
the [generated-source correction policy](references/grasp-editing.md#correct-inconsistent-generated-video-grasps).
Preserve the intended action and object region while authoring a feasible robot
grasp and arm trajectory. Exact source hand/wrist motion is a revisable prior
within that scope; record its deviation separately from contact, collision and
temporal acceptance. Carry this policy into remote case prompts explicitly.

Preserve reference human **body posture** separately from unreliable generated
finger/wrist details. When object-trajectory correction is authorized, follow
[object adjustment with body fidelity](references/grasp-editing.md#adjust-object-motion-while-preserving-reference-body-posture):
try a shared object path with original-reference guidance before compensating with pronounced torso lean,
twist or displaced elbows. Keep the original human evidence and robot posture
reference fixed across iterations. A low contact gap does not excuse body-pose
regression. Existing object-edit permission persists; carry it, any explicit bounds and
the protected body/support constraints into case plans and remote prompts.
Object correction means a small deviation from the original reference motion,
not a freely chosen path. Use positive original-trajectory position/orientation
objectives at matching source times; hard displacement and correction-rate caps
are optional, not required by this policy. Preserve key orientations and event
timing, and measure total deviation from the same original trajectory across all
rounds. Soft costs do not guarantee a small deviation, so review the actual
errors; retain any hard limits explicitly configured for that trial.
For SOMA joint initialization, a moving-source object's precontact interval
shares one adjustable SE(3) pose: all those frames may translate/rotate together
while remaining exactly stationary. The original resting pose only initializes
that common variable; positive original-reference costs softly retain it. A
finite soft stationarity penalty alone would not guarantee a static interval.
Preserving each upstream sample also fails to enforce stationarity if it drifts.
Declare the source contact boundary separately from the attraction ramp. The
first joint sweep optimizes that common prefix pose and the independent object
poses from the boundary onward. Contact gaps are diagnostics, never an unlock
or export gate. The converter deterministically preserves an exactly static
full-source object trajectory without optimization. These two policies are code,
not per-case agent decisions. See the
[SOMA object-motion option](../../../tools/soma_retargeter/README.md#optional-object-motion-inside-the-soma-converter)
for the experimental converter route and its original-reference/export contracts.
For a body-fidelity correction, configure the sequence/transition
[`posture_reference`](../../../tools/grasp/README.md#keep-the-original-native-posture-as-a-separate-reference)
with explicit named-joint deviation limits. This enforces the declared native
posture corridor; human-body alignment still needs independent source review.

Human/object reconstruction and depth calibration remain upstream, following the [upstream handoff](../../../docs/upstream-handoff.md). For initial conversion, select the source model explicitly: **SMPL-X → GMR; SOMA → NVIDIA soma-retargeter**, using the [unified retarget tools](../../../tools/retarget/README.md). Heavy runtimes remain external. Do not rerun conversion as an incidental step in an existing robot-hand edit. Initial conversion alone does not establish contact-aware refinement or artist review.

The SOMA adapter accepts the official SOMA-base BVH contract; raw SOMA parameter
checkpoints require a matching model/identity/rest-pose export first. Do not apply
the custom SMPL-X convention to SOMA or mix legacy MHR-backed SOMA with native
SOMA03. Bind BVH frame timing, units, facing convention and original robot model.
For metric reconstructed HOI, the SOMA adapter explicitly defaults
`human_scale_ratio` to `1.0` and records the effective scaler. A different ratio
must be declared in the case config; changing scale invalidates prior contact
and model/trajectory binding evidence. Old conversions are not silently relabeled
as having used this policy.
Bundled SOMA `unitree_g1` output is body-only (29 DOFs). For G1 + Dex3 use the
explicit `unitree_g1_dex3` custom registry and `extra_robot_paths`; the
[target generator](../../../tools/soma_retargeter/README.md#g1--dex3-through-the-soma-solver)
calibrates against actual native neutral FK and solves all 43 hinges with SOMA.
BVH offsets alone omit neutral rotations. Never pad a body-only result or invent
missing finger motion. Bind original object motion separately before HOI contact edits;
both backends then use the same native Blender refinement and review loop.

For SOMA hand-object initialization, the optional `contact_constraints` config
adds authored material positions (with optional normal alignment) to the same Newton IK solve
([contract and limitations](../../../tools/soma_retargeter/README.md#optional-hand-object-constraints-inside-soma-ik)).
The current chosen SOMA initializer is the
[all-frame synchronous mode](../../../docs/soma-synchronous-sequence.md), with
CUDA sequential forward IK with bounded-memory LM, CUDA Graph residual/Jacobian
evaluation and CPU sparse global solving. The forward device follows the
synchronous evaluator by default; an explicit CPU device remains available,
and an explicitly serial evaluator defaults the forward stage to CPU. The
forward CUDA adapter supports at most 64 native tangent DOFs and fails rather
than silently falling back on an unsupported runtime or device. Preserve the
original objectives, warm starts, iteration budgets and geometry when comparing
devices. Separate first-process compilation/capture time from warm execution;
speed and execution checks do not accept the resulting motion.
Use this for new SOMA agentic initializations; the comparison with an alternative
two-stage adapter did not select that adapter as the default. For objects whose
full original translation and rotation are exactly constant, omit robot-object
contact position and normal losses in both initialization stages. Keep original
object poses, collision/source/temporal terms, and diagnostic contact evidence.
A stationary prefix of a moving object is not a static case. Static cases do
not need authored anchors and may provide `contacts: []`; the contact bridge
still carries the object geometry and configured collision/temporal objectives.
For the current hand-target initializer, follow the
[packaged workflow](../../../docs/current-retargeting-workflow.md): Topo-inspired
candidate/sequence search, explicit selected-hand export, then fresh SOMA
forward and global stages. Free-hand wrists/fingers are soft targets, not
already mounted robot poses. Material anchors are optional and separately
validated; no eligible anchors means a recorded pose-only case, not fabricated
contact evidence. The older anchor-first route below remains available for
explicit material-contact experiments.
When anchor-based contact initialization is requested, construct and inspect the native
hand/object hypotheses before this solve, then explicitly configure both
`contact_constraints` and its `collision_avoidance` section. Follow the
[candidate-first initialization route](../../../tools/soma_retargeter/README.md#candidate-first-initialization).
Retain SOMA's human-reference objectives in that same solve. A preliminary raw
pose used to seed contact search is preparation, not the requested initialized
result; disclose that dependency and preserve any from-scratch restrictions.
Missing candidates return to contact authoring rather than silently dropping the
requested objectives. Verify the effective contact/collision reports before
calling the output contact-aware, and compare actual mapped human targets as
well as contact, geometry and temporal behavior.
Identify feasible robot contact layouts at source-supported keyframes first;
human contact regions or nearest surface points alone do not establish that
layout. Keep anchors fixed within a declared hold and author changed/sliding
relations as separate phases. Review the post-clamp contact measurements and
whole-hand geometry before selecting this initializer. This mode explicitly
disables upstream limb postprocessing to avoid wrist overwrite (including its
foot stabilization); full-body IK variables remain active. It supplies a contact
aware initialization hypothesis, not a collision-free or accepted sequence.
When fitting these contacts, enable the companion `collision_avoidance` section
with explicit original mesh IDs and self pairs covering the relevant fingers,
palm and wrist (plus arm/body pairs where needed). Its sampled clearance loss
runs in the same CPU/CUDA solve on approach and holding frames. Review reported
omissions and retain independent original-mesh/subframe checks: winding signs on
open meshes and finite samples are not a hard nonpenetration guarantee. Keep
targets and coverage fixed for each comparison and reject forbidden crossings;
do not accept an initializer just because the combined loss fell.

When jitter must be addressed during SOMA initialization, use the
[same-objective temporal option](references/temporal-editing.md#smooth-during-soma-initialization)
before export, retaining the original human targets, contacts and collision
terms. Review the saved same-run initial and refined trajectories; the upstream
joint-limit filter alone does not smooth time, and soft temporal costs do not
replace native speed, acceleration, jerk or geometry checks.
When authorized object changes accompany this solve, explicitly configure
`object_optimization` with `temporal_optimization`. Alternate object
updates and the same human-guided robot solve inside the converter; do not
describe this as adding an object articulation to Newton. Keep both hands and
collision transforms bound to the resulting shared path, and export that exact
report-bound path. Check object/table geometry, support, original-path deviation
and actual object derivatives separately from robot smoothness.

The [native refinement package](../../../src/__main__.py) is invoked with `python -m agentic_retargeter`: `fit-robot` creates bounded native candidates, while `init`, `verify`, `register`, `select` and `page` bind provenance, reviews and comparison media. Preserve imported human records as lineage; this package does not fit human motion. The [Blender extension](../../../extensions/blender/README.md) and [contact edit kit](../../../tools/contact_edit/README.md) provide the authoring tools. Tool success does not select a motion.

Plan and review with the host assistant's available tools; this workflow makes no remote model API calls. Existing credentials do not authorize a model call. Contact suggestions remain hypotheses and do not replace native or visual checks.

For native key edits, use the [local Blender editing loop](../../../docs/blender-agent.md). Bind the checkpoint, editable joints/windows, native visual model and multiple cameras. Use installed Blender MCP when useful, or foreground `blender -b --python`; the local `blender-edit` command performs a bounded round without model calls. This repository does not bundle a live MCP client. Maintain versioned `.blend` files as the editable state, continuing from the latest chosen checkpoint and saving new rounds beside their native exports and review evidence. Preserve native curves and scene bindings across rounds. Reopen and compare the saved scene with the exported native motion; candidates remain unselected until the corresponding motion checks pass.

Reuse verified toolchain behavior across examples. Run relevant regression tests when editing/interpolation/export code, runtime versions or supported input/model contracts change, or an observed inconsistency needs diagnosis. A new example needs input binding and its own motion checks, not another full toolchain experiment. Keep the normal save/export/reopen comparison for a delivered editable scene.

For repeated cases, write an explicit stage plan and use the [timed batch
runner](../../../tools/batch/README.md) to execute already specified commands
without a new assistant round between them. Record setup/development separately
from case execution, solver, audit and rendering. Batch alternative contact
hypotheses against the same difficult poses and short transitions. A failed or
missing gate stops that acceptance path and returns the case to its owning
agent; independent cases continue. A declared coupled repair may use a rejected
mount as diagnostic initialization, preserving its rejection and fixed parent.
It does not inherit acceptance from screening. The runner supplies orchestration and timing; it does not
generate grasps, fill missing checks or accept motions. Preserve source caches
only with their hashes and disclose them when reporting a from-scratch rerun.

For a reviewed repeated recipe, execute the fixed stage plan with `tools/batch/run.py`
and use `--resume` after interruptions. Bind stage inputs/outputs, retain the grasp
and short-transition gates, and have the owning agent inspect `execution.json`,
any `agent_request.json`, numerical evidence and images at decision boundaries.
Retain visual review after successful batches too. Do not re-explore completed choices between deterministic
commands. Use the shared audit/repair `--cache` for durable per-sample evidence;
an unchanged raw baseline should not be rescanned in every attempt. Cache failures
as well as passes, and recompute changed poses/interpolation joins. Check coverage,
invariants and motion derivatives for every current candidate. A diagnostic raw
audit may finish with failed checks; it never authorizes propagation as accepted.

Use hardware-required cached rendering (`require_hardware: true`, initially one
worker), render affected windows during iteration, then the complete trajectory
after its required gates. Record the actual graphics device. A failed hardware
guard requires diagnosing the render environment rather than silently claiming
GPU acceleration from a software fallback. Preserve the same solver objectives,
phase structure, audit density and source/visual review when measuring savings.

## Establish the input and edit scope

1. Read the selected checkpoint, decision, source video and execution state. Resolve the native model, joint order, original object mesh/transforms, camera, source timing and fallback. Selection follows recorded lineage and hashes, not the newest filename.
2. State the visible defect, per-hand roles and allowed native variables. Separate RGB observations, estimated reconstruction/depth and authored contact hypotheses. A proposed 3D contact is not a predicted ground-truth point.
3. Preserve source timing, camera, original meshes/scales, human shape/fingers, root orientation and observed object-stationary phases unless the user changes that scope. Robot fingers are editable within the original robot joint limits. Declare source-supported root/trajectory trials separately from finger/arm edits.
4. Reuse the frozen stage-specific source/depth targets and calibration with their hashes; never recalibrate on a candidate foreground. Inspect live agents, tracked executions and GPU ownership before a run; avoid duplicate work and serialize GPU use. Do not revive stopped schedules.

Read [native inputs and effective retarget constraints](references/inputs.md) when importing an upstream conversion or investigating an unexplained target/pose mismatch.

GRAIL handoffs may use the custom **G1-proportioned SMPL-X**. Preserve the handoff's bound human model when building human FK/IK targets and comparisons; the robot itself still uses its original native G1 model. Check the [custom-model handoff](references/inputs.md#custom-smpl-x-handoff) before selecting a human model or applying scale/proportion conversion. A legacy standard checkpoint or an existing robot correction does not authorize silently replacing its upstream human geometry.

For reusable pose or curve code, read the [contact edit kit](../../../tools/contact_edit/README.md) before writing another case-specific solver or interpolator. It provides fixed original-triangle contact fitting over named native hinges and Blender Graph Editor curves with protected windows and native export checks. Author each case's contacts, palm pose and phase keys explicitly. The kit has no built-in original-mesh collision or source-image validator; its outputs remain unreviewed, and its bounded pose solve does not replace the outer editing loop below. Its optional visual preview supports compatible portable G1 scenes; it is not a general rig builder or human-pose solver.

For contact, intersection or transition repair, use the [shared audit and local repair tools](../../../tools/repair/README.md) before creating case-specific audit/solver scripts. Bind the parent, original geometry and explicit mesh pairs, fixed material targets, editable hinges/window and protected frames; include relevant palm, wrist, arm, auxiliary-digit and environment geometry from the first candidate. The shared tool audits finite samples, couples short-window joint edits and rechecks the resulting native motion. Its README owns the configuration schema and check limitations. Source observations and temporal gates require explicit configuration; absent checks are not passes. Reuse unchanged evidence only within its bound scope, and recheck affected interpolation intervals and joins after edits.

Configure the [native joint-speed hard gate](references/temporal-editing.md#require-a-native-joint-speed-hard-gate)
before fitting and before accepting a full sequence. It must cover every native
hinge and consecutive frame interval, including fingers, arms, waist, protected
holds and editable/frozen joins. A local improvement does not exempt inherited
speed violations elsewhere; retain a failed full-motion status until repaired
or explicitly deliver the scoped partial. Use the named G1 + Dex3 animation
preset as an initial policy, not as a hardware velocity rating. A smoothness
loss or improved wrist displacement does not replace this gate.

Contact-normal alignment is diagnostic only under the current user-selected
workflow. Set each contact `normal_weight: 0`, use `max_normal_deg: 180` for
shared audits, and omit `contact_normal_policy`. Also set `solver.normal_weight: 0`
in shared repair configs. Pose/SOMA tools default to zero per-contact normal
weight; shared audits default to the unrestricted 180-degree range. When
continuing a legacy config, explicitly revise its weights, angle gates and
normal policy in a new config; defaults do not override historical settings.
Do not reject, rank progress or trigger another repair solely for a contact
normal error. Retain geometry normals used by collision/clearance algorithms,
optional diagnostic arrows and object-relative palm pose constraints. Restore a
contact-normal objective or gate only on an explicit task request.

When the user permits brief contact excursions, use the shared tool's
[contact tolerance profile](../../../tools/repair/README.md#brief-contact-excursions):
15 mm nominal gap / 40 mm cap, at most 5% outlier duration and 0.2 seconds per
continuous gap excursion for each contact. Record the policy in the case config;
it is the selected gap profile for the visually reviewed mocap cases, not a
universal grasp standard. Reclassify saved complete measurements before deciding
to repair. Preserve maxima as diagnostics and do not restart fitting solely for
a tolerated peak. A persistent deviation is a separate scoped tradeoff; report
it without automatically rerunning a motion the user already selected visually.
Check duration over the actual active phase and complete sampled interval,
including subframes and merged shard boundaries. Other geometry/source/motion
gates retain their own scope.

In Blender, use **GRAIL G1 → Local Window Repair** in the unified extension for the same solver. It snapshots current native curves, owns an external worker, and provides reversible Preview/Restore followed by explicit Apply. Wait for the actual Blender-subframe re-audit before applying; stale inputs or edited preview curves require a new trial. Use its documented LINEAR replay scope and retain rejection/unknown checks. Newton contact views refresh separately after a pose change; an old contact map is not evidence for a repaired pose.

Choose a repair backend explicitly when changing methods: native CPU uses local clearance witnesses; the optional material GPU backend uses fixed material anchors, opt-in normal alignment and sampled signed distances. Keep the same parent, editable scope and audit gates for comparisons, and separate setup/solve/audit time from rendering. Use GPU runtime/device bindings in the [shared tool configuration](../../../tools/repair/README.md#optional-gpu-proposals-and-audit-acceleration). A signed-distance loss, fast static Newton IK or a nearest-contact fit does not establish whole-mesh or trajectory quality; always inspect interpolation and joins. Retain source and wrist/finger speed/acceleration measurements alongside contact improvements.

When continuing from a selected smoother candidate, use the shared tool's [explicit iteration seed](../../../tools/repair/README.md#continue-from-a-selected-smoother-seed). Keep the original parent, correction corridor and source posture prior fixed; bind the chosen working seed separately. Check actual wrist and hinge derivatives across editable/frozen joins. Preserve the chosen seed when a proposal fails; do not loosen smoothness gates or replace the original reference silently. A successful preview/export proves the editing path works, while contact, source and motion acceptance remain separate decisions.

Before the first fit or after changing model/mesh/editable scope, use the shared tool's `coverage` command to enumerate missing mesh pairs and unlisted geometry. This is a lightweight topology preflight, not another toolchain regression or collision pass. Name intentional omissions, retain raw seam evidence, and audit any newly added pairs at the relevant baseline poses. Same-body/parent adjacency and an omission reason are not clearance proofs; an empty relative edit mask does not imply the source geometry is constant over time. The tool reports gaps without silently expanding or accepting the scope.

For Newton contact diagnostics, use the [collision bridge](../../../docs/collision.md)
with explicit original-mesh pairs and the pinned external runtime. The Blender
panel displays paired query points and saves frame/geometry-bound evidence;
refresh after edits. This backend detects contact candidates, not native-joint
corrections. Its unsigned paired-point distance is not penetration depth, and
zero candidates do not replace complete original-triangle and temporal review.

For an existing motion that snaps, jitters or closes too quickly, read [temporal editing](references/temporal-editing.md). Separate closure duration and peak finger speed from acceleration/jerk: a smoother motion can still close too fast. Start from the selected native checkpoint, preserve the reviewed grasp/support milestones, and edit the relevant native controls and interval. A closure-only request usually needs finger timing changes with arms/object motion frozen, rather than a new grasp solve.

Choose wrist constraints per hand and phase using [phase-aware wrist guidance](references/temporal-editing.md#choose-wrist-guidance-per-hand-and-phase). During free approach or after release, default to no extra preservation of the previous robot wrist's world position/orientation and no attraction to future object contacts. Retain human-derived posture guidance, joint limits, scene/self clearance and coupled temporal objectives. This does not disable SOMA's human-reference tracking. During a supported hold, preserve the declared object-relative grasp. Review the acquisition join with its protected hold and neighboring derivative stencils; a selected checkpoint is an initialization, not automatically an end-effector target.

For trajectory edits, use the phase-aware tool route: **Blender/native timing edits → contact-preserving joint solve where a stable hold needs it → exported-native contact, whole-mesh, source and hand-motion review**. Apply only the stages needed for the requested defect. Read [contact-preserving interpolation](references/contact-preserving-interpolation.md) before using the experimental G1 sequence solver or choosing a curve-smoothing method. Contact preservation and animation improvement are separate decisions; Continuous Acceleration remains optional, and the constrained prototype is not a universal default.

## Grasp-first editing loop

For a new grasp or propagation, follow the loop below. For a local correction to a selected motion, reuse reviewed inputs and grasp stages that the edit leaves unchanged; rerun the affected editing and review steps.

Use the [surface-grasp batch recipe](references/grasp-editing.md#batch-surface-grasps-before-trajectory-repair)
to choose candidate surfaces before freezing material points. The
[public grasp tools](../../../tools/grasp/README.md) provide `bank.py` for
batched whole-hand region proposals, `hold.py` for mounted proposals,
`evaluate.py` for common difficult-key/transition screening, and `sequence.py`
for jointly repairing a fixed-material hold and its entry/exit buffers. Supply
the original inputs, topology, regions, seeds and edit bounds in case configs;
do not rebuild those solvers inside each case job. Screening stays unselected.
After independent checks and visual review, use
[`progress.py`](../../../tools/repair/README.md#preserve-reviewed-progress) to
retain useful scoped partials separately from diagnostics and full acceptance.

Before native editing of a new case, analyze the source video and select source
keyframes. Use the [local video tools](../../../tools/video/README.md) when local
models are available: they estimate visible object masks and tracked hand
landmarks, then propose grasp intervals and image-space contact candidates.
Inspect coverage, unknown intervals, handedness and exact source timestamps.
Review initial approach, possible first touch, stable hold, object movement and
contact changes; a late detection must not be relabeled as first physical touch.
Missing detections do not establish release. Reuse reviewed source evidence for
later local corrections rather than rerunning unaffected perception.

During keyframe selection, compare each hand's grasp across keys and the intervening
video. Group source-supported continuous rigid holds under one per-hand grasp
identity, with one shared object-relative palm/wrist pose and finger/contact
layout. Test candidate grasps across the group's difficult keys before selecting
one; do not fit an unrelated grasp at each key. Regrasp, sliding, articulation and
uncertain intervals need distinct relations. Follow
[grasp continuity across source keys](references/grasp-editing.md#group-source-keys-by-grasp-continuity)
for the evidence, shared variables and interpolation checks. The local video
planner proposes evidence windows; this grouping currently needs host review.

Record the selected source frame indices before authoring edits. Use them as
Blender observation frames only after checking the source/native motion mapping.
The video planner's silhouette coordinates are 2D hypotheses, not object-local
3D contacts: author or fit original-surface 3D anchors from the bound object and
camera, then inspect those proposals alongside native measured proximity in the
canonical-hand/contact dashboard. No automatic frame or contact proposal is an
acceptance decision. For initial conversion, the [GMR tool](../../../tools/gmr/README.md)
wraps the configured upstream runtime with source/model binding checks.

Reusable implementation belongs directly in `src/` and thin command wrappers
belong in `tools/`. Keep the existing installed `agentic_retargeter` import/CLI
through package mapping; do not reintroduce a `src/agentic_retargeter/` layer.

When a new object or poor initial grasp needs alternative hand poses, use the
[optional local GraspGenX tool](../../../tools/graspgen/README.md) on demand.
It is disabled in the normal sequence-loading, retargeting and repair paths;
only its explicit `run` command loads the configured local model. Bind the
original native model, object scale, hand side and reference frame. Generated
free-wrist poses and confidence are hypotheses, not mounted IK or contact
validation. Review source-supported candidates, then fit the actual arm/fingers
and declare any changed contact targets before following the loop below. Reuse
an already reviewed grasp for local timing/trajectory work. Fast proposal
generation alone does not establish better contacts or an end-to-end speedup.

1. **Describe the grasp groups and eligible surfaces.** Record each hand's grip/support/slide role, active phases and continuity evidence. Map the G1 opposing digit to the thumb and the two long digits to the remaining human-finger group. Choose source-compatible object regions and original finger inner/side/proximal/distal or palm surfaces; neither three distal pads nor palm contact is mandatory. Keep uncertain support relations explicit. Contact-normal alignment remains disabled.
2. **Batch a small feasible candidate bank.** Declare the seeds, shared geometry, source constraints, editable variables and search budget. Inspect spacing, bending planes, opposition and native limits. Compare whole-hand region fits before freezing the chosen original-surface material anchors; use GPU batches and persistent mesh/FK caches where supported. Screen each candidate's complete relevant geometry and discard confirmed forbidden crossings early. Free-wrist results remain proposals, not mounted-pose acceptance.
3. **Fit fingers and arms together before choosing a grasp.** Use the free-hand pose as an initial hypothesis, then explicitly enable `hold.py`'s [`finger_refinement`](../../../tools/grasp/README.md#joint-arm-and-finger-fitting) to fit its fixed material contacts through actual arm/finger FK, with declared waist freedom when needed. Fit both active hands against one shared object pose; retain authorized object bounds and source-body guidance. Test the first stable hold plus later difficult orientations before full propagation. Compare one versioned contact layout across these keys rather than unrelated per-key contacts. Inspect mounted gaps, whole original meshes, self-collisions and source views, and save the native pose and scoped decision. A failed mount returns to grasp/arm-branch editing; do not defer all finger adaptation to sequence repair. Legacy rigid-finger mounting remains useful for an already reviewed hold whose hand shape is intentionally protected.
4. **Evaluate candidates under one comparison policy.** Run `tools/grasp/evaluate.py` on mounted candidates with common parent, geometry, edit scope and gates. It includes first/last hold keys, the largest object-rotation key, declared difficult frames and dense short transitions. Inspect its failed and unknown gates plus source/diagnostic views; `screening_only` reports cannot promote a checkpoint, even when every sampled key passes. Version changed contact hypotheses separately.
5. **Solve the hold and its entry/exit together.** Use `tools/grasp/sequence.py` when independent mounting leaves slip or discontinuous joins. Preserve fixed material identities and the intended common grasp while allowing declared arm/finger controls through the complete hold and acquisition/release buffers. Keep the first hold sample editable; do not pin a failed first-touch pose. A rejected mount can initialize this explicitly diagnostic repair, with its status retained. Contacts activate only in their declared phase; use separate relations for sliding/articulation. For an already reviewed hold with an approach-only defect, retain the narrower local repair. The sequence tool keeps root, object and scene fixed during that solve. For authorized object edits, first prepare and bind a revised object path shared by parent and seed, preserving the original robot posture prior; alternate path preparation and sequence repair as needed. This solver limitation is not a reason to ignore object-edit permission or widen body corridors until contact fits.
6. **Audit, review and preserve progress.** Independently audit the exact exported native file with original meshes, temporal gates and affected joins; check the full hold's fixed-anchor gap/slip and inspect source-speed playback. Record source deviations under the applicable source policy. Use `tools/repair/progress.py` with the bound audit, declared scope/contact revision and visual receipt to preserve scoped reviewed partials separately from diagnostics. A failed later branch must not erase useful earlier work. Missing checks remain unknown, and a scoped pass does not accept the full motion. If the requested defect remains, execute the next evidence-supported [contact edit](references/grasp-editing.md#diagnose-and-edit-before-solving-again) within the case budget.

When scene geometry is present, read [scene-aware repair](references/scene-aware-repair.md)
for object/table checks, full-arm self clearance, newly acquired hand keys,
protected object joins and actual wrist smoothness. These checks apply to the
case's geometry and phases; they do not require rerunning unchanged toolchain tests.

### Continue the requested correction

For user-requested bounded exploration, use the [stopping-rule design](references/stopping-rules.md)
to distinguish candidate rejection, branch stagnation and case termination.
Record the declared cumulative operational budget and any user-set outer-round limits;
the optional iteration counts do not silently become hard caps. Budget/stagnation stops preserve the best
reviewed partial and unresolved checks rather than weakening acceptance gates.

Record the requested outcome, editable scope and any user-set task budget or deadline before multi-round work. A solve's iteration cap, a job timeout or an internally chosen trial budget bounds that execution, not the whole task. Do not invent a task-wide stopping budget or a fixed minimum number of retries.

After a failed review, return to the contact-edit decision loop while authorized progress remains possible. Separate its outer loop of hypothesis, actual native edit and review from the solver's bounded inner iterations. If the failure persists, execute the next evidence-supported edit or a diagnostic that distinguishes the remaining hypotheses; do not just list them as future work. A failed grasp review prevents adopting or silently propagating it as a reviewed grasp; it does not end repair. Preserve improvements and inspect only changed scope and affected joins.

If local repair reports `affected_scope_passed` while inherited failures remain outside editable support, preserve its overall `REJECTED_BY_AUDIT` result and report the local improvement separately. Stop repeating that unchanged numerical trial; review the partial and choose the next action within the requested scope. A local pass does not accept the complete motion or authorize widening the edit window.

When a bounded trial stalls, distinguish a missed collision sample, a poor local witness direction, a material-contact conflict and a restricted joint/window corridor before changing weights. Preserve successful checkpoints. A known good pose outside the corridor is evidence for considering a separately declared scope revision, not proof that the current problem is infeasible. Additional samples or more optimizer iterations can worsen joins or create self-collisions; compare their measured result rather than assuming greater complexity is better.

Choose editable variables from the defect. For a local correction to an already reviewed finger shape, a mounted arm/palm trial with fingers fixed can preserve that work; for a new grasp, use the joint fitting stage above. Extra freedom can trade contact error for finger penetration. Inspect whether fixed-anchor error is mainly normal separation or tangential displacement, and verify that contact points lie on the original finger surfaces. Nearby nearest-surface contact does not justify silently moving a fixed target. Test the last difficult hold orientation as well as the initial grasp: a useful static pose may still reach native joint limits or the torso later.

Try the original pose before an explicit small restart bank. Keep targets, posture prior, bounds and screening rules common across starts. Additional starts are optional; a lowest-cost rejected fit remains diagnostic. Palm transport can guide a coupled trajectory, but it does not replace fixed material contacts, interpolation checks or source review.

Keep checkpoint quality separate from task status. Publishing a PARTIAL pose, full-length video or editable Blender file is progress, not completion of an unresolved requested correction. Continue within existing authorization without asking again. End the task when the requested outcome has been reviewed and delivered, the user ends or explicitly changes the scope, a user-set task limit is reached, or a documented prerequisite/resource/permission constraint prevents meaningful authorized progress after safe recovery or alternatives have been considered. Report the concrete stop reason and unmet outcome; a failed fit alone is not proof of infeasibility. A completed local edit need not resolve unrelated inherited defects. Never revive stopped schedules or promise unattended continuation without an authorized tracked execution.

Read [grasp descriptors, propagation and Blender controls](references/grasp-editing.md) before authoring or propagating a grasp. Use [robot review and delivery](references/review-delivery.md) for selection.

Use installed Blender MCP or Blender Python/CLI as available, and report what actually ran. Edit verified native controls rather than deforming display meshes. Retain the hand/contact dashboard for inspection and refresh it after changes. Save a new `.blend` per round; reopening an earlier baseline must not silently discard accumulated edits. Reopen and test the native round trip when delivering an editable scene. An exported change gets a new review status.

## Accelerate execution without changing the repair problem

For an execution-speed comparison, preserve the reviewed phase structure, grasp
roles, material targets, difficult keys, active variables, objective weights,
correction bounds and verification thresholds. Batch commands between decision
points, reuse immutable source/model caches, reject a proposal as soon as an
original-mesh crossing is confirmed, and re-audit only changed interpolation
support plus its joins. A rejected candidate does not need its remaining pair
checks; a selected sequence needs complete declared coverage and visual review.
Do not replace the phase plan with uniform keys, omit first touch or auxiliary
digit paths, or open fixed fingers to obtain a shorter runtime. Start poses from
the requested raw G1 input; prior optimized motion is comparison evidence only.
Record reused contact/planning evidence separately from fresh pose computation,
and report compute time separately from orchestration, inspection and rendering.
Record the actual initialization dependency chain and effective solver arguments,
including nondefault margins, weights and witness coverage. A named final grasp
config may inherit a precursor fit; solving that config directly from raw is a
different initialization. Reconstruct missing dependencies before labeling a
comparison execution-only. Keep rejected exploratory branches out of the selected
trajectory and include their time in the development/diagnostic total.

## Keep decisions scoped

- **Accepted for the stated scope:** identify the local correction and its checked evidence; retain inherited issues.
- **Partial:** useful authored change with remaining contact, geometry, source or temporal limitations. A user may select such an edit without making it a validated full grasp or sequence.
- **Rejected:** retain its checkpoint and failure evidence; do not blend it into a selected motion silently.
- **Unreviewed/diagnostic:** isolated hands, new keys or exported edits awaiting the corresponding native checks.

Match delivery to the request: a static-frame request gets the actual mounted native frame and useful views; a sequence request needs full-duration playback and temporal review. Neither visual preference nor zero sampled intersections proves force closure, continuous collision clearance or physical stability.
