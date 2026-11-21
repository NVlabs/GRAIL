# Phase-aware timing and contact-preserving interpolation

Read this for a robot trajectory edit involving curve smoothing, coordinated arm/hand adjustment or contact drift. For a wrong static grasp, use the [contact-edit decision loop](grasp-editing.md#diagnose-and-edit-before-solving-again) first. Preserving a poor contact does not repair it. A selected partial can support a scoped timing/preservation trial when inherited defects remain explicit.

## Tool route

| Situation | Tool and intended result | Remaining work |
|---|---|---|
| Approach, preshape, closure or release timing | The [contact edit kit](../../../../tools/contact_edit/README.md) provides native hinge controls and Blender Bézier/Auto Clamped curves. Keep `NONE` as the handle-smoothing default; `CONT_ACCEL` is an optional matched comparison. | Inspect every digit's path, actual native export, closure speed and joins. A smooth curve does not provide clearance. |
| Wrong contact, mounted palm or supporting digit path | Author revised material contacts, palm/arm pose or per-digit clearance keys; use the kit's `pose.py` where its contract fits. | Review the mounted pose and iterate the hypothesis. Repeated solves of unchanged assumptions are not artist edit rounds. |
| Contact, original-mesh intersection or joint-transition defects within a short interval | Start with [shared audit and local repair](../../../../tools/repair/README.md): fixed material anchors, explicit mesh pairs, protected frames and a coupled native solve with audit-derived clearance witnesses. | Review source alignment, hand motion and joins; configure the necessary source/temporal gates. A pass covers only declared geometry and sampled poses. |
| Stable hold/support whose coordinated interpolation drifts from contacts | Use the shared coupled solver with phase-bound material positions and normal alignment disabled. The [historical experimental G1 solver](../../../../src/contact_edit/experimental/README.md) also enforces normal inequalities and is not compatible with this policy unchanged. | Whole original meshes, source rays, auxiliary digits and hand-motion quality remain external checks. |
| Candidate selection | Use [robot review](review-delivery.md): fixed material targets, original-mesh intersections, fixed source/depth identities, native invariants and temporal/visual evidence together. | A successful contact constraint or lower joint jerk is not overall motion acceptance. |

Blender authoring and a constrained native solve are optional stages selected by the defect, not mandatory operations on every sequence. Reuse a reviewed grasp/hold for a closure-only request. Do not widen frozen arm/object variables merely because a tool exposes them.

For free approach and release, follow [phase-aware wrist guidance](temporal-editing.md#choose-wrist-guidance-per-hand-and-phase). Contact preservation inside a hold does not justify locking the old robot wrist pose or attracting the hand to the object before contact.

## Define the comparison before fitting

1. Bind the selected parent, upstream evidence, original model/meshes, native controls, camera, timing, descriptor and frozen evaluation targets. Keep the untouched fallback. Declare the edited interval and affected joins, protected keys, native joint limits and any narrower correction corridor.
2. Identify each hand's active contact/support phases and auxiliary digits. Carry the [source-reviewed grasp groups](grasp-editing.md#group-source-keys-by-grasp-continuity) into the trajectory: keys in one rigid-hold group share the same object-relative palm/wrist pose and hand/contact layout. Keep fixed original-surface anchors, pad identities and normal conventions; distinguish rigid holding, articulation and permitted sliding. Contact attraction/preservation must follow the declared phase. Inactive digits still need a path and whole-mesh review.
3. State the requested temporal improvement: closure duration/speed, a join discontinuity, or joint and FK hand smoothness. Measure RMS and peaks in the relevant world/object-relative coordinates, with explicit timestep and component versus vector definitions. Smoother joint curves alone are insufficient.
4. Declare per-contact gap allowances against the same original parent at the same times, using source uncertainty and task scope. Keep absolute contact errors visible. Normal-angle diagnostics do not reject a candidate or trigger refitting under the current workflow. Do not accumulate allowances by resetting the reference after each accepted step or move targets to hide drift. When testing a tool improvement, keep keys, targets, weights and correction bounds matched unless a separate change is explicitly declared.

## Execute the appropriate native edit

Author real native/Blender controls, then validate the exported motion using the interpolation consumed downstream. Do not smooth each link transform independently of articulated FK. Derive wrist/pad targets consistently: rigid fingers can share a wrist transform; articulated fingers need actual FK from the interpolated finger coordinates.

Equal object-relative poses at two keys do not imply equal poses between them:
independent joint splines can still make the hand slide. Reconstruct the shared
grasp target from the matching-time object transform, solve/reproject native
joints where needed, and measure actual object-relative FK at independent
subframes. If the backend only fits pad distances/normals, report that scope and
measure palm pose drift separately; do not claim it enforces full rigid-pose
equality. Keep grasp grouping in planning records unless the selected backend's
schema explicitly supports it.

For a local repair, use the shared tool's **audit → coupled window repair → audit → visual review** route before writing a case-specific solver. Bind original geometry and the complete relevant pair list before fitting; preserve target identities and protected poses. Use a suitable reference pose for collision-witness side selection; an intersecting or incomplete reference cannot establish that side. Keep the authoritative schema and sampling/geometry limitations in the [tool README](../../../../tools/repair/README.md). The tool exports a new native candidate; it does not edit or select a Blender checkpoint automatically.

Test difficult short transitions before full propagation. Reuse exact unchanged sampled evidence within a bound audit context and keep the surrounding joins in scope. If only inherited failures outside editable support remain, the local improvement may pass while the complete configured audit remains rejected; preserve both conclusions. Neither cached evidence nor a local numerical pass replaces the required final sequence review.

Once a hold passes, protect its native keys, shared grasp and material descriptor
while repairing approach/closure. Include both neighboring interpolation
endpoints in the affected scope; unchanged integer keys do not make their
changed adjacent intervals reusable. Verify protected arrays after composition.
If an object/waist edit changes a supposedly frozen hand in world or object
coordinates, invalidate that hand's affected evidence too. Expand a local repair
only for a declared need within the authorized scope; do not restart the whole
grasp search to fix a swept-finger collision confined to approach.

The current prototype requires stable contact identities/roles over its configured interval **and the reviewed neighboring frames**. Choose a compatible interval or explicitly extend the phase model; do not flatten a support change, sliding patch or contact activation merely to make the input pass. Protect phase milestones and review the resulting joins. G1 pad/joint schemas and ±parent corridors must be resolved from the actual model and selected state.

Run the implementation's preflight before fitting. For a changed implementation, independently check residual/constraint Jacobians, native interpolation and material-metric agreement, protected/frozen masks, parent feasibility and rejection of a known violating state where available. Match the evaluator's coordinate/normal convention, including saved rotation precision; a similar world-distance metric is not automatically identical to object-local material error.

The solver accepts a proposed step only after actual nonlinear FK satisfies the declared constraints and the objective-decrease condition. Version 3 includes bounded second-order feasibility corrections for curved contact boundaries; QP feasibility alone is insufficient. Save accepted checkpoints, rejected proposals/corrections, code/config/input hashes and effective solver diagnostics. Export a new unreviewed native archive and refresh/invalidate derived FK caches. Preserve original metadata provenance and use separate fields for new wrapper/descriptor identities.

An iteration cap or a tiny accepted step is not task completion. Diagnose remaining descent, QP residuals, gradients, bound activity and nonlinear feasibility before extending the same execution. Correct a demonstrated numerical failure, or return to contact/palm/path editing if the hypothesis is wrong. Do not loosen the contact gate to make a saved candidate pass. Respect user-set task limits and the [skill's scoped stopping rules](../SKILL.md#continue-the-requested-correction).

## Review and route failures

Evaluate source frames, changed joins and independently selected times between fitting samples. Refine near worst cases; once a time is added to the fit it is no longer held out for that version. Finite samples never establish continuous-time contact or collision guarantees. Hash-bind each candidate's fitting grid and independent review coverage.

| Failure after export | Next supported action |
|---|---|
| Matching-time contact drift between fitted samples | Check native interpolation/metric parity, add justified constraint times and repeat the same problem with separate off-grid checks. |
| Small/stable pad gaps but wrong grasp or palm placement | Return to contact definition and mounted palm/arm authoring; preserving the parent more precisely cannot fix its inherited error. |
| New object/self intersections or growing inherited overlaps | Locate the original mesh components and revise palm, active-hand transition or auxiliary-digit clearance keys as indicated. Contact inequalities are not collision constraints. |
| Lost own-hand rays, wrong-body hits or worse fixed common-depth error | Inspect source alignment against the frozen targets; do not recalibrate the candidate foreground or accept gained coverage as compensation. |
| Joint jerk improves but hand motion, peaks or closure speed worsen | Inspect the relevant coordinate space and interval, then revise the temporal objective, keys or coupled controls within scope. A new objective is a new declared trial. |

Keep implementation correctness, contact preservation, requested temporal benefit and overall selection as separate conclusions. Report unsupported source checks explicitly; for example, absent fixed depth in an edited interval cannot count as a pass. Preserve all original geometry and check inactive digits as well as active pads. Use full-duration source-speed comparisons for sequence requests and state when visual review used only ordered stills.
