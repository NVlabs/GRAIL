# Robot review, selection and delivery

Freeze a comparison baseline, target definitions and tolerances before each trial. Use tolerances appropriate to the source uncertainty, mesh and task; do not import a successful clip's millimeters, frame boundaries or sample count as universal gates.

## Evidence for a selected native motion

| Dimension | Check and report |
|---|---|
| Depth, where supplied by the handoff | Reuse fixed reviewed rays/targets and calibration for changed robot poses. Report per-region camera-z error/bias, common-pixel error and total/covered/missing rays separately. Name actual hit links where another body part can occupy a hand region; missing rays cannot vanish from the denominator. |
| Source alignment | Calibrated source overlays; reliable image landmarks, masks, finger direction/placement and object silhouette. Name whether a target comes from RGB, a detector or a partial reconstruction. For authorized generated-hand corrections, distinguish intended action/region preservation from authored hand/arm deviations; the latter alone do not fail the scoped correction. Keep protected source checks. Coverage alone can reward the wrong finger or palm. |
| Body fidelity | Compare human torso/pelvis directions and shoulder/elbow arrangement at matching times, plus robot deviation from the original posture reference. Keep these separate from revised finger/wrist guidance and contact gains. Report peak/RMS deviations, worst frames and declared corridors; a pronounced lean or twist cannot be hidden by a small hand gap. A robot prior or missing source metric is not proof of human alignment. |
| Contact | Fixed hand-surface-to-material-target gap, object-relative slip and active phase. Contact-normal angles are optional diagnostics, excluded from acceptance and progress ranking unless explicitly requested. Report absolute errors and per-pad changes against the original comparison parent at matching times, not only a difference between aggregate maxima. Evaluate gaps and binding independently. |
| Original geometry | Whole relevant original robot/object triangle intersections, with both directions where the chosen method requires them. Include finger/palm self-collisions, body, cross-hand and environment checks where applicable; state omissions. Add occupancy only if the original mesh supports a meaningful inside/outside test. Vertex-only, pad-only or proxy checks miss triangle-interior crossings. |
| Native contracts | Actual model FK and joint limits; frozen fields, camera, scales, meshes, timeline and stationary phases. Derive checks from saved parameters, not screenshots alone. |
| Authorized placement | Record the robot/object or table world translation against the unchanged baseline. Check finite-table support and gap during on-table phases, foot support and all affected collisions; bind the same placement in native data, solver, audit and render. Keep the calibrated camera and report intentional overlay deviations separately under the declared source policy. See [placement and support](scene-aware-repair.md#adjust-placement-while-retaining-table-support). |
| Authorized object trajectory | Report translation/rotation corrections against the original object path, absolute and offset derivatives, protected phases and joins. Both hands must use the same corrected path. Verify intended action/orientation and finite-table support during resting phases, then recheck object/robot/scene collisions. A static placement adjustment does not substitute for this time-varying review. |
| Temporal behavior | Require the configured native joint-speed hard gate over all hinges and all consecutive integer frames, including protected holds and editable/frozen joins; inherited violations still fail full-motion acceptance. Also review subframes of the actual native interpolation, with denser samples around approach, closure, release, curve joins and worst cases. Compare joint and world-space wrist/contact-surface/object velocity, acceleration and jerk at the actual timestep. See [temporal editing](temporal-editing.md#require-a-native-joint-speed-hard-gate). |
| Visual judgment | Source view and at least two useful diagnostic views, close-ups and full body. Check source-speed playback when available; ordered stills alone are a more limited audit and should be labeled as such. |

Revisable generated hand/arm targets do not waive the separately declared body-posture corridors. When an elbow or shoulder constraint conflicts with a stable grasp, revise the shared grasp or use the authorized object-path scope before widening that corridor.

For a configured `posture_reference`, full promotion also requires its passing
gate bound to the exact exported candidate and original reference. The independent
geometry audit and `progress.py` do not compute this posture policy. The progress
tool blocks a candidate tagged `REJECTED_BY_POSTURE_REFERENCE` from full acceptance;
absence of that tag is not evidence of a pass. Preserve the solver's bound gate
report alongside the progress record; a failed or missing gate permits only an
explicit diagnostic/scoped partial, not full promotion.

For edited object motion, enforce the declared small-deviation policy against
the immutable original path at matching timestamps: report translation-norm and
rotation-angle peak/RMS, worst frames, and retained path/orientation/timing
milestones. Include accumulated placement and trajectory corrections. A local
solver offset cap against a revised parent or a lower contact error does not
prove that the original-reference bounds pass.

Report newly introduced collision pairs/intervals separately from totals. Preserve the distinction between known assembly/mount overlaps, inherited defects and new penetrations; a lower total count cannot hide new failures. Mesh-face pair counts are not penetration depth or independent physical collision events.

An existing pair is not an exemption: compare its spatial extent at matching times. If an inherited overlap grows, locate the original intersection points on the affected components and inspect them in a useful view. Distinguish an assembly seam from intersections extending along a finger or limb; intersection-point spread is not penetration depth. Screen the remaining trajectory early when repairing local windows, and reuse an earlier geometry result only when the relevant native state, object transforms, original meshes and evaluator match exactly.

Check whether occupancy is defined reliably for the original mesh. Open pans and other open/nonmanifold surfaces can make inside/outside ambiguous; do not run undefined solid occupancy or remesh/close the original surface to manufacture this evidence. If it is inapplicable, report that limitation and the justified alternative evidence; do not convert an unavailable check into a zero-collision result or repair/replace original geometry silently.

A local collision correction can clear one thumb/handle intersection while introducing a long-finger/neck intersection elsewhere. Recheck all original object components and new self-collision pairs after each retained edit; a zero pad penalty does not substitute for these checks.

Finite subframe sampling supports only the checked states. It is not a continuous collision proof. Low contact drift does not prove small contact gaps; a rigid relation can stably hold the hand in the wrong place. Lower objective loss or numerical convergence is not an acceptance decision.

For shared-grasp trials, report per-hand object-relative wrist/palm translation
and rotation separately from physical material-contact gap. Specify whether a
position value is RMS, range diameter or maximum deviation from the shared pose,
and which phase and timestamps it covers. Report object translation/rotation
offsets from the original trajectory and their temporal changes alongside any
slip improvement. Keep a contact-definition revision separate from a matched
motion comparison. A local approach repair must also demonstrate that protected
hold keys/targets were retained and that joins were rechecked.

For a contact-constrained interpolation trial, independently evaluate times between the fitting samples and both sides of edited joins. If a violation is added to the next fit, mark it as a fitted time and retain separate off-grid checks for the revised candidate. Keep each version's code/config/checkpoint/review bindings distinct. A passing material constraint cannot waive auxiliary-digit collisions, inherited-overlap growth, protected source-ray identity loss or worsening world/object-relative hand motion. Apply source-error gates under the declared source policy; intentional generated-hand corrections remain reported deviations. Record tool correctness, preservation, smoothness and overall selection separately; a successful method comparison need not adopt a motion. Read the [interpolation procedure](contact-preserving-interpolation.md) for scope and failure routing.

Use useful independent reviewers: one can reconstruct native outputs/check geometry while another reviews fixed source and novel-view assets. Give them the exact candidate/baseline, scope, target definitions and evidence. Let them inspect shortcomings; do not ask only for confirmation. Serialize GPU evaluation with fitting/rendering when resources conflict.

## Artifact and decision record

Keep a run directory with a short plan, executed commands/scripts/configs, input/model/code hashes, fixed evaluation definitions, candidate checkpoints (including rejected trials), review evidence and a selected-native pointer. Record effective code overrides, not just the nominal config. Maintain journal/execution state for multi-round work so another turn can resume without repeating it. Historical records and original fallbacks remain intact.

An embedded descriptor or native metadata may still describe an earlier draft. Resolve selection from the explicit reviewed decision, checkpoint hash and lineage; document stale labels instead of silently rewriting history or trusting whichever timestamp is newest. If those records do not resolve the conflict, keep the candidate unreviewed until its provenance is established.

For composed trajectories, bind the arrays actually loaded, not a trial label. Assert retained keys against their intended native states and material descriptors, apply explicit frame/joint masks, reject conflicting overlaps and rebuild FK. Review the resulting joins using their actual endpoints; an earlier subframe check is reusable only if both interpolation endpoints and the interpolation rule are unchanged.

User acceptance of a visual tradeoff can select a scoped partial; record that choice and the actual remaining defect without calling it collision-free or physically validated. The decision must state its scope: e.g. “accepted left-hand anatomical correction; contact remains partial.” Preserve visual tradeoffs even when numerical gates pass. Do not loosen a gate retrospectively to label the existing result successful; make a new controlled comparison if the gate itself is inappropriate. Do not require a completely solved reconstruction before delivering a useful, honestly labeled local correction.

For multi-round editing, record requested-work status separately from artifact review status. Keep the unresolved in-scope defects, last tested hypothesis and actual edit, review result, next justified action, and any task-level stop reason in the execution record. A PARTIAL artifact is not a completion reason. If a feasible in-scope next edit is identified, execute it rather than ending with “next round” or asking for authorization already given. Share concise progress during long edits and deliver useful checkpoints while continuing. If blocked, distinguish a failed individual run from the actual prerequisite that prevents further work, document attempted recovery/alternatives, and state what is needed to resume. Completion of a requested local edit remains valid even when the larger motion is partial.

## Deliverables

Match deliverables to the requested scope rather than requiring every artifact for a static correction:

- For a frame/grasp request: actual mounted native pose, source/full-body and useful hand/novel views, material descriptor and a concise decision. An isolated free-wrist preview is supplemental and labeled accordingly.
- For a sequence request: full source-duration playable comparison and diagnostic views at the original timing, native robot motion, and the checked integer/subframe scope. Keep crops/slow motion supplemental.
- For an editable Blender request: original geometry, verified native controls, usage instructions and a tested reopen/native round trip. Do not claim an editable scene was produced if only native parameters or renders were saved.
- For each selected result: report actual changes, fixed-comparison evidence, inherited/unresolved contact/source/collision/temporal issues and input/model/code/checkpoint/review hashes. Include useful rejected evidence without obscuring which checkpoint was selected.

Where a depth comparison is useful, show it in the fixed calibrated coordinate system; do not re-align each candidate for display. No loss-only, nearest-distance-only, coverage-only, convergence or physical-validation claims.

Use the current environment's supported delivery route and retain successful receipts; never claim an attachment before it succeeds. Do not commit/push or initiate unrelated services as part of packaging. Do not restart old schedules.
