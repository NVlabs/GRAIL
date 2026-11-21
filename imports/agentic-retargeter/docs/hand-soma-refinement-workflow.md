# Hand trajectory, synchronous SOMA and native collision feedback

This is a tested research workflow, not an automatic motion-acceptance policy. Keep an immutable initial trajectory and exact input/model hashes. Agent decisions choose and document the next bounded experiment; a lower loss or successful process exit does not select a motion automatically.

The [current workflow](current-retargeting-workflow.md) owns the selected
support-block setting, packaged Topo/hand-target entrypoints and one-case,
one-GPU job policy. In first-frame-preserving refinement, keep frame zero out
of editable key intervals and verify the exported pose against its declared
reference. A soft global target alone does not guarantee exact preservation.

## Initialization

1. Bind the original source clock, robot model, object geometry, scene and contact-phase/region annotations. Retain separate thumb and both long-finger coverage checks. A static object has no material/normal contact loss; collision terms remain enabled.
2. Optimize the complete hand sequence, preserving source timing. Use world coordinates outside contact and object-relative coordinates during contact, with explicit transitions, native joint limits and phase-aware trajectory terms. Evaluate source adherence, contact acquisition, holding, support and sliding separately. A denser temporal representation may recover contact detail lost by a very coarse spline. Do not select a smooth sequence by jerk alone.
3. Package the exact chosen hand result and available sparse material anchors with provenance. If re-extracting anchors changes their identity, disclose this: the downstream experiment is not a pure trajectory-weight ablation. Never invent anchors for missing coverage.
4. Run fresh whole-body forward initialization and synchronous trajectory optimization. The current GPU implementation evaluates residuals/Jacobians and forward small systems on CUDA; full-sequence sparse LSMR is on CPU. Preserve source targets, original geometry and object policies. Record actual accepted iteration counts, cache state, compilation, solver time and complete allocation separately.

## Diagnosis and native repair

Review the actual final native trajectory, not a hand-only proxy or an earlier result. Inspect world and object-frame trajectories in physical time, relative SO(3) increments, hand/arm speeds, amplitude and source/body proxies. Check original mesh intersections at every source frame and at additional, independently chosen fractional times near edits. Report unresolved geometry queries. Open surfaces do not establish physical penetration depth; preserve exactly zero-area original facets and account for them as segments/points instead of silently dropping them. Near-degenerate facets with nonzero area remain unresolved when the triangle query cannot certify them.

Choose a small, explicit joint/time support around the collision. Author a smooth native correction while retaining native limits and the original scene, root and object policy. Search can use simpler differentiable witnesses, but final screening must use original geometry, unobserved fractional times, body/source corridors and all declared no-regression checks. A failed candidate may seed a new declared attempt; it must never be silently promoted.

Save the authored native Blender scene, reopen it, and re-export the actual evaluated joint/object state. Extract keyframes from that reopened result. Bind all evidence by hashes. Guard intervals around an edit should include the actual unchanged parent state, including the return to the original trajectory. Fractional collisions can occur in that return even when integer keys are clear.

## Feed back and recheck

Use `prior_global_seed` to bind the exact previous full-sequence state while retaining fresh forward/source-target evidence. Keyframe losses remain additional objectives; they are not hard safety constraints. A global optimization can move an unkeyed root or object enough to create a new table intersection even if all keyed hinges barely move.

When a repair should be local, explicitly declare `refinement_scope` native hinges and half-open frame intervals. All original residual rows and cross-boundary temporal stencils remain. Only the declared columns participate in the sparse solve; roots, objects and unselected native coordinates remain byte exact. See [soma-refinement-scope.md](soma-refinement-scope.md). This is constrained trajectory refinement, not unrestricted full-body convergence.

Re-evaluate the actual exported result. Check native key errors, fixed-state equality, original objective/source bindings, original mesh intersections, held-out fractional times and contact/source/speed regressions. If an edit is locally successful, state the exact target pairs and sampled interval. Inherited collisions, speed overshoots and contact losses elsewhere still prevent whole-motion acceptance. Preserve the previous selected result and failed feedback outputs.
