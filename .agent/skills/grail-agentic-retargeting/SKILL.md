---
name: grail-agentic-retargeting
description: Retarget a reviewed native SOMA03 reconstruction and refine robot grasps, support, timing and trajectory continuity.
---

# Robot retargeting

1. Bind the native source pose, identity, source clock, original CAD and support. Use the [case YAML](../../../docs/agentic_retargeting.md) to select the target robot and run plan, preflight and conversion.
2. Inspect the initial pose and approach, contact and release frames. Use the [native grasp tools](../../../imports/agentic-retargeter/docs/hand-inspection.md) to edit the robot's actual grasp and check object-frame material targets.
3. Check support against the registered floor and table. The [optional stance solver](../../../docs/source/retargeting.md#optional-stance-floor-correction) writes a separate candidate.
4. Check full-duration continuity and original-mesh clearance, save and reopen the native scene, then bind the selected result to its reviewed source for the [handoff](../../../docs/source/retargeting.md#hand-off-a-retargeted-motion).
