---
name: grail-agentic-hoi-recon
description: Fit SOMA03 human and object motion to video, review native placement, and refine phase-specific contacts before robot retargeting.
---

# Native human–object reconstruction

1. Bind the original video, clock, SOMA03 identity, camera, masks, depth and object CAD. Use the [input and fitting guide](references/fitting.md).
2. Fit Stage 1 and inspect source overlays and oblique views. Check both rotation and translation over the full sequence before using a static skip.
3. Bind a placement checkpoint and author contact phases on the original object faces. Run Stage 2 from that exact parent. Preserve unknown or no-contact intervals.
4. Verify the exported native body mesh and object in the source coordinate frame. Record the selected parent, evidence, units and remaining limitations for [robot retargeting](../grail-agentic-retargeting/SKILL.md).
