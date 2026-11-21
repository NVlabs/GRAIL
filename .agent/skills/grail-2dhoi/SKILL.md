---
name: grail-2dhoi
description: Configure GRAIL scenes, object placement and camera composition, review renders, and generate human-object source videos. Use for manipulation, pickup, sitting or terrain datasets; Kling is the default provider and MiniMax-H3 is explicit opt-in.
---

# GRAIL source videos

Start from the requested assets, action, scene family and video count. Bind the
original mesh/scale, character, scene, camera and random-scene settings. A request
to configure or render stops there; an authorized video-generation task proceeds
after its render review without an extra blanket approval step.

## Configure and review

1. Select manipulation, pickup, sitting or terrain using the current configs.
   Read [scene and asset selection](references/scenes.md) for floor/table choices,
   mesh visibility checks and object YAML structure. Measure actual dimensions
   and units; category names alone do not establish scale or up-axis.
2. Choose orientation, placement and camera parameters using
   [composition](references/composition.md). Keep shared scene defaults in
   `configs/scenes/`; write dataset-specific overrides in the object YAML or CLI.
3. Render before submitting the corresponding provider requests. Inspect every
   new render selected for generation, using labelled contact sheets when useful;
   inspect ambiguous images individually. Check object visibility, body framing,
   support, intended interaction and informative orientation under the task's
   accepted framing policy. Reuse already-reviewed unchanged renders and records.
4. Correct only the affected categories in a new output namespace, keeping the
   old render and its camera/depth/placement metadata together. Review replacements
   before generation. Do not clear old output trees or alter a frozen source video
   as an incidental reconstruction repair.

## Generate with the selected provider

Use [launch and resume](references/generation.md) for render-only and video-only
commands. Kling remains the default; retain the combined concurrency cap of 20
across this workload's floor/table batches, subject to any tighter live limit.
Splitting datasets does not create another provider quota.

For explicit MiniMax-H3, read [its runtime contract](references/minimax-h3.md)
before preparing the video command. Rendering is shared; runtime, capacity,
provider flags and output directories are provider-specific. Do not require H3
weights for Kling or silently switch providers after a failure.

Reuse outputs only when the input/config/provider identity and requested work
match. Check the actual skip behavior and recorded provider request before
resuming. Keep errors and partial results; neither `--skip_done` nor an existing
MP4 proves a new request is needed or that the video is acceptable.

## Handoff

Report generated/failed/pending counts, provider request identifiers and exact
video paths. Carry original asset, scene/camera, render review, provider metadata,
frame count and FPS into reconstruction. Decode completion and visual/source
acceptance remain separate. Use [grail-agentic-hoi-recon](../grail-agentic-hoi-recon/SKILL.md)
for the next requested stage or [grail-pipeline](../grail-pipeline/SKILL.md) for
multi-stage orchestration.
