# Agentic Retargeter

Read `.agents/skills/grail-agentic-retargeting/SKILL.md` for robot contact and native animation work. `.agents/skills` is the skill source of truth.

Preserve original model geometry, limits, source timing and input hashes. Export edits as new UNREVIEWED candidates; separate tool tests from motion acceptance. Contact, whole-mesh, source/depth and temporal checks remain necessary at the requested scope.

The upstream reconstruction and initial GMR conversion run in GRAIL. This repository owns the robot refinement tools, workflow and Blender extension. Keep those tools independent of local GRAIL checkout paths and historical artifact directories.

The host assistant plans and reviews edits using its available local tools; this workflow makes no remote model API calls. Do not read, copy or change credentials as part of routine Blender work.

Use installed Blender MCP when available and useful, or foreground `blender -b --python` execution. The local `blender-edit` command in `docs/blender-agent.md` performs one bounded inspect/edit round without model calls. This repository does not bundle a live MCP client. Keep a versioned `.blend` as the editable state: continue from the latest chosen checkpoint, save each round to a new path, and retain exported native data and review evidence alongside it. Verify the saved scene/native round trip; a render or tool success never selects or accepts a motion automatically.

Use `docs/hand-inspection.md` for canonical hand snapshots and measured/predicted contact maps. The sidebar and worker share the same original-mesh inspection engine. Bind anatomy, mesh identities and exact frames explicitly; measured unsigned proximity, model predictions and authored hypotheses are different evidence. Missing predictions are unknown, and inspection views must refresh after pose changes. Preserve mounted robot views alongside isolated wrist-local hands.

Build the extension with `python extensions/blender/build.py --out /absolute/new/path`. Use Blender 4.3.2 for installation/edit/export checks; isolate its user profile and preserve user scenes. Do not include model assets, case motions, credentials, caches or generated outputs in commits.
