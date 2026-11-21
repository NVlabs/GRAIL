---
name: grail-pipeline
description: Connect local asset and source-video generation to native SOMA03 reconstruction, robot retargeting and downstream motion handoff.
---

# Motion pipeline

Start at the first missing artifact whose parent is known. Use [3D asset generation](../grail-gen3d/SKILL.md) and [source-video generation](../grail-2dhoi/SKILL.md) when creating a new source. For an existing video, start at [human–object reconstruction](../grail-agentic-hoi-recon/SKILL.md). Bind the selected reconstruction to [robot retargeting](../grail-agentic-retargeting/SKILL.md) and hand off the reviewed trajectories and CAD. See [data export](../../../docs/source/data_export.md) when consuming existing rollouts.
