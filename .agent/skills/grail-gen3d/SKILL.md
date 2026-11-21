---
name: grail-gen3d
description: Generate rigid 3D objects with Hunyuan3D or procedural curbs, slopes and stairs for GRAIL. Use for new mesh assets, object prompt collections, or terrain geometry; scene composition and source-video generation belong to grail-2dhoi.
---

# GRAIL 3D assets

Choose the generator from the requested geometry. Resolve asset count, dimensions,
output dataset and existing inputs before execution. Preserve supplied object
choices; explain a concrete modeling limitation rather than silently replacing
the object or imposing an unrelated category filter.

| Asset | Procedure | Typical output |
|---|---|---|
| Textured rigid object from a description | [Hunyuan3D generation](references/hunyuan.md) | `data/gen_<dataset>/<slug>/model.obj` and materials/textures |
| Parametric curb, slope or stairs | [Procedural terrain](references/terrain.md) | `data/syn_<dataset>/<type>_<index>/model.obj` and materials/textures |

## Generate within the requested scope

1. For object collections, write concrete, distinct descriptions and record the
   chosen list. Consider reach, rigidity, visual orientation cues and scene size;
   the [prompt guidance](references/hunyuan.md#object-descriptions) is a starting
   point, not a license to discard an explicitly requested object.
2. Use a unique persistent config/output namespace for the requested batch.
   Inspect existing destinations; do not overwrite another run or use a shared
   temporary YAML. Reuse inputs only when their settings and identity match.
3. Use the generator's actual environment. Hunyuan and inference dependencies
   differ from procedural terrain and reconstruction; do not change a shared
   runtime's dependencies as incidental setup. Keep credentials out of output.
4. Execute authorized work in the foreground or a tracked cluster job. Retain
   the config, logs, actual exit and incomplete outputs. Do not delete failed
   directories automatically or infer success from a running process.

## Inspect and deliver

Check mesh readability, material/texture paths, scale/units, up-axis, dimensions
and support surface. Rigid geometry does not validate articulation or dynamics.
Report requested, produced and failed counts separately, with actual asset paths
and unresolved defects. Existing `model.obj` files can trigger a skip even when
they have not passed these checks.

Use [grail-2dhoi](../grail-2dhoi/SKILL.md) for an authorized source-video stage,
carrying the original mesh, scale and coordinate convention. For multi-stage work
return the asset handoff to [grail-pipeline](../grail-pipeline/SKILL.md).
