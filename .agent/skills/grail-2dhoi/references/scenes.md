# Scene and asset selection

Read the actual mesh and current scene/config YAML before setting overrides.
Keep the dataset field consistent with `data/<dataset>/<category>/model.obj`
and the generation output namespace.

| Action | Pipeline config under `configs/gen_2dhoi/` | Scene family |
|---|---|---|
| Manipulation | `manipulation.yaml` | `indoor1-manipulation` floor or `indoor2-manipulation` table |
| Pickup | Select the compatible manipulation settings with the pickup scene | `indoor2-pickup` |
| Sitting | `sitting.yaml` | `indoor1-sitting` |
| Curb / slope / stairs | `terrain_curbs.yaml`, `terrain_slope.yaml`, `terrain_stairs.yaml` | Corresponding terrain scene in `configs/scenes/default.yaml` |

## Dimensions and scene fit

Inspect all three bounding-box extents after resolving units, scale and up-axis.
The historical manipulation heuristic assigned floor placement for a largest
dimension of at least 0.60 m, or a largest dimension of at least 0.40 m with a
second dimension of at least 0.30 m. Heavy objects and functional furniture may
also belong on the floor even when smaller. These are composition starting
points, not physical measurements of mass or universal classification rules.

Use specific category semantics: a wheel rim differs from a small flywheel, and
`motorcycle_helmet` must not be classified as an engine by a `motor` substring.
A coffee table is itself a support surface; a spice rack can sit on a table.
When the fit is ambiguous, inspect both reach and actual support footprint.

Historical visibility screens for the narrow tabletop scene used largest extent
0.10 m, the product of the two smaller extents 0.003 m², and bounding-box volume
0.0008 m³. Apply such screens only when appropriate to the chosen camera and
dataset policy. A sparse mesh may pass them and still render as a sliver; an
intentionally thin tool may be usable with a better view. Record exclusions and
their reason; do not delete the assets or inherit a previous batch's drop list.

For sitting, verify seat height, up-axis, rigid support and fit beside the actual
character. Casters and
swivel joints are incompatible with a completely static rigid-chair assumption
unless the source and task support that representation.

## Object YAML

Keep shared scene definitions in `configs/scenes/default.yaml`. Example overrides:

```yaml
dataset: <dataset>
objects:
  default:
    obj_scale: [1, 1, 1]
    scene: indoor2-manipulation
  <category>:
    obj_scale: [1, 1, 1]
    scene: indoor2-manipulation
    obj_rot: [90, 0, 270]
    obj_rot_rand_max: [0, 0, 30]
```

The example rotation and scale must match the real mesh. Hunyuan outputs are
intended to be metric; other datasets can have different conventions. Do not
infer a scale multiplier from the dataset name alone. Each category needs an
explicit mapping or YAML alias; a bare key becomes `None` and is not an override
dictionary. Read actual category/default merge behavior when editing a config.

Use `--object_config`, `--results_dir`, `--num_rand_scenes` and supported CLI
overrides instead of copying an entire base pipeline YAML per dataset. If a
required option has no CLI override, make a scoped config rather than changing a
shared default for every dataset.

## Splitting floor and table batches

Separate object YAMLs/output namespaces can simplify review and failure isolation.
Plan the split before rendering. A new dataset alias must still resolve the same
original mesh paths; create an alias only after checking that no existing file,
directory or symlink will be replaced. Retain valid prior render/metadata bindings
instead of moving trees or replacing symlinks automatically.
