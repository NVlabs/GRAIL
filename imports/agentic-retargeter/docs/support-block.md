# Authorized floating support block

The scene-preparation tool replaces one explicitly named packed static support
mesh. It requires the case's recorded authorization. It never changes robot
poses, object motion, timing, scale, the first frame, or other scene geometry.
The original selected support and its metadata are archived alongside the new
candidate. Prior approval/review metadata is archived and cleared on the new
candidate; it cannot approve a changed scene. Original scenes remain the default
when no replacement is authorized.

The default recipe uses the original frame-zero object world bounds:

- Center the block under the object's horizontal bounding box.
- Set each horizontal dimension to the larger of the object's extent plus
  3 cm per side and the minimum `[0.60, 0.50]` metres.
- Put the top at the object's minimum world-Z coordinate and the bottom
  8 cm below it. The block is static and axis-aligned in the native z-up frame.

This is the geometry recipe used for the enlarged-block setting. It is not a
support or collision certificate: later object motion, hand/block crossings,
and robot/body collisions still need independent full-sequence review.

```bash
python tools/retarget/support_block.py \
  --native inputs/original_native.npz \
  --object-mesh inputs/original_object_mesh.npz \
  --table-name 'Original table' \
  --authorization 'Case owner requested the selected table be replaced by a floating support block' \
  --output runs/support_block_v001
```

Inputs must include a valid `grail.static_scene.v1` native scene, a native world
frame whose name ends in `z_up`, and an original object-local mesh with
`vertices` and `faces`. The native object poses and constant positive scalar or
XYZ scale determine frame-zero world bounds. If the mesh NPZ declares a
`world_frame`, it must match the native frame. The tool retains all original
object vertices/faces, including any degenerate facets; it does not perform a
distance query or silently substitute a hull.

Outputs are `candidate_UNREVIEWED.npz`, `original_support.npz`, `report.json`, and
`soma_scene_handoff.json`. The report binds input/output hashes and records exact
preservation of motion fields and unselected static geometry. A pre-existing
output directory is rejected. Old status, approval and review fields are archived
byte-exactly in `original_support.npz`, with their names and hashes in the report.
The new candidate explicitly sets `status`/`review_status` to `UNREVIEWED` and
acceptance, review and selection flags to false. These metadata resets are
listed separately among the declared changes; they do not edit the trajectory.

`soma_scene_handoff.json` is marked `NOT_RUN`. Bind the exact new scene in the
SOMA object sidecar and collision configuration before optimizing, and use the
same scene in final audits and rendering. This tool does not rewrite arbitrary
solver configurations, run SOMA, or repair inherited collisions. The Python API
is `agentic_retargeter.support_block.build_replacement`, with
`validate_replacement` for exact regeneration and `run` for file delivery.

The optional `--min-xy-m`, `--margin-m`, and `--thickness-m` arguments record a
different explicitly authorized recipe; they do not imply that it was tested
with the current motions. See the [current pipeline](current-retargeting-workflow.md)
for the subsequent initialization and refinement stages.
