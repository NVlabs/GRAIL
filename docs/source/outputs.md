# Outputs and Coordinate Conventions

Each example has its own directory under `/workspace/runs`. The Docker mount makes the same files visible under `../grail-demo/runs` beside your checkout. Use a fresh directory for a new fit or training run. Keep the original video, object mesh and scene files beside the derived results.

## Native robot motion

The native NPZ stores `qpos` per source frame. Its first seven columns are translation and a unit quaternion in **w, x, y, z** order. The G1 terrain target then has 29 joint positions (36 columns total). The manipulation target adds 14 finger joints (50 columns total). Read joint names and addresses from the file. Keep its FPS, source timestamps, object transforms, scale and original mesh hash when moving it between stages.

Some exported motion libraries use **x, y, z, w** quaternions. Select the convention for the file being opened in the [visualization guide](visualization.md#quaternion-convention). See [retargeting](retargeting.md) for target selection and [tracking](tracking.md) for training inputs.

## Stairs output layout

<span id="find-the-stairs-outputs"></span>

Under `../grail-demo/runs/stairs/` on the host, the walkthrough produced the following files. Keep the input, fit, scene geometry and review together if you intend to retarget or repack.

```text
generation/                                 original clip, camera, masks and metric mesh
selected_fit.pkl, selected_fit_geometry.npz  selected human fit and its geometry
reconstruction.mp4, review.json              human comparison and contact findings
robot.npz, retargeting.mp4                   selected robot poses and preview
retargeting_review.json                      robot contact findings
terrain/robot/stairs.pkl                     robot reference for scene-aware tracking
terrain/objects/stairs.pkl                   matching object reference
terrain/object_usd/stairs.usd                stair collider
terrain/flat_placeholder.usd                 scene input for the terrain launcher
tracking/train/model_step_000004.pt          policy after four updates
tracking/eval/step_000004/metrics_eval.json   evaluated policy progress and joint error
```

The generic retargeting CLI writes an initial `native/robot_UNREVIEWED.npz`. The tutorials ask for the **selected** native motion to be delivered as `robot.npz` after inspection. Edits may intervene; renaming the initial file cannot substitute for that review. See the {ref}`stairs measurements <stairs-track-result>` or {ref}`kettlebell measurements <kettlebell-track-result>` for the actual metric keys and units.

## Robot file validation

<span id="check-a-native-robot-file"></span>

From the container checkout root, choose **one** native archive. Replace the path with your own case. The reader prints its resolved path, mesh binding, shape, FPS and world frame; it checks finite poses, a uniform frame clock, joint count, unit root quaternions and object-array shapes. Add `--object-mesh /path/to/original.obj` to verify the mesh bytes against the hash saved in the archive. Each released case took under a second on the i9.

```bash
python scripts/standalone/check_robot.py /workspace/runs/stairs/robot.npz
```

| Native field | Stairs | Kettlebell | Interpretation |
|---|---|---|---|
| `qpos`, `timestamps` | (241, 36), 0–10 s | (121, 50), 0–5 s | One sample per source frame at 24 Hz. Root position is in metres, root quaternion is **wxyz**, hinge positions are in radians. |
| `dof_names`, `qposadr` | 29 joints | 43 joints | Look up joint columns by name and address. |
| `obj_R`, `obj_t` | 241 rotation matrices and translations | 121 rotation matrices and translations | Object rotations are 3 × 3 matrices; translations are 3-vectors in the declared z-up world. Keep `object_scale` and the original mesh hash. |

| `selected_fit_geometry.npz` | Stairs | Kettlebell |
|---|---|---|
| Animated human / object vertices per frame | 4,505 / 168 (241 frames) | 4,505 / 3,930 (121 frames) |

This companion carries `human_faces` and `object_faces` shared across frames. Preserve those arrays for the contact indices.

The stairs pack stores the robot root as **xyzw** in `robot/stairs.pkl` and the object quaternion as **wxyz** in `objects/stairs.pkl`. Reordering the native robot quaternion reproduced the packed values within 3 × 10⁻⁸. Converting the packed object quaternion back to a matrix reproduced `obj_R` within 1.2 × 10⁻⁷. Check each field's producer when consuming another library.

Array checks cannot establish hand contact or balance. The kettlebell archive retained `native_status=UNREVIEWED`; the separate visual review selected a partial result and recorded the unresolved contacts.

## Conventional batch reconstruction and training exports

A filtered reconstruction sequence can contain `hoi_data/hoi_data.pkl` (human and object parameters), `mesh_data/` (the bound object geometry) and `result_vis/`. Look there for `recon_result.mp4`, `recon_comparison.mp4`, a top view and the ScenePic HTML when those outputs were enabled. Keep the camera, original mesh and the export's coordinate convention with the sequence. Numerical filtering and visual selection address different questions.

A packed or exported library indexes robot and object PKLs and object USDs by motion stem. The corresponding `vis/<stem>.mp4` is a kinematic replay. A source video, a replay and a simulated rollout can show different trajectories. For a reproducible [export](data_export.md), keep the checkpoint step and recorded evaluation with the library.
