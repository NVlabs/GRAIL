# Outputs and coordinate conventions

Each example has its own directory under `/workspace/runs`. The Docker mount makes the same files visible under `../grail-demo/runs` beside your checkout. Use a fresh directory for a new fit or training run. Keep the original video, object mesh and scene files beside the derived results.

| File or directory inside a case | Written by | What to check |
|---|---|---|
| `generation/` | Input copied from the repository | Source RGB, camera, first-frame depth and masks, metric object mesh and pose describe the same scene. |
| Selected fit and `reconstruction.mp4` | SOMA reconstruction | Compare human and object against the source through the entire clip. Keep the selected native fit and its review together. |
| `contact_map.json` (moving object) | Source contact review | Contact intervals follow the original video clock and identify regions on the original mesh. Remeshing or reordering faces invalidates that binding. |
| `robot.npz` and `retargeting.mp4` | G1 retargeting | Check foot support, hand placement and the object trajectory. A kinematic preview can play even when physics tracking will fail. |
| `terrain/` or `handoff/` | Terrain packer or manipulation handoff | Stairs include a static collider. Manipulation carries the reviewed contacts, mesh and robot motion into the tracking pack. |
| `tracking/train/` (stairs), `tracking/finetune/train/` (manipulation) | PPO training | A numbered policy checkpoint records the completed update. It is a policy, not a robot trajectory. |
| `tracking/eval/step_*/metrics_eval.json` (stairs), `tracking/eval_final/kettlebell/summary.json` (manipulation) | Saved-policy evaluation | Read progress or reference coverage, termination reason and tracking errors. A completed evaluation can report a failed grasp or step. |

## Native robot motion

The native NPZ stores `qpos` per source frame. Its first seven columns are translation and a unit quaternion in **w, x, y, z** order. The body-only G1 then has 29 joint positions (36 columns total). The DEx3 target adds 14 finger joints (50 columns total). Read joint names and addresses from the file. Keep its FPS, source timestamps, object transforms, scale and original mesh hash when moving it between stages.

Some exported motion libraries use **x, y, z, w** quaternions. Select the convention for the file being opened in the [visualization guide](visualization.md#quaternion-convention). See [retargeting](retargeting.md) for target selection and [tracking](tracking.md) for training inputs.

## Find the stairs outputs

Under `../grail-demo/runs/stairs/` on the host, the walkthrough produced the following files. Keep the input, fit, scene geometry and review together if you intend to retarget or repack.

```text
generation/                                 original clip, camera, masks and metric mesh
selected_fit.pkl, selected_fit_geometry.npz  selected human fit and its geometry
reconstruction.mp4, review.json              human comparison and contact findings
robot.npz, retargeting.mp4                   selected robot poses and preview
retargeting_review.json                      robot contact findings
terrain/robot/stairs.pkl                     robot reference for the terrain tracker
terrain/objects/stairs.pkl                   matching object reference
terrain/object_usd/stairs.usd                stair collider
terrain/flat_placeholder.usd                 scene input for the terrain launcher
tracking/train/model_step_000004.pt          policy after four updates
tracking/eval/step_000004/metrics_eval.json   evaluated policy progress and joint error
```

The generic retargeting CLI writes an initial `native/robot_UNREVIEWED.npz`. The tutorials ask for the **selected** native motion to be delivered as `robot.npz` after inspection. Edits may intervene; renaming the initial file cannot substitute for that review. See the {ref}`stairs measurements <stairs-track-result>` or {ref}`kettlebell measurements <kettlebell-track-result>` for the actual metric keys and units.
