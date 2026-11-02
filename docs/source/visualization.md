# Data Visualization

Render kinematic-replay MP4s of any GRAIL motion-library directory — works
on the retargeting output, the data-export process, and the public-release
layout. Single IsaacSim session per batch, so an N-motion render is much
faster than spawning a full training process per clip.

The visualization subpackage lives at
{src}`grail/visualization/`.
Two CLI wrappers cover the common cases:

| Script | Use case |
|---|---|
| {blob}`scripts/visualize.sh <grail/visualization/scripts/visualize.sh>` | Batch — render every motion in a library |
| {blob}`scripts/visualize_single.sh <grail/visualization/scripts/visualize_single.sh>` | Render one motion from its `robot/<seq>.pkl` |

Underneath, both call the same two Python modules:
`grail.visualization.prepare_vis_shard` (motion-lib → trajectory shard) and
`grail.visualization.batch_render_replay` (single IsaacSim session, one MP4
per motion).

## Native SOMA G1+DEx3 scene replay

Default SOMA retargeting first emits a 50-column G1+DEx3 CSV rather than the
packaged motion-library layout described below. Inspect that intermediate result
with the source-bound target and the same validated `hoi_data.pkl`:

```bash
export DISPLAY=:1
RETARGET_DIR=<retarget_output_directory>

imports/soma-retargeter/.venv/bin/python -m grail.retargeting.soma_replay \
    "$RETARGET_DIR/g1_retarget/csv/source.csv" \
    <path/to/hoi_data.pkl> \
    "$RETARGET_DIR/g1_dex3_target" \
    "$RETARGET_DIR/g1_dex3_scene.mp4"
```

This renderer refuses 29-joint/padded CSVs. It loads all 43 native robot joints,
the reconstructed object mesh and scale, and the per-frame object trajectory.
For visualization it creates a `2.0 x 0.6 x 0.04 m` tabletop with four legs,
centered below the initial object pose and placed at the scaled/rotated mesh's
minimum height. Use `--table-size WIDTH DEPTH THICKNESS` to override the
dimensions. It is a kinematic geometry/alignment review, not a physics or
grasp-success test. Once a motion is accepted and packaged into `robot/`,
`objects/`, and `object_usd/`, use the IsaacLab batch renderer below.

## Input layout

The visualizer expects any directory shaped like:

```
<motion_lib>/
├── robot/<seq>.pkl       required
├── objects/<seq>.pkl     required
├── object_usd/<seq>.usd  required
└── meta/<seq>.pkl        optional (used for table_pos when present)
```

Three layouts in the wild all satisfy this:

| Layout | Producer | Quat convention | Hand DOFs |
|---|---|---|---|
| `data/motion_lib/<name>/` | legacy SMPL-X/GMR retargeting pipeline (`grail.retargeting.retarget`) | **wxyz** | none |
| `logs_rl/<exp>/exported/step_*/shard_*/` | data-export pipeline (`grail.data_export.export_successful_rollouts`) | **xyzw** | `hand_dof_pos: (T, 14)` |
| `data/<name>/` | released dataset | **xyzw** | inherited from exported data |

Differences between the three are handled by flags on `visualize.sh` — see
[Quaternion convention](#quaternion-convention) and
[Hand DOFs](#hand-dofs).

## Output layout

Everything lands under `<motion_lib>/vis/` (kept separate from the release
`<motion_lib>/video/` which carries the original 2D HOI render, and from
the retarget `<motion_lib>/videos/` written by older pipelines):

```
<motion_lib>/vis/
├── <seq>.mp4                    one per motion
├── all_motions_combined.mp4     concat (only when --max_videos > 0)
└── examples_grid.mp4            4×4 or 2×2 grid (only when --max_videos > 0)
```

## Batch render

```bash
conda activate sonic    # or set $GRAIL_SONIC_ENV
export DISPLAY=:1

# Retargeting output — root_rot is wxyz, must override
QUAT_CONVENTION=wxyz bash grail/visualization/scripts/visualize.sh \
    data/motion_lib/pickup_table

# Data-export dir — defaults match (xyzw)
bash grail/visualization/scripts/visualize.sh \
    logs_rl/<exp>/exported/step_010000/shard_0

# Public-release dir — defaults match (xyzw)
bash grail/visualization/scripts/visualize.sh \
    data/pickup_table
```

Positional arguments (all but the first are optional):

```
visualize.sh <motion_lib_path> [max_videos] [cam_offset_x,y,z] [quat_convention]
```

| Arg | Default | Notes |
|---|---|---|
| `motion_lib_path` | — | Full path (absolute or repo-relative) to the motion library. |
| `max_videos` | `16` | Cap on number of motions rendered. Pass `0` to render all motions; `0` also skips the post-processing concat / grid. |
| `cam_offset` | `1.5,-1.5,1.0` | Camera position relative to the motion centroid. Comma-separated, no spaces. |
| `quat_convention` | `xyzw` | One of `auto`, `wxyz`, `xyzw`. See [Quaternion convention](#quaternion-convention). |

Env-var fallbacks: `QUAT_CONVENTION` (default `xyzw`).

## Single-motion render

```bash
bash grail/visualization/scripts/visualize_single.sh \
    data/pickup_table/robot/pickup_table__apple_0__000.pkl
```

Takes the full path to a `robot/<seq>.pkl` and writes
`<motion_lib>/vis/<seq>.mp4`. Useful for spot-checking a single clip without
re-rendering the whole library.

```
visualize_single.sh <robot_pkl_path> [cam_offset_x,y,z] [quat_convention]
```

## Quaternion convention

`root_rot` is stored in two different conventions across the codebase:

- {blob}`grail/retargeting/retarget.py` writes **wxyz**.
- {blob}`grail/data_export/export_successful_rollouts.py` writes **xyzw**.

The renderer expects wxyz, so `prepare_vis_shard.py` canonicalizes before
passing data downstream. Pick the right setting via `--quat_convention`
(CLI) or `QUAT_CONVENTION` (env var):

| Value | Behavior | Use for |
|---|---|---|
| `xyzw` *(default)* | Always swap `[x,y,z,w] → [w,x,y,z]` before rendering | data-export dir + public-release dirs |
| `wxyz` | No-op pass-through | retargeting output (`data/motion_lib/<name>/`) |
| `auto` | Magnitude-based detection: whichever of slot 0 / slot 3 carries more energy is taken to be the scalar `w` | Mixed corpora; **not robust** to motions that don't start near-upright (e.g. backward-leaning) |

`prepare_vis_shard.py` prints a per-batch summary line:

```
quat conventions (root_rot): wxyz=0 xyzw=16
```

so you can confirm the choice from the log.

## Hand DOFs

The gripper renders from `hand_dof_pos: (T, 14)` in the input `robot/<seq>.pkl`
when present, otherwise the renderer zero-pads the missing 14 DOFs, so the gripper stays in its open pose.

Status appears in the Step-1 log:

```
hand DOFs from hand_dof_pos: 16 / 16
```

## Under the hood

`visualize.sh` runs in three logical steps:

```
1. prepare_vis_shard.py   motion_lib/{robot,objects,meta} → /tmp/vis_shard_<key>/
                          (per-motion trajectory.pkl + synthetic metrics_eval.json)
2. batch_render_replay.py one IsaacSim session, hot-swap object USDs between motions
                          → <motion_lib>/vis/<seq>.mp4
3. (optional) ffmpeg      add per-clip labels, concat into all_motions_combined.mp4,
                          build 4×4 (or 2×2 if fewer than 16) examples_grid.mp4
```

Step 2 also handles MuJoCo → IsaacLab DOF reordering for the 29 G1 body
joints and pads to 43 DOFs (zero-fill) when `hand_dof_pos` is absent.

You can call the Python modules directly if you want to script around the
shell wrapper:

```bash
python -m grail.visualization.prepare_vis_shard \
    --data_dir <motion_lib> --shard_dir /tmp/shard --max_motions 16 \
    --quat_convention xyzw

python -u -m grail.visualization.batch_render_replay \
    --shard_dir /tmp/shard --traj_dir /tmp/shard/trajectories \
    --object_usd_dir <motion_lib>/object_usd \
    --output_dir <motion_lib>/vis \
    --camera_offset 1.5 -1.5 1.0 --camera_target 0.0 0.0 0.8 \
    --skip_existing --headless
```

## Following up with the web visualizer

`<motion_lib>/vis/*.mp4` is exactly the input the
[web visualizer](web_visualizer.md) consumes for hover-to-play previews —
generate `vis/` first, then point `grail.web_visualizer.generate_manifest`
at the same motion library.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Robot is rotated horizontally / lying on its side | Wrong `quat_convention`. Retarget pkls are `wxyz`; data-export and release pkls are `xyzw`. Match it explicitly instead of relying on `auto`. |
| Gripper stays open even at the grasp moment | `robot/<seq>.pkl` lacks `hand_dof_pos`. Re-run the data-export step with the post-2026-06-01 code (it persists the full 14-DOF hand trajectory alongside the scalar averages). |
| `Error: required subdir missing: <path>/object_usd` | The directory isn't a motion-library layout. Verify `robot/`, `objects/`, `object_usd/` all exist; `meta/` is optional. |
| `[DOF] Padding 29-DOF trajectories to 43-DOF articulation` printed and the gripper looks open | Expected when `hand_dof_pos` is absent — the renderer pads with zeros (= open pose). Not an error. |
| Render hangs on first motion | IsaacSim init failed silently. Confirm `DISPLAY` is set and the `sonic` env activated. The renderer has a 180 s watchdog that force-exits if no per-frame heartbeat fires, so stalls do eventually self-terminate. |
| Black mujoco viewer / `glfwInit failed` | `export DISPLAY=:1` **before** activating conda (same as retargeting — `DISPLAY` is an env var, not a conda setting). |
