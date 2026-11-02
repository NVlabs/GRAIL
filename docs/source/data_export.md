# Data Export

Turn successful task-general tracking rollouts into a training-ready motion
library you can use to bootstrap the next sweep or publish as a public
dataset.

The core of the pipeline lives under
{src}`grail/data_export/` plus the
kinematic-replay renderer in the sibling
{src}`grail/visualization/` module —
all cluster-agnostic Python.

## Pipeline

```
W&B sweep
    ↓
0. select_top_checkpoints   → top-K checkpoints by eval success rate
    ↓
1. shard eval (your scheduler)
                            → per-shard {metrics_eval.json, *.trajectory.pkl}
    ↓
2. batch_render_replay      → per-shard vis/*.mp4 (kinematic replay)
   export_successful_rollouts
                            → per-shard {robot,objects,object_usd}/*.{pkl,usd}
    ↓
3. merge_exports            → single `merged/` motion library
```

| Module | What it does |
|---|---|
| `grail.data_export.select_top_checkpoints` | Rank W&B checkpoints by reported eval success rate |
| `grail.visualization.batch_render_replay` | Kinematic-replay MP4 renderer (no policy, no physics, single IsaacSim session per shard) |
| `grail.data_export.export_successful_rollouts` | Convert Phase 1 trajectory pkls → per-motion `robot/`, `objects/`, copy `object_usd/` |
| `grail.data_export.merge_exports` | Merge per-shard exports into a single `merged/` motion library |
| `grail.data_export.summarize_phase1_sr` | Aggregate per-shard Phase 1 metrics into a single SR table |

## Prerequisites

- A completed task-general tracking training run with W&B logging.
- Access to the **source motion library** the sweep trained on, in
  particular its `object_usd/` directory — those USD assets get copied
  into the exported dataset verbatim.
- The `sonic` conda env (or set `GRAIL_SONIC_ENV=<name>`).
- A GPU + IsaacLab for the kinematic-replay step. IsaacLab is already
  bundled in the `sonic` env, no separate install needed.

## Running

### Step 0 — pick the best checkpoints

```bash
conda activate sonic    # or set $GRAIL_SONIC_ENV

python -m grail.data_export.select_top_checkpoints \
    --sweep <wandb_sweep_id> --k 5
# alternatively: --group <wandb_group_id>
```

### Step 1 — shard eval (write Phase 1 trajectories)

Run your own scheduler so that for each shard `i ∈ [0, num_shards)` you
end up with:

```
{exp_dir}/eval/step_{step}/phase1_shard_{i}/
├── metrics_eval.json
└── <motion_key>.trajectory.pkl   (one per evaluated motion)
```

The eval entry point is `eval_agent_trl.py` from
{src}`imports/SONIC <imports/SONIC/>`. Run one eval shard per worker in
your scheduler and write each shard to the layout above.

### Step 2 — render + export per shard

```bash
# Per shard, two commands. Loop over shard indices in your scheduler.
SHARD_DIR={exp_dir}/eval/step_{step}/phase1_shard_{i}
OUT_DIR={exp_dir}/exported/step_{step}/shard_{i}

python -u -m grail.visualization.batch_render_replay \
    --shard_dir   "$SHARD_DIR" \
    --traj_dir    "$SHARD_DIR" \
    --object_usd_dir <source_motion_lib>/object_usd \
    --output_dir  "$OUT_DIR/vis" \
    --skip_existing --headless

python -m grail.data_export.export_successful_rollouts \
    --eval_dir    "$SHARD_DIR" \
    --source_data <source_motion_lib> \
    --output_dir  "$OUT_DIR" \
    --min_progress 1.0
```

### Step 3 — merge shards

```bash
python -m grail.data_export.merge_exports \
    --shards {exp_dir}/exported/step_{step}/shard_* \
    --output {exp_dir}/exported/step_{step}/merged
```

## Output layout

```
merged/
├── robot/<seq>.pkl           per-motion robot trajectories (29 body DOFs + hand_dof_pos)
├── objects/<seq>.pkl         per-motion object trajectories
├── object_usd/<seq>.usd      USD assets copied from source_data
│   └── textures/<seq>/…      flat or nested texture trees (both layouts preserved)
├── vis/<seq>.mp4             kinematic-replay videos
└── export_manifest.json      counts + checkpoint provenance
```

The layout is byte-identical to the established GRAIL/legacy-GMR motion library
written under `data/motion_lib/<name>/`; it is not the raw CSV emitted by the
default SOMA Retargeter. A merged-export motion library is therefore a
drop-in replacement for any training config's
`motion_lib_cfg.motion_file` / `object_motion_file` /
`object_usd_path`.

Preview the result with the {doc}`web_visualizer` (point it at the
`merged/` dir).
