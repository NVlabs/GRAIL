# Data Export

<span id="export-recorded-policy-motion"></span>

<span id="motion-data-export"></span>

<span id="which-input-do-i-have"></span>

Export produces a motion library from what the simulated robot actually did. You need the **recorded trajectory**, its evaluation metrics and the matching source library for the scene geometry. A native robot NPZ is ready for [tracking](tracking.md); a checkpoint still needs an evaluation with recording.


<span id="can-i-export-the-short-stairs-evaluation"></span>

## Single-motion export

<span id="record-and-export-one-stairs-evaluation"></span>

The tutorial’s default terrain evaluator writes a metrics JSON. Its four-update run ended early and produced no recorded trajectory. A policy checkpoint alone cannot be opened as a rollout in the motion visualizer. You can watch the kinematic reference in `retargeting.mp4` immediately, or use the [published motion library](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) for an existing trajectory.

To collect a trajectory, open the container shell at the checkout root. Re-evaluate the saved four-update stair policy with the trajectory recorder. Use a new output directory. This takes about a minute on the RTX 6000 Ada and requires the same checkpoint and packed scene as training.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
export OMNI_KIT_ACCEPT_EULA=Yes WANDB_MODE=disabled
export PYTHONPATH="$PWD/imports/SONIC:$PWD${PYTHONPATH:+:$PYTHONPATH}"
cd imports/SONIC
OUT=/workspace/runs/stairs-export/recorded
python gear_sonic/eval_agent_trl.py \
  ++checkpoint=/workspace/runs/stairs/tracking/train/model_step_000004.pt \
  ++num_envs=1 ++headless=true '++eval_callbacks=[im_eval]' \
  ++run_eval_loop=false ++use_wandb=false "++eval_output_dir=$OUT" \
  manager_env/recorders=empty \
  ++manager_env.recorders.trajectory._target_=gear_sonic.envs.manager_env.mdp.recorders.TrajectoryRecorderCfg \
  "++manager_env.recorders.trajectory.save_path=$OUT" "hydra.run.dir=$OUT/hydra"
```

Look for `000000.trajectory.pkl` alongside `metrics_eval.json`. Return to the checkout root (`cd /workspace/grail`) in the same container shell and run the exporter on those files and the original terrain library.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grail
python -m grail.data_export.export_successful_rollouts \
  --eval_dir /workspace/runs/stairs-export/recorded \
  --source_data /workspace/runs/stairs/terrain \
  --output_dir /workspace/runs/stairs-export/library
```

The recorded four-update policy reached 11.8% of the reference and terminated early. The exporter printed `SKIP: stairs (terminated)` and `No successful motions found. Nothing to export.` It returned without creating a motion library. A completed reference can be selected by the same command with its matching recorded evaluation. Recording the trajectory does not change the policy’s performance.

## Output layout

<span id="read-the-export-result"></span>

This guide ran the early-termination case only. No successful library was exported in this verification. By default the exporter requires a non-terminated motion with `--min_progress 1.0`. Set `--obj_pos_threshold` to apply an additional object-position error limit.

For an eligible recorded rollout, inspect `export_manifest.json`. Its `exported_count` should be positive and each entry should have its matching robot PKL, object PKL and USD. `skipped_no_trajectory` counts eligible motions without recordings. The exporter's `--verify <library-directory>` checks an existing library and does not create one. The short run returned zero eligible motions in under a second on the i9-14900K.

If the rollout terminated, inspect the tracking result. If a completed motion lacks a trajectory, enable recording when evaluating it. Use [Visualization](visualization.md) for an existing library.

<span id="running"></span><span id="prerequisites"></span><span id="step-0-pick-the-best-checkpoints"></span>

## Pipeline

<span id="export-a-collection-of-tracking-rollouts"></span>

For a collection, the steps are **select checkpoints → evaluate with trajectory recording → replay if needed → export completed motions → merge shards**.

| Operation | Public entry point |
|---|---|
| Rank checkpoints | `grail.data_export.select_top_checkpoints` |
| Record and summarize evaluations | `gear_sonic/eval_agent_trl.py` and `grail.data_export.summarize_phase1_sr` |
| Render a kinematic replay | `grail.visualization.batch_render_replay` |
| Export eligible trajectories | `grail.data_export.export_successful_rollouts` |
| Merge the exported shards | `grail.data_export.merge_exports` |

Use the merged `robot/`, `objects/` and `object_usd/` as the training configuration’s `motion_file`, `object_motion_file` and `object_usd_path`. When the task uses shape features or hand-contact labels, check those inputs against the original library as described in [tracking](tracking.md#preparing-retargeted-data-for-training).

Use this path for several checkpoints or motion shards. Start with the training run, its matching motion library and object USDs. A policy checkpoint alone supplies neither the evaluation trajectories nor the scene assets. In the configured environment, set `WB_RUN` to your own W&B run path if you used W&B. Select the top five checkpoints:

```bash
python -m grail.data_export.select_top_checkpoints --run "$WB_RUN" --k 5
```

<span id="step-1-shard-eval-write-phase-1-trajectories"></span><span id="step-2-render-export-per-shard"></span>

For a group, supply `--group`, `--entity` and `--project`. For each selected step, evaluate the assigned motion subset using the trajectory-saving options above. Each worker must write its own `eval/step_<step>/phase1_shard_<rank>/` containing `metrics_eval.json` and the trajectory pickles. Ensure shards have disjoint motion keys. For one shard, set `EXP_DIR` to your experiment, `SOURCE` to the original library, `STEP` to the exact directory suffix and `RANK` to its index:

```bash
SHARD="$EXP_DIR/eval/step_${STEP}/phase1_shard_${RANK}"
OUT="$EXP_DIR/exported/step_${STEP}/shard_${RANK}"
python -m grail.visualization.batch_render_replay --shard_dir "$SHARD" --traj_dir "$SHARD" \
  --object_usd_dir "$SOURCE/object_usd" --output_dir "$OUT/vis" --skip_existing --headless
python -m grail.data_export.export_successful_rollouts --eval_dir "$SHARD" \
  --source_data "$SOURCE" --output_dir "$OUT" --min_progress 1.0
```

<span id="step-3-merge-shards"></span>

The renderer uses the SONIC environment and an RTX-capable simulator. Repeat per shard. Then aggregate the metrics and merge completed shards (set `N` to the actual shard count):

```bash
python -m grail.data_export.summarize_phase1_sr --exp_dir "$EXP_DIR" --steps "$STEP"
python -m grail.data_export.merge_exports --exp_dir "$EXP_DIR" --source_data "$SOURCE" \
  --step "$STEP" --num_shards "$N"
```

Inspect `exported/step_<step>/merged/` for robot and object PKLs, USDs with textures, and replay MP4s under `vis/`. Include metadata when available. The merger checks counts, duplicate motion keys and shard overlap. Use the [web viewer](web_visualizer.md) to inspect a library. These batch interfaces were checked against the release tools; the short stairs example above ran the early-termination export path.
