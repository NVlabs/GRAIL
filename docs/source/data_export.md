# Motion data export

For an existing rollout, `grail.data_export.export_successful_rollouts` extracts robot and object motion. To merge locally exported shards, run `python -m grail.data_export.merge_exports --help` for the required experiment and source directories. Inspect the merged manifest and overlap counts before using a library.

For native retargeting output, use the [manipulation handoff](retargeting.md#hand-off-a-retargeted-motion). It preserves the source clock and object geometry needed by a downstream tracker.


## Which input do I have?

| Input in hand | Next step |
|---|---|
| A reviewed native `robot.npz` | Use the [terrain or manipulation tutorial](quick_start.md) to pack it for training. |
| A trained policy checkpoint | Evaluate it on its matching motion and scene. Inspect progress, termination and object metrics before export. |
| A rollout evaluation with `metrics_eval.json` and recorded `.trajectory.pkl` files | Supply the evaluation directory, matching source motion library and a fresh output directory to the exporter. A checkpoint or metrics file alone contains no replay trajectory. |

By default a motion must finish without early termination and reach full reference progress to be selected. When trajectories are available, the exporter writes the robot and object motions, object USD, metadata and an `export_manifest.json`. Inspect its exported count and skipped trajectories before opening the [visualizer](visualization.md). A short tracking smoke run may yield no exportable motion.

## Can I export the short stairs evaluation?

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

For a completed full-schedule stairs policy, select `tracking-full/train/model_step_005000.pt` in the recording block and choose fresh recording and export directories in both blocks. For a successful export, check that the library contains `robot/`, `objects/` and `object_usd/`. The [visualizer](visualization.md) reads those files and writes a kinematic replay under `vis/` on a host with working Isaac Sim rendering. The short run above yields no watchable stair rollout; the [production stair image](results.md#stair-climb) comes from a separate rendered evaluation on an L40S.
