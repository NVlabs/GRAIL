# Motion tracking

Use the [stairs tutorial](tutorials/stairs.md) for terrain packing or the [kettlebell tutorial](tutorials/kettlebell.md) for the hand–object handoff. A saved policy and its evaluation use different files. The [output reference](outputs.md) lists both.

Both examples finetune with 2,048 environments for 5,000 PPO updates and evaluate checkpoints at 1,000-update intervals.

- **Stairs.** Package the body-only G1 motion and stair collider with `python -m grail.retargeting.terrain`, then run the SONIC terrain trainer. Each checkpoint writes `tracking/eval/step_XXXXXX/metrics_eval.json`.
- **Kettlebell.** The [manipulation handoff](retargeting.md#hand-off-a-retargeted-motion) binds the G1+DEx3 motion, source contact map and ordered object mesh. Run `scripts/tracking/manipulation_finetune.sh` on that handoff to build the pack, train ContactFirst and evaluate each saved checkpoint after training releases the GPU. The final rollout and metrics are in `tracking/eval_final/kettlebell/`.

Install the `sonic` environment with `scripts/setup/install_env_sonic.sh`. The checkpoint downloader installs the terrain policy, frozen manipulation decoder and [ContactFirst actor warmstart](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/tree/main/checkpoint/SONIC/models/manip_warmstart). The [GRAIL dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) also includes released motion data.

From the checkout root in the active `sonic` environment, the stairs launcher is `bash scripts/tracking/terrain_finetune.sh OUTPUT_DIR TERRAIN_LIBRARY`. When increasing the schedule, reuse the existing packed library and give the launcher a fresh output directory. The trainer writes numbered checkpoints under `train/` and the launcher evaluates them after training. The launcher creates a new run directory on each invocation and has no in-place resume option. The {ref}`stairs tutorial <stairs-track-result>` shows the measured result for each schedule.

After the four-update stairs tutorial, run the full schedule in the same configured shell. Keep the existing terrain library and choose an unused output path. The defaults train 2,048 environments for 5,000 updates and evaluate all five saved checkpoints.

```bash
bash scripts/tracking/terrain_finetune.sh /workspace/runs/stairs/tracking-full /workspace/runs/stairs/terrain
```

## Quick Start tracking details

### Stairs

From the repository root in the configured `sonic` environment, the stairs launcher finetunes `models/terrain_stairs/last.pt` on the packed terrain library, then evaluates each numbered checkpoint. The default run is equivalent to the following commands, including the initial pack step:

```bash
STAIRS=/workspace/runs/stairs
LIB="$STAIRS/terrain"
OUT="$STAIRS/tracking"
python -m grail.retargeting.terrain --native "$STAIRS/robot.npz" --motion-key stairs --output "$LIB" && (
  set -e
  cd imports/SONIC
  python gear_sonic/train_agent_trl.py +exp=manager/universal_token/scene/terrain_tracking \
    num_envs=2048 headless=true ++warm_resume=true ++use_wandb=false \
    "++checkpoint=$PWD/models/terrain_stairs/last.pt" "++experiment_dir=$OUT/train" \
    ++algo.config.num_learning_iterations=5000 ++callbacks.model_save.save_frequency=1000 \
    ++callbacks.model_save.save_last_frequency=20 \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$LIB/robot" \
    "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$LIB/objects" \
    "++manager_env.config.terrain_motion_dir=$LIB" "++manager_env.config.flat_usd_path=$LIB/flat_placeholder.usd"

  for step in 001000 002000 003000 004000 005000; do
    python gear_sonic/eval_agent_trl.py "++checkpoint=$OUT/train/model_step_${step}.pt" \
      ++num_envs=1 ++headless=true '++eval_callbacks=[im_eval]' ++run_eval_loop=false \
      ++use_wandb=false "++eval_output_dir=$OUT/eval/step_${step}"
  done
)
```

On the host, the final tracking errors and termination details are in `../grail-demo/runs/stairs/tracking/eval/step_005000/metrics_eval.json`. Earlier checkpoint evaluations are in the other `step_*` directories. To run a shorter schedule, set `NUM_ENVS`, `ITERATIONS` and `EVAL_EVERY` on the launcher command. For example, `NUM_ENVS=64 ITERATIONS=4 EVAL_EVERY=4` trains for four updates and evaluates checkpoint `000004`.

### Kettlebell

The manipulation launcher builds a tracking pack from the handoff and uses the released ContactFirst actor warmstart (`models/manip_warmstart/last.pt`) and frozen decoder (`models/sonic_manipulation_base/last.pt`) under `imports/SONIC/`. Its defaults are `NUM_ENVS=2048`, `ITERATIONS=5000` and `EVAL_EVERY=1000`. Evaluation starts after training releases the GPU.

On the host, open `../grail-demo/runs/kettlebell/tracking/eval_final/kettlebell/summary.json` for reference coverage and termination, or `output/eval.mp4` in that directory for the final rollout. Earlier evaluations are under `tracking/eval/step_*/kettlebell/`. To write metrics without rollout videos, prefix the launcher command with `EVAL_NO_VIDEO=1`.
