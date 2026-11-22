# Task General Tracking

<span id="train-and-evaluate-a-tracking-policy"></span>

<span id="motion-tracking"></span>

GRAIL trains task-general loco-manipulation tracking policies from retargeted references. The scene-aware tracker follows terrain geometry. For manipulation, an object-aware latent adaptor conditions the whole-body controller on the object and adds hand actions. See [Section 3.3 of the paper](https://arxiv.org/abs/2606.05160v2). The packed library describes your scene and reference. The starting checkpoint supplies a policy trained for that kind of motion. Complete the [one-time installation](installation.md) and [tracking downloads](downloads.md), and use a fresh training directory for each schedule.

## Quickstart

<span id="quick-start-tracking-details"></span>

| Task | Inputs and command | What the short run produced |
|---|---|---|
| Stairs | {ref}`Pack the terrain and launch training <stairs-tracking-step>`. Pack the body reference and its original static collider together. The released stairs policy is the starting point. | `tracking/train/model_step_000004.pt` and `tracking/eval/step_000004/metrics_eval.json`. The policy reached 11.8% of the reference before termination. |
| Kettlebell | {ref}`Build the handoff and train the object-aware manipulation policy <kettlebell-tracking-step>`. Supply the native motion with finger joints, original CAD, reviewed contact map, `manip_warmstart/last.pt` and `sonic_manipulation_base/last.pt`. | A checkpoint and `tracking/eval_final/kettlebell/summary.json`. Inspect reference coverage, termination and object error together. The pack's default mass is 0.1 kg. |

Both tutorial commands used 64 environments, four updates and one evaluation in `nvgrail/grail:1.2.0` on an RTX 6000 Ada. Training plus evaluation took 1 min 09 s for stairs and 2 min 20 s for kettlebell. The default launchers use 2,048 environments, 5,000 updates and evaluate each 1,000-update checkpoint. The [full-schedule comparisons](results.md) took 4 h 13 min and 8 h 45 min respectively on one L40S in `nvgrail/grail:1.1.0`. For the earlier manipulation workstation benchmark, see [outputs and runtime](#output-layout). The full stairs schedule has no workstation timing or memory measurement.

## Full training schedule

<span id="run-the-full-schedule"></span>

In the tutorial's configured `sonic` shell, start at the checkout root. Reuse the existing `terrain/` or `handoff/` and pick an unused training output. The terrain packer refuses an existing directory, and neither launcher resumes training in place.

<span id="stairs"></span>

**Stairs**

```bash
(
  CASE=/workspace/runs/stairs
  bash scripts/tracking/terrain_finetune.sh "$CASE/tracking-full" "$CASE/terrain"
)
```

<span id="kettlebell"></span>

**Moving object.** The first two lines select the bundled kettlebell. For your own [handoff](retargeting.md#tracking-handoff), change both lines to its directory and motion key. For the renamed lift, use `CASE=/workspace/runs/my-lift` and `KEY=lift01`. Choose `RUN` once and keep it for the [standalone evaluation](#per-object-manipulation-fine-tuning).

```bash
CASE=/workspace/runs/kettlebell
KEY=kettlebell
RUN="$CASE/tracking-full"
HANDOFF_KEY=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["motion_key"])' "$CASE/handoff/handoff.json")
if [ -n "$KEY" ] && [ "$HANDOFF_KEY" = "$KEY" ] && [ ! -e "$RUN" ]; then
  bash scripts/tracking/manipulation_finetune.sh "$RUN" "$CASE/handoff"
else
  echo "Check the handoff motion key and choose a fresh training directory." >&2
fi
```

On a host that cannot render, set `EVAL_NO_VIDEO=1` for the manipulation launcher. Both launchers accept `NUM_ENVS`, `ITERATIONS` and `EVAL_EVERY` overrides. For stairs, expect five numbered checkpoints in `tracking-full/train/` and metrics under `tracking-full/eval/step_*/`. For manipulation, look under `tracking-full/finetune/train/` for checkpoints and `tracking-full/eval_final/<motion-key>/summary.json` for the final evaluation.

### Checkpoints

| Case | Prepare the input | Starting policy and launcher |
|---|---|---|
| Static stairs | `grail.retargeting.terrain` writes the packed library. | `terrain_stairs/last.pt`, selected by `terrain_finetune.sh`. |
| Other fixed terrain, such as a slope or curb | The terrain packer requires a fixed collider and the G1 terrain target. | The launcher selects the same stairs checkpoint. Transfer to a new terrain needs its own evaluation; only the stairs example is verified here. |
| Moving floor or tabletop object | `grail.retargeting.manipulation.handoff` binds the source. The manipulation launcher packs it, including its support. | `manip_warmstart/last.pt` plus the frozen `sonic_manipulation_base/last.pt` decoder, via `manipulation_finetune.sh`. |

If PhysX runs out of memory, keep the pack or handoff and [retry training in a new directory](troubleshooting.md). If evaluation stops early, inspect the recorded completion and error for that checkpoint. A checkpoint on disk confirms a training save; the evaluation measures the motion.

## Stair policy rollout

<span id="watch-a-stair-policy-rollout"></span>

The short stairs command writes metrics. To see the **simulated policy**, enable the video recorder during a separate evaluation. The command below was run with the 5,000-update stairs checkpoint on one L40S, driver 580.95.05, in `nvgrail/grail:1.1.0`. Use a host where Isaac Sim RTX rendering works. On the tested RTX 6000 Ada workstation with driver 595.99.02, Isaac Sim's rendering app crashed at startup even with an empty stage; the local metrics-only path worked.

Start at the checkout root in the same configured `sonic` shell as training. Set `STAIRS` to the case containing the completed training run and keep a fresh output directory. The full-schedule launcher above creates the checkpoint used here. This evaluates one environment without training it again.

```bash
STAIRS=/workspace/runs/stairs
cd imports/SONIC
OUT="$STAIRS/tracking-full/eval_video_step_005000"
test ! -e "$OUT" && \
python gear_sonic/eval_agent_trl.py "++checkpoint=$STAIRS/tracking-full/train/model_step_005000.pt" \
  ++num_envs=1 ++headless=true '++eval_callbacks=[im_eval]' ++run_eval_loop=false \
  ++use_wandb=false "++eval_output_dir=$OUT/metrics" \
  ++manager_env.config.render_results=true ++manager_env.config.render_width=960 ++manager_env.config.render_height=720 \
  '++manager_env.config.eval_camera_offset=[-2.5,1.2,1.1]' ++manager_env.config.max_render_envs=1 \
  '++manager_env.config.render_env_ids=[0]' ++manager_env.config.render_frame_skip=1 ++manager_env.config.render_skip_first_frame=false \
  ++manager_env.config.render_terminal_overlay=false "++manager_env.config.save_rendering_dir=$OUT/video" \
  ++manager_env.recorders._target_=gear_sonic.envs.manager_env.mdp.recorders.RecordersCfg \
  ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
  "++manager_env.recorders.render_envs.video_save_path=$OUT/video" ++manager_env.recorders.render_envs.video_quality=7
```

Expect `video/000000.mp4` beneath the new output directory. The recorded run closed the video writer after 500 frames. The clip decodes at 960 × 720, 50 fps; the simulation loop took about 71 seconds. Render-only evaluation prints progress and joint error. The `$OUT/metrics` override can remain empty or absent. Use the launcher's separate checkpoint evaluation for those metrics. The [stairs result](results.md#stair-climb) shows this recording. To probe rendering before a full run, select `tracking/train/model_step_000004.pt` from the short tutorial and a new output directory. If the renderer exits during startup, use the saved metrics and open your own `retargeting.mp4` for the kinematic reference. On a rendering-capable host use the same case, checkpoint and packed scene to see your policy.

The rest of this page covers your own motion libraries and the released reference policies.

<span id="install"></span>

## Preparing retargeted data for training

<span id="choose-a-policy-and-prepare-the-motion-library"></span>

Use the tutorials above for the two bundled motions. For a different dataset choose the experiment that matches its support, hand actions and object motion. The {src}`experiment files <imports/SONIC/gear_sonic/config/exp/manager/universal_token/>` select the observation layout, controller, rewards, randomization and termination rules. The trainer reads these files and the motion libraries in the **sonic** environment. Follow [GRAIL Installation Guide](installation.md) and [tracking downloads](downloads.md#tracking-weights) first.

### Task-general tracking configs overview

| Task | Experiment under `manager/universal_token/` | Data and starting weights |
|---|---|---|
| Table or floor pick-up | `hoi/pnp_table`, `hoi/pnp_ground` | Paired robot and object motions, object USDs and per-object shape features. The installed warm start is `models/pnp_table/last.pt` or `models/pnp_ground/last.pt`. |
| Table or floor advanced manipulation | `hoi/advanced_manip_table`, `hoi/advanced_manip_ground` | Same library interface. For fine-tuning supply a checkpoint trained with the matching observation and action layout. |
| Reviewed G1 grasp, including kettlebell | `hoi/advanced_manip_ground_unified_contactfirst_finetune` | A per-object contact handoff, including the actual support for a tabletop scene. The launcher binds the actor warm start and frozen decoder separately. |
| Scene-aware tracking for terrain traversal | `scene/terrain_tracking` | Robot motion, object motion and matching static USD per motion key. The Quick Start warm start is for stairs. The separate `terrain_release` reference also covers curb, slope and sitting. |

Choose by the scene used in generation. `indoor2-pickup-table` and `indoor2-pickup-ground` use `pnp_table` and `pnp_ground`. `indoor2-manipulation` and `indoor1-manipulation` use the table and ground advanced-manipulation experiments respectively. For `indoor1-terrain-curb`, `indoor1-terrain-slope`, `indoor1-syn-stairs`, `indoor1-gen-stairs` or `indoor1-sitting`, select `scene/terrain_tracking` and pack the matching collider.

A conventional pick-up library contains a robot trajectory, an object trajectory, a collision USD and a per-object shape feature stored as a basis-point-set (BPS) array. The [reference download](downloads.md#reference-policies-and-motion-libraries) includes this complete 250-frame record with the filename stem `pickup_table__apple_0__000`. On the [standard mount](installation.md#gpu-container), the host directory `../grail-demo/runs/reference/data/pickup_table/` appears as `/workspace/runs/reference/data/pickup_table/`.

```text
robot/pickup_table__apple_0__000.pkl        G1 trajectory with left and right hand actions
objects/pickup_table__apple_0__000.pkl      object trajectory and per-frame hand-contact labels
object_usd/pickup_table__apple_0__000.usd   collision scene, plus any assets referenced by this USD
bps/pickup_table__apple_0__000.npy          shape feature vector for this motion's object
```

Use the common **filename stem** for motion selection. In the released apple pair, each PKL contains one record under the same original capture ID, which differs from its published filename. The robot record has 29 joint positions in `dof`, 14 in `hand_dof_pos`, and `hand_action_left` and `hand_action_right`. The object record has `contact_points_left_hand` and `contact_points_right_hand`. Keep the library’s shared `bps/_basis.npy` and USD dependencies with it. Substitute a key present in your prepared library. The bundled kettlebell instead starts with a reviewed native handoff and uses the per-object launcher above.

If your **conventional** library still lacks hand actions, the shipped {blob}`process.sh <grail/retargeting/scripts/process.sh>` takes a library name under `data/motion_lib/` and writes `<name>_ha/{robot,objects,meta}`. It expects the original per-motion metadata and matching keys within each input PKL. Use its processed robot/object directories alongside the original collision USDs. The {blob}`compute_bps.sh <grail/retargeting/scripts/compute_bps.sh>` wrapper takes the original library name and writes `<name>/bps/<stem>.npy` and `_basis.npy` from the USDs. The downloadable apple record already has the hand actions, shape array and table pose in `meta/`.

### Quick smoke test

<span id="configure-a-single-gpu-smoke-test"></span>

Start a Bash shell at the checkout root. Set `DATA_DIR` to your prepared table pick-up library and `RUNS` to a writable location. The reference download supplies the illustrated files. For your own library, replace both its path and motion key. Choose a new output name for each invocation. Keep the arrays in the same shell for the following recipes.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
cd imports/SONIC
DATA_DIR=/workspace/runs/reference/data/pickup_table
PICKUP_KEY=pickup_table__apple_0__000
BPS_DIR="$DATA_DIR/bps"
RUNS=/workspace/runs/pickup
EXP=manager/universal_token/hoi/pnp_table
DATA=(
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=$DATA_DIR/robot"
  "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$DATA_DIR/objects"
  "++manager_env.config.object_usd_path=$DATA_DIR/object_usd"
  "++manager_env.commands.motion.motion_lib_cfg.bps_dir=$BPS_DIR"
)
ARGS=("+exp=$EXP" headless=true ++use_wandb=false "${DATA[@]}")
python -u gear_sonic/train_agent_trl.py "${ARGS[@]}" num_envs=4 \
  ++algo.config.num_learning_iterations=3 ++callbacks.model_save.save_frequency=3 \
  ++manager_env.config.gpu_collision_stack_size_exp=28 "++experiment_dir=$RUNS/smoke"
```

The earlier four-environment smoke recipe took about two minutes on one L40. That is a planning figure for this type of smoke test. Look for scene creation, rollout, PPO and checkpoint writing. Look for the resolved config and numbered checkpoint in `$RUNS/smoke`. To inspect the composed settings without allocating a simulator, add `--cfg job` after the script name. The four public pick-up/manipulation configs and the terrain recipe below were checked by configuration composition in the release image. The measured short training runs on this page are the stairs and manipulation tutorials.

<span id="finetuning-a-pick-up-policy"></span>

## Training

<span id="train-and-fine-tune-on-your-own-library"></span>

### Pick-up and advanced manipulation

Start a new task policy using the selected experiment and the released SONIC action decoder. Choose one of the four experiment names in the table above when constructing `ARGS`. To fine-tune instead, give the trainer a compatible task checkpoint and a separate output directory. For ground pick-up replace both the table experiment and the checkpoint. For advanced manipulation use your matching advanced-manipulation checkpoint.

```bash
CHECKPOINT_DIR="$RUNS/train"
python -u gear_sonic/train_agent_trl.py "${ARGS[@]}" num_envs=2048 \
  "++experiment_dir=$CHECKPOINT_DIR"
```

```bash
CHECKPOINT_DIR="$RUNS/finetune"
python -u gear_sonic/train_agent_trl.py "${ARGS[@]}" num_envs=2048 \
  ++resume=true "++checkpoint=$PWD/models/pnp_table/last.pt" \
  ++algo.config.num_learning_iterations=10000 "++experiment_dir=$CHECKPOINT_DIR"
```

After either run, continue to [evaluate one saved pick-up policy](#tracking-released-data) with the same `CHECKPOINT_DIR` and `PICKUP_KEY`.

<span id="finetuning-a-terrain-policy"></span>

### Terrain-aware tracking

Starting in `imports/SONIC`, set `LIB` to the terrain pack from the stairs tutorial or a library with the same structure. The trainer pairs robot files with terrain USDs by stem. The flat placeholder supplies a collider for non-terrain environments. To mix in flat motion, set `++manager_env.config.flat_motion_dir` and `++manager_env.config.flat_to_terrain_ratio`. With a flat library, a ratio of `R` assigns approximately one terrain environment per group of `R+1`; zero uses all terrain. Without an explicit `flat_usd_path`, the loader first checks `flat_placeholder.usd` at the library root, then under `object_usd/`. Each rank records its assigned motion stems in `/tmp/rank_<R>_motion_keys.txt`.

```bash
TERRAIN_CASE=/workspace/runs/stairs
LIB="$TERRAIN_CASE/terrain"
CHECKPOINT_DIR="$TERRAIN_CASE/train-new"
TERRAIN=(+exp=manager/universal_token/scene/terrain_tracking headless=true ++use_wandb=false
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=$LIB/robot"
  "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$LIB/objects"
  "++manager_env.config.terrain_motion_dir=$LIB"
  "++manager_env.config.flat_usd_path=$LIB/flat_placeholder.usd")
python -u gear_sonic/train_agent_trl.py "${TERRAIN[@]}" num_envs=4096 \
  "++experiment_dir=$CHECKPOINT_DIR"
```

For a direct stairs fine-tune, initialize from the stairs policy and save each 1,000-update checkpoint. The [full-schedule launcher](#full-training-schedule) also evaluates each save after training exits.

```bash
CHECKPOINT_DIR="$TERRAIN_CASE/finetune-new"
python -u gear_sonic/train_agent_trl.py "${TERRAIN[@]}" num_envs=2048 \
  ++warm_resume=true "++checkpoint=$PWD/models/terrain_stairs/last.pt" \
  "++experiment_dir=$CHECKPOINT_DIR" \
  ++algo.config.num_learning_iterations=5000 ++callbacks.model_save.save_frequency=1000 \
  ++callbacks.model_save.save_last_frequency=20
```

After the direct fine-tune, leave `CHECKPOINT_DIR="$TERRAIN_CASE/finetune-new"`. To evaluate the full-schedule launcher’s saves instead, set `CHECKPOINT_DIR=/workspace/runs/stairs/tracking-full/train`. Run the loop from `imports/SONIC` after the selected run has saved all five checkpoints. Each invocation uses a single environment and the scene paths recorded in the checkpoint configuration. Keep the terrain library available at those paths.

```bash
for step in 001000 002000 003000 004000 005000; do
  python gear_sonic/eval_agent_trl.py "++checkpoint=$CHECKPOINT_DIR/model_step_${step}.pt" \
    ++num_envs=1 ++headless=true '++eval_callbacks=[im_eval]' ++run_eval_loop=false \
    ++use_wandb=false "++eval_output_dir=$CHECKPOINT_DIR/recheck/step_${step}"
done
```

### Per-object manipulation fine-tuning

<span id="fine-tune-the-object-aware-manipulation-policy"></span>

Use the [kettlebell handoff and pack](tutorials/kettlebell.md). A single invocation accepts reviewed handoffs for one object group. {blob}`scripts/tracking/manipulation_finetune.sh` reads the pack index and gives the trainer the matching robot motions, objects, shape features and physics settings. It checks the actor loaded from `manip_warmstart/last.pt` against the warm-start tensors before the first rollout. The warm start restores the parts of the policy that command the hands and control how much training actions vary. The critic and optimizer start fresh, and the SONIC action decoder stays frozen. Changing the object's mass or table support changes the simulation even if the trajectory stays the same.

| Setting | Released per-object recipe |
|---|---|
| Control and PPO | 50 Hz control, 200 Hz physics, 24 rollout steps per environment, 5 epochs and 4 mini-batches. The setup uses 2,048 environments on one GPU. |
| Initialization | Actor-only checkpoint (984 observations, 66 actions). The decoder uses both `models/sonic_manipulation_base/model_config.yaml` and `last.pt`. |
| Sampling and hands | Difficulty sampling warms up over 100 completed episodes, with a uniform mixture of 0.10. The learned hand closure/opening slew rates are 4/s and 5/s. |
| Learning rates | Configured actor rate `2e-5`, critic rate `1e-3`. In the checked run both shared optimizer groups started at `2e-5`. Training episodes last 10 s. |
| Randomization | The recipe disables observation noise, object start-position randomization, object/table resizing, rigid-body mass and material randomization. |
| Sampling details | Success floor 0.10, minimum weight 0.05, beta cap 3 and uniform mixture 0.10. |
| Outputs | `finetune/actor_initialization.json` records the warm-start check. `finetune/per_motion_training_metrics.json` and `.jsonl` record training. The default saves and evaluates every 1,000 updates. Set `EVAL_EVERY=0` for final evaluation only. |

The launcher sets the paths from the selected handoff and pack, then starts the trainer with the arguments below. The same command runs in a fresh training directory for each object. To adapt the recipe, keep the decoder and warm-start checks together.

::::{dropdown} Show the launcher arguments

```{literalinclude} ../../scripts/tracking/manipulation_finetune.sh
:language: bash
:start-at: TRAIN_ARGS=(
:end-before: Evaluate after the trainer
```

::::

If **that same run** was interrupted after a save, restart the trainer from its saved configuration. Reuse the same scene paths and the absolute update target (5,000 includes completed updates). From `imports/SONIC` in the original environment, with `RUN` set to that manipulation run:

```bash
CHECKPOINT_DIR="$RUN/finetune/train"
export JOINT_ROSTER_MANIFEST=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["groups"][0]["roster_path"])' "$RUN/pack/tracking_pack.json")
export JOINT_ROSTER_DIRECT_DIRS=""
python gear_sonic/train_agent_trl.py --config-path "$CHECKPOINT_DIR" --config-name config \
  ++exp_base=manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_finetune \
  "++checkpoint=$CHECKPOINT_DIR/last.pt" "++experiment_dir=$CHECKPOINT_DIR" \
  ++resume=true ++warm_resume=false ++callbacks.contactfirst.verify_actor_initialization=false \
  ++algo.config.num_learning_iterations=5000
```

The fresh-actor check expects update zero and an empty optimizer, so it must be off on restart. The per-object experiment enables a separate exact-resume check for the model, optimizer, scheduler and training step. When the checkpoint and experiment directory differ, the trainer selects a warm start. It can load available optimizer state but skips the old scheduler and counters. The direct terrain fine-tune uses that warm-start mode. The restart settings were checked with a saved configuration; no interrupted training session was resumed in this verification.

To evaluate a saved manipulation policy checkpoint separately, run the following from `imports/SONIC`. Reuse the same `CASE`, `RUN` and `KEY` from the full-schedule launcher above. If you reopened the shell, restore them for your case first. Choose the actual checkpoint from that run and a fresh evaluation directory. Omit `--no-video` to record on a rendering-capable host.

```bash
CHECKPOINT_DIR="$RUN/finetune/train"
CHECKPOINT_FILE=last.pt
HANDOFF_KEY=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["motion_key"])' "$CASE/handoff/handoff.json")
if [ -n "$KEY" ] && [ "$HANDOFF_KEY" = "$KEY" ] && [ -f "$CHECKPOINT_DIR/$CHECKPOINT_FILE" ]; then
  python gear_sonic/eval_contactfirst_trl.py --pack "$RUN/pack" --motion "$KEY" \
    --checkpoint "$CHECKPOINT_DIR/$CHECKPOINT_FILE" \
    --output "$RUN/eval-standalone/${CHECKPOINT_FILE%.pt}/$KEY" --no-video
else
  echo "Check the selected case, motion key and saved checkpoint." >&2
fi
```

One environment follows the reference from frame zero with deterministic actions. Check `reference_coverage`, termination and object pose errors in `summary.json` and `output/evaluation.json`. Coverage 1.0 with no early termination means that motion reached its end. The [results page](results.md) gives the five checkpoint evaluations for each worked example.

### Multi-node `accelerate` template

<span id="multiple-gpus-or-nodes"></span>

If you enable W&B, use the public `+opt=wandb` configuration in place of the offline example’s `++use_wandb=false`. The experiment writes its resolved configuration to disk even when W&B is disabled.

Use the same Bash `ARGS` array from the smoke test on each host. Each worker needs the same code, decoder, data and writable shared experiment path. For eight GPUs on one machine:

```bash
accelerate launch --multi_gpu --num_processes=8 gear_sonic/train_agent_trl.py \
  "${ARGS[@]}" num_envs=2048 "++experiment_dir=$RUNS/eight-gpu"
```

For eight nodes with eight GPUs each, set `MASTER_ADDR` and `MASTER_PORT` to the reachable coordinator and port, and `MACHINE_RANK` to 0 through 7, one value per node. Run the command once per node. Accelerate's process count is **total** across nodes. These launch flags were checked in the release image. The distributed jobs were not run. See the [Accelerate launch guide](https://huggingface.co/docs/accelerate/basic_tutorials/launch) and [CLI reference](https://huggingface.co/docs/accelerate/package_reference/cli) for host and rank settings.

```bash
accelerate launch --multi_gpu --num_machines=8 --num_processes=64 \
  --machine_rank="$MACHINE_RANK" --main_process_ip="$MASTER_ADDR" \
  --main_process_port="$MASTER_PORT" gear_sonic/train_agent_trl.py \
  "${ARGS[@]}" num_envs=2048 "++experiment_dir=$RUNS/multi-node"
```

## Tracking released data

<span id="evaluate-a-released-motion-library"></span>

Use `gear_sonic/eval_agent_trl.py` for a single pick-up or terrain checkpoint. It reads the saved `config.yaml` beside the checkpoint or one directory up. To reproduce a released-data policy, choose its [reference checkpoint and matching library](downloads.md#reference-policies-and-motion-libraries). The Quick Start downloader’s `pnp_table`, `pnp_ground` and `terrain_stairs` directories contain warm starts. Point at the corresponding local release data. For a **pick-up** example, run the block below in the same `imports/SONIC` shell with the previously defined `DATA` array. Keep `PICKUP_KEY` from your library setup. For your own policy, keep `CHECKPOINT_DIR` from training and select `last.pt` or a numbered save in that directory. For the downloaded table **reference** policy, set `CHECKPOINT_DIR=/workspace/runs/reference/checkpoint/SONIC/models/pnp_table_release` before running this block. Override both decoder paths because the historical saved config can refer to its training-time decoder. Reset rank and world size to evaluate locally.

```bash
CHECKPOINT_FILE=last.pt
test -f "$CHECKPOINT_DIR/$CHECKPOINT_FILE" && test -f "$DATA_DIR/robot/$PICKUP_KEY.pkl" &&
  python gear_sonic/eval_agent_trl.py \
  "++checkpoint=$CHECKPOINT_DIR/$CHECKPOINT_FILE" ++headless=true ++num_envs=1 \
  '++eval_callbacks=[im_eval]' ++run_eval_loop=false ++use_wandb=false \
  "++eval_output_dir=$RUNS/eval-$(basename "$CHECKPOINT_DIR")/${CHECKPOINT_FILE%.pt}/$PICKUP_KEY" "${DATA[@]}" \
  "++manager_env.config.action_transform_module_cfg=$PWD/models/sonic_manipulation_base/model_config.yaml" \
  "++manager_env.config.action_transform_module_checkpoint=$PWD/models/sonic_manipulation_base/last.pt" \
  "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[$PICKUP_KEY]" \
  ++manager_env.commands.motion.motion_lib_cfg.multi_thread=false \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0
```

To watch the released table policy, use this complete example from `imports/SONIC` in the `sonic` environment. It uses the apple record and policy from [Downloads](downloads.md#reference-policies-and-motion-libraries) and the decoder from tracking setup. Supply a fresh video directory. Set `MOTION_KEYS` to comma-separated filename stems for additional downloaded motions.

```bash
REF=/workspace/runs/reference
DATA_DIR="$REF/data/pickup_table"
CHECKPOINT="$REF/checkpoint/SONIC/models/pnp_table_release/last.pt"
MOTION_KEYS=pickup_table__apple_0__000
NUM_ENVS=8
SAVE_DIR="$REF/renderings/apple-table-policy"
python -u gear_sonic/eval_agent_trl.py "++checkpoint=$CHECKPOINT" \
  ++headless=true "++num_envs=$NUM_ENVS" '++eval_callbacks=[im_eval]' ++run_eval_loop=false ++use_wandb=false \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=$DATA_DIR/robot" \
  "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$DATA_DIR/objects" \
  "++manager_env.config.object_usd_path=$DATA_DIR/object_usd" \
  "++manager_env.commands.motion.motion_lib_cfg.bps_dir=$DATA_DIR/bps" \
  "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[$MOTION_KEYS]" \
  ++manager_env.commands.motion.motion_lib_cfg.multi_thread=false \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
  "++manager_env.config.action_transform_module_cfg=$PWD/models/sonic_manipulation_base/model_config.yaml" \
  "++manager_env.config.action_transform_module_checkpoint=$PWD/models/sonic_manipulation_base/last.pt" \
  ++manager_env.config.render_results=true ++manager_env.config.env_spacing=10.0 \
  "++manager_env.config.max_render_envs=$NUM_ENVS" "++manager_env.config.save_rendering_dir=$SAVE_DIR" \
  ++manager_env.recorders._target_=gear_sonic.envs.manager_env.mdp.recorders.RecordersCfg \
  ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
  "++manager_env.recorders.render_envs.video_save_path=$SAVE_DIR" ++manager_env.recorders.render_envs.video_quality=5
```

| Override group | What it controls |
|---|---|
| `checkpoint`, both decoder paths | Load the released table policy and the action decoder installed on this machine. The checkpoint’s saved configuration can point to its original training machine. |
| Motion files, collision USD, BPS and motion filter | Load the paired robot and object trajectories, geometry and shape features for the named motion stems. |
| Loader threads, shard size and rank | Load the records in one process and reset the training-time shard to rank zero of one. |
| `headless`, `eval_callbacks`, `run_eval_loop`, `use_wandb` | Run without a display window. The imitation-evaluation callback performs the rollout. Disabling the subsequent evaluation loop avoids a second rollout. Local evaluation needs no W&B run. |
| `num_envs`, `env_spacing`, `render_results`, `max_render_envs` | Create eight simulated environments 10 m apart and enable their cameras. Setting `max_render_envs` switches the evaluation callback to render-only mode and caps recording at eight environments. |
| Recorder targets, both video directories and `video_quality` | Install the environment video recorder and send its numbered MP4s to the same fresh output directory. Quality 5 sets the encoder's quality parameter. |

Use the preceding metrics command to collect `metrics_eval.json`. The metrics and video examples both loaded the published apple checkpoint configuration and resolved its motion files and decoder in the release image. Video recording requires a working Isaac renderer. The [stairs policy replay](#stair-policy-rollout) includes a measured rendering run.

## Output layout

<span id="outputs-runtime-and-troubleshooting"></span>

A direct training job writes to its resolved `experiment_dir`. Look for `config.yaml`, `model_step_NNNNNN.pt` and `last.pt`. Save frequency is controlled by `callbacks.model_save`. The default is `<base_dir>/<project_name>/<experiment_name>-<timestamp>/`. For table pick-up the base is `logs_rl` and the project is `GRAB_Tracking`. Look for `meta.yaml` beside the saved config and TensorBoard events when that logger is enabled. See the {blob}`SONIC README <imports/SONIC/README.md>` and {src}`SONIC guides <imports/SONIC/docs/>` for its training stack. An explicit output path makes it easier to find a run. The launchers place metrics for each evaluated checkpoint outside the training directory. Inspect both the checkpoint and its matching evaluation before using a policy.

The November 2026 single-object manipulation finetune benchmark on a 48 GB RTX 6000 Ada reported about 29 GB in use and roughly 5 s per update with 2,048 environments. That rate projects to about 7 h for 5,000 updates. Scene and cache startup took about 8 min, an evaluation without video about 30 s per motion, and packing a 121-frame motion about 5 min on one CPU core. These figures come from the earlier manipulation benchmark. Workstation timings for full stairs training and a measured 5,000-update duration for these exact tutorial inputs in the standard image are still unavailable.

The full tutorial references on one L40S took **4 h 13 min for stairs** and **8 h 45 min for kettlebell**. The per-object release reported ground/table completion rates of 55%/95% for the 420-object set and 62%/94% for the 200-object set. These are per-motion deterministic final checks on the training references. The released manipulation sets contain [600 ground motions](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/tree/main/data/manip_ground) and [673 table motions](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/tree/main/data/manip_table). These population rates and the two tutorial runs answer different questions. The individual checkpoint measurements and hardware are in [worked results](results.md).

### Troubleshooting

| Symptom | What to inspect |
|---|---|
| No motions selected or an empty rank | Match stems and support type. For pick-up evaluation set the single-process shard overrides above. For terrain inspect the `TerrainAutoDiscover`, `PerRankUSD` and `PerRankMotion` startup lines. |
| A decoder path from an old checkpoint cannot be opened | Set both decoder overrides to the downloaded model config and weights. Start in `imports/SONIC`, or supply absolute paths. |
| `reward_grasp: gate_with_contact_label=True but no contact label found`, or wrong pose width | Check the paired hand-action robot motion, object sequence, reviewed contact map and hand/body target before packing. Conventional pick-up data can require `contact_points_left_hand` and `contact_points_right_hand`. |
| The checkpoint saved but the object falls or the episode ends early | Compare the paired robot/object replay, support, mass and evaluated motion key. Inspect termination and pose errors for the same checkpoint. |
| Simulation runs out of memory, or RTX video fails at startup | Follow the [case-specific retry instructions](troubleshooting.md). Keep the handoff/terrain pack and give the next run a fresh output. Metrics evaluation can run without RTX video. |
