# Task General Tracking

Task-general tracking trains physics-based policies on retargeted GRAIL motion
libraries: pick-and-place, manipulation, terrain-aware tracking, and locomotion.
The training implementation lives in the
{src}`imports/SONIC <imports/SONIC/>`
vendored release tree.

This page documents the GRAIL-specific pieces:

- The four self-contained HOI release configs (`pnp_*`, `advanced_manip_*`)
  used to train pick-and-place and manipulation policies.
- The shared `terrain_tracking` config used for scene/terrain-aware tracking.

For lower-level training internals see
{blob}`imports/SONIC/README.md`
and
{src}`imports/SONIC/docs/`.

## Install

The published Docker image already contains the `sonic` policy-training
environment:

```bash
conda activate sonic
```

For a source installation, create it once and then activate it:

```bash
bash scripts/setup/install_env_sonic.sh
conda activate sonic
```

The source installer sets up Isaac Lab, GRAIL, the training stack, and GMR for
explicit legacy SMPL-X inputs. Default SOMA-to-G1 retargeting uses its own
Python 3.12 environment and does not run inside `sonic`.

## Checkpoints

The pretrained base behavior-model, task-specific warmstart
checkpoints for finetuning and reference checkpoints trained on released data are all fetched by the project-wide setup script:

```bash
bash scripts/setup/download_checkpoints.sh           # all submodules
bash scripts/setup/download_checkpoints.sh \
    --skip-gem-smpl --skip-gem-x --skip-mose3 \
    --skip-foundationpose --skip-hunyuan3d           # SONIC only
```

This lands them under `imports/SONIC/models/`:

```
imports/SONIC/models/
├── sonic_manipulation_base/   # pretrained WBC base for pickup and manipulation: last.pt + model_config.yaml
├── pnp_table_warmstart/       # pickup table warm start: last.pt + config.yaml
├── pnp_ground_warmstart/      # pickup ground warm start: last.pt + config.yaml
├── terrain_stairs_warmstart/  # terrain (stairs only) warm start: last.pt + config.yaml
├── pnp_table_release/         # pickup table reference: last.pt + config.yaml
├── pnp_ground_release/        # pickup ground reference: last.pt + config.yaml
└── terrain_release/           # terrain (curb + slope + stairs + sitting) reference: last.pt + config.yaml
```

Store path references in experiment configs as relative to
`imports/SONIC/`, not to the GRAIL root — e.g. `models/pnp_table_warmstart/last.pt`.
The training commands below assume `cd imports/SONIC` first, so those
relative paths resolve correctly.

## Preparing retargeted data for training

The current training loaders consume the established GRAIL motion-library
layout, not the raw 43-joint G1+DEx3 CSV emitted by the default SOMA
Retargeter. Released data
and successful-rollout exports already use this layout. The legacy SMPL-X/GMR
pipeline can also produce it directly; new SOMA results must first be replayed,
reviewed, and packaged for SONIC.

1. Prepare a reviewed motion library containing `robot/`, `objects/`,
   `object_usd/`, and `meta/`. For legacy SMPL-X reconstruction, the GRAIL
   retargeting pipeline writes a `<name>_ha/` directory with this layout.
2. Move the retargeted folder to `data/motion_lib/<name>_ha/` — that
   prefix is what the training config loaders expect.
3. For multi-object HOI sweeps, also place the BPS encodings under
   `data/motion_lib/<name>/bps/` (BPS is data-only; it is not
   parameterized by the hand-action variant).
4. For legacy terrain data, re-retarget with `--zero_out_wrist` to skip hand
   IK. For SOMA manipulation inputs, preserve the 14 DEx3 finger joints and
   validate the robot + object + table replay before packaging it.

## Task-general tracking configs overview

Our released tracking configs are listed in `imports/SONIC/gear_sonic/config/exp/manager/universal_token/`:


| Path                             | Purpose                                                    | Supported scene config(s) |
|----------------------------------|------------------------------------------------------------|--------------------------|
| `scene/terrain_tracking.yaml`    | Shared height-map + object-state terrain-aware tracking config | `indoor1-sitting`, `indoor1-terrain-curb`, `indoor1-terrain-slope`, `indoor1-syn-stairs`, `indoor1-gen-stairs` |
| `hoi/pnp_table.yaml`             | Self-contained tabletop pick-up release config             | `indoor2-pickup-table` |
| `hoi/pnp_ground.yaml`            | Self-contained ground pick-up release config               | `indoor2-pickup-ground` |
| `hoi/advanced_manip_table.yaml`  | Self-contained advanced-manipulation tabletop config       | `indoor2-manipulation` |
| `hoi/advanced_manip_ground.yaml` | Self-contained advanced-manipulation ground config         | `indoor1-manipulation` |

The release configs share the same launch pattern: choose the Hydra config,
then pass runtime data paths through Hydra overrides. **Note: please pick the right tracking config given the scene config used in 2D HOI generation**

## Quick smoke test

A single-GPU, 4-env, 3-iteration run against the `pnp_table` release config —
enough to verify the install end-to-end. Completes in ~2 minutes on a single
L40. Set `DATA_DIR` and `BPS_DIR` to a retargeted motion library prepared as
described above.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline

cd imports/SONIC
python -u train_agent_trl.py \
    +exp=manager/universal_token/hoi/pnp_table \
    num_envs=4 headless=True \
    ++algo.config.num_learning_iterations=3 \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.object_usd_path=${DATA_DIR}/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${BPS_DIR}
```

## Training

### Pick-up and advanced manipulation

All four release configs share a single launch shape — only the Hydra config
name changes. Set `DATA_DIR` to your retargeted motion library (with
`robot/`, `objects/`, `object_usd/` subdirs) and `BPS_DIR` to the matching
BPS encodings; see [Preparing retargeted data for training](#preparing-retargeted-data-for-training)
for the expected layout. The example script below will launch training with
a single node using 8 GPUs.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC
accelerate launch --num_processes=8 train_agent_trl.py \
    +exp=${HYDRA_CONFIG} \
    num_envs=2048 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.object_usd_path=${DATA_DIR}/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${BPS_DIR}
```

Available release configs:

| Sweep                   | `HYDRA_CONFIG`                                       |
|-------------------------|------------------------------------------------------|
| `pnp_table`             | `manager/universal_token/hoi/pnp_table`              |
| `pnp_ground`            | `manager/universal_token/hoi/pnp_ground`             |
| `advanced_manip_table`  | `manager/universal_token/hoi/advanced_manip_table`   |
| `advanced_manip_ground` | `manager/universal_token/hoi/advanced_manip_ground`  |

Pick-up and manipulation launch inputs:

| Flag | Effect |
|------|--------|
| `+exp=${HYDRA_CONFIG}` | Selects one release config from the table above. |
| `++manager_env.commands.motion.motion_lib_cfg.motion_file=<path>` | Robot-motion directory, usually `${DATA_DIR}/robot`. |
| `++manager_env.commands.motion.motion_lib_cfg.object_motion_file=<path>` | Object-motion directory, usually `${DATA_DIR}/objects`. |
| `++manager_env.config.object_usd_path=<path>` | Object USD directory, usually `${DATA_DIR}/object_usd`. |
| `++manager_env.commands.motion.motion_lib_cfg.bps_dir=<path>` | BPS encoding directory for multi-object pick-up / manipulation data. |

#### Finetuning a pick-up policy

To continue from an existing pick-up run, use the matching bundles
(downloaded by `download_checkpoints.sh` above):

| Config | Warm-start checkpoint |
|--------|-----------------------|
| `manager/universal_token/hoi/pnp_table`  | `models/pnp_table_warmstart/last.pt`  |
| `manager/universal_token/hoi/pnp_ground` | `models/pnp_ground_warmstart/last.pt` |

The finetune command below selects the release config and warm-resumes
from the `_warmstart` `last.pt`. Write the new run to a separate
`experiment_dir`.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC
python -u train_agent_trl.py \
    +exp=manager/universal_token/hoi/pnp_table \
    num_envs=2048 headless=True \
    ++resume=True \
    ++checkpoint=models/pnp_table_warmstart/last.pt \
    experiment_dir=${FINETUNE_DIR} \
    ++algo.config.num_learning_iterations=10000 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.object_usd_path=${DATA_DIR}/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${BPS_DIR}
```

Use `+exp=manager/universal_token/hoi/pnp_ground` and
`++checkpoint=models/pnp_ground_warmstart/last.pt` for ground pick-up data.

### Terrain-aware tracking

The current GRAIL terrain-aware runs use one shared height-map + object-state
config. Set `DATA_DIR` to a retargeted dataset root with `robot/`, `objects/`,
and `object_usd/`.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC
python -u train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.terrain_motion_dir=${DATA_DIR}
```

Use the same config for all scene/terrain-aware datasets and pass only the
runtime data paths through Hydra overrides. If the active dataset root does
not include `flat_placeholder.usd`, set
`++manager_env.config.flat_usd_path=<path>` to a compatible placeholder USD.

Terrain launch inputs:

| Flag | Effect |
|------|--------|
| `++manager_env.config.terrain_motion_dir=<path>` | Dataset root with paired `robot/*.pkl` + `object_usd/*.usd` (1:1 stem matching). Auto-discovers all pairs. |
| `++manager_env.config.flat_motion_dir=<path>` | Optional — adds flat (non-terrain) motions, interleaved between terrain envs. |
| `++manager_env.config.flat_usd_path=<path>` | Explicit placeholder USD for non-terrain envs. Falls back to `<terrain_motion_dir>/flat_placeholder.usd` (new layout) or `<terrain_motion_dir>/object_usd/flat_placeholder.usd` (legacy). |
| `++manager_env.config.flat_to_terrain_ratio=R` | Every `(R+1)`th env is terrain; rest are flat. `R=0` → all envs terrain. |

The terrain path emits `/tmp/rank_<R>_motion_keys.txt` per GPU and logs
`[TerrainAutoDiscover]` / `[PerRankUSD]` / `[PerRankMotion]` during init —
grep those to confirm the slicer is doing what you expect.

#### Finetuning a terrain policy

Terrain finetuning uses the same warm-resume pattern: warm-resume from
`models/terrain_stairs_warmstart/last.pt`, write to a new output directory,
and point `DATA_DIR` at the next terrain-aware motion-library partition.
The published warm-start bundle (`models/terrain_stairs_warmstart/`) is the
**stairs** policy only — use it as the warm-start when finetuning on any
stairs-like dataset.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC
python -u train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking \
    num_envs=4096 headless=True \
    ++resume=True \
    ++checkpoint=models/terrain_stairs_warmstart/last.pt \
    experiment_dir=${FINETUNE_DIR} \
    ++algo.config.num_learning_iterations=20000 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.terrain_motion_dir=${DATA_DIR}
```

If the dataset root does not provide `flat_placeholder.usd`, also pass
`++manager_env.config.flat_usd_path=<path-to-placeholder-usd>`.

### Multi-node `accelerate` template

Replace the single-node launcher with the multi-node form. Same
`train_agent_trl.py` command and `${ARGS[@]}`, just different launcher
flags. Example: 8 nodes × 8 GPUs = 64 GPUs.

```bash
accelerate launch \
    --multi_gpu \
    --num_machines=8 \
    --num_processes=64 \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    train_agent_trl.py "${ARGS[@]}" num_envs=2048
```

See the
[Accelerate distributed training guide](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
and
[multi-node launcher docs](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch).

## Output layout

Each run writes to:

```
logs_rl/TRL_G1_Track/manager/<config_path>/<exp_name>-<timestamp>/
├── config.yaml              # full resolved Hydra config
├── model_step_NNNNNN.pt     # checkpoint every N iters (algo.config.save_every)
├── last.pt                  # symlink to the latest step
├── meta.yaml                # wandb_id + misc provenance
└── events.out.tfevents.*    # tensorboard (optional; wandb is primary)
```

W&B run name and project come from the `+opt=wandb` Hydra opt group
(`gear_sonic/config/opt/wandb.yaml`).

## Tracking released data

See {src}`imports/SONIC docs <imports/SONIC/docs/>`
for the full eval-loop and metric workflows. GRAIL does not add eval tooling —
eval runs against the training checkpoint directory format directly, so the
released `_release` reference bundles work as-is.

The `_release` reference checkpoints (fetched by `download_checkpoints.sh`) can
be rolled out on the released motion libraries to reproduce the reference
rollouts as videos. The command below is the single-GPU local form which invokes 
`eval_agent_trl.py` directly (single-shot, respects `+checkpoint=<path>`) in render mode.

Pick a released policy and its matching released motion library:

| Policy | Reference checkpoint | Released motion library (example) |
|--------|----------------------|-----------------------------------|
| `pnp_table`  | `models/pnp_table_release/last.pt`  | `data/pickup_table`  |
| `pnp_ground` | `models/pnp_ground_release/last.pt` | `data/pickup_ground` |
| `terrain`    | `models/terrain_release/last.pt`    | `data/stair_p1`      |

Set `CHECKPOINT` to the reference checkpoint, `DATA_DIR` to the matching
released motion library (with `meta/`, `robot/`, `objects/`, `object_usd/`), `BPS_DIR` to
its BPS encodings (`bps/` only needed for pickup), and `MOTION_KEYS` to a comma-separated list of motion stems
to render:

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC

CHECKPOINT=models/pnp_table_release/last.pt
DATA_DIR=data/pickup_table
BPS_DIR=${DATA_DIR}/bps
MOTION_KEYS="pickup_table__apple_0__000"        # comma-separated motion stems
NUM_ENVS=8
SAVE_DIR="$(dirname ${CHECKPOINT})/renderings/${MOTION_KEYS}"

python -u gear_sonic/eval_agent_trl.py \
    +checkpoint=${CHECKPOINT} \
    headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=${NUM_ENVS} \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.object_usd_path=${DATA_DIR}/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${BPS_DIR} \
    ++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[${MOTION_KEYS}] \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
    ++manager_env.config.render_results=True \
    ++manager_env.config.save_rendering_dir=${SAVE_DIR} \
    ++manager_env.config.env_spacing=10.0 \
    ++manager_env.config.max_render_envs=${NUM_ENVS} \
    ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
    ++manager_env.recorders.render_envs.video_save_path=${SAVE_DIR} \
    ++manager_env.recorders.render_envs.video_quality=5
```

Rendered videos land at `${SAVE_DIR}/*.mp4`. For `pnp_ground`, swap
`CHECKPOINT`/`DATA_DIR` for the ground row above; terrain uses the
`terrain_tracking` config's `terrain_motion_dir` layout instead of `bps_dir`.

Eval-specific overrides (beyond the shared data-path flags) and why they are
needed:

| Flag | Effect |
|------|--------|
| `+checkpoint=<path>` | Single-shot checkpoint to roll out. `eval_agent_trl.py` respects it directly (unlike `eval_exp.py`, which watches for `model_step_*.pt`). |
| `++run_eval_loop=False` + `++manager_env.recorders.render_envs.*` | Render mode: the video is produced by the injected `RenderEnvsRecorderCfg` recorder, not the eval-loop rollout. The recorder must be injected via dot overrides because the saved `config.yaml` ships an empty `recorders` section. |
| `++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[...]` | Restricts the rollout to the listed motion stems. |
| `++manager_env.config.max_render_envs=<N>` | Flips `im_eval` into render-only mode; without it the callback tries to eval the whole motion library and never renders the filtered set. |
| `++...motion_lib_cfg.motion_shard_world_size=1` / `++...motion_shard_rank=0` | Resets the multi-node sharding baked into the trained `config.yaml` (e.g. `64`) so single-GPU eval doesn't slice the motions down to an empty shard. |

## Troubleshooting

| Symptom                                         | Cause / fix                                                                                    |
|-------------------------------------------------|------------------------------------------------------------------------------------------------|
| `reward_grasp: gate_with_contact_label=True but no contact label found` | Retarget output is missing `contact_points_{left,right}_hand` — re-run `process.sh` with `--include_contact_points` (it is the default). |
| Wrong motion-lib format                         | Verify `robot/` contains per-motion pkls with keys `joint_pos`, `hand_action_{left,right}`, `table_pos`. |
