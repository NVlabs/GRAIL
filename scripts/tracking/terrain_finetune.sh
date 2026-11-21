#!/usr/bin/env bash
# Finetune the released stairs policy on a packed terrain library, then evaluate
# each numbered checkpoint. Run from the repository root in the sonic environment.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 OUTPUT_DIR TERRAIN_LIBRARY" >&2
    exit 2
fi
out=$(realpath -m "$1")
library=$(realpath -e "$2")
num_envs=${NUM_ENVS:-2048}
iterations=${ITERATIONS:-5000}
eval_every=${EVAL_EVERY:-1000}

for value in "$num_envs" "$iterations" "$eval_every"; do
    if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_ENVS, ITERATIONS and EVAL_EVERY must be positive integers." >&2
        exit 2
    fi
done
if (( eval_every > iterations )); then
    echo "EVAL_EVERY must be at most ITERATIONS to produce a checkpoint to evaluate." >&2
    exit 2
fi

cd "$(dirname "$0")/../../imports/SONIC"
mkdir -p "$(dirname "$out")"
mkdir "$out"
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/scene/terrain_tracking \
    "num_envs=$num_envs" headless=true ++warm_resume=true ++use_wandb=false \
    "++checkpoint=$PWD/models/terrain_stairs/last.pt" "++experiment_dir=$out/train" \
    "++algo.config.num_learning_iterations=$iterations" "++callbacks.model_save.save_frequency=$eval_every" \
    ++callbacks.model_save.save_last_frequency=20 \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$library/robot" \
    "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$library/objects" \
    "++manager_env.config.terrain_motion_dir=$library" "++manager_env.config.flat_usd_path=$library/flat_placeholder.usd"

for ((step=eval_every; step<=iterations; step+=eval_every)); do
    printf -v checkpoint_step '%06d' "$step"
    python gear_sonic/eval_agent_trl.py "++checkpoint=$out/train/model_step_${checkpoint_step}.pt" \
        ++num_envs=1 ++headless=true '++eval_callbacks=[im_eval]' ++run_eval_loop=false \
        ++use_wandb=false "++eval_output_dir=$out/eval/step_${checkpoint_step}"
done
