#!/usr/bin/env bash
# ============================================================================
# Single-GPU manipulation tracking finetune + deterministic evaluation for one object.
#
# Usage (repository root, `sonic` environment):
#   bash scripts/tracking/manipulation_finetune.sh OUTPUT_DIR HANDOFF_DIR [HANDOFF_DIR ...]
#
# Every HANDOFF_DIR comes from `python -m grail.retargeting.manipulation.handoff` and holds one
# retargeted motion of the same object. The script builds the pack and cohort (GRAIL data
# production), then finetunes and evaluates through SONIC itself:
#   cd imports/SONIC && python gear_sonic/train_agent_trl.py \
#       +exp=manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_finetune ...
#   cd imports/SONIC && python gear_sonic/eval_contactfirst_trl.py --pack ... --motion ...
#
# Environment overrides (production defaults):
#   ITERATIONS=5000   absolute PPO update target
#   NUM_ENVS=2048     training environments
#   CPUS=8            Kit/PhysX threads of the training process
#   EVAL_EVERY=1000   checkpoint evaluation interval after training (0 = final only)
#   EVAL_NO_VIDEO=0   1 = metrics-only evaluations (hosts without RTX rendering)
#   WARMSTART=imports/SONIC/models/manip_warmstart/last.pt
# ============================================================================
set -euo pipefail

if [ "$#" -lt 2 ]; then
    sed -n '2,22p' "$0"
    exit 2
fi
OUT="$(realpath -m "$1")"
shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SONIC="$REPO/imports/SONIC"
ITERATIONS="${ITERATIONS:-5000}"
NUM_ENVS="${NUM_ENVS:-2048}"
CPUS="${CPUS:-8}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_NO_VIDEO="${EVAL_NO_VIDEO:-0}"
WARMSTART="$(realpath "${WARMSTART:-$SONIC/models/manip_warmstart/last.pt}")"
DECODER_CFG="$(realpath "$SONIC/models/sonic_manipulation_base/model_config.yaml")"
DECODER_CKPT="$(realpath "$SONIC/models/sonic_manipulation_base/last.pt")"
export OMNI_KIT_ACCEPT_EULA=Yes WANDB_MODE="${WANDB_MODE:-disabled}" HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 1. Data production (GRAIL): pack + cohort ────────────────────────────────
HANDOFF_ARGS=()
for handoff in "$@"; do
    HANDOFF_ARGS+=(--handoff "$handoff")
done
python -m grail.retargeting.manipulation.pack "${HANDOFF_ARGS[@]}" --num-envs "$NUM_ENVS" --output "$OUT/pack"
PACK="$OUT/pack"

# ── 2. Resolve the single group's cohort view and caches ────────────────────
COHORT=$(cd "$SONIC" && PYTHONPATH="$SONIC" python - "$PACK" "$OUT/finetune" "$NUM_ENVS" <<'PYEOF'
import json, os, sys
pack, out, num_envs = sys.argv[1], sys.argv[2], int(sys.argv[3])
catalog = json.loads(open(os.path.join(pack, "tracking_pack.json")).read())
groups = catalog["groups"]
assert len(groups) == 1, "This launcher drives one object group per run"
group = groups[0]
os.environ["JOINT_ROSTER_MANIFEST"] = group["roster_path"]
os.environ["JOINT_ROSTER_DIRECT_DIRS"] = ""
from gear_sonic.rp2.joint_roster import resolve_rank_local_roster
view = resolve_rank_local_roster(group["roster_path"], num_envs=num_envs, rank=0,
                                 world_size=1, runtime_target_fps_hz=50.0)
from gear_sonic.utils.contactfirst_cache import cache_root
print(group["group_id"], group["roster_path"], group["physics_path"],
      view["motion_dir"], view["object_motion_dir"], view["bps_dir"],
      cache_root(out, "train"), sep="\n")
PYEOF
)
mapfile -t COHORT_PATHS <<< "$COHORT"
if [ "${#COHORT_PATHS[@]}" -ne 7 ]; then
    echo "Expected seven fields from the cohort resolver" >&2
    exit 1
fi
ROSTER="${COHORT_PATHS[1]}"
PHYSICS="${COHORT_PATHS[2]}"
MOTION_DIR="${COHORT_PATHS[3]}"
OBJECT_DIR="${COHORT_PATHS[4]}"
BPS_DIR="${COHORT_PATHS[5]}"
CACHE_ROOT="${COHORT_PATHS[6]}"
export JOINT_ROSTER_MANIFEST="$ROSTER" JOINT_ROSTER_DIRECT_DIRS=""

SAVE_FREQ=$(( ITERATIONS < 1000 ? ITERATIONS : 1000 ))
if [ "$EVAL_EVERY" -gt 0 ]; then SAVE_FREQ="$EVAL_EVERY"; fi
SAVE_LAST=$(python -c "import math,sys; print(math.gcd(20, int(sys.argv[1])))" "$ITERATIONS")
mkdir -p "$OUT/finetune"

TRAIN_ARGS=(
    +exp=manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_finetune
    "num_envs=$NUM_ENVS" headless=true "base_dir=$OUT/finetune"
    "++experiment_dir=$OUT/finetune/train" project_name=grail_manipulation_tracking
    "++algo.config.num_learning_iterations=$ITERATIONS"
    "++contactfirst_runtime.cpu_threads=$CPUS"
    "++callbacks.model_save.save_frequency=$SAVE_FREQ"
    "++callbacks.model_save.save_last_frequency=$SAVE_LAST"
    "++manager_env.config.action_transform_module_cfg=$DECODER_CFG"
    "++manager_env.config.action_transform_module_checkpoint=$DECODER_CKPT"
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION_DIR"
    "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$OBJECT_DIR"
    "++manager_env.commands.motion.motion_lib_cfg.bps_dir=$BPS_DIR"
    "++manager_env.config.object_physics_overrides_path=$PHYSICS"
    "++manager_env.recorders._target_=gear_sonic.envs.manager_env.mdp.group_metrics.make_recorders"
    "++manager_env.recorders.output_root=$OUT/finetune"
    "++manager_env.recorders.roster_manifest=$ROSTER"
    "++manager_env.commands.motion.motion_lib_cfg.motion_tensor_cache.mode=read_write"
    "++manager_env.commands.motion.motion_lib_cfg.motion_tensor_cache.cache_dir=$CACHE_ROOT/training_data/motion_tensor"
    "++manager_env.config.authored_stage_cache.mode=read_write"
    "++manager_env.config.authored_stage_cache.cache_dir=$CACHE_ROOT/training_data/authored_stage"
    "++manager_env.config.authored_stage_cache.attach_mode=root_copy"
    "++manager_env.commands.motion.motion_lib_cfg.joint_roster_scene_plan_cache_mode=read_write"
    "++manager_env.commands.motion.motion_lib_cfg.joint_roster_scene_plan_cache_dir=$CACHE_ROOT/training_data/rank_scene_plan"
    "++use_wandb=false"
    "++resume=false" "++warm_resume=false" "++flexible_load=false"
    "checkpoint=$WARMSTART"
    "++callbacks.contactfirst.actor_init_checkpoint=$WARMSTART"
    "++callbacks.contactfirst.verify_actor_initialization=true"
    "++callbacks.contactfirst.run_name=$(basename "$OUT")"
)
(cd "$SONIC" && python -u gear_sonic/train_agent_trl.py "${TRAIN_ARGS[@]}") 2>&1 | tee "$OUT/finetune/training.log"

# Evaluate after the trainer has released its simulation and GPU contact buffers.
EVAL_FLAG=()
[ "$EVAL_NO_VIDEO" = "1" ] && EVAL_FLAG+=(--no-video)
for handoff in "$@"; do
    KEY="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["motion_key"])' "$handoff/handoff.json")"
    if [ "$EVAL_EVERY" -gt 0 ]; then
        for ((step=EVAL_EVERY; step<ITERATIONS; step+=EVAL_EVERY)); do
            name=$(printf 'step_%06d' "$step")
            (cd "$SONIC" && python -u gear_sonic/eval_contactfirst_trl.py --pack "$PACK" --motion "$KEY" \
                --checkpoint "$OUT/finetune/train/model_${name}.pt" --output "$OUT/eval/$name/$KEY" "${EVAL_FLAG[@]}")
        done
    fi
    (cd "$SONIC" && python -u gear_sonic/eval_contactfirst_trl.py --pack "$PACK" --motion "$KEY" \
        --checkpoint "$OUT/finetune/train/last.pt" --output "$OUT/eval_final/$KEY" "${EVAL_FLAG[@]}")
done
