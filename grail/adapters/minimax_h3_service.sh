#!/usr/bin/env bash
# Launch GRAIL's portable resident MiniMax-H3 FL2VA service.
# Start the service inside the selected GPU environment.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash grail/adapters/minimax_h3_service.sh [mode] [port]

Modes: dense, gb-parity, fast, cache-fast, quality, single,
       single-gb-parity, single-gb-parity-vae-resident, single-cache-fast,
       single-cache-fast-vae-resident, single-cache-fast-fp8-resident,
       dual, dual-fast

Required environment:
  MINIMAX_H3_CHECKPOINT   Path to the MiniMax-H3 FL2VA checkpoint directory.

Optional environment:
  MINIMAX_H3_SERVICE_ROOT, MINIMAX_H3_SERVICE_FILE, MINIMAX_H3_PYTHON,
  MINIMAX_H3_SOL_ENGINE_ROOT, MINIMAX_H3_HARDWARE,
  MINIMAX_H3_BIND_HOST, MINIMAX_H3_ADVERTISE_HOST,
  MINIMAX_H3_CHECKPOINT_ID (optional human-managed provenance label).
EOF
}

DEFAULT_MODE=single-gb-parity-vae-resident
MODE=${1:-$DEFAULT_MODE}
PORT=${2:-30020}
HARDWARE=${MINIMAX_H3_HARDWARE:-A100}
HARDWARE=${HARDWARE^^}

NUM_GPUS=4
PROFILE=dense
QUANTIZATION=
QUANTIZATION_IGNORED_LAYERS=
FORCE_FP8_MARLIN=0
DIT_RESIDENT=0
VAE_RESIDENT=0
case "$MODE" in
  dense) PROFILE=dense ;;
  gb-parity) PROFILE=gb_parity ;;
  quality) PROFILE=quality ;;
  cache-fast) PROFILE=cache_only ;;
  fast) PROFILE=fullopt_exact ;;
  single|single-gb-parity|single-gb-parity-vae-resident|single-cache-fast|single-cache-fast-vae-resident|single-cache-fast-fp8-resident)
    NUM_GPUS=1
    case "$MODE" in
      single) PROFILE=dense ;;
      single-gb-parity) PROFILE=gb_parity ;;
      single-gb-parity-vae-resident)
        PROFILE=gb_parity
        VAE_RESIDENT=1
        ;;
      single-cache-fast) PROFILE=cache_only ;;
      single-cache-fast-vae-resident)
        PROFILE=cache_only
        VAE_RESIDENT=1
        ;;
      single-cache-fast-fp8-resident)
        PROFILE=cache_only
        QUANTIZATION=fp8
        FORCE_FP8_MARLIN=1
        DIT_RESIDENT=1
        QUANTIZATION_IGNORED_LAYERS=condition_proj,token_refiner,final_layer
        for ((index = 0; index < 50; index++)); do
          QUANTIZATION_IGNORED_LAYERS+=,blocks.$index.adaln_proj
        done
        ;;
    esac
    ;;
  dual|dual-fast)
    NUM_GPUS=2
    if [[ "$MODE" == dual ]]; then
      PROFILE=dense
    else
      PROFILE=fullopt_exact
    fi
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GRAIL_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
SOL_ENGINE_ROOT=${MINIMAX_H3_SOL_ENGINE_ROOT:-$GRAIL_ROOT/imports/Sana}
RUNTIME_DIR=$SOL_ENGINE_ROOT/models/minimax_h3/$HARDWARE
if [[ ! -d "$RUNTIME_DIR" ]]; then
  echo "MiniMax-H3 hardware runtime is unavailable: $RUNTIME_DIR" >&2
  exit 1
fi
if [[ "$HARDWARE" != A100 && "$HARDWARE" != H100 ]]; then
  echo "The resident MiniMax-H3 service supports A100 or H100, got $HARDWARE" >&2
  exit 1
fi
if [[ "$HARDWARE" != A100 && "$NUM_GPUS" != 4 ]]; then
  echo "MiniMax-H3 $HARDWARE service currently requires a four-GPU mode" >&2
  exit 1
fi
if [[ "$HARDWARE" == H100 && "$PROFILE" != dense && "$PROFILE" != quality && "$PROFILE" != fullopt_exact ]]; then
  echo "MiniMax-H3 H100 service modes are dense, quality, or fast" >&2
  exit 1
fi

CHECKPOINT=${MINIMAX_H3_CHECKPOINT:-}
H3_PYTHON=${MINIMAX_H3_PYTHON:-python3}
if [[ -z "$CHECKPOINT" ]]; then
  echo "MINIMAX_H3_CHECKPOINT must point to the MiniMax-H3 FL2VA checkpoint" >&2
  exit 1
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "MiniMax-H3 FL2VA checkpoint directory does not exist: $CHECKPOINT" >&2
  exit 1
fi
CHECKPOINT=$(cd "$CHECKPOINT" && pwd -P)
if [[ ! -f "$CHECKPOINT/model_index.json" ]]; then
  echo "MiniMax-H3 FL2VA checkpoint is incomplete: $CHECKPOINT" >&2
  exit 1
fi

DEFAULT_RUN_ROOT=${TMPDIR:-/tmp}/minimax-h3-service-$(id -u)
if [[ -n "${MINIMAX_H3_SERVICE_ROOT:-}" ]]; then
  SERVICE_ROOT=$MINIMAX_H3_SERVICE_ROOT
else
  mkdir -p "$DEFAULT_RUN_ROOT"
  SERVICE_ROOT=$(mktemp -d "$DEFAULT_RUN_ROOT/${MODE}.XXXXXX")
fi
SERVICE_FILE=${MINIMAX_H3_SERVICE_FILE:-$SERVICE_ROOT/service.json}
MODEL_VIEW=${MINIMAX_H3_MODEL_VIEW:-$SERVICE_ROOT/model}

mkdir -p "$SERVICE_ROOT/outputs" "$SERVICE_ROOT/cache" "$MODEL_VIEW"

# SGLang verifies components at the model root before loading the FL2VA
# partition. Build a zero-copy model view that satisfies both layouts.
cp "$CHECKPOINT/model_index.json" "$MODEL_VIEW/model_index.json.tmp"
mv "$MODEL_VIEW/model_index.json.tmp" "$MODEL_VIEW/model_index.json"
for entry in FL2VA text_encoder tokenizer processor video_vae audio_vae transformer; do
  if [[ "$entry" == FL2VA ]]; then
    target=$CHECKPOINT
  else
    target=$CHECKPOINT/$entry
  fi
  if [[ ! -e "$target" ]]; then
    echo "MiniMax-H3 FL2VA component is missing: $target" >&2
    exit 1
  fi
  if [[ -e "$MODEL_VIEW/$entry" && ! -L "$MODEL_VIEW/$entry" ]]; then
    echo "Refusing to replace non-symlink model-view entry: $MODEL_VIEW/$entry" >&2
    exit 1
  fi
  ln -sfn "$target" "$MODEL_VIEW/$entry"
done

rm -f "$SERVICE_FILE"
export H3_NUM_GPUS=$NUM_GPUS
export H3_QUANTIZATION=$QUANTIZATION
export H3_QUANTIZATION_IGNORED_LAYERS=$QUANTIZATION_IGNORED_LAYERS
export H3_DIT_RESIDENT=$DIT_RESIDENT
export H3_VAE_RESIDENT=$VAE_RESIDENT
export H3_RNG_CONTRACT=${H3_RNG_CONTRACT:-diffusers_shared}
export H3_SOL_PROFILE=$PROFILE
export SGLANG_FORCE_FP8_MARLIN=$FORCE_FP8_MARLIN
export H3_MODEL_PATH=$MODEL_VIEW
export H3_MODEL_SUBFOLDER=FL2VA
export H3_MODEL_REVISION=${H3_MODEL_REVISION:-bfc8ed0353f5a9733be73e6b2c98ec0948195b86}
export H3_MASTER_PORT=${H3_MASTER_PORT:-$((30000 + (($$ + RANDOM) % 20000)))}
export H3_EXPECTED_TORCH=${H3_EXPECTED_TORCH:-2.11.0+cu130}
export H3_EXPECTED_TRITON=${H3_EXPECTED_TRITON:-3.6.0}
export H3_EASYCACHE_NUM_FORWARDS=${H3_EASYCACHE_NUM_FORWARDS:-49}
export H3_CACHE_ROOT=${H3_CACHE_ROOT:-$SERVICE_ROOT/cache}
export HF_HOME=${HF_HOME:-$H3_CACHE_ROOT/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$H3_CACHE_ROOT/triton}
export TORCH_HOME=${TORCH_HOME:-$H3_CACHE_ROOT/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$H3_CACHE_ROOT/xdg}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export H3_SGLANG_PYTHON_ROOT=${H3_SGLANG_PYTHON_ROOT:-/sgl-workspace/sglang/python}
export PYTHONPATH=$GRAIL_ROOT:$SOL_ENGINE_ROOT:$SOL_ENGINE_ROOT/techniques/sparse_backends:$H3_SGLANG_PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader || true
fi
echo "MiniMax-H3 service: hardware=$HARDWARE mode=$MODE profile=$PROFILE gpus=$NUM_GPUS"
echo "MiniMax-H3 checkpoint: $CHECKPOINT"
echo "MiniMax-H3 service descriptor: $SERVICE_FILE"

exec "$H3_PYTHON" -m grail.adapters.minimax_h3_service \
  --model "$MODEL_VIEW" \
  --output "$SERVICE_ROOT/outputs" \
  --shared-root "$SERVICE_ROOT" \
  --service-file "$SERVICE_FILE" \
  --host "${MINIMAX_H3_BIND_HOST:-127.0.0.1}" \
  --advertise-host "${MINIMAX_H3_ADVERTISE_HOST:-127.0.0.1}" \
  --port "$PORT"
