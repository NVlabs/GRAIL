#!/usr/bin/env bash
# ============================================================================
# Download all required checkpoints for GRAIL submodules.
#
# Usage:
#   bash scripts/setup/download_checkpoints.sh
#   bash scripts/setup/download_checkpoints.sh --skip-gem-smpl --skip-gem-x
#   bash scripts/setup/download_checkpoints.sh --force   # re-download even if exists
#
# Sources:
#   GEM-SMPL       — HuggingFace nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  (checkpoint/GEM-SMPL/...)
#   GEM-X          — HuggingFace nvidia/GEM-X + nvidia/SOMA-X
#   MoSE3          — HuggingFace JoannaCCCCCC/MoSE3 (pinned release)
#   FoundationPose — HuggingFace nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  (checkpoint/FoundationPose/...)
#   Hunyuan3D      — GitHub release (RealESRGAN); shape/paint auto-download from HF
#   SONIC          — HuggingFace nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  (checkpoint/SONIC/models/...)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()      { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "  ${RED}[FAIL]${NC} $1"; }
section() { echo -e "\n${CYAN}── $1 ──${NC}"; }

# ── Parse arguments ────────────────────────────────────────────────────────
SKIP_GEM_SMPL=false
SKIP_GEM_X=false
SKIP_MOSE3=false
SKIP_FOUNDATIONPOSE=false
SKIP_HUNYUAN3D=false
SKIP_SONIC=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --skip-gem-smpl)       SKIP_GEM_SMPL=true ;;
        --skip-gem-x)          SKIP_GEM_X=true ;;
        --skip-gem-soma)       SKIP_GEM_X=true ;; # backwards-compatible alias
        --skip-mose3)          SKIP_MOSE3=true ;;
        --skip-foundationpose) SKIP_FOUNDATIONPOSE=true ;;
        --skip-hunyuan3d)      SKIP_HUNYUAN3D=true ;;
        --skip-sonic)          SKIP_SONIC=true ;;
        --force)               FORCE=true ;;
        --help|-h)
            echo "Usage: bash scripts/setup/download_checkpoints.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-gem-smpl        Skip GEM-SMPL checkpoint download"
            echo "  --skip-gem-x           Skip GEM-X checkpoint download"
            echo "  --skip-mose3           Skip MoSE3 object-tracking checkpoint download"
            echo "  --skip-foundationpose  Skip FoundationPose weight download"
            echo "  --skip-hunyuan3d       Skip Hunyuan3D-2.1 checkpoint download"
            echo "  --skip-sonic           Skip SONIC base + reference checkpoint download"
            echo "  --force                Re-download even if files already exist"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Helper: download from HuggingFace ──────────────────────────────────────
# ── Helper: download from the GRAIL HF dataset repo ────────────────────────
# Remote layout `checkpoint/<rel>` mirrors local `imports/<rel>`, so callers
# only specify <rel> once. Hardlinks the file from the HF cache into the
# target path when possible to avoid duplicating multi-GB checkpoints.
GRAIL_HF_REPO="nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"

hf_grail_download() {
    local rel_path="$1"   # e.g. "GEM-SMPL/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    local local_path="imports/$rel_path"
    local remote_path="checkpoint/$rel_path"

    if [ -f "$local_path" ] && [ "$FORCE" = false ]; then
        ok "$rel_path (already exists)"
        return
    fi

    mkdir -p "$(dirname "$local_path")"
    echo "  Downloading $rel_path from $GRAIL_HF_REPO..."
    python3 - <<PYEOF
import os, shutil
from huggingface_hub import hf_hub_download
src = hf_hub_download(
    repo_id="$GRAIL_HF_REPO",
    filename="$remote_path",
    repo_type="dataset",
)
dst = "$local_path"
os.makedirs(os.path.dirname(dst), exist_ok=True)
if os.path.exists(dst):
    os.remove(dst)
try:
    os.link(src, dst)
except OSError:
    shutil.copy2(src, dst)
PYEOF
    if [ -f "$local_path" ]; then
        ok "$rel_path"
    else
        fail "$rel_path download failed"
    fi
}

# ════════════════════════════════════════════════════════════════════════════
print_body_model_instructions() {
    local smpl_dir="imports/GEM-SMPL/inputs/checkpoints/body_models/smpl"
    local smplx_dir="imports/GEM-SMPL/inputs/checkpoints/body_models/smplx"

    if [ -f "$smpl_dir/SMPL_NEUTRAL.pkl" ] && \
       [ -f "$smplx_dir/SMPLX_NEUTRAL.npz" ]; then
        ok "Licensed SMPL and SMPL-X body models"
        return
    fi

    mkdir -p "$smpl_dir" "$smplx_dir"
    warn "Please download the licensed body models manually:"
    echo "  SMPL:   https://smpl.is.tue.mpg.de/"
    echo "          SMPL_NEUTRAL.pkl -> $smpl_dir/"
    echo "  SMPL-X: https://smpl-x.is.tue.mpg.de/"
    echo "          SMPLX_NEUTRAL.npz -> $smplx_dir/"
    echo ""
}

# 1. GEM-SMPL checkpoints (HuggingFace nvidia/PhysicalAI-...-GRAIL)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_GEM_SMPL" = false ]; then
    section "GEM-SMPL checkpoints"

    python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
        fail "huggingface_hub is required. Install with: pip install huggingface_hub"
        exit 1
    }

    # These small, redistributable auxiliaries are checked on every run so an
    # interrupted or older installation can be repaired without --force.
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl/J_regressor_extra.npy"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl/J_regressor_h36m.npy"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl/J_regressor_wham.npy"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl/smpl_mean_params.npz"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/body_models/smplx2smpl_sparse.pt"
    hf_grail_download "GEM-SMPL/inputs/checkpoints/vimo/smpl_mean_params.npz"

    GEM_SMPL_MARKER="imports/GEM-SMPL/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    HMR4D_MARKER="imports/GEM-SMPL/outputs/mocap_mixed_v1/genmo/genmo_lg_jukebox_jukebox_new/version_0/checkpoints/last.ckpt"

    if [ -f "$GEM_SMPL_MARKER" ] && [ -f "$HMR4D_MARKER" ] && [ "$FORCE" = false ]; then
        ok "GEM-SMPL checkpoints (already exist)"
    else
        # Inputs bundle: pre-trained externals. Licensed SMPL/SMPL-X model
        # files are intentionally excluded and installed manually below.
        hf_grail_download "GEM-SMPL/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
        hf_grail_download "GEM-SMPL/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
        hf_grail_download "GEM-SMPL/inputs/checkpoints/vimo/vimo_checkpoint.pth.tar"
        hf_grail_download "GEM-SMPL/inputs/checkpoints/yolo/yolov8x.pt"
        # Outputs: GRAIL's own GENMO (HMR4D) checkpoint
        hf_grail_download "GEM-SMPL/outputs/mocap_mixed_v1/genmo/genmo_lg_jukebox_jukebox_new/version_0/checkpoints/last.ckpt"
    fi

    print_body_model_instructions
else
    section "GEM-SMPL checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 2. GEM-X and MoSE3 checkpoints
# ════════════════════════════════════════════════════════════════════════════
# GEM-X's verified asset inventory lives outside the repository and is linked
# into the pinned public submodule only after every hash and byte count matches.
GEM_X_ASSETS="${GRAIL_GEM_X_ASSETS:-${XDG_CACHE_HOME:-$HOME/.cache}/grail/models/gem-x}"
if [ "$SKIP_GEM_X" = false ]; then
    section "GEM-X checkpoints"
    python3 scripts/setup/setup_gem_x.py --assets "$GEM_X_ASSETS" \
        --download-only --link-submodule
else
    section "GEM-X checkpoints (skipped)"
fi

# MoSE3 is the default dynamic-object proposal model. Preserve mismatching local
# files rather than silently replacing a user-selected checkpoint.
if [ "$SKIP_MOSE3" = false ]; then
    section "MoSE3 checkpoints"
    python3 - <<'PY'
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from huggingface_hub import hf_hub_download

revision = '5280ca2ea4c8cf41ac174645fd3aada430af7f40'
destination = Path(os.environ.get('GRAIL_MOSE3_CHECKPOINT', 'imports/MoSE3/ckpt'))
assets = {
    'config.json': '7e7c60d3ddf0781195f6e5a6f500f6321cabe031b04c7334d4bf5729f10c0d4e',
    'model.safetensors': 'fcbd51ff01c2bc2f76ca9db3037e0a4055f6bc5ea202a5e1a9f57a91107d183c',
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

destination.mkdir(parents=True, exist_ok=True)
for name, expected in assets.items():
    target = destination / name
    if target.exists():
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f'Preserving mismatched MoSE3 asset: {target}')
        continue
    source = Path(hf_hub_download('JoannaCCCCCC/MoSE3', name, revision=revision))
    if sha256(source) != expected:
        raise ValueError(f'MoSE3 download checksum mismatch: {name}')
    with tempfile.TemporaryDirectory(prefix='.mose3-', dir=destination) as temporary:
        staged = Path(temporary) / name
        shutil.copyfile(source, staged)
        try:
            os.link(staged, target)
        except FileExistsError:
            if sha256(target) != expected:
                raise ValueError(f'Preserving concurrently selected MoSE3 asset: {target}')
print(f'MoSE3 checkpoint verified: {destination}')
PY
else
    section "MoSE3 checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 3. FoundationPose weights (HuggingFace nvidia/PhysicalAI-...-GRAIL)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_FOUNDATIONPOSE" = false ]; then
    section "FoundationPose weights"

    FP_SCORER="imports/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
    FP_SCORER_CFG="imports/FoundationPose/weights/2024-01-11-20-02-45/config.yml"
    FP_REFINER="imports/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"
    FP_REFINER_CFG="imports/FoundationPose/weights/2023-10-28-18-33-37/config.yml"

    if [ -f "$FP_SCORER" ] && [ -f "$FP_SCORER_CFG" ] && \
       [ -f "$FP_REFINER" ] && [ -f "$FP_REFINER_CFG" ] && [ "$FORCE" = false ]; then
        ok "FoundationPose weights (already exist)"
    else
        python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
            fail "huggingface_hub is required. Install with: pip install huggingface_hub"
            exit 1
        }

        # Two-network pipeline: scorer ranks pose hypotheses, refiner regresses pose deltas.
        # Timestamp dir names are load-path sensitive — FoundationPose's loader hardcodes them.
        hf_grail_download "FoundationPose/weights/2024-01-11-20-02-45/config.yml"
        hf_grail_download "FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
        hf_grail_download "FoundationPose/weights/2023-10-28-18-33-37/config.yml"
        hf_grail_download "FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"
    fi
else
    section "FoundationPose weights (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 4. Hunyuan3D-2.1 (RealESRGAN; shape/paint models auto-download on first use)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_HUNYUAN3D" = false ]; then
    section "Hunyuan3D-2.1 checkpoints"

    REALESRGAN="imports/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"

    if [ -f "$REALESRGAN" ] && [ "$FORCE" = false ]; then
        ok "RealESRGAN (already exists)"
    else
        mkdir -p "$(dirname "$REALESRGAN")"
        echo "  Downloading RealESRGAN_x4plus.pth..."
        wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
            -O "$REALESRGAN"
        if [ -f "$REALESRGAN" ]; then
            ok "RealESRGAN_x4plus.pth"
        else
            fail "RealESRGAN download failed"
        fi
    fi

    echo "  Note: Hunyuan3D shape/paint models will auto-download from HuggingFace on first use."
else
    section "Hunyuan3D-2.1 checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 5. SONIC reference checkpoints (HuggingFace nvidia/PhysicalAI-...-GRAIL)
# ════════════════════════════════════════════════════════════════════════════
SONIC_BASE_DIR="SONIC/models/sonic_manipulation_base"
SONIC_PNP_TABLE_WARMSTART_DIR="SONIC/models/pnp_table_warmstart"
SONIC_PNP_GROUND_WARMSTART_DIR="SONIC/models/pnp_ground_warmstart"
SONIC_STAIRS_WARMSTART_DIR="SONIC/models/terrain_stairs_warmstart"
SONIC_PNP_TABLE_RELEASE_DIR="SONIC/models/pnp_table_release"
SONIC_PNP_GROUND_RELEASE_DIR="SONIC/models/pnp_ground_release"
SONIC_TERRAIN_RELEASE_DIR="SONIC/models/terrain_release"

if [ "$SKIP_SONIC" = false ]; then
    section "SONIC checkpoints"

    SONIC_BASE_MARKER="imports/$SONIC_BASE_DIR/last.pt"

    if [ -f "$SONIC_BASE_MARKER" ] && [ "$FORCE" = false ]; then
        ok "SONIC checkpoints (already exist)"
    else
        python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
            fail "huggingface_hub is required. Install with: pip install huggingface_hub"
            exit 1
        }

        hf_grail_download "$SONIC_PNP_TABLE_WARMSTART_DIR/last.pt"
        hf_grail_download "$SONIC_PNP_TABLE_WARMSTART_DIR/config.yaml"
        hf_grail_download "$SONIC_PNP_GROUND_WARMSTART_DIR/last.pt"
        hf_grail_download "$SONIC_PNP_GROUND_WARMSTART_DIR/config.yaml"
        hf_grail_download "$SONIC_STAIRS_WARMSTART_DIR/last.pt"
        hf_grail_download "$SONIC_STAIRS_WARMSTART_DIR/config.yaml"
        hf_grail_download "$SONIC_PNP_TABLE_RELEASE_DIR/last.pt"
        hf_grail_download "$SONIC_PNP_TABLE_RELEASE_DIR/config.yaml"
        hf_grail_download "$SONIC_PNP_GROUND_RELEASE_DIR/last.pt"
        hf_grail_download "$SONIC_PNP_GROUND_RELEASE_DIR/config.yaml"
        hf_grail_download "$SONIC_TERRAIN_RELEASE_DIR/last.pt"
        hf_grail_download "$SONIC_TERRAIN_RELEASE_DIR/config.yaml"
        hf_grail_download "$SONIC_BASE_DIR/model_config.yaml"
        hf_grail_download "$SONIC_BASE_DIR/last.pt"
    fi
else
    section "SONIC checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 6. Validation
# ════════════════════════════════════════════════════════════════════════════
section "Validating checkpoint files"

ERRORS=0

check_file() {
    local path="$1"
    local description="$2"
    if [ -f "$path" ]; then
        ok "$description"
    else
        fail "$description: $path"
        ERRORS=$((ERRORS + 1))
    fi
}

# GEM-SMPL
if [ "$SKIP_GEM_SMPL" = false ]; then
    check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"     "SMPL neutral body model (manual)"
    check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smpl/J_regressor_extra.npy" "SMPL extra joint regressor"
    check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt" "SMPL neutral joint regressor"
    check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smplx2smpl_sparse.pt"     "SMPL-X to SMPL sparse mapper"
    check_file "imports/GEM-SMPL/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" "SMPL-X body model (manual)"
    check_file "imports/GEM-SMPL/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"        "HMR2 checkpoint"
    check_file "imports/GEM-SMPL/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"      "ViTPose (GEM-SMPL)"
    check_file "imports/GEM-SMPL/inputs/checkpoints/vimo/smpl_mean_params.npz"             "VIMO SMPL mean params"
    check_file "imports/GEM-SMPL/inputs/checkpoints/vimo/vimo_checkpoint.pth.tar"          "VIMO checkpoint"
    check_file "imports/GEM-SMPL/inputs/checkpoints/yolo/yolov8x.pt"                       "YOLOv8x detector"
    check_file "imports/GEM-SMPL/outputs/mocap_mixed_v1/genmo/genmo_lg_jukebox_jukebox_new/version_0/checkpoints/last.ckpt" "HMR4D (GENMO) checkpoint"
fi

# GEM-X assets are checked by hash and byte count, not only existence.
if [ "$SKIP_GEM_X" = false ]; then
    python3 scripts/setup/setup_gem_x.py --assets "$GEM_X_ASSETS" --verify-only
fi

if [ "$SKIP_MOSE3" = false ]; then
    check_file "${GRAIL_MOSE3_CHECKPOINT:-imports/MoSE3/ckpt}/config.json" "MoSE3 config"
    check_file "${GRAIL_MOSE3_CHECKPOINT:-imports/MoSE3/ckpt}/model.safetensors" "MoSE3 model"
fi

# FoundationPose
if [ "$SKIP_FOUNDATIONPOSE" = false ]; then
    check_file "imports/FoundationPose/weights/2024-01-11-20-02-45/config.yml"            "FoundationPose ScorePredictor config"
    check_file "imports/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"         "FoundationPose ScorePredictor"
    check_file "imports/FoundationPose/weights/2023-10-28-18-33-37/config.yml"            "FoundationPose PoseRefinePredictor config"
    check_file "imports/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"         "FoundationPose PoseRefinePredictor"
fi

# Hunyuan3D-2.1
if [ "$SKIP_HUNYUAN3D" = false ]; then
    check_file "imports/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" "RealESRGAN (Hunyuan3D)"
fi

# SONIC
if [ "$SKIP_SONIC" = false ]; then
    check_file "imports/$SONIC_BASE_DIR/last.pt"            "SONIC base behavior-model checkpoint"
    check_file "imports/$SONIC_BASE_DIR/model_config.yaml"  "SONIC base behavior-model config"
    check_file "imports/$SONIC_PNP_TABLE_WARMSTART_DIR/last.pt"       "SONIC pnp_table warmstart checkpoint"
    check_file "imports/$SONIC_PNP_TABLE_WARMSTART_DIR/config.yaml"   "SONIC pnp_table warmstart config"
    check_file "imports/$SONIC_PNP_GROUND_WARMSTART_DIR/last.pt"      "SONIC pnp_ground warmstart checkpoint"
    check_file "imports/$SONIC_PNP_GROUND_WARMSTART_DIR/config.yaml"  "SONIC pnp_ground warmstart config"
    check_file "imports/$SONIC_STAIRS_WARMSTART_DIR/last.pt"         "SONIC terrain stairs warmstart checkpoint"
    check_file "imports/$SONIC_STAIRS_WARMSTART_DIR/config.yaml"     "SONIC terrain stairs warmstart config"
    check_file "imports/$SONIC_PNP_TABLE_RELEASE_DIR/last.pt"       "SONIC pnp_table checkpoint"
    check_file "imports/$SONIC_PNP_TABLE_RELEASE_DIR/config.yaml"   "SONIC pnp_table config"
    check_file "imports/$SONIC_PNP_GROUND_RELEASE_DIR/last.pt"      "SONIC pnp_ground checkpoint"
    check_file "imports/$SONIC_PNP_GROUND_RELEASE_DIR/config.yaml"  "SONIC pnp_ground config"
    check_file "imports/$SONIC_TERRAIN_RELEASE_DIR/last.pt"         "SONIC terrain checkpoint"
    check_file "imports/$SONIC_TERRAIN_RELEASE_DIR/config.yaml"     "SONIC terrain config"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checkpoints are present.${NC}"
else
    echo -e "${YELLOW}$ERRORS checkpoint file(s) missing. See above for details.${NC}"
fi
