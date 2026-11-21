#!/usr/bin/env bash
# ============================================================================
# Download all required checkpoints for GRAIL submodules.
#
# Usage:
#   bash scripts/setup/download_checkpoints.sh
#   bash scripts/setup/download_checkpoints.sh --force   # re-download even if exists
#
# Sources:
#   GEM-X       — HuggingFace nvidia/GEM-X + nvidia/soma-x
#   MoSE3         — HuggingFace JoannaCCCCCC/MoSE3 (pinned release)
#   FoundationPose — HuggingFace nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  (checkpoint/FoundationPose/...)
#   Hunyuan3D      — GitHub release (RealESRGAN); shape/paint auto-download from HF
#   SONIC          — HuggingFace nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL  (checkpoint/SONIC/models/...)
#   SONIC manip    — same repo, checkpoint/SONIC/models/manip_warmstart (manipulation finetune warm start)
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
SKIP_GEM_X=false
SKIP_MOSE3=false
SKIP_FOUNDATIONPOSE=false
SKIP_HUNYUAN3D=false
SKIP_SONIC=false
SKIP_SONIC_MANIP=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --skip-gem-x)       SKIP_GEM_X=true ;;
        --skip-mose3)          SKIP_MOSE3=true ;;
        --skip-foundationpose) SKIP_FOUNDATIONPOSE=true ;;
        --skip-hunyuan3d)      SKIP_HUNYUAN3D=true ;;
        --skip-sonic)          SKIP_SONIC=true ;;
        --skip-sonic-manip)    SKIP_SONIC_MANIP=true ;;
        --force)               FORCE=true ;;
        --help|-h)
            echo "Usage: bash scripts/setup/download_checkpoints.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-gem-x           Skip GEM-X checkpoint download"
            echo "  --skip-mose3           Skip MoSE3 object-tracking checkpoint download"
            echo "  --skip-foundationpose  Skip FoundationPose weight download"
            echo "  --skip-hunyuan3d       Skip Hunyuan3D-2.1 checkpoint download"
            echo "  --skip-sonic           Skip SONIC base + reference checkpoint download"
            echo "  --skip-sonic-manip     Skip the manipulation tracking finetune warm start"
            echo "  --force                Re-download even if files already exist"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Helper: download from the GRAIL HF dataset repo ────────────────────────
# The optional second argument names the remote file when its directory differs
# from the local path used by the training configs. Hardlinks from the HF cache into the
# target path when possible to avoid duplicating multi-GB checkpoints.
GRAIL_HF_REPO="nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"

hf_grail_download() {
    local rel_path="$1"   # e.g. "FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
    local local_path="imports/$rel_path"
    local remote_path="checkpoint/${2:-$rel_path}"

    if [ -f "$local_path" ] && [ "$FORCE" = false ]; then
        ok "$rel_path (already exists)"
        return
    fi

    mkdir -p "$(dirname "$local_path")"
    echo "  Downloading $rel_path from $GRAIL_HF_REPO..."
    python3 - <<PYEOF
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

src = Path(hf_hub_download(
    repo_id="$GRAIL_HF_REPO",
    filename="$remote_path",
    repo_type="dataset",
)).resolve(strict=True)
dst = Path("$local_path")
with tempfile.TemporaryDirectory(dir=dst.parent) as staging:
    ready = Path(staging) / dst.name
    try:
        os.link(src, ready)
    except OSError:
        shutil.copyfile(src, ready)
    os.replace(ready, dst)
PYEOF
    if [ -f "$local_path" ]; then
        ok "$rel_path"
    else
        fail "$rel_path download failed"
        return 1
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# 2. GEM-X checkpoints (HuggingFace nvidia/GEM-X)
# ════════════════════════════════════════════════════════════════════════════
# The public release uses a verified asset inventory and external storage.
GEM_X_ASSETS="${GRAIL_GEM_X_ASSETS:-${XDG_CACHE_HOME:-$HOME/.cache}/grail/models/gem-x}"
if [ "$SKIP_GEM_X" = false ]; then
    section "GEM-X checkpoints"
    python3 scripts/setup/setup_gem_x.py --assets "$GEM_X_ASSETS" \
        --download-only --link-submodule
else
    section "GEM-X checkpoints (skipped)"
fi

# MoSE3 is the default dynamic-object proposal model. Keep checkpoints separate
# from the runtime image and preserve mismatching local files for inspection.
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
    # Only publish a complete file, without replacing a concurrent selection.
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
fi

# ════════════════════════════════════════════════════════════════════════════
# 3. FoundationPose weights (HuggingFace nvidia/PhysicalAI-...-GRAIL)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_FOUNDATIONPOSE" = false ]; then
    section "FoundationPose weights"

    # Both predictors load config.yml beside model_best.pth.
    for predictor in 2024-01-11-20-02-45 2023-10-28-18-33-37; do
        hf_grail_download "FoundationPose/weights/$predictor/model_best.pth"
        hf_grail_download "FoundationPose/weights/$predictor/config.yml"
    done
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
SONIC_PNP_TABLE_DIR="SONIC/models/pnp_table"
SONIC_PNP_GROUND_DIR="SONIC/models/pnp_ground"
SONIC_STAIRS_DIR="SONIC/models/terrain_stairs"

if [ "$SKIP_SONIC" = false ]; then
    section "SONIC checkpoints"

    hf_grail_download "$SONIC_BASE_DIR/last.pt"
    hf_grail_download "$SONIC_BASE_DIR/model_config.yaml"
    for bundle in "$SONIC_PNP_TABLE_DIR" "$SONIC_PNP_GROUND_DIR" "$SONIC_STAIRS_DIR"; do
        for file in last.pt config.yaml; do
            hf_grail_download "$bundle/$file" "${bundle}_warmstart/$file"
        done
    done
else
    section "SONIC checkpoints (skipped)"
fi

# ════════════════════════════════════════════════════════════════════════════
# 5b. Manipulation tracking finetune warm start (actor-only; uses sonic_manipulation_base)
# ════════════════════════════════════════════════════════════════════════════
SONIC_MANIP_WARMSTART_DIR="SONIC/models/manip_warmstart"

if [ "$SKIP_SONIC_MANIP" = false ]; then
    section "SONIC manipulation finetune warm start"
    python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
        fail "huggingface_hub is required. Install with: pip install huggingface_hub"
        exit 1
    }
    for rel in "$SONIC_BASE_DIR/model_config.yaml" "$SONIC_BASE_DIR/last.pt" \
               "$SONIC_MANIP_WARMSTART_DIR/last.pt" "$SONIC_MANIP_WARMSTART_DIR/meta.json"; do
        hf_grail_download "$rel"
    done
else
    section "SONIC manipulation finetune warm start (skipped)"
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

# GEM-X assets are checked by hash and byte count, not only existence.
if [ "$SKIP_GEM_X" = false ]; then
    python3 scripts/setup/setup_gem_x.py --assets "$GEM_X_ASSETS" --verify-only
fi

# FoundationPose
if [ "$SKIP_FOUNDATIONPOSE" = false ]; then
    for predictor in 2024-01-11-20-02-45 2023-10-28-18-33-37; do
        check_file "imports/FoundationPose/weights/$predictor/model_best.pth" "FoundationPose $predictor weights"
        check_file "imports/FoundationPose/weights/$predictor/config.yml" "FoundationPose $predictor config"
    done
fi

# Hunyuan3D-2.1
if [ "$SKIP_HUNYUAN3D" = false ]; then
    check_file "imports/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"               "RealESRGAN (Hunyuan3D)"
fi

# SONIC
if [ "$SKIP_SONIC" = false ]; then
    check_file "imports/$SONIC_BASE_DIR/last.pt"            "SONIC base behavior-model checkpoint"
    check_file "imports/$SONIC_BASE_DIR/model_config.yaml"  "SONIC base behavior-model config"
    check_file "imports/$SONIC_PNP_TABLE_DIR/last.pt"       "SONIC pnp_table checkpoint"
    check_file "imports/$SONIC_PNP_TABLE_DIR/config.yaml"   "SONIC pnp_table config"
    check_file "imports/$SONIC_PNP_GROUND_DIR/last.pt"      "SONIC pnp_ground checkpoint"
    check_file "imports/$SONIC_PNP_GROUND_DIR/config.yaml"  "SONIC pnp_ground config"
    check_file "imports/$SONIC_STAIRS_DIR/last.pt"         "SONIC terrain checkpoint"
    check_file "imports/$SONIC_STAIRS_DIR/config.yaml"     "SONIC terrain config"
fi

if [ "$SKIP_SONIC_MANIP" = false ]; then
    check_file "imports/$SONIC_MANIP_WARMSTART_DIR/last.pt"   "SONIC manipulation finetune warm start"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All requested checkpoints are present.${NC}"
else
    echo -e "${YELLOW}$ERRORS checkpoint file(s) missing. See above for details.${NC}"
    exit 1
fi
