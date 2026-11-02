#!/usr/bin/env bash
# GRAIL full environment setup from scratch.
# Creates conda envs (grail + hunyuan, optionally sonic), installs all deps,
# builds CUDA extensions.
#
# Usage:
#   bash scripts/setup/install_env_scratch.sh [OPTIONS]
#   Options: --skip-system-deps, --skip-pytorch, --skip-cuda-extensions,
#            --skip-foundationpose, --skip-gem-smpl, --skip-gem-x,
#            --skip-moge, --skip-mose3, --skip-hunyuan, --only-hunyuan,
#            --install-sonic (opt-in; also creates the 'sonic' env with
#                             Isaac Sim 5.1 + Isaac Lab v2.3.2 — ~30 GB disk,
#                             ~30 min, needs sudo for git-lfs at the end),
#            --help
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; exit 1; }
section() { echo -e "\n${CYAN}══ $1 ══${NC}"; }

# ── Parse arguments ────────────────────────────────────────────────────────
SKIP_SYSTEM_DEPS=false; SKIP_PYTORCH=false; SKIP_CUDA_EXTENSIONS=false
SKIP_FOUNDATIONPOSE=false; SKIP_GEM_SMPL=false; SKIP_GEM_X=false
SKIP_MOGE=false; SKIP_MOSE3=false; SKIP_HUNYUAN=false; ONLY_HUNYUAN=false
INSTALL_SONIC=false

for arg in "$@"; do
    case "$arg" in
        --skip-system-deps)      SKIP_SYSTEM_DEPS=true ;;
        --skip-pytorch)          SKIP_PYTORCH=true ;;
        --skip-cuda-extensions)  SKIP_CUDA_EXTENSIONS=true ;;
        --skip-foundationpose)   SKIP_FOUNDATIONPOSE=true ;;
        --skip-gem-smpl)         SKIP_GEM_SMPL=true ;;
        --skip-gem-x)            SKIP_GEM_X=true ;;
        --skip-gem-soma)         SKIP_GEM_X=true ;; # backwards-compatible alias
        --skip-moge)             SKIP_MOGE=true ;;
        --skip-mose3)            SKIP_MOSE3=true ;;
        --skip-hunyuan)          SKIP_HUNYUAN=true ;;
        --only-hunyuan)          ONLY_HUNYUAN=true ;;
        --install-sonic)         INSTALL_SONIC=true ;;
        --help|-h) grep '^#' "$0" | head -14 | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if [ "$ONLY_HUNYUAN" = true ]; then
    SKIP_SYSTEM_DEPS=true; SKIP_PYTORCH=true
    SKIP_FOUNDATIONPOSE=true; SKIP_GEM_SMPL=true; SKIP_GEM_X=true
    SKIP_MOGE=true; SKIP_MOSE3=true; SKIP_HUNYUAN=false
fi

# ── Detect environment ─────────────────────────────────────────────────────
section "Environment detection"

CONDA_BIN=""
if command -v conda &>/dev/null; then
    CONDA_BIN="$(which conda)"
else
    for p in /root/miniconda3 /opt/conda /root/miniforge3; do
        [ -f "$p/bin/conda" ] && CONDA_BIN="$p/bin/conda" && break
    done
fi
[ -z "$CONDA_BIN" ] && fail "conda not found. Install miniconda first: https://docs.anaconda.com/miniconda/"
eval "$($CONDA_BIN shell.bash hook)"
$CONDA_BIN init bash 2>/dev/null | tail -1
ok "Conda: $($CONDA_BIN --version)"

GRAIL_ENV="grail"
if [ "$ONLY_HUNYUAN" = false ]; then
    if ! conda env list 2>/dev/null | grep -q "^${GRAIL_ENV} "; then
        conda create -y -n "$GRAIL_ENV" python=3.10 2>&1 | tail -3
    fi
    conda activate "$GRAIL_ENV"
    ok "Environment: $GRAIL_ENV ($(python --version))"

    # Compiler + headers/libs for native extensions (PyTorch3D, FoundationPose,
    # mycuda). CUDA 12.1's nvcc rejects gcc>12; libboost 1.91 made boost-system
    # header-only and breaks FoundationPose's `find_package(Boost COMPONENTS
    # system)`; conda-forge eigen provides Eigen/Dense at a discoverable prefix.
    # Installing these into the env keeps everything self-contained and avoids
    # requiring system apt packages.
    conda install -y -c conda-forge \
        'gcc_linux-64=12' 'gxx_linux-64=12' \
        'libboost-devel=1.84' 'libboost=1.84' \
        eigen 2>&1 | tail -3
    ok "Native build deps (gcc-12, boost 1.84, eigen)"
fi

PYTHON=${PYTHON:-python}
PIP="$PYTHON -m pip"

# CUDA detection — discover CUDA_HOME first so nvcc can be added to PATH
if [ -z "${CUDA_HOME:-}" ]; then
    for d in /usr/local/cuda-* /usr/local/cuda; do
        [ -x "$d/bin/nvcc" ] && export CUDA_HOME="$d" && break
    done
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
command -v nvcc &>/dev/null || fail "nvcc not found. Install CUDA toolkit first."
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
[ -z "${CUDA_HOME:-}" ] && fail "Cannot find CUDA. Set CUDA_HOME manually."
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
TORCH_LIB=$($PYTHON -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null || echo "")
[ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ] && export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
# Make conda-env-provided headers (eigen, boost) visible to native builds
# (mycuda hardcodes /usr/include/eigen3 which we don't populate via apt here)
if [ -n "${CONDA_PREFIX:-}" ]; then
    export CPATH="${CONDA_PREFIX}/include:${CONDA_PREFIX}/include/eigen3:${CPATH:-}"
    export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
fi
ok "CUDA $CUDA_VERSION, CUDA_HOME=$CUDA_HOME"

# PyTorch CUDA index URL (pinned to cu121 for torch 2.5.1 compatibility)
TORCH_INDEX="https://download.pytorch.org/whl/cu121"; TORCH_CUDA_TAG="cu121"

# ── 1. System dependencies ─────────────────────────────────────────────────
if [ "$SKIP_SYSTEM_DEPS" = false ]; then
    section "System dependencies"
    APT_CMD="apt-get"; [ "$(id -u)" -ne 0 ] && APT_CMD="sudo apt-get"
    $APT_CMD update -qq
    $APT_CMD install -y -qq \
        build-essential cmake g++ gcc git git-lfs wget curl unzip ffmpeg \
        libeigen3-dev libboost-all-dev \
        libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
        libglfw3-dev libglvnd-dev freeglut3-dev \
        libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxext6 \
        libosmesa6-dev mesa-utils-extra pkg-config \
        2>&1 | tail -3
    git lfs install --skip-repo 2>/dev/null || true
    ok "System packages"
fi

# ── 2. PyTorch ─────────────────────────────────────────────────────────────
if [ "$SKIP_PYTORCH" = false ]; then
    section "PyTorch"
    TORCH_OK=$($PYTHON -c "import torch; assert torch.cuda.is_available(); print(torch.__version__)" 2>/dev/null || echo "")
    if [ -n "$TORCH_OK" ]; then
        ok "PyTorch $TORCH_OK (CUDA)"
    else
        $PIP install --upgrade pip setuptools wheel
        $PIP install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$TORCH_INDEX"
        ok "PyTorch installed"
    fi
fi

# ── 3-9: Main grail env (skipped with --only-hunyuan) ─────────────────────
if [ "$ONLY_HUNYUAN" = false ]; then

section "Python dependencies"
$PIP install --upgrade pip setuptools wheel 2>&1 | tail -1

$PIP install \
    'numpy==2.2.6' scipy opencv-python Pillow imageio imageio-ffmpeg ffmpeg-python \
    trimesh open3d pyyaml omegaconf einops tqdm scikit-image matplotlib \
    joblib ninja pybind11 h5py \
    2>&1 | tail -3
# numpy is pinned to 2.2.6 (not <2) to match the Blender bundled python below.
# 2dhoi step 3 renders pickle data in Blender's numpy 2.x; 4dhoi step 4 then
# unpickles it in this env. Mismatched numpy major versions break the unpickle
# with `ModuleNotFoundError: No module named 'numpy._core.numeric'`.

$PIP install \
    'transformers==4.46.0' accelerate safetensors huggingface_hub sentencepiece \
    smplx roma timm kornia wandb tensorboardX lightning \
    hydra-core hydra-zen hydra_colorlog rich \
    2>&1 | tail -3

$PIP install \
    scenepic pyrender PyOpenGL PyOpenGL_accelerate av 'moviepy==1.0.3' \
    ultralytics lmdb colorama gdown openai rerun-sdk configer natsort ipdb \
    wis3d 'sam2==0.4.0' seaborn transformations ruamel.yaml fast-simplification \
    2>&1 | tail -3

# chumpy (SMPL-X body model loading — patch for numpy compat)
$PIP install chumpy --no-build-isolation 2>&1 | tail -3
CHUMPY_INIT=$($PYTHON -c "import importlib.util; spec = importlib.util.find_spec('chumpy'); print(spec.origin if spec else '')" 2>/dev/null || echo "")
[ -n "$CHUMPY_INIT" ] && sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' "$CHUMPY_INIT"

$PIP install 'warp-lang==1.8.0' 2>&1 | tail -3
ok "Python dependencies"

# Blender
BLENDER_DIR="$PROJECT_ROOT/imports/blender"
if [ ! -x "$BLENDER_DIR/blender" ]; then
    echo "  Downloading Blender 4.3.2..."
    mkdir -p "$BLENDER_DIR"
    curl -L "https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz" -o /tmp/blender.tar.xz
    tar -xf /tmp/blender.tar.xz -C "$BLENDER_DIR" --strip-components=1
    rm -f /tmp/blender.tar.xz
    ok "Blender 4.3.2 downloaded"
else
    ok "Blender already installed"
fi

# Blender Python deps — numpy pinned to match grail env so pickles exchanged
# between the two (e.g. initial_state.pickle) deserialize cleanly. Run on every
# invocation so an existing Blender install also gets the right numpy.
BLENDER_PY="$BLENDER_DIR/4.3/python/bin/python3.11"
if [ -x "$BLENDER_PY" ]; then
    $BLENDER_PY -m ensurepip 2>&1 | tail -1
    $BLENDER_PY -m pip install 'numpy==2.2.6' pyyaml trimesh Pillow imageio scipy tqdm omegaconf opencv-python 2>&1 | tail -3
    $BLENDER_PY -m pip install -e "$PROJECT_ROOT" --no-deps 2>&1 | tail -1
fi

# ── PyTorch3D ──────────────────────────────────────────────────────────────
section "PyTorch3D"
PT3D_OK=$($PYTHON -c "import pytorch3d; print(pytorch3d.__version__)" 2>/dev/null || echo "")
if [ -n "$PT3D_OK" ]; then
    ok "PyTorch3D $PT3D_OK"
else
    $PIP install fvcore iopath 2>&1 | tail -1
    # TORCH_CUDA_ARCH_LIST controls which GPU archs pytorch3d's _C.so bakes
    # binary kernels for. Without it the build auto-detects from the host's
    # GPU (build host without GPU falls back to torch's compile arch list,
    # which on cu128 wheels SKIPS sm_89). Targeting 8.0;8.6;8.9;9.0 covers
    # A100, A40/RTX30, L40/L40s/RTX40 (sm_89!), and H100 — i.e. every modern
    # cluster GPU you'd actually train on.
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" $PIP install \
        "git+https://github.com/facebookresearch/pytorch3d.git" \
        --no-build-isolation 2>&1 | tail -5
    $PYTHON -c "import pytorch3d" 2>/dev/null && ok "PyTorch3D built" || fail "PyTorch3D build failed"
fi

# ── detectron2 ─────────────────────────────────────────────────────────────
section "detectron2"
DET2_OK=$($PYTHON -c "import detectron2; print(detectron2.__version__)" 2>/dev/null || echo "")
if [ -n "$DET2_OK" ]; then
    ok "detectron2 $DET2_OK"
else
    $PIP install cloudpickle pycocotools 2>&1 | tail -1
    $PIP install "git+https://github.com/facebookresearch/detectron2.git" \
        --no-build-isolation 2>&1 | tail -5
    $PYTHON -c "import detectron2" 2>/dev/null && ok "detectron2 built" || warn "detectron2 build failed (non-fatal)"
fi

# ── FoundationPose ─────────────────────────────────────────────────────────
if [ "$SKIP_FOUNDATIONPOSE" = false ]; then
    section "FoundationPose"
    FP_DIR="$PROJECT_ROOT/imports/FoundationPose"
    if [ ! -d "$FP_DIR/mycpp" ]; then
        warn "Submodule not found. Run: git submodule update --init imports/FoundationPose"
    else
        $PIP install pysdf 'warp-lang==1.8.0' \
            "git+https://github.com/NVlabs/nvdiffrast.git" --no-build-isolation 2>&1 | tail -3

        if [ "$SKIP_CUDA_EXTENSIONS" = false ]; then
            # Patch for PyTorch 2.7+ (c++17, scalar_type)
            sed -i "s/std=c++14/std=c++17/g" "$FP_DIR/bundlesdf/mycuda/setup.py"
            sed -i "s/\.type()/\.scalar_type()/g" "$FP_DIR/bundlesdf/mycuda/common.cu"

            # Build mycpp
            if ! ls "$FP_DIR/mycpp/build/mycpp"*".so" &>/dev/null 2>&1; then
                cd "$FP_DIR/mycpp" && rm -rf build && mkdir -p build && cd build
                CMAKE_PREFIX_PATH=$($PYTHON -c "import pybind11; print(pybind11.get_cmake_dir())") \
                    cmake .. -DPYTHON_EXECUTABLE="$(which $PYTHON)" 2>&1 | tail -3
                make -j"$(nproc)" 2>&1 | tail -3
                cd "$PROJECT_ROOT"
            fi
            ok "mycpp"

            # Build mycuda
            if ! $PYTHON -c "import common" 2>/dev/null; then
                cd "$FP_DIR/bundlesdf/mycuda" && rm -rf build *egg* *.so
                $PIP install -e . --no-build-isolation 2>&1 | tail -5
                cd "$PROJECT_ROOT"
            fi
            ok "mycuda"
        fi
    fi
fi

# ── GEM-SMPL ──────────────────────────────────────────────────────────────
if [ "$SKIP_GEM_SMPL" = false ]; then
    section "GEM-SMPL"
    GEM_SMPL_DIR="$PROJECT_ROOT/imports/GEM-SMPL"
    if [ ! -f "$GEM_SMPL_DIR/setup.py" ]; then
        warn "Submodule not found. Run: git submodule update --init imports/GEM-SMPL"
    else
        $PIP install -e "$GEM_SMPL_DIR" --no-deps 2>&1 | tail -3
        $PIP install cython_bbox lapx wis3d \
            "git+https://github.com/google/aistplusplus_api.git" \
            "git+https://github.com/warmshao/WiLoR-mini.git" --no-deps \
            torch-scatter -f "https://data.pyg.org/whl/torch-$($PYTHON -c 'import torch; print(torch.__version__.split("+")[0])')+${TORCH_CUDA_TAG}.html" \
            2>&1 | tail -3

        ok "GEM-SMPL"
    fi
fi

# ── GEM-X ──────────────────────────────────────────────────────────────────
if [ "$SKIP_GEM_X" = false ]; then
    section "GEM-X isolated prediction runtime and native SOMA-X"
    GEM_X_ASSETS="${GRAIL_GEM_X_ASSETS:-${XDG_CACHE_HOME:-$HOME/.cache}/grail/models/gem-x}"
    GEM_X_ENV="${GRAIL_GEM_X_ENV:-${XDG_CACHE_HOME:-$HOME/.cache}/grail/envs/gem-x}"
    if [ ! -f "$PROJECT_ROOT/imports/GEM-X/pyproject.toml" ]; then
        fail "Submodule missing. Run: git submodule update --init imports/GEM-X"
    fi
    GEM_X_GRAIL_PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
    "$PYTHON" scripts/setup/setup_gem_x.py --assets "$GEM_X_ASSETS" --env "$GEM_X_ENV" \
        --grail-python "$GEM_X_GRAIL_PYTHON" --link-submodule
    ok "GEM-X"
fi

# ── MoGe ──────────────────────────────────────────────────────────────────
if [ "$SKIP_MOGE" = false ]; then
    section "MoGe-3 isolated inference worker"
    MOGE_DIR="$PROJECT_ROOT/imports/MoGe"
    if [ ! -f "$MOGE_DIR/pyproject.toml" ]; then
        warn "Submodule not found. Run: git submodule update --init imports/MoGe"
    else
        bash "$PROJECT_ROOT/scripts/setup/install_moge3.sh" --python "$PYTHON"
        ok "MoGe-3 (separate Triton runtime; main grail environment preserved)"
    fi
fi

if [ "$SKIP_MOSE3" = false ]; then
    section "MoSE3 isolated object-tracking worker"
    bash "$PROJECT_ROOT/scripts/setup/install_mose3.sh" --python "$PYTHON"
    ok "MoSE3 (weights downloaded separately; GRAIL PyTorch unchanged)"
fi

section "SOMA Retargeter (Unitree G1 + DEx3)"
bash "$PROJECT_ROOT/scripts/setup/install_soma_retargeter.sh" --python "$PYTHON"
ok "SOMA Retargeter"

fi  # end ONLY_HUNYUAN guard

# ── Hunyuan3D-2.1 (separate conda env) ────────────────────────────────────
if [ "$SKIP_HUNYUAN" = false ]; then
    section "Hunyuan3D-2.1"
    HY3D_DIR="$PROJECT_ROOT/imports/Hunyuan3D-2.1"
    HY3D_ENV="hunyuan"

    if [ ! -f "$HY3D_DIR/requirements.txt" ]; then
        warn "Submodule not found. Run: git submodule update --init imports/Hunyuan3D-2.1"
    else
        if ! conda env list 2>/dev/null | grep -q "^${HY3D_ENV} "; then
            conda create -y -n "$HY3D_ENV" python=3.10 2>&1 | tail -3
        fi
        eval "$(conda shell.bash hook)"
        conda activate "$HY3D_ENV"

        # Compiler for custom_rasterizer CUDA extension — same gcc>12 rejection
        # by CUDA 12.1's nvcc as in the grail env.
        conda install -y -c conda-forge 'gcc_linux-64=12' 'gxx_linux-64=12' 2>&1 | tail -3

        HY3D_PIP="pip"
        command -v uv &>/dev/null && HY3D_PIP="uv pip"

        # Pin setuptools<81: lightning_fabric (transitive dep via pytorch_lightning)
        # calls pkg_resources.declare_namespace(), which setuptools 81 removed.
        $HY3D_PIP install --upgrade pip 'setuptools<81' wheel 2>&1 | tail -1
        $HY3D_PIP install torch torchvision torchaudio --index-url "$TORCH_INDEX" 2>&1 | tail -3

        grep -v "^bpy" "$HY3D_DIR/requirements.txt" > /tmp/_hunyuan_reqs.txt
        $HY3D_PIP install -r /tmp/_hunyuan_reqs.txt 2>&1 | tail -5
        rm -f /tmp/_hunyuan_reqs.txt

        # openai + grail package (needed for image generation/verification in gen_3d_assets)
        $HY3D_PIP install openai 2>&1 | tail -3
        $HY3D_PIP install -e "$PROJECT_ROOT" --no-deps 2>&1 | tail -3

        if [ "$SKIP_CUDA_EXTENSIONS" = false ]; then
            # Detect broken custom_rasterizer: when the kernel .so was built
            # against a different torch ABI, `import custom_rasterizer` still
            # succeeds (the package object exists) but `from .render import *`
            # in __init__.py silently swallows the inner ImportError, leaving
            # the `rasterize` symbol unbound. The old check missed this case
            # and skipped the rebuild. Verify the function attribute is
            # actually exposed; force-reinstall on mismatch.
            if ! python -c "import custom_rasterizer; assert hasattr(custom_rasterizer, 'rasterize')" 2>/dev/null; then
                cd "$HY3D_DIR/hy3dpaint/custom_rasterizer"
                $HY3D_PIP install --no-build-isolation --force-reinstall . 2>&1 | tail -5
                cd "$PROJECT_ROOT"
            fi
            ok "custom_rasterizer"

            DR_DIR="$HY3D_DIR/hy3dpaint/DifferentiableRenderer"
            if ! ls "$DR_DIR"/mesh_inpaint_processor*.so &>/dev/null 2>&1; then
                cd "$DR_DIR" && bash compile_mesh_painter.sh 2>&1 | tail -3
                cd "$PROJECT_ROOT"
            fi
            ok "DifferentiableRenderer"
        fi

        REALESRGAN="$HY3D_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        if [ ! -f "$REALESRGAN" ]; then
            mkdir -p "$(dirname "$REALESRGAN")"
            wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
                -O "$REALESRGAN"
        fi
        ok "Hunyuan3D-2.1"
        conda activate "$GRAIL_ENV" 2>/dev/null || true
    fi
fi

# ── GRAIL package ─────────────────────────────────────────────────────────
if [ "$ONLY_HUNYUAN" = false ]; then
    section "GRAIL package"
    $PIP install -e "$PROJECT_ROOT" 2>&1 | tail -3
    ok "GRAIL installed"
fi

# ── Sonic (Isaac Sim + Isaac Lab + retargeting + task-general tracking) ───
# Opt-in because it's a separate heavy env (~30 GB disk, ~30 min wall-clock)
# and needs sudo for git-lfs. The actual work is delegated to the standalone
# install_env_sonic.sh, which also works on its own against a pre-existing env.
if [ "$INSTALL_SONIC" = true ] && [ "$ONLY_HUNYUAN" = false ]; then
    section "Sonic env (Isaac Sim 5.1 + Isaac Lab v2.3.2)"
    bash "$PROJECT_ROOT/scripts/setup/install_env_sonic.sh"
fi

echo ""
echo -e "${GREEN}Installation complete.${NC}"
if [ "$INSTALL_SONIC" = true ] && [ "$ONLY_HUNYUAN" = false ]; then
    echo ""
    echo "One manual step still needed for sonic:"
    echo "  sudo apt install -y git-lfs && git lfs install"
    echo "  cd imports/SONIC && git lfs pull"
    echo "Then verify with:"
    echo "  conda activate sonic"
    echo "  OMNI_KIT_ACCEPT_EULA=Yes python imports/SONIC/check_environment.py --training"
fi
