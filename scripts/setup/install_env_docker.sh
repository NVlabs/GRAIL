#!/usr/bin/env bash
# GRAIL Docker setup — installs Blender into the bind-mounted repo.
# The Docker image already has conda envs, Python deps, and CUDA extensions.
# Usage:
#   bash scripts/setup/install_env_docker.sh
#   SKIP_BLENDER=1 bash scripts/setup/install_env_docker.sh  # test native extensions only
#   SKIP_RECON_RUNTIMES=1 bash scripts/setup/install_env_docker.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

# Init conda
CONDA_BIN=""
for p in /root/miniconda3 /opt/conda /root/miniforge3; do
    [ -f "$p/bin/conda" ] && CONDA_BIN="$p/bin/conda" && break
done
[ -z "$CONDA_BIN" ] && echo -e "${RED}conda not found. Are you in the GRAIL Docker container?${NC}" && exit 1
eval "$($CONDA_BIN shell.bash hook)"
$CONDA_BIN init bash 2>/dev/null | tail -1

# Verify grail env
conda env list 2>/dev/null | grep -q "^grail " || { echo -e "${RED}'grail' env not found. Use install_env_scratch.sh instead.${NC}"; exit 1; }

run_grail() {
    conda run -n grail --no-capture-output "$@"
}

# Build CUDA extensions for the current GPU when the image ships stale
# prebuilt binaries. This matters for nvdiffrast: a wheel compiled only for
# sm_120 fails with CUDA error 209 on Ada/Hopper GPUs.
CUDA_ARCH="$(run_grail python - <<'PY' 2>/dev/null || true
import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
PY
)"
if [ -n "$CUDA_ARCH" ] && [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
fi
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    if ! run_grail python - <<'PY' >/dev/null 2>&1
import nvdiffrast.torch as dr

dr.RasterizeCudaContext()
PY
    then
        echo "Rebuilding nvdiffrast for this GPU..."
        run_grail python -m pip uninstall -y nvdiffrast >/dev/null 2>&1 || true
        run_grail python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git" 2>&1 | tail -5
        run_grail python - <<'PY'
import nvdiffrast.torch as dr

dr.RasterizeCudaContext()
print("nvdiffrast CUDA rasterizer OK")
PY
    else
        echo "nvdiffrast CUDA rasterizer OK"
    fi
else
    echo "No CUDA GPU detected; skipping nvdiffrast runtime check."
fi

# Pin setuptools<81 in hunyuan env: lightning_fabric (transitive dep via
# pytorch_lightning) calls pkg_resources.declare_namespace(), which setuptools
# 81 removed. Older/custom images may carry a newer incompatible setuptools;
# release 1.2.0 already includes a compatible version.
if conda env list 2>/dev/null | grep -q "^hunyuan "; then
    conda run -n hunyuan pip install --quiet 'setuptools<81' 2>&1 | tail -1

    # Build mesh_inpaint_processor.so if missing: the prebuilt image is missing
    # this extension, which causes meshVerticeInpaint NameError in texture inpaint.
    DR_DIR="$PROJECT_ROOT/imports/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer"
    if [ -f "$DR_DIR/compile_mesh_painter.sh" ] && ! ls "$DR_DIR"/mesh_inpaint_processor*.so &>/dev/null; then
        echo "Building mesh_inpaint_processor.so..."
        (cd "$DR_DIR" && conda run -n hunyuan --no-capture-output bash compile_mesh_painter.sh) 2>&1 | tail -3
    fi
fi

# Build FoundationPose mycpp.so if missing or built against the wrong Python
# ABI. Some image builds carried mycpp.cpython-313*.so while grail runs Python
# 3.10, causing FoundationPose to silently set mycpp=None.
FP_MYCPP_DIR="$PROJECT_ROOT/imports/FoundationPose/mycpp"
if [ -f "$FP_MYCPP_DIR/CMakeLists.txt" ]; then
    export FP_MYCPP_DIR
    if ! run_grail python - <<'PY' >/dev/null 2>&1
import os
import sys

sys.path.insert(0, os.path.dirname(os.environ["FP_MYCPP_DIR"]))
import mycpp.build.mycpp as mycpp

assert hasattr(mycpp, "cluster_poses")
PY
    then
        echo "Building FoundationPose mycpp.so..."
        run_grail bash -lc '
            set -euo pipefail
            cd "$FP_MYCPP_DIR"
            rm -rf build
            mkdir -p build
            cd build
            PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
            cmake .. -DPYTHON_EXECUTABLE=$(which python) -Dpybind11_DIR="$PYBIND11_DIR"
            make -j"$(nproc)"
        ' 2>&1 | tail -8
        run_grail python - <<'PY'
import os
import sys

sys.path.insert(0, os.path.dirname(os.environ["FP_MYCPP_DIR"]))
import mycpp.build.mycpp as mycpp

assert hasattr(mycpp, "cluster_poses")
print("FoundationPose mycpp OK")
PY
    else
        echo "FoundationPose mycpp OK"
    fi
fi

# Download Blender
BLENDER_DIR="$PROJECT_ROOT/imports/blender"
if [ "${SKIP_BLENDER:-0}" = "1" ]; then
    echo "Skipping Blender install because SKIP_BLENDER=1."
elif [ ! -x "$BLENDER_DIR/blender" ]; then
    echo "Downloading Blender 4.3.2..."
    mkdir -p "$BLENDER_DIR"
    curl -L "https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz" -o /tmp/blender.tar.xz
    tar -xf /tmp/blender.tar.xz -C "$BLENDER_DIR" --strip-components=1
    rm -f /tmp/blender.tar.xz
    echo -e "${GREEN}Blender installed.${NC}"
else
    echo "Blender already installed."
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

# Reconstruction uses isolated, mutually incompatible predictor environments.
# Model weights remain external and are handled by download_checkpoints.sh.
if [ "${SKIP_RECON_RUNTIMES:-0}" != "1" ]; then
    GRAIL_PYTHON="$(run_grail python -c 'import sys; print(sys.executable)')"
    GEM_X_ENV="${GRAIL_GEM_X_ENV:-${XDG_CACHE_HOME:-$HOME/.cache}/grail/envs/gem-x}"
    run_grail python scripts/setup/setup_gem_x.py \
        --env "$GEM_X_ENV" --env-only --grail-python "$GRAIL_PYTHON"
    GEM_X_LINK="$PROJECT_ROOT/imports/GEM-X/.venv"
    if [ -L "$GEM_X_LINK" ]; then
        [ "$(readlink -f "$GEM_X_LINK")" = "$(readlink -f "$GEM_X_ENV")" ] || {
            echo -e "${RED}GEM-X .venv points to a different environment.${NC}"; exit 1;
        }
    elif [ -e "$GEM_X_LINK" ]; then
        echo -e "${RED}Refusing to replace existing GEM-X .venv.${NC}"; exit 1
    else
        ln -s "$GEM_X_ENV" "$GEM_X_LINK"
    fi
    bash scripts/setup/install_moge3.sh --python "$GRAIL_PYTHON"
    bash scripts/setup/install_mose3.sh --python "$GRAIL_PYTHON"
    bash scripts/setup/install_soma_retargeter.sh --python "$GRAIL_PYTHON"
    run_grail python - <<'PY'
import importlib.metadata as metadata

import soma

assert soma.__version__ == "0.3.3"
assert metadata.version("py-soma-x") == "0.3.3"
print("GRAIL native SOMA runtime OK: py-soma-x=0.3.3")
PY
    run_grail python -m grail.retargeting.soma --help >/dev/null
fi

echo -e "${GREEN}Docker setup complete.${NC}"
