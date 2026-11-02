# GRAIL Installation Guide

GRAIL has been developed and tested on Ubuntu 22.04+ with NVIDIA GPUs (A6000, RTX 4090, RTX 5090, RTX 6000 Ada).

The repo ships three conda envs:

- `grail` (Python 3.10) — 2D generation, 4D reconstruction, optimization
- `hunyuan` (Python 3.10) — Hunyuan3D-2.1 asset generation
- `sonic` (Python 3.11, optional) — task-general tracking and the legacy
  SMPL-X/GMR path (Isaac Lab + Isaac Sim)

Fresh 4D reconstruction also uses three isolated workers: GEM-X for native
SOMA pose prediction, MoGe-3 for depth, and MoSE3 for object-track proposals.
Their dependency sets are intentionally kept out of the `grail` environment.
The public SOMA Retargeter uses its own Python 3.12 environment and currently
exposes only the `unitree_g1` target through GRAIL.

## Docker (Recommended)

Choose between two Linux AMD64, environment-only images:

```{list-table}
:widths: 45 55
:header-rows: 1

* - Image
  - Use it for
* - `docker.io/nvgrail/grail:1.2.0` (`:latest`)
  - Standard GRAIL with native SOMA 0.3 and Kling; no MiniMax-H3/SGLang runtime.
* - `docker.io/nvgrail/grail:1.2.0-minimax-h3` (`:minimax-h3`)
  - Standard GRAIL plus an isolated MiniMax-H3/SGLang runtime.
```

Both images ship pre-installed system dependencies, CUDA, and the `grail`,
`hunyuan`, and `sonic` Conda environments. Neither image contains the
GRAIL/Sana source, GRAIL or MiniMax-H3 checkpoints, API keys, or user data.
Bind-mount your repo and run the setup script to validate GPU/Python-specific
native extensions, download Blender, and provision the isolated GEM-X,
MoGe-3, MoSE3, and Python 3.12 SOMA Retargeter environments. The GRAIL-side
native SOMA runtime is pinned to `py-soma-x==0.3.3`. Keep `.env` and registry
credentials on the host; never add them to an image build context.

For an exactly reproducible pull, use the immutable release digest:

```text
Standard:
docker.io/nvgrail/grail@sha256:14eb489f0d6eb80b8ae198d44bfd4800695af27202a7a38b504c57d22f65ad8e

MiniMax-H3:
docker.io/nvgrail/grail@sha256:fdbee9f32bd525d4d68c7346dd677d09c24c95575084061e67e5a290a60186d7
```

```bash
git clone https://github.com/NVlabs/GRAIL.git
cd GRAIL
git submodule update --init --recursive

# The public Docker Hub images can be pulled without registry credentials.
export GRAIL_IMAGE=docker.io/nvgrail/grail:1.2.0
docker pull "$GRAIL_IMAGE"
docker run --gpus all -it --shm-size=16g \
    -v "$PWD":/workspace/grail \
    "$GRAIL_IMAGE"

# Inside the container
cd /workspace/grail
bash scripts/setup/install_env_docker.sh   # validates extensions and installs isolated runtimes
bash scripts/setup/download_checkpoints.sh # model checkpoints
bash scripts/setup/download_comasset.sh    # datasets (optional)
eval "$(conda shell.bash hook)"
conda activate grail
[ -f .env ] && source .env
```

The setup script rebuilds GPU/Python-specific native extensions when needed
(`nvdiffrast`, FoundationPose `mycpp`) and installs Blender into the mounted
checkout.

The top-level `Dockerfile` is a minimal source-install bootstrap, not the build
recipe for the published runtime. Use the prebuilt images above to retain all
three pre-installed Conda environments.

## Install from Scratch

### Quick Start

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/NVlabs/GRAIL.git
cd GRAIL

# If already cloned without --recursive:
git submodule update --init --recursive

# 2. Install everything
bash scripts/setup/install_env_scratch.sh                 # grail + hunyuan
bash scripts/setup/install_env_scratch.sh --install-sonic # + sonic (Isaac Sim/Lab, ~30 GB / ~30 min)

# 3. Activate the environment
#    (install_env_scratch.sh runs `conda init bash`, so start a new shell or source bashrc)
source ~/.bashrc
conda activate grail

# 4. Download checkpoints
bash scripts/setup/download_checkpoints.sh

# 5. Download ComAsset dataset
bash scripts/setup/download_comasset.sh
```

## Environment Setup (Manual)

### Prerequisites

- **OS:** Ubuntu 20.04+
- **GPU:** NVIDIA GPU with CUDA support (tested: A6000, RTX 4090, RTX 5090, RTX 6000 Ada)
- **CUDA:** toolkit installed at `/usr/local/cuda-<version>/` (the scripts default to cu121; use the matching NVIDIA runfile if your system only has apt-installed `nvidia-cuda-toolkit`, since that layout doesn't satisfy `$CUDA_HOME/{bin,include,lib64}`).
- **Conda:** miniconda or miniforge

### System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake g++ gcc git git-lfs wget curl unzip ffmpeg \
    libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    libglfw3-dev libglvnd-dev freeglut3-dev \
    libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxext6 \
    libosmesa6-dev mesa-utils-extra pkg-config
```

Boost, Eigen, and gcc-12 ship via conda-forge into the `grail` / `hunyuan` envs; no system packages needed.

### Python Environment

```bash
conda create -y -n grail python=3.10
conda activate grail
pip install --upgrade pip setuptools wheel

# PyTorch 2.5.1 + CUDA 12.1 (script default; cu128 + torch 2.7.1 also works)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Build-env C++ toolchain compatible with CUDA 12.1 nvcc (gcc>12 is rejected).
# libboost 1.84 because 1.91 made boost::system header-only, breaking
# FoundationPose's find_package(Boost COMPONENTS system).
conda install -y -c conda-forge \
    'gcc_linux-64=12' 'gxx_linux-64=12' \
    'libboost-devel=1.84' 'libboost=1.84' eigen

# PyTorch3D (build from source — ~5-10 min)
pip install fvcore iopath
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# detectron2 (build from source)
pip install cloudpickle pycocotools
pip install "git+https://github.com/facebookresearch/detectron2.git" --no-build-isolation

# Core dependencies
pip install numpy scipy opencv-python Pillow imageio ffmpeg-python trimesh fast_simplification open3d \
    pyyaml omegaconf einops tqdm scikit-image matplotlib joblib ninja pybind11
pip install 'py-soma-x==0.3.3' 'uv>=0.9,<1'

# ML dependencies
pip install 'transformers==4.46.0' accelerate safetensors huggingface_hub sentencepiece \
    smplx roma timm kornia wandb tensorboardX lightning hydra-core hydra-zen rich

# Visualization & utilities
pip install scenepic pyrender PyOpenGL PyOpenGL_accelerate av 'moviepy==1.0.3' \
    ultralytics colorama gdown openai rerun-sdk configer natsort \
    wis3d 'sam2==0.4.0' seaborn transformations ruamel.yaml

# Reconstruction-specific deps
pip install chumpy --no-build-isolation  # then patch for numpy compat:
# sed -i 's/from numpy import bool, int, float.*/from numpy import nan, inf/' \
#   $(python -c "import chumpy; print(chumpy.__file__)")
pip install "git+https://github.com/warmshao/WiLoR-mini.git" --no-deps
pip install "git+https://github.com/google/aistplusplus_api.git"
```

### Submodule Setup

#### FoundationPose
```bash
# Python deps
pip install pysdf warp-lang
pip install "git+https://github.com/NVlabs/nvdiffrast.git" --no-build-isolation

# Build C++ extension (requires Eigen3, Boost, pybind11 — all in the grail env from above)
cd imports/FoundationPose/mycpp && rm -rf build && mkdir build && cd build
CMAKE_PREFIX_PATH=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
    cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j$(nproc)
cd ../../../..

# Patch and build CUDA extension (required for PyTorch 2.7+)
cd imports/FoundationPose/bundlesdf/mycuda
sed -i 's/std=c++14/std=c++17/g' setup.py
sed -i 's/\.type()/\.scalar_type()/g' common.cu
CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/include/eigen3 \
    pip install -e . --no-build-isolation
cd ../../../..
```

#### GEM-SMPL
```bash
pip install -e imports/GEM-SMPL/ --no-deps
pip install cython_bbox lapx wis3d

# Build DROID-SLAM CUDA extensions
cd imports/GEM-SMPL/third-party/DROID-SLAM
export CUDA_HOME=/usr/local/cuda-12.1
pip install -e . --no-build-isolation
cd ../../../..
```

#### GEM-X (native SOMA prediction)

GEM-X uses a separate environment and stores its approximately 7.7 GB verified
asset bundle outside the checkout. The helper pins source, model revisions,
file sizes, and SHA-256 hashes before linking them into the submodule.

```bash
git submodule update --init imports/GEM-X
export GRAIL_GEM_X_ASSETS="${XDG_CACHE_HOME:-$HOME/.cache}/grail/models/gem-x"
export GRAIL_GEM_X_ENV="${XDG_CACHE_HOME:-$HOME/.cache}/grail/envs/gem-x"
python scripts/setup/setup_gem_x.py \
    --assets "$GRAIL_GEM_X_ASSETS" --env "$GRAIL_GEM_X_ENV" \
    --grail-python "$(which python)" --link-submodule
```

#### MoGe-3 (isolated depth worker)

```bash
git submodule update --init imports/MoGe
bash scripts/setup/install_moge3.sh --python "$(which python)"
```

The default worker lives at `.venvs/moge3/bin/python`. Override it with
`GRAIL_MOGE_PYTHON=/absolute/path/to/python`.

#### MoSE3 + FoundationPose hybrid tracking

```bash
git submodule update --init imports/MoSE3 imports/FoundationPose
bash scripts/setup/install_mose3.sh --python "$(which python)"
bash scripts/setup/download_checkpoints.sh \
    --skip-gem-smpl --skip-gem-x --skip-foundationpose \
    --skip-hunyuan3d --skip-sonic
```

The worker and weights default to `.venvs/mose3/bin/python` and
`imports/MoSE3/ckpt`. The runtime can be overridden with
`GRAIL_MOSE3_PYTHON`, `GRAIL_MOSE3_ROOT`, and `GRAIL_MOSE3_CHECKPOINT`.

#### SOMA Retargeter (Unitree G1 + DEx3)

```bash
git submodule update --init imports/soma-retargeter imports/GMR
bash scripts/setup/install_soma_retargeter.sh --python "$(which python)"
```

This hydrates the bundled Unitree G1 retarget assets, verifies the original GMR
G1-with-DEx3 MJCF and meshes, and installs the pinned retargeter into
`imports/soma-retargeter/.venv`. Per-source DEx3 registry bundles are generated
at retarget time; no upstream submodule is modified. The upstream runtime requires
Python 3.12 and pins Newton, Warp, USD, NumPy, and SciPy through its `uv.lock`;
the helper therefore keeps it isolated from the Python 3.10 GRAIL environment
and the Python 3.11 SONIC environment. Do not install these dependencies into
either shared environment manually.

#### Hunyuan3D-2.1 (Separate Environment)

Hunyuan3D has conflicting dependency versions and runs in a separate conda environment:

```bash
conda create -y -n hunyuan python=3.10
conda activate hunyuan
pip install --upgrade pip 'setuptools<81' wheel   # lightning_fabric needs pkg_resources
conda install -y -c conda-forge 'gcc_linux-64=12' 'gxx_linux-64=12'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r imports/Hunyuan3D-2.1/requirements.txt

# Build extensions
cd imports/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
pip install --no-build-isolation -e .
cd ../DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../../../..

# Download RealESRGAN weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -P imports/Hunyuan3D-2.1/hy3dpaint/ckpt/

conda activate grail  # return to main env
```

#### Sonic (Isaac Lab + Isaac Sim)

Required for task-general tracking training and the explicit legacy
SMPL-X/GMR retargeting path. Native SOMA-to-G1 retargeting uses the separate
Python 3.12 environment installed above and does not require SONIC. The
one-shot path is:

```bash
bash scripts/setup/install_env_sonic.sh      # creates conda env 'sonic' (Python 3.11)
```

That installs Isaac Sim 5.1.0, Isaac Lab v2.3.2, public GMR, GRAIL, and the
retargeting/training dependencies, then pulls `imports/SONIC` LFS assets.
GRAIL-specific GMR behavior is applied at runtime by `grail.adapters.gmr`; the
public GMR submodule is not modified. Time: ~30 min. Disk: ~20 GB. EULA:
auto-accepted via `OMNI_KIT_ACCEPT_EULA=Yes`.

The published Docker image already contains this environment; inside that
container, use `conda activate sonic` directly instead of running the installer.

### GRAIL Package

```bash
pip install -e .
```

## Prepare Datasets

We use [ComAsset](https://huggingface.co/datasets/HyeonwooKim/ComAsset) as our main dataset:

```bash
bash scripts/setup/download_comasset.sh
```

Rate-limited for anonymous IPs — if you hit a limit, `huggingface-cli login` with a read token and retry.

Expected structure:
```
data/
└── ComAsset/
    ├── accordion/
    │   └── <object_id>/
    │       ├── images/
    │       ├── model.obj
    │       └── model.mtl
    ├── axe/
    └── ...
```

Other supported datasets: [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/), [InterCap](https://intercap.is.tue.mpg.de/), [FullBodyManip](https://github.com/lijiaman/omomo_release), [SAPIEN](https://sapien.ucsd.edu/).

## Download Checkpoints

```bash
bash scripts/setup/download_checkpoints.sh
```

Downloads the redistributable GEM-SMPL, verified GEM-X bundle, pinned MoSE3
weights, FoundationPose, RealESRGAN, and SONIC files. Native SOMA is the
default and does not require licensed SMPL assets. Only legacy SMPL/SMPL-X
configs need the licensed models; download them from the official
[SMPL](https://smpl.is.tue.mpg.de/) and
[SMPL-X](https://smpl-x.is.tue.mpg.de/) sites and place them at:

```text
imports/GEM-SMPL/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
imports/GEM-SMPL/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```

## Environment Variables

```bash
# Required
export CUDA_HOME=/usr/local/cuda-12.1
export PYOPENGL_PLATFORM=egl                  # headless rendering
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH

# Sonic / legacy GMR only
export OMNI_KIT_ACCEPT_EULA=Yes               # Isaac Sim EULA
export DISPLAY=:1                             # GMR mujoco viewer

# For 2D generation pipeline
export OPENAI_API_KEY=<your-key>
export KLING_ACCESS_KEY=<your-key>
export KLING_SECRET_KEY=<your-key>

# Optional cache paths
export HF_HOME=/path/to/cache/huggingface
export TORCH_HOME=/path/to/cache/torch
```
