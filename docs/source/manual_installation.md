# Manual Installation

<span id="manual-environment-setup"></span>

GRAIL has been developed and tested on Ubuntu 22.04+ with NVIDIA GPUs (A6000, RTX 4090, RTX 5090, RTX 6000 Ada).

The repo ships three conda envs:

- `grail` (Python 3.10) — 2D generation, 4D reconstruction, optimization
- `hunyuan` (Python 3.10) — Hunyuan3D-2.1 asset generation
- `sonic` (Python 3.11) — tracking and evaluation in Isaac Lab and Isaac Sim

Fresh 4D reconstruction also uses three isolated workers: GEM-X for native
SOMA pose prediction, MoGe-3 for depth, and MoSE3 for object-track proposals.
Their dependency sets are intentionally kept out of the `grail` environment.
The public SOMA Retargeter uses its own Python 3.12 environment and currently
exposes only the `unitree_g1` target through GRAIL.

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
    --skip-gem-x --skip-foundationpose \
    --skip-hunyuan3d
```

The worker and weights default to `.venvs/mose3/bin/python` and
`imports/MoSE3/ckpt`. The runtime can be overridden with
`GRAIL_MOSE3_PYTHON`, `GRAIL_MOSE3_ROOT`, and `GRAIL_MOSE3_CHECKPOINT`.

<span id="soma-retargeter-unitree-g1-dex3"></span>

#### SOMA Retargeter (Unitree G1 with articulated hands)

```bash
git submodule update --init imports/soma-retargeter
git lfs pull --include="assets/robots/unitree_g1/meshes/**"
bash scripts/setup/install_soma_retargeter.sh --python "$(which python)"
```

This hydrates the bundled Unitree G1 retarget assets, verifies the pinned G1 hand model and its mesh files, and installs the pinned retargeter into
`imports/soma-retargeter/.venv`. Per-source hand-target registry bundles are generated
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

### GRAIL Package

```bash
pip install -e .
```

## Dataset preparation

<span id="prepare-datasets"></span>

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

## Checkpoints

<span id="download-checkpoints"></span>

```bash
bash scripts/setup/download_checkpoints.sh
```

## Environment Variables

```bash
# Required
export CUDA_HOME=/usr/local/cuda-12.1
export PYOPENGL_PLATFORM=egl                  # headless rendering
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH


export OMNI_KIT_ACCEPT_EULA=Yes               # Isaac Sim EULA

# For video generation, first set the provider credentials in your shell or secret manager.
export OPENAI_API_KEY
export KLING_ACCESS_KEY
export KLING_SECRET_KEY

# Optional cache paths
export HF_HOME=/path/to/cache/huggingface
export TORCH_HOME=/path/to/cache/torch
```

## Tracking environment installation

<span id="install-from-scratch-or-add-the-tracking-environment"></span>

To override the FoundationPose source used by the MoSE3 hybrid worker, set `GRAIL_FOUNDATIONPOSE_ROOT` as well as the MoSE3 paths above.

For a Linux host environment, the repository provides the {blob}`scratch installer <scripts/setup/install_env_scratch.sh>`. Initialize the submodules and hydrate the robot meshes as in [GRAIL Installation Guide](installation.md). Inspect the installer options and choose a new environment before running it:

```bash
bash scripts/setup/install_env_scratch.sh --help
```

The separate {blob}`SONIC installer <scripts/setup/install_env_sonic.sh>` sets up the Isaac Lab training environment. Follow the licence step in [tracking downloads](downloads.md#tracking-weights). For checkpoints and the optional ComAsset dataset use the download scripts described above. The tutorial timings cover the released Docker setup. No new host installation was timed.

Choose `bash scripts/setup/install_env_scratch.sh` for the scripted host install or the component-specific commands above for an existing environment. The top-level Dockerfile is a source-install bootstrap. The [release container](installation.md) already supplies the three Conda environments. The tracking installer is `bash scripts/setup/install_env_sonic.sh`; it requires Isaac Sim and acceptance of its licence.

For a custom depth-worker location, run `scripts/setup/install_moge3.sh` with `--python` set to your GRAIL interpreter and `--venv` set to the new worker directory. Point `GRAIL_MOGE_PYTHON` at the resulting interpreter. The worker shares base PyTorch and isolates the Triton dependencies. Revalidate it after changing base PyTorch. Its setup receipt is `.venvs/moge3/runtime_setup.json`; installation alone does not verify depth inference. The main reconstruction process stays in the GRAIL environment.
