#!/bin/bash
# Install / augment the `sonic` conda env used by GRAIL retargeting + SONIC training.
#
# Usage:
#   bash scripts/setup/install_env_sonic.sh              # default env 'sonic'
#   GRAIL_SONIC_ENV=my_sonic_env bash scripts/setup/install_env_sonic.sh
#   BOOTSTRAP_SONIC=0 bash scripts/setup/install_env_sonic.sh         # skip Isaac Sim/Lab
#                                                                      (assume already installed)
#   INSTALL_SYSTEM_DEPS=0 bash scripts/setup/install_env_sonic.sh      # skip apt step
#   PULL_LFS=0 bash scripts/setup/install_env_sonic.sh                 # skip git-lfs pull
#
# What this script does, in order:
#   1. (INSTALL_SYSTEM_DEPS=1 — default when apt+sudo/root are available)
#       Install vulkan/GUI libs + git-lfs via apt. Uses sudo if needed;
#       no-op if we're neither root nor have sudo.
#   2. (BOOTSTRAP_SONIC=1 — default) Create the conda env with Python 3.11,
#      pip-install Isaac Sim 5.1.0 (`isaacsim[all,extscache]`), clone Isaac
#      Lab v2.3.2 to $ISAAC_LAB_DIR (default: ~/IsaacLab), run
#      `./isaaclab.sh --install all`, pip install the core `isaaclab`
#      editable, and install `vector_quantize_pytorch`. Set BOOTSTRAP_SONIC=0
#      to skip when you already have an env with IsaacLab/IsaacSim installed
#      (e.g. gearenv).
#   3. Symlinks data/motion_lib_genhoi + models into imports/SONIC/gear_sonic/.
#   4. pip install -e imports/SONIC/gear_sonic[training]
#      + GRAIL package (editable) + huggingface_hub.
#   5. pip install retargeting-specific deps (smplx, mujoco, pxr, trimesh, ...).
#   6. (PULL_LFS=1 — default when git-lfs is on PATH) git-lfs pull on
#      imports/SONIC so the robot mesh STLs + policy ONNX materialize.
#   7. Verify critical package versions and sanity-import top-level modules.

set -eo pipefail

ENV_NAME="${GRAIL_SONIC_ENV:-sonic}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BOOTSTRAP_SONIC="${BOOTSTRAP_SONIC:-1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
PULL_LFS="${PULL_LFS:-1}"
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-$HOME/IsaacLab}"
ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-5.1.0}"
ISAAC_LAB_TAG="${ISAAC_LAB_TAG:-v2.3.2}"

# Core versions for a reproducible `sonic` environment. Keep these fixed so
# local bootstraps do not drift as unconstrained PyPI/Git dependencies change.
SONIC_PYTHON_VERSION="3.11.15"
SONIC_NUMPY_VERSION="1.26.4"
SONIC_OPENCV_VERSION="4.13.0.92"
SMPLX_COMMIT="1265df7ba545e8b00f72e7c557c766e15c71632f"
SMPL_SIM_COMMIT="b5c08720503ad5fff64050c4d289c42d947fcf8d"

echo ">>> Target conda env: ${ENV_NAME}"
echo ">>> Repo root:        ${REPO_ROOT}"
echo ">>> Bootstrap mode:   ${BOOTSTRAP_SONIC} (1=install Isaac Sim/Lab, 0=assume present)"

# --- Step 1: system deps (Vulkan/GUI/git-lfs) via apt -------------------
# Idempotent: re-installs are a fast pass. Skipped entirely on non-apt
# systems or when we can't elevate.
if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]] && command -v apt-get &>/dev/null; then
    APT_PKGS=(
        libvulkan1 vulkan-tools mesa-vulkan-drivers
        libxcb-xfixes0 libxcb-cursor0 libxrandr2 libxi6 libxcursor1
        libxtst6 libxss1 libxrender1 libgl1 libegl1
        git-lfs
    )
    if [[ "$(id -u)" -eq 0 ]]; then
        APT_CMD="apt-get"
    elif sudo -n true 2>/dev/null; then
        APT_CMD="sudo apt-get"
    else
        APT_CMD=""
        echo ">>> [skip apt] not root and no passwordless sudo; install these manually if missing:"
        echo "    ${APT_PKGS[*]}"
    fi
    if [[ -n "${APT_CMD}" ]]; then
        echo ">>> Installing system deps via ${APT_CMD} (Vulkan, GUI, git-lfs)"
        ${APT_CMD} update -qq
        ${APT_CMD} install -y --no-install-recommends "${APT_PKGS[@]}" | tail -3
    fi
fi

# --- Step 2: bootstrap the env + Isaac Sim + Isaac Lab ------------------
eval "$(conda shell.bash hook)"

if [[ "${BOOTSTRAP_SONIC}" == "1" ]]; then
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo ">>> Creating conda env '${ENV_NAME}' with Python ${SONIC_PYTHON_VERSION}"
        conda create -y -n "${ENV_NAME}" "python=${SONIC_PYTHON_VERSION}"
    fi
    conda activate "${ENV_NAME}"

    echo ">>> Installing pinned Python build tools"
    pip install 'pip==26.0.1'

    if ! python -c "import isaacsim" 2>/dev/null; then
        echo ">>> Installing Isaac Sim ${ISAAC_SIM_VERSION} (~6 GB download)"
        pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
            --extra-index-url https://pypi.nvidia.com
    fi

    # Accept EULA non-interactively on first import. The Kit kernel checks
    # for the literal file <isaacsim_pkg>/kit/EULA_ACCEPTED before showing
    # its interactive prompt — write it directly so this works in non-TTY
    # non-interactive builds where stdin is closed and the
    # `python -c "import isaacsim"` workaround silently fails.
    export OMNI_KIT_ACCEPT_EULA=Yes
    ISAACSIM_PKG=$(python -c "import isaacsim, os; print(os.path.dirname(isaacsim.__file__))")
    echo "yes" > "${ISAACSIM_PKG}/kit/EULA_ACCEPTED"
    python -c "import isaacsim" >/dev/null

    # Pre-install flatdict without build isolation. flatdict 4.0.1 (pinned by
    # Isaac Lab core) has a legacy setup.py that imports pkg_resources, which
    # setuptools 81+ removed. PEP 517 build isolation installs the latest
    # setuptools, so the wheel build fails. Pin setuptools<81 in the env
    # first, then build flatdict against it.
    pip install 'setuptools==80.10.2' 'wheel==0.46.3'
    pip install 'flatdict==4.0.1' --no-build-isolation

    if [[ ! -d "${ISAAC_LAB_DIR}" ]]; then
        echo ">>> Cloning Isaac Lab ${ISAAC_LAB_TAG} to ${ISAAC_LAB_DIR}"
        git clone --depth 1 --branch "${ISAAC_LAB_TAG}" \
            https://github.com/isaac-sim/IsaacLab.git "${ISAAC_LAB_DIR}"
    fi

    if ! python -c "import isaaclab" 2>/dev/null; then
        echo ">>> Running ./isaaclab.sh --install all (~10-15 min, ~4 GB)"
        (
            ISAACLAB_CONSTRAINTS="$(mktemp)"
            trap 'rm -f "${ISAACLAB_CONSTRAINTS}"' EXIT
            printf '%s\n' \
                'numpy==1.26.0' \
                'skrl==2.0.0' \
                'stable-baselines3==2.8.0' \
                'tensordict==0.12.2' \
                'torch==2.7.0+cu128' \
                'torchvision==0.22.0+cu128' \
                'triton==3.3.0' \
                > "${ISAACLAB_CONSTRAINTS}"
            export PIP_CONSTRAINT="${ISAACLAB_CONSTRAINTS}"
            export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com https://download.pytorch.org/whl/cu128"
            cd "${ISAAC_LAB_DIR}"
            ./isaaclab.sh --install all
        )
        # Isaac Lab's --install flag sometimes skips the core `isaaclab`
        # package when a transitive dep (e.g., flatdict) failed during the
        # first pass. Install it explicitly to be safe.
        pip install --no-deps -e "${ISAAC_LAB_DIR}/source/isaaclab"
    fi

    # Required by some gear_sonic configs.
    pip install 'vector-quantize-pytorch==1.28.2'
else
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "ERROR: conda env '${ENV_NAME}' does not exist and BOOTSTRAP_SONIC=0." >&2
        echo "       Either unset BOOTSTRAP_SONIC (default bootstrap) or create the env first." >&2
        exit 1
    fi
    conda activate "${ENV_NAME}"
fi

# --- Step 3: surface data/ and models/ into the SONIC submodule ---------
# imports/SONIC/gear_sonic/ is the cwd for training scripts; it expects
# data/motion_lib_genhoi/... and models/... to resolve from there.
GEAR_SONIC="${REPO_ROOT}/imports/SONIC/gear_sonic"
mkdir -p "${REPO_ROOT}/data/motion_lib_genhoi" "${REPO_ROOT}/models"
ln -sfn ../../../../data/motion_lib_genhoi "${GEAR_SONIC}/data/motion_lib_genhoi"
ln -sfn ../../../models "${GEAR_SONIC}/models"
echo ">>> Linked ${GEAR_SONIC}/{data/motion_lib_genhoi,models} -> repo root"

# --- Step 4: editable installs ------------------------------------------
echo ">>> Install pinned NumPy and OpenCV"
pip install --no-deps \
    "numpy==${SONIC_NUMPY_VERSION}" \
    "opencv-python==${SONIC_OPENCV_VERSION}"

echo ">>> pip install -e imports/SONIC/gear_sonic[training] + huggingface_hub"
pip install \
    'wandb==0.26.1' \
    'transformers==4.57.6' \
    'accelerate==1.13.0' \
    'tensorboard==2.20.0' \
    -e "${GEAR_SONIC}[training]"
pip install 'huggingface-hub==0.36.2'

echo ">>> pip install -e . (grail, --no-deps)"
# --no-deps: grail's setup.cfg has unpinned numpy/opencv-python, which resolve
# to numpy 2.x + opencv 4.13 and break gear_sonic (numpy==1.26.4), isaaclab-rl
# (numpy<2), and isaacsim-kernel (numpy==1.26.0). The sonic env only consumes
# grail.retargeting; its real deps (smplx, scipy, mujoco, mink, trimesh, pxr,
# isaaclab) are installed by other steps in this script.
pip install --no-deps -e "${REPO_ROOT}"

# --- Step 5: retargeting-specific deps ----------------------------------
echo ">>> pip install retargeting deps"
pip install \
    "smplx @ git+https://github.com/vchoutas/smplx@${SMPLX_COMMIT}" \
    'joblib==1.5.3' \
    'trimesh==4.5.1' \
    'usd-core==26.3' \
    'scipy==1.15.3' \
    'rich==15.0.0' \
    'tqdm==4.67.3' \
    'loop-rate-limiters==1.2.0' \
    'natsort==8.4.0' \
    'protobuf==7.34.1' \
    'redis[hiredis]==7.4.0' \
    'imageio[ffmpeg]==2.37.0' \
    'mujoco==3.7.0' \
    'mink==1.1.0' \
    'qpsolvers[proxqp]==4.11.0' \
    'warp-lang==1.12.1' \
    'simple-raycaster @ git+https://github.com/Agent-3154/simple-raycaster.git@197daa6dcb146c5ce3e675a173328e17df6b9777'

# --- Step 5b: SONIC training/eval-callback deps -------------------------
# smpl_sim is a non-PyPI package providing compute_metrics_lite, used by the
# SONIC eval-watcher's im_eval callback (gear_sonic/trl/callbacks/im_eval_callback.py).
# Without it, eval `python eval_agent_trl.py` crashes at metrics computation
# and no rendered videos get uploaded to wandb.
#
# Install the pinned dependency versions explicitly, then install SMPLSim
# without dependency resolution. SMPLSim declares only `numpy>1.16.1`,
# so resolving it normally upgrades the env to NumPy 2.x and breaks SONIC,
# Isaac Lab, dex-retargeting, and numba.
pip install \
    "numpy==${SONIC_NUMPY_VERSION}" \
    'numpy-stl==3.2.0' \
    'easydict==1.13' \
    'gymnasium==1.2.1' \
    'mediapy==1.2.6' \
    'torchgeometry==0.1.2' \
    'vtk==9.6.1'
pip install --no-deps \
    "smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git@${SMPL_SIM_COMMIT}"

# Keep runtime-sensitive transitive packages pinned. --no-deps avoids
# re-resolving the intentional NumPy/OpenCV combination.
echo ">>> Applying runtime pins"
python -m pip install --no-deps \
    'click==8.1.7' \
    'daqp==0.7.2' \
    'datasets==4.8.4' \
    'packaging==26.1' \
    'psutil==5.9.8' \
    'redis==7.4.0' \
    'skrl==2.0.0' \
    'stable-baselines3==2.8.0' \
    'tensordict==0.12.2' \
    'typing_extensions==4.15.0' \
    'warp-lang==1.12.1'

# --- Step 6: git-lfs pull for SONIC assets ------------------------------
# Mesh STLs + policy ONNX files are LFS-tracked. Without this pull, the
# preflight check fails at the size check (pointer files are <1 KB).
if [[ "${PULL_LFS}" == "1" ]] && command -v git-lfs &>/dev/null; then
    echo ">>> git lfs install + pull in imports/SONIC"
    # Avoid git's "dubious ownership" refusal when running as root inside a
    # container against a bind-mounted host repo (different UIDs).
    git config --global --add safe.directory "${REPO_ROOT}" 2>/dev/null || true
    git config --global --add safe.directory "${REPO_ROOT}/imports/SONIC" 2>/dev/null || true
    # --skip-repo: set up global LFS filters only. Without it, `git lfs install`
    # aborts with exit 2 when an identical pre-push hook already exists in the
    # cwd repo (idempotency foot-gun under `set -e`). `git lfs pull` below
    # works regardless since SONIC already has the hook.
    git lfs install --skip-repo
    if ! (cd "${REPO_ROOT}/imports/SONIC" && git lfs pull) | tail -3; then
        echo "ERROR: git lfs pull failed; required SONIC meshes/checkpoints may still be pointers." >&2
        echo "       Retry manually, or set PULL_LFS=0 only if the assets are already present." >&2
        exit 1
    fi
elif [[ "${PULL_LFS}" == "1" ]]; then
    echo "ERROR: PULL_LFS=1 but git-lfs is not on PATH." >&2
    echo "       Install git-lfs and retry, or set PULL_LFS=0 only if assets are already present." >&2
    exit 1
fi

# --- Step 7: critical version and import checks --------------------------
echo ">>> Verifying install"
if ! OMNI_KIT_ACCEPT_EULA=Yes python - <<'PY'
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

expected = {
    "accelerate": "1.13.0",
    "click": "8.1.7",
    "daqp": "0.7.2",
    "datasets": "4.8.4",
    "gear_sonic": "0.1.0",
    "grail": "0.1.0",
    "huggingface_hub": "0.36.2",
    "isaaclab": "0.54.2",
    "isaacsim": "5.1.0.0",
    "mink": "1.1.0",
    "mujoco": "3.7.0",
    "numpy": "1.26.4",
    "opencv-python": "4.13.0.92",
    "packaging": "26.1",
    "psutil": "5.9.8",
    "qpsolvers": "4.11.0",
    "redis": "7.4.0",
    "scipy": "1.15.3",
    "skrl": "2.0.0",
    "smpl_sim": "0.0.1",
    "smplx": "0.1.28",
    "stable_baselines3": "2.8.0",
    "tensorboard": "2.20.0",
    "tensordict": "0.12.2",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "transformers": "4.57.6",
    "trl": "0.28.0",
    "typing_extensions": "4.15.0",
    "wandb": "0.26.1",
    "warp-lang": "1.12.1",
}

problems = []
if platform.python_version() != "3.11.15":
    problems.append(
        f"Python: expected 3.11.15, found {platform.python_version()}"
    )

for package, wanted in expected.items():
    try:
        found = version(package)
    except PackageNotFoundError:
        problems.append(f"{package}: missing (expected {wanted})")
        continue
    if found != wanted:
        problems.append(f"{package}: expected {wanted}, found {found}")

if problems:
    print("ERROR: sonic env has unexpected critical package versions:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    raise SystemExit(1)

import gear_sonic
import isaaclab
import isaacsim
import smpl_sim
from importlib.util import find_spec
from grail.retargeting import terrain
from grail.retargeting.manipulation import handoff, pack

for module in ("gear_sonic.train_agent_trl", "gear_sonic.eval_contactfirst_trl"):
    if find_spec(module) is None:
        raise ImportError(f"Missing tracking module: {module}")

print("  GRAIL tracking (terrain pack, manipulation handoff and pack, SONIC train and evaluate) + gear_sonic + smpl_sim: OK")
print("  Isaac Lab + Isaac Sim: OK")
print("  Critical package versions verified")
PY
then
    echo "ERROR: sonic environment verification failed." >&2
    exit 1
fi

echo ""
echo "Setup complete. Quick start:"
echo "  python -m grail.retargeting.manipulation.handoff --help"
echo ""
echo "Full preflight:"
echo "  OMNI_KIT_ACCEPT_EULA=Yes python imports/SONIC/check_environment.py --training"
