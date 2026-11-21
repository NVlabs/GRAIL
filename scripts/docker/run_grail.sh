#!/bin/bash
# Launch a persistent GRAIL container with GPU + GUI (X11) forwarding, repo
# bind-mount, and Isaac Sim cache volumes.
#
# Usage:
#   bash scripts/docker/run_grail.sh                   # create & enter (or re-enter)
#   bash scripts/docker/run_grail.sh --rebuild         # destroy existing and recreate
#   bash scripts/docker/run_grail.sh -- <cmd> <args>   # exec <cmd> instead of bash
#
# Env var overrides:
#   GRAIL_IMAGE          image tag    (default: docker.io/nvgrail/grail:latest)
#   GRAIL_CONTAINER      container    (default: grail-sonic)
#   GRAIL_CACHE_DIR      host cache   (default: ~/.grail-sonic-cache)
#   GRAIL_HF_CACHE       HF cache     (default: ~/.cache/huggingface)
#
# Behavior:
#   - First run: `docker run` creates the named container; you land in a
#     `(grail) root@...:/workspace/grail#` shell. Repo is at /workspace/grail.
#   - To exit without stopping: Ctrl+P then Ctrl+Q.
#   - Subsequent runs: reuses the existing container via `docker start -ai`
#     (or `docker exec` if it's already running).
#   - --rebuild: `docker rm -f` the container first, then recreate.
#
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${GRAIL_IMAGE:-docker.io/nvgrail/grail:latest}"
NAME="${GRAIL_CONTAINER:-grail-sonic}"
CACHE_DIR="${GRAIL_CACHE_DIR:-$HOME/.grail-sonic-cache}"
HF_CACHE="${GRAIL_HF_CACHE:-$HOME/.cache/huggingface}"

REBUILD=0
EXEC_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild) REBUILD=1; shift ;;
        --) shift; EXEC_ARGS=("$@"); break ;;
        -h|--help) grep '^#' "$0" | head -30 | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown arg: $1 (use -- to pass command to container)" >&2; exit 1 ;;
    esac
done
[[ ${#EXEC_ARGS[@]} -eq 0 ]] && EXEC_ARGS=("bash")

mkdir -p "${CACHE_DIR}"/{kit,ov,pip,glcache,computecache,logs,data,documents} "${HF_CACHE}"

if [[ -n "${DISPLAY:-}" ]]; then
    xhost +local:root >/dev/null 2>&1 || true
fi

if [[ "${REBUILD}" -eq 1 ]] && docker inspect "${NAME}" >/dev/null 2>&1; then
    echo ">>> Removing existing container '${NAME}'"
    docker rm -f "${NAME}" >/dev/null
fi

# Re-use existing container when possible.
if docker inspect "${NAME}" >/dev/null 2>&1; then
    RUNNING="$(docker inspect --format='{{.State.Running}}' "${NAME}")"
    if [[ "${RUNNING}" == "true" ]]; then
        echo ">>> Exec'ing into running container '${NAME}'"
        exec docker exec -it "${NAME}" "${EXEC_ARGS[@]}"
    else
        echo ">>> Starting stopped container '${NAME}'"
        exec docker start -ai "${NAME}"
    fi
fi

# First-time create.
#
# `--entrypoint ""` clears the image's baked-in entrypoint
# (`tools/docker-entrypoint.sh`, a relative path that resolves to a
# non-executable host file once we bind-mount, and which also `gosu`s down
# to the host UID — stripping root, which breaks `apt install`). We run as
# root with a real DISPLAY forwarded; the entrypoint isn't doing anything
# useful in that setup, so we pass CMD directly instead.
echo ">>> Creating container '${NAME}' from image '${IMAGE}'"
exec docker run -it \
    --name "${NAME}" \
    --gpus all \
    --net=host \
    --ipc=host \
    --privileged \
    --entrypoint="" \
    --env="DISPLAY=${DISPLAY:-}" \
    --env="OMNI_KIT_ACCEPT_EULA=Yes" \
    --env="PRIVACY_CONSENT=Y" \
    --env="XDG_RUNTIME_DIR=/tmp" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${REPO_ROOT}:/workspace/grail" \
    -v "${HF_CACHE}:/root/.cache/huggingface:rw" \
    -v "${CACHE_DIR}/kit:/isaac-sim/kit/cache:rw" \
    -v "${CACHE_DIR}/ov:/root/.cache/ov:rw" \
    -v "${CACHE_DIR}/pip:/root/.cache/pip:rw" \
    -v "${CACHE_DIR}/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "${CACHE_DIR}/computecache:/root/.nv/ComputeCache:rw" \
    -v "${CACHE_DIR}/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "${CACHE_DIR}/data:/root/.local/share/ov/data:rw" \
    -v "${CACHE_DIR}/documents:/root/Documents:rw" \
    -w /workspace/grail \
    "${IMAGE}" \
    "${EXEC_ARGS[@]}"
