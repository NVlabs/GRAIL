# Installation

These instructions use the standard image for both released examples. Run host commands on your GPU workstation and keep the container shell open throughout the tutorial. For building individual environments directly on Linux, see the [manual installation reference](manual_installation.md).

## Check Codex on the host

Reconstruction and retargeting use OpenAI Codex CLI to run commands against the named GPU container from a second host terminal. Install the CLI and sign in using the [official CLI guide](https://developers.openai.com/codex/cli/). Check your sign-in before pulling the image.

```bash
codex --version
codex login status
```

Allow a few seconds. Look for a CLI version and a signed-in status. The walkthroughs used version 0.170.0, model `gpt-6.2-sol` and a host account with Docker access. Without `-m`, Codex uses the model selected by your CLI configuration or its default. Only the named model was validated here. With `--approve-for-me`, Codex sends approval requests to automatic review under the workspace-write sandbox. Approved Docker commands run with your host account’s Docker access. See the [Codex CLI reference](https://developers.openai.com/codex/cli/reference/). Review the resulting files and movies after each agent step. Available models, billing and limits depend on your account; we did not measure a cost per case. For a manual fit starting from prepared inputs, see the [reconstruction guide](recon_4dhoi.md).

## Get the code and the image

**Host.** Use an empty working directory on the disk selected in [Getting started](requirements.md). Clone the public repository, initialize its pinned dependencies and download Git LFS payloads. The source files alone contain only pointers for large assets. Allow a few minutes on a fast network. The release checkout contained about 2.94 GB of LFS payloads across the repository, in addition to ordinary Git objects and submodules.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/GRAIL.git GRAIL
cd GRAIL
git submodule update --init --recursive
git lfs pull
```

**Host.** Pull the image once. Docker prints its digest on success. A cold pull took about nine minutes on a fast connection; an existing image takes seconds to check.

```bash
docker pull nvgrail/grail:1.2.0
```

## Open the GPU shell

**Host, from the checkout root.** Start a container called `grail-demo`. Mount the checkout, a persistent model cache and a results directory. Expect a shell at `/workspace/grail` within seconds. The checkout and mounted directories persist when you exit the shell. Docker removes the container when the shell exits; finish both tutorials before closing it, or repeat setup in a new container using the retained mounts.

```bash
mkdir -p ../grail-demo/runs ../grail-demo/cache
docker run --rm -it --gpus all --shm-size=16g --name grail-demo \
  -v "$PWD:/workspace/grail" -w /workspace/grail \
  -v "$PWD/../grail-demo/runs:/workspace/runs" \
  -v "$PWD/../grail-demo/cache:/root/.cache" \
  nvgrail/grail:1.2.0 bash
```

**Inside the container.** Confirm that Docker can see the GPU. The model and memory should match the host reading. If the device is missing, configure NVIDIA Container Toolkit on the host before proceeding.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

## Install the reconstruction and robot workers

**Container shell.** Set up Blender, the GPU rasterizer and the separate pose, depth, object-tracking and SOMA retargeting environments. On a fast connection, allow 5–15 minutes for the first install. Look for `SOMA Retargeter Unitree G1 runtime OK`, `GRAIL native SOMA runtime OK` and finally `Docker setup complete.`.

```bash
bash scripts/setup/install_env_docker.sh
```

## Returning to a run

The checkout, `../grail-demo/cache` and `../grail-demo/runs` are host mounts. Weights stored there, reviews and checkpoints survive closing the shell. Files created as root inside those mounts may be root-owned on the host. The disposable container’s own filesystem goes away when its shell exits.

For a long run over SSH, start the container inside a terminal multiplexer such as tmux and reconnect to that session. If the container has already exited, repeat **Open the GPU shell** with the same three mounts, then repeat the worker installer and the [tracking setup](downloads.md#tracking-weights) if you need tracking. The worker installer reused the cache in about one minute; tracking setup and the weight check took about eight minutes. Keep any completed stage files and use a fresh directory for the next training attempt.

## Where to type the tutorial commands

Keep the container shell for setup, packing and training. Open a second **host terminal** in the checkout for the two Codex prompts. With the container running, check Docker access from that host account without sudo:

```bash
docker exec grail-demo nvidia-smi -L
```

Expect a line naming your GPU within seconds. If access is denied or the container is missing, resolve it before starting a fit. See [Troubleshooting](troubleshooting.md) and the [worker glossary](configuration.md).

Next, [download the checkpoints and verify the example inputs](downloads.md).
