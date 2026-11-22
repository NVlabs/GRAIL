# GRAIL Installation Guide

<span id="installation"></span>

These instructions use the standard image for both released examples. Run host commands on your GPU workstation and keep the container shell open throughout the tutorial. For building individual environments directly on Linux, see the [manual installation reference](manual_installation.md).

## Host prerequisites

<span id="check-codex-on-the-host"></span>

Reconstruction and retargeting use OpenAI Codex CLI to run commands against the named GPU container from a second host terminal. Install the CLI and sign in using the [official CLI guide](https://developers.openai.com/codex/cli/). Check your sign-in before pulling the image.

```bash
codex --version
codex login status
```

Allow a few seconds. Look for a CLI version and a signed-in status. The walkthroughs used version 0.170.0, model `gpt-6.2-sol` and a host account with Docker access. Without `-m`, Codex uses the model selected by your CLI configuration or its default. Only the named model was validated here. With `--approve-for-me`, Codex sends approval requests to automatic review under the workspace-write sandbox. Approved Docker commands run with your host account’s Docker access. See the [Codex CLI reference](https://developers.openai.com/codex/cli/reference/). Review the resulting files and movies after each agent step. Available models, billing and limits depend on your account; we did not measure a cost per case. For a manual fit starting from prepared inputs, see the [reconstruction guide](recon_4dhoi.md).

## Docker (Recommended)

<span id="get-the-code-and-the-image"></span>

**Host.** Use an empty working directory on the disk selected in [System Requirements](requirements.md). Clone the public repository, initialize its pinned dependencies and download Git LFS payloads. The source files alone contain only pointers for large assets. Allow a few minutes on a fast network. The release checkout contained about 2.94 GB of LFS payloads across the repository, in addition to ordinary Git objects and submodules.

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

For a reproducible pull, the immutable standard image reference is `nvgrail/grail@sha256:14eb489f0d6eb80b8ae198d44bfd4800695af27202a7a38b504c57d22f65ad8e`. The corresponding MiniMax-H3 image is `nvgrail/grail@sha256:fdbee9f32bd525d4d68c7346dd677d09c24c95575084061e67e5a290a60186d7`. Either reference can replace its tag in the Docker command.

## GPU container

<span id="open-the-gpu-shell"></span>

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

## Worker installation

<span id="install-the-reconstruction-and-robot-workers"></span>

**Container shell.** Set up Blender, the GPU rasterizer and the separate pose, depth, object-tracking and SOMA retargeting environments. On a fast connection, allow 5–15 minutes for the first install. Look for `SOMA Retargeter Unitree G1 runtime OK`, `GRAIL native SOMA runtime OK` and finally `Docker setup complete.`.

```bash
bash scripts/setup/install_env_docker.sh
```

## Existing runs

<span id="returning-to-a-run"></span>

The checkout, `../grail-demo/cache` and `../grail-demo/runs` are host mounts. Weights stored there, reviews and checkpoints survive closing the shell. Files created as root inside those mounts may be root-owned on the host. The disposable container’s own filesystem goes away when its shell exits.

For a long run over SSH, start the container inside a terminal multiplexer such as tmux and reconnect to that session. If the container has already exited, repeat **Open the GPU shell** with the same three mounts, then repeat the worker installer and the [tracking setup](downloads.md#tracking-weights) if you need tracking. The worker installer reused the cache in about one minute; tracking setup and the weight check took about eight minutes. Keep any completed stage files and use a fresh directory for the next training attempt.

## Host and container commands

<span id="where-to-type-the-tutorial-commands"></span>

Keep the container shell for setup, packing and training. Open a second **host terminal** in the checkout for the two Codex prompts. With the container running, check Docker access from that host account without sudo:

```bash
docker exec grail-demo nvidia-smi -L
```

Expect a line naming your GPU within seconds. If access is denied or the container is missing, resolve it before starting a fit. See [Troubleshooting](troubleshooting.md) and the [worker glossary](configuration.md).

Next, [download the checkpoints and verify the example inputs](downloads.md).

## Alternative installations

<span id="more-installation-routes"></span>

<span id="quick-start"></span>

[Start the release container](#gpu-container).

<span id="prerequisites"></span>

[Check GPU, OS and disk requirements](requirements.md).

<span id="install-from-scratch"></span><span id="environment-setup-manual"></span><span id="system-dependencies"></span><span id="python-environment"></span><span id="submodule-setup"></span><span id="foundationpose"></span><span id="gem-x-native-soma-prediction"></span><span id="moge-3-isolated-depth-worker"></span><span id="mose3-foundationpose-hybrid-tracking"></span><span id="soma-retargeter-unitree-g1-dex3"></span><span id="hunyuan3d-2-1-separate-environment"></span><span id="sonic-isaac-lab-isaac-sim"></span><span id="grail-package"></span>

[Manual installation](manual_installation.md) covers host packages, isolated workers, robot assets and SONIC.

<span id="gem-smpl"></span>

GEM-X and the fixed SOMA identity are the supported human prediction route. See [worker installation](manual_installation.md).

<span id="prepare-datasets"></span><span id="download-checkpoints"></span><span id="environment-variables"></span>

[Get examples and checkpoints](downloads.md). See [manual installation](manual_installation.md) for ComAsset and environment settings.
