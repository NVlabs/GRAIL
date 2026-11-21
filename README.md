<div align="center">

# GRAIL: Generating Humanoid Loco-Manipulation from World Models

<p><strong>CoRL 2026 · Best Paper Award Finalist</strong></p>

<p>
  <a href="https://research.nvidia.com/labs/dair/grail/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page"/>
  </a>
  <a href="https://arxiv.org/pdf/2606.05160">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=flat-square" alt="Paper"/>
  </a>
  <a href="https://nvlabs.github.io/GRAIL/">
    <img src="https://img.shields.io/badge/Docs-online-success?style=flat-square" alt="Docs"/>
  </a>
  <a href="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=000" alt="Dataset"/>
  </a>
</p>



<img src="assets/videos/teaser.gif" alt="GRAIL teaser" width="100%"/>

</div>

**GRAIL** is a fully digital data-generation pipeline for humanoid loco-manipulation. It composes 3D assets, simulator-ready scenes, robot-proportioned characters, and video foundation model priors to synthesize metric 4D human-object interaction (HOI) trajectories, then retargets them to a Unitree G1 and trains task-general policies for pick-up, whole-body manipulation, sitting, and terrain traversal. Using only GRAIL-generated data, the resulting egocentric visual policies transfer to real-world object pick-up and stair-climbing.

## News

- **[2026-11-20]** 🧩 **Agentic reconstruction and retargeting with SOMA body support** — released the agentic 4D HOI reconstruction and Unitree G1 retargeting workflow on the SOMA body model.
- **[2026-08-25]** 🎬 **MiniMax-H3 video generation** — added an optional local MiniMax-H3/Sol-Engine backend for the 2D-HOI pipeline. See the [2D-HOI generation guide](https://nvlabs.github.io/GRAIL/gen_2dhoi.html#minimax-h3-provider).
- **[2026-07-15]** 🤖 **Task-general tracking checkpoints** — released policy checkpoints trained on the released GRAIL data. See the [tracking guide](https://nvlabs.github.io/GRAIL/tracking.html#tracking-released-data) and [dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL).
- **[2026-06-03]** 🎉 **Code release** — released the GRAIL codebase and documentation.

## Motion Gallery

| Tabletop Pickup | Ground Pickup |
|:---:|:---:|
| <img src="assets/videos/pickup_table.gif" width="420"/> | <img src="assets/videos/pickup_ground.gif" width="420"/> |

| Tabletop Manipulation | Ground Manipulation |
|:---:|:---:|
| <img src="assets/videos/manip_tabletop.gif" width="420"/> | <img src="assets/videos/manip_large.gif" width="420"/> |

| Sitting | Curb |
|:---:|:---:|
| <img src="assets/videos/sitting.gif" width="420"/> | <img src="assets/videos/terrain_curbs.gif" width="420"/> |

| Slope | Stairs |
|:---:|:---:|
| <img src="assets/videos/terrain_slopes.gif" width="420"/> | <img src="assets/videos/terrain_stairs.gif" width="420"/> |

## Sim-to-Real Deployment

**Rendered Egocentric Views**

<img src="assets/videos/deployment_egocentric_views.gif" width="100%"/>

| Pick-up | Stair-Climbing |
|:---:|:---:|
| <img src="assets/videos/deployment_pickup.gif" width="420"/> | <img src="assets/videos/deployment_stairs.gif" width="420"/> |

## Quick Start

### 1. Get the Docker image and open a GPU shell

Start in the GRAIL repository root on a Linux GPU host with NVIDIA-enabled Docker. The examples require the Git LFS assets and the reconstruction and retargeting submodules. See the [installation guide](docs/source/installation.md) for dependency setup and the available runtime environments.

The two examples already include their input videos. Use the standard image for these examples or for generation with Kling. To generate new videos locally with MiniMax-H3, use `nvgrail/grail:1.2.0-minimax-h3`, which also supports Kling. The [MiniMax-H3 guide](docs/source/gen_2dhoi.md#minimax-h3-provider) shows the GPU requirements and checkpoint mount.

```bash
git submodule update --init --recursive
git lfs pull --include="examples/g1-2dhoi/*/generation/**,assets/robots/unitree_g1/meshes/**"
GRAIL_IMAGE=nvgrail/grail:1.2.0
docker pull "$GRAIL_IMAGE"
mkdir -p "$HOME/grail-runs"
docker run --rm -it --gpus all --shm-size=16g --name grail \
  -v "$PWD:/workspace/grail" -w /workspace/grail \
  -v "$HOME/grail-runs:/workspace/runs" \
  -v grail-model-cache:/root/.cache/grail \
  -v grail-hf-cache:/root/.cache/huggingface \
  "$GRAIL_IMAGE"
```

Inside the container, install GRAIL and its isolated reconstruction workers, then download the weights. The first setup provisions the workers in the mounted checkout and cache.

```bash
bash scripts/setup/install_env_docker.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grail
bash scripts/setup/download_checkpoints.sh --skip-hunyuan3d --skip-sonic --skip-sonic-manip
```

For the robot stage, make the SOMA Retargeter checkout, robot Python and G1 assets available inside the container. See the [retargeting setup](docs/source/installation.md#soma-retargeter-unitree-g1--dex3).

### 2. Reconstruct the human motion

Keep the `grail` container running. Run the reconstruction and retargeting commands from the repository root in a second host terminal where Codex can run `docker exec grail`. Sign in to Codex on the host. If the CLI rejects its default model, add `-m <model>` after `codex exec` with a model available to your account. If sandbox restrictions prevent access to Docker, add `--approve-for-me` on CLI versions that list it in `codex exec --help`. This lets Codex request automatic review for the Docker commands.

```bash
CASE=stairs  # or kettlebell
codex exec "Inside the grail container, use the grail-agentic-hoi-recon workflow on examples/g1-2dhoi/$CASE/generation.
Copy the inputs to /workspace/runs/$CASE/generation, reconstruct from the video, and save the reviewed fit and reconstruction.mp4 (source and fit side by side) under /workspace/runs/$CASE. For kettlebell, also save the reviewed source contact map there as contact_map.json."
```

### 3. Retarget the motion to Unitree G1

```bash
codex exec "Inside the grail container, use the grail-agentic-retargeting workflow on the reviewed fit in /workspace/runs/$CASE.
Use body-only Unitree G1 for stairs and G1 with DEx3 hands for kettlebell. Save the selected native motion as robot.npz, with its original object mesh binding, and save retargeting.mp4 in the same directory."
```

On the host, open `~/grail-runs/<case>/reconstruction.mp4` for the human fit and `~/grail-runs/<case>/retargeting.mp4` for the robot motion.

### 4. Track the retargeted motion

In the GPU container, set up SONIC and download the tracking checkpoints once. Both examples default to 2,048 environments and 5,000 PPO updates, with evaluation every 1,000 updates. Use a fresh output directory for each run.

```bash
cd /workspace/grail
bash scripts/setup/install_env_sonic.sh
conda activate sonic
bash scripts/setup/download_checkpoints.sh --skip-gem-x --skip-mose3 --skip-foundationpose --skip-hunyuan3d
export PYTHONPATH="$PWD/imports/SONIC:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=Yes WANDB_MODE=disabled
```

**Stairs.** Pack the body-only G1 motion with its stair collider and finetune the terrain policy.

```bash
STAIRS=/workspace/runs/stairs
python -m grail.retargeting.terrain --native "$STAIRS/robot.npz" --motion-key stairs --output "$STAIRS/terrain" &&
  bash scripts/tracking/terrain_finetune.sh "$STAIRS/tracking" "$STAIRS/terrain"
```

**Kettlebell.** Create the G1+DEx3 handoff from the motion and reviewed contact map, then finetune ContactFirst.

```bash
KETTLE=/workspace/runs/kettlebell
python -m grail.retargeting.manipulation.handoff --native "$KETTLE/robot.npz" --contact-map "$KETTLE/contact_map.json" --support native --motion-key kettlebell --output "$KETTLE/handoff" &&
  bash scripts/tracking/manipulation_finetune.sh "$KETTLE/tracking" "$KETTLE/handoff"
```

Results are under `~/grail-runs/<case>/tracking/` on the host. See the [tracking guide](docs/source/tracking.md#quick-start-tracking-details) for commands, checkpoints and output paths.

## Documentation

Full documentation can be found at [docs](https://nvlabs.github.io/GRAIL/)
(rendered HTML) and markdown sources are linked below.

### Getting Started
- [Installation](https://nvlabs.github.io/GRAIL/installation.html)
- [Quick Start](https://nvlabs.github.io/GRAIL/quick_start.html)

### Pipeline
- [3D Asset Generation](https://nvlabs.github.io/GRAIL/gen_3d_assets.html)
- [2D HOI Generation](https://nvlabs.github.io/GRAIL/gen_2dhoi.html)
- [4D HOI Reconstruction](https://nvlabs.github.io/GRAIL/recon_4dhoi.html)
- [Retargeting](https://nvlabs.github.io/GRAIL/retargeting.html)
- [Task General Tracking](https://nvlabs.github.io/GRAIL/tracking.html)
- [Data Export](https://nvlabs.github.io/GRAIL/data_export.html)
- [Data Visualization](https://nvlabs.github.io/GRAIL/visualization.html)
- [Web Visualizer](https://nvlabs.github.io/GRAIL/web_visualizer.html)

## TODOs

- [x] Release task-general tracking policy checkpoints
- [x] Provide quick-start demo script
- [x] Release GRAIL manipulation dataset

## Citation

If you find GRAIL useful in your research, please cite:

```bibtex
@inproceedings{grail2026,
  title     = {{GRAIL}: Generating Humanoid Loco-Manipulation from World Models},
  author    = {Tianyi Xie and Haotian Zhang and Jinhyung Park and Zi Wang and
               Bowen Wen and Jiefeng Li and Xueting Li and Qingwei Ben and
               Haoyang Weng and Yufei Ye and David Minor and Tingwu Wang and
               Chenfanfu Jiang and Sanja Fidler and Jan Kautz and Linxi Fan and
               Yuke Zhu and Zhengyi Luo and Umar Iqbal and Ye Yuan},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.05160},
}
```

## License

This project is released under the NVIDIA License; see [LICENSE](LICENSE) for details. The Work and any derivative works may be used only non-commercially, except by NVIDIA Corporation and its affiliates. Third-party components are subject to their own licenses.
