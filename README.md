<div align="center">

# GRAIL: Generating Humanoid Loco-Manipulation from World Models

<p><strong>CoRL 2026</strong> <em>(Oral Presentation)</em></p>

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

**Choose a Docker image** and install local extras inside the bind-mounted checkout:

| Image | Use it for |
| --- | --- |
| `docker.io/nvgrail/grail:1.2.0` (`:latest`) | Standard GRAIL with native SOMA 0.3 and Kling; no MiniMax-H3/SGLang runtime |
| `docker.io/nvgrail/grail:1.2.0-minimax-h3` (`:minimax-h3`) | Standard GRAIL plus an isolated MiniMax-H3/SGLang runtime |

Both Linux AMD64 images contain runtime environments only. They do not contain
the GRAIL/Sana source, GRAIL or MiniMax-H3 checkpoints, API keys, or user data.
The repo setup installs the isolated GEM-X, MoGe-3, MoSE3, and Python 3.12 SOMA
Retargeter runtimes and validates native SOMA 0.3 support in the GRAIL env.
Keep credentials outside the image and provide them only at runtime.

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

# inside the container
cd /workspace/grail
bash scripts/setup/install_env_docker.sh   # validates native extensions, downloads Blender
bash scripts/setup/download_checkpoints.sh # GEM-X / MoSE3 / FoundationPose and legacy weights
bash scripts/setup/download_comasset.sh --category cordless_drill # quick-start object
eval "$(conda shell.bash hook)"
conda activate grail
[ -f .env ] && source .env                  # OPENAI_API_KEY, KLING_*, HF_TOKEN
```

The setup script rebuilds GPU/Python-specific native extensions when needed
(`nvdiffrast`, FoundationPose `mycpp`) and installs Blender into the mounted
checkout.

Run any stage end-to-end. Pipeline stages are package entrypoints; invoke them
with `python -m grail.pipelines.*` rather than project-root wrapper scripts.

```bash
# 3D asset generation (procedural terrain or AI-generated objects)
python -m grail.pipelines.gen_terrain --type stairs --num 50 --output_dir data/syn_stairs
conda run -n hunyuan python -m grail.pipelines.gen_3d_assets \
    -i configs/gen_3d/example_objects.yaml -o data/gen_example

# 2D HOI generation (bundled G1 DHGen character + Kling video)
python -m grail.pipelines.gen_2dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --video_model_api kling-ai

# 4D HOI reconstruction (default: native SOMA + hybrid object tracking)
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill --results_dir results

# Native SOMA -> Unitree G1 + DEx3 (the output directory must not already exist)
HOI_DATA=$(find results/generation/4dhoi_recon_soma03_valid/ComAsset/cordless_drill \
    -type f -path '*/hoi_data/hoi_data.pkl' -print -quit)
python -m grail.retargeting.soma "$HOI_DATA" results/retargeted/cordless_drill_g1_dex3 \
    --model-config configs/retargeting/soma_g1.json
```

Full install, dataset, and config notes: see [Documentation](#documentation) below.

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
- [ ] Provide quick-start demo script
- [ ] Release GRAIL manipulation dataset

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
