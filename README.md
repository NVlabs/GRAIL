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

- **[2026-11-20]** 🧩 **Agentic reconstruction and retargeting with SOMA body support** — released the agent-guided workflow for interaction-aware HOI reconstruction and retargeting to Unitree G1 on the SOMA body model.
- **[2026-08-25]** 🎬 **MiniMax-H3 video generation** — added an optional local MiniMax-H3/Sol-Engine backend for robot-centric human video generation. See the [video generation guide](https://nvlabs.github.io/GRAIL/gen_2dhoi.html#minimax-h3-provider).
- **[2026-07-15]** 🤖 **Task-general tracking checkpoints** — released policy checkpoints trained on the released GRAIL data. See the [tracking guide](https://nvlabs.github.io/GRAIL/tracking.html) and [dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL).
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

Run either bundled video from human reconstruction through Unitree G1 retargeting and task-general tracking in simulation. Start with the [GPU and storage requirements](https://nvlabs.github.io/GRAIL/requirements.html), [install the Docker environment](https://nvlabs.github.io/GRAIL/installation.html) and [download the checkpoints](https://nvlabs.github.io/GRAIL/downloads.html).

- [Stair climbing](https://nvlabs.github.io/GRAIL/tutorials/stairs.html) with the scene-aware tracker for terrain traversal.
- [Kettlebell lifting](https://nvlabs.github.io/GRAIL/tutorials/kettlebell.html) with the object-aware manipulation policy.

The [GRAIL documentation site](https://nvlabs.github.io/GRAIL/) includes the pipeline guides, output formats and troubleshooting.

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
