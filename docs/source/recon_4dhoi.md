# 4D HOI Reconstruction

Reconstructs metric human and object motion from generated videos. The public
default is native SOMA 0.3 human motion predicted by GEM-X, MoGe-3 depth, and
MoSE3 + FoundationPose hybrid object tracking. The output keeps the existing
GRAIL contact-label optimization and post-processing path.

## Quickstart

```bash
# Dynamic object: manipulation / pickup (default)
python -m grail.pipelines.recon_4dhoi \
    --dataset ComAsset --category cordless_drill --results_dir results

# Static object: terrain / sitting
python -m grail.pipelines.recon_4dhoi \
    --config configs/recon_4dhoi/loco_soma.yaml \
    --dataset syn_stairs --results_dir results
```

The default config is `configs/recon_4dhoi/manip_soma.yaml`. Validated output
is written below `results/generation/4dhoi_recon_soma03_valid/`.

## Pipeline stages

1. **Human pose** — GEM-X predicts native SOMA pose, translation, identity,
   scale, hands, and camera parameters while preserving source FPS and frame
   count. GRAIL applies the fixed public G1 human identity downstream.
2. **Masks and depth** — SAM2 tracks human/object masks; the isolated MoGe-3
   worker predicts depth without changing the main GRAIL PyTorch/Triton stack.
3. **Object pose** — dynamic-object configs generate metric-CAD proposals from
   MoSE3 and select/refine them in a closed FoundationPose loop. Static terrain
   configs use the known rendered object pose.
4. **HOI optimization** — the original `contact_labels` cache, contact loss,
   and OpenAI vision contact-joint detection remain unchanged. Set
   `OPENAI_API_KEY`; the default vision model is `gpt-4o`.
5. **Filtering** — removes invalid camera, mask, duration, and task-motion cases.
6. **Post-processing and visualization** — writes canonical human/object motion,
   contact points, videos, and the ScenePic viewer.

This release does not include autonomous contact authoring or review workflows.

## Configs

```{list-table}
:widths: 32 18 50
:header-rows: 1

* - Config
  - Human model
  - Use
* - `configs/recon_4dhoi/manip_soma.yaml`
  - SOMA 0.3 (default)
  - Dynamic manipulation/pickup with hybrid object tracking.
* - `configs/recon_4dhoi/loco_soma.yaml`
  - SOMA 0.3
  - Static terrain and sitting.
* - `configs/recon_4dhoi/manip_smplx.yaml`
  - G1-proportioned SMPL-X
  - Explicit legacy dynamic-object route.
* - `configs/recon_4dhoi/loco_smplx.yaml`
  - G1-proportioned SMPL-X
  - Explicit legacy static-object route.
```

Both SOMA configs use
`data/body_profiles/unitree_g1/soma/soma03_custom_smplx_identity.npz`.
The DHGen character is a 2D rendering asset and is not substituted for the
native SOMA body model.

## Runtime selection

Default paths are:

```text
imports/GEM-X/.venv/bin/python
.venvs/moge3/bin/python
.venvs/mose3/bin/python
imports/MoSE3/ckpt
```

Override isolated workers with `GRAIL_MOGE_PYTHON`, `GRAIL_MOSE3_PYTHON`,
`GRAIL_MOSE3_ROOT`, `GRAIL_MOSE3_CHECKPOINT`, or the corresponding YAML keys.
See [installation](installation.md) for pinned setup commands.

## Resume individual stages

```bash
# Human estimation only
python -m grail.pipelines.recon_4dhoi --skip_step2 --skip_step3 \
    --skip_step4 --skip_step5 --skip_step6

# Reuse pose and rerun masks/depth/object tracking
python -m grail.pipelines.recon_4dhoi --skip_step1 \
    --skip_step4 --skip_step5 --skip_step6

# Reuse prediction/tracking and rerun optimization onward
python -m grail.pipelines.recon_4dhoi --skip_step1 --skip_step2 --skip_step3
```

Use a new `results_dir` when changing body representation or predictor binding;
SOMA and SMPL-X caches are intentionally not interchangeable.
