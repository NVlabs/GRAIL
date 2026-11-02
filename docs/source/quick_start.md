# Quick Start

A 5-minute smoke run that exercises every stage of the pipeline on a single
example asset (`cordless_drill`). Assumes you've completed
[`installation`](installation.md) and have the docker image (or the
`grail` / `hunyuan` conda envs) available.

Pipeline stages are launched as package modules, for example
`python -m grail.pipelines.gen_2dhoi` and
`python -m grail.pipelines.recon_4dhoi`. Do not use project-root wrapper
scripts for pipeline stages.

## 1. Source environment keys

```bash
source .env   # OPENAI_API_KEY, KLING_ACCESS_KEY, KLING_SECRET_KEY, HF_TOKEN
```

`OPENAI_API_KEY` is used by the OpenAI API for prompt refinement and
contact/scale helpers. Chat and vision helpers default to `gpt-4o`.

## 2. Generate a 3D object (procedural — no GPU needed)

```bash
python -m grail.pipelines.gen_terrain --type stairs --num 5 --output_dir data/syn_stairs
```

Outputs `data/syn_stairs/stairs_{000..004}/model.obj` plus `texture.jpg`.

## 3. Generate a 2D HOI video for one object

```bash
python -m grail.pipelines.gen_2dhoi \
    --dataset ComAsset --category cordless_drill \
    --results_dir results --video_model_api kling-ai
```

The pipeline runs four stages: physics simulation, optional scale
optimization, multi-view Blender rendering, and Kling-AI video
generation. Total wall time: ~6–10 min per object on an L40S.

## 4. Reconstruct 4D motion from the video

```bash
python -m grail.pipelines.recon_4dhoi \
    --dataset ComAsset --category cordless_drill --results_dir results
```

Six stages: GEM-X native SOMA pose, SAM2 + MoGe-3 preprocess, MoSE3 +
FoundationPose hybrid object tracking, original contact-label HOI optimization,
filter, and visualize. Runs in
~30–40 min per video on an L40S.

## 5. Inspect outputs

```bash
ls results/generation/4dhoi_recon_soma03_valid/ComAsset/cordless_drill/*/result_vis/
```

You'll see `recon_result.mp4`, `recon_comparison.mp4`,
`recon_result_top_view.mp4`, and an HTML viewer. The validated motion data
(native SOMA params + object 6-DOF poses) lives in `hoi_data/hoi_data.pkl`.

## 6. Retarget native SOMA motion to Unitree G1 + DEx3

The validated `hoi_data.pkl` is the authoritative SOMA reconstruction input.
Retarget it with the pinned public SOMA Retargeter:

```bash
HOI_DATA=$(find results/generation/4dhoi_recon_soma03_valid/ComAsset/cordless_drill \
    -type f -path '*/hoi_data/hoi_data.pkl' -print -quit)

python -m grail.retargeting.soma \
    "$HOI_DATA" results/retargeted/cordless_drill_g1_dex3 \
    --model-config configs/retargeting/soma_g1.json
```

The command emits an audited SOMA BVH, a source-bound G1+DEx3 target, a
50-column CSV with all 43 robot joints (including 14 finger joints), and JSON
receipts. The result remains `RETARGETED_UNREVIEWED` until scene replay and
contact review. See [Retargeting](retargeting.md#replay-robot-object-and-table)
for the robot + object + table replay command.

## Next steps

| You want to … | Doc |
|---|---|
| Generate Hunyuan3D textured assets | [`gen_3d_assets`](gen_3d_assets.md) |
| Render and re-render scenes with new characters | [`gen_2dhoi`](gen_2dhoi.md) |
| Tune the 4D reconstruction optimizer | [`recon_4dhoi`](recon_4dhoi.md) |
| Retarget to Unitree G1 + DEx3 | [`retargeting`](retargeting.md) |
| Train a SONIC policy | [`tracking`](tracking.md) |
| Browse a motion library in the browser | [`web_visualizer`](web_visualizer.md) |
