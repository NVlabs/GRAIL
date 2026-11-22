# Checkpoints and example data

Complete [Installation](installation.md) first. Keep the same checkout and cache for both examples. The two example videos and their scene inputs ship in Git LFS. The model weights are separate downloads. You can leave the full published motion dataset undownloaded.

## Verify the example inputs

**GPU container, a few seconds.** Inspect both videos and count the G1 mesh files.

```bash
find examples/g1-2dhoi -name '*.mp4' -exec ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames:format=filename -of compact {} \;
find assets/robots/unitree_g1/meshes -name '*.STL' | wc -l
```

Expect two 1280 × 720 videos at 24 fps: 241 frames for stairs and 121 for kettlebell. The robot mesh count is 49. Each case under `examples/g1-2dhoi/<case>/generation/` also contains the camera, metric mesh, first-frame pose, depth and masks. If a file exists but FFmpeg or the installer cannot read it, check [Git LFS hydration](troubleshooting.md).

## Reconstruction weights

**GPU container.** Activate the main environment and download the human pose and object-tracking weights. Allow 30–60 minutes for a cold download on a network shared with other users. Transfer speed can vary substantially between models. The first GEM-X download can stay on the same console line for several minutes. Wait for `All requested checkpoints are present.` before starting a fit. If the server returns HTTP 429, keep the cache and follow [rate-limit recovery](troubleshooting.md).

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grail
bash scripts/setup/download_checkpoints.sh --skip-hunyuan3d --skip-sonic --skip-sonic-manip
```

| Component | Used for | Where to look |
|---|---|---|
| GEM-X and SOMA | Human pose and body geometry | `/root/.cache/grail/models/gem-x/`, linked into `imports/GEM-X/` |
| MoSE3 and FoundationPose | Moving-object pose proposals and tracking | `imports/MoSE3/ckpt/` and `imports/FoundationPose/weights/`, including predictor YAMLs |
| MoGe and SAM2 | Depth and segmentation | First inference can download additional weights, including a roughly 5 GB MoGe-3 checkpoint for stairs. Keep the cache mounted between runs |

## Tracking weights

**Before you run this block**, review the [Isaac Sim licences](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/legal.html) and the [Omniverse agreement](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/NVIDIA_Omniverse_License_Agreement.html). The installer sets `OMNI_KIT_ACCEPT_EULA=Yes` and writes the Kit acceptance marker without prompting. Continue only if you accept the terms. If you only want the G1 motion and preview, you can skip this block.

**GPU container.** Set up the [SONIC tracking environment](configuration.md) once and fetch the released policies. On a high-bandwidth host this takes a few minutes with the image installed. The downloader again reports that all requested checkpoints are present.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
bash scripts/setup/install_env_sonic.sh
conda activate sonic
bash scripts/setup/download_checkpoints.sh --skip-gem-x --skip-mose3 --skip-foundationpose --skip-hunyuan3d
```

Under `imports/SONIC/models/`, stairs uses `terrain_stairs/last.pt`. Kettlebell uses `manip_warmstart/last.pt` for the actor and `sonic_manipulation_base/` for its frozen decoder and config. The [published dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) also provides released motion libraries for users who want to inspect existing results.

Continue to the [Quick Start](quick_start.md) and choose a motion.
