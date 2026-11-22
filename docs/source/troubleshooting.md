# Troubleshooting

1. **The model installer reports “Invalid binary mesh” or a video will not open.** A small text file beginning with `version https://git-lfs.github.com/spec/v1` is an LFS pointer. Install Git LFS on the host, hydrate the checkout with the [installation commands](installation.md#get-the-code-and-the-image), then [verify the input files](downloads.md#verify-the-example-inputs). Initializing submodules alone does not hydrate the parent repository.
2. **Setup fails while creating the CUDA rasterizer.** Use `nvgrail/grail:1.2.0` for the bundled videos. The `1.2.0-minimax-h3` variant is for local MiniMax-H3 video generation. Its setup rasterizer failed on an RTX 6000 Ada in testing. The bundled videos need neither a video-generation service nor that image.
3. **Training works but kettlebell video evaluation exits during RTX startup.** On an RTX 6000 Ada with driver 595.99.02, Isaac Sim also crashed creating an empty rendering stage outside Docker. Video evaluation worked on an L40S with driver 580.95.05. Use the metrics-only setting in the [kettlebell tutorial](tutorials/kettlebell.md) to evaluate without rendering. These observations do not establish a compatible driver range.
4. **The worker reports a missing package while opening the SOMA fit.** Use the Python interpreter from the `grail` environment for source export and the pinned SOMA retargeter's own virtual environment for robot conversion. The [Docker installer](installation.md) configures both. GRAIL's released examples use SOMA03. Running the standalone legacy SMPL-X route requires your own compatible legacy GMR installation.
5. **Codex reports an unavailable model or cannot access Docker.** Run Codex on the host from the checkout, signed in to an account with access to the selected model. Choose a model available to your account. Confirm the named container is still running and that your host user can access Docker. CLI versions offering `--approve-for-me` can request automatic review of the Docker commands.
6. **The tracking launcher reports an existing output directory, or the handoff rejects a mesh.** Use a fresh output directory for training. Keep the source mesh bytes and its original face order. For stairs, the collider must match the bound terrain. For a handheld object, the contact map must use the same mesh and source clock as the robot motion.

7. **The checkpoint downloader exits with HTTP 429 (Too Many Requests).** The public weight server has rate-limited the download. The same limit can delay the depth worker on its first inference. Wait for its indicated retry interval, or a few minutes if your client only shows the status. Run the same download block again with the same mounted cache. The GEM-X installer verifies completed files by size and SHA-256 and continues with missing ones.

8. **MoSE3 reports `torch.OutOfMemoryError` during kettlebell object tracking.** Check free GPU memory while tracking is running. On the shared 48 GB card one allocation requested 770 MiB when only 441 MiB was free. The agent reduced the MoSE3 window from 120 to 40 target frames and reran that worker on the same case. It recovered a pose for every frame. Retain the case directory so that the completed source masks, human pose and depth can be reused. Verify the object trajectory covers every source frame before fitting.

(retry-manipulation)=
## Retry tracking after a simulator memory error

If PhysX ran out of GPU memory before creating the first checkpoint, a later missing-`last.pt` error is a consequence. Once GPU memory is available, reuse the terrain library or manipulation handoff from the previous attempt.

**Stairs.** In the tutorial’s configured `sonic` shell, with `CASE` set to your stairs run, retry the short schedule with `NUM_ENVS=64 ITERATIONS=4 EVAL_EVERY=4 bash scripts/tracking/terrain_finetune.sh "$CASE/tracking-retry" "$CASE/terrain"`. The first argument must be unused. The second points to the existing terrain library.

**Kettlebell.** Reuse the existing successful handoff. Run only the launcher, with a fresh output directory. Run the following in the container from `/workspace/grail`. Set `CASE` to the directory containing your successful handoff.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonic
export PYTHONPATH="$PWD/imports/SONIC:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=Yes WANDB_MODE=disabled
CASE=/workspace/runs/kettlebell
EVAL_NO_VIDEO=1 NUM_ENVS=64 ITERATIONS=4 EVAL_EVERY=4 bash scripts/tracking/manipulation_finetune.sh "$CASE/tracking-retry" "$CASE/handoff"
```

The launcher packs the existing handoff into the new run. Open `tracking-retry/eval_final/kettlebell/summary.json` under the case directory. If `tracking-retry/` already exists from another attempt, choose another unused output name.
