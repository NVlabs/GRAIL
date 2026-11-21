# MiniMax-H3 video-provider recipe

Read this reference only when the skill-level option is `--video-provider minimax-h3`. Keep the default Kling path unchanged when the option is absent or equals `kling`.

- [Provider contract](#provider-contract)
- [Canonical flags and command](#canonical-provider-flags)
- [Downstream 4D reconstruction](#downstream-4d-reconstruction)

## Provider contract

- Treat `--video-provider` as a skill selector, not a Python CLI flag.
- Map it to `--video_model_api minimax-h3` in `grail.pipelines.gen_2dhoi`.
- Use `generation/videos_minimax_h3` for MiniMax outputs. Pass the same directory to downstream `recon_4dhoi` with `--video_dir generation/videos_minimax_h3`; never mix these videos with `generation/videos_kling` in one run.
- Use a combined GRAIL + MiniMax-H3 image. It must contain the GRAIL conda runtime and the isolated Sol Engine/SGLang runtime. Supply the model and runtime separately.
- Keep the portable launcher, service, and runtime bridge in `grail/adapters/minimax_h3_service.sh`, `minimax_h3_service.py`, and `minimax_h3_runtime.py`. Treat `imports/Sana` as a pinned, pristine Sol Engine source/model dependency on `PYTHONPATH`; do not add or patch GRAIL service files there.
- Require an initialized, unmodified `imports/Sana` submodule and a complete local FL2VA checkpoint. Set its container-visible path with `MINIMAX_H3_CHECKPOINT`.
- Use a 5- or 10-second duration and a single condition image; this backend does not accept an explicit tail image. GRAIL normalizes the result to the condition-image dimensions at 24 fps.
- Keep cache, dense/quality, checkpoint, and prompt experiments in distinct output directories. Every new MP4 has a `.mp4.json` provenance sidecar; `--skip_done` regenerates a valid video when its recorded mode, launcher, seed, duration, or condition image conflicts with the current request. It retains legacy MP4s without sidecars with a warning.
- Retry transient transport/service failures only. Configuration, checkpoint, request-contract, permission, OOM, and service-mode failures are permanent and must stop on the first attempt even when `video_max_retries` is larger.

## Canonical provider flags

Append this fragment only for MiniMax-H3:

```bash
--video_output_dir generation/videos_minimax_h3 \
--video_model_api minimax-h3 \
--minimax_h3_mode quality \
--minimax_h3_autostart \
--minimax_h3_launcher_mode single-gb-parity-vae-resident
```

Do not append any of these flags for Kling.

## Canonical Python command

Substitute the config, object config, results directory, scene count, and stage selection:

```bash
/root/miniconda3/envs/grail/bin/python -u -m grail.pipelines.gen_2dhoi \
    --config configs/gen_2dhoi/<base>.yaml \
    --object_config configs/objects/<dataset>.yaml \
    --results_dir results_collection/<results_subdir> \
    --num_rand_scenes <num_rand_scenes> \
    --video_output_dir generation/videos_minimax_h3 \
    --video_model_api minimax-h3 \
    --minimax_h3_mode quality \
    --minimax_h3_autostart \
    --minimax_h3_launcher_mode single-gb-parity-vae-resident \
    <stage-selection-flags> \
    --skip_done
```

- For one-process 2D-HOI Steps 1–4, omit `<stage-selection-flags>`. The H3 service starts lazily when that same GRAIL process reaches Step 4.
- When Steps 1–3 already produced the condition renders, replace it with `--skip_step1 --skip_step2 --skip_step3`.

This is the current validated one-GPU default:

- one A100 per pipeline worker;
- dense BF16 denoising without FirstBlockCache (`quality` request mode);
- resident video VAE;
- layerwise-offloaded DiT and text encoder;
- `diffusers_shared` RNG for condition, video, and audio noise;
- lazy service startup on the first real video request, reuse for later requests in that worker, and cleanup when `gen_2dhoi` exits.

Use `single-cache-fast-vae-resident` with request mode `fast` only as an
explicit throughput/quality experiment.
When the launcher flag is omitted, autostart derives it from the request mode:
the default `quality` request selects no-cache resident-VAE, while an explicit
`fast` request selects the cache profile for backward compatibility.

Do not start a separate service job. `--minimax_h3_autostart` owns the resident child process in the same local process environment. With `--skip_done`, no service is started if the worker has no video left to generate.

For a standalone service, run `bash grail/adapters/minimax_h3_service.sh <mode> <port>`. The shell launcher invokes `python -m grail.adapters.minimax_h3_service`; `grail.adapters.minimax_h3_runtime` registers the selected pristine Sana model implementation with SGLang.

## Downstream 4D reconstruction

Add the provider video directory to every recon command:

```bash
python -u -m grail.pipelines.recon_4dhoi \
    ... \
    --video_dir generation/videos_minimax_h3 \
    --skip_done
```

Keep Kling's existing command and default `generation/videos_kling` directory untouched. Provider selection changes video generation and the recon input directory only; render, camera, depth, FoundationPose, optimization, and filtering logic remain the same.
