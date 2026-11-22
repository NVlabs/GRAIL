# MiniMax-H3 Video Service

<span id="generate-video-with-a-minimax-h3-service"></span>

The public generator can use the optional H3 image and an existing FL2VA checkpoint. The commands below describe the service interface. The released example validation used existing videos and did not start a new generation service.

## Runtime boundary

`grail.pipelines.gen_2dhoi` stays in the normal GRAIL Python environment. The
Sol-Engine/SGLang service can use a separate compatible interpreter selected
with `MINIMAX_H3_PYTHON` (default: `python3`). On the first real MiniMax
request, `grail.adapters.minimax_h3_process` starts
`grail/adapters/minimax_h3_service.sh`, waits for `/health`, and reuses the
service for subsequent Step-4 requests. Kling never starts or imports this
runtime.

The ownership boundary is:

- `grail/adapters/minimax_h3.py`: provider client and output normalization.
- `grail/adapters/minimax_h3_process.py`: lazy ownership, health, reuse, and cleanup.
- `grail/adapters/minimax_h3_service.sh`: scheduler-neutral launcher.
- `grail/adapters/minimax_h3_service.py`: resident HTTP generation service.
- `grail/adapters/minimax_h3_runtime.py`: SGLang registration bridge.
- `imports/Sana`: pinned source/model implementation only; keep it pristine.

## Setup and autostart

Initialize the public Sana submodule, point GRAIL at an FL2VA checkpoint, and
select the provider explicitly:

```bash
git submodule update --init imports/Sana
export MINIMAX_H3_CHECKPOINT=/absolute/path/to/MiniMax-H3/FL2VA

python -m grail.pipelines.gen_2dhoi \
    --config configs/gen_2dhoi/terrain_stairs.yaml \
    --category stairs_001 \
    --results_dir results \
    --video_output_dir generation/videos_minimax_h3 \
    --video_model_api minimax-h3 \
    --minimax_h3_mode quality \
    --minimax_h3_autostart \
    --minimax_h3_launcher_mode single-gb-parity-vae-resident \
    --skip_done
```

The checkpoint must contain `model_index.json` and the FL2VA component
directories. The launcher adds the pinned Sana checkout to `PYTHONPATH` and
creates a zero-copy model view with symlinks; it does not duplicate weights or
modify `imports/Sana`.

Autostart is lazy: cached outputs, disabled Step 4, invalid requests, and prompt
failures do not load the model. It reuses the service for the rest of the
process and stops it only when GRAIL owns it. The single-A100 quality profile
uses dense BF16 denoising without FirstBlockCache, a resident VAE, and the
`diffusers_shared` RNG contract.

Every completed `video.mp4` receives an atomic `video.mp4.json` provenance
sidecar with the condition-image digest, prompt, seed, duration, request and
launcher modes, normalized output shape, inference metrics, runtime profile,
and checkpoint identity. With `--skip_done`, a result is reused only when its
request metadata still matches. Legacy videos without sidecars remain
resumable with a warning.

Only transient transport and service errors consume `video_max_retries`.
Invalid requests, missing checkpoints, service-mode mismatches, permission
errors, CUDA OOM, and other deterministic failures stop after the first
attempt.

`single-cache-fast-vae-resident` with request mode `fast` remains an explicit
throughput/quality experiment; it is not the default. If
`--minimax_h3_launcher_mode` is omitted, `quality` selects the no-cache
resident-VAE profile, while an explicit `fast` request selects the cache
profile.

## Standalone or remote service

Inside a compatible H3/SGLang runtime, launch the same GRAIL-owned service used
by autostart:

```bash
export MINIMAX_H3_CHECKPOINT=/absolute/path/to/MiniMax-H3/FL2VA
bash grail/adapters/minimax_h3_service.sh \
    single-gb-parity-vae-resident 30020
```

The launcher starts the service in the selected runtime. Without an explicit service root it creates a unique temporary
directory and prints the descriptor path. Useful overrides are:

- `MINIMAX_H3_SERVICE_ROOT`: descriptor, logs, cache, staging, and outputs.
- `MINIMAX_H3_SERVICE_FILE`: descriptor path.
- `MINIMAX_H3_PYTHON`: H3 runtime interpreter; defaults to `python3`.
- `MINIMAX_H3_SOL_ENGINE_ROOT`: Sana checkout; defaults to `imports/Sana`.
- `MINIMAX_H3_HARDWARE`: `A100` (default) or the four-GPU `H100` runtime.
- `MINIMAX_H3_CHECKPOINT_ID`: human-managed checkpoint label for sidecars.
- `MINIMAX_H3_BIND_HOST` / `MINIMAX_H3_ADVERTISE_HOST`: service exposure;
  both default to `127.0.0.1`.

To use a separately managed service, omit `--minimax_h3_autostart` and pass
`--minimax_h3_service_file /path/to/service.json` (or export
`MINIMAX_H3_SERVICE_FILE`). GRAIL never stops a borrowed service. Use
`0.0.0.0` only behind an authenticated outer environment because the service
does not provide authentication. For H100, allocate four GPUs and choose
launcher mode `dense`, `quality`, or `fast`; the default
`single-gb-parity-vae-resident` mode is A100-only.

## Container and scheduler use

Use the MiniMax-enabled runtime image; the standard `:1.2.0`/`:latest` image
intentionally omits SGLang:

```bash
export GRAIL_IMAGE=docker.io/nvgrail/grail:1.2.0-minimax-h3
export MINIMAX_H3_CHECKPOINT=/absolute/path/to/MiniMax-H3/FL2VA

docker pull "$GRAIL_IMAGE"
docker run --rm --interactive --tty \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --volume "$PWD:/workspace/grail" \
    --volume "$MINIMAX_H3_CHECKPOINT:/models/MiniMax-H3/FL2VA:ro" \
    --env MINIMAX_H3_CHECKPOINT=/models/MiniMax-H3/FL2VA \
    "$GRAIL_IMAGE"
```

Use the versioned release tag for a repeatable setup. The image
sets `MINIMAX_H3_PYTHON=/usr/bin/python3` for its isolated CUDA 13 Sol-Engine
environment, but contains neither GRAIL/Sana source nor MiniMax-H3 weights.
Initialize `imports/Sana` in the mounted checkout, then run the same
`grail.pipelines.gen_2dhoi` command above.

A scheduler only allocates resources and provides mounts and environment
variables; no MiniMax-specific scheduler integration is required. The service
binds to localhost by default.
