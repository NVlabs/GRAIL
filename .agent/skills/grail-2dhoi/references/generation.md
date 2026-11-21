# Launch and resume source-video generation

Run from the repository root in the validated generation runtime. Select an
existing base config and the reviewed object YAML. Inspect actual skip flags and
output paths in `grail.pipelines.gen_2dhoi` before continuing a previous run.

## Render without a video request

```bash
python -u -m grail.pipelines.gen_2dhoi \
    --config configs/gen_2dhoi/<base>.yaml \
    --object_config configs/objects/<dataset>.yaml \
    --results_dir results_collection/<attempt> \
    --num_rand_scenes <count> \
    --skip_step4
```

The selected YAML controls other skipped stages, including scale optimization.
Review those effective settings; a render-only flag does not enable a disabled
simulation or scale stage. Review the newly selected renders before video work.

## Kling videos from the selected renders

```bash
python -u -m grail.pipelines.gen_2dhoi \
    --config configs/gen_2dhoi/<base>.yaml \
    --object_config configs/objects/<dataset>.yaml \
    --results_dir results_collection/<attempt> \
    --num_rand_scenes <count> \
    --video_model_api kling-ai \
    --skip_step1 --skip_step2 --skip_step3
```

This uses the reviewed render and metadata in that exact output namespace.
Add `--skip_done` only after verifying its per-scene/provider behavior and the
matching input bindings. A new random-scene index needs its own render/review;
raising `--num_rand_scenes` is not permission to bypass review of new images.

Keep the combined Kling concurrency at or below 20, and respect any tighter
current limit. Pure API work does not need a GPU simply because rendering did.
MiniMax-H3 is a different deployment path: use [its reference](minimax-h3.md),
provider-specific output directory and GPU allocation, without changing Kling's
flags or requiring H3 checkpoints for the default path.

## Resources and recovery

Check credentials by presence without logging them. Blender has its own Python
runtime; verify actual serialized-input compatibility with the consuming runtime.
Do not apply the historical NumPy downgrade to a shared reconstruction runtime as
a generic repair. Missing categories usually require checking the object YAML's
`dataset`, `objects`, per-category mappings and `scene_config` path.

On failure, preserve logs, provider IDs and partial artifacts. Distinguish a
resource/read failure from a provider rejection or a source-quality failure.
Inspect any outstanding request before deciding whether another submission is
eligible. Resume only authorized unfinished work with matching provenance;
never use wholesale cleanup or deletion of failure markers as a recovery policy.
