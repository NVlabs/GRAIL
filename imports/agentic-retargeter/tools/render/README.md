# Cached, parallel sequence previews

Use this tool to render an exported native sequence from a bound Blender
template. It runs foreground Blender workers, splits frames across workers and
reuses exact unchanged frame/view images. It does not run a solver, modify the
input `.blend`/NPZ, save an edited scene, or accept a motion.

```bash
python tools/render/run.py --config render.json --output runs/render_001 \
  --cache runs/render_cache --workers 4
```

The output must be a new directory. Reuse the cache directory in later rounds.
Paths in the JSON are relative to that JSON; output/cache paths are relative to
the command's working directory.

```json
{
  "schema_version": 1,
  "blender": "/path/to/blender-4.3.2",
  "blend": "template.blend",
  "native": "exported_native.npz",
  "frames": [0, 2, 4, 6, 8],
  "resolution": [480, 480],
  "require_hardware": false,
  "views": [
    {"name": "full", "camera": "SourceCamera"},
    {
      "name": "hand",
      "follow_wrist": "right_wrist_yaw_link",
      "direction": [1, -2, 0.8],
      "ortho_scale": 0.55
    }
  ]
}
```

Use original zero-based **native integer frame indices**, with no duplicates.
This path renders exported samples; it does not approximate Blender subframes.
At 60 Hz, frames `0, 2, 4, ...` can be encoded as a 30 fps preview. Label sparse
or lower-resolution interim previews as drafts; retain source timing and render
the final full-duration preview separately. Do not reduce collision-audit sample
density along with preview density.

The template must support the existing `native_fk` preview contract and contain
the contact kit's native/model/plan data. The visual binding comes from the
template's `FreshWorkerSpec.json`, or an explicit `spec` object in this config
(same `preview` and `cameras` fields as `blender-edit`). Templates preserve their
Workbench material/studio lighting and world settings; output uses RGB PNG with
compositing/sequencer disabled. Static cameras are taken from the template. A
mounted camera points at the midpoint between the named wrist and object, with
the supplied world-space direction and orthographic scale.

The optional `native` NPZ overrides qpos only. Its shape, object trajectory,
scale, fps and coordinate-convention fields must match the template. If omitted,
the tool evaluates the template's current native curves at integer frames.
Export new Blender edits first, then supply that export with the same template
to reuse images outside the changed samples. A changed `.blend` invalidates the
whole cache conservatively; this includes changes to geometry, materials,
cameras, contact overlays or the table. Use a new template for those changes.

Cache keys include exact qpos bytes and frame number, template and external
asset hashes, package implementation, Blender executable/build, actual graphics
renderer/driver, resolution, binding and view definition. The full candidate NPZ
hash is recorded as provenance; it is deliberately not the per-frame cache key.
Checksums reject missing or damaged cached images. Workers validate body/object
FK after rendering; cache receipts bind those checks to the same inputs. Input
changes during execution and failed workers prevent a completed render report.
Stateful simulations, particles, geometry nodes and object drivers are rejected.
Use immutable inputs and pack external assets that cannot be represented as
ordinary files; this is not a general-purpose arbitrary Blender scene cache.

`images/` contains the combined frame set. `result.json` records total wall time,
actual renderer, reused/rendered counts, native fps, per-image checksums and FK
error. Per-worker requests/logs remain under `shard_*/`. Timings overlap across
workers: report wall time, not the sum of worker durations. Each worker performs
a tiny unexported warmup render to identify the graphics backend before cache
lookup; a warm cache therefore still has Blender startup/warmup overhead.

`require_hardware: true` fails if Blender falls back to llvmpipe/softpipe/software
rendering. `nvidia-smi` alone is not evidence that Workbench used the GPU; inspect
`shards[].device` from the actual rendering process. The tool does not install or
change host drivers. Start with one worker on a GPU and measure before increasing
concurrency. Four workers were tested for software rendering; more workers are
not automatically faster. `worker_timeout` optionally bounds each child (default
1800 seconds).

Reuse unchanged raw/reference renders, cache unchanged native samples during
local edits, and render short draft windows during iteration. These execution
optimizations do not replace contact, full-mesh, source or temporal review, nor
the editable scene's normal save/export/reopen verification.

For the normal GPU path use `require_hardware: true` and one worker. On NVIDIA
containers the launcher registers an available `libEGL_nvidia` library using a
worker-local GLVND JSON when no explicit EGL vendor selection was supplied.
This handles images that inject NVIDIA libraries but retain only Mesa's vendor
JSON. It does not install drivers or change system configuration. Explicit
`__EGL_VENDOR_LIBRARY_FILENAMES` is preserved. Missing/broken hardware still fails
the actual render guard; software fallback must never be reported as GPU success.

Execute the fixed batch directly on the GPU host. Persist
image caches outside disposable attempts. Render changed windows while iterating,
then the complete duration once after its required gates.
