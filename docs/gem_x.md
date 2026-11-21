# GEM-X native SOMA prediction

GEM-X replaces the paired GEP/GEM predictor with the official released Body
Motion model, pinned as an unmodified public Git submodule. The vendored
GEP/GEM source and its installers have been removed. Existing predictions,
identity manifests and archived reconstruction runs retain their original
provenance; use their frozen source snapshots to reproduce historical runs.

## Pinned components

| Component | Version or revision |
| --- | --- |
| [GEM-X source](https://github.com/NVlabs/GEM-X) in `imports/GEM-X` | `20555befe2fbbe5b2015e1fff7ff214dc18ea9d6` |
| [GEM-X weights](https://huggingface.co/nvidia/GEM-X) | `4fce057963671fd95cd4154297944397e7b5a9a0` |
| [SOMA-X assets](https://huggingface.co/nvidia/SOMA-X/tree/v0.3.3) | `d281db2d01553f7230c56e764cec00fe27ef23f7` (`v0.3.3`) |
| Native SOMA runtime | `py-soma-x==0.3.3` |
| Predictor runtime | Python 3.10+, PyTorch 2.10.0, Transformers 5.3.0 |

Body Pose uses `body_s240000.ckpt`; temporal Motion uses
`motion_native77_e774_s292175.ckpt` and the matched `motion_stats_raw.npz`.
The observation network is `vitpose.pth` with the **native SOMA77** layout.
The old Sapiens30 observations and B2/A620 Motion checkpoints are incompatible
with this released native77 model. The old Body Pose/GEP s240000 and new Body
Pose checkpoint have 672 bit-identical tensors; the checkpoint containers differ
because the public export strips training metadata.

## Clean standalone installation

Install system `git`, Python with `venv`/pip (or `uv`), and FFmpeg first. When
the system Python omits `ensurepip`, the helper uses `uv venv --seed` if available.
For optional
headless rendering on Linux x86-64, install Mesa EGL (`libegl-mesa0` and
`libgl1-mesa-dri`). The setup helper creates a separate environment and obtains
its dependencies directly; it does not inherit the previous GEP/GEM environment,
site packages, `PYTHONPATH`, or bundle sources. It requires the exact source
commit above and rejects tracked source edits. It builds an installed wheel
from that pinned Git tree, excluding untracked checkout files; this is not an
editable install.

From the repository root:

```bash
git submodule update --init imports/GEM-X
python3.12 scripts/setup/setup_gem_x.py \
  --env /absolute/external/envs/gem-x \
  --assets /absolute/external/models/gem-x \
  --grail-python /absolute/path/to/grail/bin/python \
  --link-submodule --render
```

The explicit `--grail-python` option installs SOMA-X 0.3.3 into the separate
GRAIL fitting interpreter as well. GEM-X prediction runs in its new environment;
downstream native mesh/fitting runs in GRAIL's chosen interpreter and also needs
the validated SOMA runtime. Omit this option only if that interpreter already
has the correct version. This helper does not install all of GRAIL's dependencies.

The default PyTorch wheel index is CUDA 12.8. Select
`--torch-index https://download.pytorch.org/whl/cpu` for a CPU-only environment.
CUDA inference requires a compatible driver. The detector defaults to CPU
ONNXRuntime independently of the Body Pose/Motion device. Rendering is optional;
omit `--render` for prediction alone.

Weights and geometry remain outside Git (about 7.7 GB). Every selected file is
checked against a pinned SHA256 and byte count, including the YOLOX ONNX extracted
from its official OpenMMLab archive. Valid existing files are reused; mismatches
fail without overwriting them. The helper preserves existing environments unless
they carry its exact setup marker, and refuses to change a differently configured
one. `--link-submodule` creates only missing, matching links:

```text
imports/GEM-X/.venv       -> external environment
imports/GEM-X/weights     -> external models/weights
imports/GEM-X/assets/soma -> external models/soma
```

Exact symlink exclusions are appended to the submodule's local `info/exclude`;
upstream tracked files are unchanged. Existing files, directories, or different
links are preserved. Downloads can run separately with `--download-only`.
Use `--verify-only` with `--assets` for read-only asset verification.
The helper writes `gem_x_assets.json` and an absolute-path native77 configuration
under the external model directory, plus `gem_x_runtime.json` in the environment.
This metadata is for local reproducibility; the maintained YAMLs use the
submodule-relative layout. GRAIL's worker explicitly imports that selected
submodule checkout and binds its source hashes; the standalone installed
`gem-infer` CLI uses the pinned wheel.

## Inference and source timing

The maintained SOMA configuration selects `human_pose_backend: gem_x`, the
submodule's `docs/configs/body_video_native77.yaml`, and its `.venv/bin/python`.
GRAIL's declared `static_cam` choice overrides the upstream camera default.
Run the maintained standalone predictor with source-aligned SAM2 masks and
their completion receipt:

```bash
/absolute/path/to/grail/bin/python -m grail.cli.estimate_soma_motion \
  --video /absolute/path/to/source.mp4 \
  --config configs/recon_4dhoi/manip_soma.yaml --backend gem_x \
  --human-crop-masks /absolute/path/to/sam2/masks.npz \
  --human-crop-receipt /absolute/path/to/sam2/completion.json \
  --cache-dir /absolute/path/to/new_gem_x_cache \
  --output /absolute/path/to/new_motion.npz
```

The previous `scripts/standalone/estimate_soma_motion.py` entrypoint remains a
compatibility shim. The implementation is included in the installable `grail` package.

The masks and receipt must bind the same source video and frame clock; the
adapter verifies those bindings before using mask-assisted crops. This command
writes raw predicted identity and a JSON provenance report. Fixed robot identity
is applied later in reconstruction. An explicit `--no-human-crop-repair` is
available for detector-only controls.

To exercise the upstream CLI directly with a known static camera, generate
`body_video_native77_static.json` using setup's `--camera-mode static`, then run:

```bash
imports/GEM-X/.venv/bin/gem-infer motion-video --task body \
  --input /absolute/path/to/video.mp4 \
  --config /absolute/external/models/gem-x/body_video_native77_static.json \
  --device cuda --output /absolute/path/to/new_motion.pt
```

Upstream `motion-video` normalizes a complete clip of at most 60 seconds to
30 Hz. Its frame IDs refer to this normalized grid. GRAIL must restore the exact
decoded source count/FPS/timestamps with explicit rotation-aware interpolation,
including the tail. A 121-frame 24 Hz clip becomes 151 native frames at 30 Hz
before returning to 121 source frames. Preserve both clocks in provenance.

Native77 observations are joint centers and include finger joints. They are not
Sapiens surface landmarks or COCO17 points with different labels. Confidence
scores are heatmap scores, sometimes above 1, not calibrated probabilities.
Do not silently copy old confidence calibration or treat projected mesh points
as observed 2D evidence. Keep all estimator/observation IDs, model hashes,
source hashes, crop policy and timing in the cache identity. Old prediction or
condition caches must not satisfy a fresh GEM-X request.

## Native geometry and fixed identities

Raw Motion retains its predicted identity128, positive local56 scales, and
**global scale exactly 1**. Camera/global rotations and translations are native
SOMA parameters, not a scaled display mesh. Downstream GRAIL may apply a separate
hash-bound fixed identity using its existing validated manifest contract.
Some fixed identities have a non-unit global scale; that does not permit a
non-unit global scale in the raw predictor output.

Native geometry defaults to `soma_runtime: gem_x` and the pinned SOMA assets.
Before migrating fixed identities, verify joint ordering, root placement, mesh
topology, correctives and native posed geometry against the selected parent.
The public neutral NPZ differs from the archived bundle even though the rig,
procedural metadata and corrective weights match. Source-bound historical
identity manifests must retain their original bindings; verified compatibility
is a separate record, not permission to rewrite history.

Optional upstream Contact/IK is **disabled** for the raw predictor workflow.
It adjusts shared body pose and world translation from learned contact logits;
it is not GRAIL's video-authored object-contact refinement. If explicitly tested,
save processed results separately and preserve `raw_predictions` and the method
receipt. Do not silently activate it when replacing the estimator.
