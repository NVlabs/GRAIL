# Optional local GraspGenX proposals

Use this tool when a new object or poor initial hand pose needs alternative
grasp hypotheses. It is **off in the normal workflow**: only an explicit
`graspgenx run` starts the local model. Loading a sequence, converting SMPL-X or
SOMA, editing in Blender, and local-window repair do not invoke it.

The current adapter supports the native **G1 Dex3** hand on either side. It
builds the hand description from the supplied original model, samples the
original object in physical metres, and generates object-relative grasp frames.
The current native pose supplies frame alignment and displacement diagnostics;
it is not a human observation or a fixed contact target.

## Commands

```bash
python tools/graspgen/run.py plan \
  --config graspgenx.json --output runs/grasp_proposals_001
python tools/graspgen/run.py preflight \
  --config graspgenx.json --output runs/grasp_proposals_001
python tools/graspgen/run.py run \
  --config graspgenx.json --output runs/grasp_proposals_001
```

An installed wheel exposes the same commands through
`agentic-retargeter graspgenx ...` or
`python -m agentic_retargeter graspgenx ...`.

`plan` reads configuration and binds files without importing the model.
`preflight` checks the native model, hand description, pinned local assets,
imports and CUDA availability without generating grasps. `run` performs these
required validations itself; a separate preflight is useful during setup but
is not required before every run. The worker runs in the foreground; optional
`--timeout SECONDS` bounds that external process. Failed runs retain diagnostics
and never become accepted candidates. Each run requires a new output directory.

Configuration paths are relative to the JSON file, except absolute paths:

```json
{
  "label": "Right-hand initial grasp alternatives",
  "native": "/data/motion/native.npz",
  "model_xml": "/data/models/g1_mocap_29dof_with_hands.xml",
  "object_mesh": "/data/object_original.npz",
  "hand_profile": "g1_dex3",
  "side": "right",
  "frame": 32,
  "point_count": 4096,
  "num_grasps": 2000,
  "batch_size": 200,
  "seed": 20260929,
  "preview": true,
  "runtime": {
    "python": "/data/envs/graspgenx/bin/python",
    "repo": "/data/deps/GraspGenX",
    "checkpoint_dir": "/data/checkpoints/GraspGenXModel",
    "device": "cuda:0"
  }
}
```

`frame` is a zero-based native frame; confirm its source-video correspondence.
The native NPZ must bind `qpos`, `dof_names`, `qposadr`, `fps`, `obj_R`, `obj_t`
and `object_scale` to the actual original model. The object NPZ contains
`vertices` and `faces`; scale is applied once from the native motion. Model
joint names, limits, frame transforms and mesh identity are checked. Body-only
G1 output without fingers is unsupported. An explicit
`expected_input_sha256` object can additionally bind `native`, `model_xml` and
`object_mesh` to previously reviewed hashes.

## External runtime and pinned weights

Keep the heavy model environment outside the ordinary retargeter/Blender Python
environment. The base package does not install Torch, GraspGenX or its model
weights. The tool performs no network requests, automatic downloads or inference
API calls. Supply a configured local runtime before using `run`.

The tested implementation is the public
[GraspGenX revision b942909](https://github.com/NVlabs/GraspGenX/tree/b9429097728cb1c430dd78b92edf17ba318aad03).
Its `graspgenx` package tree matches the HOI repair dependency used in our trial.
Use a clean checkout at that revision and follow its Python environment setup.
Its optional serving dependencies `pyzmq` and `msgpack-numpy` are also needed by
the raw hand-description import path, even though this tool runs a local process.
The tested runtime uses Python 3.11, CUDA PyTorch 2.7.0, NumPy 1.26.4,
diffusers 0.11.1, huggingface-hub 0.25.2 and transformers 4.45.2. GraspGenX's
runtime dependencies are separate from this repository's optional video stack.

Download the following four files explicitly from
[GraspGenXModel revision 7c83404](https://huggingface.co/adithyamurali/GraspGenXModel/tree/7c834043c11a11417e31d6d5ea9355801e40a2c1),
preserving their paths under `checkpoint_dir`:

```text
release/gen/config.yaml
release/gen/epoch_736.pth
release/dis/config.yaml
release/dis/epoch_1056.pth
```

For example, in a provisioned environment with `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

for name in ("release/gen/config.yaml", "release/gen/epoch_736.pth",
             "release/dis/config.yaml", "release/dis/epoch_1056.pth"):
    hf_hub_download("adithyamurali/GraspGenXModel", filename=name,
                    revision="7c834043c11a11417e31d6d5ea9355801e40a2c1",
                    local_dir="/data/checkpoints/GraspGenXModel")
```

This is an explicit setup operation, not part of normal execution. Before any
model import, the worker verifies pinned source bytes and checkpoint sizes and
SHA256 hashes. It reports missing or changed assets rather than downloading
replacements or falling back to another backend.

## Outputs and handoff

| File | Meaning |
| --- | --- |
| `input.npz`, `inputs.json` | Original physical geometry, sampled triangle IDs/barycentrics, hand/native frame and hashes |
| `hand_description.json` | Native open/mid swept-volume description supplied to GraspGenX |
| `model/worker_report.json` | Pinned runtime/model identity, measured load/generation timing and diagnostics |
| `proposals.npz` | All generated transforms, learned confidence and native-reference wrist displacement/rotation |
| `preview.html` | Offline interactive original-surface point-cloud view with selectable free-wrist proposals |
| `run.json` | Run status and bound inputs/outputs; always unselected |

Transforms use column vectors: `grasps_object` maps the canonical grasp frame
into the physical object frame. `W_G` maps that canonical frame into the native
wrist frame. Therefore `wrist_object = grasps_object @ inverse(W_G)`;
`wrist_world` applies the bound object transform at the selected frame. Original
proposal IDs remain array indices; display ranking does not renumber them.

The preview's hand is an **open, free-wrist hypothesis**. Learned confidence is
not collision clearance, a contact map, or successful robot IK. This tool does
not generate native arm/finger motion, alter the current sequence, or select a
candidate. The earlier research trial's IK, closure/backoff and CEM-style
upstream ranking are not silently included in this interface.

After choosing a source-supported proposal, fit it to the real native arm and
fingers, declare its material contact targets, and use the existing
[hand refinement guide](../../docs/object-relative-hand-refinement.md)
and [shared audit/repair tools](../repair/README.md). Check original meshes,
source correspondence, approach/closure and affected temporal joins before
adopting it. Existing reviewed grasps and smooth trajectories can continue
without running this tool. No general end-to-end speedup is established by the
fast candidate-generation timings alone.
