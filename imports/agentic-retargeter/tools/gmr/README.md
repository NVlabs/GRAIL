# External GMR conversion

This thin tool calls an existing GRAIL checkout and its GMR environment. Shared
implementation lives in [`src/gmr.py`](../../src/gmr.py). GMR, model weights and
upstream human reconstruction are not copied into this repository. It performs
initial conversion, before the native Blender contact-refinement loop.

```bash
python tools/gmr/run.py plan --config /path/to/gmr.json --output /path/to/new-run
python tools/gmr/run.py preflight --config /path/to/gmr.json --output /path/to/new-run
python tools/gmr/run.py run --config /path/to/gmr.json --output /path/to/new-run
```

The installed-package equivalent is `python -m agentic_retargeter.gmr` with the
same arguments. Commands run in the foreground; no model API is used. `plan`
only reads inputs. `preflight` additionally checks the external CLI imports,
required options and resolved GMR installation; it does not convert a motion.

Example configuration (relative paths resolve against the configuration file):

```json
{
  "grail_root": "/path/to/GRAIL",
  "python": "/path/to/sonic/bin/python",
  "source": "/path/to/case/hoi_data/hoi_data.pkl",
  "handoff": "/path/to/reviewed-upstream-handoff.json",
  "human_model": {
    "convention": "custom_g1",
    "body_models": "/path/to/body_models"
  },
  "robot": "unitree_g1_with_hands",
  "fps": 24,
  "process_hands": false
}
```

`human_model.convention` defaults to `custom_g1`. The upstream loader consumes
`<grail_root>/data/body_profiles/unitree_g1/smplx/g1_smplx_tpose.obj`,
`g1_smplx_param.npz`, the base model directory, and its model-initialization code.
Their hashes are bound in the plan. Arbitrary custom-model snapshots cannot be
passed to this upstream CLI: if the reviewed source uses different actual model
parameters, adapt and verify the upstream loader first. Changing a label or
setting a flag does not establish compatible geometry.

For an explicitly bound legacy standard SMPL-X source, choose
`"convention": "standard"`; the adapter passes `--no_g1_proportions` and preserves
that convention. It never silently relabels a standard source as custom.

The reviewed handoff is a JSON object with the exact `schema_version`,
`source_sha256`, `fps` and `human_model` fields shown by `plan.required_handoff`.
`human_model` contains the convention, `asset_sha256` mapping and loader SHA-256.
These fields must come from the upstream source/model review. The plan's copy is
a description of required evidence, **not proof** that this checkpoint used those
assets. Additional review metadata can be retained at the handoff's top level.
An absent or mismatched handoff blocks execution, but still allows inspection of
the plan and external runtime diagnostics. All supplied pickles must be trusted
upstream inputs; the external pipeline deserializes them during conversion.

`process_hands: true` adds an explicit second stage with
`--treat_hands_equally --include_contact_points --per_object` and zero grasp
anticipation. It preserves both arms, uses each hand's contact timing, and does
not implicitly impose an object-lift-only pickup rule. `skeleton_meta` can bind a
different upstream skeleton metadata file; otherwise the upstream
`data/body_profiles/unitree_g1/smplx/g1_skeleton_meta.pkl` is used. Review any inferred closure against
the video before editing or acceptance. No BPS export or upload is performed.

The output path must be new and outside the upstream checkout, source-case tree
and human-model tree. Each run writes `raw/`, optional `processed/`, stage logs
and `run.json`. The report binds source meshes, model assets, the robot asset
tree, IK configuration, relevant upstream code, handoff and outputs with hashes.
Inputs are rechecked around execution. Nonzero exit, changed inputs or missing
motion outputs produces `FAILED_UNREVIEWED`; a completed conversion is
`CONVERTED_UNREVIEWED`, never an accepted contact-aware result. Partial failure
outputs remain available for diagnosis. A later edit should create a separate
versioned `.blend` through the local refinement tools.
