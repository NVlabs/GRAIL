# Retargeting

Native SOMA-to-Unitree-G1+DEx3 retargeting is the public default. GRAIL exports the
reconstructed SOMA 0.3 forward kinematics to a hash-bound BVH and invokes the
pinned public [NVIDIA SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter).
The older SMPL-X-to-G1 GMR path remains available for explicit legacy configs.

## Install

```bash
git submodule update --init imports/soma-retargeter imports/GMR
bash scripts/setup/install_soma_retargeter.sh
```

The helper verifies the pinned source revisions and the original GMR
G1-with-DEx3 MJCF, hydrates the bundled Unitree G1 assets, and creates
`imports/soma-retargeter/.venv` from its lock file. This isolated
Python 3.12 environment owns the pinned Newton, Warp, USD, NumPy, and SciPy
dependencies; they must not be installed into GRAIL or SONIC. Other robot
embodiments are not exposed by the GRAIL wrapper in this release.

## SOMA to Unitree G1 + DEx3

Retarget one trusted reconstruction output:

```bash
python -m grail.retargeting.soma \
    <path/to/hoi_data.pkl> \
    <new_output_directory> \
    --model-config configs/retargeting/soma_g1.json
```

For the default reconstruction, inputs live below
`<results_dir>/generation/4dhoi_recon_soma03_valid/`. Never feed an SMPL-X
cache to this command; the exporter validates the native SOMA identity and
runtime bindings before invoking the official retargeter.

The output contains:

- `soma_export/source.bvh` — native SOMA hierarchy, pose, per-joint
  translations, and source frame rate.
- `soma_export/native_source.npz` and `native_rest.npz` — canonical source and
  identity-preserving rest transforms.
- `g1_dex3_target/` — source-bound SOMA registry containing the original
  G1+DEx3 MJCF, original joint limits, and wrist-relative hand-task calibration.
- `g1_retarget/csv/*.csv` — 50-column result: frame, six root channels, 29 G1
  body/wrist joints, and 14 DEx3 finger joints.
- JSON manifests with source/model/runtime hashes and FK round-trip errors.

Existing output directories are refused, and the result is marked
`RETARGETED_UNREVIEWED`. Replay and validate it before policy training or
hardware use. The wrapper rejects padded or relabeled 29-DoF files: the default
target must expose the native 43 hinges and 50 generalized coordinates.

SOMA thumb, index, and middle segments are mapped anatomically to the three
DEx3 digits on each hand. Human ring/pinky motion is not represented by
independent DEx3 joints. This is a source-guided IK initialization, not a
contact-aware grasp acceptance test.

The retargeter may apply its configured kinematic foot-plant stabilization.
This is separate from GRAIL reconstruction contact: the original
`contact_labels`, object contacts, and contact loss are neither regenerated nor
replaced by this wrapper.

The G1 CSV is the audited kinematic retarget result, not the legacy SONIC
`robot/`, `objects/`, `object_usd/`, `meta/` motion-library layout. Do not point
tracking training directly at the CSV; simulation replay and motion-library
packaging are separate validation steps.

## Replay robot, object, and table

Use the source-bound target from the same retarget run and the exact validated
`hoi_data.pkl`. The replay loads the 43-joint robot, object mesh, per-frame
`obj_R`/`obj_t`, and stored `obj_scale`. It creates a `2.0 x 0.6 x 0.04 m`
tabletop with four legs directly below the initial object mesh:

```bash
export DISPLAY=:1
RETARGET_DIR=<new_output_directory>

imports/soma-retargeter/.venv/bin/python -m grail.retargeting.soma_replay \
    "$RETARGET_DIR/g1_retarget/csv/source.csv" \
    <path/to/hoi_data.pkl> \
    "$RETARGET_DIR/g1_dex3_target" \
    "$RETARGET_DIR/g1_dex3_scene.mp4"
```

The table height uses the minimum world-space vertex after applying the initial
object rotation and scale. Override the tabletop dimensions with
`--table-size WIDTH DEPTH THICKNESS`; replay does not depend on
`scene_data.table`.

This is a kinematic alignment replay; it does not run physics or collision
response. Object material textures are omitted, but the original mesh geometry,
scale, and trajectory are used. Review contact, penetration, temporal alignment,
and grasp geometry before accepting the motion.

For a deliberate body-only diagnostic, pass `--target unitree_g1`. That fallback
emits 29 body/wrist joints and no actuated fingers; it is not the public default
for manipulation.

## Legacy SMPL-X / GMR path

Use this only for outputs reconstructed with `manip_smplx.yaml` or
`loco_smplx.yaml`.

```bash
bash scripts/setup/install_env_sonic.sh
conda activate sonic

RECON_DIR="<your_results_dir>/generation/4dhoi_recon_smplx_valid"
OUTPUT_FOLDER="<your_output_folder>"

bash grail/retargeting/scripts/retarget_pipeline.sh \
    "$RECON_DIR" "$OUTPUT_FOLDER"
```

This writes `data/motion_lib/<your_output_folder>/` with `robot/`, `objects/`,
`object_usd/`, and `meta/`, plus the manipulation post-processed
`<your_output_folder>_ha/` twin and BPS encodings where applicable.

For terrain and sitting, skip hand-action processing and zero wrist motion:

```bash
bash grail/retargeting/scripts/retarget.sh \
    "<your_results_dir>/generation/4dhoi_recon_smplx_valid" \
    "<your_terrain_output_folder>" \
    --zero_out_wrist
```

The G1 metadata, baked SMPL-X shape, and GMR mapping are under
`data/body_profiles/unitree_g1/smplx/`. The legacy entry points load that
canonical directory directly.

## Troubleshooting

```{list-table}
:widths: 36 64
:header-rows: 1

* - Symptom
  - Fix
* - SOMA Retargeter Python is missing
  - Rerun `bash scripts/setup/install_soma_retargeter.sh`.
* - Native identity differs from the model config
  - Use the exact identity/config that produced the reconstruction; the exporter refuses a silent shape replacement.
* - SOMA hierarchy or FK round-trip check fails
  - Update all pinned submodules together and use a new output directory; do not bypass the check.
* - `ModuleNotFoundError: general_motion_retargeting`
  - This is the legacy SMPL-X path; rerun `bash scripts/setup/install_env_sonic.sh`.
```
