# Staged HOI Fitting and Placement

Fit native human and object motion to the source video using the original
geometry and frame timing. The default predictor is GEM-X native77 with
SOMA-X 0.3.3. See [GRAIL Installation Guide](installation.md) and
[reconstruction backends](recon_4dhoi.md) for setup and observation inputs.

## Stage boundaries

| Stage | Work and required evidence |
| --- | --- |
| Observations | Source-bound human prediction, SAM2 masks, MoGe-3 depth/calibration and MoSE3/FoundationPose tracking. Keep raw and processed trajectories and actual producer receipts. |
| Numerical Stage 1 | Fit pose/depth/placement with contact refinement disabled. Save native output and geometry before edits. |
| Reviewed placement | Inspect source contact and object-motion timing independently. Keep the human fixed; apply a declared native object translation and transition with an exact protected tail, or use an evidenced unchanged-placement decision. |
| Numerical Stage 2 | Start from the selected placement checkpoint. Use this video's authored contact regions and temporal scope; Stage 2 depth loss is disabled in the maintained manipulation configs. |
| Final evidence and review | Decode the complete source-rate video, inspect source/oblique views and native numerical checks, and record the selected checkpoint, limitations and quality decision. |
| Further human/robot refinement | When requested, perform native Stage 2 Blender contact repair, then hand off the selected human/object pair to robot retargeting and robot review. These are separate scopes. |

The maintained manipulation defaults are 400 Stage 1 and 200 Stage 2 iterations.
The executed configuration and receipt are authoritative. Iteration counts and
normal process exit establish neither deliverable completeness nor motion quality.
`PARTIAL`, `REJECTED` and `UNREVIEWED` remain distinct from accepted motion even
when the requested numerical/evidence pipeline is complete. Report still-only
review as such; a decoded video does not imply full-speed replay was reviewed.

### Static, uncertain and no-contact cases

A Stage 2 static skip requires full original and selected-parent translation
**and rotation** checks, preserved output bytes, and source/geometry review.
A stationary prefix, low motion score or unknown contact is insufficient.
The receipt records `SKIPPED_STATIC_OBJECT`; it does not accept contact quality.

Contact, alignment, object-motion onset and lifting are independent events.
An unresolved onset can remain explicitly `unknown` / `null` with dense,
source-bound review. It cannot authorize a static skip or an invented onset
at frame zero. Preserve the varying original track and run numerical Stage 2.
When the evidence supports only `no_contact` / `unknown` contact phases,
`reviewed_unchanged` placement preserves the Stage 1 checkpoint exactly; it still
requires reviewed source/oblique evidence and full Stage 2 without fabricated
attraction targets.

## Native fitting tools

<span id="run-the-staged-native-tools"></span>

Prepare the case JSON and input inventory described below.
Case paths resolve relative to their declaring file. Review annotations must
bind the actual source, model topology and original object geometry.

### Case JSON and input inventory

`case.json` has exactly these keys (paths relative to the file):

```json
{
  "schema_version": 1,
  "inventory": "inventory.json",
  "backend": "soma03",
  "human_model": {
    "body_model": "soma03",
    "static_cam": true,
    "soma_runtime": "gem_x",
    "gem_x_soma_assets": "/path/to/GEM-X/assets/soma",
    "soma_identity_params_path": "/path/to/soma03_custom_smplx_identity.npz"
  },
  "optimization_config": "manip_soma_case.yaml"
}
```

Choose `optimization_config` for the interaction:

- **Manipulation:** copy `configs/recon_4dhoi/manip_soma.yaml`. Set `loss_cfg.human_foot_ground.plane` under both `init_opt` and `contact_opt` in `optimization.opt_stage_specs`. Record the measured plane and how it was obtained.
- **Locomotion:** use the locomotion configuration and supply the reviewed finite-surface support manifest. For stairs, it binds each stance interval to the original floor or tread and its boundaries.

`inventory.json` uses `"schema": "grail.cached_fitting_inputs.v1"` and records
`sequence`, `frame_count`, `fps`, `original_object_scale`, the placement events
(`contact_frame`, `object_first_motion_frame`, `recovery_start_frame`,
`recovery_end_frame`, `alignment_frame`) and a `files` map. Every file entry is
`{"path", "sha256", "bytes"}` with `path` relative to the inventory directory;
the loader rejects changed bytes and paths that leave that directory. Required
roles are `source_video`, `object_mesh`, `object_poses` (per-frame 4x4 camera
poses), `render_config` (the 2D-generation first-frame render
pickle), `mask`, `depth`, `depth_metadata` and `<backend>/human_motion`. These are
the observations produced by the default pipeline's prediction steps (GEM-X native human motion, SAM2 masks, MoGe-3 depth with its metadata sidecar, and
MoSE3 + FoundationPose object poses). Stage 2 additionally needs the reviewed
`contact_map` / `contact_temporal` pair, passed on the command line or bound as
`<backend>/contact_map` and `<backend>/contact_temporal`. GRAIL does not ship a
helper that assembles this inventory from a `recon_4dhoi` results directory;
write it once per case and keep it with the case outputs.

For the released kettle, the right-hand approach entry under `hands.right_hand` starts with `{"id":"approach","frames":[0,33],"state":"no_contact","regions":[]}` (excerpt). It covers frames 0 through 32. The next two frames were marked `unknown`. The full map also records the source-video and topology hashes, contact evidence and the other hand. Build these intervals from your own clip.

With the case inventory and measured support bound, start Stage 1 from the configured GRAIL environment. Choose an unused output path:

```bash
CASE=/workspace/runs/my-lift
python -m grail.agentic_kinematic.hoi_stages stage1 --case "$CASE/case.json" --output "$CASE/stage1"
```

After Stage 1 and placement review, point Stage 2 at the exact selected parent described in the placement decision. A reviewed unchanged placement can select the original Stage 1 fit when supported by that decision. The following argument pattern uses a case whose reviewed placement produced `placement/stage1_object_refined.pkl`; substitute the actual parent and choose a fresh output.

```bash
CASE=/workspace/runs/my-lift
python -m grail.agentic_kinematic.hoi_stages stage2 \
  --case "$CASE/case.json" --stage1-dir "$CASE/stage1" \
  --parent "$CASE/placement/stage1_object_refined.pkl" \
  --contact-map "$CASE/contact_map.json" --contact-temporal "$CASE/contact_temporal.json" \
  --output "$CASE/stage2-reviewed"
```

Stage 1 accepts new contact annotations later; Stage 2 validates its selected
parent and annotation bindings. The {blob}`native Blender refinement guide <docs/hoi_refinement.md>`
covers body-rotation editing, local contact repair, native export and save/reopen
checks for SOMA03. Arbitrary display-mesh changes
are not converted into native body parameters.


## Retargeting

<span id="retarget-a-reviewed-motion"></span>

Pass the selected native human/object result, source clock, identity, object CAD and support scene to the [robot retargeting workflow](retargeting.md).
