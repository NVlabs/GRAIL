# Native human fitting for static locomotion scenes

`python -m grail.agentic_kinematic.hoi_locomotion` fits native SOMA03 observations
against an immutable terrain and calibrated source camera. It does not run the
predictors, infer support phases, move the stairs, or invoke manipulation hand
contact and object-placement stages. Use the case's native human-prediction and
MoGe3 outputs as hash-bound inputs. Reusing saved predictions is an explicit
fitting rerun, not fresh inference.

Select `configs/recon_4dhoi/loco_soma.yaml`, matching the
native model, in the case's `optimization_config`. Each fixes
`optimization.foot_support_mode: terrain` and enables `human_scene_support`,
`human_scene_support_temporal` and planar `human_foot_contact` directly in the
existing single `optimization.opt_stage_specs.init_opt.loss_cfg` stage.
Positive-iteration fitting requires the reviewed support manifest and original
native reference geometry. Missing support does not select a base-loss fallback.
The existing `--iterations 0` geometry export remains available without support
annotations; it performs no numerical fitting. There is no locomotion Stage 2.

Native root-speed and canonical root-relative joint-velocity references remain.
Keypoint confidence is estimator-specific: SOMA uses strict `>0.9` for body and
hands. The selected YAML sets these
thresholds explicitly for fitting and evaluation. Predictor precision is recorded
separately in the input provenance; fitting reuses those saved observations.

## Inputs

The case JSON contains exactly `schema_version: 1`, `inventory`,
`backend`, `human_model`, and `optimization_config`. Supported backends are
`soma03`; `human_model.body_model` must match.
Relative inventory
and config paths resolve against the case directory. Rebase model/runtime paths
to paths mounted inside the job; preserve the paired identity manifest.

The inventory uses `schema: "grail.static_locomotion_inputs.v1"`,
`sequence: "dataset/category/video"`, `frame_count`, `fps`,
`original_object_scale`, `is_static_obj: true`, and `files`. Each file record
contains `path`, `sha256`, and `bytes`. Paths are relative to and confined inside
the inventory directory. Required file roles are:

- `source_video`, `object_mesh`, `object_poses`, `render_config`;
- `mask`, `depth`, `depth_metadata`;
- `soma03/human_motion`.

Optional `character_data` is read explicitly when bound. Additional predictor
receipts and calibration inputs can be bound as extra roles. The depth sidecar
must occupy the location expected by `depth_cache_metadata_path`. The original
static pose track must exactly repeat one valid SE(3) pose and agree with the
original rendering after camera-to-world conversion. There are no dummy hand
contact or first-object-motion frame fields.

## Reviewed support fitting

Use the existing reviewed native manifest and its original reference geometry.
When preparing native material-region annotations, the existing zero-iteration
command can export the original fitted-identity initialization and foot IDs:

```bash
python -m grail.agentic_kinematic.hoi_locomotion \
  --case case.json --output probe --iterations 0
```

Review source video support intervals and their finite floor/tread assignments.
Construct native sole subsets from `probe/initial_geometry.npz` and
`probe/foot_regions.json`; bind foot regions to the selected native topology.
Copy the exact
`probe/support_binding.json` into the support manifest's top-level `binding`.
The manifest must follow `grail.scene_support.v1`; the runner checks the source,
inventory, native topology and foot-region binding before fitting. Its phases
remain fixed throughout the solve; GEM probabilities are observations, not
contact ground truth.

The selected YAML determines the loss settings; the support manifest supplies
case-specific observations, not solver options. Supplying the reviewed manifest
and original probe geometry binds the YAML's enabled scene-support losses.
Omitting them is allowed only for the zero-iteration geometry export; a
positive-iteration fit fails. No scene-floor probe or separate base-floor input
is required. The runner uses the YAML's `niter` unless `--iterations` is
explicitly supplied.

For a paired comparison against the earlier support-only setting, copy the
adopted YAML into the experiment directory as a baseline YAML. In its
`optimization.opt_stage_specs.init_opt.loss_cfg`, set
`human_scene_support_temporal.component_weights.window_velocity: 0` and
`human_scene_support_temporal.component_weights.window_acceleration: 0`.
Keep its top-level `weight` positive and `component_weights.no_slip: 0`, retaining
the same reviewed `human_foot_contact` schedule and other weights. This
experiment-only baseline removes the transition contribution while preserving
the required terrain-mode inputs and validation. A comparison that also changes
no-slip labels or ramps must report those additional changes.
Copy the case JSON to `baseline_case.json` and point its `optimization_config`
to that baseline YAML, preserving the input inventory and model bindings. Run
both branches from the same predictor initialization into fresh directories:

```bash
python -m grail.agentic_kinematic.hoi_locomotion \
  --case baseline_case.json --output baseline \
  --support-manifest support_manifest/scene_support.json \
  --support-reference-geometry probe/initial_geometry.npz
python -m grail.agentic_kinematic.hoi_locomotion \
  --case case.json --output transition \
  --support-manifest support_manifest/scene_support.json \
  --support-reference-geometry probe/initial_geometry.npz \
  --depth-reference baseline/depth_reference.npz
```

Both branches use the YAML's `human_scene_support` settings: outer weight 300,
penetration 2, support height 1, tangential anchor 2, boundary 1, robust delta
0.005 m. The transition branch also applies the temporal settings below. The
runner consumes the fixed reviewed manifest and never infers support intervals. Manipulation hand-contact terms remain outside this runner's
scope. Experiment-specific baseline copies belong with run artifacts, not as
additional maintained configs under `configs/recon_4dhoi/`.

Both runs reset their RNGs identically. The second command reuses the baseline's
exact initial depth targets and visibility masks. Compare
`depth_sampling_binding.json` to verify that these bindings match between
branches.
Keep the authored manifest beside its `receipt.json`, which binds the original
probe geometry and remapping output. New receipts use
`grail.static_support_remapping_receipt.v1` with the exact `model_type`; the
historical `grail.soma_support_remapping_receipt.v1` schema remains SOMA-only.
The runner preserves those byte hashes and
checks repeat native FK within 10 micrometers and projected points within
0.002 pixels; terrain, topology and other fixed fields must match exactly.
Without `--depth-reference`, small FK differences can still change raster
visibility at boundaries; report any depth-mask mismatch instead of calling the
comparison an exact ablation.

Both runs preserve the full source frame count/FPS, original meshes and scale,
terrain pose, camera, native identity/shape, root orientation, fingers and wrists.
Only native body residuals and declared human translation residuals are edited.
Frozen channels are restored before native FK, rather than relying on zero
gradients or identity rotation roundtrips. Output smoothing and generic
pre-evaluation frame truncation are disabled.

## Outputs and review

Each run saves `initial.pkl`, `result.pkl`, exact native geometry, source
observations, effective config, model/code/input bindings, original global-motion
reference, timing/history, and post-export frozen-field guards. `completion.json`
means the numerical run ended; it never declares the motion accepted.

Compare supported sole height/hover, all-frame penetration, stance drift, landing
velocity, acceleration, source keypoints, and source/oblique views. A finite
support-patch or column-exit proxy does not establish full original-mesh or swept
collision freedom, balance, or physical feasibility. Retain failures and the
original fallback. This endpoint performs locomotion initial reconstruction;
it does not automatically enable scene-support constraints in manipulation
Stage 1/2 or produce a robot-retargeting result.

## Default touchdown transition

The adopted settings are in
[`configs/recon_4dhoi/loco_soma.yaml`](../configs/recon_4dhoi/loco_soma.yaml).
Select the matching model's file through `case.json`'s `optimization_config`; change support loss
weights and options in `optimization.opt_stage_specs.init_opt.loss_cfg`. Save
case-specific test variants with experiment artifacts. Optional losses can use
`weight: 0`; terrain mode requires positive top-level weights for
`human_scene_support`, `human_scene_support_temporal` and `human_foot_contact`.
The temporal loss's `fps: source` resolves to the immutable inventory's
FPS; a numeric override must agree with that FPS. There is no separate transition
profile or loss-option CLI.

The configuration separates normal height activation from tangential locking,
retains planar `human_foot_contact` with the reviewed scene intervals, and adds
tapered foot-material temporal terms near touchdown and release. No-slip stencils
cannot cross an authored interval or tread switch. The temporal term's
`component_weights.no_slip: 0` avoids duplicating the retained foot-contact loss. Source contact scores are preserved in the observations. Provenance
records the selected YAML's hash and the resolved settings, including
the actual FPS.

```yaml
optimization:
  foot_support_mode: terrain
  opt_stage_specs:
    init_opt:
      niter: 400
      loss_cfg:
        human_foot_contact:
          weight: 30.0
          label_source: scene_support
          point_mode: foot_parts
          velocity_space: planar
          options:
            fps: source
            normal_velocity_weight: 0.0
            long_horizon_lag: 4
            long_horizon_weight: 0.25
            tangent_blend_frames: 3
        human_scene_support:
          weight: 300.0
          penetration_weight: 2.0
          support_weight: 1.0
          tangent_weight: 2.0
          boundary_weight: 1.0
          robust_delta_m: 0.005
          height_blend_frames: 0
          tangent_blend_frames: 3
        human_scene_support_temporal:
          weight: 1.0
          component_weights:
            no_slip: 0.0
            window_velocity: 0.1
            window_acceleration: 0.1
          options:
            fps: source
            window_frames: 4
            tangent_blend_frames: 3
            velocity_scale_mps: 0.35
            acceleration_scale_mps2: 3.0
            correction_speed_tolerance_mps: 0.15
            acceleration_tolerance_mps2: 3.0
            normal_velocity_weight: 0.25
            long_horizon_lag: 4
            long_horizon_weight: 0.25
```

These are solver defaults, not universal physical limits. The retained foot-contact
term averages planar metre displacement only within one authored part/surface
interval. The transition terms are dimensionless Huber penalties on excess correction speed relative to
the frozen initialization and excess native acceleration. Their windows include
approach/release for continuity, but do not attract swing feet to a surface or
pin swing velocity to zero. Height attraction remains active at the evidenced
first touch; nonpenetration stays active on all frames. Short or uncertain
contact intervals are not automatically merged.

Each nonzero-iteration run saves `depth_reference.npz` and its JSON binding when
building its initial depth targets/visibility. For a paired candidate, pass
`--depth-reference baseline/depth_reference.npz` to reuse the exact same rays
and visibility masks after the shared native probe parity check. The loader
checks all frame keys, numeric types/shapes, archive hashes, original inputs,
model assets, depth config and camera. Compare final `depth_sampling_binding.json`
files; a byte-identical pair eliminates the raster-mask confound. It does not
turn estimated depth or contact annotations into ground truth.

Review all phase boundaries and the whole sequence, including source/depth
alignment, hovering, penetration and drift. A reduction in landing acceleration
alone is not sufficient to accept a new motion.
