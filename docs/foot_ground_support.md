# Scene-ground constraints in Stage 1 and Stage 2

The manipulation config `manip_soma.yaml` fixes
`optimization.foot_support_mode: plane`. Their existing numerical Stage 1
(`init_opt`) and Stage 2 (`contact_opt`) enable `human_foot_ground` and
`human_foot_ground_temporal`, retaining planar `human_foot_contact`. Locomotion
uses finite terrain support in its single `init_opt` stage; see
[Finite scene support](scene_support.md).

`human_foot_contact` penalizes planar foot drift during predicted support; it does
not constrain floor height. Supply the same explicit measured plane and
provenance in `human_foot_ground.plane` in both manipulation stage loss configs.
The maintained YAMLs leave `plane: null`; fitting fails until it is supplied.
For example:

```yaml
human_foot_ground:
  weight: 300.0
  penetration_weight: 2.0
  support_weight: 1.0
  support_threshold: 0.5
  robust_delta_m: 0.005
  support_height_m: 0.002
  penetration_margin_m: 0.002
  plane:
    normal: [0.0, 0.0, 1.0]
    offset_m: 0.0  # Example for a measured plane through the world origin
    source:
      type: original_blender_floor_mesh
      object_name: floor
      scene_sha256: <SHA-256 of the original scene>
```

Measure the plane in the source scene and record its binding. The plane is `normal · world_position + offset_m = 0`, in metres,
with its normal pointing into free space. Bind the original scene/mesh and verify
that the entire foot path is supported by that planar surface. Do not derive the
plane from the candidate body's lowest point. Sloped or stepped terrain needs its
own surface representation; this single-plane loss is not a terrain solver.
An enabled loss without an explicit plane/provenance fails instead of guessing.
The existing plane field accepts nonempty source provenance as a string or
mapping; the example records the original scene hash and floor object. Provide
these existing YAML inputs directly. No separate scene-floor probe input,
automatic floor extraction, output handoff field or postprocessing policy is
introduced by these loss settings.

For finite floors, separate heel/toe support intervals, stance anchors and stair
treads, see [Finite scene support](scene_support.md). The locomotion config
family requires reviewed finite scene patches and does not infer a universal
floor height.

The loss uses the selected native SOMA foot surfaces. The ankle regions
exclude toe vertices when the model's semantic regions overlap. It penalizes the
deepest foot vertex in every frame, including swing. Height attraction uses each
heel/toe region's lowest surface only when its fixed GEM support probability is
above the threshold. This keeps the foot volume intact and allows an unsupported
heel or swinging foot to lift. The four support channels are left ankle, left toe,
right ankle and right toe; these are predictions, not observed ground truth.
The height loss alone can preserve nonpenetration when labels are absent. The
maintained manipulation configs also enable transition smoothing, which requires
finite fixed support probabilities for all four channels; missing labels fail
that enabled objective.

## Touchdown and liftoff smoothing

Both manipulation stages also contain:

```yaml
human_foot_ground_temporal:
  weight: 1.0
  component_weights:
    window_velocity: 0.1
    window_acceleration: 0.1
  options:
    fps: source
    threshold: 0.5
    window_frames: 4
    velocity_scale_mps: 0.35
    acceleration_scale_mps2: 3.0
    correction_speed_tolerance_mps: 0.15
    acceleration_tolerance_mps2: 3.0
```

The loss fixes the native heel/toe material regions, starting motion and GEM
support labels at initialization. It applies tapered windows around label
changes: `window_velocity` limits excess correction speed relative to that
starting motion, and `window_acceleration` limits excess actual acceleration.
It neither attracts swing feet to the plane nor forces their velocity to zero.
`fps: source` resolves from the actual input timeline for physical units.
Locomotion uses the existing `human_scene_support_temporal` with its reviewed
finite-surface intervals instead.

`support_height_m` is a small explicit clearance target above the unchanged
measured plane. Both manipulation configs set `penetration_margin_m` to the same
2 mm target, giving the soft barrier clearance above the measured floor without
changing its height. The optional loss parameter defaults to zero when omitted;
when supplied, it must be nonnegative and no larger than `support_height_m`.
Both remain soft penalties, without a hard guarantee of zero penetration.
Evaluate signed distance to the actual plane, not to the clearance target or
margin, and report residual penetration without rounding it away.

## Review

Retain the existing support-gated planar velocity/stance loss and source/temporal
terms. The new loss has no scene-table, hand-object or object-floor collision term.
It is a soft optimization penalty, so acceptance still requires measuring native
foot clearance over every frame. Inspect support-height error, support drift,
swing motion, source pose, hand-object alignment and transition smoothness
together. Do not infer collision clearance from the loss value alone.

For a targeted comparison, reuse the same predictor observations, camera, shape,
support labels and contacts; rerun numerical Stage 1, its object-placement step,
then Stage 2. Record the placement offset because it can change when Stage 1
changes. Stage 1 **Blender refinement** still freezes the human and only edits
object placement; the ground correction occurs during numerical fitting.
