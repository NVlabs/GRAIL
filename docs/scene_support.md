# Finite scene support for human reconstruction

`human_scene_support` extends foot support to finite floor and stair patches. It
supports native SOMA vertices through `foot_region_indices(model)`.
The locomotion config `loco_soma.yaml` fixes
`optimization.foot_support_mode: terrain` and enable `human_scene_support`,
`human_scene_support_temporal` and planar `human_foot_contact` in their existing
single Stage 1 `init_opt.loss_cfg`, with 400 iterations by default. They require
an explicit reviewed manifest; there is no added locomotion Stage 2, default
ground height or automatic
nearest-tread reassignment during fitting. Manipulation retains its existing
Stage 1/2 structure with [plane support](foot_ground_support.md).

## Fixed inputs

Before optimization, bind the source video, original scene geometry and transform,
camera, body model and support-label provenance. Use the existing reviewed
`grail.scene_support.v1` manifest of exposed, upward-facing support surfaces in
world metres. Represent each as a finite convex polygon;
split nonconvex regions and clip buried portions of stacked stair boxes. Do not
extend an upper tread into an infinite plane. This loss integration consumes the
existing manifest directly; it does not add an extraction/draft/compile workflow
or require a separate base-floor measurement.

Review heel/toe support intervals against the source video. Left/right ankle and
toe can have different intervals and surfaces. GEM support scores are predicted
labels, not ground truth. Keep the assignments fixed within an experiment. If
review finds a wrong landing interval or tread, revise the manifest explicitly
and run a new comparison instead of silently changing the surface during a solve.

The inline manifest has schema `grail.scene_support.v1`, `frame_count`, nonempty
`source` provenance, `surfaces` and `intervals`. A surface supplies `id`, `origin`,
`normal`, ordered `polygon_world` corners and provenance. An interval supplies
`part` (`L_Ankle`, `L_Toe`, `R_Ankle`, `R_Toe`), inclusive `start`, exclusive `end`,
`surface_id` and provenance. Optional fields include fixed native
`contact_vertex_ids`, `blend_frames`, `weight`, `boundary_margin_m` and an explicit
`anchor_world` on the surface. `frame_count` must match the native motion.

## Objective

```yaml
human_scene_support:
  weight: 300.0
  penetration_weight: 2.0
  support_weight: 1.0
  tangent_weight: 2.0
  boundary_weight: 1.0
  robust_delta_m: 0.005
  manifest: # insert the reviewed grail.scene_support.v1 mapping here
```

The maintained locomotion YAMLs place these weights in `init_opt.loss_cfg`;
the manifest supplies case observations rather than solver settings. Missing
support input fails a positive-iteration fit. The existing native locomotion
runner can still export initial geometry with `--iterations 0` for native
material-region authoring; that is not a fitted result. See
[the native fitting procedure](static_locomotion_fitting.md).

These are solver defaults, not universal physical limits. The four terms are:

- **Nonpenetration:** every native foot region, every frame, tested against each
  finite surface where the vertex projects inside its footprint.
- **Support height:** the lowest vertex of a fixed sole patch approaches its
  assigned surface only during that patch's support interval. Swing remains free
  to lift. Selecting fixed near-sole vertices avoids flattening the whole foot.
- **Tangential anchor:** the patch centroid stays near one shared stance position.
  By default the interval's weighted mean eliminates the anchor analytically, so
  the whole stance can adjust while frame-to-frame drift is penalized. An explicit
  world anchor can instead be supplied when independently supported.
- **Boundary:** the support patch centroid stays inside the assigned polygon.

Transition weights blend inside each interval; they do not add attraction outside
it. Existing source-pose, temporal and hand-object terms remain relevant. The
manifest and tensor constants are cached; gradients flow through native geometry.

Planar `human_foot_contact` uses `label_source: scene_support` so no-slip stencils
remain within one reviewed heel/toe surface interval. Raw GEM labels remain
observations and do not override reviewed swing. The retained
`human_scene_support_temporal` term uses tapered correction-speed and acceleration
windows at touchdown/liftoff, with `component_weights.no_slip: 0` to avoid
counting the separate no-slip objective twice. It does not attract swing feet to
a surface or lock their velocity to zero. Source FPS supplies physical units;
loss options stay in the selected YAML.

## Evaluation and scope

Compare original and candidate with identical fixed labels and surfaces. Measure
all-frame foot penetration, support hovering, tangential drift, boundary error,
swing clearance, landing speed, source reprojection and hand-object alignment.
Inspect original-camera and novel-angle playback. Preserve failed candidates and
the old sequence; a lower aggregate loss is not acceptance.

The default `penetration_mode: normal_depth` checks vertical/normal columns over
support polygons. At a tread edge, a barely overlapping foot vertex can receive
an excessively large vertical correction. For a scene composed of verified solid
columns, a manifest may instead set `penetration_mode: column_exit` and supply
separate `collision_surfaces` with the same surface schema. These must preserve
the **full original overlapping solid footprints**, including the floor; exposed
tread strips remain the surfaces used for stance assignment. The penalty uses the
smaller of depth below the top and inward side distance for each column, then the
maximum across columns. This provides a continuous horizontal escape direction
near an edge. Using exposed strips as collision columns would introduce artificial
low-cost seams and is inappropriate for stacked boxes.

This mode is a lower-bound exit-distance proxy for a union of convex downward
columns, not an exact union signed distance or a generic scene-mesh representation.
Preserve the old normal-depth diagnostic alongside the exit-distance diagnostic
when comparing modes; a change of metric alone is not evidence of improvement.
Neither mode implements full stair-riser collision, swept-mesh collision,
whole-sole support area or physical balance. Its boundary term constrains a patch
centroid, not every foot vertex. Report those limits and inspect edge/riser motion
separately. A floating or intersecting result may remain under finite soft weights.

Stage 1 Blender placement policy is separate: the human stays frozen there and
only the object placement is refined. Human support correction happens during
numerical fitting. Do not add this term to a case without verified scene support
geometry merely because it worked on another scene.
