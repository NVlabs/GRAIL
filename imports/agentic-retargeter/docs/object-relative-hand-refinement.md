# Object-relative hand refinement and SOMA

This opt-in research workflow extends [hand initialization](hand-initialization-workflow.md)
with smoother hand trajectories, phase-aware whole-body temporal costs and
bounded collision-repair rounds. It preserves the original meshes, native
limits, source clock, contact roles and source-motion reference. A hand-only
candidate remains a target for whole-body retargeting, not a feasible mounted
robot motion.

## Hand trajectories

Run the existing Topo-inspired initialization and distal-surface fitting, then
use a selected checkpoint as the paired solver's warm start. Continue measuring
source deviation against the original source, never against the newest result.
Keep each candidate and its report in a new output directory.

The optional phase spline has two live wrist curves: world coordinates outside
contact and object coordinates during authored hold, support and slide phases.
Exterior quintic transitions blend position and SO(3) orientation without
changing phase masks. A free hand therefore need not inherit an object's motion
just because its path was stored in object coordinates. Acquisition and phase
joins remain explicit review intervals. This is not a constant-grip constraint.

Native finger angles use bounded controls:
`q = B @ (lower + (upper-lower)*sigmoid(control_logits))`.
The nonnegative cubic basis preserves the original limits and avoids applying a
sigmoid to a potentially steep dense spline. Warm-fit deviations are reported;
the fitted curve does not replace the original reference.

Merge this fragment into a fully bound paired configuration:

```json
{
  "device": "cuda:0",
  "execution_backend": "eager",
  "rotation_backend": "rodrigues",
  "paired_steps": 400,
  "temporal_coordinate_space": "phase",
  "object_relative_temporal_roles": ["hold", "support", "slide"],
  "trajectory_parameterization": {
    "kind": "phase_cubic_bspline",
    "control_count": 17,
    "finger_space": "bounded_joint_controls",
    "transition_frames": 18,
    "allowed_transition_roles": [
      "no_contact", "unknown", "contact_acquisition", "approach", "release"
    ]
  },
  "feasible_spline_step": {
    "kind": "affine_projection_backtracking",
    "max_backtracks": 12,
    "shrink_factor": 0.5
  }
}
```

```bash
python -m agentic_retargeter.hand_sequence_pair paired.json --output runs/paired_v002
```

The fragment does not supply inputs, source arrays, phases, warm checkpoint or
objective weights. Those must remain explicitly bound to the case. Seventeen
controls and eighteen exterior frames are tested settings for 121-frame clips,
not universal choices. Transitions that overlap or cross undeclared roles fail.
Spline parameters require eager execution; CUDA Graph remains available for the
original per-frame parameterization.

After Adam proposes a step, the optional affine projection finds the nearest
position-control proposal satisfying the unchanged dense source-relative wrist
boxes. The phase form jointly projects world and object position controls.
The subsequent decode-only backtracking checks the original componentwise
translation and rotation-vector boxes, finite values and native finger bounds.
It restores the preceding controls if no trial is feasible. It never widens
the source corridor. Adam moments and its step counter retain the full proposal;
the report records this policy, projection failures, step fractions and zeros.
An unsuccessful projection falls back to the same feasibility filter.

This option is disabled by default. It does not support shared-hold or
frozen-frame overlays. Rotation-vector bounds remain componentwise boxes, not a
new rotation-angle norm bound. The existing numerical tolerance is retained.
An alternative shared-hold boundary blend exists for explicit constant holds,
but is a separate ablation and should not silently freeze support or slide.

## Whole-body optimization

Package the selected wrist/finger trajectory using the hash-bound handoff
contract in the baseline document. Preserve independently selected anchors and
their shortfalls. For a matched target ablation, hold the material-anchor list
fixed and retain newly generated anchor reports separately. The experiment
handoff exporter and batch drivers are not packaged CLI commands.

Run fresh CUDA forward whole-body initialization, followed by the synchronous
trajectory solve and native export. Residual/Jacobian evaluation uses CUDA;
sparse LM/LSMR assembly and solving, the optional hand translation QP and some
validation remain on CPU.

The optional SOMA `temporal_optimization` entries below operate on the actual
robot wrist, not just on target arrays:

```json
{
  "relative_hand_temporal": {
    "roles": ["hold", "support", "slide"],
    "translation_acceleration_weight": 0.008,
    "translation_jerk_weight": 0.000008,
    "rotation_acceleration_weight": 0.00008,
    "rotation_jerk_weight": 0.00000008,
    "hold_translation_velocity_weight": 0.0,
    "hold_rotation_velocity_weight": 0.0
  },
  "world_hand_temporal": {
    "roles": ["hold", "support", "slide"],
    "translation_acceleration_weight": 0.032,
    "translation_jerk_weight": 0.000032,
    "rotation_acceleration_weight": 0.00032,
    "rotation_jerk_weight": 0.00000032
  },
  "hinge_speed": {"weight": 4.0, "sigma_rad_s": 0.2}
}
```

These are example ablation weights, not an accepted default. Keep the baseline
required temporal fields and `method: "synchronous_sparse"`. The relative/world
terms require the robot hand target bridge. Relative derivatives use complete
same-role active stencils and include derivatives with respect to both robot
and adjustable object poses. World derivatives use the complementary stencils,
including free motion, acquisition and joins; their object Jacobian is zero.
Use identical role sets when these costs are intended to partition the clock.
Zero-relative-velocity costs, if enabled, apply only to holds. Support and slide
must retain their allowed motion.

Coefficients multiply sums of squared SI derivatives, with the original time
step. They are not per-frame means. The hinge-speed term penalizes only native
scalar-joint speeds above explicit gates. Its built-in gates are authored
animation-review values, not manufacturer hardware limits. It does not unwrap,
retime or smooth a trajectory after optimization. Named joint multipliers can
separately strengthen arm/waist temporal costs while retaining finger costs.

For exactly static full-source objects, continue to omit material contact
position/normal losses and retain collision/source terms. A moving object's
stationary prefix retains the shared adjustable but strictly stationary pose
policy. Do not infer a static case from that prefix.

## Finite collision-repair loop

The host agent controls a finite sequence of experiments. There is no packaged
autonomous repair daemon. Set a maximum round count and resource budget before
launching; retain the preceding candidate if a round fails or trades away an
important constraint. Each round should:

1. Inspect the mounted native result at all original frames and selected
   subframes, including hand-object, cross-hand, robot self and scene geometry.
   Keep geometric crossing evidence separate from signed-distance proxies.
2. Select a small set of observed collision frames and propose native hinge
   corrections with original limits, body posture and hand targets retained.
   Save before/after mounted views and measurements. A repaired key is still an
   unaccepted hypothesis.
3. Bind the keys to the original model, source BVH and clock, then rerun fresh
   forward initialization and synchronous optimization with those soft costs.
   Optionally add original-triangle collision witnesses as additional samples
   of already enabled mesh pairs.
4. Compare the full exported trajectory, especially neighboring frames and
   acquisition/release joins. Keep the measured improvements and regressions;
   do not accept a result because its optimizer objective decreased.

The top-level SOMA keyframe configuration is
`"pose_keyframes": {"path": "pose_keys.json", "sha256": "<actual SHA256>"}`
and requires the explicit contact bridge, even when its contact list is empty.
The artifact schema is `soma_native_pose_keyframes.v1`, with
`model_xml_sha256`, `source_bvh_sha256`, `frame_count`, `fps` and `keyframes`.
Each key contains an original `frame`, exact native hinge-name-to-radian
`joints`, positive `weight` and positive `sigma_rad`. The resulting cost is
`sqrt(weight)/sigma_rad * (q-target)` only at those named frames. Free-root
pins are outside this contract. Conflicting keys, hash/clock changes and
out-of-limit angles fail validation.

Additional SOMA collision samples live in the contact artifact's
`collision_avoidance.witness_groups`. A group names an original source geom,
an already enabled target pair, an exact mesh identity, a positive weight and
original triangle/barycentric samples. Original samples and normalization stay
intact; each added group contributes its own weighted mean. The optional
`mesh_identity_schema: "native_raw_geom_pose.v2"` binds exact compiled raw
vertices, face order, geom position and geom quaternion before body-coordinate
arithmetic. The existing body-coordinate fingerprint remains the default.
The portable identity extension has local tests; require remote preflight and
retain the separate remote quality result. Do not replace a failed identity
check with a geometry tolerance. These queries retain the winding-distance heuristic over the full target;
extra samples do not establish a safe side or clearance on an open mesh.
