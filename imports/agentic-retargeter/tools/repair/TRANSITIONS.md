# Scene-aware transition tools

Use these tools after choosing feasible mounted grasp keys. Keep the accepted
hold in the input checkpoint and edit only approach, closure or another declared
short transition. They use the same native model, geometry, correction bounds,
protected frames and original-triangle audit as `run.py`.

The referenced repair config must include the workflow's
[`temporal.max_velocity_rad_s` hard gate](README.md). It checks every hinge over
the full native sequence, including protected holds and approach/hold joins.
The solver's smoothness weights do not establish a pass. Violations fail the
final audit even outside the edited transition; preserve the rejected result
and revise the authorized repair window before another trial. Use
`joint_velocity.py` for the cheap speed-only check before expensive mesh review.

```bash
python tools/repair/transition.py --config case/transition.json --output case/transition_trial --cache case/audit_cache
python tools/repair/diagnostics.py --config case/diagnostics.json --candidate case/transition_trial/candidate_UNREVIEWED.npz --output case/motion_diagnostics.json
```

Both commands create new outputs and never select or overwrite a motion. Torch/Warp are
required only by the transition proposal worker. The diagnostics command needs
only the normal native runtime and does not construct collision query trees.

## Check scene collisions against the edit windows

With a packed or explicitly configured static scene, the command first audits
the current native seed on the configured frame/subframe grid, reusing its
exact-state cache. It writes `collision_scope_preflight.json` before seed fitting
or GPU work. This report binds the seed, effective edit mask, geometry and audit
code; an unbound historical collision summary cannot replace it.

For each detected scene crossing it lists the local limb controls, shared
ancestors, declared windows and interpolation support frames. A local joint
must be editable at a contributing native key. A waist that can move a frozen
arm indirectly does not count as local arm coverage. Protected endpoints and
common protected frames stay frozen. Having local control is necessary for
this repair, not proof of a nonzero gradient or a reachable collision-free pose.

`BLOCKED_EDIT_SCOPE` stops before solving and exports no candidate. Revise the
explicit windows/hinges, proposal coverage or starting checkpoint in response
to the report; it never unlocks joints automatically. For example, a late
left-hand grasp does not justify freezing the left arm through an earlier
hand/table collision. Scene pairs for editable geometry must contribute to the
proposal; entirely frozen geometry remains in the audit even when omitted from
the proposal. A whole-motion request also needs a full-duration baseline and
final audit: this preflight covers only the configured window and halo.

For an intentional diagnostic experiment, an explicit override is available:

```json
"collision_scope": {
  "mode": "diagnostic",
  "reason": "Measure a scoped change while preserving the reported frozen failures"
}
```

The default mode is `required`. A diagnostic override preserves all findings,
stays unselected and returns a nonzero CLI status even if later sampled checks
pass. `--proposal-only` skips the final candidate audit, not this scene-scope
preflight. Low-level Python calls are primitives; use this command for the
bound preflight, export and audit workflow.

## Free approach and wrist preservation

Use extra wrist waypoints/orientation targets only when the source or clearance
route warrants them. During free approach, retain soft source-posture guidance,
scene/self clearance and temporal objectives without automatically preserving
the previous robot wrist pose. Keep material contact constraints phase-bound;
this policy does not disable the upstream SOMA human-reference objectives.

By default, `run` seeds wrist position/orientation IK before refinement.
Setting `solver.weights.route: 0` disables the refinement route penalty but does
**not** skip that IK. For a directly authored native clearance key, use
`seed.native_input: true` to bypass waypoint IK and joint-blend seeding, then
supply `pose_keyframes` and the required routes/blends and joint windows. With
this flag, world-space waypoints do not author a new path; the explicit native
key does. To fit a new world-space waypoint first, use the normal seed path,
review its native pose, and pin those angles in the key edit. The native-input
route retains the normal preflight, export and audit. See the
[hand initialization guide](../../docs/hand-initialization-workflow.md).

## Author a route, then batch the short solve

`transition.json` points to an ordinary [repair config](README.md#configuration).
All its paths still resolve from that repair config's directory. Keep explicit
robot/object, robot/scene and self pairs in that config; packed scene geometry is
included by its existing audit rules. Declare object/scene pairs explicitly when
checking those as well. Keep contact normals disabled under the current policy.

```json
{
  "repair_config": "repair.json",
  "joint_windows": {
    "right_shoulder_pitch_joint": [0, 26],
    "right_shoulder_roll_joint": [0, 26],
    "right_shoulder_yaw_joint": [0, 26],
    "right_elbow_joint": [0, 26],
    "right_wrist_roll_joint": [0, 26],
    "right_wrist_pitch_joint": [0, 26],
    "right_wrist_yaw_joint": [0, 26],
    "right_hand_index_0_joint": [0, 26]
  },
  "routes": [{
    "body": "right_wrist_yaw_link",
    "joints": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
      "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
    "window": [0, 26],
    "waypoints": [{"frame": 8, "position_world": [0.4, -0.2, 1.1]}],
    "rotation_window": [8, 26],
    "orientation_radius_m": 0.1
  }],
  "joint_blends": [{"joints": ["right_hand_index_0_joint"], "window": [10, 26]}],
  "seed": {"max_evaluations": 55},
  "solver": {
    "device": "cuda:0", "steps": 300, "max_seconds": 120,
    "fit_substeps": 2, "samples": 96, "margin_m": 0.003,
    "allow_degenerate_subset": []
  }
}
```

These names/frames/coordinates illustrate the schema, **not a portable grasp or
table route**. Choose all arm/finger channels needed by the actual model. The
repair config's `editable_joints` must exactly match `joint_windows`; each channel
belongs to one route or blend. Each window preserves both endpoints and all
common protected frames. A right-hand window ending at frame 26 stays frozen
while a left-hand window can continue until frame 60. Shared waist joints can be
an explicit separate blend; two independent routes cannot overwrite the same
joint. Correction bounds remain relative to the fixed repair parent. An explicit
`iteration` seed keeps the common tool's binding and smoothness rules.

Read the original scene in world metres. Author a small route that clears the
relevant table edge before lifting/reaching, with allowance for the whole hand,
forearm and torso. Include robot-model reach and source evidence when choosing
the side. The tool does not infer a table plane or safe edge. `position_world`
is the named body's origin. Endpoint poses and velocities come from the native
checkpoint; interior positions use a cubic spline. Wrist orientation follows
shortest-path SO(3) interpolation with smooth timing over `rotation_window`.
`joint_blends` smoothly connect the named endpoint hinge values, useful for
preshape/closure or a shared waist transition. Splines and endpoint poses are
proposals and may overshoot; they are checked after fitting.

The seed fits native arm FK with source-posture and previous-frame regularizers.
Optional `seed.collision_pairs` can name **declared robot-geom pairs** such as
`["geom:55", "geom:39"]`. Their native convex distance witnesses guide the seed;
they do not replace original mesh collision checks. Seed settings also accept
`posture_weight`, `continuity_weight`, `clearance_m` and `clearance_weight`.

The batched CPU/CUDA stage jointly optimizes the declared coordinates against
original-surface scene/self clearance, fixed contact positions, route positions,
joint continuity and actual world wrist acceleration. Geometry is bound once;
both bodies of moving pairs receive derivatives. Frozen coordinates are copied
from the native checkpoint exactly, avoiding float32 changes to root/legs/holds.
Object trajectory, scale, scene and source timing remain fixed in this tool.

Optional solver settings:

- `proposal_pairs`: explicit subset of the repair config's pairs. All omissions
  are reported; the final audit retains the full configured scope. Without this
  setting, all configured pairs contribute to the proposal.
- `allow_degenerate_subset`: explicit mesh labels (for example `scene:000`) for
  mapped GPU query subsets. Original mesh arrays are preserved; negative audit
  coverage stays unknown where the source geometry is invalid.
- `learning_rate`, and `weights` containing any of `clearance`,
  `joint_acceleration`, `joint_velocity`, `posture`, `route`,
  `wrist_acceleration`, `joint_jerk`, `wrist_jerk`, `velocity_violation`,
  `contact`. Effective values are recorded. Contact normals
  are disabled; legacy angular objectives/gates must be revised explicitly first.

### Sampled derivative and spline options

Use `solver.derivative_units: "seconds"` for physical-time comparisons. It scales
the joint velocity/acceleration objectives by native timestamps; the
compatibility default is `"frames"`.
Recalibrate their weights when changing units. Joint jerk and FK wrist
acceleration/jerk use seconds in either mode. These losses are soft objectives.
The derivative gate uses consecutive source stencils and factorial-scaled
divided differences, equal to forward differences divided by `dt**order` on a
uniform grid. They describe sampled motion, not continuous derivatives of the
final LINEAR replay.

- `solver.spline_initializer` forwards `smoothing_lambda`, optional positive
  frame/sample `weights`, and `max_iterations` to the CPU weighted natural cubic
  smoothing proposal. Lambda has units seconds⁴. It approximates noisy samples
  under the same bounds and exact frozen mask before refinement; it does not
  retime motion or enforce contact. No parameter value is a universal default
  for a case. Its internal C2 spline loses that continuity when baked to native
  samples and replayed LINEAR.
- `solver.temporal_projection` requires `velocity_projection: true` and the
  repair config's native speed gate. Supply explicit
  `acceleration_limit_rad_s2` and `jerk_limit_rad_s3`; optional
  `joint_acceleration_limits_rad_s2` and `joint_jerk_limits_rad_s3` map actual
  hinge names to overrides. The OSQP-backed CPU proposal preserves frozen
  entries and native bounds; `preserve_endpoints` defaults to true. It runs on
  the initializer and final proposal, with intermediate projection controlled
  by `velocity_projection_interval`. `max_seconds` and `per_joint_seconds`
  bound projection work; an infeasible or unverified solve raises instead of
  exporting a partially projected result. Its timestamp-scaled derivative
  operators match the independent gate on uniform and irregular timelines.

When temporal projection is configured, the final independent derivative gate
checks every native hinge across the complete timeline, including frozen legs,
held phases and joins. A failure records `REJECTED_BY_DERIVATIVE_GATE`; improving
editable hands does not erase inherited frozen failures. Source/depth, contact,
original-mesh collision, native speed and visual checks retain their independent
scope. Neither successful projection nor an internally smooth spline selects a
motion or establishes a full-motion pass. If a declared per-leg policy retains
inherited motion while legs are frozen, report that scope explicitly; passing
those configured limits is not evidence that the legs were repaired.

`max_seconds` bounds the seed/refinement work after preflight using cooperative checks. It is
**not** the outer case deadline: setup, an in-flight solve, the final audit and
export require time too, and the baseline preflight consumes case time separately.
Use the existing case budget and stop search early enough to finish checks.
The default sequence performs the scene-scope preflight and one candidate audit
after the batch; exact unchanged rows are reused rather than rescanned inside
every optimization iteration. A confirmed original-triangle crossing stops the scan and rejects
the candidate immediately. The report lists all untested samples; partial
coverage never becomes a pass. Without a crossing, every configured sample is
checked. Geometry incompleteness remains unresolved and is not suppressed.
`--proposal-only` explicitly skips that audit, records `NOT_RUN`, and
returns exit code 2. A completed audit returns 0 only when its configured gates
pass; failed/incomplete audits return 2. All candidates remain unselected.

Outputs include `candidate_UNREVIEWED.npz` when a proposal was run, exact-input bindings, effective
settings, seed errors, solver loss history, audit results, changed frame IDs and
motion diagnostics. The historical candidate filename is retained for callers;
an audited failure sets both its stored status and report to `REJECTED_BY_AUDIT`.
A locally improved interval with inherited collisions is still rejected by the
overall configured audit. A diagnostic override cannot turn that into a pass.
A blocked preflight writes its actionable report without a candidate; a command
exception writes `error.json`. Inspect the existing
report and views before running another branch; a lower sampled loss is not a
reason to replace a better reviewed candidate.

Audit reports store original degenerate face IDs once in `geometry_issues`.
Each collision's `degenerate_face_refs` identifies the mesh catalog and exact
per-pose additions/removals. This preserves the evidence and coverage status
without duplicating large ID lists across pairs, frames, failures and caches.

## Report deviation and local peaks without rescanning geometry

A standalone `diagnostics.json` has `model_xml`, `trajectory_npz` and
`diagnostics`. `trajectory_npz` fixes the model/timeline and default baseline.
The same `diagnostics` block is accepted by the ordinary repair/audit config,
including its Blender worker. For example:

```json
{
  "model_xml": "model/robot.xml",
  "trajectory_npz": "raw_native.npz",
  "diagnostics": {
    "bodies": ["right_wrist_yaw_link", "left_wrist_yaw_link"],
    "human_reference": {
      "npz": "human_world_motion.npz",
      "positions_key": "joints_world", "names_key": "joint_names",
      "body_map": {"right_wrist_yaw_link": "RightHand", "left_wrist_yaw_link": "LeftHand"},
      "fps": 24
    },
    "review_thresholds": {"position_deviation_m": 0.2, "acceleration_peak_m_s2": 20}
  }
}
```

The example review thresholds are **optional diagnostics, not new acceptance
limits or universal defaults**. Their warnings identify the exact frame or
finite-difference support frames to inspect. `baseline_npz` optionally binds a
separate raw native reference. Human positions must be in world metres on the
same timeline, with an explicit name mapping; declare `timestamps_key` instead
of `fps` for actual stored timestamps. Timing mismatch is an error, not an
implicit resampling operation. Missing human data means no human RMS comparison.

The report includes world wrist position/orientation deviation from baseline,
velocity and acceleration RMS/peaks, worst frames, and positive absolute RMS
excess `max(0, robot RMS - human RMS)`. It covers the full integer sequence and
therefore includes editable/frozen joins. Derivatives use midpoint finite
differences and actual seconds. It does not introduce a ratio, change the current
RMS gate, add a peak gate, or trigger automatic retries. A lower RMS can coexist
with worse local peaks and large source deviations; inspect those warnings in
source and mounted views. Integer differences do not certify continuous motion
smoothness between LINEAR keys.

## Optional native velocity-constrained refinement

An optional top-level `pose_keyframes` list pins explicit native hinge poses:

```json
"pose_keyframes": [{
  "frame": 9,
  "window": [0, 26],
  "joints": {"right_wrist_pitch_joint": -0.5}
}]
```

Angles are absolute native radians; this example is syntax, not a grasp preset.
The frame must be strictly inside its window. An additive quintic correction
initializes the window, then the authored key is fixed during refinement and
temporal projection. Export checks that the key survives native serialization
exactly; unrepresentable parent precision fails explicitly. Overlapping windows
for the same hinge are rejected. Use a subsequent bounded edit to add another
key with an earlier reviewed key as its fixed window endpoint. Verify all
interpolated geometry and temporal joins; clear keys alone do not clear a path.

For an existing native trajectory, `seed.native_input: true` skips waypoint IK
and joint-blend seeding. Still declare every editable joint through the existing
routes/blends and joint windows; this does not expand the editing scope.

With configured native `temporal` velocity limits, `solver.velocity_projection:
true` projects the seed and final proposal into the joint-limit, correction-bound,
fixed-key and per-interval velocity constraints. An optional positive integer
`solver.velocity_projection_interval` also projects optimizer coordinates every
N steps (default 0). Infeasible corridors fail explicitly. The hard final speed
audit covers every native hinge, including frozen joints and editing boundaries.

`solver.route_reference` defaults to `"seed"`; `"parent"` instead targets the
original parent wrist positions when the route objective is enabled. This can
help preserve the path while redistributing articulation over time. It is not
a source-video observation or a guarantee of contact or table clearance.

Projection can alter contact and collision behavior. Keep its receipts, then
review fixed material contacts, original-mesh collisions, source/novel views and
temporal behavior before accepting any output. Speed feasibility alone does not
select a motion.
