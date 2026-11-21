# Native Stage 2 HOI refinement

`grail.agentic_kinematic.hoi_refinement` creates a new, **UNREVIEWED** human/object
checkpoint from explicitly selected native body-rotation keys and bounded local
contact repair. It supports
SOMA03 and its original model assets.
It runs native evaluation in a configured external Python environment, then
uses Blender for key authoring and source/oblique observation renders.

To inspect an existing Stage 2 result in its original furnished room without
editing it, use the [native Blender replay importer](hoi_blender_replay.md).

This is the Stage 2 refinement step after selecting the reconstruction and
reviewing contact timing. It does not run initial prediction, Stage 1 depth
fitting, contact inference, robot retargeting or a physics-based grasp solve.
See [fixed material contact authoring](fixed_contact_points.md) and the
[native SOMA contract](gem_x.md) for their prerequisites.

## Start from the selected Stage 2 result

The default refinement loop is:

1. Bind the selected Stage 2 checkpoint and its original model, source video,
   calibrated camera and original object mesh. Author dense contact phases from
   the video, including no-contact, unknown, release and material changes.
2. Select meaningful approach, palm-orientation, first-touch, support and release
   keys. Edit permitted native body rotations, then interpolate those corrections
   over the source motion and review their joins to establish a smooth working
   seed. Interpolation alone does not prove smooth native motion.
3. Propose a bounded repair over a short interval that has a diagnosed defect.
   Keep the authored dense regions fixed during that solve. Evaluate the result
   through the checkpoint's actual native model.
4. Review source and oblique views, regional contact and normals, whole original
   geometry, actual wrist motion, correction joins and feet. Review the exported
   source-rate sequence as well as useful native subframes.
5. If the result fails, revise the evidenced contact definition, mounted palm
   pose or neighboring path keys and repeat the local trial. Save a chosen
   checkpoint with its decision before extending the correction to the sequence.

Keep three states distinct: the **original source** provides lineage and source
fidelity; the **working seed** is the current edited/interpolated fallback; the
**candidate** is a new proposal. Compare a proposal with its working seed and
retain its deviation from the original source. A failed or unreviewed proposal
does not replace the seed. A repair iteration budget bounds one trial; it does
not resolve an outstanding correction or establish infeasibility.

Dense contact fields are independent human/object surface regions. The regional
objective does not assign exact vertex correspondences, infer contact phases or
prove that every colored vertex touches. Use the source to revise an incorrect
region or phase rather than fitting the same hypothesis harder. Frozen human
fingers can limit a feasible grasp; repair does not add arbitrary rig controls.

## What remains fixed

Use the original SOMA03 source and model. The permitted controls are local rotation corrections on its declared body joints.
Root rotation/translation, fingers, face/eyes, human shape and native scales,
original object geometry/scale/trajectory, camera, source frame count and fps
remain fixed. Native export checks unchanged canonical fields and noneditable
joints. It removes known stale derived caches from the new export and preserves
the original input.

See the explicit body-joint allowlist in [native.py](../grail/agentic_kinematic/hoi_refinement/native.py). A key’s quaternion represents a
**local native delta**, composed as `source_rotation * delta_rotation`.
Blender control locations are fixed markers, not IK handles.

## Configure a case

Run commands from the repository root. Paths inside a case JSON resolve relative
to that JSON file; use absolute paths for executables and model bundles when
sharing a case. `runtime.native_python` must have the dependencies required by
the selected native model and review worker. Blender itself needs no Torch,
SOMA, Warp or Newton installation.
The Python launching the CLI needs NumPy and, for native pickle input, joblib
and any libraries required to deserialize its values (for example Torch).
Using the same configured native interpreter for the CLI is supported.

This illustrative `case.json` starts with a SOMA03 source containing at least
21 frames. Replace the paths and source-selected frames with the actual case:

```json
{
  "schema_version": 1,
  "name": "reviewed_source_case",
  "input_native": "selected_stage2.pkl",
  "scene": "original_scene.npz",
  "plan": "plan.json",
  "model": {
    "model_type": "soma03",
    "backend": "native",
    "soma_runtime": "gem_x",
    "gem_x_soma_assets": "/absolute/external/models/gem-x/soma",
    "device": "cpu"
  },
  "runtime": {
    "native_python": "/absolute/native_environment/bin/python",
    "blender": "/absolute/blender/blender",
    "timeout_s": 180
  },
  "observation_frames": [10, 15, 20],
  "review_frames": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  "source_video": "source.mp4",
  "source_images": ["source_frame_015.png"],
  "evidence": {"sequence": "reviewed_source_case"},
  "render": true
}
```

`input_native` accepts a tagged canonical NPZ or an existing trusted GRAIL
`human_data`/`obj_data` checkpoint (`.pkl`, `.pickle` or `.joblib`). NPZ input
requires its own matching `model_type`. An untagged historical native checkpoint
uses the explicitly declared case model type; conflicting existing tags fail.
The canonical interchange includes `poses`, `trans`, `fps`, `obj_R`, `obj_t`
and `obj_scale`. SOMA03 additionally requires identity128 and 57 native scales;
Native scales are not an extra Blender mesh transform.

Before the first real-data edit, replay the unedited input and compare **vertices
as well as joints** with the selected Stage 2 model's output. Matching joint
centres alone does not establish matching surfaces.

For a fixed fitted SOMA identity, add `soma_identity_params_path` to `model`
and retain its paired identity manifest. The native loader validates its
binding to the checkpoint. Use the paired SOMA03 implementation/assets expected
by the bundle; a model tag alone does not establish geometry parity.

Use the bound identity and scales at each frame. Preserve their original values when evaluating subframes.

The original scene NPZ contains these required arrays:

| Field | Meaning |
| --- | --- |
| `object_vertices [V,3]` | Original unscaled object-local vertices, in metres |
| `object_faces [F,3]` | Original triangular face indices |
| `camera_matrix_world [4,4]` | Calibrated OpenCV camera-to-world transform |
| `camera_intrinsics [3,3]` | Pixel-space calibration with positive focal lengths and zero skew |
| `image_width`, `image_height` | Positive integer scalar image dimensions |
| `camera_convention` | Scalar string `opencv_camera_to_world` |

Object motion and scale come from the native checkpoint. Optional scene
`obj_R`, `obj_t` or `object_scale` must agree with those native values. The
Blender worker converts OpenCV camera axes to Blender camera axes explicitly.

## Author the bounded correction plan

Each window controls one allowed native joint. Its integer start/end frames
must lie on the original timeline; endpoint corrections are exactly zero.
Interior keys may use fractional source frames. Windows on the same joint
cannot overlap, and every delta's angle must remain below the declared bound.
Keys must be strictly sorted and include both endpoints.
`max_angle_rad` must be strictly between zero and pi; correction arcs crossing
the pi branch are rejected instead of introducing a rotation discontinuity.

Example `plan.json` for an explicitly selected SOMA body joint:

```json
{
  "schema_version": 1,
  "windows": [
    {
      "joint": 12,
      "start": 10,
      "end": 20,
      "max_angle_rad": 0.25,
      "keys": [
        {"frame": 10, "delta_rotvec": [0.0, 0.0, 0.0]},
        {"frame": 15, "delta_rotvec": [0.0, 0.0, 0.08]},
        {"frame": 20, "delta_rotvec": [0.0, 0.0, 0.0]}
      ]
    }
  ]
}
```

These numbers demonstrate the schema, not contact evidence or a recommended
anatomical correction. Choose joints, intervals and values from the actual
source and native model. `windows: []` makes a baseline observation round.

Corrections interpolate with quaternion SLERP on the original motion. Native
export preserves the integer source timeline. Blender renders replay of that
export, including requested fractional samples; `authored_vs_replay` in the
round receipt reports the sampled angular difference between authored
continuous corrections and source-rate export replay. A fractional authored
key is not a promise that the exported source-rate motion reproduces that
continuous key exactly.

## Run and continue an immutable round

```bash
python -m grail.agentic_kinematic.hoi_refinement run \
  --config /absolute/case/case.json \
  --output /absolute/case/round_001
```

The output directory must be new. The orchestrator binds source/model/scene
inputs and code hashes, evaluates the parent and candidate in the external
native worker on CPU, writes and reopens the native export, compares reconstructed
vertices/joints on the same CPU model, runs the available review checks, and asks an isolated CPU
Blender process to render, save and reopen a new authoring scene. Each worker
has a foreground timeout and saved logs. The CLI makes no remote model calls.

The round includes:

| Artifact | Purpose |
| --- | --- |
| `iteration.json` | Execution state, hashes, lineage and round-trip measurements |
| `candidate_native.pkl` or `.npz` | New export in the source's native family |
| `parent.npz`, `candidate.npz` | Parameters-only canonical comparison inputs |
| `plan.json`, `model.json`, `frames.json` | Exact authored plan and evaluation request |
| `before_geometry.npz`, `after_geometry.npz`, `reopened_geometry.npz` | Native geometry at the declared evaluation frames |
| `review.json` | Diagnostics and explicit missing-evidence statuses |
| `blender/UNREVIEWED_authoring.blend` | Saved keys, source bindings and native observation snapshots |
| `blender/report.json`, rendered PNGs and worker logs | Blender transport/render evidence |

Repair rounds additionally retain `original.npz`, `seed_plan.json`,
`repair/report.json`, the proposed `repair/proposal_plan.json` and
`iteration_review.json`. Their `parent.npz` is the evaluated working seed;
`candidate.npz` is the repaired proposal. These states do not overwrite the
original input or a prior round.

All plan keys are added to the evaluated frame batch. `review_frames` must
include `observation_frames`; only observation frames produce the standard
source/oblique before/after renders. Sparse samples do not establish full
sequence quality.

To continue a saved, edited scene:

```bash
python -m grail.agentic_kinematic.hoi_refinement run \
  --config /absolute/case/case.json \
  --input-blend /absolute/case/edited_round_001.blend \
  --output /absolute/case/round_002
```

The saved scene supplies actual authored control values and the previous
candidate's bound native bytes. The previous candidate is the **before**
comparison. The cumulative plan is applied to the **original case input**, so
continuation does not accidentally accumulate the same correction twice.
Source/model/scene bindings must still match; frozen native parameters cannot
be redefined through a saved scene. Existing round directories and the input
`.blend` remain unchanged.

Scripted edits use the same path. Supply `--operations /absolute/operations.json`
with a JSON array of `set_key` or `delete_key` operations. For example:

```json
[
  {"op": "set_key", "window": 0, "frame": 15, "delta_rotvec": [0.0, 0.0, 0.10]}
]
```

An operation can edit an interior key or add one within an existing window;
it cannot change a protected endpoint or introduce an undeclared window.
Scripted native parameters and the exported/reloaded native motion remain
authoritative. No display-mesh edits are converted into native motion.

## Propose a short-window repair

After authoring and interpolating the working seed, use the same case with a
`repair` block. For example:

```json
"repair": {
  "contact_map": "reviewed_dense_contact_map.json",
  "objective": {
    "partial_mass_fraction": 0.25,
    "activation_ramp_frames": 1,
    "phase_blend_frames": 3
  },
  "max_nfev": 20,
  "max_delta_rad": 0.08,
  "finite_difference_rad": 0.001,
  "max_time_s": 120
}
```

These solver settings illustrate the schema; select bounds and phase blending
for the actual motion. `contact_map` resolves relative to the case file. Repair
requires the actual `source_video` and a compatible, video-reviewed
`grail.contact_map.v1` map. `objective` accepts the regional loss settings from
[`PhaseContactMapLoss.from_file`](../grail/optimization/phase_contact_map_loss.py),
including coverage, temperature, gap and normal controls. Geometry, source and
model bindings come from the case; do not override them through this block.
Keep the source-derived map immutable during a solve, then version any revised
contact hypothesis for the next round.

```bash
python -m grail.agentic_kinematic.hoi_refinement repair \
  --config /absolute/case/case.json \
  --input-blend /absolute/case/reviewed_seed.blend \
  --output /absolute/case/repair_001
```

`--input-blend` is optional; without it, the case plan defines the seed. The
reversible Blender preview requires a saved scene as the seed. The output
directory must be new. Repair changes only interior rotation keys of
already declared windows. It preserves the window bounds, unchanged endpoints,
per-joint angle bounds and original source timeline. Use short source-supported
windows to delimit the repair scope; adding a new path key or changing the
contact definition remains an explicit authoring step.

The solver evaluates dense regional contact through the original native human
model. It uses finite differences over the bounded key controls, so runtime
depends on key count and native model cost. This is an external proposal step,
not instantaneous IK. The optional `weights` object controls `contact`,
`posture`, `temporal`, `source` and `feet` terms. The `source` term penalizes
original native joint projection displacement normalized by image diagonal; it
is not an observed RGB keypoint measurement. The temporal term penalizes
seed-relative native joint displacement acceleration and jerk in source-frame
units. The independent review guards below measure actual motion in physical
units using source FPS. Stage 2 depth remains outside the repair objective.
Original-mesh collisions are audited after evaluation and are **not a solver
term** in this implementation.

`max_nfev` counts scalar objective calls, including finite-difference probes;
two original/seed geometry evaluations are reported separately. `max_time_s`
is checked between calls, while `runtime.timeout_s` bounds the external worker.
An exhausted inner budget retains the best evaluated feasible proposal, which
may be unchanged from the seed and remains unreviewed.

Optional `repair.foot_joint_ids` supply model-specific joints for the foot term.
Use them only when the selected windows have source-supported foot stance;
explicit IDs do not turn a moving foot into a valid fixed support. Independently
set `iteration.foot_joint_ids` for foot review and `iteration.wrist_joint_ids`
for actual wrist motion review. IDs must come from the selected native model;
there is no universal wrist or foot numbering. Omitting wrist IDs
leaves that temporal evidence `NOT_PERFORMED` and blocks application, while
keeping diagnostic preview available. Foot review measures displacement from
the seed and a 3D joint-speed proxy; it does not establish ground contact or
physical slip.

Iteration guards compare the candidate with the working seed using SO(3) joint
rotation changes, actual wrist speed/acceleration/jerk and velocity changes at
window joins. Configurable `iteration.relative_tolerance` and
`iteration.absolute_tolerance` dictionaries use named metrics such as
`wrist_acceleration_m_s2` and `joint_boundary_velocity_jump_rad_s`.
`iteration.preserve_smoothness` defaults to `true`; default tolerances permit
only the implementation's numerical allowance. Declare a tolerance change as
part of the trial rather than relaxing a failed guard after selection.
Disabling this check records `NOT_PERFORMED` and cannot earn an application
guard pass.

Review includes native integer and half-frame replay within the edited windows
and a three-frame halo around them. This finite audit is not continuous collision
or dynamics validation. Application requires complete iteration guards and no
detected original-mesh crossings in the candidate audit. A guard pass still
leaves the motion **UNREVIEWED**. Preserve the original source, working seed,
proposal artifacts and any failed checks; only an explicit checkpoint decision
selects a reviewed result. Continue the contact/palm/path loop when an in-scope
defect remains after one bounded solve.

## Use the Blender sidebar

Build/install the HOI extension as described in
[the Blender extension README](../extensions/blender/README.md). In the
**GRAIL HOI** tab, use **Stage 2 SOMA Native Round**.

1. Open a generated Stage 2 `.blend`, then click **Enable Verified Native
   Snapshots**. A new round can also be launched from an empty Blender document
   by setting the case, external Python, repository and new round directory.
2. Select a `GRAILStage2_Key_...` control, click **Jump to Source Key**, and edit
   its **Native delta WXYZ** quaternion. These are independent static key
   controls; Blender F-curves do not define the native interpolation.
3. **Source frame 0 = Blender frame 1**, including the corresponding offset for
   subframes.
4. Changed keys or uncached frames make the displayed candidate stale and hide
   it. Camera changes, native mesh transforms, altered topology and original
   object-mesh edits are rejected or hidden. Saved observation meshes are
   caches, never a substitute for evaluating edited native parameters.
5. Set **Native case JSON**, **External GRAIL Python**, **GRAIL repository** and
   a **New round directory**, then click **Refresh via External Native
   Evaluator**. The operator saves a separate input scene copy and invokes the
   same CLI in the foreground. It records the refreshed scene path without
   silently replacing the current document.
6. Save the current document before **Open Refreshed Scene**, then enable its
   snapshots and inspect the generated images. **Export Actual Authored Plan**
   writes a new JSON plan only; it does not evaluate or accept a candidate.

For local fitting, **Propose Local Repair** evaluates a separate candidate from
the saved seed. **Preview Repair** opens the proposal for inspection;
**Restore Seed** returns to the saved seed. **Apply Repair as Unreviewed** makes
an explicit guarded transition to the repaired working state. It requires the
iteration and original-mesh checks above and never marks the motion accepted.
Application saves a new `APPLIED_UNREVIEWED.blend`; rejected previews and the
return-to-seed scene are also preserved. Editing keys after a proposal requires
a fresh evaluation before application. Keep saved scenes immutable and save a
new scene for actual subsequent edits. Proposal failure leaves the seed
available for the next contact, palm or path revision.

Opening a scene does not execute embedded source scripts. Snapshot playback
must be explicitly enabled through the installed extension after each open.
Original native meshes are visible only at evaluated observation/cache frames
with matching controls. This is deliberately an explicit refresh workflow,
not an in-Blender Torch model or a live IK solver.

## Review and optional Newton diagnostics

Every successful export retains `motion_status: "UNREVIEWED"` and
`selected: false`. `EXPORTED` is an execution result, not acceptance.
The report compares native geometry, fixed material contacts when supplied,
original-mesh intersection samples, source projection, fixed-ray depth when
supplied, and temporal finite differences where sampling permits them.
Missing optional evidence is recorded as `NOT_PERFORMED`.

An actual `source_video` file is required when the case supplies contact review,
fixed-depth evidence or a source-video hash; the orchestrator binds the file
itself. Merely declaring a hash cannot stand in for that input. The current
review measures projected before/after displacement, not observed RGB keypoint
error. Supplying `evidence.source_keypoints` is explicitly rejected.

Use `contact_review` for the independently authored material-contact review
JSON described in [fixed_contact_points.md](fixed_contact_points.md). Its
sequence/source-video/native-topology bindings must match. A contact-distance
reduction, a matching model tag or successful save/reopen does not establish
correct contact. Source projection changes alone are not evidence of improved
source alignment. Review intended material points and normals, contact timing,
source/depth agreement, original geometry and temporal behavior independently.
Retain unresolved/occluded intervals and separate inherited defects from new
regressions. Save the explicit selection decision and scope separately.

When reviewed material pairs are available, source/oblique renders include
their world-space points and normals as diagnostic overlays. The addon does
not currently provide a separate canonical-hand inspection panel. Optional
`evidence.fixed_depth` uses a frozen, hash-bound ray NPZ and the declared
background-ground-truth calibration; the review reports errors together with
missing ray coverage. Its exact fields are documented in
[`review.py`](../grail/agentic_kinematic/hoi_refinement/review.py).

Optional Newton queries are enabled by adding a case block such as:

```json
"newton": {
  "python": "/absolute/newton_environment/bin/python",
  "repo": "/absolute/pinned_ho_contact_checkout",
  "device": "cpu",
  "timeout_s": 120,
  "frames": [15]
}
```

The supplied frames must belong to the evaluated native batch. The query runs
outside Blender against explicit original human/object mesh snapshots and
records source/dependency provenance. The adapter requires the pinned
`ho_contact` and Newton revisions declared in
[`newton.py`](../grail/agentic_kinematic/hoi_refinement/newton.py); it does not
install dependencies or silently switch backends. Results are diagnostic
paired surface points with **unsigned point distances**, not signed penetration
depth. Neither zero reported contacts nor an intersection check certifies
clearance, self-collision freedom, continuous-time behavior, force, support,
stability or grasp success. No physical validation is performed.

The external Newton client/worker was adapted from `agentic-retargeter` revision
`d7c0ef38f5390170c2c04e7768f7057dbc7ec9bf` (`src/collision.py` and
`src/collision_backend.py`). GRAIL uses its own package imports and native
human/object snapshot adapter, so the worker runs independently of another
checkout's mutable files. The worker's local adapter uses a distinct
module name so it cannot shadow the upstream `newton` Python package.

## Tooling checks

To check the Blender extensions, follow [Rebuild and verify](../extensions/blender/README.md#rebuild-and-verify).
The smoke test installs both ZIPs in an isolated profile and checks that they can be enabled and unregistered together.
Verify native model parity and reconstruction quality separately on the intended case.
