# SOMA retargeter tool

This is a thin local entry point around the official [NVIDIA SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter). Implementation lives directly in [`src/soma.py`](../../src/soma.py). Install the upstream repository and its locked Python environment separately using its [installation instructions](https://github.com/NVIDIA/soma-retargeter/blob/main/documentation/installation.md); model assets and heavy runtime dependencies are not bundled in Agentic Retargeter.

The input is a **SOMA-base skeleton BVH**, not arbitrary BVH, SMPL-X, or a raw SOMA parameter NPZ. Export parameter sequences with their original SOMA model/skeleton first; renaming a file or treating axis angles as BVH Euler channels is not a conversion. SMPL-X uses the separate [GMR tool](../gmr/README.md).

Create a JSON configuration with paths relative to the config file or absolute paths:

```json
{
  "soma_root": "/path/to/soma-retargeter",
  "python": "/path/to/soma-retargeter/.venv/bin/python",
  "source": "/path/to/source_soma.bvh",
  "source_units": "cm",
  "source_facing_direction": "Mujoco",
  "robot": "unitree_g1",
  "human_scale_ratio": 1.0,
  "device": "cuda:0",
  "bindings": {
    "human_model": "/path/to/original_human_model_manifest.json",
    "object": "/path/to/original_object_trajectory.npz",
    "camera": "/path/to/original_camera.json"
  }
}
```

`bindings` is optional evidence retention. These files are hashed and copied unchanged; they are not inputs to the upstream solver. Omit unavailable evidence rather than inventing an object trajectory. Targets require local MJCF or URDF descriptions, loaded by the official Newton builder. Optional `extra_robot_paths` registers explicit custom robot directories in both preflight and conversion; these paths resolve relative to the config. Duplicate names, overrides of bundled targets and remote model downloads are rejected. URDF targets produce the official CSV and a `model_urdf` handoff; the existing MJCF native exporter and contact-refinement tools do not consume that handoff yet.

`human_scale_ratio` defaults to **1.0** for all targets, including G1, H2, T1,
X2 Ultra and A3. These workflows use embodiment-matched, metric SOMA sources;
do not apply the upstream generic-human size ratio again. A positive finite
override belongs in this config (GRAIL YAML: `backend.human_scale_ratio`). The
runtime bridge sets the scalar before constructing IK targets and foot-processing
scales, without editing the external checkout or robot assets. It records both
the upstream and effective values in `scaling.json`; a missing or inconsistent
receipt fails conversion. Joint calibration offsets and solver settings remain
unchanged. Scale 1.0 prevents additional global position/size scaling, but does
not establish correct joint correspondence or hand-object contact.

Run from this repository:

```bash
python tools/soma_retargeter/run.py plan --config soma.json --output /path/to/new_candidate
python tools/soma_retargeter/run.py preflight --config soma.json --output /path/to/new_candidate
python tools/soma_retargeter/run.py run --config soma.json --output /path/to/new_candidate
```

`plan` only reads and validates files. `preflight` invokes the configured upstream Python to check CLI support, package resolution, actual model path, runtime versions and joint order. `run` creates a new output directory, copies exactly one BVH into an isolated input folder and executes the official converter in the foreground with `--viewer null`. It rejects changed inputs, missing Git LFS assets, invalid skeleton hierarchy, nonfinite/truncated motion, unexpected CSV columns, different frame counts and false-success exits. Existing outputs are never reused.

The output contains `raw/source.csv`, `input/source.bvh`, `converter.json`, `conversion.log`, retained `evidence/`, and `run.json`. The report binds original source, robot/SOMA assets, runtime code, input and generated configuration, dependencies and all output hashes. Its `native_handoff` object provides the exact inputs for the native CSV-to-NPZ converter. Every result remains `CONVERTED_UNREVIEWED` until motion review.

## Coordinate and timing contract

- Declare `source_units: "cm"`: the upstream BVH loader always converts centimetres to metres. Metre-valued BVH requires an explicit upstream export; this tool does not guess scale.
- Declare `source_facing_direction` as `Mujoco` or `Maya`, matching the upstream `SpaceConverter` setting for your export. `Mujoco` applies a +90 degree X rotation before solving. It does **not** mean that the source file is already in the final robot world coordinates. Object and camera transforms require a separately verified shared-world mapping.
- Source frame count and raw BVH frame time are recorded. Effective FPS follows the official importer: `round(1 / frame_time, 2)`. Optional config `fps` must agree. CSV has no embedded FPS; never use its loader's default 120 Hz for another source rate.
- CSV root translation is in centimetres. Root orientation is extrinsic XYZ Euler degrees; actuated joint values are degrees. Actual column names/order come from the configured Newton robot builder. Native conversion maps by joint name and converts to metres, radians and root quaternion explicitly.
- The bundled `unitree_g1` target has **29 body/wrist DOFs and no actuated finger joints**. Its locomotion/foot contact processing is not hand-object grasp prediction or contact refinement. Finger mapping and grasp repair require an explicitly bound hand-capable robot and later native Blender editing.

After initial conversion, use the shared source keyframe/contact planning and native Blender review workflow. Do not transplant this body's CSV onto a differently configured G1 with hands by padding unknown joints or changing its model label.

## G1 + Dex3 through the SOMA solver

The custom `unitree_g1_dex3` target uses the original G1-with-hands MJCF, its
43 scalar joints (14 fingers) and unchanged joint limits. It adds 20 native SOMA
hand segment/tip tasks to the official body task table and runs the same Newton
solver. It does not call GMR or attach finger animation to an existing body CSV.
Native export maps all CSV columns by joint name and contains 50 qpos values.

Create a target bundle with the official SOMA runtime:

```bash
/path/to/soma-retargeter/.venv/bin/python src/soma_target.py \
  --soma-root /path/to/soma-retargeter \
  --model-xml /path/to/original/g1_mocap_29dof_with_hands.xml \
  --source-bvh /cases/example/source_export/source.bvh \
  --source-rest /cases/example/source_export/native_rest.npz \
  --output /cases/example/targets/g1_dex3
```

The verified GRAIL SOMA export manifest must accompany the BVH and bind its
`native_rest.npz`. The rest file is the actual native zero-pose FK with the
source identity and scale, including rotations in the declared coordinate
frame. BVH `OFFSET` translations alone do not retain neutral joint rotations.
The generator copies original model assets byte-for-byte, calibrates hand
offsets relative to each wrist, and records the model/source/code hashes.
The current generator requires metric source scale 1.0 for already
G1-proportioned sources; it does not fit or resize a general human identity.

Set these fields in the normal SOMA config:

```json
{
  "robot": "unitree_g1_dex3",
  "extra_robot_paths": ["./targets/g1_dex3"]
}
```

Thumb maps to the native opposing thumb. Human index and middle are explicit
representatives for the two long robot digits; ring and little fingers have no
independent target DOFs. Original left/right hinge signs are retained. Bind the
original object trajectory separately. Results remain unreviewed until mounted
contact, whole original meshes, source alignment and temporal behavior are
checked; finger motion from IK is not a reviewed grasp.

## Optional hand-object constraints inside SOMA IK

Set `"contact_constraints": "contacts.json"` in the usual SOMA config to add
authored material point objectives **to the same Newton IK solve** as
the human body/finger objectives. The default route is unchanged. This path uses
an isolated subprocess bridge in `src/soma_contact_runtime.py`; it does not edit
the external SOMA checkout or run a separate post-retarget finger fit. It supports
CPU and CUDA, including device-side phase weights during CUDA graph replay.

For the current SOMA agentic initializer, use the
[synchronous trajectory mode](../../docs/soma-synchronous-sequence.md) with its
default CUDA forward / CUDA Graph evaluation / CPU sparse global solve backend.
The forward LM adapter avoids residual-count-dependent shared-memory tiles and
supports up to 64 native tangent DOFs. An explicit converter `device` overrides
the forward default; a `serial` synchronous evaluator defaults forward to CPU.
Unsupported devices or runtime contracts fail explicitly. First-process
compilation and graph capture can dominate a cold run, so compare both stage
and complete-worker times. Backend validation does not accept motion quality.
When the entire original object translation and rotation are exactly constant,
the bridge omits both hand-object position and normal contact losses in both
stages. Configured object/scene/self collision, human tracking, joint limits and
temporal losses remain active. The report records `contact_loss_policy` and zero
effective contact objective counts; authored anchors remain diagnostic evidence.
Static cases may use `contacts: []`. Moving objects, including an initially
stationary prefix, retain the configured contact losses.

Example `contacts.json` (paths relative to this file):

```json
{
  "schema_version": 1,
  "world_frame": "soma_retargeter_z_up_m",
  "post_processing": "disabled",
  "object_mesh_npz": "original_object.npz",
  "object_trajectory": "object_motion.npz",
  "contacts": [{
    "name": "right_thumb_hold",
    "body_name": "right_hand_thumb_2_link",
    "point_local": [0.0075, 0.0341, 0.0],
    "normal_local": [1.0, 0.0, 0.0],
    "object_face_id": 123,
    "object_barycentric": [0.2, 0.3, 0.5],
    "object_normal_sign": 1,
    "frames": [26, 121],
    "position_weight": 400.0,
    "normal_weight": 0.0,
    "ramp_frames": 0
  }]
}
```

These illustrative pad/face values must be replaced with the actual robot's
reviewed contact descriptor. This tool does not infer robot contacts from human
vertex labels. Review source-supported hand-surface choice, grasp topology, reachable
spacing and palm orientation at the initial and difficult later keys first.
Unknown or sliding contacts need their own explicitly authored phases, not a
fixed anchor chosen independently on every frame.

The mesh NPZ contains original ordered `vertices[N,3]` and triangle `faces[M,3]`.
The trajectory NPZ contains `obj_R[T,3,3]`, `obj_t[T,3]`, positive scalar/XYZ
`object_scale`, scalar `fps`, and `timestamps[T]` starting at zero. Its timing must
match the BVH exactly and its transforms must already use the solver's world
frame. Geometry, trajectory, contact specification and bridge code are hashed in
the run report. Scale is applied before deriving material points and normals;
object transforms remain fixed inputs unless the explicit experimental
`object_optimization` mode below is enabled alongside temporal optimization.

`frames` uses zero-based, half-open source intervals. Contacts are inactive
during solver warmup and outside those intervals. Optional `ramp_frames` applies
a raised-cosine gain at **both** ends, wholly inside the interval; leave it zero
for a hold that persists to the clip boundary. Use separate explicit phases for
other onset/release schedules. The objective adds
`gain * (position_weight * gap_m² + normal_weight * ||pad_normal + object_normal||²)`.
These are soft penalties with different units, not guaranteed hard constraints.
`normal_weight` is optional and defaults to zero. The current workflow does not
fit or accept/reject by contact-normal alignment. Explicit nonzero legacy values
remain supported; revise them in a new config when adopting the current policy.
Normal metadata remains required for diagnostics; original surface normals used
by collision avoidance are unaffected. A nonzero normal term, if explicitly
requested, leaves rotation about the contact normal unconstrained.

Contact mode explicitly requires `post_processing: "disabled"`: upstream's
limb stabilizer can otherwise overwrite the newly fitted wrists. This disables
its foot postprocessing too; ordinary body/foot IK targets and native joint limits
remain. Root, legs and waist participate in the initial full-body solve; this
mode does not promise frozen lower-body motion. A no-contact comparison should
use the same disabled postprocessing setting to isolate contact effects.

Outputs additionally include `contact_targets.npz` and
`contact_initialization.json`, with per-active-frame gaps and normal errors
measured **after** upstream's final joint-limit clamp. They are initialization
diagnostics, not motion acceptance. Collision avoidance is an additional explicit
option below. Inspect full original geometry, source alignment, derivatives and
transitions before using the result for subsequent Blender refinement.

### Candidate-first initialization

For a requested contact-aware initial conversion, prepare the original human
BVH, robot geometry, object regions/trajectory and support scene first. Author
and screen each participating hand's contact layout before supplying
`contact_constraints` to the SOMA run. Free-wrist screening establishes only
that geometric hypothesis; mounted reach and full-motion review remain separate.
A source-pose preview can seed candidate construction when allowed, but record
its lineage and do not use an earlier repaired trajectory as an undeclared
initializer. Report unavailable/uncertain hand phases explicitly.

Pass the resulting fixed material pairs and `collision_avoidance` together in
the contact file. Keep the original human tracking objectives active. Check
`same_soma_ik_solve`, `active_in_same_solve`, post-clamp observations and bound
static geometry in the output; a plain successful conversion is insufficient.
Object motion remains fixed by default; separate permission for object edits
does not activate them. Configure the experimental object-motion option below
explicitly when those variables belong in this converter run.

For an objective comparison, use the same source and target configuration with
`enable_post_processing: false` in a new, bound target retarget JSON for both
runs. Contact mode disables that postpass; leaving it enabled in the control
confounds wrist and foot comparisons. Review source-target fit, root/feet,
contact and collision changes independently. Zero-weight normal objectives do
not allocate solver residual rows; all material position anchors remain active.

The upstream `smooth_joint_filter` shapes joint-limit costs; it is not a
frame-to-frame smoothness objective. Previous-frame initialization likewise
does not bound velocity, acceleration or jerk. Contact activation ramps can
soften acquisition, but the initialized sequence still needs temporal review
and, when necessary, a coupled trajectory solve with the same contacts and
clearance constraints. Do not describe this per-frame initializer as a
temporally optimized final motion.

### Optional temporal optimization inside the SOMA converter

With the explicit contact bridge enabled, `temporal_optimization` in the SOMA
run config adds time-normalized hinge velocity, acceleration and jerk costs:

```json
"temporal_optimization": {
  "sweeps": 4,
  "velocity_weight": 0.002,
  "acceleration_weight": 0.0001,
  "jerk_weight": 0.0000001,
  "max_seconds": 720
}
```

These example coefficients are a trial configuration, not universal acceptance
thresholds. They multiply sums of squared derivatives in rad/s, rad/s² and
rad/s³, calculated from the bound BVH FPS. The sequence needs at least four
source frames. Ordinary conversion without this option retains its behavior.

The converter first computes a fresh SOMA initialization, then alternates
forward/backward visits over the source frames before CSV export. Every visit
uses the **same original human position/orientation objectives, phase-specific
material contacts, configured collision terms and native limit objectives**.
With neighboring frames held fixed, all temporal difference stencils touching
the current frame collapse to one exact conditional quadratic per revolute
hinge. The compact analytic residual keeps the Newton solve small. This is
block-coordinate optimization of a sequence objective; it is neither an output
filter nor the separate `transition_repair` Adam optimizer. It does not replace
human targets with robot joint-angle priors.

`contact_initialization.json` records settings, target identity, completed
sweeps and before/after derivative metrics. `temporal_sequence_states.npz`
retains the fresh initial and refined **Newton-coordinate** sequences for
comparison; these arrays are not native MuJoCo exports. Incomplete sweeps are
discarded at the cooperative temporal deadline. A run with no completed sweep
fails rather than reporting temporal optimization. The configured deadline
bounds the sweep stage, not the initial conversion or complete case.

The current temporal residual covers scalar revolute joints only. The free
root remains in the full-body source/contact/collision solve but has no
temporal penalty. Costs are soft: finite sweeps need not converge or satisfy
derivative caps, contact equality, foot support or continuous clearance.
Evaluate the exported native motion, world-space wrist/root motion, original
human-target error and full meshes before choosing a candidate. There is no
subsequent joint-only hard projection that could silently break the contacts.

### Optional object motion inside the SOMA converter

**Experimental; case validation is still required.** With authored contact
constraints and `temporal_optimization` enabled, the top-level
`object_optimization` block permits one shared object path to change during the
converter's sequence sweeps. It optimizes one shared precontact SE(3) pose,
then alternates post-contact object SE(3) updates with robot updates retaining the original human position/orientation, contact and
collision objectives. The object update is a separate numerical block inside
the converter, not a new articulated body in the Newton robot model. Ordinary
conversion and temporal-only conversion retain their fixed-object behavior.

The contact boundary and numerical objective settings below are explicit. The
reference pose and diagnostic tolerance have defaults; hard limits are optional. Original-path position/orientation penalties are required; the default
has no hard displacement or correction-rate caps. Weights are case settings:

| Fields | Contract |
|---|---|
| `contact_frame`, `stationary_reference_frame` | Zero-based source indices. `[0, contact_frame)` shares one optimized position **and** rotation. The entire prefix can move/rotate together, but its samples cannot drift apart. Independent per-frame object updates start at `contact_frame`, including frame 0. `stationary_reference_frame` seeds the shared pose; it does not pin its final value. It defaults to 0 and must lie before contact (0 when contact starts at 0). Legacy `first_edit_frame` is accepted but normalized to `contact_frame`. |
| `contact_tolerance_m` | Optional, defaults to 0.015 m. Report contact gaps against this tolerance; it never unlocks updates, discards a completed sweep, or blocks export. A low gap alone does not establish grasp quality. |
| `max_translation_m`, `max_rotation_rad` | **Optional**, absent by default. When supplied, hard translation-norm and rotation-angle caps from the original path at matching timestamps, including the stationary correction. |
| `max_translation_offset_rate_m_s`, `max_rotation_offset_rate_rad_s` | **Optional**, absent by default. When supplied, hard rates of neighboring translation corrections and rotation-vector corrections, using source FPS. These bound corrections, not absolute object speed. |
| `reference_translation_weight`, `reference_rotation_weight` | Required positive soft penalties for departing from the immutable original translation/orientation. They do not guarantee a displacement bound. |
| `translation_velocity_weight`, `translation_acceleration_weight`, `translation_jerk_weight` | Nonnegative physical-time costs on the actual object translation trajectory, including stencils crossing the stationary/editable join. |
| `rotation_velocity_weight`, `rotation_acceleration_weight`, `rotation_jerk_weight` | Nonnegative physical-time costs on actual rotation-matrix differences. These are smoothness surrogates, not exact SO(3) angular acceleration/jerk or hard derivative caps. |
| `max_iterations` | Positive SLSQP iteration limit for each object update. It does not establish convergence or acceptance. |

Object policy is resolved in Python from the complete immutable source trajectory,
without an agent decision. Exactly constant translation **and** rotation select
`static_passthrough`: every original object pose is reused, no object optimizer
is called, and export records `object_motion_optimized: false`. Any translation
or rotation change, including rotation-only motion, selects
`joint_with_shared_precontact`, with `precontact_pose_policy: shared_optimized_pose`.
No implicit static-motion tolerance is applied.

For moving sources, the first sequence sweep already optimizes the common
precontact pose and alternates object/robot updates. Every frame from
`contact_frame` onward is independently eligible, including frames with zero
contact gain, where reference, temporal and collision terms still apply. The
shared prefix block uses its original-reference, enabled contact/collision and
temporal costs, including stencils crossing the contact boundary. There is no
robot-only contact warmup or contact-quality unlock. For example,
`contact_frame: 27` lets frames 0–26 translate/rotate together as one stationary
pose, then permits independent trajectory updates from frame 27 onward.
Historical pinned-prefix trials retain their archived configuration/results;
new plans use this shared variable and normalize the legacy delayed-edit field
to the contact boundary.

The ordinary `plan` / `run` CLI commands above apply this policy whenever
`object_optimization` is configured; no model call or agentic workflow is needed.
The plan and result record the resolved `object_motion_policy`. A moving-source
result requires evidence of at least one complete object-enabled sweep, with
a shared-prefix update when that interval is nonempty and updates covering the
entire post-contact suffix; static passthrough requires a complete robot sweep
and zero object updates. Failed
contact diagnostics remain visible in both results without suppressing export.

Before solving, distinguish the acquisition ramp, effective contact and pickup.
The precontact prefix is one adjustable pose, not per-frame copies of a
potentially drifting source and not a pose pinned to its original value. Sharing
that variable guarantees exact stationarity without a soft stationarity weight;
a finite soft velocity/rotation cost alone would only discourage drift. Positive
soft reference costs apply to every prefix sample and permit a changed common
pose. Report that change against the unchanged original path; if an explicitly
configured cap is violated, fail instead of silently widening it. Keep the reference fixed across
sweeps and report peak/RMS translation/rotation deviations for quality review.
Soft reference costs alone do not guarantee that the result stays sufficiently
close. Contact errors are measured after each sweep and on final export for
review, without gating initial-result delivery. Finite poses, exact stationary
prefixes (constancy to the optimized common pose), immutable references and any
explicit caps remain hard checks. Export records maximum precontact translation
and rotation variation separately from deviation against the original path.

Each object update refreshes both hands' paired material targets and the
collision object transforms together. With a bound support scene, include
object/scene pairs in the object-update collision cost as well as the existing
robot/object and robot/scene checks. Sampled penalties do not enforce physical
support or exact nonpenetration. Check the full original object/table geometry,
finite-table support during resting phases, both-hand gaps/slip, original-path
deviation and the stationary-to-moving join independently. A small offset rate
alone does not smooth the source path; report actual translation and SO(3)
angular derivatives, alongside offset derivatives and robot/wrist behavior.

The optimized object trajectory is a bound output of the run. Native export
must validate the supplied original object sidecar against that binding, then
consume the report-bound optimized transforms while preserving original scene,
scale and timing. Do not export the old sidecar's transforms with the newly
solved robot motion, rebase the reference to a corrected output, or apply the
correction a second time in Blender. Keep initial and final object paths and
their hashes with the robot comparison; no solver status selects the motion.

### Collision avoidance during contact initialization

Add `collision_avoidance` alongside `contacts` in `contacts.json` to include
sampled clearance penalties **in the same SOMA/Newton solve**. Use this when
fitting hand contacts; leave the section absent to reproduce contact-only IK.
It runs on the configured CPU/CUDA device, without a Torch dependency.

```json
"collision_avoidance": {
  "robot_geoms": [53, 55, 57, 59, 61, 63, 65, 67, 69],
  "self_pairs": [[61, 65], [61, 69], [65, 69]],
  "samples_per_mesh": 128,
  "margin_m": 0.001,
  "weight": 10000.0
}
```

These geom IDs are illustrative, **not a complete hand/body collision preset**.
Enumerate the original model's visual meshes and explicitly include the fingers,
palm and wrist. Every listed robot mesh is checked against the object; only
listed `self_pairs` participate in self-collision avoidance. Add arm/body and
other-hand pairs where needed. The report records included and omitted mesh/pair
IDs and open/nonmanifold topology. Same-body or adjacent assembly overlaps are
not silently waived. An omitted pair remains outside the optimization scope.

Object geometry is strict by default. An explicitly authorized query-only
exception is `collision_avoidance.object_degenerate_policy:
"exclude_zero_area_queries"`. It retains every original vertex/face and material
ID in the compiled evidence, while the normal-dependent collision query uses
the original-face map excluding exact zero-area faces. The report lists excluded
IDs and original/query/map hashes, with `query_geometry_complete: false` and no
motion acceptance. Nonzero near-degenerate triangles, float32 collapse, and
contacts authored on degenerate faces still fail. Collision pairs, weights,
source references and hard object constraints are unchanged. This setting is
separate from the grasp bank's `inputs.object_degenerate_policy`; a contact-spec
builder must propagate an explicit policy into this collision section.

When the native HOI handoff carries a table or other static scene, bind its
original packed triangles in this same section:

```json
"static_scene": {
  "native_npz": "native_with_original_scene.npz",
  "world_frame": "soma_retargeter_z_up",
  "world_to_solver": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
}
```

The path resolves relative to `contacts.json`. `world_frame` must exactly match
the native handoff and its validated `grail.static_scene.v1` manifest.
`world_to_solver` is an explicit rigid transform in metres into
`soma_retargeter_z_up_m`; use the shown identity only when the two coordinate
systems coincide. Scale or reflection transforms are rejected. The compiler
uses every packed scene mesh and its original faces, applies only this declared
world conversion, and adds bidirectional robot/scene queries for every listed
robot geom. The scene stays fixed independently of the moving object and robot
FK. Reports bind the NPZ hash, scene fingerprint, original source provenance and
world transform. `UNKNOWN` scene geometry is rejected when requested; an explicit
`EMPTY` scene is recorded. Omitting the binding retains the earlier objective
and reports `NOT_CONFIGURED`; no table is inferred. In fixed-object mode,
object/scene pairs remain outside this robot-joint objective because neither
trajectory is editable. The optional object-update block includes those pairs
when `object_optimization` and a support scene are bound.

Degenerate static-scene triangles remain in the packed original geometry and
independent audit. The Warp query uses only the original faces with usable
normals at both stored and query precision; it does not remesh or replace them.
The compiled schedule retains full original faces, query faces and their
original-face-ID map. Each mesh report gives excluded counts, float32 vertex
error and `query_geometry_complete`; any omission means incomplete geometry
coverage. A scene with no usable faces is rejected. Degenerate original geometry
prevents a negative independent audit from establishing a complete clearance
result, even when these sampled penalties are zero.

The compiler preserves original triangles and builds deterministic surface
samples from vertices and face centroids (`16..1024` samples per mesh). Queries
run in both directions for each pair. Warp winding-distance queries supply a
signed clearance estimate and analytic gradients for both moving bodies. The
added loss is `weight * sum_pairs(mean_samples(max(margin_m - signed_gap_m, 0)^2))`.
At most 32 grouped RMS residuals keep the dense IK system small: grouping retains
every pair/sample and exactly preserves this squared loss (the Gauss–Newton
curvature approximation changes). A positive margin
also pushes intended contact surfaces apart by that amount; start at **1 mm**
and compare contact gaps as well as collisions. Weights are trial settings,
not physical stiffness or a guarantee of separation.

Avoidance runs on every source frame, including approach, and is disabled only
during upstream initialization/warmup. Original object transforms stay fixed
unless the explicit object-motion mode updates them during temporal sweeps.
Reports contain post-clamp sampled deficits for every source frame. A failed
distance query fails the run; a successful query or small loss does not accept
the motion. Finite sampling can miss intersections and winding signs can be
unreliable for open meshes. Environment geometry participates only through the
explicit static-scene binding. Keep the
independent original-triangle and affected-subframe audit; reject remaining
forbidden intersections instead of treating the soft objective as a hard guard.
