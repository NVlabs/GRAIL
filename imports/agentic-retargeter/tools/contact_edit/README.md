# Contact editing tools

| File | Purpose |
| --- | --- |
| [pose.py](pose.py) | Optimize explicit limited MuJoCo hinges at one source frame against fixed original-triangle contacts. |
| [blender_curves.py](blender_curves.py) | Create editable Blender F-curves, evaluate real Blender interpolation, export source-rate native samples and subframe comparisons. |
| [native_io.py](../../src/contact_edit/native_io.py) | Validate joint bindings, protected windows, endpoints, limits and cache invalidation. |
| [experimental sequence guide](../../src/contact_edit/experimental/README.md) | Opt-in G1 contact-preserving sequence prototype, bundled objective and caller-input schema. |
| [make_demo.py](make_demo.py) | Generate a CPU toy robot and synthetic contact/curve inputs. |

Use Python with NumPy, SciPy and MuJoCo. Blender needs its bundled NumPy; it does not need the external Python environment's MuJoCo for curve editing. The Blender code targets the `Action.fcurves` API used by Blender **4.0–4.3**. A newer layered-action API needs adaptation and testing.

Run these commands from the repository root, with a suitable Python environment active and the intended Blender executable on `PATH`. Each output directory must be new; choose a different demo name when repeating the example.

```bash
contact_kit=tools/contact_edit
contact_demo=tmp/contact_edit_demo

python "$contact_kit/make_demo.py" --out "$contact_demo"
python "$contact_kit/pose.py" \
  --config "$contact_demo/pose.json" --out "$contact_demo/pose_candidate"

blender --background --factory-startup --python-exit-code 1 \
  --python "$contact_kit/blender_curves.py" -- \
  --native "$contact_demo/native.npz" --model "$contact_demo/model.npz" \
  --plan "$contact_demo/curves.json" --out "$contact_demo/curve_candidate"

blender "$contact_demo/curve_candidate/editable_curves.blend"
```

The toy pose and curve examples start from the same source archive and produce separate candidates. The toy `.blend` provides curve controls; it does not build a robot visualization rig. Synthetic contact closure demonstrates the tool contract, not an accepted grasp.

For a contact pose, start with the original object's triangle topology and explicit robot pad geometry. A minimal config is:

```json
{
  "schema_version": 1,
  "model_xml": "robot.xml",
  "trajectory_npz": "native.npz",
  "object_mesh_npz": "object_mesh.npz",
  "frame": 10,
  "editable_joints": {
    "pad_hinge": {"delta_bounds_rad": [-0.4, 0.4]}
  },
  "contacts": [{
    "name": "authored_material_contact",
    "body_name": "moving_pad",
    "point_local": [0.1, 0, 0],
    "normal_local": [0, -1, 0],
    "object_face_id": 0,
    "object_barycentric": [1, 0, 0],
    "object_normal_sign": 1,
    "position_weight": 1,
    "normal_weight": 0.0
  }],
  "solver": {"max_evaluations": 60, "pose_weight": 0}
}
```

Paths resolve relative to the config. The mesh NPZ contains original `vertices[V,3]` and integer `faces[F,3]`. The trajectory requires floating `qpos[T,nq]`, scalar `fps`, `timestamps[T]`, `obj_R[T,3,3]`, `obj_t[T,3]`, and positive scalar or XYZ `object_scale`. The solver checks optional native `dof_names`/`qposadr` and compiled-model bindings when present. Coordinates use metres and radians.

Each pad normal must already be unit length. The target normal comes from the authored triangle after applying object scale, multiplied by `object_normal_sign`; the pad normal is compared with its opposite only for diagnostics by default. Face IDs and barycentric coordinates remain fixed throughout a solve. There is no nearest-point relabeling, remeshing, automatic grasp selection or automatic finger mapping. Joint addresses and limits come from the actual compiled MuJoCo model. Each finite correction corridor includes zero and intersects the model's native hinge limits. Invalid inputs are rejected without clipping. Models with equality constraints require a different, constraint-aware solver.

The default solver budget is 100 residual evaluations, bounded above by 10,000; the default source-pose weight is `1e-5`. Contact weights multiply squared position and optional opposing-normal residuals. `normal_weight` defaults to zero; a nonzero value explicitly enables alignment. Unit normal metadata is still required by the current schema for diagnostics. It is not a default contact acceptance condition. `pose.py` writes `candidate.npz`, the exact input `config.json`, and `report.json`. Reports include per-contact gaps/normals, edited bounds, frozen checks, optimizer termination and source/model/config hashes. Candidate NPZs carry their own `UNREVIEWED` status and export kind; earlier acceptance labels are stored as explicitly parent-prefixed history. A rejected caller guard instead produces `REJECTED_BY_GUARD`.

The callable APIs are `solve_pose(config, config_dir='.', candidate_guard=None)` and `run(config_path, out_dir, candidate_guard=None)`. Both return a `PoseResult` with `.trajectory` and `.report`. A guard receives an independent copy of the whole candidate trajectory and returns a JSON-serializable dictionary such as `{"passed": false, "reason": "original-mesh intersection"}`. Implement that decision using the caller's checks. This is an optional **post-candidate guard**, not a collision objective or collision constraint inside the optimizer. The CLI does not perform original-mesh collision, source-image, contact-completeness, adjacent-frame, force-closure or dynamics checks.

For animation, a curve plan selects each hinge and its own protected source-frame window:

```json
{
  "schema_version": 1,
  "channels": [{
    "joint": "pad_hinge",
    "window": [0, 20],
    "keys": [[0, 0], [6, 0.3], [13, -0.15], [20, 0]],
    "interpolation": "BEZIER"
  }]
}
```

Keys are `[source_frame, radians]`; interior keys may use fractional frames. Window endpoints are integer source frames and their values must match the parent motion. Only each window's strict interior is editable. Other joints, both endpoints, and all samples before/after the window remain exact. Use separate windows for independently timed fingers. The model snapshot NPZ contains `joint_names`, `jnt_type`, `jnt_limited`, `jnt_qposadr`, and `jnt_range` from the actual compiled model; the toy generator shows how to save it.

**Source frame 0 is Blender frame 1**, including fractional times. Source fps and frame count remain unchanged. Frozen source free/ball-joint quaternions must have unit endpoints and nonnegative adjacent dot products; the kit rejects hemisphere jumps rather than silently rewriting source orientation. Frozen quaternion components interpolate linearly and native FK normalizes them. In the saved scene:

1. Run the embedded `Enable_ContactEdit_Preview.py` text after reopening.
2. Select `ContactEdit_Curves`, then open its Action in the Graph Editor. Properties named `qpos_XX` refer to model-derived native addresses; their descriptions give joint names.
3. Edit interior keys and handles. Keep `ContactEdit_FrozenReference` unchanged and retain the protected endpoint times/values. Do not add unlisted channels.
4. Run the embedded `Export_ContactEdit.py` text to create a fresh export directory. The parent scene's older exporter does not export this kit's Action.

The frozen reference compensates for Blender's float precision so an unchanged passage through the editor does not alter native doubles. Source object transforms, scale, camera-related metadata and frozen native coordinates are preserved. Edited qpos invalidates derived FK/task-error caches; known caches are removed and `fk_cache_valid=False` invalidates all motion-derived data. List additional custom cache keys in `derived_fields` in the pose config or curve plan, then recompute caches from the exported candidate before using them.

To use an existing compatible portable G1 scene, supply it before the Python script:

```bash
blender --background path/to/portable_g1.blend --python-exit-code 1 \
  --python "$contact_kit/blender_curves.py" -- \
  --native path/to/matching_native.npz --model path/to/matching_model.npz \
  --plan path/to/curves.json --out path/to/new_curve_candidate
```

This optional preview supports the portable scene runtime embedded as `Enable_Exact_Native_Preview.py`, with `Native_Qpos_Controller` and `Native_Qpos_Reference`. The selected native checkpoint and model must match the scene. Unexported parent-controller edits, drivers or NLA are rejected; export/rebase them first. An arbitrary `.blend` does not gain an exact native G1 preview automatically. Check `exact_native_visual_preview` in the curve report. Child Of constraints and NLA layers can be useful authoring ideas in a separately validated workflow, but this kit does not implement their baking/export; its explicit Action exporter rejects controller drivers and NLA.

Curve exports contain `candidate_UNREVIEWED.npz`, `editable_curves.blend`, `authored_curves.json` with the actual keys/handles, the initial `plan.json`, `subframe_comparison.npz`, and `report.json`. The native NPZ retains the original frame rate. A consumer that linearly interpolates those samples will differ from Blender's Bézier curve between frames. The report records `authored_vs_linear_subframe_max_rad` and maximum sampled finite-difference angular speeds for both interpretations, not analytic peak speeds; the comparison NPZ stores both trajectories at the sampled subframes. `--substeps` defaults to 4 per source interval and must be at least 2. These samples detect sampled violations; they are not a continuous-time guarantee. Review both the authored motion and the actual downstream native replay.

Use an outer editing loop across the action: inspect source evidence and original geometry; choose contacts and palm orientation; solve a pose; inspect residuals and whole-hand geometry; manually revise the contact hypothesis, permitted palm/arm pose or auxiliary approach/path keys; then shape the timing in Blender and export again. Auxiliary paths are manual planning aids, not a built-in Cartesian path solver. Preserve each candidate and its evidence. An inner solve reaching `max_evaluations` completes that bounded numerical attempt; it is **not a task-stopping condition** or a reason to accept the grasp. Continue the outer review/correction loop until the requested contact and motion are externally reviewed, or record a concrete unresolved limitation. A small gap or optimizer success alone does not establish acceptance. This kit edits native robot hinges; it is not a human-pose solver.

Run the Blender curve check on the demo's no-op plan:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python "$contact_kit/blender_curves.py" -- \
  --native "$contact_demo/native.npz" --model "$contact_demo/model.npz" \
  --plan "$contact_demo/noop_curves.json" --out "$contact_demo/noop_check"
```

The tests cover reachable fixed-material contact, frozen coordinates and limits, invalid descriptors, guard rejection, status/cache handling, independent curve windows and protected endpoints. The paired smoothing check holds synthetic keys fixed, verifies the omitted setting matches explicit `NONE` exactly, verifies that `CONT_ACCEL` changes the actual Blender curve, and records derivatives of both authored and source-rate replay samples. It does not require a lower derivative metric: that is an experiment result, not a tool-contract invariant. The Blender round-trip check exercises a real key edit, Action snapshot restoration, saved-scene reopening and exact exported native samples; a compatible portable scene also checks native FK preview. These are tooling checks. Original-mesh penetration, source alignment and full-sequence contact/temporal review remain necessary for the motion itself.
