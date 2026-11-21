# Sparse object-relative wrist feedback

`wrist_keyframes` adds authored wrist position and orientation targets at explicit
source-frame/hand pairs. It is optional and disabled when the field is absent.
It complements named hinge `pose_keyframes`: hinge values preserve the articulated
shape, while object-relative wrist targets help preserve the grasp relation when
the full-body root or object changes during SOMA optimization.

This is an experimental soft objective, not a collision guarantee or an automatic
acceptance rule. Existing body, hand, object-reference, temporal and collision
objectives are retained. It does not pin the root or object in world coordinates.

## Configuration and binding

The normal SOMA config must already include `robot_hand_targets` and
`contact_constraints`. Add an exact key artifact binding:

```json
"wrist_keyframes": {
  "path": "feedback/wrist_keys.json",
  "sha256": "<lowercase SHA256 of wrist_keys.json>"
}
```

The key JSON has these required fields; only the optional `provenance` dictionary
may be added:

```json
{
  "schema": "soma_object_relative_wrist_keyframes.v1",
  "model_xml_sha256": "<original model XML SHA256>",
  "source_bvh_sha256": "<original source BVH SHA256>",
  "frame_count": 121,
  "fps": 24.0,
  "handoff_sha256": "<bound hand handoff JSON SHA256>",
  "targets_sha256": "<bound hand target NPZ SHA256>",
  "object_mesh_sha256": "<bound original object mesh SHA256>",
  "object_reference_sha256": "<reference_hash result>",
  "authored_native": {
    "path": "candidate_UNREVIEWED.npz",
    "sha256": "<actual authored native NPZ SHA256>"
  },
  "keyframes": [
    {
      "frame": 90,
      "hand": "left",
      "wrist_body": "left_wrist_yaw_link",
      "position_object": [0.0, 0.1, 0.0],
      "rotation_object": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "translation_weight": 10.0,
      "rotation_weight": 10.0,
      "translation_sigma_m": 0.02,
      "rotation_sigma_rad": 0.2
    }
  ]
}
```

The numeric pose above is a schema illustration, not a usable grasp. Extract
values from the actual authored native trajectory. The authored file path is
resolved relative to the key artifact. The top-level artifact path is resolved
relative to the SOMA config. The enabled hand and wrist-body name must exactly
match the bound hand targets. Frame indices refer to the original zero-based
source clock; no resampling or duration changes are performed.

Use `soma_wrist_keyframes.reference_hash(rotations, positions, scale)` on the
loaded hand-target solver object reference. It hashes canonical float64 shapes
and bytes; do not substitute a hash of the raw reference file. Copy handoff,
target and object mesh identities from the validated hand-target report. The
runner also preserves the existing asset/model manifest checks.

Invalid hashes, lineage, frame rate/count, hand/body names, duplicate frame/hand
pairs, nonfinite poses, nonpositive weights/sigmas, and rotations outside the
SO(3) tolerance fail closed. A key artifact requires at least one explicit key.
Omit the config to disable the feature; an empty artifact is not a disable flag.

## Extracting authored targets

Use the saved and reopened native Blender export when the repair was authored in
Blender. Preserve that file and its review evidence. For each requested frame,
run mounted robot forward kinematics from its native `qpos` and obtain the named
wrist body's world position `p_w` and rotation `R_w`. With the actual authored
object transform `(R_o, t_o)`, store:

```python
p_relative = R_o.T @ (p_w - t_o)
R_relative = R_o.T @ R_w
```

`position_object` is in metres with the original object scale already applied;
it is not an unscaled mesh coordinate. Do not multiply it by the object scale
again. If a serialized rotation needs conversion to SO(3), use the same explicit
normalization used by the native evaluation, and record the source discrepancy.
Never silently alter the native trajectory or relax the SO(3) validator.

For two interacting hands, record both wrist relations even if the correction
edits only one hand. Preserve the other hand's actual authored pose instead of
inventing a new one. Include reviewed neighboring guard frames when needed, and
audit the transitions outside the key interval. Do not expand a repair window
without explicitly recording the added constraints and their source.

## Objective and execution

At a keyed frame the target follows the current optimizable object:

```text
p_target = R_object * p_relative + t_object
R_target = R_object * R_relative
translation residual = sqrt(weight_translation) / sigma_m * (p_target - p_wrist)
rotation residual = sqrt(weight_rotation) / sigma_rad * log(R_wrist * R_target^T)
```

The solver includes both robot and object derivatives. Existing material-position
and wrist-rotation primitives provide the robot Jacobians; the relative target
contributes analytic object translation/rotation columns to the global system.
There are exactly six residual rows per enabled wrist. Their gains are zero at
unlisted frames and during warmup. Keys are not interpolated, and they are not
disabled when the global optimizer disables sequential temporal priors.

Both the fresh forward initialization and the whole-sequence optimization apply
the added rows. CUDA execution stages the live-object targets and sparse gains
before graph replay; the CUDA bulk evaluator gathers the corresponding frame.
The original `robot_hand_targets` are never overwritten. If their goals conflict
with the authored targets, the resulting soft compromise must be measured.

## Evidence and review

`contact_initialization.json` records the `wrist_keyframes` binding, retained
lineage, explicit keys, and forward/final translation and rotation errors.
`wrist_keyframe_states.npz` records actual relative wrist FK states together with
the corresponding robot, object and time arrays. The adapter replays those
diagnostics and checks their relationship to the actual global output.

Scalar diagnostic replay permits `1e-10` absolute floating roundoff across
runtime/adapter SciPy builds. This does not relax file hashes, state-array
identity, key poses, collision tests, or any motion-quality threshold.

Review the actual exported native trajectory, not only solver success or key
error. Compare configured original self pairs, robot versus original object,
and robot versus original static scene; inspect held-out fractional times and
the joins. Retain original zero-area mesh faces and resolve their supported
segment/point queries explicitly. Open meshes do not establish solid occupancy
or physical penetration depth. Finite sampled clearance is not a continuous
collision proof, and other known collisions can remain outside the repaired
scope. World-space table clearance is not protected by an object-relative target
when the object and hand move together.

## Validation and implementation scope

Focused tests cover strict artifact binding, clock/lineage rejection, sparse
masking, robot and object finite differences, sequence assembly, CUDA graph
staging, and CUDA bulk evaluation. An actual 121-frame disabled comparison
preserved the old forward/global arrays and CSV byte-for-byte. Matched enabled
CUDA experiments retain their separate runtime and geometric evidence;
successful execution alone does not select a candidate for production.

The feature is implemented by `soma_wrist_keyframes.py` and
`soma_wrist_keyframe_runtime.py`, with narrow integration in `soma.py`,
`soma_contact_runtime.py`, `soma_sequence_runtime.py`, and `soma_sequence_cuda.py`.
Artifact-specific experiments and cluster round launchers are outside the
runtime API. Keep the optional feature disabled unless a case explicitly requests and
reviews authored feedback.
