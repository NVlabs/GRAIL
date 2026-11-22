# Retargeting

<span id="retarget-a-reviewed-fit-to-unitree-g1"></span>

<span id="robot-retargeting"></span>

Start with the selected native SOMA03 motion, the fixed body identity, source frame clock, original CAD and support scene from [reconstruction](recon_4dhoi.md). Pick the robot before converting. The terrain target has 29 articulated joints. The manipulation target adds seven joints per hand. The SOMA thumb, index and middle fingers guide the robot’s three digits. Ring and little fingers have no independent robot counterparts. Check the resulting grip on the original CAD. The retargeter can apply kinematic foot-plant stabilization when enabled for the target. Compare any foot change with the support measured during human reconstruction.

| Example | Command on the host | Result in the case directory | Time in the standard image on RTX 6000 Ada |
|---|---|---|---|
| Stairs | {ref}`Run the terrain conversion <stairs-retarget-step>` | `robot.npz` with 241 × 36 poses, `retargeting.mp4`, foot support review | 8 min 43 s |
| Kettlebell | {ref}`Convert the grasp to Unitree G1 <kettlebell-retarget-step>` | `robot.npz` with 121 × 50 poses, `retargeting.mp4`, grasp and clearance review | 13 min 56 s |

## Agent review of robot motion

<span id="how-the-agent-reviews-a-robot-motion"></span>

The {blob}`retargeting skill <.agent/skills/grail-agentic-retargeting/SKILL.md>` starts from the **selected human fit and its review**. Give it the request under [Native retargeting](#native-retargeting) for a moving object. For static terrain, use the 29-joint target, actual terrain mesh and support intervals as in the {ref}`stairs example <stairs-retarget-step>`.

**Your review before conversion.** Read the reconstruction review and select a specific human fit. If you want to choose between placements before contact refinement, append the [Stage 1 pause request](recon_4dhoi.md#agent-review-of-a-fit) to the reconstruction request. To review the robot before it goes to tracking, append this to the retargeting request:

> Show me the selected robot motion beside the source in both views. Record the parent human fit and robot candidate, output paths, units and coordinate frame, SHA-256 of the source MP4 and original OBJ, and unresolved frames. Wait for my selection before packaging the motion.

1. **Bind and plan.** The agent reads the fit, camera, frame clock, original CAD, support and the human model’s neutral joint rotations used to align the robot joints before transferring motion. It writes the case YAML, selects the G1 terrain target or G1 with articulated hands, and calls the {blob}`plan, preflight and run entry point <grail/retargeting/agentic.py>`. The native solver output is `native/robot_UNREVIEWED.npz`. A successful conversion establishes which inputs were used.
2. **Inspect the interaction.** Compare the robot to the source in the calibrated camera, then use the oblique view to see depth. On stairs, inspect each sole through approach, stance and release. For a moving handle, inspect each finger's approach, opposing contact, closure, slip and release in object coordinates. Read the original-mesh clearance and wrist measurements alongside the movie.
3. **Edit and compare.** The agent can use the {blob}`native grasp inspection and Blender tools <imports/agentic-retargeter/docs/hand-inspection.md>` for a localized robot edit. You can request a change in plain language: “At frames 94–96 the fingers close across the handle. Keep the reviewed human motion, object trajectory and earlier grip fixed. Save a separate robot candidate and show the two hands in both views before and after.” For stairs, identify the foot, tread and frame interval. Recheck the points on the object that each finger should touch. Compare the edited interval and the full motion with its parent, including other stance contacts and wrist targets.
4. **Select or stop.** Reopen the chosen motion, decode the full comparison, check the shape and timestamps in the [output reference](outputs.md), and retain the limits in the written review. You choose whether a partial motion is suitable for exploratory simulation. If it changes the CAD, misses the tread or loses the grip, revisit that binding or edit before packaging. The handoff couples the selected robot path with the original geometry and source contact map.

On the released stairs motion the right toe intersected one tread by up to 12.6 mm. On kettlebell the two wrists stayed within 4.5 and 5.1 mm of their mean positions relative to the held object, but the original-mesh audit still found intersections. Both reviews retained those observations. Their wall times are listed above. The examples establish neither a universal acceptance threshold nor a billed cost for a new case.

<span id="install"></span>

## Native retargeting

<span id="convert-your-selected-fit"></span>

The agent reads the selected fit, exports its human skeleton and neutral calibration, prepares the robot and support scene, runs the native solver, and compares the result to the source in camera and oblique views. It writes a selected native motion and records contacts that still need review.

For your own moving object, replace the case directory, object group and motion key together. Here `my-lift` and `Lab/weight/lift01-start-end` identify a repackaged released input. After you have selected its fit, give Codex this request from the host:

> Use grail-agentic-retargeting in grail-demo. Read /workspace/runs/my-lift/selected_fit.pkl, its inventory.json, review, contact_map.json for Lab/weight/lift01-start-end, and generation/mesh/Lab/weight/model.obj. Reuse the measured support geometry, source camera, FPS and SOMA03 identity from this fit. Select unitree_g1_dex3. Write retarget/retarget_case.yaml under this case; use the grail Python for source export and the installed soma-retargeter .venv for the robot solve. Check the neutral rest calibration, original CAD hash, hand contacts and floor clearance. Save robot.npz, retargeting.mp4 and the written review in /workspace/runs/my-lift. In the review identify the selected parent and output paths, units and coordinate frame, source MP4 and original OBJ hashes, and unresolved frames, including any partial grasp.

We checked the plan and built a handoff from the released robot motion with the lift inputs under new names. We have not run the own-case request through to a new robot motion.

For manual configuration, [download the case YAML](_static/retarget_case.yaml) to `<case>/retarget/`. Paths are relative to that file.

- `source.path` points to the selected SOMA fit, normally `../selected_fit.pkl`.
- `native.object_mesh` points to the original metric OBJ named in the reconstruction inventory. Compare its hash before solving.
- `native.scene_sidecar` points to `original_floor.npz`. Export the named, static support mesh from the original Blender room in world metres. The {blob}`Blender scene exporter <grail/retargeting/blender_scene_export.py>` takes the bound room file, object names and output path. The released lift used a case-specific reader for its legacy room to save the original floor vertices, faces and transform.
- `backend.bindings.camera` points to `source_camera.json`. Write the selected fit’s camera rotation and translation, focal length in pixels and image height and width. Declare the projection convention (the lift used `pytorch3d_world_to_camera` with `R` and `T`) and check it against the original conditioning camera.
- `backend.extra_robot_paths` contains the generated `target_g1_dex3/` directory. Export `source.bvh` and `native_rest.npz` from the fixed SOMA identity first. Pass both to the {blob}`target generator <grail/retargeting/soma_target.py>` with the installed SOMA retargeter and the bundled G1 with articulated hands XML.

Run `python -m grail.retargeting.agentic plan --config <case>/retarget/retarget_case.yaml --output <fresh-plan-directory>` to inspect the resolved inputs. For a raw SOMA fit, `SOURCE_EXPORT_REQUIRED_UNREVIEWED` means that `run` will export the source before solver preflight. For stairs, select 29-joint `unitree_g1` and the original fixed collider.

## Robot and object alignment

<span id="what-should-move-together"></span>

Use the neutral pose to align the joints. The first video frame can already contain a bent knee or closed fingers. Keep the frame rate, scale, support surfaces and object trajectory from the reviewed fit. If a grip looks right only in the source view, check it from the side before editing the hand target or object scale.

## Tracking handoff

<span id="hand-off-a-retargeted-motion"></span>

Check the [native array layout and timestamps](outputs.md#robot-file-validation), then read the visual review. The solver's initial native archive may still say `UNREVIEWED` even when a later review selects a partial result. Keep that review alongside the chosen file.

**Check the source before packing a grasp.** Compare `contact_map.json` → `binding.source_video_sha256` with the selected fit’s `inventory.json` → `files.source_video.sha256` and the actual MP4. Match the original OBJ to `files.object_mesh.sha256` as well. The handoff checks map schema, clock fields when present and hand intervals. A map from another clip with the same timing can pass. Resolve a missing or different hash before handing the motion to tracking.

For static terrain, pack the terrain reference together with its original collider as in the {ref}`stairs pack <stairs-tracking-step>`, substituting your own native path and motion key. For a moving object, run the following from the container checkout root. Change `CASE`, `KEY` and `OBJ` together. The example uses the case's original floor. For an actual tabletop pickup, inspect the original finite table and choose the corresponding supported table-block route.

```bash
CASE=/workspace/runs/my-lift
KEY=lift01
OBJ="$CASE/generation/mesh/Lab/weight/model.obj"
python scripts/standalone/check_robot.py "$CASE/robot.npz" --object-mesh "$OBJ"
python -m grail.retargeting.manipulation.handoff --native "$CASE/robot.npz" --object-mesh "$OBJ" \
  --contact-map "$CASE/contact_map.json" --support native --motion-key "$KEY" --output "$CASE/handoff"
```

Expect `handoff/handoff.json` with your motion key and frame count. The CPU check and handoff were exercised on the released lift with its original mesh under a different name (121 frames, 24 fps). Only proceed to [training](tracking.md#full-training-schedule) when the visual review selects this robot file. Use that case’s own handoff directory and an unused training output.

## Optional stance-floor correction

The optional {blob}`stance-floor tool <grail/retargeting/stance_floor.py>` creates a separate lower-limb candidate for a measured flat floor. Check both soles, wrist targets and object-relative grip against the native motion before selecting it. Leave it off for support surfaces at different elevations. The release image provides MuJoCo in the `sonic` environment. Use the model XML bound to the native motion, the original CAD and a fresh output path.

```bash
conda run -n sonic python -m grail.retargeting.stance_floor --source "$CASE/robot.npz" \
  --model-xml "$BOUND_MODEL_XML" --object-mesh "$ORIGINAL_CAD" --output "$CASE/stance_candidate.npz"
```

The tool checks the compiled robot model, joint order, CAD hash and native source clock. A model or schema mismatch calls for inspecting the input archive.

<span id="troubleshooting"></span><span id="legacy-smpl-x-gmr-path"></span>

## Target and support configuration

<span id="configure-targets-support-and-corrections"></span>

The {blob}`native retargeting reference <docs/agentic_retargeting.md>` gives the custom robot registry, original Blender support export, coordinate transforms and correction settings. Raw SOMA exports can ground frame zero using one shared vertical offset for the human, object, camera and support. Inspect the selected offset before solving. Moving only the robot changes its clearance relative to the scene. Use the native rest rotations to calibrate the wrists and fingers; a translation-only BVH offset cannot encode the full neutral frame.

Preserve the original floor/table geometry and object scale. A tabletop support used by the simulator is a separate approximation. Keep edits bounded to the affected frames and inspect the unedited portion, wrist targets and both feet. The [native grasp tools](https://github.com/NVlabs/GRAIL/blob/main/imports/agentic-retargeter/docs/hand-inspection.md) describe hand candidates and their review. If FK or identity validation fails, verify the source, target and installed revision before repeating the solve.

For several reviewed handoffs of the same object, supply the packer with one `--handoff` argument per motion. It writes a roster, physics data and per-motion evaluation inputs indexed by `tracking_pack.json`. The [tracking guide](tracking.md) describes the corresponding data and policy. For the legacy SMPL-X route the standalone retargeter requires a separate user-provided legacy installation.

<span id="replay-robot-object-and-table"></span>

## SOMA to Unitree G1 + DEx3

<span id="convert-a-validated-reconstruction-directly"></span>

The public distribution also includes a single-file adapter for a trusted native SOMA reconstruction. From the checkout root, set `HOI` to its `hoi_data.pkl` and `RUN` to a new output path:

```bash
python -m grail.retargeting.soma "$HOI" "$RUN" --model-config configs/retargeting/soma_g1.json
```

For the 29-joint terrain target, add `--target unitree_g1` to the command above. The default manipulation target includes articulated hands. The adapter exports `soma_export/source.bvh` plus native source/rest transforms, builds the robot hand target for this human fit and its neutral pose, and produces a CSV and hash/FK receipts. The 50-column hand-target CSV contains the frame, six root channels, 29 body joints and 14 finger joints. The initial conversion is unreviewed. To see the CSV with the object's original motion and a tabletop:

```bash
imports/soma-retargeter/.venv/bin/python -m grail.retargeting.soma_replay \
  "$RUN/g1_retarget/csv/source.csv" "$HOI" "$RUN/g1_dex3_target" "$RUN/replay.mp4"
```

This replay estimates table height from the initial transformed object vertices. Use `--table-size WIDTH DEPTH THICKNESS` to change its dimensions. Inspect the registered source scene separately when deciding contact quality. The CLI and replay arguments are available in the public distribution. The timed workflows above used the reviewed case YAML route.

## Manual execution

<span id="run-a-configured-case-without-the-agent"></span>

From the GRAIL environment in the checkout root, set `CFG` to your prepared YAML and give each action a fresh output directory outside the input tree. For example, with `CFG=/workspace/runs/my-lift/retarget/retarget_case.yaml`:

```bash
python -m grail.retargeting.agentic plan --config "$CFG" --output /workspace/runs/my-lift-plan
python -m grail.retargeting.agentic preflight --config "$CFG" --output /workspace/runs/my-lift-preflight
python -m grail.retargeting.agentic run --config "$CFG" --output /workspace/runs/my-lift-initial
```

For a raw SOMA source the standalone preflight can report that source export is required. The run action performs that export before backend preflight. The optional `workflow` action accepts a configured batch/refinement specification. Check the receipt and review each selected motion separately. To build a group pack, set the arguments to two reviewed handoffs of the same object:

```bash
python -m grail.retargeting.manipulation.pack --handoff "$FIRST_HANDOFF" \
  --handoff "$SECOND_HANDOFF" --num-envs 2048 --output "$NEW_PACK"
```
