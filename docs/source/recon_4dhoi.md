# 4D HOI Reconstruction

<span id="reconstruct-and-review-a-humanobject-motion"></span>

<span id="required-environment"></span>

Interaction-aware HOI reconstruction starts with a generated video, its original camera, depth, masks and metric CAD, and the installed [reconstruction workers](downloads.md). A fit gives you a sequence of SOMA03 body poses plus object poses on the source frame clock. You need the native fit to retarget; an MP4 lets you inspect it.


<span id="common-variants"></span>

## Quickstart

<span id="run-the-example-that-matches-your-interaction"></span>

From the host, use the command in the linked tutorial. Both were run with the tools in `nvgrail/grail:1.2.0` on one 48 GB RTX 6000 Ada.

| Scene | Command and expected output | Measured time |
|---|---|---|
| Static stairs | {ref}`Run the stairs reconstruction <stairs-fit-step>`. Keep the CAD fixed. Look for `selected_fit.pkl`, `selected_fit_geometry.npz` and `reconstruction.mp4`. | 32 min 52 s including a first depth-model download |
| Moving kettlebell | {ref}`Run the kettlebell reconstruction <kettlebell-fit-step>`. Also keep `contact_map.json`. Review where the grip begins and whether the handle rises with the hand. | 17 min 07 s |

## Agent review of a fit

<span id="how-the-agent-reviews-a-fit"></span>

The {blob}`reconstruction skill <.agent/skills/grail-agentic-hoi-recon/SKILL.md>` tells a coding agent how to use the fitting and inspection tools. You provide the case and review the motion it selects. The solver optimizes the supplied observations and contact targets. Choosing what the video actually shows is a separate step.

| Decision | What the agent runs or inspects | What you can review |
|---|---|---|
| Are these the right observations? | Bind the clip and its clock, camera, original metric CAD, fixed SOMA03 identity, human prediction, masks, depth and object pose. Check the first frame against the source overlay. | Confirm the selected sequence and conditioning frame, mesh orientation, scale and support. Return to the inputs if they disagree. |
| Is the placement credible? | For fixed stairs, {blob}`grail.agentic_kinematic.hoi_locomotion <grail/agentic_kinematic/hoi_locomotion.py>` fits the human to the finite treads in one solve. For a moving object, {blob}`grail.agentic_kinematic.hoi_stages <grail/agentic_kinematic/hoi_stages.py>` runs Stage 1 for placement and Stage 2 for contact refinement. Inspect both camera and oblique views. | Look at contact transitions as well as the quiet frames. On a lift, identify first touch and first object motion independently. Review placement before refining contact. |
| What should attract, and when? | For manipulation, select the Stage 1 parent or a reviewed native placement edit. Mark each hand's contact intervals and regions on the original mesh. Run Stage 2 from that exact parent. Leave uncertain and no-contact intervals without an invented target. | Check the named hand, the side of the handle, the first and last contact frames and the second view. A static skip requires full-sequence translation **and** rotation evidence. |
| Which fit should go forward? | Reopen the native result. Compare source and fit through all frames and measure support, slip or regional gap. Return a selected fit, geometry, comparison movie and review with unresolved intervals. | Inspect the movie and contact map before handing the pair to the robot solver. Keep a partial result labelled partial. If the evidence cannot distinguish two placements, ask for that uncertainty in the review. |

**To choose the placement yourself**, append this to the reconstruction request for a moving object:

> Stop after Stage 1. Show me the source overlay, an oblique view and the proposed contact intervals. Name the placement checkpoint and wait for my selection before starting Stage 2.

For stairs, ask to review the single terrain fit before retargeting. In either case, ask for a written review naming the selected parent checkpoint, paths to the selected outputs, coordinate frame and units, SHA-256 of the source MP4 and original OBJ, and unresolved frames or intervals. After an edit, compare the revised interval and the rest of the motion with the parent. Give the selected fit and that review to the robot stage.

For a new generated case use the **copyable request below**. To request an iteration, name the case, parent fit, frame interval, view and visible error. For example: “In the second swing the handle points away from the palm in the oblique view. Recheck the original CAD orientation against the source, preserve the first swing, and show the same frames before and after in a new output directory.” Change a contact interval only if the original frames support it. For a stair foot on the wrong tread, inspect the assigned finite surface and transition timing before adjusting a loss weight.

The [stairs and kettlebell reviews](results.md) illustrate different stopping points. The stair fit reached a 0.4 mm median labelled sole error while retaining a heel overhang elsewhere. The kettlebell fit lifted the CAD and reduced the two hand-region gaps to about 7 mm, but the hands still crossed the handle. The wall times above include the agent and tools on the released examples. Model-token usage and per-case charges were not measured. The renamed lift tested input binding; it supplies no runtime or quality measurement for a different object.

<span id="pipeline-steps"></span>

## Pipeline stages

<span id="staged-fitting-and-review"></span>

The released stairs fit used 400 iterations against the labelled floor and treads. Check heel and toe at stance transitions. The kettlebell fit used 400 placement iterations and 200 contact-refinement iterations, starting from the reviewed placement. Each hand has its own zero-based contact intervals `[start, end_exclusive)` on the original mesh. Planar foot contact limits sliding. The measured floor plane sets the contact height.

For a moving object, give the agent a file you prepare with the **upward unit normal**, the signed `offset_m` in metres (the plane satisfies `normal · world_position + offset_m = 0`), the source world frame and how the floor was measured. Record the calibration and measurement source, such as the original room floor or registered frame-zero depth and floor pixels. Both `init_opt` and `contact_opt` must use the same measured `human_foot_ground.plane` with its provenance in `source`. On stairs, supply the finite support manifest instead. [Inspect the fit in its original room](visualization.md).

## Configs

<span id="select-your-own-generated-sequence"></span>

Use a generation tree you control. The names below follow the released lift repackaged as `Lab/weight/lift01-start-end`. Replace **all** of the names with yours, including the conditioning stem if it differs from the MP4. Keep inputs under the new case root so inventory paths can be relative. Put your measured plane record at `/workspace/runs/my-lift/floor-plane.json` for the request below, or substitute its actual path. Run the [capture check](tutorials/own-video.md#example-case-layout) before starting the agent. On the host, give Codex the following request in the same checkout and `grail-demo` container as the tutorials:

> Use grail-agentic-hoi-recon. My case is /workspace/runs/my-lift, my generation root is /workspace/runs/my-lift/generation, my video sequence is Lab/weight/lift01-start-end and my conditioning stem is lift01. Read the MP4 with that exact stem in videos_kling/Lab/weight, the original mesh at mesh/Lab/weight/model.obj and the camera, masks, depth and pose for that conditioning stem. Use only this case. Copy configs/recon_4dhoi/manip_soma.yaml to my case. Read my floor measurement from /workspace/runs/my-lift/floor-plane.json and use it in both fitting stages. For the observation workers use /workspace/runs/my-lift/observations/generation with this exact video copied as lift01.mp4 alongside the lift01 calibration, as described below. Write case.json and inventory.json for the original source in my case. Bind the source hashes and frame clock before Stage 1. Review placement, first touch and first object motion; author contact_map.json and contact_temporal.json for both hands before Stage 2. Save the reviewed selected_fit.pkl, selected_fit_geometry.npz, reconstruction.mp4 and review inside my case. Record the selected parent, output paths, coordinate frame and units, SHA-256 of the original MP4 and OBJ, and unresolved frames. Report missing inputs before fitting.

When the agent prepares the inventory, compare `files.source_video.sha256` and `files.object_mesh.sha256` to the actual files you selected, and check `sequence`, `fps` and `frame_count`. All must agree before accepting a fit. The [native fitting procedure](agentic_kinematic.md#case-json-and-input-inventory) gives the manifest, contact example and Stage 2 resume command. The renamed input was checked; a fit for a new object has no worked example yet.

<span id="inspect-existing-native-results"></span>

## Retargeting criteria

<span id="decide-whether-to-retarget"></span>

Keep the selected native fit, its geometry, review and source bindings together. The example stairs fit measured a 0.4 mm median absolute sole-to-tread distance over labelled samples. The kettlebell CAD rose 0.68 m at its peak, and the review retained grip limitations. These are reference observations for those sequences. See the [full results](results.md) before interpreting a small numerical loss as a good interaction.

If the feet land on the wrong step, revisit the camera, original stairs and support intervals. If the hand appears close only in the camera view, check the oblique view, object orientation and handle vertices before rerunning contact refinement. Once the motion is selected, continue to [G1 retargeting](retargeting.md).

## Case inputs and contact maps

| Input | What the fitting stage needs |
|---|---|
| `case.json` and `inventory.json` | The case identifies the native SOMA03 backend, fixed human identity and optimization YAML. The inventory binds the sequence, source FPS and frame count, camera, metric CAD, prediction, masks and depth. Keep files with their original hashes. |
| Locomotion support | Label the floor and finite tread surfaces on the original scene. Specify source-frame intervals, foot region and support surface. Inspect entry and exit from stance in both camera and oblique views. |
| Moving-object placement | After Stage 1, inspect when the object first moves and which hand is visibly involved. If placement needs a correction, keep the reviewed parent fit and protect unaffected motion. |
| `contact_map.json` | Identify the source video, original object and human topology. Author each hand independently. A hand region and object region can have different numbers of vertices. Preserve their separate sets and weights; their index order does not define pairs. |

For uncertain or no-contact intervals leave the attraction targets empty. Use source frames to localize first touch, stable contact and release. Recheck bindings if changing the mesh, pose topology or frame clock.

<span id="sharded-fan-out"></span>

For multiple independent sequences, the reconstruction entry point supports a chunk count and zero-based index per worker. Keep each case’s source bindings, output and review separate.

```{toctree}
:maxdepth: 1

agentic_kinematic
```

<span id="resume-individual-stages"></span><span id="runtime-selection"></span>

## Observation workers

<span id="run-observation-workers-by-stage"></span>

The manual worker uses the video name to look up calibration. In the split-stem lift above it resolves `lift01-start-end` to `lift01-start` for camera intrinsics and masks, and looks up metric depth under the full video name. The capture checker's `--scene-id` does not change the worker. Use one basename throughout the **worker copy**. Choose a worker sequence ID with no lowercase `end` in any path segment, including the category. The current name resolver checks the entire ID: it turns `Lab/pendulum/lift01` into `Lab/pendulum/lift0`. Use a safe alias such as `Lab/weight/lift01` throughout the worker tree and keep the original ID in the case inventory.

From the checkout root in the configured GRAIL environment, stage the renamed lift in a fresh directory. Copy the MP4 bytes to the calibration basename. Keep the original intact. If the original already has a safe shared basename, keep the same case layout and skip the MP4 rename.

```bash
SOURCE=/workspace/runs/my-lift/generation
ROOT=/workspace/runs/my-lift/observations
VIDEO=Lab/weight/lift01
mkdir "$ROOT" && cp -R "$SOURCE" "$ROOT/generation" &&
  cp "$SOURCE/videos_kling/${VIDEO}-start-end.mp4" "$ROOT/generation/videos_kling/$VIDEO.mp4" &&
  cmp "$SOURCE/videos_kling/${VIDEO}-start-end.mp4" "$ROOT/generation/videos_kling/$VIDEO.mp4" &&
  python scripts/standalone/check_capture.py "$ROOT/generation" "$VIDEO" "$ROOT/frame-zero.jpg"
```

Check the overlay before continuing. The worker now reads the `lift01` MP4, `foundation_pose/Lab/weight/lift01/cam_K.txt`, both frame-zero masks beside it, and `depth_maps/Lab/weight/lift01/000000.png` under `generation/`. Write paths relative to `/workspace/runs/my-lift/inventory.json`. Bind `source_video` to the original `generation/videos_kling/Lab/weight/lift01-start-end.mp4` and `object_mesh` to `generation/mesh/Lab/weight/model.obj`. Keep the original frame-zero pose under `generation/foundation_pose/`. Bind the human prediction, mask, depth tensor and metadata, and object poses under `observations/generation/`. Stage 1 checks each record’s size, hash and resolved location; every file must live within the directory containing the inventory. In the same shell, run the human predictor, then image preprocessing and object tracking:

```bash
OBS=(--config configs/recon_4dhoi/manip_soma.yaml --video_id "$VIDEO" --results_dir "$ROOT")
python -m grail.pipelines.recon_4dhoi "${OBS[@]}" --skip_step2 --skip_step3 --skip_step4 --skip_step5 --skip_step6
```

```bash
python -m grail.pipelines.recon_4dhoi "${OBS[@]}" --skip_step1 --skip_step4 --skip_step5 --skip_step6
```

**Before Stage 1**, open the outputs under `$ROOT/generation/`. Check the human prediction (`hmr_gem_x_native77/$VIDEO.npz`), mask archive (`4dhoi_recon_soma03_cache/masks/$VIDEO.npz`), metric-depth metadata (`4dhoi_recon_soma03_cache/depth_moge3_vitg/$VIDEO.pt.json`) and complete object track (`object_pose_mose3_fp/$VIDEO/pose_estimation_output/poses_in_cam.pkl`). Compare frame counts with the MP4, inspect the masks and verify that the depth metadata names the intended calibration. A worker can log a per-clip error and still finish with `OK`. Stop on an error or a missing, mismatched or incomplete artifact. For static terrain select `configs/recon_4dhoi/loco_soma.yaml` and the actual terrain video. For a collection, select `--dataset` and optionally `--category` in place of `--video_id`. Use `--skip_done` with matching inputs, or give workers disjoint `--job_chunk_idx` values under the same `--num_job_chunks`.

With reviewed contact inputs and a matching configuration, `--skip_step1 --skip_step2 --skip_step3` starts at the optimization, filtering and post-processing stages. For a moving-object tutorial, use the [explicit Stage 1, placement and Stage 2 procedure](agentic_kinematic.md) so the chosen placement is recorded. For the [stairs tutorial](tutorials/stairs.md), use the static terrain fitter. A validated batch result normally contains `hoi_data/hoi_data.pkl` and visualizations under the configured results root. The tutorial's selected-fit and review files are separate deliverables.

GEM-X provides native human motion. The MoGe depth worker and MoSE3 object worker use isolated Python environments. The dynamic object initializer scores metric CAD proposals with FoundationPose, then smooths the selected trajectory while preserving frame zero. The {blob}`hybrid tracking configuration <configs/object_tracking/mose3_fp_hybrid.yaml>` uses a 10 px PnP reprojection threshold, 10 px median-residual threshold and a 0.05 score margin. Post-processing uses a nine-frame, third-order Savitzky–Golay filter with the initial pose preserved. Compare the raw and processed rotations and translations if a handle appears to flip. Static terrain uses its verified fixed pose. The [manual setup](manual_installation.md) gives the worker installations. The CLI can prepare observations by itself. It does not author the video-specific contact map or decide that a fit is ready. The batch flags above were checked at the CLI; the two timed examples use the tutorial requests.
