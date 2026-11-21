# Local video grasp planning

Implementation is directly in `src/video_perception.py`, `src/video_grasps.py`
and `src/video_cli.py`. `run.py` is a thin checkout launcher.

This is a local, single-person/single-target-object evidence frontend. It uses
learned object/hand detection and a heuristic temporal planner. The planner
does not call a language-model API or edit a Blender scene.

## Setup

Use a separate Python environment from Blender. Install the package and optional
video dependencies, choosing the appropriate PyTorch build for your machine:

```bash
python -m pip install -e '.[video]'
```

`requirements-local.txt` records the independently tested local runtime. Download
the original model assets separately; weights are not bundled in this repository:

- [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- [SAM2.1 Hiera small](https://huggingface.co/facebook/sam2.1-hiera-small)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index)

Keep complete local Hugging Face model directories (configuration, tokenizer,
processor and weights) and the MediaPipe `.task` file. Runtime loading uses
`local_files_only=True`; it does not download weights or contact an inference
service. `ffprobe` must be installed for exact source presentation timestamps.

## Analyze a video

```bash
python tools/video/run.py analyze \
  --video /data/source.mp4 --object-prompt 'frying pan' \
  --detector-dir /models/grounding-dino-tiny \
  --segmenter-dir /models/sam2.1-hiera-small \
  --hand-model /models/hand_landmarker.task \
  --device cuda --output runs/video_001
```

The installed equivalent is `python -m agentic_retargeter video analyze ...`.
`--device cpu` is the default. Analyze every frame by default; a larger
`--stride` is explicit and may leave too little temporal evidence to identify
a short grasp. Use a new output directory each time.

You can separate heavy perception from lightweight planning:

```bash
python tools/video/run.py extract --help
python tools/video/run.py plan \
  --observations runs/video_001/observations/observations.json \
  --output runs/video_replan_001
```

Outputs:

- `observations/observations.json` and `masks/`: exact frame/PTS mapping, source
  hash, local model hashes, object identity uncertainty and tracked landmarks.
- `plan/grasp_plan.json`: settings, coverage, unknown intervals, evidence hashes,
  hypothesized events, selected source frames and image-space anchors.
- `plan/grasp_events.json`, `keyframes.json`, `contact_candidates.json`: small
  views of that bound plan for downstream tools.
- `plan/keyframe_*.png`, `keyframes_*.png`: English source views and contact zooms.

## Interpretation and Blender handoff

The planner checks fingertip proximity to the **predicted visible object mask**,
finger curvature/thumb opposition, sustained observations, and supporting
hand/object image co-motion. Scores are heuristic support, not calibrated
contact probabilities. Handedness confidence does not measure landmark accuracy.
Object identity ties, hand occlusion and sampling gaps remain uncertain.

`first_near_frame` means the first near observation connected to a supported
candidate. It is not a verified first-touch frame. `object_image_motion_frame`
does not establish lift-off, and ending a support cue does not establish release.
An empty event list can mean missing evidence. Preserve the coverage report.
Events are supported evidence windows: flickering cues may split one physical
hold into several windows. Do not count them as distinct grasps. A track with
inconsistent handedness is exported as `side: unknown`; it cannot select a
left/right native robot joint group automatically. Clip endpoints and sustained
object displacement also provide context keys even when hands are missing.

Review selected frames against the source, then use their `source_frame` indices
in the local Blender configuration's `spec.observation_frames` (at most 32).
Verify the source/native motion frame correspondence; G1 source frame 0 maps to
Blender frame 1. Continue from the chosen `.blend` with a new `blender-edit` round.

During that review, compare each hand's grip across keys and the intervening
video. A supported continuous rigid hold should use one shared object-relative
robot palm pose and hand/contact layout across its keys and interpolation.
Record regrasp, sliding, articulation and uncertainty separately, following the
[shared hold guide](../../docs/hand-shared-hold-feasibility.md).
The current CLI does not classify same-grasp identity, recover a 3D palm pose, or
compile shared-pose solver constraints; these are host review and downstream
authoring steps. Fragmented detector events alone must not create different
robot grasps.

`object_silhouette_xy_px` is an image coordinate. Hidden surfaces, depth, surface
normals and force closure cannot be inferred from it. Author or fit native 3D
anchors using the bound original object and camera, then inspect them in the
canonical-hand/contact dashboard. This frontend does not automatically convert
2D anchors into a 3D prediction map or run native edits. All outputs are UNREVIEWED.

Actual pan testing used all 121 frames: masks covered 121, hands 106, and both
hands 90. Automatic object-context hand crops recovered an early near candidate
at frame 24, supported evidence at 25 and confirmation at 28, consistent with
source inspection around initial acquisition. Cue flicker still fragments the
hold and anatomical handedness is inconsistent. This is a reusable proposal
stage, not a validated grasp-onset benchmark. Extinguisher hands were largely
missed by the initial full-image hand-detector probe; that case has not yet been
validated with the updated crop frontend.
