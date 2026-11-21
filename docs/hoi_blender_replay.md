# Replay a Stage 2 reconstruction in its original Blender scene

Use the exported native `stage2_geometry.npz`, the original room `.blend`, and
the source FPS. The geometry must already use that room's world coordinates.
The importer does not infer a new alignment, change body shape, or retarget to a
different human rig. It replays SOMA through its exported vertices.

```bash
imports/blender/blender --background --disable-autoexec --threads 8 \
  --python grail/visualization/hoi_blender_scene.py -- \
  --geometry /path/to/stage2_geometry.npz \
  --background-scene /path/to/original_room.blend \
  --output-dir /path/to/new_replay_directory \
  --fps 24 --model-label SOMA03 --quality-status PARTIAL \
  --frames 0,20,60,120
```

The saved `stage2_native_replay.blend` contains the original scene and the full
native human/object animation as baked mesh shape keys. Source frame 0 maps to
Blender frame 1. Playback requires no embedded Python or model installation.
The importer reopens the file, compares every source frame's vertices, and
checks the calibrated source-camera projection before rendering. It also saves
input hashes and a verification receipt. It preserves the original scene file.

Inspect the initial views, then use the same command with `--render-only
--frames all` for a complete source/oblique replay. `--render-percent` defaults
to 50; this changes only image resolution. Workbench preview renders retain
the room geometry and use diagnostic colors (blue human, orange object).
Original materials remain in the saved scene.

This is visualization, not Stage 2 contact refinement or physical validation.
Use the checkpoint's existing quality status. Importing successfully does not
resolve inherited contact gaps, foot sliding, or depth/alignment defects. Baked
shape keys are for replay; use native model controls for subsequent refinement.

If the feet penetrate the original floor, use the explicit
[scene-ground constraints](foot_ground_support.md) during numerical Stage 1/2.
Keep the imported scene and native replay unchanged as the comparison baseline.
