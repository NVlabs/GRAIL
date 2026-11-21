# Hand and contact inspection

The Blender extension provides read-only inspection beside its native motion editor. In version **0.7.0**, open the matching robot scene, enable its native preview, move to a source frame, and load an inspection configuration under **N → GRAIL G1 → Hand & Contact Dashboard**. Click **Open Inspection Dashboard** to see all maps together inside one Blender viewport.

Columns show the declared hands in wrist coordinates and the object with both hands. Rows show **Measured proximity**, **Predicted / authored**, and **Difference**. All tiles share the same source frame and threshold. The display uses actual original mesh copies with normalized tile framing, while distances remain in native metres. No external image viewer or camera switching is needed.

**Return to Editing View** restores the previous local viewport state; the source scene camera stays unchanged. After reopening a file, it returns to the source camera if session-local navigation history is unavailable. **Render Inspection Views** exports one combined dashboard PNG and the bound measurements JSON. Choose **Individual views** to use the earlier camera-by-camera layout.

**Follow timeline (on pause)** is optional and is not saved as enabled in a `.blend`. It marks the dashboard stale during playback and refreshes after scrubbing or pose edits settle. It evaluates an already enabled native preview; it never enables scene Python itself. Native scenes without an enabled preview receive an explicit error. With Follow off, use **Load / Refresh Inspection Snapshot** after changes; the dashboard remains a labeled snapshot. Clear, Return, file load, and add-on removal stop automatic following. These tools are not a live contact solver and do not promise playback-rate map recomputation.

The reusable tools cover:

| Tool | What it shows |
| --- | --- |
| Canonical hand views | The current articulated hand transformed into its declared wrist coordinate system, with axes and digit labels. Global wrist placement is removed; the finger pose is preserved. |
| Measured contact map | Unsigned vertex-to-triangle distance from hand to object and object to the union of the configured hands. Close surfaces receive higher proximity scores. |
| Prediction / proposal map | Imported per-vertex scores, with producer, source reference, exact mesh hashes and source frame. Model predictions and authored hypotheses retain distinct labels. |
| Difference map | Predicted score minus measured proximity score, only where prediction coverage exists. Positive means predicted contact exceeds measured proximity. |
| JSON evidence | Geometry identities, coordinate/frame conventions, threshold, measured values and prediction coverage for reproducible agent observations. |

“Measured” means measured from the current geometry. It is not sensor contact, ground truth, signed penetration, normal agreement, force closure or stable grasp certification. Vertex samples can miss triangle-interior intersections and containment. The original meshes and their materials, motion controls and source camera are preserved; generated diagnostic copies have their own display materials.

Colors interpolate between original vertices across each triangle. Coarse meshes can therefore show broad color bands; the numerical report retains the actual per-vertex measurements. The inspector does not remesh or densify source geometry.

The [Newton canonical contact view](collision.md#canonical-contact-map) can show measured heatmaps, contact points and normal arrows together beside the mounted robot. Its sequence cache keeps the articulated hands, measured maps, points and normals visible throughout playback. Automatic refresh fills missing or edited frames after a pause. The dashboard described here remains the separate route for comparing measured maps with imported predictions or authored hypotheses.

## Bind a scene

The configuration names the actual scene objects. Do not infer anatomy from an arbitrary link name or copy another case's mapping blindly. For G1 + Dex3, the opposing digit corresponds to the human thumb; the other two digits collectively represent the remaining four human fingers. Their individual roles depend on the video/reference-motion grasp or support phase, rather than a fixed human index/middle correspondence. `canonical_rotation`, when supplied, explicitly defines the wrist-local orientation used for display; identity shows the declared wrist axes without guessing a palm normal or mirroring the hand.

Proposed contact regions may cover distal/proximal surfaces, finger sides or
backs, and the palm when supported by the observed arrangement. They need not
be finger-pad masks. Bind regions on both the original robot hand and object
meshes, preserve phase and evidence-frame identities, and mark hidden 3D choices
as authored hypotheses. These inspection maps are distinct from whichever
discrete material anchors a solver consumes; a few displayed anchors do not
mean only those points are expected to touch. See the
[hand refinement](object-relative-hand-refinement.md)
for object-relative contact targets.

```json
{
  "schema_version": 1,
  "object": "Original_Object",
  "threshold_m": 0.005,
  "hands": [{
    "name": "left",
    "wrist": "G1_left_wrist_yaw_link",
    "meshes": ["PalmMesh", "ThumbMesh", "Long1Mesh", "Long2Mesh"],
    "digits": {
      "thumb": ["ThumbMesh"],
      "long_1": ["Long1Mesh"],
      "long_2": ["Long2Mesh"]
    }
  }]
}
```

Replace the example mesh names with those in your scene; add the right hand as another entry. Positions are in metres. Source frame 0 is Blender frame 1. The inspector reads the current scene pose, so enable/update the native preview before refreshing. Unsupported deformation modifiers or shape keys must be resolved into a validated original-mesh binding instead of treating altered vertex indices as matching predictions.

For scripted use inside Blender:

```python
from agentic_retargeter.blender_inspection import (
    build_inspection, render_inspection, clear_inspection)

report = build_inspection(bpy.context.scene, config, source_frame=48,
                          layer="measured")
for index, camera in enumerate(report["cameras"]):
    render_inspection(bpy.context.scene, report, camera,
                      f"/new/evidence/hand_{index}.png")
# Save report as JSON alongside the images. Clear removes owned diagnostics only.
clear_inspection(bpy.context.scene)
```

The standalone Python package must be importable for that example. An installed ZIP supplies the equivalent bundled module and sidebar operators without installing Python dependencies into Blender.

## Import predictions

Put a `predictions` object in the inspection JSON. The first version imports predictions; it does not include a trained contact predictor. A separate predictor, dataset adapter or authored proposal can supply the schema below. Agent-authored guesses use `authored_hypothesis`, never a ground-truth label. The inspector does not call a model API.

```json
{
  "schema_version": 1,
  "provenance": {
    "kind": "model_prediction",
    "producer": "your-contact-model/checkpoint-id",
    "reference": "source-case-and-evidence-version"
  },
  "mesh_bindings": {"ThumbMesh": "<SHA256 from the inspection report>"},
  "maps": [{
    "source_frame": 48,
    "mesh": "ThumbMesh",
    "values": [0.1, 0.8, 0.9]
  }]
}
```

The three values are illustrative: supply exactly one finite value in `[0,1]` per original vertex, in original order. `mesh_digest(vertices, triangles)` binds local mesh coordinates and triangle order. Export inspection evidence first to obtain each mesh's actual identity. Hash, length and frame mismatches are rejected; missing mesh/frame coverage is explicitly unavailable, never an all-zero prediction. There is no automatic interpolation between prediction frames.

Measured scores are `clip(1 - distance_m / threshold_m, 0, 1)`. Imported scores may have different calibration, so the difference map is a diagnostic comparison rather than an error metric with statistical meaning. Threshold changes affect measured scores and must accompany comparisons.

## Local editing loop

Add the same configuration as `spec.inspection` in a [`blender-edit` config](blender-agent.md), with optional `spec.inspection_layers`:

```json
{
  "inspection": {"schema_version": 1, "object": "Original_Object", "hands": []},
  "inspection_layers": ["measured", "predicted", "difference"]
}
```

For the combined Blender dashboard as the agent observation, set `spec.inspection_layout` to `"dashboard"` and omit `inspection_layers`. The local worker produces one dashboard image per before/after frame with all three layers, alongside the mounted robot views. The default `"views"` retains individual-camera observations. The host assistant can review these files through its available tools; the worker makes no inference calls.

The empty hand list is a placeholder; supply the complete binding shown above. The worker requires the inspection wrist and mesh parent chains to match its declared native FK bindings, and the inspected object to follow its frozen object binding. It updates actual native FK, computes inspection maps at every configured observation frame before and after each edit, renders the diagnostic views, and binds the JSON/images by hash. Compact per-mesh measurements accompany the normal mounted robot views; dense per-vertex arrays remain in the bound JSON report. Keep the prediction binding fixed across comparable rounds, and record any deliberate target change. Rendering uses Cycles on CPU.

The loop still edits only declared native joint keys. Continue from the latest chosen `.blend` and save the next round to a new checkpoint; refresh the dashboard after pose or frame changes. An available Blender MCP connection can use the same add-on operators, while foreground Blender Python remains the local fallback. These tools add observations; they do not authorize new joints, change source timing, or accept a motion. Reopened candidates retain `UNREVIEWED` status.
