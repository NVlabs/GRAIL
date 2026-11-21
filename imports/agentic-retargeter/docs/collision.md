# Newton collision diagnostics in Blender

The optional backend is an external contact-detection backend that is not distributed with GRAIL. It provides the importable `ho_contact` package with a pinned Newton submodule; the adapter verifies the exact backend and Newton revisions recorded as `HO_CONTACT_COMMIT` and `NEWTON_COMMIT` in `src/collision.py` and fails on any other checkout.

This version supplies collision queries and contact extraction. It does **not** supply a robot pose repair solver. The integration queries the currently displayed original meshes, displays paired contact points, their connecting segments and Newton normal arrows, and saves bound diagnostic evidence. Optional wrist-local hand views show the same samples alongside the mounted robot. It does not change joints, animation, original geometry, materials or the source camera.

## External runtime

Newton/Warp run in a separate Python environment. The Blender extension bundles only our adapter and visualization; it does not bundle upstream code, model assets, RP2 loading or the upstream UV remeshing pipeline. Keep the checkout at the stated revision with its implementation unchanged. No remote model API is used.

```bash
# /your/tools/ho_contact_detection: a separately obtained checkout of the
# external backend at the pinned revision, with its Newton submodule initialized.
python3 -m venv /your/envs/ho-contact
/your/envs/ho-contact/bin/python -m pip install numpy scipy warp-lang==1.9.1
/your/envs/ho-contact/bin/python -m pip install --no-deps -e /your/tools/ho_contact_detection/third_party/newton
```

The adapter imports `ho_contact.collision` directly from the configured checkout. Its collision-only route does not require `rp2_loader`, PyTorch, xatlas or the contact texture exporter. Use the upstream README for their RP2/UV CLI; those inputs and exports are a separate workflow. CPU queries need no GPU. Set the device explicitly to `cuda:0` for a compatible Warp/CUDA runtime. The first query may compile kernels.

The verified CPU environment uses Python 3.12.3, Newton 0.1.1a2 at the pinned commit, Warp 1.9.1, NumPy 2.2.6 and SciPy 1.15.3. Use Python 3.11+ for this adapter. The upstream declared Warp 1.7 minimum does not import successfully with this Newton revision; keep this environment separate from the SOMA retargeter runtime.

## Scene configuration

Create a JSON file using actual Blender mesh object names. List every desired pair explicitly; no robot adjacency or assembly exclusions are inferred. For example:

```json
{
  "schema_version": 1,
  "meshes": ["RightThumb", "RightPalm", "OriginalPan"],
  "pairs": [
    ["RightThumb", "OriginalPan"],
    ["RightPalm", "OriginalPan"],
    ["RightThumb", "RightPalm"]
  ],
  "runtime": {
    "python": "/your/envs/ho-contact/bin/python",
    "repo": "/your/tools/ho_contact_detection",
    "device": "cpu",
    "timeout_s": 120
  }
}
```

Include all relevant object components, fingers, palms, arms, cross-hand pairs and environment meshes for the desired review scope. An unlisted pair is unchecked. Existing assembly overlaps remain observations, not automatic exemptions. Unsupported deformation modifiers and shape keys are rejected; the adapter never remeshes the source to obtain a detection result.

In extension **0.7.0**, open **N → GRAIL G1 → Newton Collision Diagnostics**:

1. Open the matching scene, enable its trusted native preview if it has native robot controls, and move to the required frame/subframe.
2. Select the collision JSON and a new query output folder.
3. Click **Run Newton Collision Query**. This is an explicit bounded foreground query; Blender waits for the external process. Its output folder retains the input, report and logs.
4. Inspect the overlay together with the original robot and object. The panel lists each pair's returned sample count. Query positions use metres; the Blender bridge applies the scene unit scale for display.
5. Use **Export Collision Evidence** to save a current bound report. Save a new `.blend` normally to retain the editable scene and diagnostic snapshot. **Clear Collision Overlay** removes owned diagnostics only.

For a separately executed query, use **Export Collision Input**, run the [collision CLI](../tools/collision/README.md), and choose its report with **Load Newton Result**. Loading requires the same frame, meshes, transforms and pair selection. It never runs external Python automatically on file open.

Frame, geometry or native-control changes invalidate the snapshot. Refresh the query after edits; stale diagnostics cannot be exported as current evidence. Dependency revisions, source hashes, device and query settings accompany each backend report. These records describe the queried geometry and software, not acceptance of the motion.

## Contact normals and simultaneous canonical hands

Add optional hand bindings to the same configuration. Each wrist is an existing rigid scene object, and each hand mesh must also be a configured collision input. For example:

```json
{
  "hands": [
    {"name": "right", "wrist": "RightWrist", "meshes": ["RightThumb", "RightPalm"]}
  ],
  "visualization": {
    "point_limit": 2048,
    "normal_limit": 192,
    "point_radius_m": 0.0015,
    "normal_length_m": 0.025
  }
}
```

After running or loading a query, click **Show Canonical Hand View**. This splits the active 3D viewport: the mounted robot remains on the left and the persistent wrist-local hands appear beside it, with the Newton panel on the right. Both views use the same source frame and sampled contact indices. The hand copies preserve the current finger articulation; removing wrist translation and rotation does not replace the hand with a default pose. Wrist transforms are included in stale checks.

Green points represent mesh A, orange points mesh B, and red segments join the paired query points. Cyan arrows start at the A points and show the Newton query normal oriented **from mesh B toward mesh A**. They have a fixed display length and do not encode force or penetration depth. The canonical view rotates these directions into the wrist frame without translating them. This is the detector's normal, not a separately estimated finger-pad normal. Zero or undefined normals are omitted; legacy reports without normals remain usable for points but require a new query to display arrows.

Display limits are balanced across pairs; the full result arrays remain in the report. **Refresh Contact Display** applies changed hand bindings or display settings to the existing current result without rerunning Newton. Running or loading a new result refreshes an already-open canonical viewport too. The canonical hand geometry follows the timeline continuously. Its contact layers require matching evidence for each pose; use the sequence cache below for playback. Missing or invalidated evidence turns the hands neutral gray and hides only contact glyphs, with a pending label.

## Canonical contact map

Add `contact_map` alongside the hand and visualization bindings:

```json
"contact_map": {
  "enabled": true,
  "threshold_m": 0.005,
  "targets": ["OriginalPan"]
}
```

Only the canonical hand copies receive the map. The mounted robot keeps its materials and contact glyphs. Blue means the original hand vertex is at least the configured distance from the target surface; yellow means it is close to zero distance. A legend shows the range in millimetres. Points and normal arrows remain visible on top. **Canonical map** in the panel can use the config, override the measured range, or turn coloring off; **Refresh Contact Display** applies these settings without rerunning Newton when the pose is unchanged.

This map measures unsigned distance from every original hand vertex to the original target triangles, with `score = clip(1 - distance / threshold, 0, 1)`. It is independent of Newton candidate density and its 10 mm margin. The report retains distances and scores in original vertex order, target/mesh bindings, frame and threshold. Colors interpolate across the unchanged triangles. Near-surface color is not proof of contact, penetration clearance or contact force; it is not a prediction or authored contact target. Missing target bindings are rejected instead of painted as zero contact.

## Sequence playback

In **Canonical sequence playback**, expand **Sequence cache files**, select a new output folder, and click **Build Full Sequence Contact Cache**. The explicit build covers the scene's inclusive integer frame range and restores the original selected frame and controls afterward. It captures original geometry once, computes dense per-vertex surface distances for every frame, and queries Newton in one external worker with reusable per-pair collision scenes. Blender waits for this bounded build; no motion is edited.

The resulting `sequence_cache.npz` is loaded automatically. **Load Sequence Contact Cache** can reuse it in a matching scene. Click **Show Canonical Hand View** to keep the mounted robot and articulated canonical hands visible together while playing or scrubbing. Both displays read the current frame's cached points and normals; canonical copies also show that frame's contact map. Playback itself launches no Newton processes.

Each entry is bound to its source frame, original geometry, evaluated pose, native controls, wrist basis, units and configuration. Uncached subframes are not interpolated as measured contact: the current articulated hand remains visible in neutral gray until matching evidence is available. Geometry, pose or configuration edits invalidate the affected evidence; unchanged matching frames remain reusable. Enable automatic contacts to recompute a missing or edited frame after playback/editing pauses. A fresh current-frame query updates the sequence in memory.

Use **Save Sequence Cache Revision** with a new NPZ path to retain those updates, then save a new `.blend`. The scene stores the cache path; keep that file alongside it when moving the scene. Reloading a cache does not execute Newton. Enable the scene's trusted native preview to animate native robot controls. Clearing the collision overlay removes the persistent view and disables automatic queries.

Caches contain sampled Newton glyph arrays, dense original-vertex distances, pose signatures and backend provenance. Full per-pair counts and sampled-index provenance remain in the build's `report.json`; full single-frame query reports remain in their query folders. Cached diagnostics retain the same coverage limitations as single-frame queries.

## Automatic refresh in the add-on

Newton is integrated as a query backend of the add-on, with an isolated external Python runtime. It can compute from current Blender geometry; precomputed reports are optional. Keeping Warp/Newton outside Blender avoids coupling their binary dependencies to Blender's bundled Python.

Configure the runtime, mesh pairs, hands and map targets once, choose an **Automatic query folder**, then click **Enable Automatic Contacts**. After a short pause in editing or frame selection, the add-on runs a managed background query and refreshes the mounted points/normals and canonical map together. Playback does not launch a query for every frame. Superseded results are discarded, and pending work is cancelled when the feature is disabled or the add-on is unloaded. Status and per-query logs remain accessible; the viewport is available while Newton runs.

The enabled config can resume when its sequence is reloaded in the same Blender session. **Remember configured runtime**, followed by saving Blender preferences, retains that binding across restarts. The remembered executable, config and output binding lives in add-on preferences, not solely in a `.blend`; changed bindings require explicit enablement again. Native sequences still need their exact preview enabled so the displayed mesh follows the editable joints. Loading a file does not implicitly execute embedded scene scripts. Unconfigured scenes remain available for manual setup.

During automatic refresh, stale contact layers are hidden until the new snapshot is validated. A configured sequence keeps the current canonical hand geometry visible in neutral gray; valid cached frames display immediately and need no new query. **Run Newton Collision Query** requests a fresh automatic query when the feature is enabled. **Disable Automatic Contacts** stops it; **Clear Collision Overlay** also disables it before removing the generated diagnostics.

## Interpret the result

The pinned Newton mesh path uses vertex-to-surface queries with a default **0.01 m candidate margin**. Candidates can occur on nearby surfaces before intersection. Edge/face-only crossings, deep penetrations beyond the query radius, open meshes and inconsistent winding can produce missed or ambiguous contacts. It is not an exhaustive original-triangle intersection audit.

The upstream `penetration_depths` array is computed as the Euclidean distance between paired query points. Our report calls this `point_distances_m`: **unsigned paired-point distance**, not signed penetration depth. A zero-contact report means only that this detector returned no candidates for the listed pairs and pose. It does not prove collision clearance, force closure or physical stability.

Use these samples to locate likely contact regions and prioritize edits. Retain the existing original-mesh, source-video and temporal review for motion acceptance. The hand dashboard's unsigned proximity maps, authored contact targets and Newton candidates are separate kinds of evidence.
