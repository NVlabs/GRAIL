# Topo-inspired hand candidate initialization

`tools/grasp/topo.py` packages the **Topo-inspired structure stage** without
importing a historical experiment directory. Standalone distal-pad correction
is disabled by default and remains available as an explicit ablation. The tool
generates a bank of free-wrist/native-finger poses at explicitly chosen source
frames. It does not mount the hand on an arm, select a candidate, propagate a
hold, or run whole-body SOMA. The subsequent sequence and whole-body contracts
are in [hand initialization](hand-initialization-workflow.md).

## Objective and geometry

The structure stage builds a fixed interaction graph from the source human
hand and deterministic original-object face centroids in object coordinates.
It uses the existing `grasp_interaction_graph.py` and `grasp_paper_objectives.py`
implementation: Delaunay adjacency (recorded kNN fallback for degenerate input),
distance-weighted graph coordinates, and optional adjacent bone-direction
differences. Native FK and free wrist variables move the robot nodes toward
those source graph coordinates. The reference graph and source posture stay
fixed through optimization. This is **TopoRetarget-inspired**, not a complete
paper reproduction or learned grasp generator.

The graph's anatomical correspondence uses wrist, thumb, index and middle nodes.
That correspondence is separate from contact roles: thumb regions guide the
robot thumb; the other source fingers' regions may be pooled for the two native
long fingers, which must still be declared as separate contact-patch members.
The caller supplies phase-supported regions; the tool does not infer them.

When explicitly enabled, the pad stage starts from the corresponding structure-stage candidate and
removes the graph term. Its attraction and anchor selection use the original
distal closing-side surfaces, selected by the packaged `grasp_distal_pad.py`
anatomy heuristic. This prior is explicit and only suitable when the intended
grasp uses those surfaces. Other grasp types can use the general surface-bank
route. **All original collision samples, triangles and screening pairs remain
in every executed stage**; the pad mask never reduces collision geometry. Native limits,
source-relative wrist bounds and source priors remain active.

Skipping this separate correction pass does not remove contact or collision
objectives from the structure bank or subsequent trajectory optimizer. The
sequence optimizer still samples the original contact surfaces; surface
sampling and an additional pad-only optimization pass are separate choices.

## Configuration and execution

Install the existing GPU extras and run in the chosen CPU/CUDA runtime:

```bash
python tools/grasp/topo.py --config hand_keys.json --out runs/hand_keys_v001
# Equivalent installed-package entry:
python -m agentic_retargeter.topo_hand_initialization \
  --config hand_keys.json --out runs/hand_keys_v001
```

Run this CLI directly in a compatible CUDA runtime. The generic sandbox-owned
`surface_grasp` GPU worker does **not** accept `multistart`, `topo_objective`,
`contact_surface`, or `warm_start_bank`; it cannot launch this multistart route.
A deployment must provide direct execution or separately extend and validate the
typed worker contract. Do not send invented worker operations or bypass the
agent's sandbox to obtain CUDA access.

The output directory must be new. The configuration has this shape:

```json
{
  "bank": {
    "inputs": {
      "native_npz": "original_native.npz",
      "model_xml": "original_robot.xml",
      "object_mesh_npz": "original_object.npz",
      "hands": "REPLACE_WITH_EXPLICIT_HAND_BINDINGS"
    },
    "seed_frames": {"right": [27, 73, 120]},
    "device": "cuda:0",
    "screen_backend": "numba",
    "samples_per_geom": 512,
    "translation_limit_m": 0.16,
    "rotation_limit_rad": 1.0,
    "multistart": {
      "count_per_frame": 32,
      "seed": 20261024,
      "translation_radius_m": 0.07,
      "rotation_radius_rad": 1.0,
      "finger_radius_rad": 0.8
    },
    "contact_patch": "REPLACE_WITH_EXPLICIT_PER_DIGIT_PATCH_POLICY"
  },
  "source_arrays": "source_human_and_object.npz",
  "topo_steps": 200,
  "pad_steps": 0,
  "topo_objective": {"graph_weight": 500.0, "direction_weight": 0.01}
}
```

This is a schema illustration, not a runnable case: replace both marked fields
with the original model's named hands, mesh IDs and patch policy described in
[surface-grasp inputs](../tools/grasp/README.md). Source frames, regions and
bounds must come from the case; do not reuse the example frame numbers blindly.
`bank` accepts the usual surface-grasp settings except `steps`,
`topo_objective`, `contact_surface` and `warm_start_bank`, which this runner owns.
Explicit `contact_patch` and `multistart` are required. The structure stage
defaults to 200 optimizer steps; `pad_steps` defaults to 0 and skips the
standalone pad pass. Set `pad_steps: 400` explicitly to run the historical
two-stage variant as an ablation. Each enabled stage allows at most 2000 steps.
When enabled, pad refinement uses the same rows: 32 starts at 3 frames means
96 candidates, not 192. Existing configs that explicitly set a positive
`pad_steps` retain that behavior; changing the default does not rewrite or
rerun historical experiments.

All input paths resolve relative to the config. `source_arrays` must contain
the same-time human joint names, world positions and rotations, plus object
rotation/translation/scale exactly matching the native input. Graph options
also allow `graph_decay`, `object_sample_count`, `fallback_neighbors` and
`node_set` using the packaged objective's validation. Missing source data is an
error; this runner never silently reruns SOMA or substitutes an old result.

This contact-key runner checks the **complete original** object rotation and
translation using the shared static-object policy. An exactly static object
skips contact-key search with an explicit error directing the caller to the
`hand_sequence`/SOMA no-contact path. A stationary prefix followed by motion is
still a moving case. An optional `is_static_obj` flag must agree with the source;
it cannot override the policy. The general one-frame surface bank remains
available separately and does not infer a whole-sequence contact policy.

## Receipts and next stage

Each executed stage saves `executed_config.json`, `bank.json` and `summary.json`
under `topo/` or, when enabled, `pad/`. A skipped pad pass creates no `pad/`
directory. The root `report.json` records configuration/code hashes, wall times,
executed stage counts/rejections, `skipped_stages`, and the actual `final_stage`,
`final_bank`, and `final_bank_sha256`. Downstream selection must consume those
final-bank fields instead of assuming `pad/bank.json` exists. For example:

```python
import hashlib
import json
from pathlib import Path

report = json.loads(Path("runs/hand_keys_v001/report.json").read_text())
bank_path = Path(report["final_bank"])
assert hashlib.sha256(bank_path.read_bytes()).hexdigest() == report["final_bank_sha256"]
bank = json.loads(bank_path.read_text())
```

Candidate IDs stay bound
to hand, source frame and start index. Pad warm starts require the exact parent
bank SHA256, native/model/object binding, named finger order and candidate IDs.
Finite proper rotations, original hinge limits and wrist bounds are checked
before optimization. The actual warm delta and inherited rejection record are
retained on each pad candidate.

A rejected structure pose can initialize a diagnostic refinement, but its
rejection remains visible. The pad candidate gets a fresh screen; neither a
successful optimizer nor a low loss accepts it. Original whole-hand screens
stop at the first witness, so they do not establish complete clearance.
Missing spaced contact coverage remains a shortfall. Every emitted bank stays
`UNREVIEWED`; selection, phase-aware sequence fitting, mounted reachability,
scene/body collision and final motion review happen afterward.

To initialize a full sequence, explicitly inspect and choose rows from a bank,
retaining its hash, row IDs and rejection evidence. Supply their object-relative
wrist position/rotation and ordered finger angles to
`hand_sequence_seed.build_seed` for declared strict holds, or
`hand_sequence_path_seed` for varying support/sliding paths. These existing
utilities preserve the original clock and immutable source/body/object arrays.
Never turn a support or slide into a constant no-slip hold. Their output is a
warm start for `hand_sequence` or `hand_sequence_pair`, followed by the explicit
SOMA handoff; the Topo bank CLI does not automatically select or propagate a
candidate. Selection as a diagnostic seed does not remove its rejection.

For lower-level use, `surface_grasp.propose` exposes the same opt-in hooks:
`topo_objective`, `contact_surface: "dex3_distal_pad"`, and
`warm_start_bank: {"path": "bank.json", "sha256": "..."}`. Without these
options its previous objective and initialization remain unchanged. The
multistart route does not replace the packaged all-frame `hand_sequence.py`
route. That solver now defaults to Topo → temporal optimization, with
`mode: "temporal"` and `pad_steps: 0`; an explicit positive `pad_steps` inserts
the optional pad pass. It emits snapshots only for executed stages, plus the
source and any loaded warm start, and exports the actual last result as
`hand_sequence.npz` with `final_stage` recorded in its report.
