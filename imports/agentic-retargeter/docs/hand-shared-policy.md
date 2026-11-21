# Shared numerical policy for hand initialization

Version 004 adds the **same bound static support geometry to hand initialization**.
Topo key proposals and their original-triangle screens include the packed native
scene. Base and refined paired sequences query that scene at every source frame
and the configured half-frame collision samples, including approach and release
where hand/object contact is inactive. Object-relative hand samples are converted
to native world coordinates at the matching time; the table stays fixed. All
packed scene triangles are retained, without object scaling or hull replacement.
The source body, object motion, protected first hand frame and scene remain fixed.

The profile uses scene weight 1 in both the bank and paired solves; paired scene
clearance uses the same sampled mean/peak form and global peak coefficient as its
object clearance. Existing low-level configurations remain scene-disabled unless
they explicitly set `scene_clearance_weight` (bank) or `scene_collision` in the
sequence loss weights. Frozen v003 experiment code/configurations are unchanged.
Missing scene data is reported as unknown, never as clearance. A source pose
already colliding in a protected frame remains an inherited limitation.
Reports bind the exact scene fingerprint and effective settings. Sampled signed
distances are optimization guidance; original whole-mesh and subframe review
remains required. SOMA still needs its own binding of the same support scene.

The agent identifies intervals with the same object-relative grasp and records
the source frames supporting that interpretation. It labels support, sliding,
approach, release and uncertainty separately. Native anatomy, input bindings,
contact masks and inherited object regions remain case data. Contact-mask
conflicts require an explicit upstream evidence correction; changing a weight
is not an alternative to correcting the interpretation.

The optimizer uses one shared object-relative wrist position/orientation and
finger pose per observed grasp interval. Different adjacent grasp IDs receive
different controls; repeated IDs separated by another phase do not couple those
intervals. Support/slide and unknown intervals remain per-frame. If a grasp
begins at frame zero, zero is explicitly excluded from sharing and stays at the
original raw SOMA pose. The remaining frames still enter the same temporal
objectives; the exception is recorded without relabeling the source phase.

The profile fixes 16 starts per key, 200 Topo steps, zero standalone pad steps,
300 base paired steps followed by 300 refinement steps, all
source/contact/collision/temporal weights, sampling,
learning rates and source-relative bounds. Grasp keys are interval midpoints;
support/slide keys are their start, midpoint and end. A fixed evidence ranking
chooses an unreviewed hold seed; support/slide uses the existing path DP over up
to eight candidates per key. Rejected candidates retain their reasons. This
does not establish a valid grasp or full-body reachability.

The routine pipeline selects one hand trajectory for the downstream SOMA solve.
The 16 Topo starts search for key-pose initializations; they do not each run a
complete hand sequence and SOMA solve. Running several complete trajectories is
a separate, explicitly requested experiment, not part of this shared profile.
Adding static scene clearance does not change that execution policy.

Long key lists are divided deterministically into batches of at most 128
candidates per hand (eight keys at 16 starts). No key or start is removed; random
perturbations are keyed by hand/frame/start and remain identical across batch
partitions. Each bank retains its own input and configuration hash, and a bank
index preserves all candidate-selection provenance.

Before the paired solve, the shared seed is projected into the intersection of
the unchanged source-relative wrist bounds over its entire hold and allowed
transition buffer. The buffer's base starts at the original source pose. This
keeps one shared pose and the declared phase boundaries; it does not loosen
bounds or improve contact by definition. Empty translation intersections or a
failed finite rotation-feasibility search are reported with attempted values.
The original warm seed and projection receipt are retained. Every proposed
shared update then passes the actual all-frame decoder's feasibility check,
using a common backtracked step fraction when necessary. Ordinary per-frame
base controls retain their native projection before that check; shared/blended
outputs are never clamped independently frame by frame.

Transitions request 0.25 seconds (six source frames at 24 FPS). The same rule
clips buffers at protected/uncertain/support/slide frames and neighboring holds;
competing buffers are assigned to the nearest hold, with ties to the earlier
interval. Requested and effective lengths are reported. There is no timing
resampling or guarantee of smooth derivatives at every discrete join. Midpoint
collision collocation is enabled for every original frame interval, alongside
original-frame sampling; finite sampling is not complete collision clearance.

Removing the separate pad pass does not remove the sequence contact objective
or its current original distal-surface sample membership. Material anchors and
normal alignment are not added by this route. Entirely static source objects
skip contact-key search and disable attraction/material terms, while collision,
source fidelity and temporal terms remain. Their source annotations can be empty.

## Filter and contact-aware temporal refinement

The base solve retains the v002 numerical settings so the tested warm-start
lineage is preserved. Apply the [phase-aware filter](hand-sequence-filter.md)
with Savitzky–Golay window 9 and polynomial order 3. It uses object coordinates
during eligible active contact phases and world coordinates otherwise, with
SO(3) log/exp rotation handling. Raw frame zero and complete strict-hold poses
remain exact. Filtering can change contact and never constitutes acceptance.

The final paired solve starts from that filtered sequence. It sets contact
weight 256, enables four distinct samples per digit with 3 mm minimum spacing,
and multiplies every existing temporal weight by 100. Source priors, native
limits, source-relative bounds, collision terms, source timing and phase
semantics remain unchanged. No material anchors or normal-alignment objectives
are fabricated. An optimized strict hold may move coherently as one shared
control; the original first hand frame remains protected.

An exported sequence already contains the shared-hold boundary blend. If using
it as new base controls would apply that blend again and violate a source bound,
the solver re-encodes only the offending boundary position/rotation/finger
groups. It first inverts the blend within the original base/native limits, then
tries bounded source-base backtracking if the shortest SO(3) arc or those limits
prevent inversion. Valid starts are unchanged. Shared grip controls, strict-hold
poses, frame zero and other frames remain unchanged during this initialization
correction; the full decoder must pass the original bounds before optimization.
An infeasible strict grip still fails. `shared_hold_initialization.json` records
the attempted correction, clipping and fallback fractions, including failures;
it is a control-encoding correction, not a new grasp or collision repair.

The two-case comparison reduced object-relative wrist and finger jerk against
the preceding unified profile, but retained deficient digit coverage and sampled
penetrations. Lower average contact distance can hide a missing digit. Inspect
per-digit coverage, speed, motion amplitude, temporal joins and original-mesh
diagnostics; a user-selected initialization candidate remains unaccepted motion.

## Bound inputs and execution

`case.json` contains only these fields:

```json
{
  "inputs": {
    "native_npz": "native.npz",
    "model_xml": "original.xml",
    "object_mesh_npz": "object.npz",
    "hands": {"right": "REPLACE_WITH_ORIGINAL_GRASPINPUTS_HAND_DESCRIPTOR"}
  },
  "source_arrays": "source_human_arrays.npz",
  "contact_mask": {"right": [false, true, true, true]},
  "contact_patch_members": {"right": "REPLACE_WITH_ORIGINAL_ANATOMICAL_MEMBER_MAPPING"}
}
```

This is a field illustration, not a runnable asset config. Optional
`contact_region_faces` retains original triangle IDs. Optional `is_static_obj`
must agree with the full original trajectory. All solver knobs are rejected
from the case schema. Source frame count/FPS come from the native input; existing
nonuniform timing is rejected rather than resampled. Per-case numeric IDs such
as original mesh/face/joint indices are anatomy and geometry, not loss settings.

`phases.json` binds the source video and covers observed intervals; uncovered
frames remain `unknown`:

```json
{
  "schema_version": 1,
  "source_video": {"path": "source.mp4", "sha256": "REPLACE_WITH_64_HEX_SHA256"},
  "frame_count": 121,
  "fps": 24.0,
  "hands": {
    "right": [{
      "start": 30, "end_exclusive": 90, "role": "hold", "grasp_id": "handle_grip",
      "evidence": {
        "frames": [30, 60, 89], "certainty": "observed",
        "note": "The source shows the same grip on the handle throughout this interval."
      }
    }]
  }
}
```

Use half-open original frame ranges. An uncertain phase must not claim a shared
grasp. Notes are agent-authored observations, not a contact probability or an
automatic video verification. Bind every configured hand, using an empty list
where no phase is known. Original video hash is checked before execution.

```bash
python -m agentic_retargeter.hand_initialization_policy \
  --case /case/case.json --phases /case/phases.json \
  --out /case/hand_shared_v001 --device cuda:0
```

Add `--plan-only` to validate and write the exact profile/configs without GPU
optimization. Always use a new output directory. Device is a deployment option;
numerical profile changes belong in a new globally versioned recipe and an
across-case comparison. Shared holds use eager CUDA; per-frame-only cases can
use CUDA Graph. There is no unsupported graph fallback.

Outputs preserve `plan.json` with profile and input hashes, the raw source
snapshot, full Topo bank, `seed_receipt.json` with ranking and failed evidence,
`warm.npz`, executed paired config, checkpoints and paired report. Exact native
body/object/timing bindings and the raw first hand frame are retained. A shared
pose can still be infeasible within the original source-relative bounds; that
failure stays a failure. No per-case relaxation or automatic alternative branch
is attempted. Check contact/source deviation, collisions and temporal joins
before selecting a checkpoint for SOMA.

The final selection names the refined configuration, solver report and NPZ by
hash. Preserve the base paired result and SG output as intermediate evidence;
do not accidentally export them as the new default target.

This entrypoint handles hand initialization only. Full-body forward SOMA and
synchronous SOMA remain later stages. Collision-key refinement is a separate
optional scope; initialization-only runs stop after SOMA export and review. For
many cases, run one case per GPU process with your own scheduler.
The generic typed GPU worker does not automatically expose this new command;
use a compatible provisioned operation/runtime rather than bypassing its sandbox.
