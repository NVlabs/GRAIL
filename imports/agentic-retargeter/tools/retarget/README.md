# Initial retargeting: SMPL-X and SOMA

For the current hand-first pipeline, see the
[workflow guide](../../docs/current-retargeting-workflow.md). Two preparation
tools are separate from the whole-body conversion:

- `hand_targets.py`: export an explicitly selected hand-sequence checkpoint to
  validated SOMA soft targets; [input/selection contract](../../docs/hand-initialization-workflow.md#hand-target-handoff).
- `support_block.py`: replace an explicitly authorized support mesh while
  preserving robot/object trajectories; [geometry and provenance](../../docs/support-block.md).

Neither preparation tool runs SOMA or accepts a motion. Their output paths must
be new; the downstream solver must bind the prepared targets and scene.

One explicit routing command selects the human source model:

| Source model | Backend | Current input |
| --- | --- | --- |
| `smplx` | External GRAIL/GMR | Reviewed GRAIL `hoi_data/hoi_data.pkl` and bound SMPL-X assets |
| `soma` | [NVIDIA soma-retargeter](https://github.com/NVIDIA/soma-retargeter) | SOMA-base skeleton `.bvh`, with explicit units and facing convention |

```bash
python -m agentic_retargeter retarget plan \
  --source-model smplx --config smplx_case.json --output runs/smplx_001
python -m agentic_retargeter retarget preflight \
  --source-model soma --config soma_case.json --output runs/soma_001
python -m agentic_retargeter retarget run \
  --source-model soma --config soma_case.json --output runs/soma_001
```

From an uninstalled checkout, replace `python -m agentic_retargeter retarget` with
`python tools/retarget/run.py`. `plan` is read-only; `preflight` checks the chosen
external runtime; `run` performs foreground conversion to a new output directory.
A failed preflight exits nonzero. The router never guesses from a file extension
or falls back to the other human model.

Configuration is backend-specific: see [GMR](../gmr/README.md) and
[SOMA](../soma_retargeter/README.md). The selected adapter's report is `run.json`;
the unified command also writes `retarget_result.json`, preserving the source
model, backend, results and UNREVIEWED status. A completed conversion does not
select a motion or certify grasp/contact quality.

The implementations are directly in `src/retarget.py`, `src/gmr.py`, `src/soma.py`
and `src/retarget_native.py`. The heavy runtimes and original model assets remain
external. The old `tools/gmr/run.py` entry point remains supported.

## Model and timing contracts

SMPL-X keeps the existing custom G1-proportioned model convention for new GRAIL
cases. A reviewed standard SMPL-X source remains an explicit supported choice.
Neither convention applies to SOMA. Preserve each SOMA source's skeleton,
rest-pose, identity/scale and coordinate lineage when exporting BVH.

The official SOMA tool accepts SOMA-base BVH, not arbitrary BVH or raw SOMA
parameter arrays. GEM's legacy MHR-backed SOMA and native SOMA03 are different
parameter/model contracts. Export using the matching model and verify source FK;
renaming an NPZ/PKL or reusing the SMPL-X loader is not a conversion. Raw SOMA
checkpoint-to-BVH model export is not bundled in this adapter.

Official SOMA CSV uses centimeters and degrees; it does not carry source FPS.
The adapter preserves source BVH timing and records the actual robot joint order.
Native handoff must convert units and root orientation explicitly and use the
original, bound robot model. Do not apply a second axis/scale conversion to
already converted native arrays.

## Shared Blender refinement

Both routes lead to native robot motion, then the same contact-authoring,
canonical-hand inspection and versioned `.blend` editing workflow. Native state
and the visual robot model must agree on topology and joint order. Use
`retarget-native` to validate/export a compatible native handoff; its
output records remaining scene requirements rather than generating an accepted
grasp or automatically creating a Blender scene.

```bash
python -m agentic_retargeter retarget-native \
  --report runs/soma_001/run.json \
  --output runs/soma_native_001/robot_UNREVIEWED.npz
```

The checkout equivalent is `python tools/retarget/native.py ...`. It checks the
completed run's CSV, original BVH, robot XML and dependency hashes before native
conversion. It writes a `.npz` and matching `.npz.json` report. Optional
`--object-sidecar` supplies an explicitly bound original object trajectory; its
schema is documented in [the native converter](../../src/retarget_native.py).

The official bundled SOMA G1 is a **29-DOF body model without robot finger
joints**. Its foot-contact stabilization does not reconstruct hand/object grasps.
Do not fill missing finger joints with invented values or import its motion as a
different G1-with-hands model. To solve the original G1 + Dex3 model directly,
register the custom `unitree_g1_dex3` target with `extra_robot_paths` and its
verified native SOMA rest calibration. See the
[target generator](../soma_retargeter/README.md#g1--dex3-through-the-soma-solver).
This retains SOMA/Newton and solves all 43 body/finger joints. Subsequent grasp
and approach refinement is still required.

For HOI, preserve the original object motion, mesh/scale, camera and source-frame
mapping as separately bound evidence. A body-only CSV has no object trajectory.
Robot-only native output remains explicitly incomplete for object contact edits
until those scene inputs are supplied. Once bound, use the existing
[Blender editing loop](../../docs/blender-agent.md); no remote model API is required.
