# Agentic Retargeter

> **Vendored in GRAIL.** This directory is a snapshot of the Agentic Retargeter at
> commit `41264543`, distributed with GRAIL under the same NVIDIA License (see
> [LICENSE](LICENSE)). Its tests, cluster launchers and the disabled hosted-inference
> client are omitted from this snapshot. GRAIL calls these tools through
> `python -m grail.retargeting.agentic`; see the
> [GRAIL adapter guide](../../docs/agentic_retargeting.md).

Reusable robot contact refinement and native animation tools extracted from GRAIL. This repository owns the **robot refinement** workflow, native backend, contact editing kit and G1 Blender extension. Start from an existing native robot checkpoint or use the initial retargeting adapters: **SMPL-X → GMR; SOMA → NVIDIA soma-retargeter**.

The current [retargeting workflow](docs/current-retargeting-workflow.md) is **source phases and regions → Topo-inspired hand keys → base paired hand sequence → SG initial guess and contact-aware temporal refinement → GPU forward whole-body SOMA → synchronous trajectory optimization → native export and review**. The [shared hand initializer](docs/hand-shared-policy.md) uses one numerical profile across cases and supplies free-wrist/finger targets. Whole-body SOMA fits those targets through the robot's arm kinematics; reachability and motion quality still require review. Post-SOMA collision-key refinement is a separate requested scope. Material anchors are optional additional targets with separate provenance. Fully static original objects omit robot-object contact losses while retaining collision checks. A moving object's source-declared precontact prefix shares one adjustable stationary pose; contact quality does not unlock it.

Preserve the original support scene unless the case explicitly authorizes a replacement or placement change. Bind an authorized floating block consistently in optimization, audit and rendering, retaining the original as provenance. When the first frame is protected, exclude it from edits and verify the exported state; a soft target alone does not establish preservation. Current scene and protection settings are documented in the [workflow guide](docs/current-retargeting-workflow.md).

For existing-grasp edits and fixed-material experiments, the [grasp tools](tools/grasp/README.md) support bank generation, mounted difficult-key screening, shared-hold fitting and local approach/closure repair. Reuse unaffected reviewed stages. Sliding and uncertain support require separate phase relations, and video observations and proposed 3D contacts remain hypotheses until reviewed.

For many cases, run one case per GPU process with your own scheduler; keep inputs read-only and give each case its own output directory and ledger. The agent runs common tools for its case and records decisions and evidence. The batch runner's 30-minute cumulative case execution ledger with a delivery reserve bounds tool execution; it does not define the user's overall task duration or accept an unresolved motion when a process ends.

| Component | Location |
| --- | --- |
| Current hand/SOMA/refinement pipeline and scene policy | [Workflow guide](docs/current-retargeting-workflow.md) |
| Packaged Topo-inspired hand initialization | [Hand initializer](docs/topo-hand-initialization.md) |
| Explicitly authorized floating support replacement | [Support-block recipe](docs/support-block.md) |
| Optional Newton contact queries and Blender overlays | [collision guide](docs/collision.md) |
| Runtime implementation, directly under `src/` | [src](src/) |
| Local video perception, grasp hypotheses and source keyframes | [video tools](tools/video/README.md) |
| External GMR conversion and input/model preflight | [GMR tools](tools/gmr/README.md) |
| Unified SMPL-X/SOMA routing and native handoff | [Retarget tools](tools/retarget/README.md) |
| External NVIDIA SOMA BVH conversion | [SOMA tools](tools/soma_retargeter/README.md) |
| On-demand local GraspGenX initial-grasp proposals | [GraspGenX tools](tools/graspgen/README.md) |
| Standalone contact pose fitting and Blender curve editing | [contact edit kit](scripts/standalone/contact_edit_kit/README.md) |
| Shared original-mesh audit and coupled short-window repair | [local repair tools](tools/repair/README.md) |
| Optional contact-constrained G1 sequence prototype | [experimental tools](scripts/standalone/contact_edit_kit/experimental/README.md) |
| Installable G1 native editing panel | [Blender extension](extensions/blender/README.md) |
| Upstream reconstruction / initial conversion contract | [handoff guide](docs/upstream-handoff.md) |
| Local Blender editing rounds and persistent scene checkpoints | [Blender editing loop](docs/blender-agent.md) |
| Canonical hands, measured/predicted contact maps and agent observations | [Hand inspection tools](docs/hand-inspection.md) |

## Install

Python 3.11+:

```bash
cd imports/agentic-retargeter   # inside a GRAIL checkout
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
agentic-retargeter --help
```

Add `.[experimental]` only for the optional constrained sequence solver. `ffmpeg` and `ffprobe` are required for media checks/pages. Blender runs in its own bundled Python environment; use **Blender 4.3.2** for the tested extension/API. No GRAIL import or checkout is needed for the tools included here. Original robot XML/assets, object meshes, case motions and licensed human model assets are not bundled.

Runtime code lives directly in `src/`; there is no `src/agentic_retargeter/` directory. Setuptools maps `src/` to the existing Python import name `agentic_retargeter`, preserving installed commands. Thin checkout launchers live in `tools/`; the installed wheel also contains the contact-edit backend. A Blender binary and original model assets remain external.

```text
src/
  video_perception.py   # Local video → object masks + tracked hand landmarks
  video_grasps.py       # Image evidence → grasp hypotheses + source keyframes
  video_cli.py          # Video command dispatch
  gmr.py               # External GRAIL/GMR adapter and provenance checks
  soma.py              # External NVIDIA SOMA BVH retargeter adapter
  retarget.py          # Explicit source-model routing
  retarget_native.py   # Strict robot-model/timing/native-state handoff
  graspgen.py          # Explicit optional grasp proposals; no automatic motion edits
  graspgen_inputs.py, graspgen_runtime.py, graspgen_worker.py, graspgen_preview.py
  blender_edit.py      # One local inspect/edit/review round
  blender_runtime.py   # Provider-free worker transport and bindings
  blender_worker.py    # Native Blender edits, renders, export and reopen checks
  inspection.py        # Canonical-hand and measured/predicted map calculations
  blender_inspection.py
  contact_edit/        # Native pose/curve tools and experimental sequence solver
  motion_audit.py      # Explicit contact, mesh, source and temporal audit scopes
  local_repair.py      # Coupled local repair with protected poses and fixed anchors
  repair_geometry.py   # Original-triangle intersections and nearest queries
  robot.py, native.py, contact.py, manifest.py, media.py
tools/
  video/               # Thin video launcher and local model setup
  gmr/                 # Thin conversion launcher; upstream GMR stays external
  soma_retargeter/     # Thin SOMA conversion launcher
  retarget/            # Unified SMPL-X/SOMA entry point
  graspgen/            # On-demand local model launcher and proposal preview
  contact_edit/        # Thin pose/curve launchers
  repair/              # Shared motion audit and local repair launcher
extensions/blender/    # Installable Blender panels and package builder
```

```bash
python -m agentic_retargeter retarget preflight \
  --source-model soma --config soma_case.json --output runs/soma_001
python -m agentic_retargeter retarget run \
  --source-model smplx --config smplx_case.json --output runs/smplx_001
```

See the [dual-backend guide](tools/retarget/README.md) for configuration and native
handoff. SOMA currently accepts the official SOMA-base BVH format. Its bundled G1
target has 29 body DOFs. An explicit custom `unitree_g1_dex3` registry solves all
43 body/finger joints with the same SOMA backend; see the
[target generator](tools/soma_retargeter/README.md#g1--dex3-through-the-soma-solver).
Original object motion and contact refinement remain separately bound inputs.

## Select video evidence before editing

The optional local frontend uses Grounding DINO, SAM2 and MediaPipe. Supply the
source video, object description and downloaded model paths; no manual contact
frames or remote model calls are required. It saves hashed observations, masks,
grasp-event hypotheses, source-frame key selections and contact-candidate sheets.
See [setup and commands](tools/video/README.md). Detection gaps remain unknown;
the first detected candidate is not necessarily the first physical touch.

Review those source frames before authoring native contacts and copy the chosen
source indices into the Blender case's `spec.observation_frames`. Blender's native
motion, camera and source-video lineage must match that case. Image silhouette
anchors are **2D hypotheses**: they cannot be supplied directly as object-local
3D surface anchors or loaded as a 3D contact-map prediction. The canonical-hand
dashboard continues to show native measured proximity and separately bound 3D
predictions/proposals. See [architecture and contracts](docs/architecture.md).

## Agentic editing without remote model calls

The host assistant reviews the scene, makes bounded native edits, and inspects the resulting views and measurements. Use an available Blender MCP connection for interactive work, or run Blender in the foreground with `blender -b --python`. This repository provides the latter through `blender-edit`; it does not bundle an MCP client. No remote model API is called, and no model credentials are needed.

Keep a versioned `.blend` as the editable state across rounds. Save each new checkpoint beside its exported native candidate and review evidence, then start the next round from that checkpoint. The hand/contact dashboard remains part of Blender and the rendered evidence.

```bash
# Inspect the input without changing its keys.
python -m agentic_retargeter blender-edit \
  --config case.json --output runs/inspect_001

# Apply explicit operations, then continue from the resulting editable scene.
python -m agentic_retargeter blender-edit \
  --config case.json --operations edit_001.json --output runs/edit_001
```

See the [local editing guide](docs/blender-agent.md) for scene bindings, operation files, checkpoint continuation and optional MCP use.

## Try a synthetic contact edit

Use new output paths for every run:

```bash
python scripts/standalone/contact_edit_kit/examples/make_demo.py --out runs/demo
python scripts/standalone/contact_edit_kit/pose.py \
  --config runs/demo/pose.json --out runs/demo/pose_candidate

/path/to/blender-4.3.2 --background --factory-startup --python-exit-code 1 \
  --python scripts/standalone/contact_edit_kit/blender_curves.py -- \
  --native runs/demo/native.npz --model runs/demo/model.npz \
  --plan runs/demo/curves.json --out runs/demo/curve_candidate
```

The toy robot tests native parameter transport and contact fitting. It is not an accepted grasp or a G1 geometry preview. For a real case, provide the original robot/model binding, fixed object material patches/normals, source-supported phases and permitted joints. Start from the reviewed grasp before propagating it.

## Native backend and review records

```bash
python -m agentic_retargeter fit-robot --config case/robot_trial.json --output runs/trial_01
python -m agentic_retargeter init --spec case/benchmark.json --output runs/manifest.json
python -m agentic_retargeter verify runs/manifest.json
python -m agentic_retargeter page runs/manifest.json --output runs/review_page
```

Use `--help` for `register` and `select`, and see the [input and review contract](docs/upstream-handoff.md). The legacy bounded robot backend produces `candidate_UNREVIEWED.npz`; the newer kit has its own documented config/schema. These are distinct solvers, not interchangeable config formats. Prior acceptance labels and derived geometry do not establish acceptance after an edit.

Preserve the original camera, models/scales, timing, joint limits and source/depth evidence. Compare fixed material gaps, object-relative slip, entire original meshes, source alignment, missing depth coverage and affected temporal intervals together; contact-normal angles are optional diagnostics. Original open surfaces do not define a solid interior. Numerical convergence or smooth curves alone do not validate contact, collision clearance or physical stability.

## Build the Blender extension

```bash
python extensions/blender/build.py --out dist/blender-v001
```

Install the generated `grail_g1_retargeting-0.4.0.zip` using Blender's **Preferences → Get Extensions → Install from Disk**. Open **N → GRAIL G1** in the 3D View or Graph Editor. Source frame 0 maps to Blender frame 1. Native export, protected milestones, original limits and optional exact preview use the same checks from the panel and saved Text buttons. The hand inspection panel adds wrist-local hand views and separate measured, predicted/proposed and difference maps; see the [configuration and usage guide](docs/hand-inspection.md).

## License and scope

The Blender guide documents the isolated ZIP installation and native edit/export/reopen check. The code is distributed with the original NVIDIA [LICENSE](LICENSE). HOI reconstruction and training-data processing remain upstream. Initial conversion adapters call the configured external GMR or SOMA runtime; their heavy dependencies and original assets are not copied into this package.
