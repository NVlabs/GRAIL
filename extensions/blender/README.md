# GRAIL Blender refinement extensions

Two reusable, offline Blender panels for the native editing part of the GRAIL workflow. Version 0.1.0 targets Blender 4.2–4.3 and is tested with **Blender 4.3.2**. Blender 4.4+ uses a different animation API and is not supported by this version.

| Extension | Sidebar | Native editing |
| --- | --- | --- |
| GRAIL HOI Recon Refinement | 3D View → N → **GRAIL HOI** | Bound human-model preview, selected human contact keys, phase-window interpolation, bounded constant camera-depth placement trials |
| GRAIL G1 Retargeting Refinement | 3D View or Graph Editor → N → **GRAIL G1** | Actual robot joint channels, approach/closure timing keys, Blender interpolation, original-limit checks and native export |

Both export new checkpoints and keep review decisions separate. An export is **UNREVIEWED**. The panels are refinement tools: initial Stage 1/2 fitting, initial SOMA retargeting, optional contact solvers, and geometric/source/depth review remain in the existing Python workflow. They do not automatically locate contact, solve a grasp, or certify collisions.

## Install

1. In Blender 4.3.2, open **Edit → Preferences → Get Extensions**.
2. Use the top-right menu → **Install from Disk**, select one built ZIP, and enable it. Repeat for the second ZIP. Both can coexist.
3. Open the relevant sidebar with **N**. No pip installation is needed inside Blender. Original models and case data are supplied separately.

Save your current document before creating an authoring session. The human setup requires an empty scene; it refuses to remove existing objects. The robot extension preserves existing scene objects. A **Create / Save** or **Save / Export** action saves a new `.blend` and changes the current document's save path. It never overwrites an existing checkpoint directory.

## Human/object workflow

Use **Stage 2 SOMA Native Round** to author bounded rotation keys and refresh the preview against the external native model. Source frame 0 corresponds to Blender frame 1. For each round, bind the model, selected reconstruction, original object and case configuration. See the [CLI and refresh guide](../../docs/hoi_refinement.md) for the frozen controls, repair steps and review criteria.

## Robot workflow

1. Select the native robot NPZ and its compiled model snapshot. Initial conversion from human motion remains in the official SOMA retargeter.
2. For geometry preview, open the matching portable G1 `.blend`. An empty or unrelated scene supports **curve controls only**; this version does not import arbitrary robot rigs.
3. Load actual model joints. Explicitly enable the permitted limited hinges; no joints are selected by default. Choose a source-frame window, closure onset/completion, native values, and optional protected milestones. Alternatively load an existing curve plan.
4. Create controls and edit the selected channels in the Graph Editor. Source frame **0 = Blender frame 1**. Default interpolation is **Bézier / Auto Clamped**, with handle smoothing **None**. Continuous Acceleration is optional and must be evaluated on the exported motion.
5. Exact mesh preview requires the matching portable scene and the explicit **Enable Exact Preview** action, which executes that trusted scene's preview code. It does not run automatically at installation or file open.
6. Save/export to a new directory. Original joint limits, declared window endpoints, protected milestones, frozen coordinates and dense fractional curve samples are checked. The result includes the native candidate, curves, comparison report and editable `.blend`.
7. Review the intended thumb/long-finger roles, auxiliary-digit clearance, matching-time material contact, whole original meshes, source/depth alignment and temporal behavior. Return to contact/pose/path editing after a failure. A smoother F-curve or a solver iteration cap is not a completed contact correction.

Saved robot export/preview Text buttons require the matching installed extension and use the same guarded operators. Reopening and exporting curves does not require enabling mesh preview. Scene or bundled-code mismatches are rejected instead of silently switching implementations.

The reusable external contact tools remain `contact_edit_kit/pose.py`. Their material descriptors and reviewed pose results can feed the authoring plans. They require their documented Python environments; these extensions do not bundle SciPy, MuJoCo, licensed model assets or case motions.

## Rebuild and verify

From the repository root, use Python 3.11 or newer, or install the GRAIL `dev` extra on Python 3.10.

```bash
python extensions/blender/build.py --out /absolute/new/build-directory
```

The builder copies the current repository-owned kits into two self-contained ZIPs, records original/packaged SHA256 hashes and any import adapter, and preserves the repository LICENSE. Do not maintain separate hand-edited kit copies inside the extension source directories. Runtime scene scripts and model assets are not downloaded.

`tests/install_smoke.py` exercises actual ZIP installation in a fresh local Blender extension repository, checks coexistence/import isolation, and checks unregistration cleanup. Run it only with an isolated Blender user profile and new output directory; the file documents the command. Native edit/export/reopen tests should also cover the kit fixtures and the project's actual custom human/portable robot scenes.

A successful tooling test establishes installation and native round-trip behavior. It does not upgrade the quality status of an existing reconstruction or retargeted motion.
