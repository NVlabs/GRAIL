# GRAIL to Agentic Retargeter handoff

GRAIL owns human/object reconstruction, depth estimation and calibration, and custom human models. Agentic Retargeter's initial-conversion adapters route SMPL-X to external GRAIL/GMR and SOMA-base BVH to external NVIDIA soma-retargeter. Subsequent robot grasp, contact, native trajectory and Blender editing share the native robot handoff. A reviewed human reconstruction is useful input evidence; it is not a reviewed robot grasp. See the [dual-backend contract](../tools/retarget/README.md).

In GRAIL this snapshot is vendored under `imports/agentic-retargeter`; GRAIL's adapter `python -m grail.retargeting.agentic` invokes it.

## Upstream entry points

Use the guides from the GRAIL revision recorded in the case handoff.

| Upstream GRAIL path | Responsibility |
| --- | --- |
| [Staged HOI fitting](../../../docs/source/agentic_kinematic.md) | Source observations, native fitting and reviewed placement. |
| [Native refinement](../../../docs/hoi_refinement.md) | Human and object editing and source checkpoints. |
| [Robot retargeting](../../../docs/source/retargeting.md) | G1 conversion and native motion handoff. |

Run upstream stages only when the requested work requires them. Existing robot corrections should retain their selected upstream references. Missing historical evidence is an explicit limitation; it does not itself require a reconstruction rerun. This repository does not install GRAIL, GMR, SMPL-X assets, or the reconstruction stack, and does not provide `fit-human` or `fit-hoi` commands.

For two-handed manipulation, inspect upstream hand-action processing: the legacy route may zero the left arm and leave its hand open. Check the effective `--treat_hands_equally` behavior and the actual exported motion, not just the requested flag. Record effective CLI arguments and model/proportion overrides with the converted native checkpoint.

## Records to transfer

Use a manifest or equivalent explicit records with content hashes and stable relative paths where practical. Preserve existing native containers; do not convert formats merely to rename fields. Original models, meshes, source media and case motions are caller inputs and are not bundled with this repository.

| Record | Required meaning |
| --- | --- |
| Selection and lineage | Case identity; selected human/object and robot checkpoint paths and SHA256 hashes; parent chain; untouched fallback; explicit selection decision and inherited limitations. Do not select by timestamp. |
| Native robot state | Original MuJoCo model and referenced assets; model hashes; native joint names/order/addresses and limits; finite native motion; units, coordinate conventions and interpolation used by the downstream consumer. Bind the robot to its actual upstream human checkpoint. |
| Human lineage | Actual human model class/version and asset hashes, native parameters and any target arrays used by the robot conversion. Retain model proportions and scale conventions. |
| Original object | Original mesh/component identity and hash, face/vertex indexing, scale and per-frame transforms; distinguish scaled object-local coordinates from raw mesh coordinates. |
| Static scene | Original table/support/environment triangles, source hashes and explicit world alignment. These remain independent of body model, robot scale and moving-object transforms. Absent evidence is UNKNOWN; never infer a replacement table. |
| Camera and source | Fixed intrinsics/extrinsics, source images/video and hashes, frame correspondence, image preprocessing, masks/observations and occlusion uncertainty. Never align the camera separately for a new candidate. |
| Timing and phases | Original fps/count, stationary-object intervals, contact/support onset and release per hand, protected keys and checked joins/subframes. Distinguish observed times from authored timing choices and uncertain event brackets. |
| Contacts | Human-side roles and observed/estimated/authored provenance; original-material face/barycentric anchors or bounded patches; normal and gap conventions; active phase and permitted slip. Define robot digit mapping and pads downstream from the actual robot. |
| Frozen source/depth evidence | Target arrays/rays, backend/checkpoint/preprocessing hashes, existing calibration, fixed denominators and coverage/hit identities. Mark unavailable evidence explicitly. No candidate-foreground recalibration. |
| Scope and review | Editable/frozen variables, source-supported bounds, executed code/config hashes, numerical and visual review, actual inspected frames/subframes, omitted checks, inherited/new failures and accepted-for-scope/partial/rejected/unreviewed decision. |

Uncertain or hidden contacts remain hypotheses. A partial human handoff may support a scoped robot diagnostic or correction; robot improvements do not retroactively validate it. Source phase times guide robot key authoring but do not prove native reach, closure feasibility or auxiliary-digit clearance.

### Packed static scene

The object sidecar and native NPZ may carry a scalar JSON string
`static_scene_json` with `schema: "grail.static_scene.v1"`, `world_frame` equal
to the native scalar `world_frame`, `status` (`PRESENT`, `EMPTY`, or `UNKNOWN`),
`meshes`, and a `provenance` object. Each mesh has a unique `name`, a `role`
(for example `table`), `source: {path, sha256}`, and `vertices_key` /
`faces_key` naming `static_scene_000_vertices` / `static_scene_000_faces`
(consecutive zero-based indices). Vertices are finite floating `N x 3`
coordinates already baked into the native world frame in metres; faces are
integer `M x 3` triangles with valid vertex indices. The consumer preserves
these arrays exactly and never applies body scaling or object animation.
`PRESENT` requires meshes; `EMPTY` and `UNKNOWN` require none. Missing metadata
remains compatible and means `UNKNOWN`. Source paths record provenance;
rendering uses the packed geometry without requiring those paths to be mounted.

The native Blender preview imports this geometry into `GRAIL_StaticScene` with
identity transforms, visible meshes and no animation or constraints. Reopening
checks a content fingerprint and the actual triangles; name collisions, stale
geometry, hidden objects or changed transforms fail instead of being silently
replaced. Render candidates must retain identical static-scene fields. Blender
stores coordinates at its normal float32 precision, while native NPZ geometry
retains its input precision. Previewing the table alone does not establish
clearance or stability.

Native local repair automatically loads a packed `PRESENT` scene and screens
every original robot mesh against every carried scene mesh, including body
parts outside the editable joint scope. Those inherited collisions remain
reported; a hand-only edit cannot silently exempt a colliding wrist or leg.
Optional repair config `static_scene: {clearance_m, reference_frame}` sets the
clearance proposal and safe-side reference. The scene geometry stays frozen in
candidate audits. Primitive robot geoms, object/table support contact, temporal
gaps and incomplete mesh queries retain their separately reported limitations.

For initial SOMA IK, enable the static scene in the contact collision objective
using the explicit coordinate binding in the [SOMA adapter guide](../tools/soma_retargeter/README.md).
Its sampled soft penalty is a proposal mechanism; follow it with original-mesh
screening and source/novel-view review. Neither a low loss nor the presence of
the scene in a render means collisions were cleared.

## Custom human model lineage

Some GRAIL handoffs use the G1-proportioned `G1ProportionSMPLX` / `G1SmplxHumanModel`. Bind the base SMPL-X asset, custom T-pose/parameter assets, initialized model-array export and FK parity evidence. Under that custom contract, reference scale is baked into the template and pose correctives, with native human scale 1. Verify the recorded convention before building targets; do not apply the same height/proportion or source-to-native pelvis conversion again.

Standard-SMPL-X and legacy inputs retain their recorded model lineage. A robot-only correction does not authorize replacing the upstream human geometry. The robot always uses its original native robot model and limits; any reach adaptation is an explicit robot-stage edit.

## Begin native refinement

From a checkout with the package installed, inspect the available contracts before binding a case:

```bash
python -m agentic_retargeter --help
python -m agentic_retargeter fit-robot --help
```

`fit-robot` creates bounded native candidates from caller-authored configuration and correspondences. It does not infer a grasp or perform initial GMR conversion. The package's `init`, `verify`, `register`, `select` and `page` commands manage provenance, separate review decisions and comparison media. Human records in a manifest represent imported lineage; their presence does not add a human fitting runtime.

Record the selected baseline, editable variables, contact definitions, mounted palm pose and auxiliary paths. The [contact edit kit](../tools/contact_edit/README.md) and [Blender extension](../extensions/blender/README.md) provide additional authoring tools from the clone. Every new export is **UNREVIEWED**. Contact, whole original meshes, fixed source/depth evidence and temporal/visual review establish the requested outcome together; solver termination or a successful Blender round trip alone does not.
