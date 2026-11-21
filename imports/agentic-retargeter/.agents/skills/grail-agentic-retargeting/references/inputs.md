# Native inputs and effective retarget constraints

## Reconstruction handoff or existing robot motion

Initial conversion routes SMPL-X to external GRAIL/GMR and SOMA-base BVH to NVIDIA soma-retargeter through the [unified retarget tools](../../../../tools/retarget/README.md). Import native robot motion with the selected reconstruction records described in the [upstream handoff contract](../../../../docs/upstream-handoff.md). Preserve their limitations. Transfer native human lineage, original object transforms, source timing, per-hand roles/contact phases and reviewed material descriptors. Missing or hidden contacts remain uncertain; do not silently turn reconstruction estimates into exact robot constraints. SOMA's bundled body-only G1 is not the fingered G1 model; bind actual robot topology and preserve missing-input labels.

Carry source event times, authored timing changes, protected support/contact milestones and the actual exported interpolation convention into robot planning. Human timing/phase evidence guides robot keys; it does not prove a G1 closure or auxiliary-digit path is feasible. Where older records omit these details, inspect the selected native/source motion and document the uncertainty instead of inventing observations or forcing an upstream refit. Bind new robot descriptors and interpolation controls separately from the human-stage record.

For a downstream robot correction, use the selected native robot motion and its original upstream references. Human Stage 1/2 fitting and depth calibration remain unchanged. A partial upstream handoff may support a declared robot diagnostic or local edit; a better robot grasp does not retroactively validate reconstruction.

Record the parent and output hashes, actual model/mesh versions, joint ordering, units, coordinate conventions, camera, object scale/transforms, fixed targets, editable variables and decision. Retain the untouched fallback and pre-edit outputs.

## Check what the runtime actually does

### Custom SMPL-X handoff

GRAIL human reconstructions may use `G1ProportionSMPLX` / `G1SmplxHumanModel`. Follow the [custom-model handoff](../../../../docs/upstream-handoff.md#custom-human-model-lineage). Resolve the handoff's actual base/custom asset hashes, initialized model-array export, native scale convention and FK parity evidence before computing human targets. Reuse those bound proportions for human reference geometry and target construction; do not regenerate a different human from a neutral template merely because the native pose vectors have standard SMPL-X dimensions. Check effective upstream retarget CLI behavior and overrides against this lineage.

For the GRAIL custom contract, reference scale has already been baked into template/pose correctives; native human scale is 1. Verify that convention from the bound handoff. Do not apply the same proportion/height conversion twice, and do not rerun source-to-native pelvis conversion on native data. Robot-side reach adaptations remain explicit retargeting controls and must not silently alter the upstream human reconstruction. A standard-SMPL-X checkpoint retains its recorded model lineage; an existing robot-only correction does not force a custom-model conversion.

### Effective runtime constraints

For initial conversion details, inspect the upstream `grail-retarget-batch` skill and implementation in the GRAIL checkout selected for that handoff; [upstream entry points](../../../../docs/upstream-handoff.md#upstream-entry-points) lists their locations. For native refinement here, inspect [robot.py](../../../../src/robot.py) and `python -m agentic_retargeter fit-robot --help`. Check that upstream manipulation processing preserved both arms: the legacy hand-action path can leave the left hand open and zero the left arm. Inspect the effective upstream `--treat_hands_equally` behavior rather than assuming a two-handed input survived processing.

Before tuning a mismatched pose, inspect:

- Native FK, handedness, link axes, joint indices/limits and pad locations/normals from the actual robot model. Link names alone do not establish anatomical correspondence.
- The solver's effective wrist and shoulder targets, including offsets from reconstructed finger centers, scale overrides, task frames and residual definitions. Raw human-wrist error may measure a different point from the actual retarget target.
- Active variables, frozen coordinates, task weights, phase schedules, joint/velocity limits and actual call arguments. A nominal configuration may be overridden or passed to an unintended parameter.
- Whether intended material contact constraints are active and routed to the correct hand. Check their gradient/variable path when introducing a new term; source-pose or orientation priors can oppose a contact update.

Preserve original joint limits. A failed bounded IK search is not proof that the grasp is globally unreachable. Diagnose initialization, correspondence, arm/torso coordination and authorized source-supported controls before attributing the failure to the robot.

## Reuse fixed source and depth evidence

Bind evaluation to the existing source frames, masks/rays, target arrays, calibration, model/checkpoint and preprocessing hashes. For MoGe3 with background ground truth, reuse that frozen background calibration. Robot candidates do not authorize a new foreground calibration or candidate-specific alignment.

Report per-region camera-z error and signed bias, common-pixel error, total/covered/missing rays, and actual hit link identities where relevant. A shoulder occupying a hand-region ray is not evidence of hand alignment. Use the same evaluated pixels across candidates and retain missing-coverage penalties/denominators. Read [robot review](review-delivery.md) for contact, geometry and temporal evidence alongside depth.
