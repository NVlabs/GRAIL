# Robot retargeting

For a hands-on conversion, follow [body-only stairs](tutorials/stairs.md) or [kettlebell with DEx3](tutorials/kettlebell.md). Both start from a reviewed fit. Keep the source frame rate and original object binding through export. Review the robot preview before using its [native NPZ as a tracking input](outputs.md).

Start from a reviewed SOMA03 human and object reconstruction. Set `source_model: soma03` in the case YAML. Bind the source motion, fixed identity, object mesh, support scene, units and timing. GRAIL exports the native 77-joint pose through verified FK to SOMA-base BVH and runs the NVIDIA SOMA retargeter. Select the body-only G1 target or the hand-capable Dex3 target before conversion.

```bash
python -m grail.retargeting.agentic plan --config case.yaml --output /runs/case/plan
python -m grail.retargeting.agentic preflight --config case.yaml --output /runs/case/preflight
python -m grail.retargeting.agentic run --config case.yaml --output /runs/case/initial
```

Use a fresh output path per invocation. The native result is the starting point for grasp and trajectory refinement. Review body posture, contact, original table geometry, support and temporal continuity before selecting the motion for downstream training.

See {blob}`the retargeting guide <docs/agentic_retargeting.md>` for the case YAML, installation and review steps.

## Hand off a retargeted motion

For every motion, collect the native G1 + DEx3 robot/object motion written by
`python -m grail.retargeting.agentic run` (`<run>/native/robot_UNREVIEWED.npz`, 50 qpos /
43 named DOFs, object poses, scale and support scene) and the reviewed contact map of the
same video from the HOI stage:

```bash
python -m grail.retargeting.manipulation.handoff \
    --native runs/table_046_s02/retarget/native/robot_UNREVIEWED.npz \
    --contact-map runs/table_046_s02/recon/contact_map.json \
    --motion-key table_046_s02 --output runs/table_046_s02/tracking_handoff
```

The handoff copies the inputs unchanged, adds the object mesh in original OBJ order and
uses `native` support for floor-only scenes, including an explicitly labeled floor or ground mesh. Table scenes and older unlabeled support meshes use `table-block`. Use `--support` to override the choice.


## Optional stance-floor correction

For a G1+DEx3 motion with a registered finite floor and bound object mesh, run the following in an environment with MuJoCo, OSQP and hpp-fcl. The command creates a separate, unselected NPZ. Inspect sole heights, wrist targets and original-mesh clearance before selecting it.
```bash
python -m grail.retargeting.stance_floor --source native.npz --model-xml robot.xml --object-mesh original.obj --output stance_candidate.npz
```
