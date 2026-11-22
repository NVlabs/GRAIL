# Citation, Licences and Support

**Cite GRAIL.** Use the paper *GRAIL: Generating Humanoid Loco-Manipulation from World Models*, published at CoRL 2026. The repository provides a ready-to-copy {blob}`BibTeX entry <README.md#citation>`. [Read the paper](https://arxiv.org/abs/2606.05160).

**Check the terms for the files you use.** Section 3.3 of the {blob}`NVIDIA licence <LICENSE>` limits use of the work and derivatives to non-commercial research or evaluation, with an exception for NVIDIA Corporation and its affiliates. Third-party assets have their own terms; the Unitree robot assets ship with their {blob}`Unitree notice <assets/robots/unitree_g1/LICENSE>` and {blob}`GMR notice <assets/robots/unitree_g1/GMR-LICENSE>`. Consult the dataset card for the terms of the [published motions and checkpoints](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL).

**Get help.** Check [Troubleshooting](troubleshooting.md) for installation, geometry and evaluation errors. For a reproducible bug, open a [GRAIL issue](https://github.com/NVlabs/GRAIL/issues) with the example name, image tag, GPU and driver, failing stage, command and first error. Include a short log excerpt and the expected output.

## Bundled components

<span id="components-used-by-the-examples"></span>

GEM-X predicts the human pose and SOMA represents the body. MoGe estimates depth; MoSE3 and FoundationPose track moving objects. The SOMA retargeter converts the reviewed motion to G1, and SONIC trains the tracking policy. Optional scene-generation components include Hunyuan3D and Sana. [GRAIL Installation Guide](installation.md) initializes the pinned dependencies.

For a source checkout, [initialize submodules and hydrate LFS](installation.md). FoundationPose supplies model-based pose refinement, GEM-X supplies the human predictor and MoGe supplies depth. Hunyuan3D produces optional textured assets. Sana provides the optional MiniMax service implementation. The unified SONIC code is vendored. The robot model and its licence files live under `assets/robots/unitree_g1/`.
