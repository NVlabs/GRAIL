# Quick Start

<span id="quick-start-choose-a-tutorial"></span>

Start with a bundled video and its metric scene. GRAIL fits a moving human to the footage, transfers that motion to Unitree G1, then trains a policy to follow the reference in simulation.

**Already set up?** Start at {ref}`stairs, step 1 <stairs-fit-step>` or {ref}`kettlebell, step 1 <kettlebell-fit-step>`. Each tutorial gives the host prompts and container command. If this is your first visit, check the [hardware budget](requirements.md), [open the container](installation.md) and [get the weights](downloads.md) once.

| Tutorial | What you will see | Tracking result |
|---|---|---|
| [Stair climbing](tutorials/stairs.md) | A human fit aligned to the stair treads, then a Unitree G1 following the reference | Terrain policy checkpoint and progress metrics |
| [Kettlebell lifting](tutorials/kettlebell.md) | A moving object, reviewed hand contact, then a G1 grasp with articulated fingers | manipulation policy checkpoint, reference coverage and object error |

Find the two review movies, `reconstruction.mp4` and `retargeting.mp4`, in `../grail-demo/runs/<case>/` beside the checkout. Evaluation writes into that case's `tracking/` directory. The [output reference](outputs.md) explains which files to open, retain or pass to another stage.

Compare [example results at each stage](results.md), including the five full-schedule checkpoints.

For a new object or recording, see [New Scenes and Videos](tutorials/own-video.md). The [glossary](configuration.md) explains the stage names.

## Next steps

<span id="start-before-the-example-video-or-inspect-an-output"></span>

[Start with the released stairs or kettlebell tutorial](#quick-start).

<span id="source-environment-keys"></span>

[Configure the generation environment](manual_installation.md). The bundled videos already include their scene inputs.

<span id="generate-a-3d-object-procedural-no-gpu-needed"></span><span id="generate-a-2d-hoi-video-for-one-object"></span>

[Create an asset](gen_3d_assets.md) and [generate video and camera inputs](gen_2dhoi.md).

<span id="reconstruct-4d-motion-from-the-video"></span><span id="retarget-native-soma-motion-to-unitree-g1-dex3"></span><span id="inspect-outputs"></span>

[Fit the human and object](recon_4dhoi.md), [retarget to Unitree G1](retargeting.md) and [inspect the output files](outputs.md).
