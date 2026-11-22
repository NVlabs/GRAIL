# Quick Start: choose a tutorial

Start with a bundled video and its metric scene. GRAIL fits a moving human to the footage, transfers that motion to Unitree G1, then trains a policy to follow the reference in simulation.

**Already set up?** Start at {ref}`stairs, step 1 <stairs-fit-step>` or {ref}`kettlebell, step 1 <kettlebell-fit-step>`. Each tutorial gives the host prompts and container command. If this is your first visit, check the [hardware budget](requirements.md), [open the container](installation.md) and [get the weights](downloads.md) once.

| Tutorial | What you will see | Tracking result |
|---|---|---|
| [Climb stairs](tutorials/stairs.md) | A human fit aligned to the stair treads, then a body-only G1 following the reference | Terrain policy checkpoint and progress metrics |
| [Lift a kettlebell](tutorials/kettlebell.md) | A moving object, reviewed hand contact, then a G1 grasp with DEx3 fingers | ContactFirst checkpoint, reference coverage and object error |

Find the two review movies, `reconstruction.mp4` and `retargeting.mp4`, in `../grail-demo/runs/<case>/` beside the checkout. Evaluation writes into that case's `tracking/` directory. The [output reference](outputs.md) explains which files to open, retain or pass to another stage.

Compare [example results at each stage](results.md), including the five full-schedule checkpoints.

For a new object or recording, see [Prepare a new scene and video](tutorials/own-video.md). The [glossary](configuration.md) explains the stage names.
