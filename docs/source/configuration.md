# Glossary and configuration

| Term | Meaning in GRAIL | Where it matters |
|---|---|---|
| HOI | Human–object interaction. The source video supplies timing and visual evidence for the human, object and their contact. | Reconstruction and grasp review |
| SOMA03 | The native human body representation used by the released reconstruction and retargeting workflows. The fixed body identity must match the fitted sequence. | Human fit, export to the robot solver |
| CAD binding | The identity, metric scale and original ordering of the object mesh. A replacement mesh with the same appearance can change contact indices. | Object tracking, contact maps, collision geometry |
| Static scene | Terrain or furniture stays fixed throughout the source sequence. Feet can switch between the floor and different support surfaces. | Stairs and other locomotion; review support in side/oblique views |
| Stage 1 / Stage 2 | Fit the human/object to the observations, then refine with the reviewed placement and relevant contact evidence. Static terrain and moving-object cases take different paths. | Selecting the parent fit |
| G1 / DEx3 | Body-only Unitree G1, or the G1 with articulated fingers. The target determines joint count and the available tracking handoff. | Retargeting and packing |
| Handoff / pack | A handoff binds the native motion to its scene and contact evidence. Packing converts those bindings to the training library expected by the tracker. | Passing between kinematic output and dynamics |
| PPO update | One optimization update of the tracking policy. The number of simulated environments controls how much experience is collected in parallel. | Training duration, memory and evaluation cadence |
| Case / worker | A case is one recording with its scene and outputs. A worker runs a particular model in its own software environment. | Installation and both guided stages |
| GEM-X / MoGe / MoSE3 | Models used for the human pose, monocular depth and moving-object pose, respectively. | Reconstruction weights |
| SONIC / terrain tracker | SONIC trains a policy in simulation. On stairs the policy follows the reference body motion over the terrain. | Stairs training |
| ContactFirst | The manipulation tracking policy used for the G1 and the moving object. | Kettlebell training |
| Motion library | A directory of reference robot and scene trajectories consumed by a tracker. | Terrain pack and manipulation pack |
| Kling | A video generation service. The example MP4s have already been generated. | Generating a new synthetic scene |

Use [Reconstruction](recon_4dhoi.md) for scene and solver configuration, [Retargeting](retargeting.md) for the robot target and [Outputs and coordinate conventions](outputs.md) when writing a consumer. Preserve the source frame clock when moving between them.
