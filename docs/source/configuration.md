# Configuration and Glossary

<span id="glossary-and-configuration"></span>

| Term | Meaning in GRAIL | Where it matters |
|---|---|---|
| HOI | Human–object interaction. The source video supplies timing and visual evidence for the human, object and their contact. | Reconstruction and grasp review |
| SOMA03 | The native human body representation used by the released reconstruction and retargeting workflows. The fixed body identity must match the fitted sequence. | Human fit, export to the robot solver |
| CAD binding | The identity, metric scale and original ordering of the object mesh. A replacement mesh with the same appearance can change contact indices. | Object tracking, contact maps, collision geometry |
| Static scene | Terrain or furniture stays fixed throughout the source sequence. Feet can switch between the floor and different support surfaces. | Stairs and other locomotion; review support in side/oblique views |
| Stage 1 / Stage 2 | Fit the human/object to the observations, then refine with the reviewed placement and relevant contact evidence. Static terrain and moving-object cases take different paths. | Selecting the parent fit |
| Unitree G1 | Unitree G1 with 29 articulated joints, or the G1 with articulated fingers. The target determines joint count and the available tracking handoff. | Retargeting and packing |
| Handoff / pack | A handoff binds the native motion to its scene and contact evidence. Packing converts those bindings to the training library expected by the tracker. | Passing between kinematic output and dynamics |
| PPO update | One optimization update of the tracking policy. The number of simulated environments controls how much experience is collected in parallel. | Training duration, memory and evaluation cadence |
| Case / worker | A case is one recording with its scene and outputs. A worker runs a particular model in its own software environment. | Installation and both guided stages |
| GEM-X / MoGe / MoSE3 | Models used for the human pose, monocular depth and moving-object pose, respectively. | Reconstruction weights |
| SONIC / scene-aware tracker | SONIC trains a policy in simulation. On stairs the policy follows the reference body motion over the terrain. | Stairs training |
| Object-aware latent adaptor | The manipulation tracking policy used for the G1 and the moving object. | Kettlebell training |
| Motion library | A directory of reference robot and scene trajectories consumed by a tracker. | Terrain pack and manipulation pack |
| Kling | Image-to-video service used for the bundled clips. | Video generation |
| Packed library | Robot, object and collider data in the layout a tracking policy reads. A handoff supplies the inputs to make it. | Tracking |
| Shard | A set of trajectories and a metrics index that can be processed together. | Export and rendering |
| Kinematic replay | Playback of stored reference poses without a simulated policy. | Checking what the robot was asked to do |
| Native | The fitted model's original parameter or geometry output, before packing for another tool. | SOMA geometry and G1 NPZ |
| Fixed body identity / rest calibration | The subject's shape and neutral joint transforms used to map human joints to the robot. | Pose transfer, including when the first video frame is bent |

Use [Reconstruction](recon_4dhoi.md) for scene and solver configuration, [Retargeting](retargeting.md) for the robot target and [Outputs and Coordinate Conventions](outputs.md) when writing a consumer. Preserve the source frame clock when moving between them.

## Configuration by motion type

<span id="choose-a-configuration-by-what-moves"></span>

| File | Use it when | What you must provide |
|---|---|---|
| `configs/gen_2dhoi/terrain_stairs.yaml` | Fixed terrain | A stair object list, room, character and metric mesh. [Render one seed](gen_2dhoi.md). |
| `configs/gen_2dhoi/pickup_ground.yaml` | Pick up a rigid object from the floor | `indoor2-pickup-ground`, your object list and metric mesh. The preset renders start and end images, uses the settled initial state and skips prompt refinement. |
| `configs/gen_2dhoi/pickup_table.yaml` | Pick up from a tabletop | `indoor2-pickup-table`, your object list and measured table support. The preset uses the starting image and enables prompt refinement. |
| `configs/recon_4dhoi/loco_soma.yaml` | The scene stays rigid and a foot may land above the floor | A case with reviewed support intervals and finite floor/tread patches tied to the source and native geometry. |
| `configs/recon_4dhoi/manip_soma.yaml` | The human moves a rigid object | A measured ground plane in both fitting stages, first-frame placement, real object motion and a hand contact map for refinement. |
| `configs/agentic_kinematic/robot.json` | Configuring native G1 arm and hand edits | Replace the sample trajectory, targets and mesh paths with those of the case you reviewed. Follow [G1 retargeting](retargeting.md) for terrain traversal and manipulation with articulated hands. |

A **support interval** names which foot region bears weight on which finite surface over which source frames. A **contact interval** names evidence for hand contact, no contact or uncertainty. Motion timestamps use seconds; both interval types refer to source frames. A **review** records which geometry and frames were inspected and what remains unresolved. The file diagram and runnable reader are in [Outputs and coordinates](outputs.md).

## Pipeline stages

<span id="how-the-stages-fit-together"></span>

The paper describes robot-centric human video generation, interaction-aware HOI reconstruction and task-general loco-manipulation tracking. Generation starts from a known camera, metric assets and a simulator-ready scene. Reconstruction recovers human and object motion from that clip. Retargeting maps the reviewed motion to Unitree G1. For terrain traversal the policy uses scene geometry; for object manipulation the object-aware latent adaptor supplies hand actions and adapts a pretrained whole-body controller. In the implementation, `unitree_g1` and `unitree_g1_dex3` select different joint layouts. The manipulation target has seven finger joints per hand. [GRAIL paper, Sections 3.1–3.3](https://arxiv.org/abs/2606.05160v2).
