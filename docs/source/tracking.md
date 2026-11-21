# Motion tracking

Follow the {blob}`stair and kettlebell Quick Start <README.md#4-track-the-retargeted-motion>` for the full training and evaluation commands. Both examples finetune with 2,048 environments for 5,000 PPO updates and evaluate checkpoints at 1,000-update intervals.

- **Stairs.** Package the body-only G1 motion and stair collider with `python -m grail.retargeting.terrain`, then run the SONIC terrain trainer. Each checkpoint writes `tracking/eval/step_XXXXXX/metrics_eval.json`.
- **Kettlebell.** The [manipulation handoff](retargeting.md#hand-off-a-retargeted-motion) binds the G1+DEx3 motion, source contact map and ordered object mesh. Run `scripts/tracking/manipulation_finetune.sh` on that handoff to build the pack, train ContactFirst and evaluate each saved checkpoint after training releases the GPU. The final rollout and metrics are in `tracking/eval_final/kettlebell/`.

Install the `sonic` environment with `scripts/setup/install_env_sonic.sh`. The checkpoint downloader installs the terrain policy, frozen manipulation decoder and [ContactFirst actor warmstart](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/tree/main/checkpoint/SONIC/models/manip_warmstart). The [GRAIL dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) also includes released motion data.
