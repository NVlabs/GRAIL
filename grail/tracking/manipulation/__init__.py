"""Manipulation tracking finetune on agentic retargeting output.

Pipeline (see docs/source/agentic_kinematic.md, "Manipulation tracking finetune"):

1. ``handoff``  - collect a retargeted G1 + DEx3 motion, its object mesh and HOI contact map.
2. ``pack``     - convert handoffs into training packs and per-object 2048-slot cohorts,
                  with the small table-support block for table scenes and 0.1 kg object mass.
3. ``train``    - per-object PPO finetune from the released warm-start actor.
4. ``evaluate`` - deterministic single-environment evaluation of a checkpoint on one motion.

The simulation side runs on the vendored SONIC fork in ``imports/SONIC-manipulation``;
``runtime/`` holds the modules loaded inside its training and evaluation processes.
"""
