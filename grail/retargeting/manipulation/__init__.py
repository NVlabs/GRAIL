"""Manipulation tracking data production for the SONIC ContactFirst finetune.

This package produces the training data (see docs/source/retargeting.md,
"Manipulation finetune data (handoff + pack)"):

1. ``handoff`` - collect a retargeted G1 + DEx3 motion, its object mesh and HOI contact map.
2. ``pack``    - convert handoffs into training packs and per-object 2048-slot cohorts,
                 with the small table-support block for table scenes and 0.1 kg object mass.

Training and evaluation are SONIC commands on the unified ``imports/SONIC`` tree::

    cd imports/SONIC
    python gear_sonic/train_agent_trl.py \
        +exp=manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_finetune ...
    python gear_sonic/eval_contactfirst_trl.py --pack ... --motion ... --checkpoint ...

``scripts/tracking/manipulation_finetune.sh`` chains pack + finetune + final evaluation
for one object.
"""
