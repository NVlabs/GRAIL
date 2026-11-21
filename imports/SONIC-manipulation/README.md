# SONIC manipulation tracking tree (vendored)

This directory is a plain-source snapshot of a SONIC (`gear_sonic`) fork with the
manipulation tracking extensions that trained the GRAIL manipulation policies
(per-object finetune from a generalist warm start). GRAIL drives it through
`python -m grail.tracking.manipulation.{handoff,pack,train,evaluate}`; see
[Agentic workflows: manipulation tracking finetune](../../docs/source/agentic_kinematic.md#manipulation-tracking-finetune).

Upstream revision: `fb8563f9a5b5889d50bccba4dc07a607f2e4a71c`.

## Why a separate tree

The fork changes shared SONIC modules (motion commands, motion library, rewards,
observations, terminations, environment wrapper, PPO trainer) and the shared
`advanced_manip_ground` experiment, while the main [`imports/SONIC`](../SONIC/) tree carries
later public fixes. Merging them would change the behavior of either the released public
tracking configs or the manipulation finetune. They therefore stay separate:

- `imports/SONIC` - public SONIC release; all configs in [tracking](../../docs/source/tracking.md) are unchanged.
- `imports/SONIC-manipulation` - this tree; only the manipulation finetune uses it.

Put exactly one of the two trees on `PYTHONPATH`; both provide the `gear_sonic` package.
The GRAIL entry points set the path themselves.

## Contents

| Path | Content |
| --- | --- |
| `gear_sonic/` | Fork package: environments, rewards, observations, motion library, PPO trainer, Hydra configs. The manipulation experiment is `gear_sonic/config/exp/manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_ablation_d.yaml`; `gear_sonic/rp2/` holds the cohort, contact-geometry and support-context modules. |
| `gear_sonic/data` | Symlink to `../../SONIC/gear_sonic/data`. The robot assets are byte-identical to the public SONIC tree; the fork's additional collision layer `robots/g1/g1_43dof_s3_yam_core_collision.usda` is stored there. Run `git lfs pull` to hydrate them. |
| `tools/` | Cohort builder (`build_unified_no_vof.py`), rigid active-object and articulation pack writers, static-context USD writer, checkpoint step reader. |
| `LICENSE` | License of this snapshot. |

Not included: the FBX motion loader and its prebuilt binaries, cluster launch scripts,
tests, research notes and dataset-specific converters. Comments that referred to private
infrastructure were reworded and the deployment-specific mount-alias table in
`gear_sonic/rp2/rank_scene_plan_cache.py` is empty; program behavior is unchanged.

## License

Source code is under the Apache License 2.0 and model weights under the NVIDIA Open
Model License, as stated in [LICENSE](LICENSE).
