#!/usr/bin/env python3
"""Deterministic evaluation of a trained manipulation tracking checkpoint on one motion.

Run in the ``sonic`` environment from the repository root::

    python -m grail.tracking.manipulation.evaluate --pack runs/table_046/pack \\
        --motion table_046_s02 --checkpoint runs/table_046/finetune/train/model_step_005000.pt \\
        --output runs/table_046/eval_5000

One environment, seed 0, policy mean actions, start at reference frame 0, no optimizer, no PPO
update and no reset during the reference. The checkpoint's saved ``train/config.yaml`` (default:
next to the checkpoint) supplies the task, reward and observation settings; only the decoder
paths, roster and physics file are replaced. Outputs under ``--output/output``: ``eval.mp4``,
``frames.json``, ``evaluation.json`` (end reason, reference coverage, object position/orientation
errors) and ``evaluation_config.yaml``; ``--no-video`` skips the camera (metrics only) for hosts
without RTX rendering. A technically valid video can still show a policy failure;
``end_reason`` and ``reference_coverage`` report it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from grail.tracking.manipulation import paths


def find_motion(pack: Path, motion: str, group_id: str | None) -> dict:
    catalog = json.loads((pack / "tracking_pack.json").read_text())
    matches = []
    for group in catalog["groups"]:
        if group_id and group["group_id"] != group_id:
            continue
        for item in group["motions"]:
            if motion in (item["motion_id"], item["source_motion_key"]):
                matches.append((group, item))
    if len(matches) != 1:
        raise ValueError(f"Motion {motion!r} matched {len(matches)} entries in {pack}")
    return {**matches[0][1], "physics_path": matches[0][0]["physics_path"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--motion", required=True, help="Motion key or motion id from tracking_pack.json")
    p.add_argument("--group", default=None)
    p.add_argument("--checkpoint", type=Path, required=True, help="Full trained checkpoint (model_step_*.pt)")
    p.add_argument("--train-config", type=Path, default=None, help="Saved training config (default: <checkpoint dir>/config.yaml)")
    p.add_argument("--output", type=Path, required=True, help="New evaluation directory")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--no-video", action="store_true",
                   help="Simulate the same episode without cameras (metrics only) on hosts without RTX rendering")
    p.add_argument("--decoder-config", type=Path, default=paths.SONIC_DECODER_CONFIG)
    p.add_argument("--decoder-checkpoint", type=Path, default=paths.SONIC_DECODER_CHECKPOINT)
    p.add_argument("--sonic-root", type=Path, default=paths.SONIC_MANIP_ROOT)
    args = p.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new evaluation directory: " + str(output))
    motion = find_motion(args.pack.resolve(), args.motion, args.group)
    checkpoint = args.checkpoint.resolve(strict=True)
    train_config = (args.train_config or checkpoint.parent / "config.yaml").resolve(strict=True)
    output.mkdir(parents=True)
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.load(train_config)
    with open_dict(cfg):
        cfg.manager_env.config.action_transform_module_cfg = str(args.decoder_config.resolve(strict=True))
        cfg.manager_env.config.action_transform_module_checkpoint = str(args.decoder_checkpoint.resolve(strict=True))
    source = output / "source_train_config.yaml"
    OmegaConf.save(cfg, source)
    import torch
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = value["state"]
    step = state.get("global_step") if isinstance(state, dict) else state.global_step
    del value
    request = dict(remote_repo=str(args.sonic_root.resolve()), source_commit=paths.UPSTREAM_SONIC_COMMIT,
                   case_id=motion["source_motion_key"], cpus=args.cpus, phase="final_eval",
                   checkpoint=str(checkpoint), checkpoint_sha256=paths.sha256(checkpoint), checkpoint_step=step,
                   source_train_config=str(source), source_train_config_sha256=paths.sha256(source),
                   output_root=str(output / "output"), num_envs=1, deterministic=True, start_from_first_frame=True,
                   seed=0, parent_wandb_run_id=None, parent_train_run_name=None, parent_train_output_root=str(output),
                   roster_manifest=motion["single_env_roster_path"],
                   roster_manifest_sha256=paths.sha256(motion["single_env_roster_path"]), direct_dirs=None,
                   motion_id=motion["motion_id"], source_motion_key=motion["source_motion_key"],
                   object_mesh_sha256=motion["object_mesh_sha256"],
                   source_reference_frames=motion["source_reference"]["frames"],
                   source_reference_fps=motion["source_reference"]["fps"],
                   physics_overrides=motion["physics_path"], shader_cache_seed=None, render_video=not args.no_video)
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    child = paths.RUNTIME_DIR / "eval_entry.py"
    env = dict(os.environ, WANDB_MODE="disabled", OMNI_KIT_ACCEPT_EULA="Yes", HYDRA_FULL_ERROR="1",
               PYTHONPATH=os.pathsep.join([str(paths.RUNTIME_DIR), str(args.sonic_root.resolve())]),
               OMP_NUM_THREADS=str(args.cpus))
    with (output / "eval.log").open("w") as stream:
        result = subprocess.run([sys.executable, "-u", str(child), str(request_path)], cwd=args.sonic_root.resolve(),
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    with (output / "verify.log").open("w") as stream:
        verified = subprocess.run([sys.executable, str(child), "--verify-completion", str(request_path), str(result.returncode)],
                                  cwd=args.sonic_root.resolve(), env=env, stdout=stream, stderr=subprocess.STDOUT)
    if verified.returncode:
        print(f"Evaluation failed; see {output / 'eval.log'} and {output / 'verify.log'}", file=sys.stderr)
        return 1
    evaluation = json.loads((output / "output/evaluation.json").read_text())
    rollout = evaluation["rollout"]
    summary = {"motion": motion["source_motion_key"], "checkpoint_step": step, "end_reason": rollout["end_reason"],
               "reference_coverage": rollout["reference_clock"]["recorded_runtime_fraction"],
               "tracking_success": rollout.get("tracking_success"), "object_metrics": evaluation.get("object_metrics", {}),
               "video": None if args.no_video else str(output / "output/eval.mp4")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
