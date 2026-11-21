#!/usr/bin/env python3
"""Per-object manipulation tracking finetune from the released warm-start actor.

Run in the ``sonic`` environment (Isaac Lab) from the repository root::

    python -m grail.tracking.manipulation.train --pack runs/table_046/pack \\
        --output runs/table_046/finetune --iterations 5000

``--pack`` is a directory written by ``python -m grail.tracking.manipulation.pack``. The run
composes the vendored SONIC experiment
``manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_ablation_d`` with the
production launch settings (2048 environments, 4 PPO mini-batches, 24 rollout steps per
environment, actor LR 2e-5, hand-target slew close 4/s / open 5/s, no observation corruption
or domain randomization, beta-capped adaptive motion sampling), initializes the actor exactly
from the warm-start checkpoint with a fresh critic and optimizer, and saves full checkpoints
every 1000 updates plus the final update. With ``--eval-every N`` (production: 1000) training
pauses at every N-th update and evaluates that checkpoint deterministically on each motion.

All outputs (checkpoints, logs, receipts) go under ``--output``; inputs are read only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import traceback

from grail.tracking.manipulation import paths

FATAL_PHYSICS_MESSAGES = (
    "PhysX ABORT error: GPU SDF cooking failed!",
    "Cuda context manager error, simulation will be stopped",
)
DEPENDENCIES = {
    "torch": "torch", "numpy": "numpy", "scipy": "scipy", "hydra-core": "hydra",
    "accelerate": "accelerate", "transformers": "transformers", "trl": "trl",
    "joblib": "joblib", "easydict": "easydict", "loguru": "loguru", "trimesh": "trimesh",
    "opencv-python": "cv2", "isaaclab": "isaaclab",
}
# Tuning surface a caller may override with --override (Hydra syntax).
ALLOWED_OVERRIDE_PREFIXES = ("algo.config.", "manager_env.rewards.", "manager_env.terminations.",
                             "manager_env.events.", "manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.")
ALLOWED_OVERRIDE_KEYS = {"manager_env.config.latent_residual_scale", "manager_env.config.hand_target_slew.enabled",
                         "manager_env.config.hand_target_slew.close_per_s", "manager_env.config.hand_target_slew.open_per_s"}
MANAGED_KEYS = {"algo.config.num_learning_iterations", "algo.config.num_mini_batches"}
# Production hyperparameters, passed explicitly so the launch is self-describing.
PRODUCTION_OVERRIDES = [
    "++algo.config.actor_learning_rate=2e-5",
    "++algo.config.critic_learning_rate=1e-3",
    "++algo.config.use_log_std=false",
    "++algo.config.freeze_noise_std=false",
    "++manager_env.config.hand_target_slew.enabled=true",
    "++manager_env.config.hand_target_slew.close_per_s=4.0",
    "++manager_env.config.hand_target_slew.open_per_s=5.0",
]


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def real_file(value, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} must name a nonempty absolute file: {path}")
    with path.open("rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{label} is an unhydrated Git LFS pointer (run `git lfs pull`): {path}")
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", type=Path, required=True, help="Directory written by grail.tracking.manipulation.pack")
    p.add_argument("--group", default=None, help="Object group id inside the pack (default: the only group)")
    p.add_argument("--output", type=Path, required=True, help="New, empty run directory")
    p.add_argument("--iterations", type=int, default=5000, help="Absolute PPO update target (production: 5000)")
    p.add_argument("--num-envs", type=int, default=2048, help="Training environments on one GPU (production: 2048)")
    p.add_argument("--cpus", type=int, default=8, help="CPU threads for Kit/TBB/PhysX in the training process")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Deterministic per-motion evaluation every N updates (production: 1000; 0 disables)")
    p.add_argument("--eval-cpus", type=int, default=2, help="CPU threads for each evaluation child")
    p.add_argument("--eval-no-video", action="store_true",
                   help="Periodic evaluation without cameras (metrics only), for hosts without RTX rendering")
    p.add_argument("--warmstart", type=Path, default=paths.WARMSTART_ACTOR, help="Actor-only warm-start checkpoint")
    p.add_argument("--decoder-config", type=Path, default=paths.SONIC_DECODER_CONFIG)
    p.add_argument("--decoder-checkpoint", type=Path, default=paths.SONIC_DECODER_CHECKPOINT)
    p.add_argument("--sonic-root", type=Path, default=paths.SONIC_MANIP_ROOT, help="Vendored SONIC manipulation tree")
    p.add_argument("--resume", action="store_true", help="Continue this run from its own train/last.pt")
    p.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    p.add_argument("--wandb-project", default="grail-manipulation-tracking")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--override", action="append", default=[],
                   help="Extra Hydra override on the tuning surface (algo.config.*, manager_env.rewards.*, ...)")
    p.add_argument("--allow-warmstart-mismatch", action="store_true",
                   help="Accept a warm-start actor whose SHA-256 differs from the released one")
    return p.parse_args(argv)


def select_group(pack: Path, group_id: str | None) -> dict:
    catalog = json.loads((pack / "tracking_pack.json").read_text())
    groups = catalog["groups"]
    if group_id is None:
        if len(groups) != 1:
            raise ValueError("Pack has several object groups; choose one with --group: " + ", ".join(g["group_id"] for g in groups))
        return groups[0]
    for group in groups:
        if group["group_id"] == group_id:
            return group
    raise ValueError(f"Unknown group {group_id}")


def build_config(args) -> dict:
    output = args.output.resolve()
    group = select_group(args.pack.resolve(), args.group)
    config = {
        "schema": "grail.tracking.manipulation.run_config.v1",
        "sonic_root": str(args.sonic_root.resolve()), "source_commit": paths.UPSTREAM_SONIC_COMMIT,
        "output_root": str(output), "mode": "train", "iterations": args.iterations,
        "num_envs": args.num_envs, "num_gpus": 1, "cpus": args.cpus, "eval_cpus": args.eval_cpus,
        "resume": bool(args.resume), "physx_buffers": dict(paths.PHYSX_BUFFERS),
        "actor_init_checkpoint": str(args.warmstart.resolve()),
        "actor_init_sha256": paths.WARMSTART_ACTOR_SHA256,
        "model_config": str(args.decoder_config.resolve()), "model_checkpoint": str(args.decoder_checkpoint.resolve()),
        "roster_manifest": group["roster_path"], "physics_overrides": group["physics_path"],
        "group_id": group["group_id"], "motions": group["motions"],
        "run_name": args.run_name or output.name,
        "wandb_mode": args.wandb_mode, "wandb_project": args.wandb_project, "wandb_entity": args.wandb_entity,
        "periodic_eval": {"frequency": args.eval_every, "video": not args.eval_no_video} if args.eval_every > 0 else None,
        "overrides": PRODUCTION_OVERRIDES + list(args.override),
    }
    for override in args.override:
        key = override.lstrip("+").split("=", 1)[0]
        if "=" not in override or key in MANAGED_KEYS or not (key.startswith(ALLOWED_OVERRIDE_PREFIXES) or key in ALLOWED_OVERRIDE_KEYS):
            raise ValueError(f"Override is outside the training tuning surface: {key}")
    if args.iterations < 1 or args.num_envs < 4 or args.num_envs % 4:
        raise ValueError("Need a positive iteration target and num_envs divisible by the four PPO mini-batches")
    if output == Path(config["sonic_root"]) or Path(config["sonic_root"]) in output.parents:
        raise ValueError("--output must be outside the vendored source tree")
    return config


def claim_output(config: dict, output: Path):
    """One advisory lease per output; a fresh run needs an empty directory, resume its own identity."""
    lease = (output / ".runtime.lock").open("a+")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise RuntimeError("Another process owns this output directory") from None
    identity = {key: config.get(key) for key in ("run_name", "output_root", "roster_manifest", "iterations", "num_envs")}
    path = output / "run_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity or not config["resume"]:
            lease.close()
            raise ValueError("Output already holds a run; use --resume with identical settings or a new --output")
    elif config["resume"]:
        lease.close()
        raise ValueError("--resume requires this output's existing run_identity.json")
    elif any(child.name != ".runtime.lock" for child in output.iterdir()):
        lease.close()
        raise ValueError("Fresh --output must be empty")
    if config["resume"]:
        last = output / "train/last.pt"
        if not last.is_file():
            lease.close()
            raise ValueError("--resume requires train/last.pt")
        config["resume_checkpoint_sha256"] = paths.sha256(last)
        sys.path[:0] = [config["sonic_root"]]
        from tools.ckpt_global_step import read_global_step
        config["resume_from_step"] = read_global_step(str(last))
        if config["resume_from_step"] is None or config["resume_from_step"] >= config["iterations"]:
            lease.close()
            raise ValueError("Resume target must exceed the saved checkpoint step")
    else:
        write_json(path, identity)
    return lease


def source_check(config: dict, report: dict) -> Path:
    repo = Path(config["sonic_root"]).resolve()
    for relative in ("gear_sonic/train_agent_trl.py", f"gear_sonic/config/exp/{paths.EXPERIMENT}.yaml"):
        if not (repo / relative).is_file():
            raise ValueError(f"Vendored SONIC tree is incomplete: missing {relative}")
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(paths.RUNTIME_DIR)]
    sonic = importlib.import_module("gear_sonic")
    imported = Path(sonic.__file__).resolve()
    if imported != repo / "gear_sonic/__init__.py":
        raise ValueError(f"gear_sonic imports from the wrong checkout: {imported} (use the vendored tree only)")
    for relative in ("gear_sonic/data/robots/g1/g1_43dof_s3.usda",
                     "gear_sonic/data/robots/g1/g1_43dof_s3_yam_core_collision.usda",
                     "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"):
        real_file(str(repo / relative), relative)
    report["source"] = {"upstream_commit": config["source_commit"], "vendored_tree": str(repo), "import_path": str(imported)}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, timeout=30)
        if head.returncode == 0:
            report["source"]["grail_git_head"] = head.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return repo


def dependencies_check(report: dict) -> None:
    errors, found = [], {}
    for package, module in DEPENDENCIES.items():
        try:
            imported = importlib.import_module(module)
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(imported, "__version__", "unknown")
            found[module] = str(version)
        except Exception as exc:
            errors.append(f"{module}: {type(exc).__name__}: {exc}")
    report["dependencies"] = found
    if not errors:
        try:
            importlib.import_module("gear_sonic.trl.trainer.ppo_trainer")
        except Exception as exc:
            errors.append(f"PPO implementation import: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("Runtime dependencies are missing (use the `sonic` environment): " + "; ".join(errors))


def data_check(config: dict, report: dict) -> dict:
    os.environ["JOINT_ROSTER_MANIFEST"] = config["roster_manifest"]
    os.environ["JOINT_ROSTER_DIRECT_DIRS"] = ""
    from gear_sonic.rp2.joint_roster import resolve_rank_local_roster
    view = resolve_rank_local_roster(str(real_file(config["roster_manifest"], "roster_manifest")),
                                     num_envs=config["num_envs"], rank=0, world_size=1, runtime_target_fps_hz=50.0)
    report["data"] = {"status": "roster_resolved", "motion_count": view["resolved_motion_count"], "num_envs": view["num_envs"]}
    return view


def training_overrides(config: dict, output: Path, view: dict) -> list[str]:
    """Hydra arguments of the production manipulation finetune (values unchanged from production)."""
    from local_cache import cache_root
    prefix = "manager_env.commands.motion.motion_lib_cfg."
    values = {
        "algo.config.num_learning_iterations": config["iterations"], "algo.config.num_mini_batches": 4,
        "callbacks.model_save.save_frequency": min(1000, config["iterations"]),
        "callbacks.model_save.save_last_frequency": math.gcd(20, config["iterations"]),
        "manager_env.config.action_transform_module_cfg": config["model_config"],
        "manager_env.config.action_transform_module_checkpoint": config["model_checkpoint"],
        "manager_env.config.motion_meta_info_path": None, "manager_env.config.safe_nan": True,
        prefix + "max_num_objects": 1, prefix + "max_unique_motions": None, prefix + "remove_motion_keys": None,
        prefix + "asset.assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
        prefix + "smpl_motion_file": "dummy", prefix + "multi_thread": False,
        prefix + "fp16_reference_tensors": True, prefix + "bps_object_name_resolution": False,
        "motion_shard_by_rank": True,
        "manager_env.commands.motion.sample_before_contact": False,
        "manager_env.commands.motion.sample_start_truncate_time_s": None,
        "manager_env.commands.motion.start_from_first_frame": False,
        "manager_env.terminations.time_out.params.cap_by_episode_length": True,
        "manager_env.commands.motion.object_position_randomize": False,
        "manager_env.commands.motion.use_dynamic_target_object_mask": False,
        "manager_env.config.multi_object_per_env": False, "manager_env.commands.motion.full_scene_per_env": False,
        "manager_env.config.enable_multi_object_contact_sensors": False,
        "manager_env.config.enable_multi_object_support_contact_sensors": False,
        "manager_env.config.enable_feet_terrain_contact": True,
        "manager_env.config.render_results": False,
    }
    use_wandb = config["wandb_mode"] != "disabled"
    values.update({"use_wandb": use_wandb, "wandb.wandb_project": config["wandb_project"],
                   "wandb.wandb_entity": config["wandb_entity"], "wandb.wandb_dir": str(output / "wandb")})
    if config["periodic_eval"]:
        # The same-step checkpoint must exist when the periodic evaluation runs.
        values["callbacks.model_save.save_frequency"] = config["periodic_eval"]["frequency"]
        for key, value in {"_target_": "contactfirst_callbacks.PeriodicEvalCallback", "output_root": str(output),
                           "plan_file": str(output / "eval_plan.json"),
                           "frequency": config["periodic_eval"]["frequency"],
                           "video": config["periodic_eval"]["video"]}.items():
            values["callbacks.periodic_eval." + key] = value
    callback = {"_target_": "contactfirst_callbacks.ContactFirstCallback", "output_root": str(output),
                "run_name": config["run_name"]}
    if config.get("actor_init_checkpoint") and not config["resume"]:
        callback.update(actor_init_checkpoint=config["actor_init_checkpoint"], verify_actor_initialization=True)
    for key, value in callback.items():
        values["callbacks.contactfirst." + key] = value
    values["manager_env.recorders._target_"] = "group_metrics.make_recorders"
    values["manager_env.recorders.output_root"] = str(output)
    for key in ("add_joint_default_pos", "base_com", "randomize_rigid_body_mass", "randomize_table_size",
                "randomize_object_size", "physics_material"):
        values[f"manager_env.events.{key}"] = None
    for key in ("policy", "tokenizer", "policy_atm"):
        values[f"manager_env.observations.{key}.enable_corruption"] = False
    for key in ("adaptive_sampling.enable", "joint_source_tracking", "adaptive_sampling.distributed_keyed_sync",
                "adaptive_sampling.update_on_episode_end_only"):
        values[prefix + key] = True
    for key, value in {"priority_mode": "beta_capped_difficulty", "difficulty_warmup_completed_episodes": 100,
                       "difficulty_soft_success_floor": 0.10, "difficulty_min_weight": 0.05, "difficulty_beta_cap": 3.0,
                       "uniform_sampling_rate": 0.10, "init_num_failures": 1}.items():
        values[prefix + "adaptive_sampling." + key] = value
    for key, field in (("motion_file", "motion_dir"), ("object_motion_file", "object_motion_dir"), ("bps_dir", "bps_dir")):
        values[prefix + key] = view[field]
    values["manager_env.config.object_physics_overrides_path"] = config["physics_overrides"]
    for key, directory in ((prefix + "motion_tensor_cache", "motion_tensor"), ("manager_env.config.authored_stage_cache", "authored_stage")):
        values[key + ".mode"] = "read_write"
        values[key + ".cache_dir"] = str(cache_root(output, "train") / "training_data" / directory)
    values["manager_env.config.authored_stage_cache.attach_mode"] = "root_copy"
    values[prefix + "joint_roster_scene_plan_cache_mode"] = "read_write"
    values[prefix + "joint_roster_scene_plan_cache_dir"] = str(cache_root(output, "train") / "training_data/rank_scene_plan")
    args = [f"+exp={paths.EXPERIMENT}", f"num_envs={config['num_envs']}", "headless=true", f"base_dir={output}",
            "project_name=grail_manipulation_tracking", f"++experiment_dir={output / 'train'}"]
    args.extend(f"++{key}={json.dumps(value)}" for key, value in values.items())
    if config["resume"]:
        args.extend(["++resume=true", "++warm_resume=false", "++flexible_load=false",
                     "++algo.config.require_exact_checkpoint_actor_keys=true",
                     "++algo.config.reset_noise_std_on_checkpoint_load=false",
                     f"checkpoint={json.dumps(str(output / 'train/last.pt'))}"])
    else:
        args.extend(["++resume=false", f"checkpoint={json.dumps(config['actor_init_checkpoint'])}", "++warm_resume=false",
                     "++flexible_load=false", "++algo.config.require_exact_checkpoint_actor_keys=true",
                     "++algo.config.reset_noise_std_on_checkpoint_load=false"])
    args.extend(config["overrides"])
    return args


def compose_check(repo: Path, args: list[str], config: dict, output: Path, report: dict) -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from gear_sonic.utils.config_utils import register_rl_resolvers
    register_rl_resolvers()
    with initialize_config_dir(config_dir=str(repo / "gear_sonic/config"), version_base="1.1"):
        resolved = compose(config_name="base", overrides=args)
    if int(resolved.algo.config.num_learning_iterations) != config["iterations"]:
        raise ValueError("Overrides cannot change the iteration target")
    if int(resolved.num_envs) != config["num_envs"] or int(resolved.algo.config.num_mini_batches) != 4:
        raise ValueError("Overrides cannot change num_envs or the four-minibatch launch contract")
    if not any(o.startswith(("++algo.config.", "algo.config.")) for o in config["overrides"][len(PRODUCTION_OVERRIDES):]):
        protected = {"actor_learning_rate": 2e-5, "critic_learning_rate": 1e-3, "num_steps_per_env": 24,
                     "use_log_std": False, "freeze_noise_std": False}
        for key, expected in protected.items():
            if resolved.algo.config.get(key) != expected:
                raise ValueError("Production hyperparameter differs: " + key)
    if resolved.manager_env.config.robot.type != "g1_43dof_yam_core_collision":
        raise ValueError("Unexpected robot variant for the manipulation experiment")
    names = list(resolved.callbacks)
    if "periodic_eval" in names and not names.index("model_save") < names.index("periodic_eval") < names.index("contactfirst"):
        raise ValueError("Periodic evaluation must follow the checkpoint save and precede the W&B finish callback")
    (output / "composed_config.yaml").write_text(OmegaConf.to_yaml(resolved, resolve=False))
    report["config"] = {"experiment": paths.EXPERIMENT, "robot": resolved.manager_env.config.robot.type,
                        "iterations": config["iterations"], "num_envs": config["num_envs"]}


def decoder_check(config: dict, report: dict) -> None:
    """Strict load of the frozen SONIC action-transform decoder, as the environment wrapper does."""
    import torch
    from omegaconf import OmegaConf
    from gear_sonic.trl.utils.common import custom_instantiate
    raw = Path(config["model_config"]).read_text()
    for old, new in (("groot.rl.trl.", "gear_sonic.trl."), ("groot.rl.envs.", "gear_sonic.envs."),
                     ("groot.rl.utils.", "gear_sonic.utils."),
                     ("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module."),
                     ("groot.rl.agents.", "gear_sonic.trl.")):
        raw = raw.replace(old, new)
    exported = OmegaConf.create(raw)
    env_config, algo_config = exported.get("env_config", {}), exported.get("algo_config", {})
    model = custom_instantiate(algo_config.actor, env_config=env_config, algo_config=algo_config, _resolve=False).cpu()
    checkpoint = torch.load(config["model_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    report["decoder"] = {"status": "strict_weights_loaded", "checkpoint_sha256": paths.sha256(config["model_checkpoint"])}
    del checkpoint, model
    gc.collect()


def actor_initialization_check(config: dict, report: dict, allow_mismatch: bool) -> None:
    if config["resume"]:
        return
    import torch
    digest = paths.sha256(config["actor_init_checkpoint"])
    if digest != config["actor_init_sha256"] and not allow_mismatch:
        raise ValueError("Warm-start actor SHA-256 differs from the released checkpoint (--allow-warmstart-mismatch to accept)")
    config["actor_init_sha256"] = digest
    checkpoint = torch.load(config["actor_init_checkpoint"], map_location="cpu", weights_only=False)
    permitted = {"policy_state_dict", "state", "initialization_metadata"}
    if not isinstance(checkpoint, dict) or "policy_state_dict" not in checkpoint or set(checkpoint) - permitted:
        raise ValueError("Warm-start must be an actor-only initialization checkpoint")
    state = checkpoint.get("state")
    step = state.get("global_step", 0) if isinstance(state, dict) else getattr(state, "global_step", 0)
    if step != 0:
        raise ValueError("Actor-only initialization must have global_step=0")
    for name, value in checkpoint["policy_state_dict"].items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
            raise ValueError(f"Invalid actor initialization tensor: {name}")
    report["actor_initialization"] = {"status": "actor_only", "sha256": digest,
                                      "state_entries": len(checkpoint["policy_state_dict"])}
    del checkpoint
    gc.collect()


def write_eval_plan(config: dict, output: Path) -> None:
    plan = {"schema": "grail.tracking.manipulation.eval_plan.v1", "group_id": config["group_id"], "motions": config["motions"]}
    write_json(output / "eval_plan.json", plan)


def stop_child(process, timeout: float = 30) -> int:
    if process.poll() is None:
        process.terminate()
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=30)


def monitor_child(process, log) -> dict:
    """Stream the child log; stop it only on explicit fatal PhysX/CUDA messages."""
    result = {"fatal_traceback": False, "fatal_physics_error": None, "physx_buffer_overflow_messages": 0, "perf_rows": []}
    try:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            result["fatal_traceback"] |= bool(re.search(r"Traceback \(most recent call last\):|torch\.OutOfMemoryError|CUDA out of memory", line))
            capacity_error = bool(re.search(r"buffer overflow|(?:increase|exceed|insufficient).*?(?:totalAggregatePairsCapacity|foundLostPairsCapacity|foundLostAggregatePairsCapacity)", line, re.I))
            result["physx_buffer_overflow_messages"] += int(capacity_error)
            perf = re.search(r"\[PERF\] it=(\d+) rank=0 wall_end=([\d.]+) collection_s=([\d.]+) learn_s=([\d.]+) log_s=([\d.]+) gc_s=([\d.]+) callbacks_s=([\d.]+)", line)
            if perf:
                values = perf.groups()
                result["perf_rows"].append(dict(zip(("step", "wall_end", "collection_s", "learn_s", "log_s", "gc_s", "callbacks_s"),
                                                    [int(values[0])] + [float(x) for x in values[1:]])))
            marker = next((message for message in FATAL_PHYSICS_MESSAGES if message in line), None)
            if capacity_error:
                marker = "PhysX GPU capacity overflow; simulation results are not accepted"
            if marker:
                result["fatal_physics_error"] = {"message": marker, "log_line": line.rstrip("\r\n")}
                result["exit_code"] = stop_child(process)
                break
        else:
            result["exit_code"] = process.wait()
        return result
    except BaseException:
        stop_child(process)
        raise
    finally:
        process.stdout.close()


def launch(config: dict, repo: Path, output: Path, args: list[str], report: dict) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a visible CUDA GPU")
    command = [sys.executable, "-u", str(paths.RUNTIME_DIR / "train_entry.py"), *args]
    child_env = dict(os.environ)
    child_env.update(CONTACTFIRST_REPO=str(repo), CONTACTFIRST_RUN_CONFIG_FILE=str(output / "run_config.json"),
                     CONTACTFIRST_CPU_THREADS=str(config["cpus"]), OMNI_KIT_ACCEPT_EULA="Yes",
                     PYTHONPATH=os.pathsep.join([str(paths.RUNTIME_DIR), str(repo)]),
                     WANDB_MODE=config["wandb_mode"], WANDB_DIR=str(output / "wandb"))
    from local_cache import cache_root
    child_env["CONTACTFIRST_SHADER_CACHE_ROOT"] = str(cache_root(output, "train") / "cache/kit_shaders")
    (output / "launch_command.txt").write_text(shlex.join(command) + "\n")
    log_path = output / "training.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=repo, env=child_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        outcome = monitor_child(process, log)
    perf = outcome.pop("perf_rows")
    write_json(output / config["runtime_attempt"] / "perf_rows.json", perf)
    steady = perf[-100:]
    if steady:
        base = [sum(row[x] for x in ("collection_s", "learn_s", "log_s", "gc_s")) for row in steady]
        report["timing"] = {"measured_updates": len(perf), "update_seconds_median": statistics.median(base),
                            "update_seconds_mean": statistics.mean(base),
                            "projected_hours_for_target": statistics.mean(base) * config["iterations"] / 3600}
    checkpoint = output / "train/last.pt"
    step = None
    if checkpoint.is_file():
        from tools.ckpt_global_step import read_global_step
        step = read_global_step(str(checkpoint))
    report["training"] = {**outcome, "checkpoint": str(checkpoint), "global_step": step, "target": config["iterations"]}
    if outcome["fatal_physics_error"]:
        raise RuntimeError("Fatal PhysX failure stopped the training child: " + outcome["fatal_physics_error"]["message"])
    if outcome["exit_code"] or outcome["fatal_traceback"] or step is None or step < config["iterations"]:
        raise RuntimeError("Training did not reach its target; see training.log and result.json")
    final = output / "train" / f"model_step_{config['iterations']:06d}.pt"
    if final.is_file():
        report["training"]["final_checkpoint"] = {"path": str(final), "sha256": paths.sha256(final)}
    metrics = json.loads((output / "per_motion_training_metrics.json").read_text())
    expected = {m["motion_id"] for m in config["motions"]}
    if metrics["ppo_step"] != step or set(metrics["motions"]) != expected:
        raise RuntimeError("Final per-motion training metrics do not cover every motion")
    report["per_motion_training_metrics"] = {k: v["cumulative"] for k, v in metrics["motions"].items()}
    if not config["resume"] and not (output / "actor_initialization.json").is_file():
        raise RuntimeError("Missing live verification of the warm-start actor")
    if config["periodic_eval"]:
        receipts = sorted((output / "periodic_eval").glob("step_*/receipt.json"))
        report["periodic_eval"] = [json.loads(p.read_text()) for p in receipts]


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    lease = claim_output(config, output)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "output_root": str(output)}
    attempt = "runtime_attempts/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    (output / attempt).mkdir(parents=True, exist_ok=False)
    config["runtime_attempt"] = attempt
    try:
        sys.path[:0] = [str(paths.RUNTIME_DIR)]
        from local_cache import activate
        report["local_cache"] = activate(output, "train", output / attempt / "cache_diagnostics")
        repo = source_check(config, report)
        dependencies_check(report)
        for name in ("model_config", "model_checkpoint", "physics_overrides", "actor_init_checkpoint"):
            real_file(config[name], name)
        actor_initialization_check(config, report, args.allow_warmstart_mismatch)
        write_json(output / "run_config.json", config)
        if config["periodic_eval"]:
            write_eval_plan(config, output)
        view = data_check(config, report)
        hydra_args = training_overrides(config, output, view)
        compose_check(repo, hydra_args, config, output, report)
        decoder_check(config, report)
        write_json(output / "preflight.json", {**report, "status": "passed"})
        launch(config, repo, output, hydra_args, report)
        report["status"] = "passed"
        print(f"Manipulation finetune finished; receipt: {output / 'result.json'}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "result.json", report)
        lease.close()


if __name__ == "__main__":
    sys.exit(main())
