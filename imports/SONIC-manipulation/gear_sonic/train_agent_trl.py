#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fix sys.path: when running as `python gear_sonic/train_agent_trl.py`, Python adds
# gear_sonic/ to sys.path[0], causing `from trl import ...` to resolve to our local
# gear_sonic/trl/ instead of the HuggingFace trl package. Replace with repo root.
import sys
import os
import time
_PROCESS_STARTED_MONOTONIC = time.monotonic()
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for training but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    sys.exit(1)

import glob
import logging
import os
from pathlib import Path
import re
import sys

from filelock import FileLock
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml

from gear_sonic.trl.utils.common import (
    custom_instantiate,
    get_filtered_state_dict,
    materialize_lazy_params,
)
from gear_sonic.trl.utils.wandb_identity import choose_wandb_run_id
from gear_sonic.utils.common import seeding
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.obs_utils import get_group_term_obs_shape

register_rl_resolvers()


def resume_training(config):
    if config.get("checkpoint", None) is not None:
        last_existing_checkpoint = config.checkpoint
        # Warm-resume: if checkpoint and experiment_dir point to different paths,
        # load weights + optimizer state from the checkpoint but keep the
        # explicit experiment_dir (fresh wandb run, fresh output dir).
        explicit_dir = config.get("experiment_dir", None)
        if explicit_dir is not None and os.path.abspath(explicit_dir) != os.path.abspath(
            os.path.dirname(last_existing_checkpoint)
        ):
            from omegaconf import open_dict

            with open_dict(config):
                config.checkpoint = last_existing_checkpoint
                config.warm_resume = True
            print(
                f"Warm-resuming (weights + optimizer) from {last_existing_checkpoint} "
                f"into new experiment_dir {explicit_dir}"
            )
            return
    else:
        # Use experiment_dir to find the checkpoint, rather than reconstructing
        # from config.project_name which can differ from the actual filesystem path.
        experiment_dir = config.get("experiment_dir", None)
        if experiment_dir is None:
            print("No experiment_dir set for resume, starting fresh")
            return
        current_run_last = os.path.join(experiment_dir, "last.pt")
        if os.path.isfile(current_run_last):
            last_existing_checkpoint = current_run_last
        else:
            experiment_dir_base = re.sub(r"-\d{8}_\d{6}$", "", experiment_dir)
            checkpoints = sorted(glob.glob(os.path.join(f"{experiment_dir_base}-*", "last.pt")))
            if not checkpoints:
                print(f"No checkpoint found matching {experiment_dir_base}-*/last.pt, starting fresh")
                return
            last_existing_checkpoint = checkpoints[-1]
    experiment_dir = os.path.dirname(last_existing_checkpoint)
    config.experiment_dir = experiment_dir
    config.checkpoint = last_existing_checkpoint
    print(f"Resuming training from {last_existing_checkpoint}")


def resume_checkpoint(config):
    config.checkpoint = config.checkpoint


def create_manager_env(config, device, args_cli):

    # import wandb

    from isaaclab.envs import (
        ManagerBasedRLEnv,
    )

    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper

    env_instance_cfg = custom_instantiate(config.manager_env)

    # Iteratively check the difference in attribute of env_instance_cfg1 and env_instance_cfg, print out the difference
    def compare_attrs(obj1, obj2, prefix=""):
        # Only compare attributes that do not start with '__' and are not methods
        attrs1 = set(dir(obj1))
        attrs2 = set(dir(obj2))
        common_attrs = attrs1 & attrs2
        for attr in sorted(common_attrs):
            if (
                attr.startswith("__")
                or callable(getattr(obj1, attr))
                or callable(getattr(obj2, attr))
            ):
                continue
            try:
                val1 = getattr(obj1, attr)
                val2 = getattr(obj2, attr)
            except Exception:
                continue
            # Recursively compare if both are objects with __dict__ or are dicts
            if isinstance(val1, dict | DictConfig) and isinstance(val2, dict | DictConfig):
                compare_attrs(val1, val2, prefix + attr + ".")
            elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
                compare_attrs(val1, val2, prefix + attr + ".")
            else:
                if isinstance(val1, list):
                    val1 = tuple(val1)
                if isinstance(val2, list):
                    val2 = tuple(val2)
                if val1 != val2:
                    print(
                        f"\nDifference found at '{prefix}{attr}':\n"
                        f"  - env_instance_cfg1: {val1!r}\n"
                        f"  - env_instance_cfg : {val2!r}\n"
                    )

    env_instance_cfg.seed = config.seed
    env_instance_cfg.sim.device = device
    env_instance_cfg.config["headless"] = args_cli.headless
    # Optional authored-USD-stage accelerator (gear_sonic/isaaclab_cache; off unless
    # manager_env.config.authored_stage_cache.mode is set). Keyed by ordinary metadata already
    # on env_instance_cfg; miss / any cache problem before stage mutation -> native scene.
    # Installed FIRST (swaps the InteractiveScene class object), restored LAST.
    restore_scene_class = lambda: None  # noqa: E731
    try:
        from gear_sonic.isaaclab_cache import install_authored_stage_cache

        try:
            _asc_rank = int(config.get("global_rank", os.environ.get("RANK", 0)) or 0)
        except (TypeError, ValueError):
            _asc_rank = 0
        try:
            _asc_world_size = int(
                config.algo.config.get("world_size", os.environ.get("WORLD_SIZE", 1)) or 1
            )
        except (TypeError, ValueError, AttributeError):
            _asc_world_size = 1
        restore_scene_class = install_authored_stage_cache(
            env_instance_cfg.config.get("authored_stage_cache"),
            env_cfg=env_instance_cfg,
            rank=_asc_rank,
            world_size=_asc_world_size,
        )
    except Exception as exc:  # noqa: BLE001 - optional accelerator never blocks native construction
        print(f"[authored_stage] not installed: {exc!r}", flush=True)
    from gear_sonic.physx_replication import install as install_physx_replication

    restore_physx_replication = install_physx_replication(env_instance_cfg.config)
    # Observation-only startup stamps (gear_sonic/scene_timing.py; SCENE_TIMING=0 disables).
    # Any failure here prints and leaves the native construction path untouched.
    restore_scene_timing = lambda: None  # noqa: E731
    try:
        from gear_sonic.scene_timing import install as install_scene_timing

        restore_scene_timing = install_scene_timing()
    except Exception as exc:  # noqa: BLE001
        print(f"[scene_timing] not installed: {exc!r}", flush=True)
    try:
        env = ManagerBasedRLEnv(
            cfg=env_instance_cfg, render_mode="rgb_array" if not args_cli.headless else None
        )
    finally:
        try:
            restore_scene_timing()
        except Exception:  # noqa: BLE001
            pass
        restore_physx_replication()
        try:
            restore_scene_class()
        except Exception:  # noqa: BLE001
            pass

    env = ManagerEnvWrapper(env, env_instance_cfg.config)
    return env


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    simulator_type = "IsaacSim"
    env_config = config.manager_env
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    # Setup model components
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))

    if config.get("resume", False):
        resume_training(config)
    elif config.get("checkpoint", None) is not None:
        resume_checkpoint(config)

    config.algo.trl.output_dir = str(Path(config.experiment_dir))

    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)

    # Add exp_name from main config to training_args
    training_args.exp_name = config.experiment_name

    from datetime import timedelta

    from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
    import torch  # noqa: E402

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=6000))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, kwargs],
    )

    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes

        # Motion sharding: each GPU loads only its slice of motions/objects.
        # Drive MotionLibBase's rank-aware slicer from accelerate's process index
        # when the top-level cfg sets ++motion_shard_by_rank=true (used by the
        # GRAIL intermimic sweeps bp2/bp3/g21).
        if config.get("motion_shard_by_rank", False):
            from omegaconf import open_dict

            _rank = accelerator.process_index
            _ws = accelerator.num_processes
            with open_dict(config):
                config.manager_env.commands.motion.motion_lib_cfg.motion_shard_rank = _rank
                config.manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size = _ws
            print(f"[Motion Shard] Enabled: rank {_rank}/{_ws}")
    seeding(config.seed)

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    meta = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as file:
            loaded_meta = yaml.safe_load(file)
        if loaded_meta is not None and not isinstance(loaded_meta, dict):
            raise ValueError(f"Expected a mapping in {meta_path}, got {type(loaded_meta)}")
        meta = loaded_meta or {}

    full_resume = bool(
        config.get("resume", False)
        and config.get("checkpoint", None)
        and not config.get("warm_resume", False)
    )
    # explicit config id, else this run's meta.yaml (written by the first leg); the checkpoint is read once, by the
    # restore itself (PPOTrainer.load_checkpoint) - no separate identity preload (user decision 2026-10-14)
    requested_wandb_run_id = choose_wandb_run_id(config.wandb.wandb_id, meta.get("wandb_run"))
    config.wandb.wandb_id = requested_wandb_run_id
    if requested_wandb_run_id is not None:
        print(f"resume wandb from run: {requested_wandb_run_id}")

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    initialized_wandb_run_id = None
    if config.use_wandb and accelerator.is_main_process:
        local_project_name = f"{config.project_name}"
        wandb_project = config.wandb.wandb_project or local_project_name
        project_name = f"{wandb_project}"
        run_name = config.experiment_dir.replace(
            f"{config.base_dir}/{local_project_name}/", ""
        )
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        wandb_group = None if config.wandb.wandb_id is not None else config.wandb.wandb_group
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb_run = wandb.init(
            project=project_name,
            entity=config.wandb.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=unresolved_conf,
            dir=wandb_dir,
            id=config.wandb.wandb_id,
            group=wandb_group,
            resume="must" if full_resume and requested_wandb_run_id is not None else "allow",
        )
        if os.environ.get("WANDB_MODE", "").strip().lower() != "disabled":
            initialized_wandb_run_id = choose_wandb_run_id(getattr(wandb_run, "id", None))

    persistent_wandb_run_id = initialized_wandb_run_id or requested_wandb_run_id
    training_args.wandb_run_id = persistent_wandb_run_id

    # Setup simulator similar to train_agent.py

    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher  # 4.2

            # from isaaclab.app import AppLauncher # not working
            # from omni.isaac.lab.app import AppLauncher

        import argparse

        parser = argparse.ArgumentParser(description="Train an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        ######################################################### ZL: fix isaacsim 4.5 rendering #########################################################
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing  # config.env_spacing
        args_cli.output_dir = config.output_dir
        # Enable cameras if enable_cameras, render_results, render_ego, or overview_camera is True
        args_cli.enable_cameras = (
            env_config.config.get("enable_cameras", False)
            or env_config.config.get("render_results", False)
            or env_config.config.get("render_ego", False)
            or env_config.config.get("overview_camera", False)
        )
        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        # Kit boot settings: quiet logs + persistent UJITSO collision cooking through the mounted /root/.cache
        # (gear_sonic/kit_boot_args.py; PHYSX_UJITSO_COOKING=0 opts out). Printed once so the leg log shows them.
        from gear_sonic.kit_boot_args import kit_args as _kit_boot_args

        # the Kit log file lands in the experiment directory: EXPERIMENT_DIR when the stage exported it, else the
        # trainer output dir (the same directory that holds last.pt)
        _kit_env = dict(os.environ)
        _kit_env.setdefault("EXPERIMENT_DIR", str(config.output_dir))
        args_cli.kit_args = _kit_boot_args(_kit_env)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[kit_boot] kit_args={args_cli.kit_args}", flush=True)

        # AppLauncher can't handle multiple processes creating it at the same time so we need a lock
        _lock_path = "/tmp/isaaclab_app_launcher.lock"
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        with FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)

        simulation_app = app_launcher.app
        # what Kit actually resolved (store path, cooking switches, log file) - one line per rank, in the leg log
        from gear_sonic.kit_boot_args import resolved_settings_line as _kit_resolved_settings_line

        print(f"{_kit_resolved_settings_line()} rank={_local_rank}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from gear_sonic.utils.logging import HydraLoggerBridge

    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    # Setup wandb if enabled
    os.chdir(hydra.utils.get_original_cwd())

    # Save config and meta BEFORE env creation so eval jobs can postprocess
    # checkpoint configs even if training crashes during env init.
    experiment_save_dir = Path(config.experiment_dir)
    if accelerator.is_main_process:
        experiment_save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)
        meta = {"wandb_run": persistent_wandb_run_id}
        meta["max_train_steps"] = config.algo.config.num_learning_iterations
        yaml.safe_dump(meta, open(meta_path, "w"))
        print("saved meta:", meta)

    # Initialize environment
    env_config.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env_config.config.experiment_dir = str(Path(config.experiment_dir))

    env = create_manager_env(config, device, args_cli)
    if config.get("replay_batch_manifest", None):
        from gear_sonic.utils.replay_batch_v1 import run_replay_batch

        receipt = run_replay_batch(
            env,
            Path(str(config.replay_batch_manifest)),
            process_start_monotonic=_PROCESS_STARTED_MONOTONIC,
        )
        print(
            "ONE_ISAAC_APP_REPLAY_BATCH="
            + __import__("json").dumps(receipt, sort_keys=True),
            flush=True,
        )
        os._exit(0)
    if config.get("replay", False):
        _save_video_path = config.get("replay_save_video", None)
        _titles_json = config.get("replay_env_titles_json", None)
        _env_titles = None
        if _titles_json:
            import json as _json

            with open(str(_titles_json)) as _handle:
                _env_titles = _json.load(_handle)
        env.run_replay(
            start_time_step=-1,
            loop=config.get("replay_loop_num", True),
            save_video_path=_save_video_path,
            grid_spacing=config.get("replay_grid_spacing", 2.0),
            hold_completed=config.get("replay_hold_completed", False),
            per_env_video_dir=config.get("replay_per_env_video_dir", None),
            env_titles=_env_titles,
        )
        os._exit(0)
    if config.get("vplanner_replay", False):
        vplanner_checkpoint = config.get("vplanner_checkpoint", None)
        if vplanner_checkpoint is None:
            raise ValueError("vplanner_checkpoint must be specified for vplanner_replay")
        env.run_vplanner_replay(
            checkpoint_path=vplanner_checkpoint,
            max_frames=config.get("vplanner_max_frames", 500),
            replan_interval=config.get("vplanner_replan_interval", 0),
            speed=config.get("vplanner_speed", 1.0),
            loop=config.get("vplanner_loop", True),
            save_images=config.get("vplanner_save_images", False),
            output_dir=config.get("vplanner_output_dir", None),
            dof_noise=config.get("vplanner_dof_noise", 0.0),
            dof_vel_noise=config.get("vplanner_dof_vel_noise", 0.0),
            quat_noise=config.get("vplanner_quat_noise", 0.0),
        )
        os._exit(0)

    ref_model = None
    value_model = None
    disc_model = None
    # import ipdb; ipdb.set_trace()

    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy_backbone_kwargs = {}
        critic_backbone_kwargs = {}
        env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
        env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
            "policy"
        ].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
            "critic"
        ].shape[-1]
        example_obs = env.reset(flatten_dict_obs=False)
        for key in env.env.observation_space:
            if key not in ["policy", "critic"]:
                group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(
                    example_obs, key
                )
                env.config["obs"]["group_obs_dims"][key] = group_obs_dims
                env.config["obs"]["group_obs_names"][key] = group_obs_names
                env.config["obs"]["obs_dims"][key] = group_obs_total_dim
                env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim
        for key, value in example_obs.items():
            if key in env.config["robot"]["algo_obs_dim_dict"]:
                continue
            if key in {"policy", "critic"}:
                continue
            obs_shape = tuple(value.shape[1:])
            env.config["obs"]["obs_dims"][key] = int(np.prod(obs_shape).item())
            env.config["obs"]["group_obs_dims"][key] = {key: obs_shape}
            env.config["obs"]["group_obs_names"][key] = [key]
            env.config["robot"]["algo_obs_dim_dict"][key] = int(np.prod(obs_shape).item())
        if config.manager_env.config.get("meta_action_dim", None) is not None:
            env.config["robot"]["actions_dim"] = config.manager_env.config.meta_action_dim
        else:
            env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

        policy = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=policy_backbone_kwargs,
            _resolve=False,
        ).to(device)

        if getattr(config.algo.config, "use_dagger", False):
            # Get teacher input key from config or default to "teacher"
            teacher_input_key = config.algo.config.get("teacher_input_key", "teacher")
            ref_model = custom_instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
                input_key=teacher_input_key,
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False):
            value_model = custom_instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                backbone_kwargs=critic_backbone_kwargs,
                _resolve=False,
            ).to(device)
        if config.algo.config.get("use_amp", False):
            disc_model = custom_instantiate(
                config.algo.config.disc,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
            ).to(device)
    else:
        raise ValueError("No longer supported")

    materialize_lazy_params(policy, env)
    if hasattr(policy, "attach_sonic"):
        # SONIC decoder LoRA arm: adapters must exist before the optimizer/checkpoint load.
        policy.attach_sonic(env)

    if config.algo.config.get("pretrained_model", None) is not None:
        pretrained_cfg = config.algo.config.pretrained_model
        sd_key = pretrained_cfg.get("state_dict_key", "state_dict")
        strict = pretrained_cfg.get("strict", True)
        state_dict = torch.load(pretrained_cfg.path, map_location=device, weights_only=False)[
            sd_key
        ]
        for (
            module_name,
            state_dict_key,
        ) in pretrained_cfg.module_mapping.items():
            module = eval(module_name)
            filtered_state_dict = get_filtered_state_dict(state_dict, state_dict_key)
            missing, unexpected = module.load_state_dict(filtered_state_dict, strict=strict)
            if missing:
                logger.info(f"Pretrained loading '{module_name}': missing keys: {missing}")
            if unexpected:
                logger.info(f"Pretrained loading '{module_name}': unexpected keys: {unexpected}")

    accelerator.wait_for_everyone()

    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    ################
    # Training
    ################
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        disc_model=disc_model,
        value_model=value_model,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        resume=config.get("resume", False),
        warm_resume=config.get("warm_resume", False),
        flexible_load=config.get("flexible_load", False),
        local_seed=config.seed,
        log_dir=experiment_save_dir,
        accelerator=accelerator,
        _resolve=False,
    )

    # Training loop
    trainer.train()

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":

    main()
