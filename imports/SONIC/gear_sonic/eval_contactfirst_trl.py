#!/usr/bin/env python3
"""Deterministic frozen-checkpoint manipulation evaluation, without a PPO trainer.

One environment, seed 0, policy mean actions (``act_inference``), start at reference frame
0, no optimizer, zero PPO updates and no reset during the reference. Writes ``eval.mp4``,
``frames.json``, ``evaluation.json`` (object metrics, reference coverage, end reason) and
``evaluation_config.yaml``. Run from ``imports/SONIC`` in the ``sonic`` environment::

    python gear_sonic/eval_contactfirst_trl.py --pack <pack> --motion <key> \
        --checkpoint <model_step_*.pt> --output <new dir> [--no-video]

``--no-video`` simulates the same episode without cameras (metrics only) for hosts
without RTX rendering. The flag form builds a request file, runs the inference in a
child process of this same script (``eval_contactfirst_trl.py /absolute/request.json``)
and independently verifies the persisted evidence
(``--verify-completion REQUEST.json EXIT_CODE``), exactly like the periodic
evaluation during training.
"""
from __future__ import annotations

# Fix sys.path: when running as `python gear_sonic/eval_contactfirst_trl.py`, Python adds
# gear_sonic/ to sys.path[0], causing `from trl import ...` to resolve to our local
# gear_sonic/trl/ instead of the HuggingFace trl package. Replace with repo root.
import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_repo_root = _os.path.dirname(_script_dir)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


# Upstream revision of the vendored manipulation (ContactFirst) extensions.
UPSTREAM_SONIC_COMMIT = "fb8563f9a5b5889d50bccba4dc07a607f2e4a71c"


class OutputOwnershipError(RuntimeError):
    """An earlier attempt owns this output; do not add or overwrite evidence."""


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def persist_receipt(output, receipt):
    """Commit runtime evidence before Kit shutdown can terminate this process."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = "evaluation.json" if receipt["status"] == "passed" else "evaluation_failure.json"
    target = output / name
    if target.exists():
        raise FileExistsError(f"Never overwrite evaluation evidence: {target}")
    with tempfile.NamedTemporaryFile(mode="w", dir=output, prefix=".receipt-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def persist_failure(output, exc):
    if not (Path(output) / "evaluation_failure.json").exists():
        persist_receipt(output, {"status": "failed", "error": repr(exc),
                                "traceback": traceback.format_exc()})


def verify_completion(request, worker_exit_code):
    """Independent stdlib-only process rejects silent exit0 or partial evidence."""
    if worker_exit_code != 0:
        raise RuntimeError(f"Inference process failed with exit code {worker_exit_code}")
    output = Path(request["output_root"])
    if (output / "evaluation_failure.json").exists():
        raise RuntimeError("Inference recorded a failure before shutdown")
    result = json.loads((output / "evaluation.json").read_text())
    checkpoint, video, rollout = result["checkpoint"], result["video"], result["rollout"]
    required = (result.get("status") == "passed",
                result.get("source_commit") == request["source_commit"],
                result.get("case_id") == request["case_id"],
                result.get("parent_wandb_run_id") == request["parent_wandb_run_id"],
                result.get("source_train_config", {}).get("sha256") == request["source_train_config_sha256"],
                checkpoint.get("sha256") == request["checkpoint_sha256"],
                checkpoint.get("step") == request["checkpoint_step"],
                checkpoint.get("actor_exact_match") is True,
                result.get("actor_unchanged_after_inference") is True,
                result.get("optimizer_created") is False,
                result.get("ppo_updates_during_evaluation") == 0,
                rollout.get("start_frame") == 0,
                rollout.get("deterministic") is True,
                rollout.get("reset_count") == 0,
                (video.get("path") == "eval.mp4") if request.get("render_video", True) else video.get("path") is None,
                *[result.get(k) == request.get(k) for k in ("motion_id", "source_motion_key", "object_mesh_sha256", "roster_manifest_sha256")])
    if not all(required):
        raise ValueError("Persisted evaluation receipt does not match its exact request")
    if request.get("render_video", True):
        path = output / "eval.mp4"
        if path.stat().st_size != video["bytes"] or sha256(path) != video["sha256"]:
            raise ValueError("Persisted inference video size/hash mismatch")
    print("EVALUATION_EXIT_VERIFIED=" + json.dumps({"status": "passed", "case_id": request["case_id"],
          "checkpoint_step": checkpoint["step"], "video_sha256": video.get("sha256")}), flush=True)


def checkpoint_step(checkpoint):
    state = checkpoint.get("state")
    value = state.get("global_step") if isinstance(state, dict) else getattr(state, "global_step", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Full trained checkpoint must contain a positive global_step")
    return value


def reference_clock(request, horizon, control_dt):
    """Describe the unchanged source-to-runtime clock without promising episode success."""
    frames, fps = request.get('source_reference_frames'), request.get('source_reference_fps')
    if frames is None or fps is None:
        raise ValueError('Source frame/FPS provenance is required for full-reference inference')
    if type(frames) is not int or frames < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source reference clock')
    span = (frames - 1) / fps
    expected = math.ceil(span / control_dt - 1e-9)
    if not math.isclose(control_dt, 0.02, abs_tol=1e-8) or horizon != expected:
        raise ValueError('Runtime horizon does not cover the unchanged source reference clock')
    return {'source_frames': frames, 'source_fps_hz': fps,
            'source_endpoint_span_s': span, 'runtime_reference_steps': horizon,
            'runtime_fps_hz': 1.0 / control_dt, 'runtime_last_target_time_s': (horizon - 1) * control_dt,
            'resampling': 'Pinned native arange(0,(source_frames-1)/source_fps,1/50); endpoint excluded',
            'full_rollout_physics_duration_s': horizon * control_dt,
            'full_rollout_video_frame_count_with_initial': horizon + 1,
            'source_frames_are_not_output_video_frames': True}


def verify_source(repo, expected):
    """Record the vendored SONIC tree and the GRAIL revision that ran (provenance, not a pin)."""
    marker = repo / "gear_sonic/config/exp/manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_ablation_d.yaml"
    if not marker.is_file() or not (repo / "gear_sonic/train_agent_trl.py").is_file():
        raise ValueError(f"Not a vendored SONIC manipulation tree: {repo}")
    provenance = {"commit": expected, "method": "vendored_tree", "vendored_tree": str(repo)}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, timeout=30)
        status = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=repo, text=True, capture_output=True, timeout=60)
        if head.returncode == 0:
            provenance["grail_git_head"] = head.stdout.strip()
            provenance["vendored_tree_modified"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return provenance


def verify_actor(policy, source):
    import torch
    actual = policy.state_dict()
    if set(actual) != set(source):
        raise ValueError("Policy checkpoint key set differs from instantiated actor")
    for key, value in source.items():
        loaded = actual[key].detach().cpu()
        if not isinstance(value, torch.Tensor) or loaded.dtype != value.dtype or loaded.shape != value.shape:
            raise ValueError(f"Actor tensor contract differs: {key}")
        if not bool(torch.isfinite(value).all()) or not torch.equal(loaded, value.cpu()):
            raise ValueError(f"Actor tensor not exact: {key}")
    return {"actor_exact_match": True, "tensor_count": len(actual),
            "parameter_count": sum(value.numel() for value in actual.values())}


def prepare_config(config, request, output):
    """Preserve task physics and observations; isolate only evaluation controls."""
    from omegaconf import OmegaConf, open_dict
    with open_dict(config):
        config.num_envs = 1
        if request.get("physics_overrides"):
            config.manager_env.config.object_physics_overrides_path = request["physics_overrides"]
        config.seed = int(request.get("seed", 0))
        config.headless = True
        config.multi_gpu = False
        config.global_rank = 0
        config.use_wandb = False
        config.resume = False
        config.warm_resume = False
        config.checkpoint = str(request["checkpoint"])
        config.experiment_dir = str(output)
        config.output_dir = str(output)
        config.base_dir = str(output)
        config.callbacks = {}
        config.algo.config.global_rank = 0
        config.algo.config.world_size = 1
        env = config.manager_env
        env.config.num_envs = 1
        env.config.experiment_dir = str(output)
        env.config.save_rendering_dir = str(output / "raw_video")
        render = bool(request.get("render_video", True))
        env.config.render_results = render
        env.config.max_render_envs = 1
        env.config.render_env_ids = [0]
        env.config.render_width = 960
        env.config.render_height = 720
        env.config.render_frame_skip = 1
        env.config.render_skip_first_frame = False
        env.config.render_terminal_overlay = False
        env.config.render_text_color = [255, 255, 255]
        env.config.render_text_outline_color = [0, 0, 0]
        env.config.render_info = [[]]
        env.config.fix_camera_after_first_frame = True
        env.config.eval_camera_offset = [2.0, 2.0, 1.2]
        env.config.safe_nan = False  # Invalid state is a technical failure, never a hidden reset.
        env.recorders = {"_target_": "gear_sonic.envs.manager_env.mdp.episode_recorder.make_recorders",
                         "output_root": str(output), "render": render}
        motion = env.commands.motion
        motion.start_from_first_frame = True
        motion.sample_from_n_initial_frames = None
        motion.sample_before_contact = False
        motion.sample_start_truncate_time_s = None
        motion.object_position_randomize = False
        motion.motion_lib_cfg.motion_shard_rank = 0
        motion.motion_lib_cfg.motion_shard_world_size = 1
        motion.motion_lib_cfg.joint_source_tracking = False
        motion.motion_lib_cfg.adaptive_sampling = None
        motion.motion_lib_cfg.joint_source_manifest = request.get("roster_manifest") or ""
        motion.motion_lib_cfg.joint_roster_direct_dirs = request.get("direct_dirs") or ""
        for key in list(env.events.keys()):
            event = env.events[key]
            if key in list(env.config.get("train_only_events", [])):
                env.events[key] = None
            elif OmegaConf.is_dict(event) and str(event.get("mode", "")) == "interval":
                env.events[key] = None
        for key in ("policy", "tokenizer", "policy_atm"):
            if env.observations.get(key) is not None:
                env.observations[key].enable_corruption = False
        # One reference episode, not the optional training rollout time cap.
        if env.terminations.get("time_out") is not None:
            env.terminations.time_out.params.cap_by_episode_length = False
        # Never write into the live training cache/output from evaluation.
        for section in (env.config.get("authored_stage_cache"),
                        motion.motion_lib_cfg.get("motion_tensor_cache")):
            if section is not None:
                section.mode = "disabled"
        motion.motion_lib_cfg.joint_roster_scene_plan_cache_mode = "disabled"
    return config


def make_actor(config, env, device):
    import numpy as np
    from gear_sonic.trl.utils.common import custom_instantiate, materialize_lazy_params
    from gear_sonic.utils.obs_utils import get_group_term_obs_shape
    if not config.algo.config.get("use_new_actor_critic", False):
        raise ValueError("Pinned actor architecture is required")
    obs = env.reset(flatten_dict_obs=False)
    for name, group in (("actor_obs", "policy"), ("critic_obs", "critic")):
        size = env.env.observation_space[group].shape[-1]
        env.config.obs.obs_dims[name] = size
        env.config.robot.algo_obs_dim_dict[name] = size
    for key in env.env.observation_space:
        if key not in {"policy", "critic"}:
            dims, names, total = get_group_term_obs_shape(obs, key)
            env.config.obs.group_obs_dims[key] = dims
            env.config.obs.group_obs_names[key] = names
            env.config.obs.obs_dims[key] = total
            env.config.robot.algo_obs_dim_dict[key] = total
    for key, value in obs.items():
        if key not in env.config.robot.algo_obs_dim_dict and key not in {"policy", "critic"}:
            shape = tuple(value.shape[1:])
            total = int(np.prod(shape).item())
            env.config.obs.obs_dims[key] = total
            env.config.obs.group_obs_dims[key] = {key: shape}
            env.config.obs.group_obs_names[key] = [key]
            env.config.robot.algo_obs_dim_dict[key] = total
    env.config.robot.actions_dim = config.manager_env.config.get("meta_action_dim") or env.env.action_space.shape[-1]
    policy = custom_instantiate(
        config.algo.config.actor, env_config=env.config, algo_config=config.algo.config,
        module_dim_dict=config.algo.config.get("module_dim", {}), backbone_kwargs={}, _resolve=False,
    ).to(device)
    materialize_lazy_params(policy, env)
    return policy


def run(request):
    evaluation_started = time.perf_counter()
    output = Path(request["output_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in
           ("evaluation.json", "evaluation_failure.json", "raw_video", "frames.json", "eval.mp4")):
        raise OutputOwnershipError("Evaluation output already exists; never overwrite an attempt")
    repo = Path(request["remote_repo"]).resolve(strict=True)
    if request.get("num_envs", 1) != 1 or request.get("deterministic", True) is not True or request.get("start_from_first_frame", True) is not True:
        raise ValueError("Only deterministic single-environment frame-zero evaluation is supported")
    checkpoint_path = Path(request["checkpoint"]).resolve(strict=True)
    source_config = Path(request["source_train_config"]).resolve(strict=True)
    if sha256(checkpoint_path) != request["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash mismatch")
    if sha256(source_config) != request["source_train_config_sha256"]:
        raise ValueError("Saved training config hash mismatch")
    source_provenance = verify_source(repo, request["source_commit"])
    actual_commit = source_provenance["commit"]
    if request.get("roster_manifest") and sha256(request["roster_manifest"]) != request["roster_manifest_sha256"]:
        raise ValueError("Single-environment roster hash mismatch")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["JOINT_ROSTER_MANIFEST"] = request.get("roster_manifest") or ""
    os.environ["JOINT_ROSTER_DIRECT_DIRS"] = request.get("direct_dirs") or ""
    if bool(os.environ["JOINT_ROSTER_MANIFEST"]) == bool(os.environ["JOINT_ROSTER_DIRECT_DIRS"]):
        raise ValueError("Exactly one single-environment roster input is required")
    from gear_sonic.utils.contactfirst_cache import activate
    cache_plan = activate(request["parent_train_output_root"], "eval", output / "cache_diagnostics")
    with (output / "cache_plan.json").open("x") as stream:
        json.dump(cache_plan, stream, indent=2)
        stream.write("\n")
    if request.get("shader_cache_seed"):
        os.environ["CONTACTFIRST_SHADER_CACHE_SEED"] = request["shader_cache_seed"]
    os.environ["EXPERIMENT_DIR"] = str(output)
    sys.path[:0] = [str(repo)]
    os.chdir(repo)
    from gear_sonic.utils.contactfirst_runtime import install_thread_limit, defer_fk_float32_until_import
    from gear_sonic.utils.contactfirst_shader_cache import prepare_from_environment
    install_thread_limit(int(request.get("cpus", 2)), prepare_from_environment())
    defer_fk_float32_until_import()
    from gear_sonic.utils.config_utils import register_rl_resolvers
    register_rl_resolvers()
    from omegaconf import OmegaConf
    config = prepare_config(OmegaConf.load(source_config), request, output)
    # Resolve and preserve the exact rank-local data view with one environment.
    from gear_sonic.rp2.joint_roster import resolve_rank_local_roster
    from gear_sonic.rp2.direct_directory_roster import resolve_direct_directory_roster
    kwargs = dict(num_envs=1, rank=0, world_size=1, runtime_target_fps_hz=50.0)
    if request.get("roster_manifest"):
        view = resolve_rank_local_roster(request["roster_manifest"], **kwargs)
    else:
        view = resolve_direct_directory_roster(request["direct_dirs"], **kwargs)
    for key, source in (("motion_file", "motion_dir"), ("object_motion_file", "object_motion_dir"), ("bps_dir", "bps_dir")):
        config.manager_env.commands.motion.motion_lib_cfg[key] = view[source]
    OmegaConf.save(config, output / "evaluation_config.yaml")
    import torch
    from gear_sonic.utils.common import seeding
    seeding(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Physical policy evaluation requires CUDA")
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless, args.enable_cameras, args.device = True, bool(request.get("render_video", True)), "cuda:0"
    args.num_envs, args.seed, args.env_spacing = 1, config.seed, config.manager_env.config.env_spacing
    args.output_dir, args.multi_gpu, args.distributed = str(output), False, False
    from gear_sonic.kit_boot_args import kit_args
    args.kit_args = kit_args()
    sys.argv = [sys.argv[0]]  # The request JSON is not a Kit positional argument.
    app = AppLauncher(args).app
    env = None
    try:
        from gear_sonic.train_agent_trl import create_manager_env
        env = create_manager_env(config, "cuda:0", args)
        if env.num_envs != 1:
            raise RuntimeError("Instantiated environment count differs from one")
        env.set_is_evaluating(True, global_rank=0)
        policy = make_actor(config, env, "cuda:0")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        step = checkpoint_step(checkpoint)
        if step != request["checkpoint_step"]:
            raise ValueError("Checkpoint PPO step mismatch")
        source = checkpoint.get("policy_state_dict")
        if not isinstance(source, dict) or "optimizer_state_dict" not in checkpoint:
            raise ValueError("Expected a full trained policy checkpoint")
        policy.load_state_dict(source, strict=True)
        actor = verify_actor(policy, source)
        policy.requires_grad_(False)
        policy.eval()
        if hasattr(policy, "eval_mode"):
            policy.eval_mode()
        policy.init_rollout()
        obs = env.reset()
        recorder = env.env.recorder_manager._terms["render_envs"]
        command = env.motion_command
        if request.get("motion_id"):
            selected = int(command.motion_lib._curr_motion_ids[command.motion_ids[0]].item())
            actual_motion_key = str(command.motion_lib._motion_data_keys[selected])
            if actual_motion_key != request["motion_id"]:
                raise ValueError("Child sampled another motion: " + actual_motion_key)
        if int(command.motion_start_time_steps[0]) != 0 or int(command.time_steps[0]) != 0:
            raise RuntimeError("Native evaluation reset did not produce reference frame zero")
        horizon = int(command.motion_lib.get_time_step_total(command.motion_ids)[0])
        if horizon < 2 or horizon > 100000:
            raise ValueError("Invalid reference horizon")
        clock = reference_clock(request, horizon, float(env.env.step_dt))
        from gear_sonic.envs.manager_env.mdp.episode_recorder import EpisodeComplete
        rollout_started = time.perf_counter()
        recorder.begin()
        try:
            with torch.inference_mode():
                for _ in range(horizon + 2):
                    actions = policy.act_inference(
                        obs_dict=obs, cur_dones=torch.zeros(1, dtype=torch.bool, device="cuda:0"),
                        skip_episode_attnmask=True,
                    )
                    if not bool(torch.isfinite(actions).all()):
                        raise RuntimeError("Nonfinite policy action")
                    obs, _, dones, _ = env.step({"actions": actions, "obs_dict": obs})
                    if bool(dones.any()):
                        raise RuntimeError("Terminal transition returned after recorder should have stopped")
                raise RuntimeError("Reference horizon exhausted without controlled episode end")
        except EpisodeComplete:
            pass
        rollout, metrics = recorder.finalize()
        rollout_finished = time.perf_counter()
        clock.update(recorded_physics_duration_s=rollout['physics_steps'] * rollout['control_dt_s'],
                     full_runtime_reference_recorded=rollout['reference_completed'],
                     recorded_runtime_fraction=rollout['physics_steps'] / horizon,
                     actual_end_reason=rollout['end_reason'])
        rollout['reference_clock'] = clock
        verify_actor(policy, source)
        if sha256(checkpoint_path) != request["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint snapshot changed during inference")
        if request.get("render_video", True):
            raw_video = output / "raw_video/000000.mp4"
            video = output / "eval.mp4"
            raw_video.replace(video)
            if video.stat().st_size < 1024:
                raise RuntimeError("Evaluation video is missing or empty")
            # Decode every frame; codec failures must not count as a delivered video.
            import cv2
            capture = cv2.VideoCapture(str(video))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise RuntimeError("Empty decoded video frame")
                count += 1
            capture.release()
            if count != rollout["video_frame_count"] or abs(fps - 1.0 / rollout["control_dt_s"]) > 0.1:
                raise RuntimeError("Video frame count or frame rate does not match inference")
            video_record = {"path": "eval.mp4", "sha256": sha256(video), "bytes": video.stat().st_size, "fps": fps, "frame_count": count}
        else:
            video_record = {"path": None, "reason": "metrics-only evaluation (render_video=false)"}
        receipt = {
            "status": "passed", "status_definition": "technical inference and video verification; not a policy success claim",
            "kind": "deterministic_frozen_checkpoint_policy_inference",
            "phase": request.get("phase", "current"),
            "observed_at": datetime.now(timezone.utc).isoformat(),
            **{key: request.get(key) for key in ("case_id", "parent_wandb_run_id", "parent_train_run_name", "parent_train_output_root", "motion_id", "source_motion_key", "object_mesh_sha256", "roster_manifest_sha256")},
            "source_commit": actual_commit,
            "source_provenance": source_provenance,
            "source_train_config": {"path": str(source_config), "sha256": sha256(source_config)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path), "step": step, **actor},
            "actor_unchanged_after_inference": True, "optimizer_created": False,
            "ppo_updates_during_evaluation": 0,
            "rollout": rollout, "object_metrics": metrics,
            "video": video_record,
            "events_path": "frames.json",
            "evaluation_scope": "one deterministic episode on the training reference, not a held-out benchmark",
            "timing": {"initialization_seconds": rollout_started - evaluation_started,
                       "rollout_wall_seconds": rollout_finished - rollout_started,
                       "elapsed_before_shutdown_seconds": time.perf_counter() - evaluation_started,
                       "clock": "perf_counter wall time; excludes simulator shutdown",
                       "simulated_seconds_are_not_wall_time": True},
        }
        persist_receipt(output, receipt)
        print("EVALUATION_COMPLETE=" + json.dumps(receipt, sort_keys=True), flush=True)
        return receipt
    except BaseException as exc:
        persist_failure(output, exc)
        raise
    finally:
        if env is not None:
            env.env.close()
        app.close()


def find_motion(pack, motion, group_id):
    catalog = json.loads((Path(pack) / "tracking_pack.json").read_text())
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


def run_cli(argv):
    """Flag form: build the request, run the inference child and verify its evidence."""
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--motion", required=True, help="Motion key or motion id from tracking_pack.json")
    p.add_argument("--group", default=None)
    p.add_argument("--checkpoint", type=Path, required=True, help="Full trained checkpoint (model_step_*.pt)")
    p.add_argument("--train-config", type=Path, default=None,
                   help="Saved training config (default: <checkpoint dir>/config.yaml)")
    p.add_argument("--output", type=Path, required=True, help="New evaluation directory")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--no-video", action="store_true",
                   help="Simulate the same episode without cameras (metrics only)")
    p.add_argument("--decoder-config", type=Path, default=repo / "models/sonic_manipulation_base/model_config.yaml")
    p.add_argument("--decoder-checkpoint", type=Path, default=repo / "models/sonic_manipulation_base/last.pt")
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
    request = dict(remote_repo=str(repo), source_commit=UPSTREAM_SONIC_COMMIT,
                   case_id=motion["source_motion_key"], cpus=args.cpus, phase="final_eval",
                   checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), checkpoint_step=step,
                   source_train_config=str(source), source_train_config_sha256=sha256(source),
                   output_root=str(output / "output"), num_envs=1, deterministic=True, start_from_first_frame=True,
                   seed=0, parent_wandb_run_id=None, parent_train_run_name=None, parent_train_output_root=str(output),
                   roster_manifest=motion["single_env_roster_path"],
                   roster_manifest_sha256=sha256(motion["single_env_roster_path"]), direct_dirs=None,
                   motion_id=motion["motion_id"], source_motion_key=motion["source_motion_key"],
                   object_mesh_sha256=motion["object_mesh_sha256"],
                   source_reference_frames=motion["source_reference"]["frames"],
                   source_reference_fps=motion["source_reference"]["fps"],
                   physics_overrides=motion["physics_path"], shader_cache_seed=None, render_video=not args.no_video)
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    child = Path(__file__).resolve()
    env = dict(os.environ, WANDB_MODE="disabled", OMNI_KIT_ACCEPT_EULA="Yes", HYDRA_FULL_ERROR="1",
               PYTHONPATH=str(repo), OMP_NUM_THREADS=str(args.cpus))
    with (output / "eval.log").open("w") as stream:
        result = subprocess.run([sys.executable, "-u", str(child), str(request_path)], cwd=repo,
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    with (output / "verify.log").open("w") as stream:
        verified = subprocess.run([sys.executable, str(child), "--verify-completion", str(request_path), str(result.returncode)],
                                  cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
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


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--verify-completion":
        request = json.loads(Path(sys.argv[2]).read_text())
        verify_completion(request, int(sys.argv[3]))
        return 0
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        request = json.loads(Path(sys.argv[1]).read_text())
        output = Path(request["output_root"])
        try:
            run(request)
        except OutputOwnershipError:
            raise
        except Exception as exc:
            persist_failure(output, exc)
            raise
        return 0
    return run_cli(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
