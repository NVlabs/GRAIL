"""Trainer callbacks for the ContactFirst manipulation finetune (instantiated by Hydra).

``ContactFirstCallback``
    Verifies before the first rollout that the live actor equals the actor-only warm-start
    checkpoint tensor by tensor, with a fresh optimizer; flushes per-motion training metrics
    (``per_motion_training_metrics.json`` / ``.jsonl``) after every PPO update; mirrors them to
    W&B when a run is active and finishes that run at the final update (the vendored trainer
    stops at ``max_steps`` without calling ``on_train_end``).

``PeriodicEvalCallback``
    Every ``frequency`` PPO updates (production: 1000), after the same-step checkpoint has been
    saved, pauses training and evaluates that checkpoint on every motion of the cohort, one
    deterministic single-environment child process per motion
    (``gear_sonic/eval_contactfirst_trl.py``). The parent
    PPO/optimizer/RNG/environment state is checked unchanged around the children. Evaluation
    failures are recorded in ``eval_failures.jsonl`` and never stop training.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from transformers import TrainerCallback

CHILD_TIMEOUT_SECONDS = 900
MAX_EVAL_ATTEMPTS = 2


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _wandb_run():
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run


def verify_actor_initialization(checkpoint_path, state, model, optimizer, accelerator):
    """Verify the actual, already-loaded actor without changing training state."""
    import torch

    if state.global_step != 0:
        raise RuntimeError("Actor initialization verification requires a fresh step-zero run")
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    with checkpoint_path.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    permitted = {"policy_state_dict", "state", "initialization_metadata"}
    if not isinstance(checkpoint, dict) or "policy_state_dict" not in checkpoint or set(checkpoint) - permitted:
        raise RuntimeError("Live actor verification requires an actor-only initialization checkpoint")
    source_state = checkpoint.get("state")
    source_step = source_state.get("global_step") if isinstance(source_state, dict) else getattr(source_state, "global_step", None)
    if source_step != 0:
        raise RuntimeError("Actor-only initialization checkpoint must declare global_step=0")
    policy = accelerator.unwrap_model(model).policy
    expected, actual = checkpoint["policy_state_dict"], policy.state_dict()
    if not isinstance(expected, dict) or set(expected) != set(actual):
        raise RuntimeError("Loaded actor keys do not exactly match the initialization checkpoint")
    for key, source in expected.items():
        loaded = actual[key].detach().cpu()
        if not isinstance(source, torch.Tensor) or source.shape != loaded.shape or source.dtype != loaded.dtype:
            raise RuntimeError(f"Loaded actor tensor contract differs from initialization: {key}")
        if not torch.isfinite(source).all() or not torch.equal(source, loaded):
            raise RuntimeError(f"Loaded actor tensor differs from initialization: {key}")
    if optimizer.state_dict()["state"]:
        raise RuntimeError("Actor initialization requires fresh optimizer state before the first PPO update")
    actor_ids = {id(parameter) for parameter in policy.parameters()}
    groups = [{"lr": float(group["lr"]), "parameter_count": sum(p.numel() for p in group["params"]),
               "actor_parameter_count": sum(p.numel() for p in group["params"] if id(p) in actor_ids)}
              for group in optimizer.param_groups]
    return {"status": "verified_before_first_rollout", "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256, "global_step": int(state.global_step),
            "exact_tensor_equality": True, "tensor_count": len(expected),
            "actor_parameter_count": sum(p.numel() for p in policy.parameters()),
            "optimizer_state_entries": 0, "optimizer_param_groups": groups,
            "critic_initialization": "fresh_by_actor_only_checkpoint_contract",
            "initialization_metadata": checkpoint.get("initialization_metadata", {})}


class ContactFirstCallback(TrainerCallback):
    def __init__(self, output_root, actor_init_checkpoint=None, verify_actor_initialization=False,
                 run_name=None, tags=None):
        self.output = Path(output_root)
        self.actor_init_checkpoint = actor_init_checkpoint
        self.verify_actor_initialization = bool(verify_actor_initialization)
        if self.verify_actor_initialization and not actor_init_checkpoint:
            raise ValueError("Actor initialization verification requires actor_init_checkpoint")
        self.run_name, self.tags = run_name, list(tags or [])
        self.group_recorder = None
        self.finished = False

    def on_train_begin(self, args, state, control, **kwargs):
        initialization = None
        if self.verify_actor_initialization:
            initialization = verify_actor_initialization(self.actor_init_checkpoint, state, kwargs["model"],
                                                         kwargs["optimizer"], kwargs["accelerator"])
        if not state.is_world_process_zero:
            return control
        if initialization is not None:
            _write_json(self.output / "actor_initialization.json", initialization)
            print("[actor_initialization] verified exact checkpoint equality before first rollout: "
                  f"{initialization['tensor_count']} tensors, sha256={initialization['checkpoint_sha256']}, "
                  f"fresh optimizer, lr_groups={[g['lr'] for g in initialization['optimizer_param_groups']]}", flush=True)
        self.group_recorder = kwargs["env"].env.recorder_manager._terms["group_metrics"]
        run = _wandb_run()
        if run is not None:
            import wandb
            if self.run_name:
                run.name = self.run_name
            if self.tags:
                run.tags = tuple(self.tags)
            if initialization is not None:
                run.config.update({"actor_initialization": initialization})
                run.summary["initialization/actor_exact_match"] = True
            wandb.define_metric("tracking_train/ppo_step")
            wandb.define_metric("tracking_train/*", step_metric="tracking_train/ppo_step")
            _write_json(self.output / "wandb_run.json", {"id": run.id, "name": run.name, "project": run.project,
                                                          "url": run.url, "status": "running"})
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self.finished:
            return control
        step = int(state.global_step)
        if self.group_recorder is not None:
            grouped = self.group_recorder.metrics.flush(step)
            _write_json(self.output / "per_motion_training_metrics.json", grouped)
            with (self.output / "per_motion_training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(grouped, allow_nan=False) + "\n")
            run = _wandb_run()
            if run is not None:
                import wandb
                payload = {"tracking_train/ppo_step": step}
                for motion_id, row in grouped["motions"].items():
                    prefix = "tracking_train/motions/" + motion_id + "/"
                    for scope in ("latest_window", "cumulative"):
                        payload.update({prefix + scope + "/" + k: v for k, v in row[scope].items() if v is not None})
                wandb.log(payload, commit=False)
        if step >= int(state.max_steps) or control.should_training_stop:
            self._finish(step)
        return control

    def _finish(self, step):
        if self.finished:
            return
        self.finished = True
        run = _wandb_run()
        if run is None:
            return
        import wandb
        # The vendored PPO loop returns at max_steps without on_train_end or wandb.finish;
        # flush here before train_agent_trl's deliberate os._exit(0).
        wandb.log({}, commit=True)
        wandb.finish()
        path = self.output / "wandb_run.json"
        if path.is_file():
            receipt = _read(path)
            receipt.update(status="finished", final_step=step)
            _write_json(path, receipt)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._finish(int(state.global_step))
        return control


def parent_guard(state, model, optimizer, scheduler, env):
    """Fingerprint PPO/optimizer/RNG/environment state without mutating it."""
    import numpy as np
    import torch

    def versions(value):
        if isinstance(value, torch.Tensor):
            return (tuple(value.shape), str(value.dtype), value.data_ptr(), value._version)
        if isinstance(value, dict):
            return {k: versions(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [versions(v) for v in value]
        return copy.deepcopy(value)

    numpy_state = np.random.get_state()
    return {
        "global_step": int(state.global_step), "model_training": bool(model.training),
        "policy_training": bool(model.policy.training), "env_is_evaluating": bool(env.is_evaluating),
        "env_common_step_counter": int(env.env.common_step_counter) if hasattr(env.env, "common_step_counter") else None,
        "env_sim_step_counter": int(env.env._sim_step_counter) if hasattr(env.env, "_sim_step_counter") else None,
        "policy_versions": versions(model.policy.state_dict()),
        "value_versions": versions(model.value_model.state_dict()) if getattr(model, "value_model", None) is not None else None,
        "optimizer_versions": versions(optimizer.state_dict()), "scheduler": versions(scheduler.state_dict()),
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], hashlib.sha256(numpy_state[1].tobytes()).hexdigest(), *numpy_state[2:]),
        "torch_rng_sha256": hashlib.sha256(torch.random.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "cuda_rng_sha256": [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()],
    }


def child_environment(output):
    """Evaluation children get W&B disabled and their own task-local eval cache."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_")}
    env.update(WANDB_MODE="disabled", PYTHONUNBUFFERED="1", OMNI_KIT_ACCEPT_EULA="Yes")
    from gear_sonic.utils.contactfirst_cache import configure
    owner = env.get("CONTACTFIRST_CACHE_OWNER")
    if not owner:
        raise ValueError("Evaluation child requires the parent cache owner")
    env, _ = configure(owner, "eval", environ=env, receipt_dir=Path(output) / "cache_diagnostics")
    return env


def summarize_evaluation(output_root):
    """Compact view of one child's evaluation.json."""
    result = _read(Path(output_root) / "evaluation.json")
    rollout = result["rollout"]
    return {"end_reason": rollout["end_reason"], "tracking_success": rollout.get("tracking_success"),
            "reference_coverage": rollout["reference_clock"]["recorded_runtime_fraction"],
            "physics_steps": rollout["physics_steps"], "object_metrics": result.get("object_metrics", {}),
            "video_sha256": result["video"].get("sha256")}


def run_child(argv, *, env, directory, output_root, timeout=CHILD_TIMEOUT_SECONDS):
    """Run one evaluation child; retry once after a timeout, keeping failed outputs."""
    directory, output_root = Path(directory), Path(output_root)
    for attempt in range(1, MAX_EVAL_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            with (directory / f"child_attempt_{attempt}.log").open("x") as stream:
                proc = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            _write_once(directory / f"child_attempt_{attempt}.json", {"attempt": attempt, "returncode": proc.returncode,
                        "wall_seconds": time.perf_counter() - started, "timeout_seconds": timeout})
            return proc
        except subprocess.TimeoutExpired as exc:
            archive = directory / "failed_child_attempts" / f"attempt_{attempt}"
            archive.mkdir(parents=True, exist_ok=False)
            if output_root.exists():
                output_root.rename(archive / "output")
            _write_once(archive / "timeout.json", {"attempt": attempt, "error": str(exc), "timeout_seconds": timeout,
                        "observed_at": datetime.now(timezone.utc).isoformat()})
            if attempt == MAX_EVAL_ATTEMPTS:
                raise


class PeriodicEvalCallback(TrainerCallback):
    """Pack-driven periodic deterministic evaluation (motions and physics come from the pack)."""

    def __init__(self, output_root, pack, group=None, frequency=1000,
                 child_timeout_seconds=CHILD_TIMEOUT_SECONDS, video=True, eval_cpus=2, run_name=None):
        self.output = Path(output_root)
        catalog = _read(Path(pack) / "tracking_pack.json")
        groups = [g for g in catalog["groups"] if group is None or g["group_id"] == group]
        _require(len(groups) == 1, "Periodic evaluation needs exactly one pack group (use the group parameter)")
        self.group = groups[0]
        self.frequency = int(frequency)
        _require(self.frequency >= 1, "Periodic evaluation frequency must be positive")
        self.child_timeout = int(child_timeout_seconds)
        self.video = bool(video)
        self.eval_cpus = int(eval_cpus)
        self.run_name = run_name or self.output.name
        self.repo = Path(__file__).resolve().parents[3]
        self.child = self.repo / "gear_sonic/eval_contactfirst_trl.py"
        _require(self.child.is_file(), "Deterministic evaluation script missing from the SONIC tree")
        self.steps = None
        self.seen = set()

    def record_failure(self, *, step, motion_id, stage, exception, directory=None):
        row = {"observed_at": datetime.now(timezone.utc).isoformat(), "checkpoint_step": step, "motion_id": motion_id,
               "stage": stage, "error_type": type(exception).__name__, "error": str(exception),
               "traceback": "".join(traceback.format_exception(type(exception), exception, exception.__traceback__)),
               "training_continues": True, "directory": str(directory) if directory else None}
        with (self.output / "eval_failures.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print("[periodic_eval_failure_nonfatal] " + json.dumps({k: row[k] for k in ("checkpoint_step", "motion_id", "stage", "error")}), flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if self.steps is None:
            total = int(state.max_steps)
            self.steps = set(range(self.frequency, total + 1, self.frequency))
        if step not in self.steps or step in self.seen or not state.is_world_process_zero:
            return control
        self.seen.add(step)
        try:
            self._evaluate_boundary(step, state, kwargs)
        except Exception as exc:  # never fail training for an evaluation problem
            self.record_failure(step=step, motion_id=None, stage="boundary", exception=exc)
        return control

    def _evaluate_boundary(self, step, state, kwargs):
        import torch
        from gear_sonic.trl.callbacks import completed_saves
        checkpoint = self.output / "train" / f"model_step_{step:06d}.pt"
        saved = completed_saves.completed(step).get("regular")
        _require(saved is not None and Path(saved) == checkpoint, "Same-step completed checkpoint save required")
        boundary = self.output / "periodic_eval" / f"step_{step:06d}"
        boundary.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        model = kwargs["accelerator"].unwrap_model(kwargs["model"])
        torch.cuda.synchronize()
        before = parent_guard(state, model, kwargs["optimizer"], kwargs["lr_scheduler"], kwargs["env"])
        summaries = {}
        for motion in self.group["motions"]:
            directory = boundary / motion["motion_id"]
            try:
                summaries[motion["motion_id"]] = self._evaluate_motion(step, checkpoint, directory, motion)
            except Exception as exc:
                self.record_failure(step=step, motion_id=motion["motion_id"], stage="motion", exception=exc, directory=directory)
        after = parent_guard(state, model, kwargs["optimizer"], kwargs["lr_scheduler"], kwargs["env"])
        _require(before == after, "Parent PPO/optimizer/RNG/environment state changed during evaluation")
        receipt = {"checkpoint_step": step, "checkpoint_sha256": _sha(checkpoint), "motions": summaries,
                   "evaluated_motions": len(summaries), "expected_motions": len(self.group["motions"]),
                   "callback_wall_seconds": time.perf_counter() - started, "parent_state_unchanged": True,
                   "semantics": "one deterministic frame-zero episode per training motion; not a held-out success rate"}
        _write_once(boundary / "receipt.json", receipt)
        run = _wandb_run()
        if run is not None:
            import wandb
            payload = {"eval/checkpoint_step": step}
            for motion_id, row in summaries.items():
                prefix = "eval/motions/" + motion_id + "/"
                payload[prefix + "reference_coverage"] = row["reference_coverage"]
                payload[prefix + "tracking_success"] = int(bool(row["tracking_success"]))
                payload.update({prefix + k: v for k, v in row["object_metrics"].items() if isinstance(v, (int, float))})
                video = boundary / motion_id / "output" / "eval.mp4"
                if video.is_file():
                    payload[prefix + "video"] = wandb.Video(str(video), format="mp4")
            wandb.log(payload, commit=False)
        print(f"[periodic_eval] step {step}: " + json.dumps({k: {"coverage": v["reference_coverage"], "end": v["end_reason"]}
                                                           for k, v in summaries.items()}), flush=True)

    def _evaluate_motion(self, step, checkpoint, directory, motion):
        directory.mkdir(parents=True, exist_ok=False)
        source_cfg = self.output / "train/config.yaml"
        copied_cfg = directory / "source_train_config.yaml"
        with copied_cfg.open("xb") as stream:
            stream.write(source_cfg.read_bytes())
        from gear_sonic.eval_contactfirst_trl import UPSTREAM_SONIC_COMMIT
        request = {"remote_repo": str(self.repo), "source_commit": UPSTREAM_SONIC_COMMIT,
                   "case_id": self.run_name, "cpus": self.eval_cpus,
                   "phase": "periodic_eval", "motion_id": motion["motion_id"],
                   "source_motion_key": motion["source_motion_key"], "object_mesh_sha256": motion["object_mesh_sha256"],
                   "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint), "checkpoint_step": step,
                   "source_train_config": str(copied_cfg), "source_train_config_sha256": _sha(copied_cfg),
                   "output_root": str(directory / "output"), "num_envs": 1, "deterministic": True,
                   "start_from_first_frame": True, "seed": 0, "parent_wandb_run_id": None,
                   "parent_train_run_name": self.run_name, "parent_train_output_root": str(self.output),
                   "roster_manifest": motion["single_env_roster_path"], "roster_manifest_sha256": _sha(motion["single_env_roster_path"]),
                   "direct_dirs": None, "source_reference_frames": motion["source_reference"]["frames"],
                   "source_reference_fps": motion["source_reference"]["fps"],
                   "physics_overrides": self.group["physics_path"], "shader_cache_seed": None,
                   "render_video": self.video}
        _write_once(directory / "request.json", request)
        env = child_environment(request["output_root"])
        argv = [sys.executable, "-u", str(self.child), str(directory / "request.json")]
        proc = run_child(argv, env=env, directory=directory, output_root=request["output_root"], timeout=self.child_timeout)
        with (directory / "verify.log").open("x") as stream:
            subprocess.run([sys.executable, "-u", str(self.child), "--verify-completion",
                            str(directory / "request.json"), str(proc.returncode)], env=env, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=60, check=True)
        summary = summarize_evaluation(request["output_root"])
        _require(math.isfinite(summary["reference_coverage"]), "Invalid reference coverage")
        return summary
