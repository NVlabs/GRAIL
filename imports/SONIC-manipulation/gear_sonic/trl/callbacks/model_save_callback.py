import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

from loguru import logger
import torch

from gear_sonic.trl.callbacks.adaptive_sync_gate import sync_once_per_step
from gear_sonic.trl.callbacks import completed_saves
from transformers import TrainerCallback
import wandb

from gear_sonic.trl.utils.common import current_wandb_run_id, wandb_run_exists
from gear_sonic.trl.utils.wandb_identity import choose_wandb_run_id


class ModelSaveCallback(TrainerCallback):
    """Callback to save model state_dict during training."""

    def __init__(
        self,
        save_dir,
        save_frequency=1000,
        save_last_frequency=50,
        max_disk_usage=None,
        save_last_frequency_cap=None,
    ):
        """
        Args:
            save_dir (str): Directory to save model checkpoints
            save_frequency (int): Save model every N steps
            save_last_frequency (int): Overwrite last.pt every N steps
            max_disk_usage (float): Maximum disk usage in TB
            save_last_frequency_cap (int | None): Upper bound on the last.pt period.
                Lets a config shorten the resume rollback even when the launcher
                pins save_last_frequency on the command line. None keeps it.
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        if save_last_frequency_cap is not None:
            cap = int(save_last_frequency_cap)
            if cap < 1:
                raise ValueError(f"save_last_frequency_cap must be >= 1, got {cap}")
            save_last_frequency = min(int(save_last_frequency), cap)
        self.save_last_frequency = save_last_frequency
        logger.info(
            f"Checkpoint cadence: model_step every {save_frequency}, last.pt every "
            f"{self.save_last_frequency}"
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_last_only = False
        self.max_disk_usage = max_disk_usage

    def check_disk_usage(self):
        """Check if current working directory has more than 0.9TB of usage."""
        if self.save_last_only or self.max_disk_usage is None:
            return

        try:
            result = subprocess.run(["du", "-sb", "."], capture_output=True, text=True, check=True)
            size_bytes = int(result.stdout.split()[0])
            size_tb = size_bytes / (1024**4)
            if size_tb > self.max_disk_usage:
                self.save_last_only = True
                logger.info(
                    f"Directory size {size_tb:.2f}TB > {self.max_disk_usage}TB, setting save_last_only=True"
                )
                if wandb_run_exists():
                    wandb.alert(
                        title="Disk usage warning",
                        text=f"Directory size {size_tb:.2f}TB > {self.max_disk_usage}TB, setting save_last_only=True",
                        level="WARN",
                    )
        except Exception as e:
            logger.error(f"Error checking disk usage: {e}. Skip!")

    def on_step_end(self, args, state, control, **kwargs):
        """Save model state_dict at the end of each step if frequency matches."""
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        env = kwargs.get("env")
        accelerator = kwargs.get("accelerator")

        regular_due = (
            not self.save_last_only
            and state.global_step % self.save_frequency == 0
        )
        last_due = state.global_step % self.save_last_frequency == 0
        save_due = not env.is_evaluating and (regular_due or last_due)
        if save_due and accelerator is not None:
            # the checkpoint needs this step's globally synced sampler state; when the trainer (or another callback)
            # already synced this very step, the collective is not repeated (adaptive_sync_gate)
            sync_once_per_step(env, accelerator, state.global_step, source="save")
            accelerator.wait_for_everyone()

        if state.is_world_process_zero and save_due:
            regular_path = f"{self.save_dir}/model_step_{state.global_step:06d}.pt"
            last_path = f"{self.save_dir}/last.pt"
            # Same-step reuse (in memory, never a readback): a name already completed at this step by any callback is
            # not serialized again; a second name is published as a hard link / copy of the completed file. The
            # environment state (with adaptive sampling ON: 166-287 s at 8x8192) is built only when something is
            # actually serialized. Every save writes a NEW file through a temp + os.replace, so a later last.pt never
            # touches a model_step file.
            prior = completed_saves.completed(state.global_step)
            regular_needed = regular_due and "regular" not in prior
            last_needed = last_due and "last" not in prior
            env_state_dict = None
            if regular_needed:
                if not (prior.get("last") and ModelSaveCallback.publish_same_checkpoint(prior["last"], regular_path, state.global_step)):
                    env_state_dict = env.get_env_state_dict()
                    ModelSaveCallback.save_checkpoint(model, optimizer, lr_scheduler, state, env_state_dict, args, regular_path)
                completed_saves.record(state.global_step, "regular", regular_path)
            if last_needed:
                source = regular_path if regular_due else prior.get("regular")
                if not (source and ModelSaveCallback.publish_same_checkpoint(source, last_path, state.global_step)):
                    if env_state_dict is None:
                        env_state_dict = env.get_env_state_dict()
                    ModelSaveCallback.save_checkpoint(model, optimizer, lr_scheduler, state, env_state_dict, args, last_path)
                completed_saves.record(state.global_step, "last", last_path)
            if not (regular_needed or last_needed):
                print(f"[model_save] step {state.global_step}: checkpoint names already completed at this step ({prior}); nothing serialized again", flush=True)
                # self.export_policy_to_onnx(env, state, model)
        if save_due and accelerator is not None:
            accelerator.wait_for_everyone()

    def get_example_obs(self, env):
        obs_dict = copy.deepcopy(env.obs_buf_dict)
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()[0:1]
        return obs_dict

    def export_policy_to_onnx(self, env, state, model):
        checkpoint_path = os.path.join(env.config.experiment_dir, "last.pt")
        cmd = [
            sys.executable,
            "gear_sonic/eval_agent_trl.py",
            f"+checkpoint={checkpoint_path}",
            "+num_envs=1",
            "+headless=true",
            "+export_onnx_only=true",
        ]
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd())
        onnx_last_path = os.path.join(env.config.experiment_dir, "exported", "last.onnx")
        onnx_step_path = os.path.join(
            env.config.experiment_dir, "exported", f"model_step_{state.global_step:06d}.onnx"
        )
        shutil.copy(onnx_last_path, onnx_step_path)

    @staticmethod
    def publish_same_checkpoint(source_path, target_path, global_step):
        """Make ``target_path`` a second name of the checkpoint file just written at ``source_path`` (hard link, else a
        byte copy), atomically through a temp name + os.replace. Returns False when neither worked (the caller then
        serializes the checkpoint a second time, the previous behaviour). Never raises."""
        import shutil
        import tempfile
        import time

        started = time.perf_counter()
        tmp_path = None
        try:
            if not os.path.isfile(source_path):
                return False
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(target_path), delete=False, suffix=".pt.tmp") as tmp_file:
                tmp_path = tmp_file.name
            os.remove(tmp_path)  # os.link needs a free name; the directory entry is re-created atomically below
            how = "hard link"
            try:
                os.link(source_path, tmp_path)
            except OSError:
                how = "copy"
                shutil.copyfile(source_path, tmp_path)
            os.replace(tmp_path, target_path)
            print(
                f"Saved model checkpoint to {target_path} "
                f"(step {global_step}, {how} of {os.path.basename(source_path)}, {time.perf_counter() - started:.2f}s)"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - optional shortcut only; the second full save remains the fallback
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f"Could not publish {target_path} as a second name of {source_path} ({type(exc).__name__}: {exc}); saving it again")
            return False

    @classmethod
    def save_checkpoint(
        cls, model, optimizer, lr_scheduler, state, env_state_dict, args, save_path
    ):
        if model is not None:
            # Save model, optimizer, scheduler and training state. The TrainerState is copied WITHOUT its log_history
            # (one dict per logged step, never consumed by a resume): deep-copying the whole state and then dropping
            # the history was work growing with the leg length (user decision 2026-10-14: no work PPO does not need).
            _state = copy.copy(state)                 # new object with its own (shallow) attribute dict
            _state.__dict__.pop("log_history", None)   # the live state keeps its history; the copy is written without it
            checkpoint = {
                "policy_state_dict": model.policy.state_dict(),
                "value_state_dict": (
                    model.value_model.state_dict() if model.value_model is not None else None
                ),  # Value model is not always preset (e.g. for distillation)
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "lr_scheduler_state_dict": (
                    lr_scheduler.state_dict() if lr_scheduler is not None else None
                ),
                "state": _state,
                "args": args,
                "env_state_dict": env_state_dict,
                "wandb_run": choose_wandb_run_id(getattr(args, "wandb_run_id", None), current_wandb_run_id()),
            }

            if hasattr(model, "disc_model") and model.disc_model is not None:
                checkpoint["disc_state_dict"] = model.disc_model.state_dict()

            import tempfile
            import time

            save_dir = os.path.dirname(save_path)

            max_attempts = 5
            for attempt in range(max_attempts):
                tmp_path = None
                try:
                    # Save to a temp file first, then atomically rename
                    # This prevents corrupted partial checkpoints on filesystem failures
                    with tempfile.NamedTemporaryFile(
                        dir=save_dir, delete=False, suffix=".pt.tmp"
                    ) as tmp_file:
                        tmp_path = tmp_file.name

                    save_start = time.perf_counter()
                    torch.save(checkpoint, tmp_path)

                    # Atomic rename (os.replace is atomic on POSIX systems)
                    os.replace(tmp_path, save_path)
                    print(
                        f"Saved model checkpoint to {save_path} "
                        f"(step {state.global_step}, {time.perf_counter() - save_start:.2f}s)"
                    )
                    break
                except Exception as e:
                    # Clean up temp file if it exists
                    if tmp_path is not None and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                    if attempt == max_attempts - 1:
                        raise RuntimeError(
                            f"Failed to save checkpoint to {save_path} after "
                            f"{max_attempts} attempts; refusing to continue with a "
                            "potentially stale checkpoint"
                        ) from e
                    print(f"Attempt {attempt + 1} failed to save checkpoint. Retrying...")
                    time.sleep(
                        5
                    )  # Wait a bit before retrying (helps with transient filesystem issues)
