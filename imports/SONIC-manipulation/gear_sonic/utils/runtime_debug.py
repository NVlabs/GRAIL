from __future__ import annotations

import math
import os

import torch


def log_actor_init(
    *,
    use_log_std: bool,
    init_noise_std: float,
    algo_config,
    clamp_noise_std: bool,
) -> None:
    print(
        "[ACTOR INIT] "
        f"use_log_std={use_log_std} "
        f"init_noise_std={init_noise_std} "
        f"use_clampped_std={algo_config.get('use_clampped_std', False)} "
        f"std_clamp_min={algo_config.get('std_clamp_min', None)} "
        f"std_clamp_max={algo_config.get('std_clamp_max', None)} "
        f"clamp_noise_std={clamp_noise_std} "
        f"max_noise_std={algo_config.get('max_noise_std', None)}"
    )


def summarize_tensor(name, tensor, mark_nonpositive_invalid=False) -> str:
    if tensor is None:
        return f"{name}: None"

    tensor_detached = tensor.detach()
    flat = tensor_detached.reshape(-1).float()
    nan_count = torch.isnan(flat).sum().item()
    posinf_count = torch.isposinf(flat).sum().item()
    neginf_count = torch.isneginf(flat).sum().item()
    invalid_mask = torch.isnan(flat) | torch.isinf(flat)
    if mark_nonpositive_invalid:
        invalid_mask = invalid_mask | (flat <= 0)

    summary = [
        f"{name}: shape={tuple(tensor_detached.shape)}",
        f"nan={nan_count}",
        f"+inf={posinf_count}",
        f"-inf={neginf_count}",
    ]

    finite = flat[torch.isfinite(flat)]
    if finite.numel() > 0:
        summary.extend(
            [
                f"min={finite.min().item():.6g}",
                f"max={finite.max().item():.6g}",
                f"mean={finite.mean().item():.6g}",
            ]
        )

    if invalid_mask.any():
        bad_idx = invalid_mask.nonzero(as_tuple=False).flatten()[:8].tolist()
        bad_vals = flat[bad_idx].detach().cpu().tolist()
        summary.append(f"bad_idx={bad_idx}")
        summary.append(f"bad_vals={bad_vals}")

    return ", ".join(summary)


def log_actor_distribution_state(
    context: str,
    *,
    input_key: str,
    steps: int,
    is_eval_mode: bool,
    use_log_std: bool,
    obs_dict=None,
    mean=None,
    std=None,
    log_std_param=None,
    std_param=None,
) -> None:
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(
        "[ACTOR DIST DEBUG] "
        f"context={context} rank={rank} local_rank={local_rank} "
        f"step={steps} eval={is_eval_mode} use_log_std={use_log_std}"
    )
    if obs_dict is not None:
        if input_key in obs_dict:
            print("  " + summarize_tensor(f"obs[{input_key}]", obs_dict[input_key]))
        else:
            print(f"  obs_keys={list(obs_dict.keys())}")
    if mean is not None:
        print("  " + summarize_tensor("action_mean", mean))
    if std is not None:
        print("  " + summarize_tensor("action_std", std, mark_nonpositive_invalid=True))
    if use_log_std:
        print("  " + summarize_tensor("raw_log_std_param", log_std_param))
    else:
        print("  " + summarize_tensor("raw_std_param", std_param, mark_nonpositive_invalid=True))


def tensor_batch_invalid_stats(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None

    flat = tensor.detach().reshape(tensor.shape[0], -1).float()
    invalid_mask = torch.isnan(flat) | torch.isinf(flat)
    if not invalid_mask.any():
        return None

    bad_env_ids = invalid_mask.any(dim=1).nonzero(as_tuple=False).flatten()
    finite = flat[torch.isfinite(flat)]
    stats = {
        "shape": tuple(tensor.shape),
        "nan": int(torch.isnan(flat).sum().item()),
        "posinf": int(torch.isposinf(flat).sum().item()),
        "neginf": int(torch.isneginf(flat).sum().item()),
        "bad_env_ids": bad_env_ids.detach().cpu().tolist(),
    }
    if finite.numel() > 0:
        stats["min"] = float(finite.min().item())
        stats["max"] = float(finite.max().item())
        stats["mean"] = float(finite.mean().item())
    return stats


def log_invalid_stats(prefix: str, stats) -> None:
    if stats is None:
        return

    msg = [
        prefix,
        f"shape={stats['shape']}",
        f"nan={stats['nan']}",
        f"+inf={stats['posinf']}",
        f"-inf={stats['neginf']}",
    ]
    if "min" in stats:
        msg.extend(
            [
                f"min={stats['min']:.6g}",
                f"max={stats['max']:.6g}",
                f"mean={stats['mean']:.6g}",
            ]
        )
    if stats["bad_env_ids"]:
        msg.append(f"bad_env_ids={stats['bad_env_ids'][:8]}")
    print(" ".join(msg))


def debug_invalid_obs_group(
    group_name,
    group_value,
    *,
    observation_manager,
    obs_debug_step: int,
    invalid_obs_debug_count: int,
):
    if observation_manager is None:
        return []

    group_bad_env_ids = set()

    if isinstance(group_value, dict):
        obs_names = observation_manager._group_obs_term_names.get(group_name, list(group_value.keys()))
        invalid_term_stats = []
        for obs_name in obs_names:
            if obs_name not in group_value:
                continue
            stats = tensor_batch_invalid_stats(group_value[obs_name])
            if stats is None:
                continue
            invalid_term_stats.append((obs_name, stats))
            group_bad_env_ids.update(stats["bad_env_ids"])
        if invalid_term_stats:
            print(
                f"[OBS DEBUG] invalid dict group={group_name} obs_step={obs_debug_step} "
                f"debug_count={invalid_obs_debug_count}"
            )
            for obs_name, stats in invalid_term_stats:
                log_invalid_stats(f"[OBS DEBUG] group={group_name} term={obs_name}", stats)
        return sorted(group_bad_env_ids)

    if not isinstance(group_value, torch.Tensor):
        return []

    group_stats = tensor_batch_invalid_stats(group_value)
    if group_stats is None:
        return []

    print(
        f"[OBS DEBUG] invalid group={group_name} obs_step={obs_debug_step} "
        f"debug_count={invalid_obs_debug_count}"
    )
    log_invalid_stats(f"[OBS DEBUG] group={group_name}", group_stats)
    group_bad_env_ids.update(group_stats["bad_env_ids"])

    obs_names = observation_manager._group_obs_term_names.get(group_name, [])
    obs_dims = observation_manager._group_obs_term_dim.get(group_name, [])
    if len(obs_names) != len(obs_dims):
        return sorted(group_bad_env_ids)

    flat = group_value.detach().reshape(group_value.shape[0], -1)
    start = 0
    for obs_name, obs_dim in zip(obs_names, obs_dims):
        if isinstance(obs_dim, (tuple, list, torch.Size)):
            term_dim = int(math.prod(obs_dim))
        else:
            term_dim = int(obs_dim)
        term_tensor = flat[:, start : start + term_dim]
        start += term_dim
        stats = tensor_batch_invalid_stats(term_tensor)
        if stats is None:
            continue
        group_bad_env_ids.update(stats["bad_env_ids"])
        log_invalid_stats(f"[OBS DEBUG] group={group_name} term={obs_name}", stats)

    return sorted(group_bad_env_ids)


def debug_invalid_env_state(env, bad_env_ids, *, obs_debug_step: int) -> None:
    if not bad_env_ids:
        return

    env_ids = torch.tensor(bad_env_ids[:8], device=env.device, dtype=torch.long)
    print(
        f"[OBS DEBUG] invalid env state snapshot obs_step={obs_debug_step} "
        f"env_ids={env_ids.detach().cpu().tolist()}"
    )

    if hasattr(env, "episode_length_buf"):
        ep_lens = env.episode_length_buf[env_ids].detach().cpu().tolist()
        print(f"[OBS DEBUG] episode_length_buf={ep_lens}")

    if "robot" in env.scene.keys():
        robot = env.scene["robot"]
        for attr_name in [
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
        ]:
            tensor = getattr(robot.data, attr_name, None)
            if tensor is None:
                continue
            stats = tensor_batch_invalid_stats(tensor[env_ids])
            if stats is not None:
                log_invalid_stats(f"[OBS DEBUG] robot.{attr_name}", stats)

    if "object" in env.scene.keys():
        obj = env.scene["object"]
        for attr_name in [
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
        ]:
            tensor = getattr(obj.data, attr_name, None)
            if tensor is None:
                continue
            stats = tensor_batch_invalid_stats(tensor[env_ids])
            if stats is not None:
                log_invalid_stats(f"[OBS DEBUG] object.{attr_name}", stats)
