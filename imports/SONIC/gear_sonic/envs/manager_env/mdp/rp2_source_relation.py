"""Source-SOMA relational observations, rewards, reset, and termination terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)

from gear_sonic.envs.manager_env.mdp import utils
from gear_sonic.envs.manager_env.mdp.observations import PolicyCfg, PrivilegedCfg, TokenizerCfg
from gear_sonic.envs.manager_env.mdp.rewards import RewardsCfg
from gear_sonic.envs.manager_env.mdp.terminations import TerminationsCfg
from gear_sonic.rp2.reward_kernels import (
    bilateral_reference_contact_gate,
    reference_aware_support_contact_penalty,
    tolerance_tracking_score,
    update_bilateral_contact_failure_counter,
)
from gear_sonic.rp2.discrete_contact import (
    CONTACT_BODY_GROUPS,
    CONTACT_ENTITY_CLASSES,
    G1_CONTACT_GROUP_LINKS,
)
from gear_sonic.rp2.source_relation import DEFAULT_SOURCE_RELATION_MAPPING
from gear_sonic.trl.utils import torch_transform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


DEFAULT_ROBOT_BODY_NAMES = tuple(
    robot_body for robot_body, _ in DEFAULT_SOURCE_RELATION_MAPPING
)
RELATION_GEOMETRIES = ("center", "surface_sdf")


@configclass
class RP2SourceRelationPolicyCfg(PolicyCfg):
    """Policy observation group for source-SOMA relation control."""

    rp2_source_relation_residual_sparse = None
    rp2_source_relation_mask_sparse = None
    rp2_source_contact_residual_sparse = None
    rp2_source_bilateral_contact_force = None
    rp2_source_object_pos_residual_sparse = None
    rp2_source_object_ori_residual_sparse = None


@configclass
class RP2SourceRelationPrivilegedCfg(PrivilegedCfg):
    """Critic group with uncorrupted source targets and simulator state."""

    rp2_source_relation_residual_sparse = None
    rp2_source_relation_target_sparse = None
    rp2_source_relation_current = None
    rp2_source_relation_mask_sparse = None
    rp2_source_contact_residual_sparse = None
    rp2_source_contact_target_sparse = None
    rp2_source_bilateral_contact_force = None
    rp2_source_object_pos_residual_sparse = None
    rp2_source_object_ori_residual_sparse = None


@configclass
class RP2SourceRelationTokenizerCfg(TokenizerCfg):
    """Shape-compatible tokenizer group containing no retargeted reference."""


@configclass
class RP2SourceRelationRewardsCfg(RewardsCfg):
    """Reward group for direct raw-human relation learning."""

    rp2_source_relation_tracking = None
    rp2_source_bilateral_contact = None
    rp2_reference_aware_undesired_contacts = None
    # Opt-in wrist surface-proximity reward for labeled-contact frames
    # (rp2_wrist_contact_surface_proximity_reward); enabled via overrides.
    rp2_wrist_contact_proximity = None


@configclass
class RP2SourceRelationTerminationsCfg(TerminationsCfg):
    """Physical and task terminations without G1-reference deviations."""

    rp2_source_fall = None
    rp2_source_relation_deviation = None
    rp2_source_contact_failure = None


def zero_tokenizer_observation(
    env: ManagerBasedEnv, shape: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Return a fixed-shape compatibility tensor independent of any G1 reference."""
    return torch.zeros((env.num_envs, *tuple(int(value) for value in shape)), device=env.device)


def _command(env: ManagerBasedEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def _future_steps(command, offsets: list[int] | tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    offset_tensor = torch.as_tensor(offsets, device=command.device, dtype=torch.long)
    current = (command.motion_start_time_steps + command.time_steps).long()
    steps = current[:, None] + offset_tensor[None, :]
    max_steps = torch.clamp(command.motion_num_steps.long() - 1, min=0)
    steps = torch.minimum(torch.clamp(steps, min=0), max_steps[:, None])
    motion_ids = command.motion_ids[:, None].expand_as(steps)
    return motion_ids, steps


def _source_channel(command, getter_name: str, offsets: list[int] | tuple[int, ...]):
    getter = getattr(command.motion_lib, getter_name, None)
    if getter is None:
        raise RuntimeError(f"Motion library does not provide required source channel {getter_name}")
    motion_ids, steps = _future_steps(command, offsets)
    values = getter(motion_ids.reshape(-1), steps.reshape(-1))
    if values is None:
        raise RuntimeError(f"Motion library source channel {getter_name} is absent")
    return values.reshape(command.num_envs, len(offsets), *values.shape[1:])


def _gather_target_object_channel(command, values: torch.Tensor) -> torch.Tensor:
    """Gather [env, offsets, objects, ...] using the online target object index."""
    if values.ndim < 4:
        raise ValueError(f"Expected full-scene object channel, got {tuple(values.shape)}")
    object_indices = command.target_object_indices().to(values.device).long()
    max_objects = values.shape[2]
    object_indices = torch.clamp(object_indices, min=0, max=max_objects - 1)
    gather_shape = (
        values.shape[0],
        values.shape[1],
        1,
        *values.shape[3:],
    )
    gather_idx = object_indices.view(-1, 1, 1, *([1] * (values.ndim - 3))).expand(gather_shape)
    return torch.gather(values, dim=2, index=gather_idx).squeeze(2)


def _targeted_source_channel(
    command,
    *,
    single_getter_name: str,
    full_scene_getter_name: str,
    offsets: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Return source channel for the currently selected online target object.

    New full-sequence RP2 packs store [T, O, ...] source labels.  Older composed
    packs store one preselected object as [T, ...].  Prefer the full-scene
    channel when present, and preserve the legacy one-object path as fallback.
    """
    full_getter = getattr(command.motion_lib, full_scene_getter_name, None)
    if full_getter is not None:
        motion_ids, steps = _future_steps(command, offsets)
        values = full_getter(motion_ids.reshape(-1), steps.reshape(-1))
        if values is not None:
            values = values.reshape(command.num_envs, len(offsets), *values.shape[1:])
            return _gather_target_object_channel(command, values)
    return _source_channel(command, single_getter_name, offsets)


def _validate_relation_geometry(relation_geometry: str) -> str:
    geometry = str(relation_geometry).lower()
    if geometry not in RELATION_GEOMETRIES:
        raise ValueError(
            f"relation_geometry must be one of {RELATION_GEOMETRIES}, got "
            f"{relation_geometry!r}"
        )
    return geometry


def _source_relation_channel(
    command,
    offsets: list[int] | tuple[int, ...],
    relation_geometry: str = "center",
) -> torch.Tensor:
    geometry = _validate_relation_geometry(relation_geometry)
    if geometry == "surface_sdf":
        return _targeted_source_channel(
            command,
            single_getter_name="get_rp2_source_surface_relation_vector",
            full_scene_getter_name=(
                "get_rp2_full_scene_source_surface_relation_vector"
            ),
            offsets=offsets,
        )
    return _targeted_source_channel(
        command,
        single_getter_name="get_rp2_source_relation_vector",
        full_scene_getter_name="get_rp2_full_scene_source_relation_vector",
        offsets=offsets,
    )


def _source_relation_mask_channel(
    command,
    offsets: list[int] | tuple[int, ...],
    relation_geometry: str = "center",
) -> torch.Tensor:
    geometry = _validate_relation_geometry(relation_geometry)
    if geometry == "surface_sdf":
        return _targeted_source_channel(
            command,
            single_getter_name="get_rp2_source_surface_relation_mask",
            full_scene_getter_name=(
                "get_rp2_full_scene_source_surface_relation_mask"
            ),
            offsets=offsets,
        )
    return _targeted_source_channel(
        command,
        single_getter_name="get_rp2_source_relation_mask",
        full_scene_getter_name="get_rp2_full_scene_source_relation_mask",
        offsets=offsets,
    )


def _source_surface_point_local_channel(
    command, offsets: list[int] | tuple[int, ...]
) -> torch.Tensor:
    return _targeted_source_channel(
        command,
        single_getter_name="get_rp2_source_surface_point_local",
        full_scene_getter_name=(
            "get_rp2_full_scene_source_surface_point_local"
        ),
        offsets=offsets,
    )


def _source_contact_probability_channel(command, offsets: list[int] | tuple[int, ...]) -> torch.Tensor:
    return _targeted_source_channel(
        command,
        single_getter_name="get_rp2_source_contact_probability",
        full_scene_getter_name="get_rp2_full_scene_source_contact_probability",
        offsets=offsets,
    )


def _body_indices(command, robot_body_names: list[str] | tuple[str, ...]) -> list[int]:
    missing = [name for name in robot_body_names if name not in command.cfg.body_names]
    if missing:
        raise KeyError(f"Source relation robot bodies are not tracked by the command: {missing}")
    return [command.cfg.body_names.index(name) for name in robot_body_names]


def _sim_human_centric_relation(
    env: ManagerBasedEnv,
    command,
    robot_body_names: list[str] | tuple[str, ...],
) -> torch.Tensor:
    body_indices = _body_indices(command, robot_body_names)
    body_pos_w = command.robot_body_pos_w[:, body_indices]
    object_pos_w, _, _, _, _ = utils.get_target_object_current_state(env, command)
    world_vectors = object_pos_w[:, None, :] - body_pos_w
    heading = torch_transform.get_heading_q(command.robot_anchor_quat_w)
    heading_inv = quat_inv(heading)[:, None, :].expand(-1, len(body_indices), -1)
    return quat_apply(
        heading_inv.reshape(-1, 4), world_vectors.reshape(-1, 3)
    ).reshape(env.num_envs, len(body_indices), 3)


def _sim_surface_relation(
    env: ManagerBasedEnv,
    command,
    robot_body_names: list[str] | tuple[str, ...],
    offsets: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Relate G1 links to the homologous reference surface anchors."""
    body_indices = _body_indices(command, robot_body_names)
    body_pos_w = command.robot_body_pos_w[:, body_indices]
    surface_point_local = _source_surface_point_local_channel(command, offsets)
    object_pos_w, object_quat_w, _, _, _ = utils.get_target_object_current_state(
        env, command
    )
    object_quat = object_quat_w[:, None, None, :].expand(
        -1,
        len(offsets),
        len(body_indices),
        -1,
    )
    surface_point_w = quat_apply(
        object_quat.reshape(-1, 4),
        surface_point_local.reshape(-1, 3),
    ).reshape_as(surface_point_local)
    surface_point_w = surface_point_w + object_pos_w[:, None, None, :]
    world_vectors = surface_point_w - body_pos_w[:, None, :, :]
    heading_inv = quat_inv(
        torch_transform.get_heading_q(command.robot_anchor_quat_w)
    )
    heading_inv = heading_inv[:, None, None, :].expand(
        -1,
        len(offsets),
        len(body_indices),
        -1,
    )
    return quat_apply(
        heading_inv.reshape(-1, 4), world_vectors.reshape(-1, 3)
    ).reshape_as(world_vectors)


def _sim_relation_for_offsets(
    env: ManagerBasedEnv,
    command,
    robot_body_names: list[str] | tuple[str, ...],
    offsets: list[int] | tuple[int, ...],
    relation_geometry: str,
) -> torch.Tensor:
    geometry = _validate_relation_geometry(relation_geometry)
    if geometry == "surface_sdf":
        return _sim_surface_relation(
            env,
            command,
            robot_body_names,
            offsets,
        )
    current = _sim_human_centric_relation(env, command, robot_body_names)
    return current[:, None].expand(-1, len(offsets), -1, -1)


def _heading_quat_from_xy(heading_xy: torch.Tensor) -> torch.Tensor:
    heading_xy = heading_xy / torch.clamp(
        torch.linalg.vector_norm(heading_xy, dim=-1, keepdim=True), min=1.0e-6
    )
    yaw = torch.atan2(heading_xy[..., 1], heading_xy[..., 0])
    quat = torch.zeros((*yaw.shape, 4), dtype=heading_xy.dtype, device=heading_xy.device)
    quat[..., 0] = torch.cos(0.5 * yaw)
    quat[..., 3] = torch.sin(0.5 * yaw)
    return quat


def _source_relation_in_current_robot_heading(
    command,
    target: torch.Tensor,
    offsets: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Express future source-local vectors in the current robot heading frame."""
    source_heading_xy = _source_channel(
        command, "get_rp2_source_root_heading_xy", offsets
    )
    source_heading = _heading_quat_from_xy(source_heading_xy)
    robot_heading = torch_transform.get_heading_q(command.robot_anchor_quat_w)
    robot_heading = robot_heading[:, None, :].expand_as(source_heading)
    source_to_robot = quat_mul(quat_inv(robot_heading), source_heading)
    source_to_robot = source_to_robot[:, :, None, :].expand(
        -1, -1, target.shape[2], -1
    )
    return quat_apply(
        source_to_robot.reshape(-1, 4), target.reshape(-1, 3)
    ).reshape_as(target)


def _env_perf_config(env) -> dict:
    """Return the manager_env ``config`` mapping (empty when unavailable)."""
    cfg = getattr(env, "cfg", None)
    config = getattr(cfg, "config", None)
    if config is None:
        return {}
    return config


def _perf_flag(env, key: str) -> bool:
    """Opt-in D performance-migration flag under ``manager_env.config``."""
    try:
        return bool(_env_perf_config(env).get(key, False))
    except AttributeError:
        return False


def _sync_free_contact_cache_invalidation_enabled(env) -> bool:
    """Resolve ``runtime_sync_cleanup.sync_free_contact_cache_invalidation``.

    Same config location as the matching wrapper option so one yaml block
    enables the wrapper and detector parts together.
    """
    try:
        cleanup = _env_perf_config(env).get("runtime_sync_cleanup", {}) or {}
        return bool(
            cleanup.get("enabled", False)
            and cleanup.get("sync_free_contact_cache_invalidation", False)
        )
    except AttributeError:
        return False


def _contact_sensor_parity_probe_enabled(env) -> bool:
    """``manager_env.config.contact_sensor_parity_probe.enabled`` (diagnostics only)."""
    try:
        probe = _env_perf_config(env).get("contact_sensor_parity_probe", {}) or {}
        return bool(probe.get("enabled", False))
    except AttributeError:
        return False


def _record_reward_phase_parity_snapshot(env, step: int) -> None:
    """Probe-only: compare every primary sensor with its ``shadow__`` twin and the
    grouped canonical family forces at the reward phase (the consumption point
    of the required-contact and grasp rewards), including environments that the
    same step will reset afterwards. Stored on the env for the wrapper probe."""
    from gear_sonic.utils.contact_sensor_parity import reward_phase_snapshot

    sensors = getattr(env.scene, "sensors", {})
    env._contact_sensor_parity_reward_phase = reward_phase_snapshot(  # noqa: SLF001
        sensors,
        step,
        env=env,
        family_force_fn=lambda e, family: _canonical_sensor_force(e, family, allow_missing=True),
    )


def _episode_length_mutation_version(episode_length: torch.Tensor) -> int | None:
    """Return PyTorch's host-side in-place mutation generation (no device sync).

    ``Tensor._version`` advances on every in-place write to ``episode_length_buf``
    (the per-step ``+= 1`` and every ``[env_ids] = 0`` reset write), so together
    with ``common_step_counter`` it identifies "same step, no reset since".

    Returns ``None`` when the tensor has no version counter (inference-mode
    tensors do not track one): callers then fall back to the legacy value
    comparison, so evaluation under ``torch.inference_mode`` keeps working with
    the flag enabled (one host sync per read, exactly the pre-migration cost).
    """
    try:
        if episode_length.is_inference():
            return None
        return int(episode_length._version)  # noqa: SLF001
    except (AttributeError, RuntimeError):
        return None


def _step_observation_cache_get(env: ManagerBasedEnv, attr: str, key: tuple):
    """Return a same-step memoized tensor, else ``None``.

    Sync-free invalidation: valid only for the same ``common_step_counter`` AND
    the same in-place mutation generation of ``episode_length_buf`` (any reset
    write between two reads bumps ``Tensor._version``). The ObservationManager
    clones every term result per group, so one tensor may serve several groups.
    """
    step = getattr(env, "common_step_counter", None)
    episode_length = getattr(env, "episode_length_buf", None)
    if step is None or episode_length is None:
        return None
    try:
        version = int(episode_length._version)  # noqa: SLF001
    except (AttributeError, RuntimeError):
        return None
    cache = getattr(env, attr, None)
    if (
        cache is not None
        and cache["step"] == int(step)
        and cache["episode_length_buffer"] is episode_length
        and cache["episode_length_version"] == version
    ):
        return cache["values"].get(key)
    return None


def _step_observation_cache_put(
    env: ManagerBasedEnv, attr: str, key: tuple, value: torch.Tensor
) -> torch.Tensor:
    """Store a per-step memoized tensor keyed by the full parameter tuple."""
    step = getattr(env, "common_step_counter", None)
    episode_length = getattr(env, "episode_length_buf", None)
    if step is None or episode_length is None:
        return value
    try:
        version = int(episode_length._version)  # noqa: SLF001
    except (AttributeError, RuntimeError):
        return value
    cache = getattr(env, attr, None)
    if (
        cache is None
        or cache["step"] != int(step)
        or cache["episode_length_buffer"] is not episode_length
        or cache["episode_length_version"] != version
    ):
        cache = {
            "step": int(step),
            "episode_length_buffer": episode_length,
            "episode_length_version": version,
            "values": {},
        }
        setattr(env, attr, cache)
    cache["values"][key] = value
    return value


def rp2_source_relation_residual_sparse(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    offsets: list[int] = [1, 10],
    robot_body_names: list[str] = list(DEFAULT_ROBOT_BODY_NAMES),
    express_future_in_current_robot_heading: bool = False,
    relation_geometry: str = "center",
    invalid_fill_value: float = 0.0,
) -> torch.Tensor:
    """Sparse future source relation minus the current simulated relation."""
    # Relation-stack memo (opt-in
    # ``manager_env.config.relation_stack_memo``): when several observation
    # groups request this term with identical parameters in one step, the whole
    # stack (pack gathers + simulated relation) is computed once. The memo key is
    # the FULL parameter tuple; invalidation is sync-free (step + reset/mutation
    # generation). Note: the shipped config family configures no relation observation
    # terms (only the reward uses the stack), so this path is inactive there.
    memo_enabled = _perf_flag(env, "relation_stack_memo")
    memo_key = None
    if memo_enabled:
        memo_key = (
            str(command_name),
            tuple(int(offset) for offset in offsets),
            tuple(str(name) for name in robot_body_names),
            bool(express_future_in_current_robot_heading),
            str(relation_geometry).lower(),
            float(invalid_fill_value),
        )
        cached = _step_observation_cache_get(
            env, "_rp2_source_relation_residual_step_cache", memo_key
        )
        if cached is not None:
            return cached
    command = _command(env, command_name)
    target = _source_relation_channel(command, offsets, relation_geometry)
    if express_future_in_current_robot_heading:
        target = _source_relation_in_current_robot_heading(command, target, offsets)
    mask = _source_relation_mask_channel(command, offsets, relation_geometry)
    current = _sim_relation_for_offsets(
        env,
        command,
        robot_body_names,
        offsets,
        relation_geometry,
    )
    residual = torch.where(
        mask[..., None] > 0.5,
        target - current,
        torch.full_like(target, float(invalid_fill_value)),
    )
    residual = residual.reshape(env.num_envs, -1)
    if memo_enabled:
        return _step_observation_cache_put(
            env, "_rp2_source_relation_residual_step_cache", memo_key, residual
        )
    return residual


def rp2_source_relation_target_sparse(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    offsets: list[int] = [1, 10],
    express_future_in_current_robot_heading: bool = False,
    relation_geometry: str = "center",
) -> torch.Tensor:
    """Privileged sparse source relation targets."""
    command = _command(env, command_name)
    target = _source_relation_channel(command, offsets, relation_geometry)
    if express_future_in_current_robot_heading:
        target = _source_relation_in_current_robot_heading(command, target, offsets)
    mask = _source_relation_mask_channel(command, offsets, relation_geometry)
    return (target * mask[..., None]).reshape(env.num_envs, -1)


def rp2_source_relation_current(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    robot_body_names: list[str] = list(DEFAULT_ROBOT_BODY_NAMES),
    relation_geometry: str = "center",
) -> torch.Tensor:
    """Privileged current G1 link-to-object vectors in the G1 heading frame."""
    command = _command(env, command_name)
    return _sim_relation_for_offsets(
        env,
        command,
        robot_body_names,
        [0],
        relation_geometry,
    )[:, 0].reshape(env.num_envs, -1)


def rp2_source_relation_mask_sparse(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    offsets: list[int] = [1, 10],
    relation_geometry: str = "center",
) -> torch.Tensor:
    command = _command(env, command_name)
    return _source_relation_mask_channel(
        command, offsets, relation_geometry
    ).reshape(
        env.num_envs, -1
    )


def _sensor_link_names(sensor) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return one contact sensor's (body link names, filter link names), cached.

    ``ContactSensor.body_names``
    materializes ``body_physx_view.prim_paths`` (``num_envs x num_bodies`` USD path
    strings) on EVERY access; the canonical family reads it for 50 sensors per
    policy step (~410k throwaway strings per step at 8192 envs). The value is a
    pure function of the sensor's PhysX view (``_body_physx_view`` and
    ``_num_bodies`` are assigned once in ``_initialize_impl`` and never mutated
    while stepping) and of the configured filter expressions, so the cache is
    keyed on the IDENTITY of the view object plus the filter tuple: a sensor
    rebuild installs a new view and misses. Ordering is the view's ordering,
    which is the same ordering that indexes ``force_matrix_w``.
    """
    view = getattr(sensor, "_body_physx_view", None)
    cfg = getattr(sensor, "cfg", None)
    filter_paths = tuple(getattr(cfg, "filter_prim_paths_expr", ()) or ())
    cached = getattr(sensor, "_rp2_sensor_link_name_cache", None)
    if (
        cached is not None
        and view is not None
        and cached[0] is view
        and cached[1] == filter_paths
    ):
        return cached[2], cached[3]
    body_names = tuple(
        str(name).rsplit("/", 1)[-1] for name in getattr(sensor, "body_names", ())
    )
    filter_names = tuple(str(path).rsplit("/", 1)[-1] for path in filter_paths)
    if view is not None:
        try:
            sensor._rp2_sensor_link_name_cache = (  # noqa: SLF001
                view,
                filter_paths,
                body_names,
                filter_names,
            )
        except AttributeError:
            pass
    return body_names, filter_names


def _canonical_sensor_force(
    env: ManagerBasedEnv,
    sensor_name: str,
    *,
    allow_missing: bool,
) -> torch.Tensor:
    """Aggregate valid one-body filtered sensors into canonical link groups.

    With ``manager_env.config.contact_force_family_cache`` the per-sensor link
    names come from :func:`_sensor_link_names` (identity-keyed cache) instead of
    re-materializing ``sensor.body_names`` on every call; the aggregation math is
    identical.
    """
    link_name_cache = _perf_flag(env, "contact_force_family_cache")
    sensors = getattr(env.scene, "sensors", {})
    matching_sensors = [
        (str(name), sensor)
        for name, sensor in sensors.items()
        if str(name) == sensor_name
        or str(name).startswith(f"{sensor_name}__")
    ]
    if not matching_sensors:
        if allow_missing:
            return torch.zeros(
                env.num_envs,
                len(CONTACT_BODY_GROUPS),
                dtype=torch.float32,
                device=env.device,
            )
        raise KeyError(f"Missing canonical contact sensor {sensor_name!r}")

    canonical_links = tuple(
        dict.fromkeys(
            link
            for group_name in CONTACT_BODY_GROUPS
            for link in G1_CONTACT_GROUP_LINKS[group_name]
        )
    )
    canonical_index = {
        name: index for index, name in enumerate(canonical_links)
    }
    link_force = torch.zeros(
        env.num_envs,
        len(canonical_links),
        dtype=torch.float32,
        device=env.device,
    )
    observed_links: set[str] = set()
    for actual_name, sensor in matching_sensors:
        force = sensor.data.force_matrix_w
        if force is None:
            raise RuntimeError(
                f"Canonical contact sensor {actual_name!r} has no filtered "
                "force matrix"
            )
        if (
            force.shape[0] != env.num_envs
            or force.shape[-1] != 3
            or force.ndim < 3
        ):
            raise ValueError(
                f"Unexpected force matrix for {actual_name!r}: "
                f"{tuple(force.shape)}"
            )
        magnitude = torch.linalg.vector_norm(force, dim=-1)
        if link_name_cache:
            body_names, filter_names = _sensor_link_names(sensor)
        else:
            body_names = [
                str(name).rsplit("/", 1)[-1]
                for name in getattr(sensor, "body_names", ())
            ]
        mapped = False
        if (
            magnitude.ndim >= 2
            and magnitude.shape[1] == len(body_names)
        ):
            body_magnitude = magnitude
            while body_magnitude.ndim > 2:
                body_magnitude = body_magnitude.amax(dim=-1)
            for index, name in enumerate(body_names):
                if name not in canonical_index:
                    continue
                target = canonical_index[name]
                link_force[:, target] = torch.maximum(
                    link_force[:, target],
                    body_magnitude[:, index],
                )
                observed_links.add(name)
                mapped = True

        if not link_name_cache:
            cfg = getattr(sensor, "cfg", None)
            filter_paths = list(
                getattr(cfg, "filter_prim_paths_expr", ()) or ()
            )
            filter_names = [
                str(path).rsplit("/", 1)[-1] for path in filter_paths
            ]
        if (
            magnitude.ndim >= 3
            and magnitude.shape[2] == len(filter_names)
        ):
            filter_magnitude = magnitude.amax(dim=1)
            while filter_magnitude.ndim > 2:
                filter_magnitude = filter_magnitude.amax(dim=-1)
            for index, name in enumerate(filter_names):
                if name not in canonical_index:
                    continue
                target = canonical_index[name]
                link_force[:, target] = torch.maximum(
                    link_force[:, target],
                    filter_magnitude[:, index],
                )
                observed_links.add(name)
                mapped = True
        if not mapped:
            raise ValueError(
                f"Canonical contact sensor {actual_name!r} maps neither its "
                "source bodies nor filters to canonical G1 links"
            )

    missing = sorted(
        set(canonical_links) - observed_links
    )
    if missing:
        raise KeyError(
            f"Canonical contact sensor family {sensor_name!r} is missing "
            f"G1 links: {missing}"
        )

    grouped_force = torch.zeros(
        env.num_envs,
        len(CONTACT_BODY_GROUPS),
        dtype=torch.float32,
        device=env.device,
    )
    for group_index, group_name in enumerate(CONTACT_BODY_GROUPS):
        indices = [
            canonical_index[name]
            for name in G1_CONTACT_GROUP_LINKS[group_name]
        ]
        grouped_force[:, group_index] = link_force[:, indices].amax(
            dim=-1
        )
    return grouped_force


def _update_hysteretic_contact(
    force: torch.Tensor,
    state: torch.Tensor,
    engage_count: torch.Tensor,
    release_count: torch.Tensor,
    *,
    engage_threshold_n: float,
    release_threshold_n: float,
    persistence_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a Schmitt-trigger contact state by one simulation frame."""
    if not 0.0 <= release_threshold_n <= engage_threshold_n:
        raise ValueError(
            "Contact thresholds must satisfy 0 <= release <= engage"
        )
    if persistence_frames < 1:
        raise ValueError("Contact persistence must be at least one frame")
    engaging = force >= float(engage_threshold_n)
    releasing = force < float(release_threshold_n)
    engage_count = torch.where(
        engaging,
        engage_count + 1,
        torch.zeros_like(engage_count),
    )
    release_count = torch.where(
        releasing,
        release_count + 1,
        torch.zeros_like(release_count),
    )
    state = torch.where(
        engage_count >= int(persistence_frames),
        torch.ones_like(state),
        state,
    )
    state = torch.where(
        release_count >= int(persistence_frames),
        torch.zeros_like(state),
        state,
    )
    return state, engage_count, release_count


def _aggregate_canonical_family_force(
    env: ManagerBasedEnv,
    *,
    active_sensor_name: str,
    static_sensor_name: str,
    ground_sensor_name: str,
) -> torch.Tensor:
    """Stack the active/static/ground canonical group forces ``(N, 3, groups)``."""
    active = _canonical_sensor_force(
        env,
        active_sensor_name,
        allow_missing=False,
    )
    static = _canonical_sensor_force(
        env,
        static_sensor_name,
        allow_missing=True,
    )
    ground = _canonical_sensor_force(
        env,
        ground_sensor_name,
        allow_missing=False,
    )
    return torch.stack((active, static, ground), dim=1)


def _canonical_contact_family_force(
    env: ManagerBasedEnv,
    *,
    active_sensor_name: str,
    static_sensor_name: str,
    ground_sensor_name: str,
    episode_length_buffer: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Aggregate the canonical sensor family once per step for all threshold keys.

    Opt-in via ``manager_env.config.contact_force_family_cache``. The
    stacked force tensor is a pure function of the sensor family and the
    simulation step; the hysteresis-threshold cache keys do not affect it, so
    detector instances with different thresholds (or repeated reads within one
    step) share one aggregation. Reuse is invalidated sync-free by the
    episode-length buffer's in-place mutation generation (``Tensor._version``),
    which advances on every reset write -- the only event that can change
    sensor readings between manager phases within one policy step. Without a
    version counter the family is aggregated on every call (legacy behaviour).
    """
    names = (active_sensor_name, static_sensor_name, ground_sensor_name)
    try:
        version = int(episode_length_buffer._version)  # noqa: SLF001
    except (AttributeError, RuntimeError):
        version = None
    cache = getattr(env, "_canonical_contact_force_cache", None)
    if (
        version is not None
        and cache is not None
        and cache["step"] == step
        and cache["names"] == names
        and cache["episode_length_buffer"] is episode_length_buffer
        and cache["episode_length_version"] == version
    ):
        return cache["force"]
    force = _aggregate_canonical_family_force(
        env,
        active_sensor_name=active_sensor_name,
        static_sensor_name=static_sensor_name,
        ground_sensor_name=ground_sensor_name,
    )
    if version is not None:
        setattr(
            env,
            "_canonical_contact_force_cache",
            {
                "step": step,
                "names": names,
                "episode_length_buffer": episode_length_buffer,
                "episode_length_version": version,
                "force": force,
            },
        )
    return force


def canonical_measured_contact_state(
    env: ManagerBasedEnv,
    *,
    active_sensor_name: str = "canonical_active_entity_contact_sensor",
    static_sensor_name: str = "canonical_static_context_contact_sensor",
    ground_sensor_name: str = "canonical_ground_contact_sensor",
    force_threshold_n: float = 1.0,
    force_release_threshold_n: float | None = None,
    persistence_frames: int = 1,
) -> torch.Tensor:
    """Return one cached, reset-aware binary contact state in canonical order.

    Two opt-in performance paths; with
    both flags absent the legacy code path below is unchanged:

    * ``manager_env.config.contact_force_family_cache``: check the
      per-threshold-key state cache BEFORE touching the sensors, and obtain the
      stacked family force from the shared per-step cache
      (:func:`_canonical_contact_family_force`), so repeated reads and
      different thresholds within one step aggregate the 50-sensor family once.
    * ``manager_env.config.runtime_sync_cleanup.sync_free_contact_cache_invalidation``:
      the common-case hit test uses the buffer's in-place mutation generation
      (``Tensor._version``) instead of a clone + ``torch.equal`` host sync, and
      resets are applied with ``torch.where`` instead of boolean-mask writes
      (``torch.any(reset)`` host sync). Semantics are the LEGACY semantics in
      every case: when the generation changed within the same step (a same-step
      re-read after a reset/mutation) the legacy value comparison is performed
      (one sync, only in that rare case) and, on a miss, every environment is
      re-ticked exactly as the legacy path does. A variant that advanced only the
      reset environments on such a re-read is deliberately NOT used, to
      preserve legacy semantics.
    """
    release_threshold = (
        float(force_threshold_n)
        if force_release_threshold_n is None
        else float(force_release_threshold_n)
    )
    key = (
        active_sensor_name,
        static_sensor_name,
        ground_sensor_name,
        float(force_threshold_n),
        release_threshold,
        int(persistence_frames),
    )
    family_cache_enabled = _perf_flag(env, "contact_force_family_cache")
    sync_free_invalidation = _sync_free_contact_cache_invalidation_enabled(env)
    episode_length_buffer = env.episode_length_buf
    step = int(getattr(env, "common_step_counter", 0))
    if _contact_sensor_parity_probe_enabled(env):
        # Diagnostics only (never in production runs): reward-phase snapshot.
        _record_reward_phase_parity_snapshot(env, step)

    force = None
    if not family_cache_enabled:
        # Legacy ordering: aggregate the sensor family before the cache check.
        force = _aggregate_canonical_family_force(
            env,
            active_sensor_name=active_sensor_name,
            static_sensor_name=static_sensor_name,
            ground_sensor_name=ground_sensor_name,
        )
    cache_by_key = getattr(env, "_canonical_contact_detector_cache", None)
    if cache_by_key is None:
        cache_by_key = {}
        setattr(env, "_canonical_contact_detector_cache", cache_by_key)
    cache = cache_by_key.get(key)
    episode_length_version = None
    if sync_free_invalidation:
        episode_length_version = _episode_length_mutation_version(
            episode_length_buffer
        )
        if (
            episode_length_version is not None
            and cache is not None
            and int(cache["step"]) == step
            and cache.get("episode_length_buffer") is episode_length_buffer
            and cache.get("episode_length_version") == episode_length_version
        ):
            # Same step, no in-place write since the cached read: the legacy
            # value comparison would succeed too, without a device->host sync.
            return cache["state"].to(dtype=torch.float32)
        # No usable version counter (e.g. inference-mode tensors) or a write
        # since the cached read: legacy value-based invalidation below.
        episode_length = episode_length_buffer.detach().clone()
        if (
            cache is not None
            and int(cache["step"]) == step
            and torch.equal(cache["episode_length"], episode_length)
        ):
            # Rare same-step re-read after a write that left the values equal:
            # keep the legacy hit (this is the only sync on the sync-free path).
            return cache["state"].to(dtype=torch.float32)
    else:
        episode_length = episode_length_buffer.detach().clone()
        if (
            cache is not None
            and int(cache["step"]) == step
            and torch.equal(cache["episode_length"], episode_length)
        ):
            return cache["state"].to(dtype=torch.float32)

    if force is None:
        # Cache miss for this threshold key: aggregate (or reuse) the shared
        # per-step family force AFTER the cache checks, so hits never pay for it.
        force = _canonical_contact_family_force(
            env,
            active_sensor_name=active_sensor_name,
            static_sensor_name=static_sensor_name,
            ground_sensor_name=ground_sensor_name,
            episode_length_buffer=episode_length_buffer,
            step=step,
        )

    expected_shape = force.shape
    cache_compatible = (
        cache is not None
        and tuple(cache["state"].shape) == tuple(expected_shape)
    )
    if not cache_compatible:
        state = torch.zeros(
            expected_shape, dtype=torch.bool, device=env.device
        )
        engage_count = torch.zeros(
            expected_shape, dtype=torch.int16, device=env.device
        )
        release_count = torch.zeros_like(engage_count)
        previous_episode_length = episode_length
    else:
        state = cache["state"]
        engage_count = cache["engage_count"]
        release_count = cache["release_count"]
        previous_episode_length = cache["episode_length"]

    reset = (episode_length < previous_episode_length) | (
        (episode_length == 0) & (previous_episode_length != 0)
    )
    if sync_free_invalidation:
        # Value-identical to the boolean-mask writes below, without the
        # torch.any(reset) device->host sync (out-of-place; cached tensors are
        # replaced, never aliased by consumers).
        reset_mask = reset[:, None, None]
        state = torch.where(reset_mask, torch.zeros_like(state), state)
        engage_count = torch.where(
            reset_mask, torch.zeros_like(engage_count), engage_count
        )
        release_count = torch.where(
            reset_mask, torch.zeros_like(release_count), release_count
        )
    elif torch.any(reset):
        state[reset] = False
        engage_count[reset] = 0
        release_count[reset] = 0
    # Legacy semantics on every path: one hysteresis tick for ALL environments
    # (a same-step re-read after a reset/mutation therefore re-ticks every
    # environment, exactly as the pre-migration detector did).
    state, engage_count, release_count = _update_hysteretic_contact(
        force,
        state,
        engage_count,
        release_count,
        engage_threshold_n=float(force_threshold_n),
        release_threshold_n=release_threshold,
        persistence_frames=int(persistence_frames),
    )
    cache_entry = {
        "step": step,
        "episode_length": episode_length,
        "state": state,
        "engage_count": engage_count,
        "release_count": release_count,
    }
    if sync_free_invalidation:
        cache_entry.update(
            {
                "episode_length_buffer": episode_length_buffer,
                "episode_length_version": episode_length_version,
            }
        )
    cache_by_key[key] = cache_entry
    return state.to(dtype=torch.float32)


def canonical_reference_contact_state(
    command,
    offsets: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Return ternary reference labels [N, offsets, entity, body]."""
    values = []
    current_steps = command.motion_start_time_steps + command.time_steps
    for offset in offsets:
        reference = command.motion_lib.get_reference_contact_state(
            command.motion_ids,
            current_steps + int(offset),
        )
        if reference is None:
            raise RuntimeError(
                "Motion pack is missing canonical reference_contact_state"
            )
        expected = (
            command.num_envs,
            len(CONTACT_ENTITY_CLASSES),
            len(CONTACT_BODY_GROUPS),
        )
        if tuple(reference.shape) != expected:
            raise ValueError(
                f"Canonical reference contact shape is {tuple(reference.shape)}, "
                f"expected {expected}"
            )
        if not torch.all((reference >= -1) & (reference <= 1)):
            raise ValueError("Canonical reference contact labels must be in {-1,0,+1}")
        values.append(reference.to(dtype=torch.float32))
    return torch.stack(values, dim=1)


def canonical_discrete_contact_observation(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    reference_offsets: list[int] = [0, 1, 10],
    sensor_cfg: SceneEntityCfg | None = None,
    active_sensor_name: str = "canonical_active_entity_contact_sensor",
    static_sensor_name: str = "canonical_static_context_contact_sensor",
    ground_sensor_name: str = "canonical_ground_contact_sensor",
    force_threshold_n: float = 1.0,
    force_release_threshold_n: float | None = None,
    persistence_frames: int = 1,
) -> torch.Tensor:
    """Concatenate 45 measured bits and 135 ternary reference labels."""
    # The stock observation term contributes this legacy parameter through
    # Hydra composition. Joint training deliberately uses the three named
    # canonical sensors instead.
    del sensor_cfg
    command = _command(env, command_name)
    measured = canonical_measured_contact_state(
        env,
        active_sensor_name=active_sensor_name,
        static_sensor_name=static_sensor_name,
        ground_sensor_name=ground_sensor_name,
        force_threshold_n=force_threshold_n,
        force_release_threshold_n=force_release_threshold_n,
        persistence_frames=persistence_frames,
    )
    reference = canonical_reference_contact_state(command, reference_offsets)
    return torch.cat(
        (
            measured.reshape(env.num_envs, -1),
            reference.reshape(env.num_envs, -1),
        ),
        dim=-1,
    )


def _canonical_contact_selection(
    *,
    device: torch.device,
    entity_classes: list[str] | tuple[str, ...] | None,
    body_groups: list[str] | tuple[str, ...] | None,
) -> torch.Tensor:
    selected_entities = (
        CONTACT_ENTITY_CLASSES if entity_classes is None else tuple(entity_classes)
    )
    selected_bodies = CONTACT_BODY_GROUPS if body_groups is None else tuple(body_groups)
    unknown_entities = sorted(set(selected_entities) - set(CONTACT_ENTITY_CLASSES))
    unknown_bodies = sorted(set(selected_bodies) - set(CONTACT_BODY_GROUPS))
    if unknown_entities or unknown_bodies:
        raise ValueError(
            "Unknown canonical contact selection: "
            f"entities={unknown_entities}, body_groups={unknown_bodies}"
        )
    entity_mask = torch.tensor(
        [name in selected_entities for name in CONTACT_ENTITY_CLASSES],
        dtype=torch.bool,
        device=device,
    )
    body_mask = torch.tensor(
        [name in selected_bodies for name in CONTACT_BODY_GROUPS],
        dtype=torch.bool,
        device=device,
    )
    return entity_mask[:, None] & body_mask[None, :]


def canonical_required_contact_reward(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    entity_classes: list[str] | tuple[str, ...] | None = None,
    body_groups: list[str] | tuple[str, ...] | None = None,
    active_sensor_name: str = "canonical_active_entity_contact_sensor",
    static_sensor_name: str = "canonical_static_context_contact_sensor",
    ground_sensor_name: str = "canonical_ground_contact_sensor",
    force_threshold_n: float = 1.0,
    force_release_threshold_n: float | None = None,
    persistence_frames: int = 1,
) -> torch.Tensor:
    """Reward the fraction of selected required contacts currently satisfied."""
    command = _command(env, command_name)
    measured = canonical_measured_contact_state(
        env,
        active_sensor_name=active_sensor_name,
        static_sensor_name=static_sensor_name,
        ground_sensor_name=ground_sensor_name,
        force_threshold_n=force_threshold_n,
        force_release_threshold_n=force_release_threshold_n,
        persistence_frames=persistence_frames,
    )
    reference = canonical_reference_contact_state(command, [0])[:, 0]
    selection = _canonical_contact_selection(
        device=env.device,
        entity_classes=entity_classes,
        body_groups=body_groups,
    )
    required = (reference > 0.5) & selection[None]
    required_count = required.sum(dim=(1, 2))
    matched_count = (required & (measured > 0.5)).sum(dim=(1, 2))
    reward = torch.where(
        required_count > 0,
        matched_count.float() / required_count.clamp_min(1).float(),
        torch.zeros_like(required_count, dtype=torch.float32),
    )
    if hasattr(command, "metrics"):
        command.metrics["canonical_required_contact/count"] = (
            required_count.float()
        )
        command.metrics["canonical_required_contact/matched_count"] = (
            matched_count.float()
        )
        command.metrics["canonical_required_contact/reward"] = reward
    return reward


def canonical_forbidden_contact_penalty(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    entity_classes: list[str] | tuple[str, ...] | None = None,
    body_groups: list[str] | tuple[str, ...] | None = None,
    active_sensor_name: str = "canonical_active_entity_contact_sensor",
    static_sensor_name: str = "canonical_static_context_contact_sensor",
    ground_sensor_name: str = "canonical_ground_contact_sensor",
    force_threshold_n: float = 1.0,
    force_release_threshold_n: float | None = None,
    persistence_frames: int = 1,
) -> torch.Tensor:
    """Return one violation unit per entity with forbidden body contact."""
    command = _command(env, command_name)
    measured = canonical_measured_contact_state(
        env,
        active_sensor_name=active_sensor_name,
        static_sensor_name=static_sensor_name,
        ground_sensor_name=ground_sensor_name,
        force_threshold_n=force_threshold_n,
        force_release_threshold_n=force_release_threshold_n,
        persistence_frames=persistence_frames,
    )
    reference = canonical_reference_contact_state(command, [0])[:, 0]
    selection = _canonical_contact_selection(
        device=env.device,
        entity_classes=entity_classes,
        body_groups=body_groups,
    )
    forbidden = (reference < -0.5) & selection[None]
    violation = forbidden & (measured > 0.5)
    penalty = violation.any(dim=-1).float().sum(dim=-1)
    if hasattr(command, "metrics"):
        command.metrics["canonical_forbidden_contact/count"] = (
            forbidden.sum(dim=(1, 2)).float()
        )
        command.metrics["canonical_forbidden_contact/violating_entities"] = (
            penalty
        )
    return penalty


class CanonicalRequiredContactFailure(ManagerTermBase):
    """Terminate after a required canonical contact is missed for a grace window."""

    def __init__(self, cfg, env):
        super().__init__(cfg=cfg, env=env)
        self.command_name = cfg.params.get("command_name", "motion")
        self.entity_classes = cfg.params.get(
            "entity_classes", ["active_entity"]
        )
        self.body_groups = cfg.params.get(
            "body_groups", ["left_hand", "right_hand"]
        )
        self.allowed_roles = cfg.params.get("allowed_roles")
        self.active_sensor_name = cfg.params.get(
            "active_sensor_name", "canonical_active_entity_contact_sensor"
        )
        self.static_sensor_name = cfg.params.get(
            "static_sensor_name", "canonical_static_context_contact_sensor"
        )
        self.ground_sensor_name = cfg.params.get(
            "ground_sensor_name", "canonical_ground_contact_sensor"
        )
        self.force_threshold_n = float(
            cfg.params.get("force_threshold_n", 1.0)
        )
        self.force_release_threshold_n = float(
            cfg.params.get(
                "force_release_threshold_n", self.force_threshold_n
            )
        )
        self.persistence_frames = int(
            cfg.params.get("persistence_frames", 1)
        )
        self.grace_frames = int(cfg.params.get("grace_frames", 20))
        self._missed_frames = torch.zeros(
            env.num_envs,
            len(CONTACT_ENTITY_CLASSES),
            len(CONTACT_BODY_GROUPS),
            dtype=torch.long,
            device=env.device,
        )

    def __call__(
        self,
        env,
        command_name: str = "motion",
        entity_classes: list[str] | tuple[str, ...] | None = None,
        body_groups: list[str] | tuple[str, ...] | None = None,
        active_sensor_name: str = "canonical_active_entity_contact_sensor",
        static_sensor_name: str = "canonical_static_context_contact_sensor",
        ground_sensor_name: str = "canonical_ground_contact_sensor",
        force_threshold_n: float = 1.0,
        force_release_threshold_n: float | None = None,
        persistence_frames: int = 1,
        grace_frames: int = 20,
        allowed_roles: list[str] | str | None = None,
    ) -> torch.Tensor:
        del (
            command_name,
            entity_classes,
            body_groups,
            active_sensor_name,
            static_sensor_name,
            ground_sensor_name,
            force_threshold_n,
            force_release_threshold_n,
            persistence_frames,
            grace_frames,
            allowed_roles,
        )
        command = _command(env, self.command_name)
        measured = canonical_measured_contact_state(
            env,
            active_sensor_name=self.active_sensor_name,
            static_sensor_name=self.static_sensor_name,
            ground_sensor_name=self.ground_sensor_name,
            force_threshold_n=self.force_threshold_n,
            force_release_threshold_n=self.force_release_threshold_n,
            persistence_frames=self.persistence_frames,
        )
        reference = canonical_reference_contact_state(command, [0])[:, 0]
        selection = _canonical_contact_selection(
            device=env.device,
            entity_classes=self.entity_classes,
            body_groups=self.body_groups,
        )
        missed = (
            (reference > 0.5) & (measured < 0.5) & selection[None]
        )
        if self.allowed_roles:
            role_eligible = utils.get_target_object_role_mask(
                env,
                command,
                self.allowed_roles,
            )
            missed = missed & role_eligible[:, None, None]
        self._missed_frames = torch.where(
            missed,
            self._missed_frames + 1,
            torch.zeros_like(self._missed_frames),
        )
        return (self._missed_frames > self.grace_frames).any(dim=(1, 2))

    def reset(self, env_ids):
        self._missed_frames[env_ids] = 0


def _hand_force_tensor(
    env: ManagerBasedEnv,
    command,
    *,
    hand: str,
    fallback_sensor_name: str,
    links_per_hand: int,
) -> torch.Tensor:
    force = utils.get_target_object_contact_forces(
        env,
        command,
        hand=hand,
        fallback_sensor_name=fallback_sensor_name,
    )
    if force is None:
        return torch.zeros(env.num_envs, links_per_hand, 3, device=env.device)
    if force.ndim == 4:
        force = force.sum(dim=1)
    if force.ndim != 3 or force.shape[-1] != 3:
        raise ValueError(f"Unexpected {hand} contact force shape {tuple(force.shape)}")
    if force.shape[1] > links_per_hand:
        force = force[:, :links_per_hand]
    elif force.shape[1] < links_per_hand:
        padding = torch.zeros(
            env.num_envs,
            links_per_hand - force.shape[1],
            3,
            dtype=force.dtype,
            device=force.device,
        )
        force = torch.cat((force, padding), dim=1)
    return force


def _bilateral_contact_state(
    env: ManagerBasedEnv,
    command,
    *,
    left_sensor_name: str,
    right_sensor_name: str,
    links_per_hand: int,
    force_threshold: float,
    force_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_force = _hand_force_tensor(
        env,
        command,
        hand="left_hand",
        fallback_sensor_name=left_sensor_name,
        links_per_hand=links_per_hand,
    )
    right_force = _hand_force_tensor(
        env,
        command,
        hand="right_hand",
        fallback_sensor_name=right_sensor_name,
        links_per_hand=links_per_hand,
    )
    force = torch.stack(
        (
            torch.linalg.vector_norm(left_force, dim=-1).sum(dim=-1),
            torch.linalg.vector_norm(right_force, dim=-1).sum(dim=-1),
        ),
        dim=-1,
    )
    temperature = max(float(force_temperature), 1.0e-6)
    raw_probability = torch.sigmoid((force - float(force_threshold)) / temperature)
    zero_force_baseline = 1.0 / (1.0 + math.exp(float(force_threshold) / temperature))
    probability = torch.clamp(
        (raw_probability - zero_force_baseline) / max(1.0 - zero_force_baseline, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    return torch.cat((left_force, right_force), dim=1), probability


def rp2_source_bilateral_contact_force(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    left_sensor_name: str = "object_to_left_hand_contact_sensor",
    right_sensor_name: str = "object_to_hand_contact_sensor",
    links_per_hand: int = 4,
    force_scale: float = 10.0,
) -> torch.Tensor:
    """Symmetric left/right object contact-force observation."""
    command = _command(env, command_name)
    force, _ = _bilateral_contact_state(
        env,
        command,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        links_per_hand=links_per_hand,
        force_threshold=0.1,
        force_temperature=0.05,
    )
    return torch.tanh(force / max(float(force_scale), 1.0e-6)).reshape(env.num_envs, -1)


def rp2_source_contact_target_sparse(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    offsets: list[int] = [1, 10],
) -> torch.Tensor:
    command = _command(env, command_name)
    return _source_contact_probability_channel(command, offsets).reshape(
        env.num_envs, -1
    )


def rp2_source_contact_residual_sparse(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    offsets: list[int] = [1, 10],
    left_sensor_name: str = "object_to_left_hand_contact_sensor",
    right_sensor_name: str = "object_to_hand_contact_sensor",
    links_per_hand: int = 4,
    force_threshold: float = 0.1,
    force_temperature: float = 0.05,
) -> torch.Tensor:
    """Sparse source contact probability minus current bilateral sensed contact."""
    command = _command(env, command_name)
    target = _source_contact_probability_channel(command, offsets)
    _, current = _bilateral_contact_state(
        env,
        command,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        links_per_hand=links_per_hand,
        force_threshold=force_threshold,
        force_temperature=force_temperature,
    )
    return (target - current[:, None]).reshape(env.num_envs, -1)


def _sparse_object_reference(command, offsets: list[int] | tuple[int, ...]):
    motion_ids, steps = _future_steps(command, offsets)
    pos = command.motion_lib.get_object_root_pos(motion_ids.reshape(-1), steps.reshape(-1))
    quat = command.motion_lib.get_object_root_quat(motion_ids.reshape(-1), steps.reshape(-1))
    pos = pos.reshape(command.num_envs, len(offsets), -1, 3)
    quat = quat.reshape(command.num_envs, len(offsets), -1, 4)
    if pos.shape[2] == 1:
        return pos[:, :, 0], quat[:, :, 0]
    return (
        _gather_target_object_channel(command, pos),
        _gather_target_object_channel(command, quat),
    )


def rp2_source_object_pos_residual_sparse(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    offsets: list[int] = [1, 10],
) -> torch.Tensor:
    """Sparse raw-object future position residual in the current G1 heading frame."""
    command = _command(env, command_name)
    current_pos, _, _, _, _ = utils.get_target_object_current_state(env, command)
    target_pos, _ = _sparse_object_reference(command, offsets)
    target_pos = target_pos + env.scene.env_origins[:, None, :]
    delta_w = target_pos - current_pos[:, None, :]
    heading_inv = quat_inv(torch_transform.get_heading_q(command.robot_anchor_quat_w))
    heading_inv = heading_inv[:, None, :].expand(-1, len(offsets), -1)
    delta_h = quat_apply(heading_inv.reshape(-1, 4), delta_w.reshape(-1, 3))
    return delta_h.reshape(env.num_envs, -1)


def rp2_source_object_ori_residual_sparse(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    offsets: list[int] = [1, 10],
) -> torch.Tensor:
    """Sparse raw-object future orientation residual in continuous 6D form."""
    command = _command(env, command_name)
    _, current_quat, _, _, _ = utils.get_target_object_current_state(env, command)
    _, target_quat = _sparse_object_reference(command, offsets)
    current_inv = quat_inv(current_quat)[:, None, :].expand(-1, len(offsets), -1)
    relative = quat_mul(current_inv.reshape(-1, 4), target_quat.reshape(-1, 4))
    rotation = matrix_from_quat(relative)
    return rotation[..., :2].reshape(env.num_envs, -1)


def rp2_source_relation_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    robot_body_names: list[str] = list(DEFAULT_ROBOT_BODY_NAMES),
    tolerance: float = 0.04,
    sigma: float = 0.20,
    kernel: str = "cauchy",
    body_weights: list[float] | None = None,
    reference_distance_weighting: bool = False,
    reference_distance_scale: float = 0.40,
    reference_distance_floor: float = 0.15,
    express_target_in_current_robot_heading: bool = False,
    relation_geometry: str = "center",
) -> torch.Tensor:
    """Track every mapped source-SOMA link-to-object edge at the current frame."""
    command = _command(env, command_name)
    target = _source_relation_channel(command, [0], relation_geometry)
    if express_target_in_current_robot_heading:
        target = _source_relation_in_current_robot_heading(command, target, [0])
    target = target[:, 0]
    mask = _source_relation_mask_channel(command, [0], relation_geometry)[:, 0]
    current = _sim_relation_for_offsets(
        env,
        command,
        robot_body_names,
        [0],
        relation_geometry,
    )[:, 0]
    error = torch.linalg.vector_norm(target - current, dim=-1)
    score = tolerance_tracking_score(
        error,
        tolerance=tolerance,
        sigma=sigma,
        kernel=kernel,
    )
    if body_weights is None:
        weights = torch.ones(len(robot_body_names), dtype=score.dtype, device=score.device)
    else:
        if len(body_weights) != len(robot_body_names):
            raise ValueError("body_weights must match robot_body_names")
        weights = torch.as_tensor(body_weights, dtype=score.dtype, device=score.device)
    reference_distance = torch.linalg.vector_norm(target, dim=-1)
    proximity_weight = _reference_distance_weight(
        reference_distance,
        enabled=reference_distance_weighting,
        scale=reference_distance_scale,
        floor=reference_distance_floor,
    )
    active_weight = mask * weights[None, :] * proximity_weight
    denominator = active_weight.sum(dim=-1).clamp_min(1.0e-6)
    reward = (score * active_weight).sum(dim=-1) / denominator
    command.metrics["rp2_source_relation/error_m"] = (error * active_weight).sum(
        dim=-1
    ) / denominator
    command.metrics["rp2_source_relation/reward"] = reward
    command.metrics["rp2_source_relation/active_links"] = mask.sum(dim=-1)
    valid_denominator = mask.sum(dim=-1).clamp_min(1.0)
    command.metrics["rp2_source_relation/reference_distance_m"] = (
        reference_distance * mask
    ).sum(dim=-1) / valid_denominator
    command.metrics["rp2_source_relation/proximity_weight"] = (
        proximity_weight * mask
    ).sum(dim=-1) / valid_denominator
    command.metrics["rp2_source_relation/uses_surface_sdf"] = torch.full_like(
        reward,
        float(_validate_relation_geometry(relation_geometry) == "surface_sdf"),
    )
    return reward


def rp2_reference_aware_undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    active_support_sensor_cfg: SceneEntityCfg,
    command_name: str = "motion",
    threshold: float = 1.0,
    active_support_contact_threshold: float = 1.0,
    support_role_names: list[str] = ["contact_semantics:body_support"],
    support_body_names: list[str] = [
        "pelvis",
        "torso_link",
        "left_hip_roll_link",
        "left_knee_link",
        "right_hip_roll_link",
        "right_knee_link",
    ],
    reference_proximity_threshold: float = 0.65,
    reference_proximity_temperature: float = 0.05,
    relation_geometry: str = "center",
) -> torch.Tensor:
    """Penalize non-hand contact while preserving expected body support.

    This keeps the original generic contact penalty for every selected robot
    link. A support link receives a smooth exemption only when the active
    object has body-support semantics, that exact link contacts the active
    object, and its source-SOMA relation is near the object. Contacts with the
    ground or unrelated scene geometry therefore remain penalized.
    """
    if threshold < 0.0 or active_support_contact_threshold < 0.0:
        raise ValueError("contact thresholds must be non-negative")

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    selected_contact = (
        torch.linalg.vector_norm(
            net_forces[:, :, sensor_cfg.body_ids], dim=-1
        ).amax(dim=1)
        > float(threshold)
    )
    selected_body_names = [
        contact_sensor.body_names[int(body_id)] for body_id in sensor_cfg.body_ids
    ]
    selected_support_columns = [
        selected_body_names.index(name)
        for name in support_body_names
        if name in selected_body_names
    ]
    selected_support_names = [
        name for name in support_body_names if name in selected_body_names
    ]
    if not selected_support_columns:
        return selected_contact.float().sum(dim=-1)

    support_sensor = env.scene.sensors[active_support_sensor_cfg.name]
    support_force = torch.linalg.vector_norm(
        support_sensor.data.force_matrix_w, dim=-1
    )
    if support_force.ndim == 3:
        support_force = support_force.amax(dim=1)
    elif support_force.ndim != 2:
        raise ValueError(
            "active support contact force must resolve to [env, body], got "
            f"{tuple(support_force.shape)}"
        )
    if support_force.shape[-1] != len(support_body_names):
        raise ValueError(
            "active support contact sensor/body configuration mismatch: "
            f"forces={support_force.shape[-1]}, bodies={len(support_body_names)}"
        )
    support_name_to_column = {
        name: index for index, name in enumerate(support_body_names)
    }
    support_force = support_force[:, [
        support_name_to_column[name] for name in selected_support_names
    ]]
    active_support_contact = support_force > float(active_support_contact_threshold)

    command = _command(env, command_name)
    target = _source_relation_channel(command, [0], relation_geometry)[:, 0]
    relation_valid = _source_relation_mask_channel(
        command, [0], relation_geometry
    )[:, 0].bool()
    relation_index = {
        body_name: index
        for index, body_name in enumerate(DEFAULT_ROBOT_BODY_NAMES)
    }
    missing_relations = [
        name for name in selected_support_names if name not in relation_index
    ]
    if missing_relations:
        raise KeyError(
            "support bodies are absent from the source relation mapping: "
            f"{missing_relations}"
        )
    relation_columns = [relation_index[name] for name in selected_support_names]
    reference_distance = torch.linalg.vector_norm(
        target[:, relation_columns], dim=-1
    )
    relation_valid = relation_valid[:, relation_columns]
    role_eligible = utils.get_target_object_role_mask(
        env,
        command,
        support_role_names,
    )

    support_contact = selected_contact[:, selected_support_columns]
    support_penalty, exemption = reference_aware_support_contact_penalty(
        support_contact,
        active_support_contact,
        reference_distance,
        relation_valid,
        role_eligible,
        proximity_threshold=reference_proximity_threshold,
        proximity_temperature=reference_proximity_temperature,
    )
    non_support_mask = torch.ones(
        selected_contact.shape[-1], dtype=torch.bool, device=env.device
    )
    non_support_mask[selected_support_columns] = False
    non_support_penalty = selected_contact[:, non_support_mask].float().sum(dim=-1)
    penalty = non_support_penalty + support_penalty

    command.metrics["rp2_body_contact/raw_count"] = selected_contact.float().sum(dim=-1)
    command.metrics["rp2_body_contact/support_contact_count"] = (
        support_contact.float().sum(dim=-1)
    )
    command.metrics["rp2_body_contact/exempted_count"] = (
        support_contact.float() * exemption
    ).sum(dim=-1)
    command.metrics["rp2_body_contact/penalty_count"] = penalty
    return penalty


def rp2_source_object_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    w_op: float = 0.5,
    w_or: float = 0.3,
    position_tolerance: float = 0.02,
    position_sigma: float = 0.05,
    orientation_tolerance: float = 0.0872664626,
    orientation_sigma: float = 0.2617993878,
    kernel: str = "cauchy",
) -> torch.Tensor:
    """Track object pose under a soft either-hand source-contact reference gate.

    Unlike the legacy OmniGrasp term, activation depends only on the bilateral
    source reference. It therefore neither assumes a right-hand interaction nor
    removes the learning signal before simulated contact has been established.
    """
    command = _command(env, command_name)
    (
        current_position,
        current_orientation,
        _,
        _,
        _,
    ) = utils.get_target_object_current_state(env, command)
    position_error = torch.linalg.vector_norm(
        command.target_object_root_pos - current_position,
        dim=-1,
    )
    orientation_error = quat_error_magnitude(
        command.target_object_root_quat,
        current_orientation,
    )
    position_score = tolerance_tracking_score(
        position_error,
        tolerance=position_tolerance,
        sigma=position_sigma,
        kernel=kernel,
    )
    orientation_score = tolerance_tracking_score(
        orientation_error,
        tolerance=orientation_tolerance,
        sigma=orientation_sigma,
        kernel=kernel,
    )
    reference = _source_contact_probability_channel(command, [0])[:, 0]
    reference_gate = bilateral_reference_contact_gate(reference)
    ungated_reward = float(w_op) * position_score + float(w_or) * orientation_score
    reward = ungated_reward * reference_gate

    command.metrics["rp2_source_object/position_error_m"] = position_error
    command.metrics["rp2_source_object/orientation_error_rad"] = orientation_error
    command.metrics["rp2_source_object/reference_contact_gate"] = reference_gate
    command.metrics["rp2_source_object/tracking_reward_ungated"] = ungated_reward
    command.metrics["rp2_source_object/tracking_reward"] = reward
    return reward


def _reference_distance_weight(
    reference_distance: torch.Tensor,
    *,
    enabled: bool,
    scale: float,
    floor: float,
) -> torch.Tensor:
    if not enabled:
        return torch.ones_like(reference_distance)
    if scale <= 0.0:
        raise ValueError("reference distance scale must be positive")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("reference distance floor must be in [0, 1]")
    normalized_distance = reference_distance / float(scale)
    return float(floor) + (1.0 - float(floor)) / (
        1.0 + normalized_distance.square()
    )


def rp2_source_bilateral_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_to_left_hand_contact_sensor"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_to_hand_contact_sensor"),
    links_per_hand: int = 4,
    force_threshold: float = 0.1,
    force_temperature: float = 0.05,
    false_contact_penalty: float = 0.1,
) -> torch.Tensor:
    """Symmetric smooth source-contact acquisition reward for both hands."""
    command = _command(env, command_name)
    reference = _source_contact_probability_channel(command, [0])[:, 0]
    _, simulated = _bilateral_contact_state(
        env,
        command,
        left_sensor_name=left_sensor_cfg.name,
        right_sensor_name=right_sensor_cfg.name,
        links_per_hand=links_per_hand,
        force_threshold=force_threshold,
        force_temperature=force_temperature,
    )
    per_hand = reference * simulated - float(false_contact_penalty) * (1.0 - reference) * simulated
    for index, side in enumerate(("left", "right")):
        command.metrics[f"rp2_source_contact/reference_{side}"] = reference[:, index]
        command.metrics[f"rp2_source_contact/simulated_{side}"] = simulated[:, index]
        command.metrics[f"rp2_source_contact/reward_{side}"] = per_hand[:, index]
    return per_hand.mean(dim=-1)


def build_source_only_reset_state(command, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build source-aligned reset state with optional retarget-pose bootstrap."""
    motion_ids = command.motion_ids[env_ids]
    steps = command.motion_start_time_steps[env_ids] + command.time_steps[env_ids]
    source_pos = command.motion_lib.get_rp2_source_root_pos(motion_ids, steps)
    heading_xy = command.motion_lib.get_rp2_source_root_heading_xy(motion_ids, steps)
    if source_pos is None or heading_xy is None:
        raise RuntimeError("source_only_reset requires RP2 source root position and heading channels")
    source_pos = source_pos.clone()
    source_pos += command._env.scene.env_origins[env_ids]  # noqa: SLF001
    minimum_height = float(getattr(command.cfg, "source_reset_min_root_height", 0.72))
    use_source_z = bool(getattr(command.cfg, "source_reset_use_source_z", True))
    floor_height = command._env.scene.env_origins[env_ids, 2] + minimum_height  # noqa: SLF001
    if use_source_z:
        source_pos[:, 2] = torch.maximum(source_pos[:, 2], floor_height)
    else:
        source_pos[:, 2] = floor_height

    yaw = torch.atan2(heading_xy[:, 1], heading_xy[:, 0])
    root_quat = torch.zeros(len(env_ids), 4, dtype=source_pos.dtype, device=source_pos.device)
    root_quat[:, 0] = torch.cos(0.5 * yaw)
    root_quat[:, 3] = torch.sin(0.5 * yaw)
    joint_pos = command.robot.data.default_joint_pos[env_ids].clone()
    state = {
        "root_pos": source_pos,
        "root_quat": root_quat,
        "root_lin_vel": torch.zeros_like(source_pos),
        "root_ang_vel": torch.zeros_like(source_pos),
        "joint_pos": joint_pos,
        "joint_vel": torch.zeros_like(joint_pos),
    }

    get_retarget = getattr(command.motion_lib, "get_rp2_retarget_initialization", None)
    retarget = get_retarget(motion_ids, steps) if get_retarget is not None else None
    if retarget is None:
        return state
    get_retarget_valid = getattr(
        command.motion_lib,
        "get_rp2_retarget_initialization_valid",
        None,
    )
    retarget_valid = (
        get_retarget_valid(motion_ids, steps)
        if get_retarget_valid is not None
        else None
    )
    if retarget_valid is None:
        retarget_valid = torch.ones(
            len(env_ids),
            dtype=torch.bool,
            device=source_pos.device,
        )
    else:
        retarget_valid = retarget_valid.to(
            device=source_pos.device,
            dtype=torch.bool,
        )

    def set_reset_metric(name: str, value: torch.Tensor) -> None:
        metric = command.metrics.get(name)
        if not isinstance(metric, torch.Tensor) or metric.shape != (command.num_envs,):
            metric = torch.zeros(command.num_envs, dtype=source_pos.dtype, device=source_pos.device)
        metric[env_ids] = value
        command.metrics[name] = metric

    total_steps = max(int(getattr(command.cfg, "source_reset_curriculum_steps", 5000)), 1)
    current_step = int(getattr(command._env, "common_step_counter", 0))  # noqa: SLF001
    progress = min(max(current_step / total_steps, 0.0), 1.0)
    probability_start = float(
        getattr(command.cfg, "source_reset_retarget_probability_start", 0.0)
    )
    probability_end = float(
        getattr(command.cfg, "source_reset_retarget_probability_end", probability_start)
    )
    probability = probability_start + progress * (probability_end - probability_start)
    probability = min(max(probability, 0.0), 1.0)
    use_retarget = (
        torch.rand(len(env_ids), device=source_pos.device) < probability
    ) & retarget_valid

    (
        init_root_pos,
        init_root_quat,
        init_root_lin_vel,
        init_root_ang_vel,
        init_dof,
        init_dof_vel,
    ) = retarget
    if command.has_dof_mismatch:
        default_body = joint_pos[:, command.body_joint_indices]
    else:
        default_body = joint_pos
    if bool(getattr(command.cfg, "source_reset_force_retarget_nonstanding", False)):
        posture_rms = torch.sqrt(torch.mean((init_dof - default_body).square(), dim=-1))
        use_retarget |= retarget_valid & (
            posture_rms
            > float(
                getattr(
                    command.cfg,
                    "source_reset_nonstanding_joint_rms",
                    0.35,
                )
            )
        )

    if not torch.any(use_retarget):
        set_reset_metric("rp2_source_reset/retarget_fraction", use_retarget.float())
        return state

    init_root_pos = init_root_pos + command._env.scene.env_origins[env_ids]  # noqa: SLF001
    # Preserve the authoritative source XY while retaining retargeted root height,
    # tilt, and motion. The aligned pack differs by only millimeters in XY.
    retarget_root_pos = init_root_pos.clone()
    retarget_root_pos[:, :2] = source_pos[:, :2]
    retarget_floor = command._env.scene.env_origins[env_ids, 2] + float(  # noqa: SLF001
        getattr(command.cfg, "source_reset_retarget_min_root_height", 0.30)
    )
    retarget_root_pos[:, 2] = torch.maximum(retarget_root_pos[:, 2], retarget_floor)
    state["root_pos"][use_retarget] = retarget_root_pos[use_retarget]
    state["root_quat"][use_retarget] = init_root_quat[use_retarget]
    state["root_lin_vel"][use_retarget] = init_root_lin_vel[use_retarget]
    state["root_ang_vel"][use_retarget] = init_root_ang_vel[use_retarget]
    if command.has_dof_mismatch:
        selected_joint_pos = state["joint_pos"][use_retarget]
        selected_joint_vel = state["joint_vel"][use_retarget]
        selected_joint_pos[:, command.body_joint_indices] = init_dof[use_retarget]
        selected_joint_vel[:, command.body_joint_indices] = init_dof_vel[use_retarget]
        state["joint_pos"][use_retarget] = selected_joint_pos
        state["joint_vel"][use_retarget] = selected_joint_vel
    else:
        state["joint_pos"][use_retarget] = init_dof[use_retarget]
        state["joint_vel"][use_retarget] = init_dof_vel[use_retarget]
    if bool(getattr(command.cfg, "source_reset_use_reference_hand_action", False)):
        primitive_map = (
            getattr(command._env.cfg, "config", {})  # noqa: SLF001
            .get("finger_primitive", {})
            .get("primitive_action_map", {})
        )
        for side in ("left", "right"):
            side_cfg = primitive_map.get(side)
            hand_action = command.motion_lib.get_hand_action(
                motion_ids, steps, hand=f"{side}_hand"
            )
            if not side_cfg or hand_action is None:
                continue
            joint_indices = [
                command.robot.joint_names.index(name) for name in side_cfg.get("dof_names", [])
            ]
            open_pos = torch.as_tensor(
                side_cfg.get("pos_0", []), dtype=source_pos.dtype, device=source_pos.device
            )
            closed_pos = torch.as_tensor(
                side_cfg.get("pos_1", []), dtype=source_pos.dtype, device=source_pos.device
            )
            if len(joint_indices) != len(open_pos) or len(open_pos) != len(closed_pos):
                raise ValueError(f"Invalid {side} finger primitive reset configuration")
            target = torch.where(
                (hand_action >= 0.0)[:, None],
                closed_pos[None, :],
                open_pos[None, :],
            )
            selected_joint_pos = state["joint_pos"][use_retarget]
            selected_joint_pos[:, joint_indices] = target[use_retarget]
            state["joint_pos"][use_retarget] = selected_joint_pos
    set_reset_metric("rp2_source_reset/retarget_fraction", use_retarget.float())
    set_reset_metric(
        "rp2_source_reset/retarget_probability",
        torch.full(
            (len(env_ids),), probability, dtype=source_pos.dtype, device=source_pos.device
        ),
    )
    return state


def rp2_source_relation_deviation(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    robot_body_names: list[str] = list(DEFAULT_ROBOT_BODY_NAMES),
    threshold: float = 1.25,
    min_episode_steps: int = 25,
    reference_distance_scale: float = 0.40,
    reference_distance_floor: float = 0.15,
    relation_geometry: str = "center",
) -> torch.Tensor:
    """Terminate when proximity-weighted source relation error becomes excessive."""
    command = _command(env, command_name)
    target = _source_relation_channel(command, [0], relation_geometry)[:, 0]
    mask = _source_relation_mask_channel(command, [0], relation_geometry)[:, 0]
    current = _sim_relation_for_offsets(
        env,
        command,
        robot_body_names,
        [0],
        relation_geometry,
    )[:, 0]
    error = torch.linalg.vector_norm(target - current, dim=-1)
    reference_distance = torch.linalg.vector_norm(target, dim=-1)
    weight = mask * _reference_distance_weight(
        reference_distance,
        enabled=True,
        scale=reference_distance_scale,
        floor=reference_distance_floor,
    )
    weighted_error = (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0e-6)
    command.metrics["rp2_source_relation/termination_error_m"] = weighted_error
    command.metrics["rp2_source_relation/termination_threshold_m"] = torch.full_like(
        weighted_error, float(threshold)
    )
    grace_complete = command.time_steps >= int(min_episode_steps)
    return grace_complete & (weighted_error > float(threshold))


def rp2_source_fall(
    env: ManagerBasedRLEnv,
    min_pelvis_height: float = 0.30,
    max_tilt_radians: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on a low pelvis or excessive physical tilt, without a G1 reference."""
    robot = env.scene[asset_cfg.name]
    relative_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    world_down = torch.zeros(env.num_envs, 3, dtype=relative_height.dtype, device=env.device)
    world_down[:, 2] = -1.0
    local_down = quat_apply(quat_inv(robot.data.root_quat_w), world_down)
    upright_cosine = -local_down[:, 2]
    return (relative_height < float(min_pelvis_height)) | (
        upright_cosine < math.cos(float(max_tilt_radians))
    )


class RP2SourceContactFailure(ManagerTermBase):
    """Terminate sustained failure to realize an active bilateral source contact."""

    def __init__(self, cfg, env):
        super().__init__(cfg=cfg, env=env)
        self.command_name = str(cfg.params.get("command_name", "motion"))
        self.left_sensor_name = str(
            cfg.params.get("left_sensor_name", "object_to_left_hand_contact_sensor")
        )
        self.right_sensor_name = str(
            cfg.params.get("right_sensor_name", "object_to_hand_contact_sensor")
        )
        self.links_per_hand = int(cfg.params.get("links_per_hand", 4))
        self.force_threshold = float(cfg.params.get("force_threshold", 0.1))
        self.force_temperature = float(cfg.params.get("force_temperature", 0.05))
        self.reference_threshold = float(cfg.params.get("reference_threshold", 0.5))
        self.simulated_threshold = float(cfg.params.get("simulated_threshold", 0.25))
        self.grace_steps = int(cfg.params.get("grace_steps", 50))
        self._missing_contact_steps = torch.zeros(
            env.num_envs,
            2,
            dtype=torch.long,
            device=env.device,
        )

    def __call__(
        self,
        env,
        command_name: str = "motion",
        left_sensor_name: str = "object_to_left_hand_contact_sensor",
        right_sensor_name: str = "object_to_hand_contact_sensor",
        links_per_hand: int = 4,
        force_threshold: float = 0.1,
        force_temperature: float = 0.05,
        reference_threshold: float = 0.5,
        simulated_threshold: float = 0.25,
        grace_steps: int = 50,
    ) -> torch.Tensor:
        command = _command(env, self.command_name)
        reference = _source_contact_probability_channel(command, [0])[:, 0]
        _, simulated = _bilateral_contact_state(
            env,
            command,
            left_sensor_name=self.left_sensor_name,
            right_sensor_name=self.right_sensor_name,
            links_per_hand=self.links_per_hand,
            force_threshold=self.force_threshold,
            force_temperature=self.force_temperature,
        )
        self._missing_contact_steps, terminate, required = (
            update_bilateral_contact_failure_counter(
                reference,
                simulated,
                self._missing_contact_steps,
                reference_threshold=self.reference_threshold,
                simulated_threshold=self.simulated_threshold,
                grace_steps=self.grace_steps,
            )
        )
        for index, side in enumerate(("left", "right")):
            command.metrics[f"rp2_source_contact/required_{side}"] = required[:, index].float()
            command.metrics[f"rp2_source_contact/missing_steps_{side}"] = (
                self._missing_contact_steps[:, index].float()
            )
        return terminate

    def reset(self, env_ids):
        self._missing_contact_steps[env_ids] = 0
