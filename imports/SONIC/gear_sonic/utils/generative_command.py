"""Pure helpers for reference-free generative command execution."""

from __future__ import annotations

import torch


GOAL_SAMPLER_VERSION = "table_box_heading_reach_rejection_v1"


def _expand_rows(value: torch.Tensor, rows: int, width: int, name: str) -> torch.Tensor:
    if value.dim() == 1:
        if value.shape[0] != width:
            raise ValueError(f"{name} must have width {width}, got {tuple(value.shape)}")
        value = value.unsqueeze(0).expand(rows, -1)
    if value.shape != (rows, width):
        raise ValueError(f"{name} must have shape ({rows}, {width}), got {tuple(value.shape)}")
    return value


def _world_points_to_heading(
    root_state_w: torch.Tensor, points_w: torch.Tensor
) -> torch.Tensor:
    """Express world points in a yaw-only robot frame."""
    delta = points_w - root_state_w[:, :3]
    w, x, y, z = root_state_w[:, 3:7].unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = yaw.cos(), yaw.sin()
    return torch.stack(
        [
            c * delta[:, 0] + s * delta[:, 1],
            -s * delta[:, 0] + c * delta[:, 1],
            delta[:, 2],
        ],
        dim=-1,
    )


def _heading_offsets_to_world(
    root_state_w: torch.Tensor,
    origins_w: torch.Tensor,
    offsets_b: torch.Tensor,
) -> torch.Tensor:
    """Apply yaw-only heading offsets to per-row world origins."""
    w, x, y, z = root_state_w[:, 3:7].unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = yaw.cos(), yaw.sin()
    offset_w = torch.stack(
        [
            c * offsets_b[:, 0] - s * offsets_b[:, 1],
            s * offsets_b[:, 0] + c * offsets_b[:, 1],
            offsets_b[:, 2],
        ],
        dim=-1,
    )
    return origins_w + offset_w


def _quat_rotate_inverse_wxyz(quat_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse of normalized wxyz quaternions."""
    norms = quat_wxyz.norm(dim=-1, keepdim=True)
    if (norms <= 1.0e-8).any():
        raise ValueError("support quaternion contains a zero-norm row")
    quat = quat_wxyz / norms
    scalar = quat[:, :1]
    vector = -quat[:, 1:]
    twice_cross = 2.0 * torch.cross(vector, vectors, dim=-1)
    return vectors + scalar * twice_cross + torch.cross(vector, twice_cross, dim=-1)


def goal_feasibility(
    *,
    root_state_w: torch.Tensor,
    goal_w: torch.Tensor,
    support_pos_w: torch.Tensor,
    support_quat_w: torch.Tensor,
    support_size_xyz: torch.Tensor,
    support_margin: float,
    reach_forward_min: float,
    reach_forward_max: float,
    reach_lateral_max: float,
) -> dict[str, torch.Tensor]:
    """Check tabletop footprint and robot-heading reach for world goals.

    The support check uses the live support pose and the exact spawned cuboid
    size. ``support_margin`` is a center clearance from every XY edge and must
    include any desired object-footprint allowance.
    """
    if root_state_w.dim() != 2 or root_state_w.shape[1] != 7:
        raise ValueError(f"root_state_w must have shape (N, 7), got {tuple(root_state_w.shape)}")
    rows = root_state_w.shape[0]
    goal_w = _expand_rows(goal_w, rows, 3, "goal_w")
    support_pos_w = _expand_rows(support_pos_w, rows, 3, "support_pos_w")
    support_quat_w = _expand_rows(support_quat_w, rows, 4, "support_quat_w")
    support_size_xyz = _expand_rows(
        support_size_xyz, rows, 3, "support_size_xyz"
    )
    tensors = (root_state_w, goal_w, support_pos_w, support_quat_w, support_size_xyz)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("goal feasibility inputs must all be finite")
    if support_margin < 0:
        raise ValueError("support_margin must be non-negative")
    if reach_forward_min > reach_forward_max:
        raise ValueError("reach_forward_min must not exceed reach_forward_max")
    if reach_lateral_max <= 0:
        raise ValueError("reach_lateral_max must be positive")
    usable_half_extent = support_size_xyz[:, :2] * 0.5 - support_margin
    if (usable_half_extent <= 0).any():
        raise ValueError(
            "support_margin leaves no usable tabletop area for at least one environment"
        )

    support_local = _quat_rotate_inverse_wxyz(
        support_quat_w, goal_w - support_pos_w
    )
    support_ok = (support_local[:, :2].abs() <= usable_half_extent).all(dim=-1)
    reach_b = _world_points_to_heading(root_state_w, goal_w)
    reach_ok = (
        reach_b[:, 0].ge(reach_forward_min)
        & reach_b[:, 0].le(reach_forward_max)
        & reach_b[:, 1].abs().le(reach_lateral_max)
    )
    return {
        "feasible": support_ok & reach_ok,
        "support_ok": support_ok,
        "reach_ok": reach_ok,
        "goal_support_local": support_local,
        "goal_heading_b": reach_b,
    }


def sample_feasible_goal_offsets(
    *,
    root_state_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    support_pos_w: torch.Tensor,
    support_quat_w: torch.Tensor,
    support_size_xyz: torch.Tensor,
    generator: torch.Generator,
    iteration: int,
    place_fraction_max: float,
    curriculum_iterations: int,
    min_radius: float,
    max_radius: float,
    lift_min: float,
    lift_max: float,
    support_margin: float,
    reach_forward_min: float,
    reach_forward_max: float,
    reach_lateral_max: float,
    max_attempts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int | float]]:
    """Rejection-sample object-relative goals inside support and reach bounds."""
    if root_state_w.dim() != 2 or root_state_w.shape[1] != 7:
        raise ValueError("root_state_w must have shape (N, 7)")
    rows = root_state_w.shape[0]
    object_pos_w = _expand_rows(object_pos_w, rows, 3, "object_pos_w")
    support_pos_w = _expand_rows(support_pos_w, rows, 3, "support_pos_w")
    support_quat_w = _expand_rows(support_quat_w, rows, 4, "support_quat_w")
    support_size_xyz = _expand_rows(
        support_size_xyz, rows, 3, "support_size_xyz"
    )
    if rows == 0:
        empty = object_pos_w.new_empty((0, 3))
        return empty, object_pos_w.new_empty((0,), dtype=torch.bool), empty, {
            "requested": 0,
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_support_count": 0,
            "rejected_reach_count": 0,
            "feasible_fraction": 1.0,
        }
    if not (0.0 <= place_fraction_max <= 1.0):
        raise ValueError("place_fraction_max must be in [0, 1]")
    if min_radius < 0 or max_radius < min_radius:
        raise ValueError("goal radius bounds are invalid")
    if lift_min < 0 or lift_max < lift_min:
        raise ValueError("lift bounds are invalid")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    device = root_state_w.device
    dtype = root_state_w.dtype
    ramp = min(1.0, iteration / max(curriculum_iterations, 1))
    place_fraction = place_fraction_max * ramp
    is_place = torch.rand(rows, device=device, generator=generator) < place_fraction
    offsets = torch.empty((rows, 3), device=device, dtype=dtype)
    goals = torch.empty_like(offsets)
    pending = torch.ones(rows, dtype=torch.bool, device=device)
    candidate_count = 0
    rejected_support_count = 0
    rejected_reach_count = 0

    for _attempt in range(max_attempts):
        env_ids = pending.nonzero(as_tuple=True)[0]
        if env_ids.numel() == 0:
            break
        count = int(env_ids.numel())
        angle = torch.rand(count, device=device, dtype=dtype, generator=generator) * (
            2.0 * torch.pi
        )
        radius = min_radius + (max_radius - min_radius) * torch.rand(
            count, device=device, dtype=dtype, generator=generator
        )
        lift_z = lift_min + (lift_max - lift_min) * torch.rand(
            count, device=device, dtype=dtype, generator=generator
        )
        candidate_offsets = torch.stack(
            [
                radius * angle.cos(),
                radius * angle.sin(),
                torch.where(is_place[env_ids], torch.zeros_like(lift_z), lift_z),
            ],
            dim=-1,
        )
        candidate_goals = _heading_offsets_to_world(
            root_state_w[env_ids], object_pos_w[env_ids], candidate_offsets
        )
        checked = goal_feasibility(
            root_state_w=root_state_w[env_ids],
            goal_w=candidate_goals,
            support_pos_w=support_pos_w[env_ids],
            support_quat_w=support_quat_w[env_ids],
            support_size_xyz=support_size_xyz[env_ids],
            support_margin=support_margin,
            reach_forward_min=reach_forward_min,
            reach_forward_max=reach_forward_max,
            reach_lateral_max=reach_lateral_max,
        )
        accepted = checked["feasible"]
        accepted_ids = env_ids[accepted]
        offsets[accepted_ids] = candidate_offsets[accepted]
        goals[accepted_ids] = candidate_goals[accepted]
        pending[accepted_ids] = False
        candidate_count += count
        rejected_support_count += int((~checked["support_ok"]).sum())
        # Count rejection causes exclusively so their sum equals rejected candidates.
        rejected_reach_count += int(
            (checked["support_ok"] & ~checked["reach_ok"]).sum()
        )

    if pending.any():
        failed = pending.nonzero(as_tuple=True)[0].tolist()
        accepted_count = rows - int(pending.sum())
        raise RuntimeError(
            "failed to sample feasible goals after "
            f"{max_attempts} attempts for environment rows {failed[:16]}; "
            f"accepted={accepted_count}/{rows}, candidates={candidate_count}, "
            f"rejected_support={rejected_support_count}, "
            f"rejected_reach={rejected_reach_count}, "
            f"feasible_fraction={accepted_count / candidate_count:.6f}"
        )
    return offsets, is_place, goals, {
        "requested": rows,
        "candidate_count": candidate_count,
        "accepted_count": rows,
        "rejected_support_count": rejected_support_count,
        "rejected_reach_count": rejected_reach_count,
        "feasible_fraction": rows / candidate_count,
    }


def advance_motion_time_steps(
    time_steps: torch.Tensor,
    motion_start_time_steps: torch.Tensor,
    motion_lengths: torch.Tensor,
    *,
    resample_on_motion_end: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance reference cursors and return motion-end resample environment ids.

    Tracking policies historically resample a command at the end of its source
    clip. That resample also writes the reference robot/object state into the
    simulator. A reference-free generative rollout must instead freeze its
    otherwise-unused reference cursor at the last valid frame.
    """
    next_steps = time_steps + 1
    expired = next_steps + motion_start_time_steps >= motion_lengths
    if resample_on_motion_end:
        return next_steps, expired.nonzero(as_tuple=True)[0]

    last_valid_relative_step = (
        motion_lengths - motion_start_time_steps - 1
    ).clamp_min(0)
    next_steps = torch.minimum(next_steps, last_valid_relative_step)
    return next_steps, torch.empty(
        0, dtype=torch.long, device=time_steps.device
    )


def assert_no_unreset_command_transition(
    *,
    before_motion_ids: torch.Tensor,
    before_start_steps: torch.Tensor,
    before_time_steps: torch.Tensor,
    after_motion_ids: torch.Tensor,
    after_start_steps: torch.Tensor,
    after_time_steps: torch.Tensor,
    dones: torch.Tensor,
) -> None:
    """Reject command resampling or cursor jumps outside true episode resets.

    Done rows may legitimately receive a new motion during the environment
    reset performed inside ``env.step``. Every other row must retain its
    motion/start identity and may only advance one frame or remain clamped at
    a clip boundary.
    """
    tensors = {
        "before_motion_ids": before_motion_ids,
        "before_start_steps": before_start_steps,
        "before_time_steps": before_time_steps,
        "after_motion_ids": after_motion_ids,
        "after_start_steps": after_start_steps,
        "after_time_steps": after_time_steps,
        "dones": dones,
    }
    shape = before_motion_ids.shape
    if any(value.shape != shape for value in tensors.values()):
        shapes = {name: tuple(value.shape) for name, value in tensors.items()}
        raise ValueError(f"command-transition tensors must share shape; got {shapes}")
    active = ~dones.bool()
    invalid = active & (
        (after_motion_ids != before_motion_ids)
        | (after_start_steps != before_start_steps)
        | (after_time_steps < before_time_steps)
        | (after_time_steps > before_time_steps + 1)
    )
    if invalid.any():
        rows = invalid.nonzero(as_tuple=True)[0]
        sample = rows[:16].tolist()
        raise RuntimeError(
            "generative command changed without an episode reset; "
            f"rows={sample} count={int(rows.numel())}"
        )
