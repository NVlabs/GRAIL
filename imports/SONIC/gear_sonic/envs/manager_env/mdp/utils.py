"""
Utility functions for manager environment MDP, including debug visualization.
"""

import torch


def get_active_articulation(env):
    """Return the opt-in active articulation, or ``None`` for legacy scenes."""
    articulations = getattr(env.scene, "articulations", {})
    return articulations.get("active_articulation")


def get_active_articulation_tracked_body_index(env, articulation=None) -> int:
    """Resolve the exact articulated link used as the policy's active object."""
    if articulation is None:
        articulation = get_active_articulation(env)
    if articulation is None:
        raise KeyError("No active_articulation asset found in scene.articulations")

    config = getattr(env.cfg, "config", {}) or {}
    body_name = str(config.get("active_articulation_tracked_body_name", "panel"))
    cache = getattr(env, "_active_articulation_tracked_body_cache", None)
    cache_key = (id(articulation), body_name)
    if cache is not None and cache[0] == cache_key:
        return int(cache[1])
    if body_name not in articulation.body_names:
        raise KeyError(
            f"Tracked articulation body {body_name!r} not found; "
            f"available={list(articulation.body_names)}"
        )
    body_index = articulation.body_names.index(body_name)
    env._active_articulation_tracked_body_cache = (cache_key, body_index)  # noqa: SLF001
    return body_index


def get_active_articulation_joint_tracking_state(env, command):
    """Return joint state and validity in the articulation's USD joint order."""
    articulation = get_active_articulation(env)
    if articulation is None:
        zeros = torch.zeros(env.num_envs, 0, device=env.device)
        return zeros, zeros, zeros, zeros, zeros.bool()

    reference_pos = command.active_articulation_joint_pos
    reference_vel = command.active_articulation_joint_vel
    reference_valid = command.active_articulation_joint_valid_mask
    config = getattr(env.cfg, "config", {}) or {}
    reference_names = list(config.get("active_articulation_reference_joint_names", []))
    if reference_names:
        missing = [name for name in articulation.joint_names if name not in reference_names]
        if missing:
            raise ValueError(
                "Articulation joints are absent from reference channel ordering: "
                f"missing={missing}, reference={reference_names}"
            )
        source_order = [reference_names.index(name) for name in articulation.joint_names]
        reference_pos = reference_pos[:, source_order]
        reference_vel = reference_vel[:, source_order]
        reference_valid = reference_valid[:, source_order]

    if (
        reference_pos.shape[1] != articulation.num_joints
        or reference_vel.shape[1] != articulation.num_joints
        or reference_valid.shape[1] != articulation.num_joints
    ):
        raise ValueError(
            "Active articulation reference joint channels do not match the USD: "
            f"position={reference_pos.shape[1]}, velocity={reference_vel.shape[1]}, "
            f"valid={reference_valid.shape[1]}, usd={articulation.num_joints}, "
            f"names={list(articulation.joint_names)}"
        )
    return (
        articulation.data.joint_pos,
        articulation.data.joint_vel,
        reference_pos,
        reference_vel,
        reference_valid > 0.5,
    )


def get_active_articulation_joint_future_reference(env, command):
    """Return future joint targets and masks in the articulation's USD order."""
    articulation = get_active_articulation(env)
    if articulation is None:
        shape = (env.num_envs, command.num_future_frames, 0)
        zeros = torch.zeros(*shape, device=env.device)
        return zeros, zeros.bool()

    reference_pos = command.active_articulation_joint_pos_multi_future
    reference_valid = command.active_articulation_joint_valid_mask_multi_future
    config = getattr(env.cfg, "config", {}) or {}
    reference_names = list(config.get("active_articulation_reference_joint_names", []))
    if reference_names:
        missing = [name for name in articulation.joint_names if name not in reference_names]
        if missing:
            raise ValueError(
                "Articulation joints are absent from future reference ordering: "
                f"missing={missing}, reference={reference_names}"
            )
        source_order = [reference_names.index(name) for name in articulation.joint_names]
        reference_pos = reference_pos[:, :, source_order]
        reference_valid = reference_valid[:, :, source_order]
    if (
        reference_pos.shape[2] != articulation.num_joints
        or reference_valid.shape[2] != articulation.num_joints
    ):
        raise ValueError(
            "Future articulation reference channels do not match the USD: "
            f"position={reference_pos.shape[2]}, valid={reference_valid.shape[2]}, "
            f"usd={articulation.num_joints}"
        )
    return reference_pos, reference_valid > 0.5


def get_active_entity_base_current_state(env, command):
    """Return the simulated base state for the one active rigid or articulated entity."""
    articulation = get_active_articulation(env)
    if articulation is None:
        return get_target_object_current_state(env, command)

    config = getattr(env.cfg, "config", {}) or {}
    pose_mode = str(config.get("active_entity_policy_pose_mode", "frame"))
    if pose_mode == "panel":
        return get_target_object_current_state(env, command)
    if pose_mode != "frame":
        raise ValueError(
            "active_entity_policy_pose_mode must be 'frame' or 'panel', "
            f"got {pose_mode!r}"
        )

    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return (
        articulation.data.root_pos_w,
        articulation.data.root_quat_w,
        articulation.data.root_lin_vel_w,
        articulation.data.root_ang_vel_w,
        valid,
    )


def get_object_role_mask(
    env,
    command,
    role_names,
    max_objects: int,
    valid: torch.Tensor | None = None,
):
    """Return a per-env object mask for RP2 role names.

    Role metadata is motion-specific in RP2, so the mask is built per environment
    from the command's current motion assignment. Missing role metadata returns an
    all-false mask instead of falling back to an arbitrary object.
    """
    if isinstance(role_names, str):
        role_names = [role_names]
    role_set = {str(role) for role in role_names}
    mask = torch.zeros(env.num_envs, max_objects, dtype=torch.bool, device=env.device)
    role_mask_cache = getattr(command, "_object_role_mask_cache", None)
    if role_mask_cache:
        for role_name in role_set:
            role_mask = role_mask_cache.get(role_name)
            if role_mask is not None and role_mask.shape == mask.shape:
                mask |= role_mask.to(device=env.device, dtype=torch.bool)
        if valid is not None:
            mask &= valid
        return mask
    if not role_set or not hasattr(command, "_object_role_names_by_env"):
        return mask & valid if valid is not None else mask

    roles_by_env = command._object_role_names_by_env(max_objects)  # noqa: SLF001
    for env_idx, roles in enumerate(roles_by_env):
        for obj_idx, role in enumerate(roles[:max_objects]):
            if role in role_set:
                mask[env_idx, obj_idx] = True
    if valid is not None:
        mask &= valid
    return mask


def get_target_object_role_mask(env, command, role_names, valid: torch.Tensor | None = None):
    """Return whether each environment's selected target matches role metadata."""
    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
    role_mask = get_object_role_mask(
        env,
        command,
        role_names,
        max_objects=max_objects,
        valid=valid,
    )
    target_indices = command.target_object_indices().clamp(0, max_objects - 1)
    return role_mask.gather(1, target_indices[:, None]).squeeze(1)


def reduce_masked_object_metric(
    values: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "max",
) -> torch.Tensor:
    """Reduce a per-object metric over a boolean object mask."""
    if reduction == "mean":
        mask_f = mask.float()
        denom = mask_f.sum(dim=-1).clamp_min(1.0)
        return (values * mask_f).sum(dim=-1) / denom
    return torch.where(mask, values, torch.zeros_like(values)).max(dim=-1).values


def get_target_object_scene_keys(
    env, command, object_indices: torch.Tensor | None = None
) -> list[str | None]:
    """Return the per-env scene keys for the selected target object.

    In legacy single-object environments every env maps to ``object``. In
    full-scene RP2 mode the selected index is resolved against each env's motion
    key, because envs can replay different motions with different object assets.
    """
    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
    rigid_objects = getattr(env.scene, "rigid_objects", {})
    if "object" in rigid_objects and max_objects <= 1:
        return ["object"] * env.num_envs

    from gear_sonic.envs.manager_env.mdp.commands import (  # noqa: PLC0415
        motion_order_object_scene_keys,
    )

    if object_indices is None:
        if hasattr(command, "target_object_indices"):
            object_indices = command.target_object_indices()
        else:
            object_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    object_indices_cpu = object_indices.detach().cpu().tolist()
    motion_ids_cpu = command.motion_ids.detach().cpu().tolist()
    scene_keys = []
    for motion_id, object_idx in zip(motion_ids_cpu, object_indices_cpu, strict=False):
        motion_key = command.motion_lib.curr_motion_keys[int(motion_id)]
        keys = motion_order_object_scene_keys(
            env,
            max_objects=max(int(object_idx) + 1, 1),
            motion_key=motion_key,
        )
        scene_keys.append(keys[int(object_idx)] if int(object_idx) < len(keys) else None)
    return scene_keys


def get_target_object_current_state(env, command):
    """Gather current simulated state for the selected target object in each env.

    Returns:
        Tuple ``(pos, quat, lin_vel, ang_vel, valid)`` with per-env tensors.
        Missing scene keys fall back to the reference target pose and ``valid=False``
        so validation terms do not terminate just because a scene asset is absent.
    """
    active_articulation = get_active_articulation(env)
    if active_articulation is not None:
        body_index = get_active_articulation_tracked_body_index(env, active_articulation)
        valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        return (
            active_articulation.data.body_pos_w[:, body_index],
            active_articulation.data.body_quat_w[:, body_index],
            active_articulation.data.body_lin_vel_w[:, body_index],
            active_articulation.data.body_ang_vel_w[:, body_index],
            valid,
        )

    max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)
    rigid_objects = getattr(env.scene, "rigid_objects", {})
    if "object" in rigid_objects and max_objects <= 1:
        obj = env.scene["object"]
        valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        return (
            obj.data.root_pos_w[:, :3],
            obj.data.root_quat_w,
            obj.data.root_lin_vel_w[:, :3],
            obj.data.root_ang_vel_w[:, :3],
            valid,
        )

    ref_pos = getattr(command, "target_object_root_pos", None)
    if ref_pos is None:
        ref_pos = torch.zeros(env.num_envs, 3, device=env.device)
    pos = ref_pos.clone()

    ref_quat = getattr(command, "target_object_root_quat", None)
    if ref_quat is None:
        quat = torch.zeros(env.num_envs, 4, device=env.device)
        quat[:, 0] = 1.0
    else:
        quat = ref_quat.clone()

    lin_vel = torch.zeros(env.num_envs, 3, device=env.device, dtype=pos.dtype)
    ang_vel = torch.zeros(env.num_envs, 3, device=env.device, dtype=pos.dtype)
    valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    slot_count = int(getattr(env.cfg, "object_usd_slot_count", 0) or 0)
    if slot_count > 0:
        if hasattr(command, "target_object_indices"):
            object_indices = command.target_object_indices()
        else:
            object_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        cached_valid = getattr(command, "_object_valid_mask_cache", None)
        for object_idx in range(slot_count):
            scene_key = f"object_{object_idx}"
            if scene_key not in rigid_objects:
                continue
            env_mask = object_indices == object_idx
            if cached_valid is not None and cached_valid.shape[0] == env.num_envs:
                if object_idx < cached_valid.shape[1]:
                    env_mask &= cached_valid[:, object_idx].to(
                        device=env.device, dtype=torch.bool
                    )
                else:
                    env_mask &= False
            if not env_mask.any():
                continue
            obj = env.scene[scene_key]
            pos[env_mask] = obj.data.root_pos_w[env_mask, :3]
            quat[env_mask] = obj.data.root_quat_w[env_mask]
            lin_vel[env_mask] = obj.data.root_lin_vel_w[env_mask, :3]
            ang_vel[env_mask] = obj.data.root_ang_vel_w[env_mask, :3]
            valid[env_mask] = True
        return pos, quat, lin_vel, ang_vel, valid

    scene_keys = get_target_object_scene_keys(env, command)
    for scene_key in sorted({key for key in scene_keys if key is not None}):
        if scene_key not in env.scene.rigid_objects:
            continue
        env_ids = [env_id for env_id, key in enumerate(scene_keys) if key == scene_key]
        env_ids_t = torch.tensor(env_ids, device=env.device, dtype=torch.long)
        obj = env.scene[scene_key]
        pos[env_ids_t] = obj.data.root_pos_w[env_ids_t, :3]
        quat[env_ids_t] = obj.data.root_quat_w[env_ids_t]
        lin_vel[env_ids_t] = obj.data.root_lin_vel_w[env_ids_t, :3]
        ang_vel[env_ids_t] = obj.data.root_ang_vel_w[env_ids_t, :3]
        valid[env_ids_t] = True

    return pos, quat, lin_vel, ang_vel, valid


def get_all_object_current_state(env, command, max_objects: int | None = None):
    """Gather current simulated state for all full-scene objects in motion order.

    Missing scene assets fall back to reference poses with ``valid=False`` so
    mask-aware rewards/terminations can ignore padded or unavailable objects.
    """
    if max_objects is None:
        max_objects = int(getattr(command.motion_lib, "max_num_objects", 1) or 1)

    if get_active_articulation(env) is not None:
        if max_objects != 1:
            raise ValueError(
                "active_articulation mode supports max_num_objects=1; "
                f"got {max_objects}"
            )
        pos, quat, lin_vel, ang_vel, valid = get_target_object_current_state(env, command)
        return (
            pos.unsqueeze(1),
            quat.unsqueeze(1),
            lin_vel.unsqueeze(1),
            ang_vel.unsqueeze(1),
            valid.unsqueeze(1),
        )

    rigid_objects = getattr(env.scene, "rigid_objects", {})
    if "object" in rigid_objects and max_objects == 1:
        obj = env.scene["object"]
        valid = torch.ones(env.num_envs, 1, dtype=torch.bool, device=env.device)
        return (
            obj.data.root_pos_w[:, :3].unsqueeze(1),
            obj.data.root_quat_w.unsqueeze(1),
            obj.data.root_lin_vel_w[:, :3].unsqueeze(1),
            obj.data.root_ang_vel_w[:, :3].unsqueeze(1),
            valid,
        )

    ref_pos = getattr(command, "object_root_pos", None)
    if ref_pos is None:
        pos = torch.zeros(env.num_envs, max_objects, 3, device=env.device)
    else:
        pos = ref_pos[:, :max_objects].clone()

    ref_quat = getattr(command, "object_root_quat", None)
    if ref_quat is None:
        quat = torch.zeros(env.num_envs, max_objects, 4, device=env.device)
        quat[..., 0] = 1.0
    else:
        quat = ref_quat[:, :max_objects].clone()

    lin_vel = torch.zeros(env.num_envs, max_objects, 3, device=env.device, dtype=pos.dtype)
    ang_vel = torch.zeros(env.num_envs, max_objects, 3, device=env.device, dtype=pos.dtype)
    valid = torch.zeros(env.num_envs, max_objects, dtype=torch.bool, device=env.device)

    slot_count = int(getattr(env.cfg, "object_usd_slot_count", 0) or 0)
    if slot_count > 0:
        for obj_idx in range(min(max_objects, slot_count)):
            scene_key = f"object_{obj_idx}"
            if scene_key not in rigid_objects:
                continue
            obj = env.scene[scene_key]
            pos[:, obj_idx] = obj.data.root_pos_w[:, :3]
            quat[:, obj_idx] = obj.data.root_quat_w
            lin_vel[:, obj_idx] = obj.data.root_lin_vel_w[:, :3]
            ang_vel[:, obj_idx] = obj.data.root_ang_vel_w[:, :3]
            valid[:, obj_idx] = True
        cached_valid = getattr(command, "_object_valid_mask_cache", None)
        if cached_valid is not None and cached_valid.shape[:2] == valid.shape:
            valid &= cached_valid.to(device=env.device, dtype=torch.bool)
        return pos, quat, lin_vel, ang_vel, valid

    from gear_sonic.envs.manager_env.mdp.commands import (  # noqa: PLC0415
        motion_order_object_scene_keys,
    )

    motion_ids_cpu = command.motion_ids.detach().cpu().tolist()
    scene_keys_by_env = []
    for motion_id in motion_ids_cpu:
        motion_key = command.motion_lib.curr_motion_keys[int(motion_id)]
        scene_keys_by_env.append(
            motion_order_object_scene_keys(env, max_objects=max_objects, motion_key=motion_key)
        )

    for obj_idx in range(max_objects):
        keys_for_index = [
            keys[obj_idx] if obj_idx < len(keys) else None for keys in scene_keys_by_env
        ]
        for scene_key in sorted({key for key in keys_for_index if key is not None}):
            if scene_key not in rigid_objects:
                continue
            env_ids = [env_id for env_id, key in enumerate(keys_for_index) if key == scene_key]
            env_ids_t = torch.tensor(env_ids, device=env.device, dtype=torch.long)
            obj = env.scene[scene_key]
            pos[env_ids_t, obj_idx] = obj.data.root_pos_w[env_ids_t, :3]
            quat[env_ids_t, obj_idx] = obj.data.root_quat_w[env_ids_t]
            lin_vel[env_ids_t, obj_idx] = obj.data.root_lin_vel_w[env_ids_t, :3]
            ang_vel[env_ids_t, obj_idx] = obj.data.root_ang_vel_w[env_ids_t, :3]
            valid[env_ids_t, obj_idx] = True

    return pos, quat, lin_vel, ang_vel, valid


def get_target_object_contact_forces(
    env,
    command,
    hand: str = "right_hand",
    fallback_sensor_name: str | None = None,
) -> torch.Tensor | None:
    """Return contact forces for the selected per-env target object.

    In legacy scenes this falls back to ``fallback_sensor_name``.  In RP2
    full-scene scenes it resolves the current motion key and target object index
    to the explicit per-object contact sensor created by the scene config.
    """
    contact_map = getattr(env.cfg, "multi_object_contact_sensor_map", {}) or {}
    if not contact_map:
        if fallback_sensor_name and fallback_sensor_name in env.scene.sensors:
            force = env.scene[fallback_sensor_name].data.force_matrix_w
            return force.sum(dim=1) if force.ndim == 4 else force
        return None

    target_indices = (
        command.target_object_indices()
        if hasattr(command, "target_object_indices")
        else torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    )
    motion_ids = command.motion_ids.detach().cpu().tolist()
    target_indices_cpu = target_indices.detach().cpu().tolist()
    stems_by_motion = getattr(env.cfg, "object_usd_stems_by_motion", None) or {}
    global_stems = getattr(env.cfg, "object_usd_stems", None) or []
    slot_count = int(getattr(env.cfg, "object_usd_slot_count", 0) or 0)
    sensor_key = "left_hand_sensor" if hand == "left_hand" else "right_hand_sensor"

    if slot_count > 0:
        out = None
        for object_idx in range(slot_count):
            entry = contact_map.get(str(object_idx))
            if not entry:
                continue
            sensor_name = entry.get(sensor_key)
            if not sensor_name or sensor_name not in env.scene.sensors:
                continue
            force = env.scene[sensor_name].data.force_matrix_w
            if force.ndim == 4:
                force_slot = force.sum(dim=1)
            elif force.ndim == 3:
                force_slot = force
            elif force.ndim == 2:
                force_slot = force[:, None, :]
            else:
                continue
            if out is None:
                out = torch.zeros(
                    env.num_envs,
                    *force_slot.shape[1:],
                    dtype=force_slot.dtype,
                    device=force_slot.device,
                )
            env_mask = target_indices == object_idx
            if env_mask.any():
                out[env_mask] = force_slot[env_mask]
        return out

    out = None
    for env_idx, (motion_id, object_idx) in enumerate(
        zip(motion_ids, target_indices_cpu, strict=False)
    ):
        motion_key = command.motion_lib.curr_motion_keys[int(motion_id)]
        stems = stems_by_motion.get(motion_key) or global_stems
        if int(object_idx) >= len(stems):
            continue
        if slot_count > 0:
            entry = contact_map.get(str(int(object_idx)))
        else:
            stem = str(stems[int(object_idx)]).replace("-", "_")
            entry = contact_map.get(stem)
        if not entry:
            continue
        sensor_name = entry.get(sensor_key)
        if not sensor_name or sensor_name not in env.scene.sensors:
            continue
        force = env.scene[sensor_name].data.force_matrix_w
        force_env = force[env_idx]
        if force_env.ndim == 3:
            force_env = force_env.sum(dim=0)
        if out is None:
            out = torch.zeros(
                env.num_envs,
                force_env.shape[-2],
                3,
                dtype=force_env.dtype,
                device=force_env.device,
            )
        out[env_idx] = force_env
    return out


def debug_visualize_object_projection(
    env, camera, rgb_image, debug_env_idx: int = 0, show_predicted: bool = False
):
    """
    Debug visualization: Project ground truth and predicted object positions onto camera image.

    This function computes the camera pose from the robot's d435_link and camera offset,
    transforms the object position to the camera frame, and projects it to 2D pixel coordinates.

    The coordinate transformation pipeline:
    1. Get object position in robot base (pelvis) frame
    2. Get d435_link pose in robot base frame
    3. Apply camera offset (from config) + random extrinsics delta
    4. Transform object from base frame to camera frame
    5. Map camera axes to OpenCV convention:
       - x_cv = -camera_Y  (camera left → OpenCV right)
       - y_cv = -camera_Z  (camera up → OpenCV down)
       - z_cv = camera_X   (camera forward → OpenCV depth)
    6. Project using pinhole camera model: u = fx*x/z + cx, v = fy*y/z + cy

    Args:
        env: The environment object
        camera: The TiledCamera sensor
        rgb_image: RGB image tensor [num_envs, H, W, 3]
        debug_env_idx: Which environment to visualize
        show_predicted: Whether to also visualize predicted object position (in red)
    """
    import cv2
    from isaaclab.utils.math import (
        quat_apply,
        quat_apply_inverse,
        quat_conjugate,
        quat_from_euler_xyz,
        quat_mul,
    )

    # Get the image for the specified environment
    debug_img = rgb_image[debug_env_idx].detach().cpu().numpy()
    if debug_img.max() <= 1.0:
        debug_img = (debug_img * 255).astype("uint8")
    else:
        debug_img = debug_img.astype("uint8")
    debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
    H, W = debug_img_bgr.shape[:2]

    try:
        # Find object in scene
        object_name = None
        if hasattr(env.scene, "rigid_objects"):
            for name in env.scene.rigid_objects.keys():
                if "object" in name.lower() or "obj" in name.lower():
                    object_name = name
                    break

        if object_name is not None and "d435_link" in env.scene["robot"].body_names:
            robot = env.scene["robot"]
            object_asset = env.scene[object_name]
            device = robot.data.root_pos_w.device

            # Step 1: Object position in robot base frame
            obj_pos_w = object_asset.data.root_pos_w[debug_env_idx]
            robot_pos_w = robot.data.root_pos_w[debug_env_idx]
            robot_quat_w = robot.data.root_quat_w[debug_env_idx]
            obj_rel_w = obj_pos_w - robot_pos_w
            obj_pos_base = quat_apply_inverse(robot_quat_w.unsqueeze(0), obj_rel_w.unsqueeze(0))[0]

            # Step 2: d435_link pose in robot base frame
            link_idx = robot.body_names.index("d435_link")
            link_pos_w = robot.data.body_link_pos_w[debug_env_idx, link_idx]
            link_quat_w = robot.data.body_link_quat_w[debug_env_idx, link_idx]
            link_rel_w = link_pos_w - robot_pos_w
            link_pos_base = quat_apply_inverse(robot_quat_w.unsqueeze(0), link_rel_w.unsqueeze(0))[
                0
            ]
            link_quat_base = quat_mul(
                quat_conjugate(robot_quat_w.unsqueeze(0)), link_quat_w.unsqueeze(0)
            )[0]

            # Step 3: Camera extrinsics (offset from d435_link)
            base_pos_offset = [0.0, 0.0, 0.0]
            base_rot_offset_quat = [1.0, 0.0, 0.0, 0.0]
            if hasattr(camera, "cfg") and hasattr(camera.cfg, "offset"):
                offset_cfg = camera.cfg.offset
                if hasattr(offset_cfg, "pos"):
                    base_pos_offset = list(offset_cfg.pos)
                if hasattr(offset_cfg, "rot"):
                    base_rot_offset_quat = list(offset_cfg.rot)

            # Random extrinsics delta from wrapper
            pos_delta = [0.0, 0.0, 0.0]
            rot_delta = [0.0, 0.0, 0.0]
            if hasattr(env, "wrapper") and hasattr(env.wrapper, "_camera_random_deltas"):
                deltas = env.wrapper._camera_random_deltas.get(debug_env_idx, None)
                if deltas is not None:
                    pos_delta = deltas["pos_delta"]
                    rot_delta = [deltas["roll_delta"], deltas["pitch_delta"], deltas["yaw_delta"]]

            cam_offset_pos = torch.tensor(
                [
                    base_pos_offset[0] + pos_delta[0],
                    base_pos_offset[1] + pos_delta[1],
                    base_pos_offset[2] + pos_delta[2],
                ],
                device=device,
                dtype=torch.float32,
            )
            base_rot_quat = torch.tensor(base_rot_offset_quat, device=device, dtype=torch.float32)

            # Step 4: Camera pose in robot base frame
            cam_offset_world_base = quat_apply(
                link_quat_base.unsqueeze(0), cam_offset_pos.unsqueeze(0)
            )[0]
            cam_pos_base = link_pos_base + cam_offset_world_base
            cam_quat_base = quat_mul(link_quat_base.unsqueeze(0), base_rot_quat.unsqueeze(0))[0]
            if any(r != 0 for r in rot_delta):
                delta_quat = quat_from_euler_xyz(
                    torch.tensor([rot_delta[0]], device=device),
                    torch.tensor([rot_delta[1]], device=device),
                    torch.tensor([rot_delta[2]], device=device),
                )[0]
                cam_quat_base = quat_mul(cam_quat_base.unsqueeze(0), delta_quat.unsqueeze(0))[0]

            # Step 5: Object in camera frame
            obj_rel_cam = obj_pos_base - cam_pos_base
            obj_pos_cam = quat_apply_inverse(cam_quat_base.unsqueeze(0), obj_rel_cam.unsqueeze(0))[
                0
            ]

            # Step 6: Map to OpenCV convention
            # Camera frame: X=forward, Y=left, Z=up
            # OpenCV frame: X=right, Y=down, Z=depth
            x_cv = -obj_pos_cam[1].item()  # -Y (left→right)
            y_cv = -obj_pos_cam[2].item()  # -Z (up→down)
            z_cv = obj_pos_cam[0].item()  # X (forward→depth)

            # Step 7: Project using intrinsics
            K = camera.data.intrinsic_matrices[debug_env_idx].cpu().numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            if z_cv > 0.01:  # Object in front of camera
                u = int(fx * x_cv / z_cv + cx)
                v = int(fy * y_cv / z_cv + cy)

                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(debug_img_bgr, (u, v), 12, (0, 255, 0), 2)
                    cv2.circle(debug_img_bgr, (u, v), 4, (0, 255, 0), -1)
                    cv2.putText(
                        debug_img_bgr,
                        f"GT d={z_cv:.2f}m",
                        (u + 15, v),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                else:
                    u_c = max(10, min(W - 10, u))
                    v_c = max(10, min(H - 10, v))
                    cv2.circle(debug_img_bgr, (u_c, v_c), 8, (0, 0, 255), -1)
                    cv2.putText(
                        debug_img_bgr,
                        "OOB",
                        (u_c + 10, v_c),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )
            else:
                cv2.putText(
                    debug_img_bgr,
                    "Object behind camera",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # Visualize predicted object position (in RED) if available
            # NOTE: pred_pos is in robot BASE (pelvis) frame, same as GT during training
            if show_predicted and hasattr(env, "wrapper"):
                pred_pos = env.wrapper._last_predicted_object_pos
                if pred_pos is not None and pred_pos.shape[0] > debug_env_idx:
                    # pred_pos is in robot BASE (pelvis) frame - need to transform to camera frame
                    pred_pos_base = pred_pos[debug_env_idx]

                    # Transform from base frame to camera frame (same as GT)
                    pred_rel_cam = pred_pos_base - cam_pos_base
                    pred_pos_cam = quat_apply_inverse(
                        cam_quat_base.unsqueeze(0), pred_rel_cam.unsqueeze(0)
                    )[0]

                    # Map to OpenCV convention (same as GT)
                    pred_x_cv = -pred_pos_cam[1].item()  # -Y (left→right)
                    pred_y_cv = -pred_pos_cam[2].item()  # -Z (up→down)
                    pred_z_cv = pred_pos_cam[0].item()  # X (forward→depth)

                    if pred_z_cv > 0.01:  # Predicted object in front of camera
                        pred_u = int(fx * pred_x_cv / pred_z_cv + cx)
                        pred_v = int(fy * pred_y_cv / pred_z_cv + cy)

                        if 0 <= pred_u < W and 0 <= pred_v < H:
                            # Draw RED circle for predicted position
                            cv2.circle(debug_img_bgr, (pred_u, pred_v), 12, (0, 0, 255), 2)
                            cv2.circle(debug_img_bgr, (pred_u, pred_v), 4, (0, 0, 255), -1)
                            cv2.putText(
                                debug_img_bgr,
                                f"PRED d={pred_z_cv:.2f}m",
                                (pred_u + 15, pred_v + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )
                        else:
                            # Out of bounds - draw at edge
                            pred_u_c = max(10, min(W - 10, pred_u))
                            pred_v_c = max(10, min(H - 10, pred_v))
                            cv2.circle(debug_img_bgr, (pred_u_c, pred_v_c), 8, (255, 0, 255), -1)
                            cv2.putText(
                                debug_img_bgr,
                                "PRED OOB",
                                (pred_u_c + 10, pred_v_c),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 0, 255),
                                1,
                            )

    except Exception as e:
        import traceback

        traceback.print_exc()
        cv2.putText(
            debug_img_bgr,
            f"Error: {str(e)[:50]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    # Save and display
    cv2.imwrite("/tmp/camera_debug.png", debug_img_bgr)
    try:
        cv2.imshow(f"Ego Camera (env {debug_env_idx})", debug_img_bgr)
        cv2.waitKey(1)
    except Exception:
        pass


def get_active_object(env):
    """Return the active object asset from the scene.

    In full-scene/multi-object mode the primary object is the first USD stem
    recorded on the env config. In the legacy single-object mode it is
    ``object``.
    """
    active_articulation = get_active_articulation(env)
    if active_articulation is not None:
        return active_articulation
    return env.scene[get_active_object_name(env)]


def get_active_object_name(env) -> str:
    """Return the scene key of the active object."""
    if get_active_articulation(env) is not None:
        return "active_articulation"
    rigid_objects = env.scene.rigid_objects
    if "object" in rigid_objects:
        return "object"
    stems = getattr(env.cfg, "object_usd_stems", None) or []
    for stem in stems:
        key = f"object_{stem.replace('-', '_')}"
        if key in rigid_objects:
            return key
    object_keys = sorted(k for k in rigid_objects if k.startswith("object_"))
    if object_keys:
        return object_keys[0]
    raise KeyError("No object asset found in scene.rigid_objects")
