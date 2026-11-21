#!/usr/bin/env python3
"""Batch render trajectory replays with a single IsaacSim session.

Initializes IsaacSim once, then renders each trajectory sequentially —
swapping object USDs between renders. This avoids the ~90s init overhead
per trajectory that would occur when calling multi_scene_render.py as
a subprocess.

Usage (on cluster with GPU + IsaacSim):
    python -u batch_render_replay.py \
        --shard_dir /path/to/eval/step_019500/export_shard_0 \
        --object_usd_dir /path/to/exported/step_019500/merged/object_usd \
        --output_dir /path/to/exported/step_019500/merged/vis \
        --skip_existing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

# Per-motion progress heartbeat. Updated before each motion and each frame.
# A watchdog thread force-exits the process if no heartbeat for WATCHDOG_TIMEOUT
# seconds, so a hung USD load / sim step can't stall the job indefinitely.
_last_progress_time = time.time()
_last_progress_label = "init"
WATCHDOG_TIMEOUT = 180.0  # seconds


def _heartbeat(label: str) -> None:
    global _last_progress_time, _last_progress_label
    _last_progress_time = time.time()
    _last_progress_label = label


def _watchdog_loop():
    while True:
        time.sleep(15.0)
        idle = time.time() - _last_progress_time
        if idle > WATCHDOG_TIMEOUT:
            sys.stderr.write(
                f"\n[WATCHDOG] No progress for {idle:.0f}s at '{_last_progress_label}'. "
                f"Forcing non-zero exit so the scheduler flags this job.\n"
            )
            sys.stderr.flush()
            os._exit(2)


def start_watchdog():
    t = threading.Thread(target=_watchdog_loop, daemon=True, name="render-watchdog")
    t.start()
    return t


def reconstruct_filter_keys(metrics, render_sort_by="obj_pos_error"):
    """Reconstruct filter_keys order from Phase 1 metrics.

    Replicates eval_agent_trl.py logic for env_idx -> motion_key mapping.
    """
    all_dict = metrics.get("eval/all_metrics_dict", {})
    motion_keys = all_dict.get("motion_keys", [])
    if not motion_keys:
        raise ValueError("No motion_keys in metrics")

    terminated = all_dict.get("terminated", [])
    mpjpe_l = all_dict.get("mpjpe_l", [0.0] * len(motion_keys))
    mpjpe_g = all_dict.get("mpjpe_g", [0.0] * len(motion_keys))
    obj_pos_errors = all_dict.get("obj_pos_error", None)

    sort_idx = 4 if render_sort_by == "obj_pos_error" else 1

    success_pair = [
        (motion_keys[i], mpjpe_l[i], mpjpe_g[i], True, obj_pos_errors[i] if obj_pos_errors else 0.0)
        for i in range(len(motion_keys))
        if not terminated[i]
    ]
    failed_pair = [
        (
            motion_keys[i],
            mpjpe_l[i],
            mpjpe_g[i],
            False,
            obj_pos_errors[i] if obj_pos_errors else 0.0,
        )
        for i in range(len(motion_keys))
        if terminated[i]
    ]

    success_sorted = sorted(success_pair, key=lambda x: x[sort_idx], reverse=True)
    failed_sorted = sorted(failed_pair, key=lambda x: x[sort_idx], reverse=True)

    all_pair = failed_sorted + success_sorted
    filter_keys = [p[0] for p in all_pair]
    success_set = {p[0] for p in success_pair}

    return filter_keys, success_set


def build_render_plan(shard_dir, object_usd_dir, output_dir, skip_existing=False, traj_dir=None):
    """Build list of (env_idx, motion_key, traj_path, usd_path, output_path)."""
    metrics_path = os.path.join(shard_dir, "metrics_eval.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    filter_keys, success_set = reconstruct_filter_keys(metrics)
    if traj_dir is None:
        traj_dir = os.path.join(shard_dir, "trajectories")

    # Original env→motion_key map (load order during phase1 eval). The recorder
    # writes `{env_idx:06d}.trajectory.pkl` keyed on this load order, so we
    # must look up trajectories by THIS mapping — NOT the sort-permuted
    # filter_keys above. (Bug fix: previously used enumerate(filter_keys),
    # which paired success motions with the wrong env's trajectory whenever
    # any motion failed and got pushed to position 0 by reconstruct_filter_keys.)
    motion_keys_load_order = metrics.get("eval/all_metrics_dict", {}).get("motion_keys", [])
    motion_to_env = {mk: i for i, mk in enumerate(motion_keys_load_order)}

    plan = []
    stats = {"skipped": 0, "missing_traj": 0, "missing_usd": 0}

    for motion_key in filter_keys:
        if motion_key not in success_set:
            continue
        env_idx = motion_to_env.get(motion_key)
        if env_idx is None:
            stats["missing_traj"] += 1
            continue

        traj_path = os.path.join(traj_dir, f"{motion_key}.trajectory.pkl")
        if not os.path.exists(traj_path):
            traj_path = os.path.join(traj_dir, f"{env_idx:06d}.trajectory.pkl")
        if not os.path.exists(traj_path):
            stats["missing_traj"] += 1
            continue

        output_path = os.path.join(output_dir, f"{motion_key}.mp4")
        if skip_existing and os.path.exists(output_path):
            stats["skipped"] += 1
            continue

        usd_path = os.path.join(object_usd_dir, f"{motion_key}.usd")
        if not os.path.exists(usd_path):
            usd_path = os.path.join(object_usd_dir, f"{motion_key}.usda")
        if not os.path.exists(usd_path):
            stats["missing_usd"] += 1
            usd_path = None

        plan.append((env_idx, motion_key, traj_path, usd_path, output_path))

    return plan, stats, len(filter_keys), len(success_set)


def compute_start_frame_skip(total_frames, requested_skip):
    """Clamp the optional initial frame skip to a valid frame range."""
    requested_skip = max(0, int(requested_skip))
    if total_frames <= 0:
        return 0
    return min(requested_skip, total_frames - 1)


def render_all(
    plan,
    resolution=(1920, 1080),
    camera_offset=(-3.54, 0.0, 1.2),
    camera_target=(0.0, 0.0, 0.8),
    headless=True,
    start_frame_skip=0,
    robot_usd=None,
):
    """Render all trajectories with a single IsaacSim session."""
    import imageio
    import torch

    robot_usd = robot_usd or os.environ.get("GRAIL_ROBOT_USD")
    if not robot_usd or not os.path.isfile(robot_usd):
        raise ValueError("Supply the G1 43-DOF robot asset via --robot_usd or GRAIL_ROBOT_USD")

    # ---- Launch IsaacSim ----
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=headless, enable_cameras=True)
    simulation_app = launcher.app

    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

    # Inline G1_43DOF_CFG to avoid importing groot.rl (which drags in unsynced code)
    G1_43DOF_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(robot_usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76)),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # ---- Physics simulation (CPU, kinematic only) ----
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 50.0,
        render_interval=1,
        device="cpu",
        use_fabric=True,
        gravity=(0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)

    # ---- Ground + lighting ----
    ground_cfg = sim_utils.GroundPlaneCfg(size=(500.0, 500.0))
    ground_cfg.func("/World/ground", ground_cfg)

    dome_cfg = sim_utils.DomeLightCfg(
        color=(0.45, 0.55, 0.75),
        intensity=1500.0,
    )
    dome_cfg.func("/World/DomeLight", dome_cfg)

    stage = omni.usd.get_context().get_stage()
    dome_prim = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
    if dome_prim.GetPrim().IsValid():
        dome_prim.GetTextureFileAttr().Set("")

    # ---- Robot (43-DOF, HOI) ----
    env_path = "/World/envs/env_0"
    env_prim = stage.DefinePrim(env_path, "Xform")

    robot_cfg = G1_43DOF_CFG.copy()
    robot_path = f"{env_path}/Robot"
    robot_cfg.spawn.func(robot_path, robot_cfg.spawn, translation=(0, 0, 0))

    # ---- Camera ----
    w, h = resolution
    camera_cfg = CameraCfg(
        prim_path="/World/OverviewCamera",
        offset=CameraCfg.OffsetCfg(pos=camera_offset, rot=(1, 0, 0, 0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            focus_distance=100.0,
            horizontal_aperture=10.0,
            clipping_range=(0.1, 500.0),
        ),
        width=w,
        height=h,
    )
    camera = Camera(camera_cfg)

    # ---- Articulation wrapper ----
    robot_cfg.prim_path = "/World/envs/env_.*/Robot"
    art = Articulation(robot_cfg)

    # ---- Initialize ----
    sim.reset()
    camera.reset()
    art.reset()
    print(
        f"[DOF] Articulation num_joints = {int(art.num_joints)} "
        f"(body-only trajectories will be zero-padded to this width)",
        flush=True,
    )

    cam_device = getattr(camera, "_device", "cpu")
    eye = torch.tensor([list(camera_offset)], dtype=torch.float32, device=cam_device)
    tgt = torch.tensor([list(camera_target)], dtype=torch.float32, device=cam_device)
    camera.set_world_poses_from_view(eye, tgt)

    # ---- Warmup shaders ----
    for _ in range(5):
        sim.step(render=True)
        camera.update(dt=0.0)
    print("IsaacSim initialized, shaders warmed up")
    start_watchdog()
    _heartbeat("render_loop_start")

    # ---- Render loop ----
    current_obj_usd = None
    prev_obj_path = None
    total = len(plan)
    succeeded = 0
    failed = 0
    start_time = time.time()

    # --- Scene-yaw helpers (match multi_scene_render.py cosmetic rotation) ---
    def _quat_mul_wxyz(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    def _rotate_xy(x, y, cos_a, sin_a):
        return x * cos_a - y * sin_a, x * sin_a + y * cos_a

    for idx, (env_idx, motion_key, traj_path, usd_path, output_path) in enumerate(plan):
        elapsed = time.time() - start_time
        eta = (elapsed / max(idx, 1)) * (total - idx)
        print(
            f"\n[{idx+1}/{total}] {motion_key} (env {env_idx:06d}) ETA: {eta/60:.0f}m", flush=True
        )
        _heartbeat(f"motion_{idx}_{motion_key}_load_traj")

        try:
            # Load trajectory
            with open(traj_path, "rb") as f:
                traj = pickle.load(f)

            # ---- Swap object USD if needed ----
            # Unique prim path per iteration: RemovePrim doesn't reliably clear USD
            # references at the same path, so prior object geometry leaks through.
            obj_path = f"{env_path}/Object_{idx:06d}"
            if usd_path != current_obj_usd:
                # Delete previous iteration's object prim
                if prev_obj_path and stage.GetPrimAtPath(prev_obj_path).IsValid():
                    stage.RemovePrim(prev_obj_path)
                # Delete old table
                for p in [f"{env_path}/Table"] + [f"{env_path}/TableLeg_{j}" for j in range(4)]:
                    if stage.GetPrimAtPath(p).IsValid():
                        stage.RemovePrim(p)

                # Spawn new object
                if usd_path and os.path.exists(usd_path):
                    # No rigid_props: we set xform ops directly each frame for
                    # kinematic replay. RigidBody with kinematic_enabled=True
                    # caches its own pose via PhysX/Fabric and silently overrides
                    # our xformOp updates after the first sim step, which
                    # produced stale/inconsistent stair orientations across
                    # frames. Raw USD reference lets orient_op.Set() take effect.
                    obj_spawn_cfg = sim_utils.UsdFileCfg(
                        usd_path=os.path.abspath(usd_path),
                    )
                    obj_spawn_cfg.func(obj_path, obj_spawn_cfg, translation=(0, 0, 0))

                    # Assign fallback color to meshes whose materials are missing or
                    # bound to orphaned/broken shaders (some USDs reference MDL textures
                    # that don't ship with the asset).
                    obj_prim = stage.GetPrimAtPath(obj_path)
                    if obj_prim.IsValid():
                        for desc in Usd.PrimRange(obj_prim):
                            if desc.GetTypeName() == "Mesh":
                                mesh = UsdGeom.Mesh(desc)
                                binding_api = UsdShade.MaterialBindingAPI(desc)
                                bound_mat = binding_api.GetDirectBinding().GetMaterial()
                                needs_fallback = False
                                if not bound_mat.GetPrim().IsValid():
                                    needs_fallback = True
                                else:
                                    # Material exists — check that its surface shader
                                    # actually resolves to an asset. MDL shaders with a
                                    # missing info:mdl:sourceAsset render invisible.
                                    try:
                                        surface = bound_mat.ComputeSurfaceSource()
                                        shader = (
                                            surface[0] if isinstance(surface, tuple) else surface
                                        )
                                        if shader and shader.GetPrim().IsValid():
                                            src_input = shader.GetInput("info:mdl:sourceAsset")
                                            if src_input is not None:
                                                src_asset = src_input.Get()
                                                if src_asset is None or not str(src_asset):
                                                    needs_fallback = True
                                        else:
                                            needs_fallback = True
                                    except Exception:
                                        needs_fallback = True
                                if needs_fallback:
                                    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.6, 0.65)])

                # Spawn table if trajectory has table data above ground
                if traj.get("table_pos_w") is not None and float(traj["table_pos_w"][0][2]) >= 0.1:
                    table_pos = traj["table_pos_w"][0].copy()
                    table_z = float(table_pos[2])
                    table_w, table_d, table_t = 2.0, 0.6, 0.04

                    table_prim = stage.DefinePrim(f"{env_path}/Table", "Cube")
                    cube = UsdGeom.Cube(table_prim)
                    cube.GetSizeAttr().Set(1.0)
                    xf = UsdGeom.Xformable(table_prim)
                    # Translate MUST come before Scale (USD applies outermost-first;
                    # [T,S] -> scale local point, then translate in parent frame)
                    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, table_z))
                    xf.AddScaleOp().Set(Gf.Vec3f(table_w, table_d, table_t))
                    cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.35, 0.2)])

                    # Table legs
                    leg_height = table_z - table_t / 2.0
                    if leg_height > 0:
                        leg_w = 0.04
                        hw, hd = table_w / 2.0, table_d / 2.0
                        inset, dinset = 0.03, 0.10
                        corners = [
                            (hw - inset - leg_w / 2, hd - dinset - leg_w / 2),
                            (hw - inset - leg_w / 2, -(hd - dinset - leg_w / 2)),
                            (-(hw - inset - leg_w / 2), hd - dinset - leg_w / 2),
                            (-(hw - inset - leg_w / 2), -(hd - dinset - leg_w / 2)),
                        ]
                        for j, (lx, ly) in enumerate(corners):
                            leg_path = f"{env_path}/TableLeg_{j}"
                            leg_prim = stage.DefinePrim(leg_path, "Cube")
                            leg_cube = UsdGeom.Cube(leg_prim)
                            leg_cube.GetSizeAttr().Set(1.0)
                            leg_xf = UsdGeom.Xformable(leg_prim)
                            # Translate before Scale (same convention as multi_scene_render.py)
                            leg_xf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, leg_height / 2.0))
                            leg_xf.AddScaleOp().Set(Gf.Vec3f(leg_w, leg_w, leg_height))
                            leg_cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.28, 0.15)])

                current_obj_usd = usd_path
                prev_obj_path = obj_path
                # Let IsaacSim process the new prims
                sim.step(render=True)
            else:
                # Same USD as previous iteration — reuse the existing prim path
                obj_path = prev_obj_path or obj_path

            # ---- Determine scene center ----
            has_table = (
                traj.get("table_pos_w") is not None and float(traj["table_pos_w"][0][2]) >= 0.1
            )
            has_obj = traj.get("object_pos_w") is not None
            object_scale = np.asarray(
                traj.get("object_scale", np.ones(3)), dtype=np.float32
            ).reshape(-1)
            if object_scale.size == 1:
                object_scale = np.repeat(object_scale, 3)
            if (
                object_scale.size != 3
                or not np.all(np.isfinite(object_scale))
                or np.any(object_scale <= 0)
            ):
                raise ValueError(f"invalid object_scale: {object_scale}")

            if has_table:
                cx, cy = float(traj["table_pos_w"][0][0]), float(traj["table_pos_w"][0][1])
            elif has_obj:
                cx, cy = float(traj["object_pos_w"][0][0]), float(traj["object_pos_w"][0][1])
            else:
                mid = traj["total_frames"] // 2
                cx, cy = float(traj["root_pos_w"][mid][0]), float(traj["root_pos_w"][mid][1])

            # ---- Scene yaw (cosmetic rotation mirroring multi_scene_render.py) ----
            # Stairs/terrain/sitting motions are stored in a canonical source frame;
            # the preview renderer rotates the whole scene (robot + object XY & quat)
            # so the natural rise/seat direction aligns with the camera view. The
            # relative pose between robot and object is preserved (rigid-body yaw).
            path_blob = f"{motion_key}|{traj_path}|{usd_path}|{output_path}".lower()
            obj_on_ground = has_obj and abs(float(traj["object_pos_w"][0][2])) < 0.05
            # Stair/sitting motions: source motion lib stores a FIXED object quat
            # that, applied alone, tips the USD in world frame. Matching the
            # reference preview renderer (fancy-eval-render/multi_scene_render.py),
            # we compose an extra Z-axis yaw before applying (135° for stairs, 180°
            # for sitting) to restore upright orientation and align camera framing.
            if "sitting" in path_blob:
                scene_yaw = float(np.pi)
            elif "stair" in path_blob:
                scene_yaw = float(135.0 * np.pi / 180.0)
            elif obj_on_ground:
                scene_yaw = float(135.0 * np.pi / 180.0)
            else:
                scene_yaw = 0.0
            override_obj_quat_identity = False
            scene_cos = float(np.cos(scene_yaw))
            scene_sin = float(np.sin(scene_yaw))
            scene_yaw_quat = np.array(
                [np.cos(scene_yaw / 2.0), 0.0, 0.0, np.sin(scene_yaw / 2.0)],
                dtype=np.float32,
            )
            if idx == 0 or scene_yaw != 0:
                print(f"  scene_yaw={np.degrees(scene_yaw):.0f}deg", flush=True)

            # ---- Render trajectory frames ----
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fps = traj.get("fps", 25.0)
            total_frames = traj["total_frames"]
            skip = compute_start_frame_skip(total_frames, start_frame_skip)
            if skip and idx == 0:
                print(f"[FRAME] Skipping first {skip} frame(s) per --start_frame_skip", flush=True)

            try:
                writer = imageio.get_writer(
                    output_path, fps=fps, codec="libx264", quality=5, pixelformat="yuv420p"
                )
            except Exception as e:
                print(f"  FAILED to open writer: {e}")
                failed += 1
                continue

            # Robot articulation expects `num_art_dofs` joints. Trajectories may be
            # body-only (e.g. 29 DOF for G1 terrain/sitting) while the articulation is
            # 43-DOF (body + hands) — pad the trailing hand slots with zeros so
            # write_joint_state_to_sim gets the right shape. Works for any robot whose
            # traj DOFs are a leading subset of articulation DOFs.
            num_art_dofs = int(art.num_joints)
            traj_dofs = int(traj["dof_pos"].shape[1])
            pad_dofs = num_art_dofs - traj_dofs
            if pad_dofs < 0:
                raise RuntimeError(
                    f"traj has {traj_dofs} DOFs but articulation only has {num_art_dofs} — "
                    f"check robot config matches trajectory source"
                )
            if idx == 0:
                if pad_dofs > 0:
                    print(
                        f"[DOF] Padding {traj_dofs}-DOF trajectories to "
                        f"{num_art_dofs}-DOF articulation ({pad_dofs} zero hand DOFs)",
                        flush=True,
                    )
                else:
                    print(
                        f"[DOF] Trajectory DOFs match articulation "
                        f"({num_art_dofs}) — no padding needed",
                        flush=True,
                    )

            for f in range(skip, total_frames):
                if f % 25 == 0:
                    _heartbeat(f"motion_{idx}_{motion_key}_frame_{f}")
                # Robot state (pad to 43 DOFs when traj is body-only 29)
                dof_row = traj["dof_pos"][f]
                if pad_dofs > 0:
                    dof_row = np.concatenate([dof_row, np.zeros(pad_dofs, dtype=dof_row.dtype)])
                joint_pos = torch.from_numpy(dof_row).float().unsqueeze(0)
                joint_vel = torch.zeros_like(joint_pos)
                root_state = torch.zeros(1, 13, device="cpu")

                root_pos = traj["root_pos_w"][f].copy()
                root_pos[0] -= cx
                root_pos[1] -= cy
                if scene_yaw != 0.0:
                    rx, ry = _rotate_xy(root_pos[0], root_pos[1], scene_cos, scene_sin)
                    root_pos[0], root_pos[1] = rx, ry
                root_state[0, :3] = torch.from_numpy(root_pos).float()
                root_quat_f = np.asarray(traj["root_quat_w"][f], dtype=np.float32)
                if scene_yaw != 0.0:
                    root_quat_f = _quat_mul_wxyz(scene_yaw_quat, root_quat_f)
                root_state[0, 3:7] = torch.from_numpy(root_quat_f).float()

                art.write_joint_state_to_sim(joint_pos, joint_vel)
                art.write_root_state_to_sim(root_state)

                # Object state
                obj_prim = stage.GetPrimAtPath(obj_path)
                if has_obj and obj_prim.IsValid():
                    obj_pos = traj["object_pos_w"][f].copy()
                    if override_obj_quat_identity:
                        obj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        obj_quat = np.asarray(traj["object_quat_w"][f], dtype=np.float32)  # wxyz
                    ox = float(obj_pos[0]) - cx
                    oy = float(obj_pos[1]) - cy
                    if scene_yaw != 0.0:
                        ox, oy = _rotate_xy(ox, oy, scene_cos, scene_sin)
                        obj_quat = _quat_mul_wxyz(scene_yaw_quat, obj_quat)

                    xformable = UsdGeom.Xformable(obj_prim)
                    ops = xformable.GetOrderedXformOps()
                    translate_op = orient_op = scale_op = None
                    for op in ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            translate_op = op
                        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                            orient_op = op
                        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                            scale_op = op
                    if translate_op is None:
                        translate_op = xformable.AddTranslateOp()
                    if orient_op is None:
                        orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
                    if scale_op is None:
                        scale_op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)
                    translate_op.Set(Gf.Vec3d(ox, oy, float(obj_pos[2])))
                    # obj_quat is wxyz end-to-end (convert_hoi_to_motion_lib.py stores
                    # source wxyz; motion_lib passes bytes verbatim to IsaacLab's
                    # write_root_state_to_sim which expects wxyz). USD Gf.Quatf takes
                    # (real, i, j, k) == (w, x, y, z), so [0..3] maps through directly.
                    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                        orient_op.Set(
                            Gf.Quatf(
                                float(obj_quat[0]),
                                float(obj_quat[1]),
                                float(obj_quat[2]),
                                float(obj_quat[3]),
                            )
                        )
                    else:
                        orient_op.Set(
                            Gf.Quatd(
                                float(obj_quat[0]),
                                float(obj_quat[1]),
                                float(obj_quat[2]),
                                float(obj_quat[3]),
                            )
                        )
                    if scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                        scale_op.Set(
                            Gf.Vec3f(
                                float(object_scale[0]),
                                float(object_scale[1]),
                                float(object_scale[2]),
                            )
                        )
                    else:
                        scale_op.Set(
                            Gf.Vec3d(
                                float(object_scale[0]),
                                float(object_scale[1]),
                                float(object_scale[2]),
                            )
                        )
                    if f == skip and idx == 0:
                        # Confirm the orient op actually took the stored value
                        post_ops = xformable.GetOrderedXformOps()
                        for o in post_ops:
                            try:
                                print(f"[DBG-POST] {o.GetOpName()} = {o.Get()}", flush=True)
                            except Exception:
                                print(f"[DBG-POST] {o.GetOpName()} (err)", flush=True)

                # Table state
                table_prim = stage.GetPrimAtPath(f"{env_path}/Table")
                if has_table and table_prim.IsValid():
                    tp = traj["table_pos_w"][f].copy()
                    tx = float(tp[0]) - cx
                    ty = float(tp[1]) - cy
                    tz = float(tp[2])

                    table_xf = UsdGeom.Xformable(table_prim)
                    t_ops = table_xf.GetOrderedXformOps()
                    t_translate = None
                    for op in t_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            t_translate = op
                    if t_translate is None:
                        t_translate = table_xf.AddTranslateOp()
                    t_translate.Set(Gf.Vec3d(tx, ty, tz))

                    # Update legs
                    leg_height = tz - 0.02
                    if leg_height > 0:
                        leg_w = 0.04
                        hw, hd = 1.0, 0.4
                        inset, dinset = 0.03, 0.10
                        corners = [
                            (tx + hw - inset - leg_w / 2, ty + hd - dinset - leg_w / 2),
                            (tx + hw - inset - leg_w / 2, ty - (hd - dinset - leg_w / 2)),
                            (tx - (hw - inset - leg_w / 2), ty + hd - dinset - leg_w / 2),
                            (tx - (hw - inset - leg_w / 2), ty - (hd - dinset - leg_w / 2)),
                        ]
                        for j, (lx, ly) in enumerate(corners):
                            leg_prim = stage.GetPrimAtPath(f"{env_path}/TableLeg_{j}")
                            if leg_prim.IsValid():
                                leg_xf = UsdGeom.Xformable(leg_prim)
                                l_ops = leg_xf.GetOrderedXformOps()
                                l_tr = None
                                for op in l_ops:
                                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                        l_tr = op
                                if l_tr is None:
                                    l_tr = leg_xf.AddTranslateOp()
                                l_tr.Set(Gf.Vec3d(lx, ly, leg_height / 2.0))

                # Step and capture
                sim.forward()
                sim.render()
                camera.update(dt=0.0)
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                writer.append_data(rgb)

            writer.close()
            succeeded += 1
            render_time = (time.time() - start_time) / max(idx + 1, 1)
            print(
                f"  OK: {output_path} ({total_frames - skip} frames, {render_time:.1f}s avg)",
                flush=True,
            )
            _heartbeat(f"motion_{idx}_{motion_key}_done")
        except Exception as e:
            failed += 1
            print(f"  FAILED {motion_key}: {e}", flush=True)
            traceback.print_exc()
            # Remove any partial output so a retry re-renders it cleanly
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            _heartbeat(f"motion_{idx}_{motion_key}_failed")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch render complete: {succeeded}/{total} succeeded, {failed} failed")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/max(total,1):.1f}s per trajectory)")

    # IsaacSim teardown (simulation_app.close()) can hang in USD/Hydra cleanup after
    # rendering many frames. Match eval_agent_trl.py and force-exit the process —
    # all .mp4 outputs were already flushed by the per-motion writer.close() calls,
    # so this is safe and returns exit code 0 for scheduler completion.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Batch render trajectory replays (in-process)")
    parser.add_argument("--robot_usd", help="G1 43-DOF robot asset (or set GRAIL_ROBOT_USD)")
    parser.add_argument(
        "--shard_dir", required=True, help="Path to shard directory containing metrics_eval.json"
    )
    parser.add_argument(
        "--traj_dir",
        default=None,
        help="Directory with *.trajectory.pkl files (default: {shard_dir}/trajectories)",
    )
    parser.add_argument(
        "--object_usd_dir", required=True, help="Directory with {motion_key}.usd files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for rendered .mp4 files"
    )
    parser.add_argument(
        "--resolution", default="1920x1080", help="Output video resolution (default: 1920x1080)"
    )
    parser.add_argument(
        "--camera_offset",
        type=float,
        nargs=3,
        default=[-3.54, 0.0, 1.2],
        help="Camera position [x y z]",
    )
    parser.add_argument(
        "--camera_target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.8],
        help="Camera look-at point [x y z]",
    )
    parser.add_argument(
        "--start_frame_skip",
        type=int,
        default=0,
        help="Initial frames to omit from each output video (default: 0)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="Skip trajectories with existing output videos"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print render plan without executing"
    )
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    plan, stats, n_total, n_success = build_render_plan(
        args.shard_dir,
        args.object_usd_dir,
        args.output_dir,
        skip_existing=args.skip_existing,
        traj_dir=args.traj_dir,
    )
    w, h = map(int, args.resolution.split("x"))

    print(f"Shard: {args.shard_dir}")
    print(f"Filter keys: {n_total} total, {n_success} successful")
    print(f"Render plan: {len(plan)} trajectories")
    for k, v in stats.items():
        if v:
            print(f"  {k}: {v}")

    if args.dry_run:
        for env_idx, mk, tp, up, op in plan[:10]:
            print(f"  env {env_idx:06d} -> {mk}")
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more")
        return

    if not plan:
        print("Nothing to render.")
        return

    # Sort by object USD path to minimize USD swapping
    plan.sort(key=lambda x: (x[3] or "", x[1]))

    render_all(
        plan,
        resolution=(w, h),
        camera_offset=tuple(args.camera_offset),
        camera_target=tuple(args.camera_target),
        headless=args.headless,
        start_frame_skip=args.start_frame_skip,
        robot_usd=args.robot_usd,
    )


if __name__ == "__main__":
    main()
