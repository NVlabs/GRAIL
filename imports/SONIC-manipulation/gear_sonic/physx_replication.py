"""PhysX replication of identical per-env prims in heterogeneous Isaac Lab scenes.

Our joint-multisource scenes spawn one object USD per env (MultiUsdFileCfg,
replicate_physics=False), so PhysX parses every env separately -- including the
robot, which is identical in all envs. This module registers a PhysX replicator
(omni.physx get_physx_replicator_interface) that parses ``env_0/<prim>`` once and
replicates it to every other env. Measured 2026-09-04 (L40S, 3072 env/rank):
physics step -24% (HiPHI) / -36% (GRAIL+teleop), PhysX start 150 s -> 35 s,
training +12% steps/s, learning metrics unchanged. Object-level replication of
identical-asset blocks gave no additional gain (cooked meshes are already
deduplicated), so only the robot is replicated and the roster order is untouched.

Config (manager_env.config):
  physx_robot_replication: true            # default on
  physx_replication_prims: ["Robot"]       # per-env child prims to replicate
Env override for experiments: PHYSX_ROBOT_REPLICATION=0|1.

Implementation: wraps InteractiveScene._add_entities_from_cfg, which both the
cold construction path and the authored-stage-cache warm path call after all
per-env prims exist and before PhysX parses the stage. The replicator needs the
SOURCE prims excluded from regular parsing too (it parses them itself with
shareable shapes); otherwise PhysX reports
"attachShape: shape must be shared or unowned".
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable


def _enabled(config: Any) -> bool:
    env = os.environ.get("PHYSX_ROBOT_REPLICATION")
    if env is not None:
        return env.strip() not in ("0", "false", "False", "")
    getter = getattr(config, "get", None)
    return bool(getter("physx_robot_replication", True)) if getter else True


def _scene_timing_enabled() -> bool:
    value = os.environ.get("SCENE_TIMING")
    return value is None or value.strip() not in ("0", "false", "False", "")


def _prims(config: Any) -> list[str]:
    getter = getattr(config, "get", None)
    prims = getter("physx_replication_prims", None) if getter else None
    return [str(p) for p in prims] if prims else ["Robot"]


def register(scene, prims: list[str]) -> None:
    """Register the replicator on ``scene`` (an InteractiveScene after entities exist)."""
    from omni.physx import get_physx_replicator_interface
    from pxr import UsdUtils

    n = int(scene.cfg.num_envs)
    if n < 2 or bool(scene.cfg.replicate_physics):
        return
    env_ns = scene.env_ns
    plan: list[tuple[str, list[str]]] = []
    for prim in prims:
        src = f"{env_ns}/env_0/{prim}"
        if not scene.stage.GetPrimAtPath(src).IsValid():
            print(f"[physx_replication] skip {prim}: {src} not found", flush=True)
            continue
        plan.append((src, [f"{env_ns}/env_{i}/{prim}" for i in range(1, n)]))
    if not plan:
        return
    stage_id = UsdUtils.StageCache.Get().Insert(scene.stage).ToLongInt()
    targets = {src: tg for src, tg in plan}
    exclude = [src for src, _ in plan] + [p for _, tg in plan for p in tg]

    def attach_fn(_stage_id):
        if _scene_timing_enabled():  # observation only: PhysX attach begins here (inside SimulationContext.reset)
            print(f"[scene_timing] rank={os.environ.get('RANK', '0')} phase=physx.attach_begin t_end={time.time():.1f}", flush=True)
        return exclude

    def attach_end_fn(_stage_id):
        t0 = time.perf_counter()
        for src, tg in plan:
            get_physx_replicator_interface().replicate(_stage_id, src, len(tg), False, False)
        print(
            f"[physx_replication] replicated {[s.rsplit('/', 1)[-1] for s, _ in plan]} "
            f"x{n - 1} envs in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        if _scene_timing_enabled():
            print(f"[scene_timing] rank={os.environ.get('RANK', '0')} phase=physx.attach_end t_end={time.time():.1f}", flush=True)

    def rename_fn(replicate_path, index):
        return targets[str(replicate_path)][index]

    get_physx_replicator_interface().register_replicator(stage_id, attach_fn, attach_end_fn, rename_fn)
    print(f"[physx_replication] registered prims={prims} envs={n}", flush=True)


def install(config: Any) -> Callable[[], None]:
    """Enable robot replication for the next InteractiveScene construction(s).

    Returns a restore callable. Must be called after the Isaac Sim app is running.
    """
    if not _enabled(config):
        print("[physx_replication] disabled by config/env", flush=True)
        return lambda: None
    from isaaclab.scene import InteractiveScene

    prims = _prims(config)
    original = InteractiveScene._add_entities_from_cfg

    def wrapped(self):
        original(self)
        if getattr(self, "_physx_replication_done", False):
            return
        self._physx_replication_done = True
        register(self, prims)

    InteractiveScene._add_entities_from_cfg = wrapped

    def restore() -> None:
        InteractiveScene._add_entities_from_cfg = original

    return restore
