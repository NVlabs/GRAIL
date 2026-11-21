"""Inexpensive wall-clock stamps for Isaac Lab scene construction and PhysX start.

Observation only (default on; SCENE_TIMING=0 disables). Wraps a few construction
boundaries so a production log splits the IsaacLab "Time taken for scene creation"
(InteractiveScene USD authoring: env xform cloning, per-asset spawning, collision
filtering) from "Time taken for simulation start" (SimulationContext.reset: PhysX
parse/attach/cook) per rank, without changing any behaviour. Every wrapper calls the
native function and re-raises; a failure to install just prints and keeps the native
code. Lines look like::

  [scene_timing] rank=3 phase=interactive_scene.clone_env_xforms seconds=12.3 t_end=1791830202.1 detail=...

Wrapped (class attributes, restored by the returned callable):
  isaacsim.core.cloner.GridCloner.clone / filter_collisions
  isaaclab.scene.InteractiveScene.__init__ / _add_entities_from_cfg
  isaaclab.sim.spawners.wrappers.wrappers.spawn_multi_asset   (module global, per call)
  isaaclab.sim.SimulationContext.reset
The PhysX attach begin/end stamps live in gear_sonic/physx_replication.py (same SCENE_TIMING switch).
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable


def _enabled() -> bool:
    value = os.environ.get("SCENE_TIMING")
    return value is None or value.strip() not in ("0", "false", "False", "")


def _rank() -> str:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))


def stamp(phase: str, seconds: float | None = None, **detail: Any) -> None:
    extra = " ".join(f"{key}={value}" for key, value in detail.items())
    secs = "" if seconds is None else f" seconds={seconds:.3f}"
    print(f"[scene_timing] rank={_rank()} phase={phase}{secs} t_end={time.time():.1f}" + (f" detail={extra}" if extra else ""), flush=True)


def _timed(phase: str, function: Callable, detail_fn: Callable[..., dict] | None = None) -> Callable:
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            detail = {}
            if detail_fn is not None:
                try:
                    detail = detail_fn(*args, **kwargs) or {}
                except Exception:  # noqa: BLE001 - observation only
                    detail = {}
            try:
                stamp(phase, time.perf_counter() - started, **detail)
            except Exception:  # noqa: BLE001 - observation only (e.g. broken stdout)
                pass

    wrapped.__wrapped__ = function  # type: ignore[attr-defined]
    return wrapped


def install() -> Callable[[], None]:
    """Install the stamps for the next environment construction; returns a restore callable."""
    if not _enabled():
        return lambda: None
    restores: list[Callable[[], None]] = []
    try:
        from isaacsim.core.cloner import GridCloner

        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from isaaclab.sim.spawners.wrappers import wrappers as spawn_wrappers

        def _patch(owner: Any, name: str, phase: str, detail_fn=None) -> None:
            original = getattr(owner, name)
            had_own = name in vars(owner)  # inherited methods are removed again on restore
            setattr(owner, name, _timed(phase, original, detail_fn))

            def restore(owner=owner, name=name, original=original, had_own=had_own):
                if had_own:
                    setattr(owner, name, original)
                else:
                    try:
                        delattr(owner, name)
                    except AttributeError:
                        setattr(owner, name, original)

            restores.append(restore)

        def _clone_detail(self, source_prim_path=None, prim_paths=None, *args, **kwargs):
            paths = prim_paths if prim_paths is not None else kwargs.get("prim_paths")
            return {
                "envs": len(paths) if paths is not None else "?",
                "copy_from_source": kwargs.get("copy_from_source", "?"),
                "replicate_physics": kwargs.get("replicate_physics", "?"),
            }

        def _filter_detail(self, physicsscene_path=None, collision_root_path=None, prim_paths=None, *args, **kwargs):
            paths = prim_paths if prim_paths is not None else kwargs.get("prim_paths")
            return {"envs": len(paths) if paths is not None else "?"}

        def _multi_asset_detail(prim_path, cfg, *args, **kwargs):
            assets = getattr(cfg, "assets_cfg", None)
            return {"prim_path": str(prim_path).rsplit("/", 1)[-1], "assets": len(assets) if assets is not None else "?"}

        def _scene_detail(self, cfg):
            return {"num_envs": getattr(cfg, "num_envs", "?"), "replicate_physics": getattr(cfg, "replicate_physics", "?")}

        _patch(GridCloner, "clone", "cloner.clone", _clone_detail)
        _patch(GridCloner, "filter_collisions", "cloner.filter_collisions", _filter_detail)
        _patch(spawn_wrappers, "spawn_multi_asset", "spawn_multi_asset", _multi_asset_detail)
        _patch(InteractiveScene, "_add_entities_from_cfg", "interactive_scene.add_entities_from_cfg")
        _patch(InteractiveScene, "__init__", "interactive_scene.init_total", _scene_detail)
        _patch(SimulationContext, "reset", "simulation_context.reset")
        stamp("install", None, wrapped=len(restores))
    except Exception as exc:  # noqa: BLE001 - observation only; keep native code
        for restore in reversed(restores):
            try:
                restore()
            except Exception:  # noqa: BLE001
                pass
        print(f"[scene_timing] install skipped: {exc!r}", flush=True)
        return lambda: None

    def restore_all() -> None:
        for restore in reversed(restores):
            try:
                restore()
            except Exception:  # noqa: BLE001
                pass

    return restore_all
