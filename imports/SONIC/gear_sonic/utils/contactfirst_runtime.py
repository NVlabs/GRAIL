#!/usr/bin/env python3
"""ContactFirst process-level runtime adapters (training and evaluation).

Activated through config (``contactfirst_runtime`` in the experiment yaml) or by
the deterministic evaluation script; nothing here runs for the released
task-general tracking configs.

* ``install_fk_float32`` feeds float64 archived motion samples to the vendored FK model's
  float32 ABI (stored payloads are unchanged).
* ``install_thread_limit`` sizes Kit/TBB/PhysX thread pools to the CPU budget of the run.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
from functools import wraps
import os
import re
import sys
from gear_sonic.utils.contactfirst_cache import kit_arguments, install_kit_observer


THREAD_SETTINGS = (
    "/plugins/carb.tasking.plugin/threadCount",
    "/plugins/omni.tbb.globalcontrol/maxThreadCount",
    "/persistent/physics/numThreads",
)
FK_MODULE = "gear_sonic.utils.motion_lib.torch_humanoid_batch"


def install_fk_float32() -> None:
    """Adapt archived float64 motion samples to the pinned FK model's float32 ABI."""
    import torch
    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    original = Humanoid_Batch.fk_batch
    if getattr(original, "_contactfirst_float32_inputs", False):
        return
    logged_conversion = False

    @wraps(original)
    def float32_fk(self, pose, trans, *args, **kwargs):
        nonlocal logged_conversion
        if self._local_rotation_mat.dtype != torch.float32:
            raise RuntimeError("ContactFirst FK adapter requires the pinned float32 skeleton model")
        if not isinstance(pose, torch.Tensor) or not isinstance(trans, torch.Tensor):
            raise TypeError("ContactFirst FK inputs must be torch tensors")
        if not pose.is_floating_point() or not trans.is_floating_point():
            raise TypeError("ContactFirst FK inputs must be floating point")
        pose32, trans32 = pose.to(dtype=torch.float32), trans.to(dtype=torch.float32)
        if not logged_conversion:
            print(
                "[contactfirst_fk_dtype] float32 FK input adapter active; "
                f"pose={tuple(pose.shape)} {pose.dtype}->float32, "
                f"trans={tuple(trans.shape)} {trans.dtype}->float32; "
                "source tensors and stored payloads are unchanged",
                flush=True,
            )
            logged_conversion = True
        return original(self, pose32, trans32, *args, **kwargs)

    float32_fk._contactfirst_float32_inputs = True
    Humanoid_Batch.fk_batch = float32_fk


def defer_fk_float32_until_import() -> None:
    """Install the adapter after the upstream FK module's natural import."""
    if FK_MODULE in sys.modules:
        install_fk_float32()
        return
    if any(getattr(finder, "_contactfirst_fk_module", None) == FK_MODULE for finder in sys.meta_path):
        return

    class DeferredLoader(importlib.abc.Loader):
        def __init__(self, delegate, finder):
            self.delegate, self.finder = delegate, finder

        def create_module(self, spec):
            return self.delegate.create_module(spec)

        def exec_module(self, module):
            self.delegate.exec_module(module)
            install_fk_float32()
            sys.meta_path.remove(self.finder)
            print("[contactfirst_fk_dtype] adapter installed after upstream FK module import", flush=True)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    class DeferredFinder(importlib.abc.MetaPathFinder):
        _contactfirst_fk_module = FK_MODULE

        def find_spec(self, fullname, path=None, target=None):
            if fullname != FK_MODULE:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise ImportError(f"Pinned FK module is unavailable: {fullname}")
            spec.loader = DeferredLoader(spec.loader, self)
            return spec

    sys.meta_path.insert(0, DeferredFinder())


def install_thread_limit(threads: int, shader_arguments: list[str] | None = None) -> None:
    if threads < 1:
        raise ValueError("CONTACTFIRST_CPU_THREADS must be positive")
    from gear_sonic import kit_boot_args

    original = kit_boot_args.kit_args

    def bounded_kit_args(environ=os.environ):
        # Upstream calls this after importing AppLauncher, before creating it.
        # Installed AppLauncher drops limit_cpu_threads kwargs from its config
        # whitelist. Set SimulationApp's documented default here so its own
        # PXR/OPENBLAS environment assignment also uses the allocation limit.
        launcher = importlib.import_module("isaaclab.app.app_launcher")
        install_kit_observer(launcher)
        launcher.SimulationApp.DEFAULT_LAUNCHER_CONFIG["limit_cpu_threads"] = threads
        arguments = original(environ)
        for setting in THREAD_SETTINGS:
            arguments = re.sub(r"(?:^|\s)--" + re.escape(setting) + r"=\S+", "", arguments)
        # SimulationApp appends sys.argv/kit_args after its default32 settings;
        # these final CLI settings also override AppLauncher's host-based count.
        return " ".join([arguments.strip(), *(f"--{setting}={threads}" for setting in THREAD_SETTINGS), *(shader_arguments or []), *kit_arguments(environ)])

    kit_boot_args.kit_args = bounded_kit_args
    os.environ["PXR_WORK_THREAD_LIMIT"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    print(f"[contactfirst_cpu] Kit tasking/TBB/physics threads per rank: {threads}", flush=True)


def install_from_config(config, base_dir=None):
    """Config-driven activation from ``train_agent_trl`` (before AppLauncher).

    ``config.contactfirst_runtime``: ``enabled`` (default false), ``cpu_threads``
    (Kit/TBB/PhysX pool size), ``local_cache`` (redirect disposable caches to the
    node-local root), ``fk_float32`` (feed float64 archived motion samples to the
    pinned float32 FK model). Returns the cache receipt (or None).
    """
    section = {}
    if config is not None:
        raw = config.get("contactfirst_runtime", None)
        if raw is not None:
            section = dict(raw)
    if not section.get("enabled", False):
        return None
    receipt = None
    if section.get("local_cache", False):
        if base_dir is None:
            raise ValueError("contactfirst_runtime.local_cache requires base_dir")
        from pathlib import Path

        from gear_sonic.utils.contactfirst_cache import activate

        diagnostics = Path(base_dir) / "cache_diagnostics"
        receipt = activate(base_dir, os.environ.get("CONTACTFIRST_CACHE_SCOPE", "train"), diagnostics)
        os.environ.setdefault("CONTACTFIRST_SHADER_CACHE_ROOT", receipt["environment_paths"]["CONTACTFIRST_SHADER_CACHE_ROOT"])
    threads = section.get("cpu_threads", None)
    if threads is not None:
        from gear_sonic.utils.contactfirst_shader_cache import prepare_from_environment

        install_thread_limit(int(threads), prepare_from_environment())
    if section.get("fk_float32", False):
        defer_fk_float32_until_import()
    return receipt
