"""Kit boot settings for the training app (stdlib only; applied through AppLauncher's ``kit_args`` at every boot).

PhysX collision cooking (omni.physx 107.3, installed image): ``/physics/cooking/ujitsoCollisionCooking`` routes mesh
cooking through UJITSO, whose local data store lives at ``/UJITSO/datastore/localCachePath`` = ``${omni_cache}/DerivedDataCache``
(``allowLocalDataStore`` defaults to true). Measured on the installed image (job 1318970, all four ranks): ``${omni_cache}``
resolves to ``<site-packages>/isaacsim/kit/cache`` INSIDE the container image, not to ``~/.cache/ov`` - so the store was
ephemeral and nothing persisted across legs even with the cooking args on. The formal stage bind-mounts the per-user
persistent directory ``$BASE/.cache`` at the fixed container path ``/root/.cache`` (``/root/.cache/ov`` already holds the
omniverse client cache); the store is therefore pointed explicitly at ``/root/.cache/ov/DerivedDataCache`` whenever that
mount is a writable directory at boot (``PHYSX_UJITSO_STORE_DIR=<dir>`` overrides; ``KIT_PERSISTENT_CACHE_ROOT`` renames the
mount root; no writable root -> Kit's own default, logged by the resolved line). Stable across pin / run root / path alias:
the path is fixed and keyed by the per-user mount, not by any checkout or experiment. The OmniPhysics guide states these
options are not persistent and must be set at every boot, hence boot args rather than a runtime call. Nothing here is a
gate: if the UJITSO cooking service cannot start, PhysX logs the error and cooks in-process as before (native fallback).

``PHYSX_UJITSO_COOKING=0`` keeps the previous in-process cooking (opt-out); the remote (gRPC/hub) stores stay off.
"""
from __future__ import annotations

import os

QUIET_LOG_ARGS = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
# the same quiet settings without the file-logger threshold (kit_args emits that threshold exactly once)
QUIET_LOG_BASE_ARGS = "--/log/level=error --/log/outputStreamLevel=error"
DEFAULT_PERSISTENT_CACHE_ROOT = "/root/.cache"   # the stage's fixed mount point of the per-user persistent cache ($BASE/.cache)
UJITSO_COOKING_ARGS = (
    "--/physics/cooking/ujitsoCollisionCooking=true "
    "--/physics/cooking/ujitsoRemoteCacheEnabled=false "
    "--/UJITSO/datastore/allowLocalDataStore=true "
    "--/UJITSO/datastore/allowHubDataStore=false"
)


def ujitso_cooking_enabled(environ=os.environ) -> bool:
    return str(environ.get("PHYSX_UJITSO_COOKING", "1")).strip().lower() not in ("0", "false", "no", "off")


def ujitso_store_dir(environ=os.environ) -> str | None:
    """Persistent UJITSO local data store: PHYSX_UJITSO_STORE_DIR when set, else <persistent cache root>/ov/DerivedDataCache
    when that root is a writable directory (the stage's /root/.cache mount); created if missing; None otherwise (Kit default)."""
    explicit = str(environ.get("PHYSX_UJITSO_STORE_DIR", "")).strip()
    if explicit:
        target = explicit
    else:
        root = str(environ.get("KIT_PERSISTENT_CACHE_ROOT", "")).strip() or DEFAULT_PERSISTENT_CACHE_ROOT
        if not (os.path.isdir(root) and os.access(root, os.W_OK)):
            return None
        target = os.path.join(root, "ov", "DerivedDataCache")
    try:
        os.makedirs(target, exist_ok=True)
    except OSError:
        return None
    return target if os.access(target, os.W_OK) else None


# Kit log persistence (2026-10-15, Root 09:11Z): Kit's own log (cooking / UJITSO / datastore service lines) lived only in the
# ephemeral container, so whether the cooking service started, fell back or where its store resolved could not be read
# after a leg. With an experiment directory known, the Kit log file goes to <EXPERIMENT_DIR>/kit_logs/ per rank, at
# warning level for everything plus info for the few cooking/store channels (narrow verbosity). KIT_LOG_PERSIST=0 opts
# out; a missing/unwritable directory skips it (native Kit default, never a failure).
# Measured 2026-10-15 (1318970): with the file threshold at warning only one rank's file appeared, holding one cooking
# warning - a message must pass BOTH its channel level and the file logger's threshold, so the info lines of the named
# channels never reached the file. The file threshold is therefore info while the global level stays error: the file
# receives errors from everything plus info/warnings from exactly these channels (still narrow; nothing on stdout changes).
KIT_LOG_CHANNELS = ("omni.physx.cooking.plugin", "omni.physx.plugin", "carb.ujitso.default.plugin",
                    "carb.ujitsoagent.plugin", "carb.datastore.plugin", "omni.datastore.ext.plugin")


def kit_log_persist_enabled(environ=os.environ) -> bool:
    return str(environ.get("KIT_LOG_PERSIST", "1")).strip().lower() not in ("0", "false", "no", "off")


def kit_log_file(environ=os.environ) -> str | None:
    """<EXPERIMENT_DIR or output dir>/kit_logs/kit_rank<R>_job<J>_r<restart>.log, or None when not applicable."""
    if not kit_log_persist_enabled(environ):
        return None
    base = environ.get("KIT_LOG_DIR") or environ.get("EXPERIMENT_DIR")
    if not base:
        return None
    rank = environ.get("RANK") or environ.get("LOCAL_RANK") or "0"
    job = environ.get("SLURM_JOB_ID", "local")
    restart = environ.get("SLURM_RESTART_COUNT", "0")
    directory = os.path.join(base, "kit_logs")
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return None
    return os.path.join(directory, f"kit_rank{rank}_job{job}_r{restart}.log")


def kit_log_args(environ=os.environ) -> str:
    path = kit_log_file(environ)
    if path is None:
        return ""
    channels = " ".join(f"--/log/channels/{name}=info" for name in KIT_LOG_CHANNELS)
    return f"--/log/file={path} --/log/fileLogLevel=info {channels}"


def kit_args(environ=os.environ) -> str:
    """The Kit argument string for AppLauncher: quiet logs, UJITSO collision cooking with its local store on the persistent
    per-user mount unless opted out / no writable mount, and the per-rank Kit log file in the experiment directory when one
    is known (each setting appears once)."""
    log_args = kit_log_args(environ)
    parts = [QUIET_LOG_BASE_ARGS if log_args else QUIET_LOG_ARGS]
    if ujitso_cooking_enabled(environ):
        parts.append(UJITSO_COOKING_ARGS)
        store = ujitso_store_dir(environ)
        if store:
            parts.append(f"--/UJITSO/datastore/localCachePath={store}")
    if log_args:
        parts.append(log_args)
    return " ".join(parts)


def resolved_settings_line(environ=os.environ) -> str:
    """One line with the settings/tokens Kit actually resolved (call after the app exists): where the UJITSO local
    store and omni_cache resolve, whether UJITSO collision cooking / the local store / the local mesh cache are on, and
    the Kit log file. Never raises (returns the error text instead)."""
    try:
        import carb.settings  # type: ignore
        import carb.tokens  # type: ignore

        settings = carb.settings.get_settings()
        tokens = carb.tokens.get_tokens_interface()
        def res(value):
            try:
                return tokens.resolve(str(value)) if value is not None else None
            except Exception:  # noqa: BLE001
                return str(value)
        return (
            "[kit_boot] resolved "
            f"omni_cache={res('${omni_cache}')} "
            f"localCachePath={res(settings.get('/UJITSO/datastore/localCachePath'))} "
            f"allowLocalDataStore={settings.get('/UJITSO/datastore/allowLocalDataStore')} "
            f"ujitsoCollisionCooking={settings.get('/physics/cooking/ujitsoCollisionCooking')} "
            f"ujitsoRemoteCacheEnabled={settings.get('/physics/cooking/ujitsoRemoteCacheEnabled')} "
            f"useLocalMeshCache={settings.get('/persistent/physics/useLocalMeshCache')} "
            f"localMeshCacheSizeMB={settings.get('/persistent/physics/localMeshCacheSizeMB')} "
            f"kit_log_file={settings.get('/log/file')}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"[kit_boot] resolved settings unavailable: {type(exc).__name__}: {exc}"
