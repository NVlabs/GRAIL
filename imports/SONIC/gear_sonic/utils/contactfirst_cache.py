"""ContactFirst task-local disposable caches; durable experiment artifacts keep their original owner.

Kit, PhysX, CUDA, shader and Python tool caches are redirected below one node-local root
(default ``/tmp/contactfirst-local-cache``; override with ``CONTACTFIRST_CACHE_BASE``),
separately for the training process and the evaluation children. The root must live on a
local filesystem; remote/network filesystems are rejected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


BASE = Path(os.environ.get('CONTACTFIRST_CACHE_BASE', '/tmp/contactfirst-local-cache'))
LOCAL_ROOT = Path(os.environ.get('CONTACTFIRST_CACHE_LOCAL_ROOT', '/tmp'))
ENV_PATHS = {
    'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
    'XDG_CACHE_HOME': 'cache/general', 'XDG_DATA_HOME': 'kit/data',
    'XDG_CONFIG_HOME': 'kit/config', 'OMNI_CONFIG_PATH': 'kit/config',
    'CUDA_CACHE_PATH': 'cache/cuda', '__GL_SHADER_DISK_CACHE_PATH': 'cache/nvidia',
    'MESA_SHADER_CACHE_DIR': 'cache/mesa', 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
    'TRITON_CACHE_DIR': 'cache/triton', 'NUMBA_CACHE_DIR': 'cache/numba',
    'MPLCONFIGDIR': 'cache/matplotlib', 'HF_HOME': 'cache/huggingface',
    'WANDB_CACHE_DIR': 'cache/wandb', 'WANDB_DATA_DIR': 'cache/wandb_staging',
    'KIT_PERSISTENT_CACHE_ROOT': 'cache/ov', 'PHYSX_UJITSO_STORE_DIR': 'cache/cooking',
    'CONTACTFIRST_SHADER_CACHE_ROOT': 'cache/kit_shaders',
}
KIT_TOKENS = {'cache': 'cache/ov', 'omni_cache': 'cache/ov',
              'omni_global_cache': 'cache/ov', 'data': 'kit/data',
              'omni_data': 'kit/data', 'omni_global_data': 'kit/data'}
REMOTE_FILESYSTEMS = {'lustre', 'nfs', 'nfs4', 'cifs', 'smb3', 'ceph', 'fuse.s3fs', 'fuse.gcsfuse'}


def _local_path(path):
    """Resolve symlinks and reject any writable cache escaping the node-local cache root."""
    value = Path(path)
    if not value.is_absolute() or any(c.isspace() for c in str(value)):
        raise ValueError('Local cache paths must be absolute without whitespace')
    resolved = value.resolve()
    local = LOCAL_ROOT.resolve()
    if not resolved.is_relative_to(local) or resolved == local:
        raise ValueError(f'Writable cache must be contained in the node-local root {local}')
    return resolved


def mount_evidence(path, mountinfo=None):
    path = _local_path(path)
    lines = Path('/proc/self/mountinfo').read_text() if mountinfo is None else mountinfo
    matches = []
    for line in lines.splitlines():
        left, right = line.split(' - ', 1)
        fields = left.split()
        target = Path(fields[4].replace('\\040', ' '))
        if path == target or target in path.parents:
            matches.append((len(target.parts), str(target), right.split()[0]))
    if not matches:
        raise RuntimeError('Cannot establish the filesystem containing task-local cache')
    _, target, filesystem = max(matches)
    if filesystem in REMOTE_FILESYSTEMS or 'lustre' in filesystem.lower():
        raise RuntimeError('Task /tmp is backed by a remote filesystem: ' + filesystem)
    return {'mountpoint': target, 'filesystem': filesystem, 'remote_filesystem_rejected': False}


def cache_root(owner, scope):
    if scope not in {'train', 'eval'}:
        raise ValueError('Cache scope must be train or eval')
    # Scope deliberately excludes eval step and motion: sequential child processes
    # reuse one writable cache, isolated from the concurrently resident parent.
    identity = hashlib.sha256(str(Path(owner).resolve()).encode()).hexdigest()[:24]
    return _local_path(BASE / identity / scope)


def configure(owner, scope, environ=None, receipt_dir=None):
    env = dict(os.environ if environ is None else environ)
    root = cache_root(owner, scope)
    mount = mount_evidence(root)
    root.mkdir(parents=True, exist_ok=True)
    for key, suffix in ENV_PATHS.items():
        target = _local_path(root / suffix)
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    (root / 'kit/config/omniverse.toml').write_text(
        '[paths]\n' + ''.join(f'{key} = {json.dumps(str(root / suffix))}\n' for key, suffix in
        [('data_root', 'kit/data'), ('cache_root', 'cache/ov'), ('logs_root', 'kit/logs')]))
    env['CONTACTFIRST_LOCAL_CACHE_ROOT'] = str(root)
    env['CONTACTFIRST_CACHE_SCOPE'] = scope
    env['CONTACTFIRST_CACHE_OWNER'] = str(Path(owner).resolve())
    if receipt_dir is not None:
        env['CONTACTFIRST_CACHE_RECEIPT_DIR'] = str(Path(receipt_dir))
    else:
        env.pop('CONTACTFIRST_CACHE_RECEIPT_DIR', None)
    receipt = {'schema': 'contactfirst.task_local_cache.plan.v1', 'scope': scope,
               'owner_output_root': str(Path(owner).resolve()), 'cache_root': str(root),
               'mount': mount, 'environment_paths': {k: env[k] for k in ENV_PATHS},
               'per_eval_step_cache_copy': False, 'persistent_output_cache_created': False,
               'observed_at': datetime.now(timezone.utc).isoformat()}
    return env, receipt


def activate(owner, scope, receipt_dir):
    env, receipt = configure(owner, scope, receipt_dir=receipt_dir)
    os.environ.update(env)
    tempfile.tempdir = None  # Python may have memoized the inherited TMPDIR.
    print('[contactfirst_local_cache] ' + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def kit_arguments(environ=None):
    env = os.environ if environ is None else environ
    root = _local_path(env['CONTACTFIRST_LOCAL_CACHE_ROOT'])
    result = [f'--/app/tokens/{key}={root / suffix}' for key, suffix in KIT_TOKENS.items()]
    result.extend([f'--/app/userConfigPath={root / "kit/data/user.config.json"}',
                   f'--/UJITSO/datastore/localCachePath={root / "cache/cooking"}'])
    return result


def observe_kit_paths(environ=None):
    """Called after real SimulationApp initialization; fail if any cache escaped its local root."""
    import carb.settings
    import carb.tokens
    env = os.environ if environ is None else environ
    expected = _local_path(env['CONTACTFIRST_LOCAL_CACHE_ROOT'])
    settings = carb.settings.get_settings()
    tokens = carb.tokens.get_tokens_interface()
    paths = {key: tokens.resolve('${' + key + '}') for key in KIT_TOKENS}
    for key in ('/rtx/shaderDb/shaderCachePath', '/rtx/shaderDb/driverShaderCachePath',
                '/UJITSO/datastore/localCachePath', '/app/userConfigPath'):
        paths[key] = tokens.resolve(str(settings.get(key)))
    for name, value in paths.items():
        if not _local_path(value).is_relative_to(expected):
            raise RuntimeError(f'Kit cache path escaped its owned local root: {name}={value}')
    receipt = {'schema': 'contactfirst.task_local_cache.kit_resolved.v1', 'status': 'verified',
               'observed_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
               'scope': env['CONTACTFIRST_CACHE_SCOPE'], 'cache_root': str(expected),
               'mount': mount_evidence(expected), 'resolved_paths': paths}
    directory = Path(env['CONTACTFIRST_CACHE_RECEIPT_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'kit_cache_resolved_pid{os.getpid()}.json'
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print('[contactfirst_local_cache] Kit resolved cache paths verified: ' + str(path), flush=True)
    return receipt


def install_kit_observer(launcher_module):
    original = launcher_module.SimulationApp.__init__
    if getattr(original, '_contactfirst_cache_observer', False):
        return

    def initialized(self, *args, **kwargs):
        original(self, *args, **kwargs)
        observe_kit_paths()

    initialized._contactfirst_cache_observer = True
    launcher_module.SimulationApp.__init__ = initialized
