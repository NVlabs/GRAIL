"""Managed, opt-in Newton refresh after native edits stop.

Newton/Warp run in the configured external interpreter, never Blender's Python.
The add-on owns and reaps each child; timers only poll a bounded current query.
A saved scene alone cannot authorize execution: enable explicitly trusts exact
configuration bytes and executable identity in this session (or preferences).
"""
import atexit
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import bpy
from bpy.app.handlers import persistent
from bpy.props import StringProperty

from .inspection import blender_collision as collision
from .inspection import collision as client
from .inspection import blender_collision_sequence as sequence

_DEBOUNCE = .6
_INTERVAL = .2
_states = {}
_trusted = {}
_messages = {}
_busy = set()
_registered = False
_restore_pending = False


class GRAILCollisionPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    trusted_bindings_json: StringProperty(default='{}', options={'HIDDEN'})

    def draw(self, context):
        self.layout.label(text='Newton automatic refresh uses explicitly approved runtimes.')
        self.layout.label(text='Enable or disable it in the Newton Collision Diagnostics panel.')


def _preferences():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


def _stored_trust():
    preferences = _preferences()
    try:
        value = json.loads(preferences.trusted_bindings_json) if preferences else {}
    except (ValueError, AttributeError):
        value = {}
    return value if isinstance(value, dict) else {}


def _save_trust(key, record):
    preferences = _preferences()
    if preferences is None:
        raise ValueError('Enable the installed add-on before remembering a runtime in preferences.')
    records = _stored_trust()
    if record is None:
        records.pop(key, None)
    else:
        records[key] = record
    preferences.trusted_bindings_json = json.dumps(records, sort_keys=True)
    # Respect Blender's preference saving policy. We never force save_userpref.


def _resolved_path(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Choose ' + label + '.')
    return Path(bpy.path.abspath(value)).expanduser().absolute()


def _approval(scene):
    settings = scene.grail_collision
    path = _resolved_path(settings.config_path, 'a collision config JSON').resolve()
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or not isinstance(config.get('runtime'), dict):
        raise ValueError('Collision config needs an explicit external runtime.')
    runtime = dict(config['runtime'])
    python = Path(runtime.get('python', '')).expanduser()
    repo = Path(runtime.get('repo', '')).expanduser()
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Runtime python must be an absolute executable path.')
    if not repo.is_absolute() or not (repo / 'ho_contact/collision.py').is_file():
        raise ValueError('Runtime repo must be an absolute ho_contact checkout.')
    device = runtime.get('device', 'cpu')
    if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda:[0-9]+', device):
        raise ValueError('Runtime device must be cpu or cuda:N.')
    timeout = runtime.get('timeout_s', 120)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Runtime timeout_s must be finite and positive.')
    output = _resolved_path(settings.live_output_dir, 'an automatic-query output folder').resolve()
    stat = python.stat()
    # Do not resolve Python for invocation: venv/bin/python is often a symlink,
    # and invoking its resolved base interpreter would lose the environment.
    runtime.update(python=str(python), repo=str(repo.resolve()), device=device, timeout_s=float(timeout))
    config['runtime'] = runtime
    record = dict(config_path=str(path), config_sha256=hashlib.sha256(raw).hexdigest(),
                  python=str(python), python_realpath=str(python.resolve()),
                  python_identity=[stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
                  repo=str(repo.resolve()), device=device, timeout_s=float(timeout),
                  output_root=str(output))
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return key, record, config


def _message(scene, message):
    _messages[scene.as_pointer()] = message
    if hasattr(scene, 'grail_collision'):
        scene.grail_collision.message = message
    for window in bpy.context.window_manager.windows:
        if window.scene == scene:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()


def status(scene):
    state = _states.get(scene.as_pointer())
    return dict(enabled=state is not None, busy=bool(state and state.get('job')),
                message=_messages.get(scene.as_pointer(), 'Automatic refresh is off.'),
                last_output=state.get('last_output') if state else None)


def _new_state(scene, key, record, config):
    names = set(config.get('meshes', []))
    names.update(hand.get('wrist') for hand in config.get('hands', []) if isinstance(hand, dict))
    names.update(('Native_Qpos_Controller', 'ContactEdit_Curves'))
    watched = set(names)
    for name in names:
        obj = scene.objects.get(name) if isinstance(name, str) else None
        while obj:
            watched.add(obj.name)
            if obj.data:
                watched.add(obj.data.name)
            if obj.animation_data and obj.animation_data.action:
                watched.add(obj.animation_data.action.name)
            obj = obj.parent
    return dict(key=key, record=record, config=config, watched=watched, job=None,
                dirty=True, changed_at=time.monotonic(), last_output=None,
                last_stamp=None, completed_digest=None, completed_report=None)


def enable(context, remember=False):
    """Explicitly authorize this configured runtime and start automatic refresh."""
    scene = context.scene
    key, record, config = _approval(scene)
    if remember:
        _save_trust(key, record)
    old = _states.pop(scene.as_pointer(), None)
    if old:
        _cancel_job(old, 're-enabled')
    _trusted[key] = record
    _states[scene.as_pointer()] = _new_state(scene, key, record, config)
    _message(scene, 'Automatic refresh enabled; waiting for the current pose to settle.')
    _ensure_timer()
    return status(scene)


def disable(scene):
    """Stop/reap work and revoke the matching automatic-load authorization."""
    state = _states.pop(scene.as_pointer(), None)
    if state:
        _cancel_job(state, 'disabled')
        _trusted.pop(state['key'], None)
        if state['key'] in _stored_trust():
            _save_trust(state['key'], None)
    _message(scene, 'Automatic refresh is off.')


def refresh(scene):
    """Request the latest pose without launching a duplicate blocking query."""
    state = _states.get(scene.as_pointer())
    if state is None:
        raise ValueError('Enable automatic refresh first.')
    _cancel_job(state, 'explicit refresh')
    state.update(dirty=True, changed_at=time.monotonic() - _DEBOUNCE,
                 completed_digest=None, completed_report=None, force_query=True)
    _message(scene, 'Automatic refresh requested for the current pose.')


@contextmanager
def suspend_queries(scene):
    """Pause a bounded capture without revoking the user's runtime binding."""
    pointer = scene.as_pointer()
    state = _states.pop(pointer, None)
    if state:
        _cancel_job(state, 'sequence capture')
    try:
        yield
    finally:
        if state:
            state.update(dirty=True, changed_at=time.monotonic(), last_stamp=None)
            _states[pointer] = state
            _message(scene, 'Automatic contacts resumed after sequence capture.')


def _journal(job, status, **extra):
    payload = dict(status=status, source_frame=job['snapshot']['source_frame'],
                   input_digest=job['digest'], **extra)
    (job['output'] / 'lifecycle.json').write_text(json.dumps(payload, indent=2) + '\n')


def _close_job(job):
    for stream in job['streams']:
        stream.close()


def _cancel_job(state, reason):
    job = state.get('job')
    if job is None:
        return
    process = job['process']
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=.05)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=.25)
    _close_job(job)
    state['job'] = None
    try:
        _journal(job, 'cancelled', reason=reason)
    except OSError:
        # An output directory removed by the user must not prevent reaping.
        pass


def _touch(scene, state):
    _cancel_job(state, 'superseded by a frame or pose edit')
    state.update(dirty=True, changed_at=time.monotonic())
    collision.mark_collision_stale(scene)
    _message(scene, 'Pose changed; refreshing after editing/playback stops.')


def _stamp(scene, state):
    names = set(state['config'].get('meshes', []))
    names.update(hand.get('wrist') for hand in state['config'].get('hands', []) if isinstance(hand, dict))
    matrices = [(name, tuple(value for row in scene.objects[name].matrix_world for value in row))
                for name in sorted(names) if isinstance(name, str) and name in scene.objects]
    return (float(scene.frame_current_final), float(scene.unit_settings.scale_length),
            collision._native_controls(scene), matrices,
            scene.grail_collision.contact_map_mode,
            float(scene.grail_collision.contact_map_threshold_m))


def _paused(scene):
    windows = [window for window in bpy.context.window_manager.windows if window.scene == scene]
    if not bpy.app.background and not windows:
        return False
    return not any(window.screen.is_animation_playing for window in windows)


@contextmanager
def _context(scene):
    window = next((window for window in bpy.context.window_manager.windows if window.scene == scene), None)
    if window:
        with bpy.context.temp_override(window=window, scene=scene):
            yield
    else:
        with bpy.context.temp_override(scene=scene):
            yield


def _capture(scene, state):
    from . import collision_ui
    pointer = scene.as_pointer()
    _busy.add(pointer)
    try:
        with _context(scene):
            collision_ui._sync(scene)
            effective = collision_ui._config(scene)
            effective['runtime'] = state['config']['runtime']
            state['config'] = effective
            snapshot = collision.extract_snapshot(scene, state['config'])
            stamp = _stamp(scene, state)
        state['last_stamp'] = stamp
        return snapshot, stamp
    finally:
        _busy.discard(pointer)


def _launch(scene, state, snapshot, stamp):
    runtime = state['config']['runtime']
    root = Path(state['record']['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:10]
    output = root / ('source_' + f'{snapshot["source_frame"]:g}'.replace('.', '_') + '_' + suffix)
    output.mkdir()
    source = output / 'snapshot.json'
    source.write_text(json.dumps(snapshot, separators=(',', ':'), allow_nan=False) + '\n')
    command = [runtime['python'], str(Path(client.__file__).with_name('collision_backend.py')),
               '--input', str(source), '--output', str(output / 'report.json'),
               '--repo', runtime['repo'], '--device', runtime['device']]
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    if runtime['device'] == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    streams = [(output / 'stdout.log').open('w'), (output / 'stderr.log').open('w')]
    try:
        process = subprocess.Popen(command, stdout=streams[0], stderr=streams[1],
                                   env=env, cwd=output, stdin=subprocess.DEVNULL)
    except Exception:
        for stream in streams:
            stream.close()
        raise
    job = dict(process=process, streams=streams, snapshot=snapshot, stamp=stamp,
               digest=client.snapshot_digest(snapshot), output=output, started=time.monotonic())
    state.update(job=job, dirty=False, last_output=str(output))
    _journal(job, 'running')
    _message(scene, f'Computing Newton contacts for source {snapshot["source_frame"]:g}…')


def _apply(scene, state, snapshot, report):
    from . import collision_ui
    pointer = scene.as_pointer()
    _busy.add(pointer)
    try:
        with _context(scene):
            collision_ui.apply_live_report(scene, snapshot, report, state['config'])
    finally:
        _busy.discard(pointer)
    state.update(completed_digest=client.snapshot_digest(snapshot), completed_report=report,
                 dirty=False, force_query=False, last_stamp=_stamp(scene, state))
    _message(scene, f'Automatic contacts current at source {snapshot["source_frame"]:g}.')


def _finish(scene, state):
    job = state['job']
    process = job['process']
    if process.poll() is None:
        if time.monotonic() - job['started'] > state['config']['runtime']['timeout_s']:
            _cancel_job(state, 'timeout')
            raise RuntimeError('Newton query timed out; use Refresh to retry. Logs: ' + str(job['output']))
        return
    _close_job(job)
    state['job'] = None
    if process.returncode:
        _journal(job, 'failed', returncode=process.returncode)
        raise RuntimeError(f'Newton query failed ({process.returncode}); logs: {job["output"]}')
    snapshot, stamp = _capture(scene, state)
    if client.snapshot_digest(snapshot) != job['digest'] or stamp != job['stamp']:
        _journal(job, 'discarded', reason='pose changed before result application')
        state.update(dirty=True, changed_at=time.monotonic())
        _message(scene, 'Discarded an outdated query; waiting for the latest pose.')
        return
    report = json.loads((job['output'] / 'report.json').read_text())
    client.validate_report(report, snapshot)
    _apply(scene, state, snapshot, report)
    _journal(job, 'displayed')


def _tick():
    global _restore_pending
    if not _registered:
        return None
    if _restore_pending:
        _restore_pending = False
        _load_post(None)
    scenes = {scene.as_pointer(): scene for scene in bpy.data.scenes}
    for pointer, state in list(_states.items()):
        scene = scenes.get(pointer)
        if scene is None:
            _cancel_job(state, 'scene removed')
            _states.pop(pointer, None)
            continue
        try:
            key, record, _config = _approval(scene)
            if key != state['key'] or _trusted.get(key) != record:
                disable(scene)
                _message(scene, 'Configuration/runtime changed; enable automatic refresh again to approve it.')
                continue
            stamp = _stamp(scene, state)
            if state['last_stamp'] is not None and stamp != state['last_stamp']:
                _touch(scene, state)
            state['last_stamp'] = stamp
            if not _paused(scene) or bpy.app.is_job_running('RENDER'):
                if state['job']:
                    _touch(scene, state)
                state['changed_at'] = time.monotonic()
                continue
            if state['job']:
                _finish(scene, state)
            elif state['dirty'] and time.monotonic() - state['changed_at'] >= _DEBOUNCE:
                sequence.update(scene)
                cached = sequence.status(scene)
                if cached.get('enabled') and cached.get('cache_hit') and not state.get('force_query'):
                    state['dirty'] = False
                    _message(scene, f'Sequence contact cache current at source {cached["source_frame"]:g}.')
                    continue
                try:
                    snapshot, stamp = _capture(scene, state)
                except ValueError as error:
                    # Loading never executes embedded preview scripts. Wait for
                    # the separately authorized native-preview operation.
                    if 'preview' not in str(error).lower():
                        raise
                    state['changed_at'] = time.monotonic() + .4
                    _message(scene, str(error))
                    continue
                digest = client.snapshot_digest(snapshot)
                if digest == state['completed_digest'] and state['completed_report'] is not None:
                    if collision.collision_is_current(scene):
                        state['dirty'] = False
                    else:
                        _apply(scene, state, snapshot, state['completed_report'])
                else:
                    _launch(scene, state, snapshot, stamp)
        except Exception as error:
            _cancel_job(state, 'error')
            state['dirty'] = False
            _message(scene, 'Automatic query paused: ' + str(error))
    return _INTERVAL


def _ensure_timer():
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=_INTERVAL, persistent=True)


@persistent
def _frame(scene, *_unused):
    state = _states.get(scene.as_pointer())
    if state and scene.as_pointer() not in _busy:
        _touch(scene, state)


@persistent
def _geometry(scene, depsgraph):
    pointer = scene.as_pointer()
    state = _states.get(pointer)
    if not state or pointer in _busy or pointer in collision._building:
        return
    for update in depsgraph.updates:
        source = update.id.original
        if source.name not in state['watched'] or source.get('grail_collision_owner'):
            continue
        # Object selection and viewport navigation can also update depsgraph.
        # They do not change the contact input and must not restart Newton.
        geometry_changed = isinstance(source, (bpy.types.Mesh, bpy.types.Object)) and update.is_updated_geometry
        if geometry_changed or _stamp(scene, state) != state['last_stamp']:
            _touch(scene, state)
            state['last_stamp'] = _stamp(scene, state)
            break


@persistent
def _load_pre(_unused):
    for state in _states.values():
        _cancel_job(state, 'loading another file')
    _states.clear()
    _messages.clear()
    _busy.clear()


@persistent
def _load_post(_unused):
    for scene in bpy.data.scenes:
        if not hasattr(scene, 'grail_collision'):
            continue
        try:
            key, record, config = _approval(scene)
            if _trusted.get(key) == record and scene.as_pointer() not in _states:
                _states[scene.as_pointer()] = _new_state(scene, key, record, config)
                _message(scene, 'Approved sequence loaded; refresh waits for its enabled native preview.')
        except (ValueError, OSError, TypeError, AttributeError):
            pass
    _ensure_timer()


def _handlers():
    return ((bpy.app.handlers.frame_change_post, _frame),
            (bpy.app.handlers.depsgraph_update_post, _geometry),
            (bpy.app.handlers.load_pre, _load_pre),
            (bpy.app.handlers.load_post, _load_post))


def _shutdown():
    # Blender may close without explicitly disabling each installed add-on.
    for state in list(_states.values()):
        _cancel_job(state, 'Blender closing')


def register():
    global _registered, _restore_pending
    if _registered:
        return
    bpy.utils.register_class(GRAILCollisionPreferences)
    _registered = True
    atexit.register(_shutdown)
    _trusted.update(_stored_trust())
    for handlers, callback in _handlers():
        if callback not in handlers:
            handlers.append(callback)
    # Add-on registration runs in Blender's restricted data context. Defer
    # reading loaded scenes until the first normal event-loop callback.
    _restore_pending = True
    _ensure_timer()


def unregister():
    global _registered
    _registered = False
    atexit.unregister(_shutdown)
    _load_pre(None)
    _trusted.clear()
    for handlers, callback in _handlers():
        if callback in handlers:
            handlers.remove(callback)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if GRAILCollisionPreferences.is_registered:
        bpy.utils.unregister_class(GRAILCollisionPreferences)
