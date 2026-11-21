"""Window-local dashboard navigation and opt-in, paused-timeline refresh.

Handlers only queue work. Geometry is inspected on the main-thread UI timer
after native preview evaluation; scene scripts are never invoked here.
"""
import json
from pathlib import Path
import time
import sys

import bpy
from bpy.app.handlers import persistent

from .inspection import blender_inspection as inspection
from . import runtime_adapter as adapter

_watchers = {}
_views = {}
_building = set()


def read_config(scene):
    raw = scene.grail_g1.inspection_config_path.strip()
    if not raw:
        raise ValueError('Choose an inspection config JSON file.')
    path = Path(bpy.path.abspath(raw)).expanduser().resolve()
    return json.loads(path.read_text(encoding='utf-8')), (str(path), path.stat().st_mtime_ns)


def _source_ids(scene, config):
    names = {config.get('object'), 'Native_Qpos_Controller', 'ContactEdit_Curves'}
    for hand in config.get('hands', []):
        names.add(hand.get('wrist'))
        names.update(hand.get('meshes', []))
    ids = set()
    for name in names:
        obj = scene.objects.get(name) if isinstance(name, str) else None
        while obj is not None:
            ids.add(obj.as_pointer())
            if obj.data is not None:
                ids.add(obj.data.as_pointer())
            if obj.animation_data and obj.animation_data.action:
                ids.add(obj.animation_data.action.as_pointer())
            obj = obj.parent
    return ids


def _native_control_state(scene):
    """Record current native channels even when their visual preview is off.

    Depsgraph notifications describe reevaluation, not necessarily a changed
    pose. Geometry alone is insufficient because a disabled native preview
    leaves its display meshes fixed while editable qpos channels can change.
    """
    result = []
    for name in ('Native_Qpos_Controller', adapter.runtime.CONTROLLER):
        obj = scene.objects.get(name)
        if obj is None:
            continue
        properties = tuple((key, float(obj[key])) for key in sorted(obj.keys())
                           if key.startswith('qpos_'))
        action = obj.animation_data.action if obj.animation_data else None
        channels = tuple((curve.data_path, curve.array_index, curve.mute,
                          float(curve.evaluate(scene.frame_current_final)))
                         for curve in action.fcurves) if action else ()
        result.append((name, obj.as_pointer(), properties,
                       action.as_pointer() if action else None, channels))
    return tuple(result)


def _camera_spaces(scene, report):
    dashboard = report.get('dashboard', {}) if report else {}
    names = {dashboard.get('camera')}
    result = []
    for window in bpy.context.window_manager.windows:
        if window.scene != scene:
            continue
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            if space.use_local_camera and space.camera and space.camera.name in names:
                result.append((window, area))
    return result


def configure_shading(space, scene):
    # Workbench does not expose Material Preview. Use original-vertex colors
    # in that viewport without changing the source scene's render engine.
    if scene.render.engine == 'BLENDER_WORKBENCH':
        space.shading.type = 'SOLID'
        space.shading.color_type = 'VERTEX'
        space.shading.light = 'STUDIO'
    else:
        space.shading.type = 'MATERIAL'


def show(context, report, remember=True):
    if context.area is None or context.area.type != 'VIEW_3D':
        raise ValueError('Open the dashboard from a 3D View sidebar.')
    scene, space = context.scene, context.space_data
    region = space.region_3d
    key = (context.window.as_pointer(), context.area.as_pointer())
    already_dashboard = (space.use_local_camera and space.camera
                         and space.camera.get('grail_inspection_group') == 'dashboard')
    if remember and key not in _views and not already_dashboard:
        _views[key] = dict(scene=scene, use_local=space.use_local_camera,
            camera=space.camera.name if space.camera else None,
            perspective=region.view_perspective, location=region.view_location.copy(),
            rotation=region.view_rotation.copy(), distance=region.view_distance,
            camera_zoom=region.view_camera_zoom, camera_offset=tuple(region.view_camera_offset),
            shading=space.shading.type, shading_color=space.shading.color_type,
            shading_light=space.shading.light, overlays=space.overlay.show_overlays)
    camera = inspection.set_inspection_view(scene, report, report['dashboard']['camera'], context.view_layer)
    space.use_local_camera = True
    space.camera = camera
    region.view_perspective = 'CAMERA'
    region.view_camera_zoom = 0
    region.view_camera_offset = (0, 0)
    configure_shading(space, scene)
    space.overlay.show_overlays = False
    context.area.tag_redraw()


def restore(context):
    key = (context.window.as_pointer(), context.area.as_pointer())
    state = _views.pop(key, None)
    context.scene.grail_g1.inspection_follow_timeline = False
    if not state or state['scene'] != context.scene:
        # Navigation history is session-local. A reopened dashboard still has
        # an unchanged source camera, so returning remains useful after load.
        space = context.space_data
        space.use_local_camera = False
        space.camera = context.scene.camera
        space.region_3d.view_perspective = 'CAMERA' if context.scene.camera else 'PERSP'
        space.overlay.show_overlays = True
        stop_scene(context.scene)
        context.area.tag_redraw()
        return
    space, region = context.space_data, context.space_data.region_3d
    space.use_local_camera = state['use_local']
    space.camera = bpy.data.objects.get(state['camera']) if state['camera'] else None
    region.view_perspective = state['perspective']
    region.view_location = state['location']
    region.view_rotation = state['rotation']
    region.view_distance = state['distance']
    region.view_camera_zoom = state['camera_zoom']
    region.view_camera_offset = state['camera_offset']
    space.shading.type = state['shading']
    space.shading.color_type = state['shading_color']
    space.shading.light = state['shading_light']
    space.overlay.show_overlays = state['overlays']
    stop_scene(context.scene)
    context.area.tag_redraw()


def _sync_enabled_preview(scene, required=False):
    """Evaluate only an already enabled, scene-bound native preview."""
    if adapter.runtime.CONTROLLER in scene.objects:
        controller = scene.objects[adapter.runtime.CONTROLLER]
        if (controller.get('grail_g1_owner_scene') == scene
                and adapter.runtime.STATE.get('preview') is not None):
            adapter.runtime.contact_edit_preview(scene)
            bpy.context.view_layer.update()
            return
        if required:
            raise ValueError('Enable Exact Preview for this native scene before following timeline edits.')
    elif 'Native_Qpos_Controller' in scene.objects:
        runtime = sys.modules.get('grail_portable_native')
        if runtime and getattr(runtime, 'exact_native_preview', None) in bpy.app.handlers.frame_change_post:
            runtime.validate_scene_contract()
            runtime.exact_native_preview(scene)
            bpy.context.view_layer.update()
            return
        if required:
            raise ValueError('Enable this portable scene\'s trusted native preview before following the timeline.')


def refresh(scene, require_native=False):
    config, stamp = read_config(scene)
    views = _camera_spaces(scene, inspection.snapshot_metadata(scene))
    pointer = scene.as_pointer()
    _building.add(pointer)
    try:
        _sync_enabled_preview(scene, required=require_native)
        report = inspection.build_dashboard(scene, config, scene.frame_current + scene.frame_subframe - 1)
        # Never leave a viewport pointing at a deleted snapshot camera.
        for window, area in views:
            with bpy.context.temp_override(window=window, area=area):
                show(bpy.context, report, remember=False)
        _watchers[pointer] = dict(scene=scene, ids=_source_ids(scene, config), stamp=stamp,
            frame=(scene.frame_current, scene.frame_subframe), scale=scene.unit_settings.scale_length,
            controls=_native_control_state(scene), dirty=None, marked=False)
        if not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick, first_interval=.2)
        return report
    finally:
        _building.discard(pointer)


def pending(scene):
    state = _watchers.get(scene.as_pointer())
    return bool(state and state['dirty'] is not None)


def follow_changed(settings, context):
    scene = settings.id_data
    pointer = scene.as_pointer()
    if not settings.inspection_follow_timeline:
        # Keep lightweight stale tracking; this does not rebuild snapshots.
        return
    try:
        config, stamp = read_config(scene)
        report = inspection.snapshot_metadata(scene)
        if not report or 'dashboard' not in report:
            raise ValueError('Open the dashboard before enabling Follow Timeline.')
        _building.add(pointer)
        try:
            _sync_enabled_preview(scene, required=True)
        finally:
            _building.discard(pointer)
        _watchers[pointer] = dict(scene=scene, ids=_source_ids(scene, config), stamp=stamp,
            frame=(scene.frame_current, scene.frame_subframe), scale=scene.unit_settings.scale_length,
            controls=_native_control_state(scene), dirty=time.monotonic(), marked=False)
        if not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick, first_interval=.2)
    except Exception as exc:
        settings.inspection_follow_timeline = False
        settings.last_message = 'Dashboard follow stopped: ' + str(exc)


@persistent
def _frame_changed(scene, *_unused):
    state = _watchers.get(scene.as_pointer())
    if state and scene.as_pointer() not in _building:
        state['dirty'] = time.monotonic()


@persistent
def _geometry_changed(scene, depsgraph):
    pointer = scene.as_pointer()
    state = _watchers.get(pointer)
    if not state or pointer in _building:
        return
    if any(update.id.original.as_pointer() in state['ids'] for update in depsgraph.updates):
        state['dirty'] = time.monotonic()


def tick():
    for pointer, state in list(_watchers.items()):
        try:
            scene = state['scene']
            if not inspection.snapshot_metadata(scene):
                _watchers.pop(pointer, None)
                continue
            windows = [w for w in bpy.context.window_manager.windows if w.scene == scene]
            if not windows or scene != bpy.context.scene:
                continue
            config_path = Path(bpy.path.abspath(scene.grail_g1.inspection_config_path)).expanduser().resolve()
            stamp = (str(config_path), config_path.stat().st_mtime_ns)
            if (state['frame'] != (scene.frame_current, scene.frame_subframe)
                    or state['scale'] != scene.unit_settings.scale_length or state['stamp'] != stamp):
                if state['dirty'] is None:
                    state['dirty'] = time.monotonic()
            if state['dirty'] is None:
                continue
            if bpy.app.is_job_running('RENDER'):
                continue
            # Showing new diagnostic objects can schedule deferred reevaluation
            # of unchanged source IDs after _building is cleared. Confirm a
            # semantic change before labeling a fresh dashboard stale. Keep
            # changed native controls stale even if preview-disabled geometry
            # still matches the last snapshot.
            if (not state['marked']
                    and state['frame'] == (scene.frame_current, scene.frame_subframe)
                    and state['scale'] == scene.unit_settings.scale_length
                    and state['stamp'] == stamp
                    and state['controls'] == _native_control_state(scene)):
                report = inspection.current_report(scene)
                if report is not None and inspection.inspection_is_current(scene, report):
                    state['dirty'] = None
                    continue
            if not state['marked']:
                inspection.mark_dashboard_stale(scene)
                state['marked'] = True
            if (not scene.grail_g1.inspection_follow_timeline
                    or any(w.screen.is_animation_playing for w in windows)
                    or time.monotonic() - state['dirty'] < .3):
                continue
            refresh(scene, require_native=True)
            scene.grail_g1.last_message = 'Dashboard refreshed after timeline / pose change.'
        except Exception as exc:
            _watchers.pop(pointer, None)
            try:
                state['scene'].grail_g1.inspection_follow_timeline = False
                state['scene'].grail_g1.last_message = 'Dashboard follow stopped: ' + str(exc)
            except ReferenceError:
                pass
    return .2 if _watchers else None


def stop_scene(scene):
    scene.grail_g1.inspection_follow_timeline = False
    _watchers.pop(scene.as_pointer(), None)


def reset():
    _watchers.clear()
    _building.clear()
    _views.clear()
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)


def register():
    for handlers, callback in ((bpy.app.handlers.frame_change_post, _frame_changed),
                               (bpy.app.handlers.depsgraph_update_post, _geometry_changed)):
        if callback not in handlers:
            handlers.append(callback)


def unregister():
    reset()
    for handlers, callback in ((bpy.app.handlers.frame_change_post, _frame_changed),
                               (bpy.app.handlers.depsgraph_update_post, _geometry_changed)):
        if callback in handlers:
            handlers.remove(callback)
