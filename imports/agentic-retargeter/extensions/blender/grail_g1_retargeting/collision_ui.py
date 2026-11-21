"""Explicit external Newton queries and read-only collision overlays."""
import json
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty

from . import dashboard_ui, collision_live, collision_sequence_ui
from .inspection import blender_collision as collision
from .inspection import blender_collision_sequence as sequence


def _config(scene):
    value = scene.grail_collision.config_path.strip()
    if not value:
        raise ValueError('Choose a collision config JSON with explicit mesh pairs and runtime.')
    config = json.loads(Path(bpy.path.abspath(value)).expanduser().read_text())
    settings = scene.grail_collision
    if settings.contact_map_mode != 'CONFIG':
        contact_map = dict(config.get('contact_map', {}))
        contact_map['enabled'] = settings.contact_map_mode == 'MEASURED'
        if contact_map['enabled']:
            contact_map['threshold_m'] = settings.contact_map_threshold_m
        config['contact_map'] = contact_map
    return config


def _path(value, label):
    if not value.strip():
        raise ValueError('Choose ' + label + '.')
    return Path(bpy.path.abspath(value)).expanduser().absolute()


def _sync(scene):
    # Reuse an explicitly enabled preview; never execute embedded scene code here.
    dashboard_ui._sync_enabled_preview(scene, required=True)
    bpy.context.view_layer.update()


def _failure(operator, scene, error):
    scene.grail_collision.message = str(error)
    operator.report({'ERROR'}, str(error))
    return {'CANCELLED'}


def _canonical_views(context):
    """Remember existing diagnostic views before replacement removes a camera."""
    return _canonical_views_for_scene(context.scene)


def _canonical_views_for_scene(scene):
    views = []
    for window in bpy.context.window_manager.windows:
        if window.scene != scene:
            continue
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                camera = area.spaces.active.camera
                if camera and camera.get('grail_collision_role') == 'canonical_camera':
                    views.append(area)
    return views


def _show_canonical(area, scene, report):
    camera = scene.objects.get(report.get('canonical_camera', ''))
    view = report.get('canonical_view')
    if camera is None or not view:
        raise ValueError('Add explicit hands with wrist and mesh bindings to the collision config, then refresh the display.')
    space = area.spaces.active
    space.use_local_camera = True
    space.camera = camera
    space.show_region_toolbar = False
    space.shading.type = 'SOLID'
    space.shading.color_type = 'VERTEX' if report.get('contact_map', {}).get('enabled') else 'MATERIAL'
    space.overlay.show_extras = False
    space.overlay.show_floor = False
    space.overlay.show_axis_x = False
    space.overlay.show_axis_y = False
    space.clip_start, space.clip_end = .001, 100000.
    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
    aspect = max(.1, region.width / max(1, region.height)) if region else 1.
    space.region_3d.view_rotation = camera.matrix_world.to_quaternion()
    space.region_3d.view_location = view['center']
    space.region_3d.view_distance = max(view['height'], view['width'] / aspect) * .78
    space.region_3d.view_perspective = 'ORTHO'
    area.tag_redraw()


def _refresh_canonical_views(views, scene, report):
    if report.get('canonical_camera'):
        for area in views:
            _show_canonical(area, scene, report)


def apply_live_report(scene, snapshot, result, config):
    """Apply one verified live snapshot while preserving the user's navigation."""
    views = _canonical_views_for_scene(scene)
    navigation = [(area, area.spaces.active.region_3d.view_rotation.copy(),
        area.spaces.active.region_3d.view_location.copy(),
        area.spaces.active.region_3d.view_distance) for area in views]
    if sequence.status(scene).get('enabled'):
        report = sequence.update_frame(scene, snapshot, result, config)
    else:
        report = collision.display_report(scene, snapshot, result, config=config)
    _refresh_canonical_views(views, scene, report)
    for area, rotation, location, distance in navigation:
        region = area.spaces.active.region_3d
        region.view_rotation, region.view_location, region.view_distance = rotation, location, distance
    return report


class GRAILCollisionSettings(bpy.types.PropertyGroup):
    config_path: StringProperty(name='Collision config JSON', subtype='FILE_PATH')
    output_dir: StringProperty(name='New query folder', subtype='DIR_PATH')
    input_path: StringProperty(name='New input JSON', subtype='FILE_PATH')
    report_path: StringProperty(name='Result JSON', subtype='FILE_PATH')
    export_path: StringProperty(name='New evidence JSON', subtype='FILE_PATH')
    message: StringProperty(name='Status', options={'SKIP_SAVE'})
    show_pairs: BoolProperty(name='Pair counts', default=False)
    show_files: BoolProperty(name='External query / evidence', default=False)
    contact_map_mode: EnumProperty(name='Canonical map', default='CONFIG', items=(
        ('CONFIG', 'From config', 'Use the explicit contact_map configuration'),
        ('MEASURED', 'Measured proximity', 'Color canonical copies by original-surface distance'),
        ('OFF', 'Off', 'Keep canonical hand materials with points and normals')))
    contact_map_threshold_m: FloatProperty(name='Map range (m)', default=.005, min=.000001, max=1., precision=4)
    live_output_dir: StringProperty(name='Automatic query folder', subtype='DIR_PATH')
    remember_live_runtime: BoolProperty(name='Remember configured runtime', default=False,
        description='Allow this matching config and runtime to resume after reload; save Blender preferences to retain across restarts')
    sequence_output_dir: StringProperty(name='New sequence cache folder', subtype='DIR_PATH')
    sequence_cache_path: StringProperty(name='Sequence cache NPZ', subtype='FILE_PATH')
    sequence_save_path: StringProperty(name='New cache revision NPZ', subtype='FILE_PATH')
    show_sequence_files: BoolProperty(name='Sequence cache files', default=False)


class GRAILG1_OT_collision_run(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_run'
    bl_label = 'Run Newton Collision Query'
    bl_description = 'Query configured pairs with an external Python runtime; bounded foreground run, no motion edits'

    def execute(self, context):
        scene, settings = context.scene, context.scene.grail_collision
        try:
            if collision_live.status(scene)['enabled']:
                collision_live.refresh(scene)
                return {'FINISHED'}
            config = _config(scene)
            output = _path(settings.output_dir, 'a new query folder')
            _sync(scene)
            if sequence.status(scene).get('enabled'):
                snapshot = collision.extract_snapshot(scene, config)
                result = collision.run_query(snapshot, config.get('runtime'), output)
                apply_live_report(scene, snapshot, result, config)
                settings.message = 'Updated the current sequence contact frame.'
                return {'FINISHED'}
            views = _canonical_views(context)
            report = collision.build_collision(scene, config, output)
            _refresh_canonical_views(views, scene, report)
            settings.message = f'Newton snapshot at source {report["source_frame"]:g}. Inspect declared pairs; no motion was changed.'
            self.report({'INFO'}, settings.message)
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_OT_collision_input(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_input'
    bl_label = 'Export Collision Input'
    bl_description = 'Write the current bound original meshes for an external query without running Newton'

    def execute(self, context):
        scene = context.scene
        try:
            _sync(scene)
            snapshot = collision.extract_snapshot(scene, _config(scene))
            path = _path(scene.grail_collision.input_path, 'a new input JSON file')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                json.dump(snapshot, stream, allow_nan=False)
                stream.write('\n')
            scene.grail_collision.message = 'Saved original-mesh query input: ' + path.name
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_OT_collision_load(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_load'
    bl_label = 'Load Newton Result'
    bl_description = 'Load results only when their mesh, pose, frame and pair bindings match the current scene'

    def execute(self, context):
        scene = context.scene
        try:
            _sync(scene)
            config = _config(scene)
            snapshot = collision.extract_snapshot(scene, config)
            path = _path(scene.grail_collision.report_path, 'a result JSON file')
            report = json.loads(path.read_text())
            # Exported Blender evidence wraps the original backend result.
            apply_live_report(scene, snapshot, report.get('result', report), config)
            scene.grail_collision.message = 'Loaded current-frame Newton contact samples.'
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_OT_collision_export(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_export'
    bl_label = 'Export Collision Evidence'

    def execute(self, context):
        scene = context.scene
        try:
            if sequence.status(scene).get('enabled'):
                raise ValueError('Use Save Sequence Cache Revision to export the sequence contact evidence.')
            collision.export_collision(scene, _path(scene.grail_collision.export_path, 'a new evidence JSON file'))
            scene.grail_collision.message = 'Saved current collision evidence.'
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_OT_collision_clear(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_clear'
    bl_label = 'Clear Collision Overlay'

    def execute(self, context):
        try:
            collision_live.disable(context.scene)
            sequence.clear(context.scene)
            collision.clear_collision(context.scene)
            context.scene.grail_collision.message = 'Cleared generated collision diagnostics.'
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, context.scene, error)


class GRAILG1_OT_collision_live_enable(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_live_enable'
    bl_label = 'Enable Automatic Contacts'
    bl_description = 'Query the configured current pose after loading or paused edits; keep Newton in a managed external runtime'

    def execute(self, context):
        try:
            collision_live.enable(context, remember=context.scene.grail_collision.remember_live_runtime)
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, context.scene, error)


class GRAILG1_OT_collision_live_disable(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_live_disable'
    bl_label = 'Disable Automatic Contacts'

    def execute(self, context):
        collision_live.disable(context.scene)
        return {'FINISHED'}


class GRAILG1_OT_collision_display(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_display'
    bl_label = 'Refresh Contact Display'
    bl_description = 'Rebuild points, normals and bound canonical hands from the current result without running Newton'

    def execute(self, context):
        scene = context.scene
        try:
            _sync(scene)
            if sequence.status(scene).get('enabled'):
                sequence.update(scene)
                scene.grail_collision.message = sequence.status(scene)['message']
                return {'FINISHED'}
            report = collision.current_report(scene)
            if not collision.collision_is_current(scene, report):
                raise ValueError('Contact snapshot is missing or stale; run or load a current query first.')
            config = _config(scene)
            snapshot = collision.extract_snapshot(scene, config)
            views = _canonical_views(context)
            report = collision.display_report(scene, snapshot, report['result'], config=config)
            _refresh_canonical_views(views, scene, report)
            scene.grail_collision.message = 'Refreshed contact display using the existing current query.'
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_OT_collision_canonical_view(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_canonical_view'
    bl_label = 'Show Canonical Hand View'
    bl_description = 'Keep articulated canonical hands and matching per-frame contacts beside the mounted robot'

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def execute(self, context):
        scene = context.scene
        try:
            views = _canonical_views(context)
            _sync(scene)
            report = sequence.status(scene)
            if not report.get('enabled'):
                config = _config(scene)
                prior = collision.current_report(scene)
                was_current = collision.collision_is_current(scene, prior)
                report = sequence.setup(scene, config)
                if was_current:
                    snapshot = collision.extract_snapshot(scene, config)
                    report = sequence.update_frame(scene, snapshot, prior['result'], config)
            if not report.get('canonical_camera'):
                raise ValueError('Add explicit hands with wrist and mesh bindings to the config.')
            if views:
                _refresh_canonical_views(views, scene, report)
            else:
                original = context.area
                before = {area.as_pointer() for area in context.screen.areas}
                if bpy.ops.screen.area_split(direction='VERTICAL', factor=.52) != {'FINISHED'}:
                    raise ValueError('Could not split the current viewport.')
                added = [area for area in context.screen.areas if area.as_pointer() not in before]
                if len(added) != 1:
                    raise ValueError('Could not identify the new canonical viewport.')
                left, right = sorted((original, added[0]), key=lambda area: area.x)
                left.spaces.active.show_region_ui = False
                right.spaces.active.show_region_ui = True
                _show_canonical(right, scene, report)
            scene.grail_collision.message = 'Mounted and wrist-local views share the same frame, contact points and Newton normals.'
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_OT_collision_sequence_build(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_sequence_build'
    bl_label = 'Build Full Sequence Contact Cache'
    bl_description = 'Compute exact integer-frame contacts and maps for the scene range; preserve source animation'

    def execute(self, context):
        scene = context.scene
        try:
            output = _path(scene.grail_collision.sequence_output_dir, 'a new sequence cache folder')
            config = _config(scene)
            _sync(scene)
            # A sequence capture owns frame evaluation until the saved frame is
            # restored; no duplicate automatic single-frame worker is needed.
            views = _canonical_views(context)
            with collision_live.suspend_queries(scene):
                result = collision_sequence_ui.build_cache(scene, config, output, sync=_sync)
            path = Path(result['cache_path'])
            scene.grail_collision.sequence_cache_path = str(path)
            sequence.setup(scene, config)
            report = sequence.attach_cache(scene, path)
            _refresh_canonical_views(views, scene, report)
            scene.grail_collision.message = 'Full sequence contact cache loaded.'
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_OT_collision_sequence_load(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_sequence_load'
    bl_label = 'Load Sequence Contact Cache'

    def execute(self, context):
        scene = context.scene
        try:
            _sync(scene)
            views = _canonical_views(context)
            sequence.setup(scene, _config(scene))
            report = sequence.attach_cache(scene, _path(scene.grail_collision.sequence_cache_path, 'a sequence cache NPZ'))
            _refresh_canonical_views(views, scene, report)
            scene.grail_collision.message = 'Sequence contacts follow the time line; unmatched poses remain visible with pending maps.'
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_OT_collision_sequence_save(bpy.types.Operator):
    bl_idname = 'grail_g1.collision_sequence_save'
    bl_label = 'Save Sequence Cache Revision'

    def execute(self, context):
        scene = context.scene
        try:
            path = _path(scene.grail_collision.sequence_save_path, 'a new sequence cache revision NPZ')
            sequence.save_cache(scene, path)
            scene.grail_collision.sequence_cache_path = str(path)
            scene.grail_collision.message = 'Saved sequence contact cache with refreshed frames.'
            return {'FINISHED'}
        except Exception as error:
            return _failure(self, scene, error)


class GRAILG1_PT_collision(bpy.types.Panel):
    bl_label = 'Newton Collision Diagnostics'
    bl_idname = 'GRAILG1_PT_collision'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GRAIL G1'
    bl_order = 5
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout, settings = self.layout, context.scene.grail_collision
        layout.prop(settings, 'config_path')
        layout.prop(settings, 'output_dir')
        layout.operator('grail_g1.collision_run', icon='VIEWZOOM')
        auto = layout.box()
        live = collision_live.status(context.scene)
        auto.label(text='Automatic contacts on pause')
        auto.prop(settings, 'live_output_dir')
        auto.prop(settings, 'remember_live_runtime')
        auto.operator('grail_g1.collision_live_disable' if live['enabled'] else 'grail_g1.collision_live_enable',
                      icon='PAUSE' if live['enabled'] else 'PLAY')
        if live['message']:
            import textwrap
            for line in textwrap.wrap(live['message'], width=42):
                auto.label(text=line)
        layout.prop(settings, 'contact_map_mode')
        if settings.contact_map_mode == 'MEASURED':
            layout.prop(settings, 'contact_map_threshold_m')
        seq = sequence.status(context.scene)
        box = layout.box()
        box.label(text='Canonical sequence playback')
        if seq.get('enabled'):
            box.label(text=seq.get('message', 'Canonical hands follow the current frame.'))
            box.label(text=f'Source {seq.get("source_frame", 0):g} | {seq.get("cached_frames", 0)} cached frames')
            box.label(text='Matching contact cache' if seq.get('cache_hit') else 'Current hand pose | contact data pending',
                      icon='CHECKMARK' if seq.get('cache_hit') else 'TIME')
        box.operator('grail_g1.collision_canonical_view', icon='WINDOW')
        box.prop(settings, 'show_sequence_files', icon='TRIA_DOWN' if settings.show_sequence_files else 'TRIA_RIGHT')
        if settings.show_sequence_files:
            box.prop(settings, 'sequence_output_dir')
            box.operator('grail_g1.collision_sequence_build')
            box.prop(settings, 'sequence_cache_path')
            box.operator('grail_g1.collision_sequence_load')
            box.prop(settings, 'sequence_save_path')
            box.operator('grail_g1.collision_sequence_save')
        report = None if seq.get('enabled') else collision.current_report(context.scene)
        if report:
            current = collision.collision_is_current(context.scene, report)
            box = layout.box()
            box.label(text='Current snapshot' if current else 'STALE: run or load a matching query', icon='CHECKMARK' if current else 'ERROR')
            box.label(text=f'Source frame {report["source_frame"]:g}')
            visual = report['visualization']
            box.label(text=f'{visual["total_contacts"]:,} candidates / {len(report["result"]["pairs"])} pairs')
            box.label(text=f'{visual["displayed_contacts"]:,} point pairs displayed')
            normals_available = any('normals_world' in pair for pair in report['result']['pairs'])
            if normals_available:
                box.label(text=f'{visual.get("displayed_normals", 0)} cyan arrows: Newton normals, B to A')
            else:
                box.label(text='Normals unavailable in this older result.')
            box.label(text='Green: A | Orange: B | Red: paired gap')
            contact_map = report.get('contact_map', {})
            if contact_map.get('enabled'):
                box.label(text=f'Canonical map: 0 to {contact_map["threshold_m"] * 1000:g} mm')
                box.label(text='Yellow: near | Blue: far | unsigned distance')
            box.prop(settings, 'show_pairs', icon='TRIA_DOWN' if settings.show_pairs else 'TRIA_RIGHT')
            if settings.show_pairs:
                for pair in report['result']['pairs']:
                    box.label(text=f'{pair["mesh_a"]} / {pair["mesh_b"]}: {pair["contact_count"]}')
            row = layout.row()
            row.enabled = current
            row.operator('grail_g1.collision_canonical_view', icon='WINDOW')
            row = layout.row()
            row.enabled = current
            row.operator('grail_g1.collision_display', icon='FILE_REFRESH')
        layout.operator('grail_g1.collision_clear')
        files = layout.box()
        files.prop(settings, 'show_files', icon='TRIA_DOWN' if settings.show_files else 'TRIA_RIGHT')
        if settings.show_files:
            for prop, operator in [('input_path', 'collision_input'), ('report_path', 'collision_load'), ('export_path', 'collision_export')]:
                files.prop(settings, prop)
                files.operator('grail_g1.' + operator)
        layout.label(text='No samples does not prove clearance.')
        if settings.message:
            import textwrap
            for line in textwrap.wrap(settings.message, width=46):
                layout.label(text=line)


_CLASSES = (GRAILCollisionSettings, GRAILG1_OT_collision_run, GRAILG1_OT_collision_input,
            GRAILG1_OT_collision_load, GRAILG1_OT_collision_export, GRAILG1_OT_collision_clear,
            GRAILG1_OT_collision_live_enable, GRAILG1_OT_collision_live_disable,
            GRAILG1_OT_collision_display, GRAILG1_OT_collision_canonical_view,
            GRAILG1_OT_collision_sequence_build, GRAILG1_OT_collision_sequence_load,
            GRAILG1_OT_collision_sequence_save,
            GRAILG1_PT_collision)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.grail_collision = PointerProperty(type=GRAILCollisionSettings)
    collision.register_handlers()
    sequence.register_handlers(sync_callback=_sync, config_callback=_config)
    collision_live.register()


def unregister():
    collision_live.unregister()
    sequence.unregister_handlers()
    collision.unregister_handlers()
    if hasattr(bpy.types.Scene, 'grail_collision'):
        del bpy.types.Scene.grail_collision
    for cls in reversed(_CLASSES):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
