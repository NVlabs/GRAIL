"""GRAIL G1 native joint-curve refinement for Blender 4.2–4.3."""
from datetime import datetime, timezone
from pathlib import Path
import json

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import runtime_adapter as adapter


bl_info = {
    'name': 'GRAIL G1 Retargeting Refinement',
    'author': 'NVIDIA',
    'version': (0, 1, 0),
    'blender': (4, 2, 0),
    'location': '3D View / Graph Editor > Sidebar > GRAIL G1',
    'description': 'Bounded native G1 joint curves and UNREVIEWED native exports',
    'category': 'Animation',
}


def _failure(operator, context, exc):
    message = str(exc) or type(exc).__name__
    context.scene.grail_g1.last_message = message
    operator.report({'ERROR'}, message)
    return {'CANCELLED'}


def _success(operator, context, message):
    context.scene.grail_g1.last_message = message
    operator.report({'INFO'}, message)
    return {'FINISHED'}


def _new_output(settings):
    raw = settings.output_dir.strip()
    if not raw:
        raise ValueError('Choose a new output directory.')
    output = Path(bpy.path.abspath(raw)).expanduser().resolve()
    if output.exists():
        raise ValueError('Output directory already exists; choose a new directory for each checkpoint.')
    return output


def _active_joint(settings):
    if not 0 <= settings.joint_index < len(settings.joints):
        raise ValueError('Load and select a native joint first.')
    return settings.joints[settings.joint_index]


def _load_joint_rows(settings, native, model):
    joints = adapter.native_io._model_joints(model)
    hinges = [joint for joint in joints if joint['joint_type'] == 3 and joint['limited']]
    for joint in hinges:
        adapter.native_io._editable_joint(joint)
    if not hinges:
        raise ValueError('The compiled model has no explicitly limited native hinge joints.')
    count = len(native['qpos'])
    if count < 4:
        raise ValueError('At least four source frames are required for two interior closure keys.')
    settings.joints.clear()
    settings.window_start, settings.window_end = 0, count - 1
    settings.closure_onset = max(1.0, (count - 1) / 3)
    settings.closure_complete = min(float(count - 2), 2 * (count - 1) / 3)
    for joint in hinges:
        item = settings.joints.add()
        item.name = joint['joint']
        item.address = joint['address']
        item.minimum, item.maximum = joint['limits']
        item.onset_rad = float(native['qpos'][int(settings.closure_onset), joint['address']])
        item.closed_rad = float(native['qpos'][int(settings.closure_complete), joint['address']])
        item.enabled = False  # No implicit finger/arm scope.
    settings.joint_index = 0
    settings.binding_native_sha256 = adapter.runtime.digest(Path(bpy.path.abspath(settings.native_path)).expanduser())
    settings.binding_model_sha256 = adapter.runtime.digest(Path(bpy.path.abspath(settings.model_path)).expanduser())


class GRAILG1Joint(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='Edit', default=False)
    address: IntProperty(name='Native qpos address', default=0)
    minimum: FloatProperty(name='Original minimum (rad)')
    maximum: FloatProperty(name='Original maximum (rad)')
    onset_rad: FloatProperty(name='Onset value (rad)', precision=5)
    closed_rad: FloatProperty(name='Closure value (rad)', precision=5)


class GRAILG1Settings(bpy.types.PropertyGroup):
    native_path: StringProperty(name='Native G1 NPZ', subtype='FILE_PATH', description='Native G1 qpos/fps checkpoint from the external retargeting pipeline')
    model_path: StringProperty(name='Actual model snapshot NPZ', subtype='FILE_PATH', description='Compiled native model with actual joint names, qpos addresses and original limits')
    plan_path: StringProperty(name='Curve plan JSON', subtype='FILE_PATH')
    output_dir: StringProperty(name='New output directory', subtype='DIR_PATH')
    substeps: IntProperty(name='Subframe samples / interval', default=4, min=2, max=128)
    joints: CollectionProperty(type=GRAILG1Joint)
    joint_index: IntProperty(default=0)
    binding_native_sha256: StringProperty(options={'HIDDEN'})
    binding_model_sha256: StringProperty(options={'HIDDEN'})
    window_start: IntProperty(name='Window start (source)', default=0, min=0)
    window_end: IntProperty(name='Window end (source)', default=20, min=1)
    closure_onset: FloatProperty(name='Closure onset (source)', default=6, min=0, precision=3)
    closure_complete: FloatProperty(name='Closure complete (source)', default=13, min=0, precision=3)
    protect_closure: BoolProperty(name='Protect closure milestones', default=False, description='Require the two interior closure keys to retain their time and value on export')
    auto_smoothing: EnumProperty(name='Handle smoothing', items=[('NONE', 'None (default)', 'Bezier / Auto Clamped; does not validate contact'), ('CONT_ACCEL', 'Continuous Acceleration (optional)', 'May influence neighboring intervals; review actual native replay')], default='NONE')
    key_source_frame: FloatProperty(name='Interior key (source frame)', default=6, min=0, precision=3)
    key_value_rad: FloatProperty(name='Interior key value (rad)', precision=5)
    last_message: StringProperty(name='Status', default='Select native/model inputs. Initial robot conversion remains external.')
    last_export_dir: StringProperty(name='Last exported checkpoint', subtype='DIR_PATH')
    review_author: StringProperty(name='Reviewer')
    review_scope: StringProperty(name='Reviewed scope')
    review_evidence: StringProperty(name='Evidence / notes', description='Record actual contact, mesh, source and temporal evidence; not an automatic validation')
    review_decision: EnumProperty(name='Manual decision', items=[('UNREVIEWED', 'Unreviewed', ''), ('PARTIAL', 'Partial', ''), ('REJECTED', 'Rejected', ''), ('ACCEPTED_FOR_SCOPE', 'Accepted for stated scope', '')], default='UNREVIEWED')


class GRAILG1_OT_validate_inputs(bpy.types.Operator):
    bl_idname = 'grail_g1.validate_inputs'
    bl_label = 'Validate Native Inputs and Plan'
    bl_description = 'Read and validate input bindings, limits and windows without running scene scripts'

    def execute(self, context):
        try:
            native, model, plan, channels, paths = adapter.load_inputs(context.scene.grail_g1)
            adapter.validate_plan_protection(plan)
            preview = adapter.passive_parent_check(native, model)
            return _success(self, context, f'{len(native["qpos"])} source frames; {len(channels)} bounded joints. ' + ('Compatible preview inputs; enable explicitly.' if preview else 'Controls-only scene.'))
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_OT_load_joints(bpy.types.Operator):
    bl_idname = 'grail_g1.load_joints'
    bl_label = 'Load Actual Native Joint Names'
    bl_description = 'Populate the plan builder from the selected native archive and compiled model'

    def execute(self, context):
        try:
            settings = context.scene.grail_g1
            native, model, plan, channels, paths = adapter.load_inputs(settings, with_plan=False)
            _load_joint_rows(settings, native, model)
            return _success(self, context, 'Loaded original limited hinges. Select only the joints this edit may change.')
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_OT_create_plan(bpy.types.Operator):
    bl_idname = 'grail_g1.create_plan'
    bl_label = 'Write New Curve Plan'
    bl_description = 'Write selected native joints, protected source endpoints and authored interior closure keys; never overwrite a plan'

    def execute(self, context):
        try:
            settings = context.scene.grail_g1
            native, model, unused, unused_channels, paths = adapter.load_inputs(settings, with_plan=False)
            if not settings.plan_path.strip():
                raise ValueError('Choose a new curve plan JSON path.')
            target = Path(bpy.path.abspath(settings.plan_path)).expanduser().resolve()
            if target.exists():
                raise ValueError('Curve plan already exists; select a new JSON path to preserve it.')
            if settings.binding_native_sha256 != adapter.runtime.digest(paths['native']) or settings.binding_model_sha256 != adapter.runtime.digest(paths['model']):
                raise ValueError('Native/model input changed; load the actual joint names again.')
            start, end = settings.window_start, settings.window_end
            onset, complete = settings.closure_onset, settings.closure_complete
            if not 0 <= start < onset < complete < end < len(native['qpos']):
                raise ValueError('Use window start < closure onset < closure complete < window end within the original source frames.')
            by_name = {joint['joint']: joint for joint in adapter.native_io._model_joints(model)}
            channels = []
            for item in settings.joints:
                if not item.enabled:
                    continue
                joint = by_name[item.name]
                address = joint['address']
                channel = dict(joint=item.name, window=[start, end], interpolation='BEZIER', auto_smoothing=settings.auto_smoothing,
                               keys=[[start, float(native['qpos'][start, address])], [onset, item.onset_rad], [complete, item.closed_rad], [end, float(native['qpos'][end, address])]])
                if settings.protect_closure:
                    channel['protected_source_frames'] = [onset, complete]
                channels.append(channel)
            plan = dict(schema_version=1, channels=channels, authoring_note='Authored closure timing; source frame 0 maps to Blender frame 1. Endpoints come from the native checkpoint.')
            adapter.native_io.resolve_channels(native, model, plan)
            adapter.validate_plan_protection(plan)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('x') as stream:
                stream.write(json.dumps(plan, indent=2) + '\n')
            return _success(self, context, f'Wrote {len(channels)} native channels. All other joints and window endpoints stay frozen.')
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_OT_create_controls(bpy.types.Operator):
    bl_idname = 'grail_g1.create_controls'
    bl_label = 'Create Native Curve Controls'
    bl_description = 'Create the contact edit kit controls and original timeline without clearing the scene or enabling embedded preview code'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.grail_g1
            adapter.reject_conflicts()
            _new_output(settings)
            native, model, plan, channels, paths = adapter.load_inputs(settings)
            adapter.validate_plan_protection(plan)
            adapter.runtime.setup(native, model, plan, adapter.provenance(paths))
            adapter.bind_controller(context.scene)
            adapter.embed()
            # Supplied plans are also usable without first visiting the builder.
            settings.joints.clear()
            for channel in channels:
                item = settings.joints.add()
                item.name = channel['joint']
                item.address = channel['address']
                item.minimum, item.maximum = channel['limits']
                item.enabled = True
            settings.joint_index = 0
            return _success(self, context, 'Native curves created; source 0 = Blender 1. Save/export creates a new UNREVIEWED checkpoint.')
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_OT_enable_preview(bpy.types.Operator):
    bl_idname = 'grail_g1.enable_preview'
    bl_label = 'Enable Exact Preview (Existing G1 Scene)'
    bl_description = 'Explicitly execute this compatible portable scene\'s embedded preview runtime; use only a trusted scene'

    confirmed: BoolProperty(default=False, options={'HIDDEN'})

    def invoke(self, context, event):
        self.confirmed = True
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        try:
            if not self.confirmed:
                raise ValueError('Explicit consent required to run embedded portable scene code; invoke the button or pass confirmed=True.')
            adapter.enable_preview()
            return _success(self, context, 'Exact native FK preview enabled for this compatible portable G1 scene. Contact is not validated.')
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_OT_select_joint(bpy.types.Operator):
    bl_idname = 'grail_g1.select_joint'
    bl_label = 'Select Joint Curve'
    bl_description = 'Select this native joint and its interior keys in the Graph Editor; protected endpoints remain checked at export'
    open_graph: BoolProperty(name='Open Graph Editor', default=False)
    joint_name: StringProperty(name='Native joint', default='')

    def execute(self, context):
        try:
            settings = context.scene.grail_g1
            state = adapter.initialize()
            name = self.joint_name or _active_joint(settings).name
            channel = next((item for item in state['channels'] if item['joint'] == name), None)
            if channel is None:
                raise ValueError('This joint is not in the current curve plan: ' + name)
            controller = bpy.data.objects[adapter.runtime.CONTROLLER]
            action = controller.animation_data.action
            path = f'["qpos_{channel["address"]:02d}"]'
            for selected in context.selected_objects:
                selected.select_set(False)
            controller.select_set(True)
            context.view_layer.objects.active = controller
            for fc in action.fcurves:
                chosen = fc.data_path == path
                fc.select = chosen
                fc.hide = not chosen
                for key in fc.keyframe_points:
                    selected = chosen and channel['window'][0] < key.co[0] - 1 < channel['window'][1]
                    key.select_control_point = key.select_left_handle = key.select_right_handle = selected
            if self.open_graph and context.area is not None:
                context.area.type = 'GRAPH_EDITOR'
            if context.screen is not None:
                for area in context.screen.areas:
                    if area.type == 'GRAPH_EDITOR':
                        area.spaces.active.dopesheet.show_only_selected = True
                        area.tag_redraw()
            return _success(self, context, name + ': edit interior source times at Blender frame = source + 1.')
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_OT_set_key(bpy.types.Operator):
    bl_idname = 'grail_g1.set_key'
    bl_label = 'Set Interior Native Key'
    bl_description = 'Insert/update a selected joint key inside its declared window and original native limits'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.grail_g1
            state = adapter.initialize()
            adapter.runtime.validate_curves()
            adapter.validate_protected_keys()
            name = _active_joint(settings).name
            channel = next((item for item in state['channels'] if item['joint'] == name), None)
            if channel is None:
                raise ValueError('Selected joint is not in the current curve plan.')
            frame, value = settings.key_source_frame, settings.key_value_rad
            if not channel['window'][0] < frame < channel['window'][1]:
                raise ValueError('Only strict interior keys may be edited; window endpoints are protected.')
            initial = next(item for item in state['plan']['channels'] if item['joint'] == name)
            if any(abs(frame - time) < 1e-6 for time in initial.get('protected_source_frames', [])):
                raise ValueError('This authored milestone is protected in the curve plan.')
            adapter.native_io.check_sample_limits(state['model'], [channel['address']], [[value]])
            action = bpy.data.objects[adapter.runtime.CONTROLLER].animation_data.action
            fc = action.fcurves.find(f'["qpos_{channel["address"]:02d}"]', index=0)
            key = fc.keyframe_points.insert(frame + 1, value, options={'FAST'})
            key.interpolation = 'BEZIER'
            key.handle_left_type = key.handle_right_type = 'AUTO_CLAMPED'
            fc.update()
            context.scene.frame_set(context.scene.frame_current)
            return _success(self, context, f'{name}: authored key at source {frame:g} (Blender {frame + 1:g}). Export to check sampled limits.')
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_OT_export_native(bpy.types.Operator):
    bl_idname = 'grail_g1.export_native'
    bl_label = 'Save Scene + Export UNREVIEWED Native'
    bl_description = 'Save a new .blend and native NPZ with provenance, frozen/limit checks and subframe comparisons; changes Blender\'s current file path'

    def execute(self, context):
        try:
            settings = context.scene.grail_g1
            output = _new_output(settings)
            # Set before saving so the reopened checkpoint remembers its native output.
            old_last = settings.last_export_dir
            settings.last_export_dir = str(output)
            try:
                report = adapter.export_checkpoint(output, settings.substeps)
            except Exception:
                settings.last_export_dir = old_last
                raise
            return _success(self, context, f'UNREVIEWED native and editable scene saved: {output.name}; {report["changed_values"]} changed qpos values. Native archive reopened and checked.')
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_OT_record_review(bpy.types.Operator):
    bl_idname = 'grail_g1.record_review'
    bl_label = 'Write Separate Manual Review Record'
    bl_description = 'Record a human decision and its evidence for the last exported NPZ; does not auto-accept or rewrite the candidate'

    def execute(self, context):
        try:
            settings = context.scene.grail_g1
            if not settings.last_export_dir.strip():
                raise ValueError('Export a native candidate first.')
            directory = Path(bpy.path.abspath(settings.last_export_dir)).expanduser().resolve()
            candidate = directory / 'candidate_UNREVIEWED.npz'
            if not candidate.is_file():
                raise ValueError('Last exported native candidate is missing.')
            if not settings.review_author.strip() or not settings.review_scope.strip() or not settings.review_evidence.strip():
                raise ValueError('A manual review needs a reviewer, scope and evidence/notes.')
            now = datetime.now(timezone.utc)
            record = dict(schema_version=1, kind='manual_review_record', candidate_path=str(candidate), candidate_sha256=adapter.runtime.digest(candidate),
                          reviewer=settings.review_author.strip(), scope=settings.review_scope.strip(), decision=settings.review_decision,
                          evidence=settings.review_evidence.strip(), created_at=now.isoformat(), automatic_acceptance=False,
                          candidate_archive_status='UNREVIEWED; this separate record does not rewrite the native archive')
            target = directory / ('manual_review_' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
            with target.open('x') as stream:
                stream.write(json.dumps(record, indent=2) + '\n')
            return _success(self, context, 'Manual review record saved separately with the exported candidate hash.')
        except Exception as exc:
            return _failure(self, context, exc)


class GRAILG1_UL_joints(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, 'enabled', text='')
        row.label(text=item.name)


def _draw_panel(panel, context):
    layout, settings = panel.layout, context.scene.grail_g1
    layout.label(text='Source frame 0 = Blender frame 1', icon='TIME')
    inputs = layout.box()
    for field in ('native_path', 'model_path', 'plan_path', 'output_dir'):
        inputs.prop(settings, field)
    inputs.operator('grail_g1.validate_inputs', icon='CHECKMARK')
    builder = layout.box()
    builder.label(text='Optional curve plan builder')
    builder.operator('grail_g1.load_joints')
    if settings.joints:
        builder.template_list('GRAILG1_UL_joints', '', settings, 'joints', settings, 'joint_index', rows=5)
        joint = settings.joints[min(max(settings.joint_index, 0), len(settings.joints) - 1)]
        builder.label(text=f'Native limits: {joint.minimum:.4f} to {joint.maximum:.4f} rad')
        builder.prop(joint, 'onset_rad')
        builder.prop(joint, 'closed_rad')
        row = builder.row(align=True)
        row.prop(settings, 'window_start')
        row.prop(settings, 'window_end')
        builder.prop(settings, 'closure_onset')
        builder.prop(settings, 'closure_complete')
        builder.prop(settings, 'protect_closure')
        builder.prop(settings, 'auto_smoothing')
        builder.operator('grail_g1.create_plan')
    layout.operator('grail_g1.create_controls', icon='ACTION')
    layout.operator('grail_g1.enable_preview', icon='PLAY')
    if adapter.runtime.CONTROLLER in bpy.data.objects:
        editor = layout.box()
        editor.label(text='Graph Editor: selected native joint')
        editor.operator('grail_g1.select_joint', text='Open Selected Joint in Graph Editor').open_graph = True
        editor.prop(settings, 'key_source_frame')
        editor.prop(settings, 'key_value_rad')
        editor.operator('grail_g1.set_key')
        editor.label(text='Endpoints and frozen reference are protected.')
    layout.prop(settings, 'substeps')
    layout.operator('grail_g1.export_native', icon='EXPORT')
    layout.label(text='Every export starts UNREVIEWED.')
    layout.label(text='Contact / mesh / source checks remain external.')
    review = layout.box()
    review.label(text='Optional manual review record')
    for field in ('review_author', 'review_scope', 'review_decision', 'review_evidence'):
        review.prop(settings, field)
    review.operator('grail_g1.record_review')
    layout.label(text='Initial robot conversion: external pipeline.')
    layout.label(text='MuJoCo/SciPy pose fitting: external CLI.')
    if settings.last_message:
        box = layout.box()
        # Narrow sidebars wrap long messages at natural word boundaries.
        import textwrap
        for line in textwrap.wrap(settings.last_message, width=48):
            box.label(text=line)


class GRAILG1_PT_main(bpy.types.Panel):
    bl_label = 'G1 Retargeting Refinement'
    bl_idname = 'GRAILG1_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GRAIL G1'
    draw = _draw_panel


class GRAILG1_PT_graph(bpy.types.Panel):
    bl_label = 'G1 Retargeting Refinement'
    bl_idname = 'GRAILG1_PT_graph'
    bl_space_type = 'GRAPH_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'GRAIL G1'
    draw = _draw_panel


@persistent
def _on_load(_unused):
    adapter.reset()  # Reopening never executes embedded preview code.


_CLASSES = (GRAILG1Joint, GRAILG1Settings, GRAILG1_OT_validate_inputs,
            GRAILG1_OT_load_joints, GRAILG1_OT_create_plan, GRAILG1_OT_create_controls,
            GRAILG1_OT_enable_preview, GRAILG1_OT_select_joint, GRAILG1_OT_set_key,
            GRAILG1_OT_export_native, GRAILG1_OT_record_review, GRAILG1_UL_joints,
            GRAILG1_PT_main, GRAILG1_PT_graph)


def register():
    adapter.require_version()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.grail_g1 = PointerProperty(type=GRAILG1Settings)
    if _on_load not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load)


def unregister():
    adapter.reset()
    if _on_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load)
    if hasattr(bpy.types.Scene, 'grail_g1'):
        del bpy.types.Scene.grail_g1
    for cls in reversed(_CLASSES):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
