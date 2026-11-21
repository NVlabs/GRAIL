"""Consent and scene-boundary adapter around the bundled contact edit kit.

The numerical curve evaluator and native exporter remain the kit's code.
Reading a .blend or inspecting inputs never executes its embedded Python.
"""
from pathlib import Path
import json
import sys

import bpy
import numpy as np

from .kit import blender_curves as runtime
from .kit import native_io


_original_parent_runtime = runtime.parent_runtime
_original_preview_handler = runtime.contact_edit_preview
_preview_runtime = None
TEXT_NAMES = (
    'ContactEdit_Source.npz.b64', 'ContactEdit_Model.npz.b64',
    'ContactEdit_Plan.json', 'ContactEdit_Provenance.json',
    'ContactEdit_native_io.py', 'ContactEdit_Runtime.py',
    'Enable_ContactEdit_Preview.py', 'Export_ContactEdit.py',
    'README_ContactEdit.txt',
)


def require_version():
    if not (4, 2, 0) <= bpy.app.version < (4, 4, 0):
        raise ValueError('Use Blender 4.2 or 4.3 (tested with 4.3.2); layered Actions in 4.4+ are unsupported.')


def passive_parent_check(native, model):
    """Compare native arrays and existing curves without running scene scripts."""
    markers = ('Enable_Exact_Native_Preview.py', 'NativeModel.npz.b64', 'PortableReference.npz.b64')
    present = any(name in bpy.data.texts for name in markers)
    present |= 'Native_Qpos_Controller' in bpy.data.objects
    if not present:
        return False
    if any(name not in bpy.data.texts for name in markers):
        raise ValueError('Incomplete portable G1 scene; use its matching original scene or a fresh controls-only scene.')
    if 'Native_Qpos_Controller' not in bpy.data.objects or 'Native_Qpos_Reference' not in bpy.data.actions:
        raise ValueError('Portable G1 scene is missing its native controller/reference.')
    if 'Native_Qpos_Controller' not in bpy.context.scene.objects:
        raise ValueError('The portable G1 controller belongs to another scene. Switch to its scene first.')
    source = runtime.unpack('PortableReference.npz.b64')
    scene_model = runtime.unpack('NativeModel.npz.b64')
    for key in ('qpos', 'fps', 'obj_R', 'obj_t', 'object_scale'):
        if key not in native or key not in source or not np.array_equal(native[key], source[key]):
            raise ValueError('Existing scene and selected native checkpoint differ: ' + key)
    # A reduced or unrelated model archive is not evidence of an exact preview.
    for key in ('joint_names', 'jnt_type', 'jnt_limited', 'jnt_range', 'jnt_qposadr'):
        if key not in scene_model or not np.array_equal(model[key], scene_model[key]):
            raise ValueError('Existing scene and selected model differ: ' + key)
    for key in model:
        if key in scene_model and not np.array_equal(model[key], scene_model[key]):
            raise ValueError('Existing scene and selected model differ: ' + key)
    runtime.validate_parent_controller()
    return True


def guarded_parent_runtime(native, model):
    passive_parent_check(native, model)
    if _preview_runtime is not None:
        _preview_runtime.validate_scene_contract()
    return _preview_runtime


runtime.parent_runtime = guarded_parent_runtime


@bpy.app.handlers.persistent
def scoped_preview(scene, *unused):
    controller = bpy.data.objects.get(runtime.CONTROLLER)
    if controller is None or controller.get('grail_g1_owner_scene') != scene or scene != bpy.context.scene:
        return
    _original_preview_handler(scene, *unused)


scoped_preview.__name__ = 'contact_edit_preview'
runtime.contact_edit_preview = scoped_preview


def bind_controller(scene):
    bpy.data.objects[runtime.CONTROLLER]['grail_g1_owner_scene'] = scene


def reset():
    global _preview_runtime
    for handler in list(bpy.app.handlers.frame_change_post):
        if handler is runtime.contact_edit_preview:
            bpy.app.handlers.frame_change_post.remove(handler)
    _preview_runtime = None
    runtime.STATE.clear()


def reject_conflicts():
    conflicts = [name for name in TEXT_NAMES if name in bpy.data.texts]
    conflicts += [name for name in (runtime.CONTROLLER, 'ContactEdit_FrozenReference') if name in bpy.data.objects]
    conflicts += [name for name in ('ContactEdit_AuthoredAction', runtime.REFERENCE) if name in bpy.data.actions]
    if conflicts:
        raise ValueError('Existing ContactEdit data must be preserved; reopen/export it or use another scene: ' + ', '.join(conflicts[:3]))


def load_inputs(settings, with_plan=True):
    require_version()
    paths = {}
    for key in ('native', 'model') + (('plan',) if with_plan else ()):
        raw = getattr(settings, key + '_path').strip()
        if not raw:
            raise ValueError('Choose a ' + key + ' file first.')
        path = Path(bpy.path.abspath(raw)).expanduser().resolve()
        if not path.is_file():
            raise ValueError('Missing ' + key + ' file: ' + str(path))
        paths[key] = path
    native = native_io.load_npz(paths['native'])
    model = native_io.load_npz(paths['model'])
    qpos = native_io._native_qpos(native)
    joints = native_io._model_joints(model)
    native_io._validate_native_binding(native, joints)
    widths = {0: 7, 1: 4, 2: 1, 3: 1}
    covered = []
    for joint in joints:
        covered.extend(range(joint['address'], joint['address'] + widths[joint['joint_type']]))
    if sorted(covered) != list(range(qpos.shape[1])):
        raise ValueError('Compiled model joint addresses do not cover native qpos exactly.')
    runtime.validate_source_quaternions(native, model)
    plan = json.loads(paths['plan'].read_text()) if with_plan else None
    channels = native_io.resolve_channels(native, model, plan) if with_plan else []
    passive_parent_check(native, model)
    return native, model, plan, channels, paths


def provenance(paths):
    result = {name: {'path': str(path), 'sha256': runtime.digest(path)} for name, path in paths.items()}
    for name, path in (('script', Path(runtime.__file__)), ('native_io', Path(native_io.__file__)), ('addon_adapter', Path(__file__))):
        result[name] = {'path': str(path), 'sha256': runtime.digest(path)}
    if bpy.data.filepath and Path(bpy.data.filepath).is_file():
        result['parent_blend'] = {'path': bpy.data.filepath, 'sha256': runtime.digest(bpy.data.filepath)}
    result['frame_mapping'] = 'source frame 0 = Blender frame 1'
    result['initial_global_retargeting'] = 'External robot retargeting pipeline; not performed by this extension'
    return result


def initialize():
    require_version()
    for name in TEXT_NAMES[:4]:
        if name not in bpy.data.texts:
            raise ValueError('No complete ContactEdit checkpoint in this scene. Create controls first.')
    if runtime.CONTROLLER not in bpy.data.objects:
        raise ValueError('ContactEdit_Curves is missing from the scene.')
    controller = bpy.data.objects[runtime.CONTROLLER]
    if controller.name not in bpy.context.scene.objects or controller.get('grail_g1_owner_scene') != bpy.context.scene:
        raise ValueError('These native controls belong to another scene or were not created by this extension. Use their original GRAIL G1 scene.')
    for name, path in (('ContactEdit_Runtime.py', Path(runtime.__file__)), ('ContactEdit_native_io.py', Path(native_io.__file__))):
        if name not in bpy.data.texts or bpy.data.texts[name].as_string() != path.read_text():
            raise ValueError('Embedded kit differs from the installed version; use its matching extension or recreate controls from the recorded native/plan inputs.')
    return runtime.initialize()


def enable_preview():
    """Only called by the explicit, confirmed enable-preview operator."""
    global _preview_runtime
    state = initialize()
    if not passive_parent_check(state['native'], state['model']):
        raise ValueError('Controls-only scene: exact preview needs an existing compatible portable G1 scene. No generic robot rig is built.')
    handlers = list(bpy.app.handlers.frame_change_post)
    previous_preview = _preview_runtime
    previous_state = dict(runtime.STATE)
    previous_module = sys.modules.get('grail_portable_native')
    transforms = [(obj, obj.matrix_basis.copy(), obj.rotation_mode) for obj in bpy.context.scene.objects]
    try:
        _preview_runtime = _original_parent_runtime(state['native'], state['model'])
        runtime.initialize()
    except Exception:
        _preview_runtime = previous_preview
        runtime.STATE.clear()
        runtime.STATE.update(previous_state)
        bpy.app.handlers.frame_change_post[:] = handlers
        for obj, matrix, mode in transforms:
            obj.rotation_mode = mode
            obj.matrix_basis = matrix
        if previous_module is None:
            sys.modules.pop('grail_portable_native', None)
        else:
            sys.modules['grail_portable_native'] = previous_module
        raise
    return _preview_runtime


def embed():
    """Preserve kit provenance; all advertised exporters use identical guards."""
    runtime.text('ContactEdit_native_io.py', Path(native_io.__file__).read_text())
    runtime.text('ContactEdit_Runtime.py', Path(runtime.__file__).read_text())
    loader = '''import bpy
if not hasattr(bpy.types.Scene, 'grail_g1'):
    raise RuntimeError('Install/enable the matching GRAIL G1 Retargeting Refinement extension first.')
'''
    runtime.text('Enable_ContactEdit_Preview.py', '# Running this text explicitly consents to execute the existing portable G1 scene runtime.\n' + loader + "\nbpy.ops.grail_g1.enable_preview(confirmed=True)\n")
    runtime.text('Export_ContactEdit.py', '# Explicit manual export through the installed extension and all of its guards.\n' + loader + '''
from pathlib import Path
from datetime import datetime, timezone
folder = Path(bpy.data.filepath).parent / ('UNREVIEWED_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
bpy.context.scene.grail_g1.output_dir = str(folder)
bpy.ops.grail_g1.export_native()
print(folder)
''')
    runtime.text('README_ContactEdit.txt', 'GRAIL G1 Retargeting Refinement\nSource frame 0 = Blender frame 1. Reopen this file and use the GRAIL G1 sidebar to select native joint curves or save/export a new UNREVIEWED checkpoint. The installed extension initializes its trusted bundled runtime without executing this scene\'s embedded Python. Existing compatible portable G1 scenes require explicit Enable Exact Preview consent. Otherwise this scene contains editable controls only. Preserve ContactEdit_FrozenReference and all window endpoints. Bézier / Auto Clamped with smoothing NONE is the default; Continuous Acceleration is optional. Original joint limits and frozen coordinates are checked during export. Contact, original-mesh collisions, source/depth alignment and physical validity are not established. Initial global retargeting and optional MuJoCo/SciPy contact pose fitting remain external.\n')


def validate_protected_keys():
    """Additional authored milestones can be protected explicitly in plan JSON."""
    state = runtime.STATE
    action = bpy.data.objects[runtime.CONTROLLER].animation_data.action
    plans = {channel['joint']: channel for channel in state['plan']['channels']}
    for channel in state['channels']:
        declared = plans[channel['joint']].get('protected_source_frames', [])
        keys = {float(t): float(value) for t, value in channel['keys']}
        fc = action.fcurves.find(f'["qpos_{channel["address"]:02d}"]', index=0)
        for source_frame in declared:
            if isinstance(source_frame, bool) or not isinstance(source_frame, (int, float)) or source_frame not in keys:
                raise ValueError('protected_source_frames must identify keys in the initial plan.')
            matches = [key for key in fc.keyframe_points if float(key.co[0]) == float(np.float32(source_frame + 1))]
            if len(matches) != 1 or matches[0].co[1] != np.float32(keys[source_frame]):
                raise ValueError('Protected milestone changed for ' + channel['joint'])


def validate_plan_protection(plan):
    for channel in plan['channels']:
        keys = {float(t) for t, value in channel['keys']}
        protected = channel.get('protected_source_frames', [])
        if not isinstance(protected, list) or any(isinstance(t, bool) or not isinstance(t, (int, float)) or t not in keys for t in protected):
            raise ValueError('protected_source_frames must be a list of existing source key times.')


def export_checkpoint(out, substeps):
    initialize()
    runtime.validate_curves()
    validate_protected_keys()
    # Export's native invariants and dense sampled limits are the upstream kit's.
    report = runtime.export(out, substeps)
    candidate = native_io.load_npz(Path(out) / 'candidate_UNREVIEWED.npz')
    report['native_archive_reopened'] = True
    report['native_archive_status_verified'] = str(candidate['status']) == 'UNREVIEWED' and not bool(candidate['selected'])
    report['extension'] = 'GRAIL G1 Retargeting Refinement'
    report['executed_code_sha256'] = {
        name: runtime.digest(path) for name, path in (
            ('blender_curves.py', Path(runtime.__file__)),
            ('native_io.py', Path(native_io.__file__)),
            ('runtime_adapter.py', Path(__file__)),
            ('__init__.py', Path(__file__).with_name('__init__.py')),
        )
    }
    report['protected_milestones_checked'] = True
    (Path(out) / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
