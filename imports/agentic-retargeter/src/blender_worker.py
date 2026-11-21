"""Constrained JSON bridge to real Blender native curves (no model Python).

Invoke Blender with --disable-autoexec and --python-exit-code 1, a copied or
read-only input checkpoint, and ``--python this_file -- --request ... --response
...``. The controller owns the spec and output paths; only operations originate
from a model. Native previews use declared body bindings and compiled-model
arrays, never embedded scene scripts. Every exported motion is UNREVIEWED.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(label + ' must be a finite number')
    return float(value)


def quat_rotation(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid native quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quat_slerp(a, b, weight):
    """Normalized shortest-arc SLERP, including nearly identical endpoints.

    Blender's small-angle quaternion interpolation may return a non-unit value;
    converting that directly to a matrix introduces scale/shear into the object.
    Sinc weights retain true SLERP without an unnormalized linear fallback.
    """
    values = np.asarray([a, b], dtype=float)
    if values.shape != (2, 4) or not np.isfinite(values).all():
        raise ValueError('Invalid object interpolation quaternions')
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < 1e-12) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Invalid object quaternion interpolation')
    first, second = values / norms[:, None]
    dot = float(first @ second)
    if dot < 0:
        second, dot = -second, -dot
    angle = np.arccos(np.clip(dot, 0., 1.)) / np.pi
    result = ((1-weight) * np.sinc((1-weight)*angle) * first
              + weight * np.sinc(weight*angle) * second) / np.sinc(angle)
    return result / np.linalg.norm(result)


class NativeFK:
    """NumPy FK for compiled MuJoCo bodies with at most one joint per body.

    Native G1 trees and the portable synthetic fixture meet this contract.
    Unsupported compound bodies fail explicitly instead of drawing stale rigs.
    """
    def __init__(self, model):
        self.model = model
        required = ('body_names', 'body_parentid', 'body_pos', 'body_quat',
                    'body_jntadr', 'body_jntnum', 'jnt_pos', 'jnt_axis', 'qpos0')
        missing = [key for key in required if key not in model]
        if missing:
            raise ValueError('Native visual model lacks compiled FK arrays: ' + ', '.join(missing))
        self.names = model['body_names'].astype(str).tolist()
        count, nj = len(self.names), len(model['jnt_type'])
        if len(set(self.names)) != count or count < 2:
            raise ValueError('Native body names must be unique and include world')
        shapes = {'body_parentid': (count,), 'body_pos': (count, 3),
                  'body_quat': (count, 4), 'body_jntadr': (count,),
                  'body_jntnum': (count,), 'jnt_pos': (nj, 3), 'jnt_axis': (nj, 3)}
        for key, shape in shapes.items():
            if model[key].shape != shape or not np.isfinite(model[key]).all():
                raise ValueError('Invalid compiled FK array ' + key)
        for body in range(1, count):
            if not 0 <= int(model['body_parentid'][body]) < body:
                raise ValueError('Compiled bodies must have topologically ordered parents')
            n = int(model['body_jntnum'][body])
            if n not in (0, 1):
                raise ValueError('Native preview supports at most one joint per body')
            if n and not 0 <= int(model['body_jntadr'][body]) < nj:
                raise ValueError('Invalid compiled body joint address')

    def forward(self, q):
        m, count = self.model, len(self.names)
        positions = np.zeros((count, 3))
        rotations = np.tile(np.eye(3), (count, 1, 1))
        for body in range(1, count):
            parent = int(m['body_parentid'][body])
            pos = m['body_pos'][body].astype(float).copy()
            rot = quat_rotation(m['body_quat'][body])
            if m['body_jntnum'][body]:
                joint = int(m['body_jntadr'][body])
                kind, address = int(m['jnt_type'][joint]), int(m['jnt_qposadr'][joint])
                if kind == 0:
                    if parent != 0:
                        raise ValueError('Free joint must be attached to world')
                    positions[body] = q[address:address+3]
                    rotations[body] = quat_rotation(q[address+3:address+7])
                    continue
                pivot = m['jnt_pos'][joint]
                if kind == 2:
                    pos += rot @ m['jnt_axis'][joint] * (q[address] - m['qpos0'][address])
                else:
                    if kind == 1:
                        change = quat_rotation(q[address:address+4])
                    elif kind == 3:
                        angle = q[address] - m['qpos0'][address]
                        axis = m['jnt_axis'][joint]
                        if not np.isclose(np.linalg.norm(axis), 1., atol=1e-6):
                            raise ValueError('Native hinge axis is not unit length')
                        change = quat_rotation(np.r_[np.cos(angle/2), axis*np.sin(angle/2)])
                    else:
                        raise ValueError('Unsupported native joint type')
                    pos += rot @ (pivot - change @ pivot)
                    rot = rot @ change
            positions[body] = positions[parent] + rotations[parent] @ pos
            rotations[body] = rotations[parent] @ rot
        return positions, rotations


def load_runtime():
    """Load repository-owned kit; disable its legacy embedded-Python preview."""
    folder = Path(__file__).resolve().parent / 'contact_edit'
    if not (folder / 'blender_curves.py').is_file():
        raise ValueError('Blender worker requires its packaged contact edit runtime')
    import runpy
    runpy.run_path(str(Path(__file__).with_name('_bootstrap.py')))['bootstrap']()
    from agentic_retargeter.contact_edit import blender_curves as runtime
    runtime.parent_runtime = lambda native, model: None
    runtime.initialize()
    runtime.validate_curves()
    return runtime


def validate_spec(spec, runtime):
    if not isinstance(spec, dict):
        raise ValueError('Trusted spec must be an object')
    channels = {c['joint']: c for c in runtime.STATE['channels']}
    scope = spec.get('editable_joints')
    if not isinstance(scope, dict) or not scope:
        raise ValueError('Trusted spec needs explicit editable_joints')
    for joint, bounds in scope.items():
        if joint not in channels or not isinstance(bounds, dict):
            raise ValueError('Unknown or invalid editable native joint: ' + joint)
        channel = channels[joint]
        window, limits = bounds.get('window'), bounds.get('limits_rad')
        if not isinstance(window, list) or len(window) != 2 or not isinstance(limits, list) or len(limits) != 2:
            raise ValueError('Each joint needs window and limits_rad pairs')
        low, high = [number(v, 'window') for v in window]
        minimum, maximum = [number(v, 'limits_rad') for v in limits]
        if not channel['window'][0] <= low < high <= channel['window'][1]:
            raise ValueError('Trusted window exceeds native plan for ' + joint)
        if not channel['limits'][0] <= minimum < maximum <= channel['limits'][1]:
            raise ValueError('Trusted limits exceed original native limits for ' + joint)
        if 'max_delta_rad' in bounds and number(bounds['max_delta_rad'], 'max_delta_rad') <= 0:
            raise ValueError('max_delta_rad must be positive')
    frames = spec.get('observation_frames')
    total = len(runtime.STATE['native']['qpos'])
    if not isinstance(frames, list) or not 1 <= len(frames) <= 32:
        raise ValueError('Use 1 to 32 observation_frames')
    for frame in frames:
        if not 0 <= number(frame, 'observation frame') <= total - 1:
            raise ValueError('Observation frame outside native timeline')
    cameras = spec.get('cameras')
    if not isinstance(cameras, list) or not 2 <= len(cameras) <= 8 or len(set(cameras)) != len(cameras):
        raise ValueError('At least two distinct configured cameras are required (maximum eight)')
    return channels


def protected_frames(runtime, joint):
    channel = next(c for c in runtime.STATE['plan']['channels'] if c['joint'] == joint)
    return set(channel['window']) | set(channel.get('protected_source_frames', []))


def apply_operations(operations, spec, runtime):
    import bpy
    if not isinstance(operations, list) or len(operations) > 256:
        raise ValueError('operations must be a list of at most 256 operations')
    channels = {c['joint']: c for c in runtime.STATE['channels']}
    action = bpy.data.objects[runtime.CONTROLLER].animation_data.action
    seen = set()
    for operation in operations:
        if not isinstance(operation, dict) or operation.get('op') not in ('set_joint_key', 'delete_joint_key'):
            raise ValueError('Only set_joint_key and delete_joint_key operations are allowed')
        expected = {'op', 'joint', 'source_frame'}
        if operation['op'] == 'set_joint_key':
            expected.add('value_rad')
        if set(operation) != expected:
            raise ValueError('Unexpected or missing operation fields')
        joint = operation['joint']
        if not isinstance(joint, str) or joint not in spec['editable_joints']:
            raise ValueError('Joint is outside the trusted editable scope')
        frame = number(operation['source_frame'], 'source_frame')
        bounds = spec['editable_joints'][joint]
        if not bounds['window'][0] < frame < bounds['window'][1] or frame in protected_frames(runtime, joint):
            raise ValueError('Edit hits a protected frame or lies outside the trusted window')
        if (joint, frame) in seen:
            raise ValueError('Duplicate operation on the same joint/frame')
        seen.add((joint, frame))
        channel = channels[joint]
        curve = action.fcurves.find(f'["qpos_{channel["address"]:02d}"]', index=0)
        matches = [key for key in curve.keyframe_points if float(key.co[0]) == float(np.float32(frame+1))]
        if operation['op'] == 'delete_joint_key':
            if len(matches) != 1:
                raise ValueError('delete_joint_key requires one existing editable key')
            curve.keyframe_points.remove(matches[0])
        else:
            value = number(operation['value_rad'], 'value_rad')
            if not bounds['limits_rad'][0] <= value <= bounds['limits_rad'][1]:
                raise ValueError('Requested key exceeds trusted/native joint limits')
            if len(matches) > 1:
                raise ValueError('Duplicate native key times')
            key = matches[0] if matches else curve.keyframe_points.insert(frame+1, value)
            key.co[1] = value
            key.interpolation = channel['interpolation']
            key.handle_left_type = key.handle_right_type = 'AUTO_CLAMPED'
        curve.update()
    runtime.validate_curves()
    # The kit protects window endpoints. Also honor extension-authored milestones.
    for channel in runtime.STATE['channels']:
        curve = action.fcurves.find(f'["qpos_{channel["address"]:02d}"]', index=0)
        initial = dict(channel['keys'])
        for frame in protected_frames(runtime, channel['joint']):
            matches = [key for key in curve.keyframe_points if key.co[0] == np.float32(frame+1)]
            if len(matches) != 1 or matches[0].co[1] != np.float32(initial[frame]):
                raise ValueError('Protected native key changed')


class Preview:
    def __init__(self, spec, runtime):
        import bpy
        self.runtime = runtime
        self.spec = spec
        config = spec.get('preview', {})
        if config.get('mode') != 'native_fk':
            raise ValueError('Rendered observations require trusted preview.mode native_fk')
        model_text = config.get('model_text', 'ContactEdit_Model.npz.b64')
        if model_text not in bpy.data.texts:
            raise ValueError('Missing configured native visual model text')
        model = runtime.unpack(model_text)
        for key, value in runtime.STATE['model'].items():
            if key not in model or not np.array_equal(model[key], value):
                raise ValueError('Visual model differs from native edit model: ' + key)
        self.fk = NativeFK(model)
        self.bindings = config.get('body_objects', {})
        if set(self.bindings) != set(self.fk.names[1:]):
            raise ValueError('body_objects must bind every compiled non-world native body')
        self.object_name = config.get('object')
        names = list(self.bindings.values()) + [self.object_name]
        if any(not isinstance(name, str) or name not in bpy.context.scene.objects for name in names):
            raise ValueError('Native body/object binding is missing from the current scene')
        if len(set(names)) != len(names) or any(name in spec['cameras'] for name in names):
            raise ValueError('Native body/object/camera bindings must be distinct')
        for name in names:
            obj = bpy.data.objects[name]
            if obj.constraints or (obj.animation_data and (obj.animation_data.drivers or obj.animation_data.nla_tracks)):
                raise ValueError('Native body/object binding has unsupported constraints, drivers or NLA')
            if obj.animation_data and obj.animation_data.action:
                # Rendering can reevaluate an Action after update() assigns FK.
                # A second transform owner then shows the old pose despite a
                # correct native export. Require explicit scene rebasing.
                transforms = {'location', 'rotation_euler', 'rotation_quaternion',
                              'rotation_axis_angle', 'scale', 'delta_location',
                              'delta_rotation_euler', 'delta_rotation_quaternion', 'delta_scale'}
                if any(fc.data_path in transforms for fc in obj.animation_data.action.fcurves):
                    owner = 'object' if name == self.object_name else 'body'
                    raise ValueError(
                        'Native ' + owner + ' binding has transform animation that overrides FK/native object interpolation: '
                        + name + '; explicitly rebase native transform ownership before preview')
            if name != self.object_name and not np.allclose(obj.matrix_world.to_scale(), 1., atol=1e-6, rtol=0):
                raise ValueError('Native body bindings must have unit world scale; preserve geometry in child meshes')
        for name in spec['cameras']:
            if name not in bpy.context.scene.objects or bpy.data.objects[name].type != 'CAMERA':
                raise ValueError('Configured camera is missing or is not a camera: ' + str(name))
        native = runtime.STATE['native']
        for key, expected in [('scene_scale', 1.), ('height_scale', 1.), ('height_offset', 0.), ('pelvis_rotation', 0.)]:
            if key in native and not np.allclose(native[key], expected):
                raise ValueError('Native preview needs direct world coordinates; unsupported ' + key)
        for key, shape in [('obj_R', (len(native['qpos']), 3, 3)), ('obj_t', (len(native['qpos']), 3))]:
            if key not in native or native[key].shape != shape or not np.isfinite(native[key]).all():
                raise ValueError('Missing/invalid frozen native object field ' + key)
        scale = np.asarray(native.get('object_scale', []))
        if scale.size not in (1, 3) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Native object_scale must be positive scalar or XYZ')
        from agentic_retargeter.static_scene import ensure_blender_static_scene
        self.static_scene = ensure_blender_static_scene(native)

    def update(self, frame):
        import bpy
        from mathutils import Matrix
        scene = bpy.context.scene
        scene.frame_set(int(math.floor(frame)) + 1, subframe=frame % 1)
        native = self.runtime.STATE['native']
        positions, rotations = self.fk.forward(self.runtime.q_at(frame))
        for index, name in enumerate(self.fk.names[1:], start=1):
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotations[index], positions[index]
            bpy.data.objects[self.bindings[name]].matrix_world = Matrix(transform)
        low, high = int(math.floor(frame)), min(int(math.ceil(frame)), len(native['qpos'])-1)
        weight = frame-low
        a, b = Matrix(native['obj_R'][low]).to_quaternion(), Matrix(native['obj_R'][high]).to_quaternion()
        rotation = quat_rotation(quat_slerp(a, b, weight))
        transform = np.eye(4)
        transform[:3, :3] = rotation @ np.diag(np.broadcast_to(np.asarray(native['object_scale']).reshape(-1), (3,)))
        transform[:3, 3] = (1-weight)*native['obj_t'][low] + weight*native['obj_t'][high]
        bpy.data.objects[self.object_name].matrix_world = Matrix(transform)
        bpy.context.view_layer.update()
        from agentic_retargeter.static_scene import ensure_blender_static_scene
        self.static_scene = ensure_blender_static_scene(native)


def observe(runtime, spec, preview):
    import bpy
    action = bpy.data.objects[runtime.CONTROLLER].animation_data.action
    controls = []
    for channel in runtime.STATE['channels']:
        curve = action.fcurves.find(f'["qpos_{channel["address"]:02d}"]', index=0)
        controls.append(dict(joint=channel['joint'], native_property=f'qpos_{channel["address"]:02d}',
                             address=channel['address'], window=channel['window'], limits_rad=channel['limits'],
                             editable=channel['joint'] in spec['editable_joints'],
                             protected_source_frames=sorted(protected_frames(runtime, channel['joint'])),
                             keys=[{'source_frame': float(k.co[0]-1), 'value_rad': float(k.co[1])} for k in curve.keyframe_points]))
    states = []
    for frame in spec['observation_frames']:
        preview.update(frame)
        states.append(dict(source_frame=frame, qpos=runtime.q_at(frame).tolist(),
                           object_matrix_world=[list(row) for row in bpy.data.objects[preview.object_name].matrix_world],
                           body_matrix_world={name: [list(row) for row in bpy.data.objects[obj].matrix_world]
                                              for name, obj in preview.bindings.items()}))
    return dict(controls=controls, states=states, fps=float(runtime.STATE['native']['fps']),
                frame_count=len(runtime.STATE['native']['qpos']), static_scene=preview.static_scene,
                status='UNREVIEWED')


def render_observations(stage, runtime, spec, preview, out):
    import bpy
    scene = bpy.context.scene
    config = spec.get('render', {})
    resolution = config.get('resolution', [512, 512])
    if not isinstance(resolution, list) or len(resolution) != 2 or any(type(v) is not int or not 64 <= v <= 2048 for v in resolution):
        raise ValueError('Render resolution must contain two integers in [64, 2048]')
    engine = config.get('engine', 'CYCLES')
    if engine not in ('CYCLES', 'BLENDER_WORKBENCH'):
        raise ValueError('Render engine must be CYCLES or BLENDER_WORKBENCH')
    samples = config.get('samples', 16)
    if type(samples) is not int or not 1 <= samples <= 128:
        raise ValueError('Render samples must be an integer in [1, 128]')
    scene.render.engine = engine
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    if engine == 'CYCLES':
        scene.cycles.device = 'CPU'
        scene.cycles.samples = samples
    records = []
    for frame_index, frame in enumerate(spec['observation_frames']):
        preview.update(frame)
        for camera_index, camera in enumerate(spec['cameras']):
            path = out / f'{stage}_frame{frame_index:03d}_view{camera_index:02d}.png'
            scene.camera = bpy.data.objects[camera]
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            if not path.is_file():
                raise RuntimeError('Blender did not write configured render')
            records.append(dict(stage=stage, source_frame=frame, camera=camera,
                                path=str(path), sha256=sha256(path)))
    return records


def render_hand_inspections(stage, spec, preview, out):
    """Read-only canonical/contact evidence from the same native FK pose."""
    if 'inspection' not in spec:
        return [], []
    import bpy
    # This file also runs as a Blender --python script outside the package.
    import runpy
    runpy.run_path(str(Path(__file__).with_name('_bootstrap.py')))['bootstrap']()
    from agentic_retargeter.blender_inspection import (
        build_inspection, build_dashboard, clear_inspection, render_inspection)
    config = spec['inspection']
    if not isinstance(config, dict):
        raise ValueError('inspection must be an explicit scene binding object')
    def follows(name, ancestors):
        obj = bpy.context.scene.objects.get(name) if isinstance(name, str) else None
        while obj is not None:
            if obj.name in ancestors:
                return True
            obj = obj.parent
        return False
    if not follows(config.get('object'), {preview.object_name}):
        raise ValueError('Inspection object must follow the frozen native object binding')
    body_objects = set(preview.bindings.values())
    for hand in config.get('hands', []):
        if hand.get('wrist') not in body_objects:
            raise ValueError('Inspection wrist must be a declared native FK body')
        if any(not follows(name, body_objects) for name in hand.get('meshes', [])):
            raise ValueError('Inspection hand meshes must follow declared native FK bodies')
    layers = spec.get('inspection_layers', ['measured'])
    if (not isinstance(layers, list) or not layers or len(set(layers)) != len(layers)
            or any(layer not in ('measured', 'predicted', 'difference') for layer in layers)):
        raise ValueError('inspection_layers must select measured, predicted or difference once each')
    layout = spec.get('inspection_layout', 'views')
    if layout not in ('views', 'dashboard'):
        raise ValueError('inspection_layout must be views or dashboard')
    if layout == 'dashboard':
        if 'inspection_layers' in spec:
            raise ValueError('The dashboard compares all three layers; omit inspection_layers')
        layers = ['dashboard']
    scene = bpy.context.scene
    records, reports = [], []
    for frame_index, frame in enumerate(spec['observation_frames']):
        preview.update(frame)
        for layer in layers:
            try:
                report = (build_dashboard(scene, config, frame) if layout == 'dashboard'
                          else build_inspection(scene, config, frame, layer=layer))
                path = out / f'{stage}_inspection_frame{frame_index:03d}_{layer}.json'
                with path.open('x') as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
                    stream.write('\n')
                reports.append(dict(stage=stage, source_frame=frame, layer=layer,
                    path=str(path), sha256=sha256(path), summary=report.get('summary', {}),
                    provenance=report.get('provenance'),
                    interpretation='Unsigned surface proximity; prediction coverage may be absent.'))
                cameras = [report['dashboard']['camera']] if layout == 'dashboard' else report['cameras']
                for camera_index, camera in enumerate(cameras):
                    image_path = out / f'{stage}_inspection_frame{frame_index:03d}_{layer}_view{camera_index:02d}.png'
                    resolution = (report['dashboard']['recommended_resolution'] if layout == 'dashboard'
                                  else min(spec.get('render', {}).get('resolution', [512, 512])))
                    render_inspection(scene, report, camera, str(image_path), resolution=resolution)
                    if not image_path.is_file():
                        raise RuntimeError('Blender did not write hand inspection render')
                    records.append(dict(stage=stage, source_frame=frame, camera=camera,
                        kind='hand_inspection', layer=layer,
                        path=str(image_path), sha256=sha256(image_path)))
            finally:
                clear_inspection(scene)
    return records, reports


def run(request):
    import bpy
    if not (4, 2, 0) <= bpy.app.version < (4, 4, 0):
        raise ValueError('Use Blender 4.2/4.3; tested with 4.3.2')
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise ValueError('Launch with --disable-autoexec and an isolated Blender profile')
    if request.get('schema_version') != 1:
        raise ValueError('Expected worker schema_version 1')
    original = Path(bpy.data.filepath).resolve()
    if not original.is_file() or sha256(original) != request.get('input_blend_sha256'):
        raise ValueError('Input Blender checkpoint hash mismatch')
    out = Path(request['output_dir']).resolve()
    if out.exists():
        raise ValueError('Worker output_dir must be new')
    # No handlers from a saved scene execute. Autoexec is disabled before load.
    for handlers in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post,
                     bpy.app.handlers.render_pre, bpy.app.handlers.render_post):
        if handlers:
            raise ValueError('Use a clean Blender process without scene evaluation handlers')
    runtime = load_runtime()
    spec = request['spec']
    validate_spec(spec, runtime)
    # An add-on checkpoint may contain static inspection copies. Refreshing
    # removes those datablocks; discard them before holding any RNA references.
    scene = bpy.context.scene
    discarded_snapshot = bool(scene.get('grail_inspection_owner'))
    if discarded_snapshot:
        import runpy
        runpy.run_path(str(Path(__file__).with_name('_bootstrap.py')))['bootstrap']()
        from agentic_retargeter.blender_inspection import clear_inspection
        diagnostic_camera = scene.camera and scene.camera.get('grail_inspection_owner')
        clear_inspection(scene)
        if diagnostic_camera:
            scene.camera = scene.objects[spec['cameras'][0]]
    preview = Preview(spec, runtime)
    substeps = spec.get('substeps', 4)
    if type(substeps) is not int or not 2 <= substeps <= 64:
        raise ValueError('substeps must be an integer in [2, 64]')
    times = np.arange((len(runtime.STATE['native']['qpos'])-1)*substeps+1) / substeps
    baseline = np.array([runtime.q_at(frame) for frame in times])
    scene = bpy.context.scene
    # Diagnostic rendering must not alter saved camera, render settings or pose.
    saved_camera = scene.camera
    saved_frame = (scene.frame_current, scene.frame_subframe)
    saved_render = {name: getattr(scene.render, name) for name in (
        'engine', 'resolution_x', 'resolution_y', 'resolution_percentage', 'filepath', 'use_compositing', 'use_sequencer')}
    saved_format = scene.render.image_settings.file_format
    saved_cycles = (scene.cycles.device, scene.cycles.samples)
    saved_transforms = [(obj, obj.matrix_basis.copy()) for obj in scene.objects]
    before = observe(runtime, spec, preview)
    # An in-memory copy keeps exact original handles without reloading the open
    # file as a library (Blender 4.3 can crash when self-loading an Action).
    original_action = bpy.data.objects[runtime.CONTROLLER].animation_data.action.copy()
    apply_operations(request.get('operations', []), spec, runtime)
    after_samples = np.array([runtime.q_at(frame) for frame in times])
    scope = spec['editable_joints']
    editable_addresses = {channel['address'] for channel in runtime.STATE['channels'] if channel['joint'] in scope}
    frozen = [index for index in range(baseline.shape[1]) if index not in editable_addresses]
    if not np.array_equal(baseline[:, frozen], after_samples[:, frozen]):
        raise ValueError('Frozen native coordinates changed')
    for channel in runtime.STATE['channels']:
        if channel['joint'] not in scope:
            continue
        bounds, address = scope[channel['joint']], channel['address']
        outside = (times <= bounds['window'][0]) | (times >= bounds['window'][1])
        if not np.array_equal(baseline[outside, address], after_samples[outside, address]):
            raise ValueError('Curve interpolation changes samples outside trusted window')
        if np.any(after_samples[:, address] < bounds['limits_rad'][0]-1e-6) or np.any(after_samples[:, address] > bounds['limits_rad'][1]+1e-6):
            raise ValueError('Curve interpolation exceeds trusted joint limits')
        if 'max_delta_rad' in bounds:
            native = runtime.STATE['native']['qpos']
            source = np.interp(times, np.arange(len(native)), native[:, address])
            if np.max(np.abs(after_samples[:, address]-source)) > bounds['max_delta_rad']+1e-6:
                raise ValueError('Curve interpolation exceeds source-relative max_delta_rad')
    after = observe(runtime, spec, preview)
    # Only allocate evidence after all operations and numerical guards pass.
    out.mkdir(parents=True, exist_ok=False)
    renders, inspection_reports = [], []
    action = bpy.data.objects[runtime.CONTROLLER].animation_data.action
    try:
        bpy.data.objects[runtime.CONTROLLER].animation_data.action = original_action
        renders.extend(render_observations('before', runtime, spec, preview, out))
        images, evidence = render_hand_inspections('before', spec, preview, out)
        renders.extend(images)
        inspection_reports.extend(evidence)
        bpy.data.objects[runtime.CONTROLLER].animation_data.action = action
        renders.extend(render_observations('after', runtime, spec, preview, out))
        images, evidence = render_hand_inspections('after', spec, preview, out)
        renders.extend(images)
        inspection_reports.extend(evidence)
    finally:
        bpy.data.objects[runtime.CONTROLLER].animation_data.action = action
        bpy.data.actions.remove(original_action)
        scene.camera = saved_camera
        for name, value in saved_render.items():
            setattr(scene.render, name, value)
        scene.render.image_settings.file_format = saved_format
        scene.cycles.device, scene.cycles.samples = saved_cycles
        scene.frame_set(saved_frame[0], subframe=saved_frame[1])
        for obj, matrix in saved_transforms:
            # Reassigning an unchanged matrix decomposes its quaternion/scale
            # again in Blender. Repeated edit rounds can then drift immutable
            # original mesh offsets, even though no tool edited those meshes.
            if not np.array_equal(np.array(obj.matrix_basis), np.array(matrix)):
                obj.matrix_basis = matrix
        # Save native edited pose at the original frame, preserving source camera.
        preview.update(float(np.clip(saved_frame[0]-1+saved_frame[1], 0, len(runtime.STATE['native']['qpos'])-1)))
    report = runtime.export(out / 'native', substeps)
    checkpoint = out / 'native/editable_curves.blend'
    candidate = out / 'native/candidate_UNREVIEWED.npz'
    with np.load(candidate, allow_pickle=False) as archive:
        qpos = archive['qpos'].copy()
        if not np.array_equal(qpos, after_samples[::substeps]):
            raise ValueError('Exported native samples differ from actual Blender curves')
        if str(archive['status']) != 'UNREVIEWED' or bool(archive['selected']):
            raise ValueError('Native candidate status must remain UNREVIEWED')
        for key in ('obj_R', 'obj_t', 'object_scale', 'fps', 'timestamps'):
            if key in runtime.STATE['native'] and not np.array_equal(archive[key], runtime.STATE['native'][key]):
                raise ValueError('Frozen native field changed: ' + key)
        from agentic_retargeter.static_scene import assert_static_scene_equal
        assert_static_scene_equal(runtime.STATE['native'], archive)
    if sha256(original) != request['input_blend_sha256']:
        raise ValueError('Original Blender checkpoint was modified')
    # Reopen the actual saved file with autoexec disabled, then compare evaluation.
    runtime.STATE.clear()
    bpy.ops.wm.open_mainfile(filepath=str(checkpoint), use_scripts=False)
    runtime.initialize()
    runtime.validate_curves()
    reopened = np.array([runtime.q_at(frame) for frame in range(len(qpos))])
    if not np.array_equal(reopened, qpos):
        raise ValueError('Saved Blender checkpoint failed native round trip')
    from agentic_retargeter.static_scene import ensure_blender_static_scene
    ensure_blender_static_scene(runtime.STATE['native'])
    diagnostics = dict(original_preserved=True, sampled_frozen_qpos_exact=True, frozen_object_exact=True,
        source_timing_exact=True, native_archive_reopened=True, blend_reopened=True,
        sampled_joint_limits_checked=True, substeps=substeps, sample_count=len(times),
        embedded_scene_python_executed=False, native_fk_visual_preview=True,
        max_native_change_rad=float(np.max(np.abs(after_samples-baseline))),
        checks_not_performed=['contact', 'mesh collisions', 'source/depth alignment', 'physical validity'])
    diagnostics['unsigned_proximity_maps_measured'] = bool(inspection_reports)
    diagnostics['previous_static_inspection_removed_from_candidate'] = discarded_snapshot
    return dict(schema_version=1, status='ok', review_status='UNREVIEWED',
        blender_version=bpy.app.version_string, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        native_motion=str(candidate), native_motion_sha256=sha256(candidate), before=before, after=after,
        renders=renders, inspection_reports=inspection_reports, diagnostics=diagnostics, export_report=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    # Refuse an existing response: every execution gets its own evidence record.
    if args.response.exists():
        raise FileExistsError(args.response)
    try:
        request = json.loads(args.request.read_text())
        response = run(request)
    except Exception as exc:
        response = dict(schema_version=1, status='rejected' if isinstance(exc, (ValueError, KeyError, TypeError)) else 'error',
                        error={'type': type(exc).__name__, 'message': str(exc)}, review_status='UNREVIEWED')
    args.response.parent.mkdir(parents=True, exist_ok=True)
    with args.response.open('x') as stream:
        json.dump(response, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('AGENTIC_BLENDER_WORKER', json.dumps({'status': response['status'], 'response': str(args.response)}))
    if response['status'] != 'ok':
        raise RuntimeError(response['error']['message'])


if __name__ == '__main__':
    main()
