"""Author bounded native hinge curves with Blender's real F-curve evaluator.

Run with Blender --background [existing.blend] --python blender_curves.py -- ...
The saved scene embeds this runtime and native_io.py for reopening and exporting.
Adapted from an earlier tested native-controller/float-reference runtime pattern.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import sys
import types

import bpy
import numpy as np
from mathutils import Matrix
from bpy.app.handlers import persistent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_io import load_npz, resolve_channels, merge_curve_samples, check_sample_limits

CONTROLLER = 'ContactEdit_Curves'
REFERENCE = 'ContactEdit_FrozenReference'
STATE = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(name, value):
    block = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    block.clear()
    block.write(value)


def pack(name, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = base64.b64encode(stream.getvalue()).decode('ascii')
    # Blender's text buffer is slow on multi-megabyte single lines. Wrapping
    # preserves base64 bytes and keeps real robot model archives practical.
    text(name, '\n'.join(raw[i:i+100] for i in range(0, len(raw), 100)))


def unpack(name):
    data = base64.b64decode(bpy.data.texts[name].as_string())
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def curve(action, path, keys, interpolation, auto_smoothing='NONE'):
    if not hasattr(action, 'fcurves'):
        raise RuntimeError('This kit uses Blender 4.0–4.3 Action.fcurves; adapt/test the layered-action API before using a newer version.')
    fc = action.fcurves.new(path, index=0)
    fc.extrapolation = 'CONSTANT'
    fc.auto_smoothing = auto_smoothing
    for frame, value in keys:
        key = fc.keyframe_points.insert(float(frame) + 1, float(value), options={'FAST'})
        key.interpolation = interpolation
        key.handle_left_type = key.handle_right_type = 'AUTO_CLAMPED'
    fc.update()
    return fc


def setup(native, model, plan, provenance):
    channels = resolve_channels(native, model, plan)
    validate_source_quaternions(native, model)
    parent_runtime(native, model)  # Reject unreconciled parent edits before authoring.
    if bpy.data.objects.get(CONTROLLER) or bpy.data.actions.get(REFERENCE):
        raise ValueError('ContactEdit controls already exist; export them or start from a new parent scene.')
    ob = bpy.data.objects.new(CONTROLLER, None)
    bpy.context.scene.collection.objects.link(ob)
    ob.animation_data_create()
    ob.animation_data.action = bpy.data.actions.new('ContactEdit_AuthoredAction')
    reference = bpy.data.actions.new(REFERENCE)
    reference.use_fake_user = True
    for channel in channels:
        address = channel['address']
        prop = f'qpos_{address:02d}'
        path = f'["{prop}"]'
        ob[prop] = float(native['qpos'][0, address])
        ob.id_properties_ui(prop).update(description=channel['joint'], min=channel['limits'][0], max=channel['limits'][1])
        curve(ob.animation_data.action, path, channel['keys'], channel['interpolation'], channel['auto_smoothing'])
        curve(reference, path, enumerate(native['qpos'][:, address]), 'LINEAR')
    pack('ContactEdit_Source.npz.b64', native)
    pack('ContactEdit_Model.npz.b64', model)
    text('ContactEdit_Plan.json', json.dumps(plan, indent=2))
    text('ContactEdit_Provenance.json', json.dumps(provenance, indent=2))
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, len(native['qpos'])
    scene.render.fps = max(1, round(float(native['fps'])))
    scene.render.fps_base = scene.render.fps / float(native['fps'])
    scene['contact_edit_status'] = 'UNREVIEWED; native controls, not collision or contact acceptance'
    scene['contact_edit_controller'] = CONTROLLER
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    initialize()


def validate_source_quaternions(native, model):
    for kind, address in zip(model['jnt_type'], model['jnt_qposadr']):
        if int(kind) not in (0, 1):
            continue
        start = int(address) + (3 if int(kind) == 0 else 0)
        q = native['qpos'][:, start:start + 4]
        if q.shape[1] != 4 or not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError('Source native quaternion must have unit endpoints')
        if np.any(np.sum(q[:-1] * q[1:], axis=1) < 0):
            raise ValueError('Source quaternion hemisphere discontinuity; review/rebase source before editing hinges')


def parent_runtime(native, model):
    if 'Enable_Exact_Native_Preview.py' in bpy.data.texts:
        exec(bpy.data.texts['Enable_Exact_Native_Preview.py'].as_string(), {'bpy': bpy})
        rt = sys.modules.get('grail_portable_native')
        if rt is None:
            raise ValueError('Unsupported existing native-preview runtime')
        rt.validate_scene_contract()
        for key in ('qpos', 'fps', 'obj_R', 'obj_t', 'object_scale'):
            if key not in native or not np.array_equal(native[key], rt.SOURCE[key]):
                raise ValueError('Existing scene and selected native checkpoint differ: ' + key)
        for key in model:
            if key in rt.MODEL and not np.array_equal(model[key], rt.MODEL[key]):
                raise ValueError('Existing scene and selected model differ: ' + key)
        validate_parent_controller()
        return rt
    return None


def validate_parent_controller():
    parent = bpy.data.objects['Native_Qpos_Controller']
    reference = bpy.data.actions['Native_Qpos_Reference']
    animation = parent.animation_data
    if animation is None or animation.action is None or animation.drivers or animation.nla_tracks:
        raise ValueError('Export/rebase existing parent controller edits, drivers or NLA first')
    current = animation.action.fcurves
    if len(current) != len(reference.fcurves):
        raise ValueError('Export/rebase existing parent controller edits first')
    for ref in reference.fcurves:
        fc = current.find(ref.data_path, index=ref.array_index)
        if fc is None or fc.modifiers or fc.mute or any(k.interpolation != 'LINEAR' for k in fc.keyframe_points):
            raise ValueError('Export/rebase existing parent controller edits first')
        if not np.array_equal(np.array([k.co[:] for k in fc.keyframe_points]), np.array([k.co[:] for k in ref.keyframe_points])):
            raise ValueError('Export/rebase existing parent controller edits first')


def initialize():
    native = unpack('ContactEdit_Source.npz.b64')
    model = unpack('ContactEdit_Model.npz.b64')
    plan = json.loads(bpy.data.texts['ContactEdit_Plan.json'].as_string())
    validate_source_quaternions(native, model)
    STATE.update(native=native, model=model, plan=plan, channels=resolve_channels(native, model, plan))
    # Optional exact visual preview for previously reviewed portable scenes.
    # A new empty scene is explicitly a curve-authoring scene, not a robot rig.
    STATE['preview'] = parent_runtime(native, model)
    if STATE['preview'] is not None:
        for handler in list(bpy.app.handlers.frame_change_post):
            if getattr(handler, '__name__', '') in ('exact_native_preview', 'contact_edit_preview'):
                bpy.app.handlers.frame_change_post.remove(handler)
        bpy.app.handlers.frame_change_post.append(contact_edit_preview)
        contact_edit_preview(bpy.context.scene)
    return STATE


def q_at(source_frame):
    native = STATE['native']
    qpos = native['qpos']
    t = float(source_frame)
    if not np.isfinite(t) or not 0 <= t <= len(qpos) - 1:
        raise ValueError('Frame outside the original source timeline')
    lo, hi = int(np.floor(t)), min(int(np.floor(t)) + 1, len(qpos) - 1)
    q = qpos[lo].copy() if t == lo else (1 - (t - lo)) * qpos[lo] + (t - lo) * qpos[hi]
    authored = bpy.data.objects[CONTROLLER].animation_data.action
    reference = bpy.data.actions[REFERENCE]
    for c in STATE['channels']:
        if c['window'][0] < t < c['window'][1]:
            path = f'["qpos_{c["address"]:02d}"]'
            # Difference against a frozen float32 reference avoids changing a
            # double-precision native value just by passing through Blender.
            q[c['address']] += authored.fcurves.find(path, index=0).evaluate(t + 1) - reference.fcurves.find(path, index=0).evaluate(t + 1)
    return q


@persistent
def contact_edit_preview(scene, *unused):
    rt = STATE.get('preview')
    if rt is None:
        return
    t = float(np.clip(scene.frame_current_final - 1, 0, len(STATE['native']['qpos']) - 1))
    # Let the original runtime preserve its object/camera convention; override
    # only the native robot body transforms with the authored joint curves.
    rt.exact_native_preview(scene)
    positions, rotations = rt.FK.forward(q_at(t))
    for body in range(1, rt.FK.n):
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotations[body], positions[body]
        bpy.data.objects['G1_' + rt.FK.body_names[body]].matrix_world = Matrix(transform)


def validate_curves():
    native, model = STATE['native'], STATE['model']
    scene = bpy.context.scene
    expected_fps = max(1, round(float(native['fps'])))
    expected_base = float(np.float32(expected_fps / float(native['fps'])))
    if (scene.frame_start, scene.frame_end) != (1, len(native['qpos'])) or scene.render.fps != expected_fps or scene.render.fps_base != expected_base:
        raise ValueError('Source frame count/fps changed')
    ob = bpy.data.objects[CONTROLLER]
    if ob.animation_data.drivers or ob.animation_data.nla_tracks:
        raise ValueError('Bake/review drivers or NLA separately; this exporter evaluates the explicit Action only.')
    action = ob.animation_data.action
    if len(action.fcurves) != len(STATE['channels']):
        raise ValueError('Unexpected native channels in the edit Action')
    reference = bpy.data.actions[REFERENCE]
    if len(reference.fcurves) != len(STATE['channels']):
        raise ValueError('Frozen reference action changed')
    for c in STATE['channels']:
        path = f'["qpos_{c["address"]:02d}"]'
        fc, ref = action.fcurves.find(path, index=0), reference.fcurves.find(path, index=0)
        if fc is None or ref is None or fc.modifiers or ref.modifiers or fc.mute or ref.mute:
            raise ValueError('Missing, muted or modified curve')
        if fc.auto_smoothing not in ('NONE', 'CONT_ACCEL'):
            raise ValueError('Unsupported curve auto_smoothing')
        points = np.array([p.co[:] for p in ref.keyframe_points])
        expected = np.column_stack([np.arange(len(native['qpos'])) + 1, native['qpos'][:, c['address']].astype(np.float32)])
        if not np.array_equal(points, expected) or any(p.interpolation != 'LINEAR' for p in ref.keyframe_points):
            raise ValueError('Frozen native reference changed')
        authored = np.array([p.co[:] for p in fc.keyframe_points])
        if len(authored) < 2 or not np.isfinite(authored).all() or np.any(np.diff(authored[:, 0]) <= 0):
            raise ValueError('Invalid or duplicate authored keys')
        if not np.array_equal(authored[[0, -1], 0] - 1, c['window']):
            raise ValueError('Authored keys must retain the declared window endpoints')
        if np.any(authored[:, 1] < c['limits'][0] - 1e-6) or np.any(authored[:, 1] > c['limits'][1] + 1e-6):
            raise ValueError('Authored key exceeds native limits')
        for index, t in zip((0, -1), c['window']):
            if authored[index, 1] != np.float32(native['qpos'][t, c['address']]):
                raise ValueError('Protected grasp/window endpoint changed')
        if any(p.interpolation not in ('BEZIER', 'LINEAR') for p in fc.keyframe_points):
            raise ValueError('Only declared linear/Bezier curves are supported')
    if STATE['preview'] is not None:
        STATE['preview'].validate_scene_contract()
        validate_parent_controller()


def export(out_dir, substeps=4):
    if not isinstance(substeps, int) or substeps < 2:
        raise ValueError('Use at least two samples per source interval')
    validate_curves()
    native, model, plan = STATE['native'], STATE['model'], STATE['plan']
    times = np.arange((len(native['qpos']) - 1) * substeps + 1, dtype=float) / substeps
    sampled = np.array([q_at(t) for t in times])
    addresses = [c['address'] for c in STATE['channels']]
    check_sample_limits(model, addresses, sampled[:, addresses], tolerance=1e-6)
    candidate, report = merge_curve_samples(native, model, plan, sampled[::substeps])
    integer = candidate['qpos']
    lo = np.floor(times).astype(int)
    hi = np.minimum(lo + 1, len(integer) - 1)
    weight = (times - lo)[:, None]
    baked = (1 - weight) * integer[lo] + weight * integer[hi]
    action = bpy.data.objects[CONTROLLER].animation_data.action
    actual_curves = []
    for c in STATE['channels']:
        fc = action.fcurves.find(f'["qpos_{c["address"]:02d}"]', index=0)
        actual_curves.append({
            'joint': c['joint'], 'address': c['address'], 'window': c['window'],
            'extrapolation': fc.extrapolation, 'auto_smoothing': fc.auto_smoothing,
            'keys': [{'source_frame': float(k.co[0] - 1), 'value_rad': float(k.co[1]),
                      'interpolation': k.interpolation,
                      'handle_left': [float(k.handle_left[0] - 1), float(k.handle_left[1])],
                      'handle_right': [float(k.handle_right[0] - 1), float(k.handle_right[1])],
                      'handle_left_type': k.handle_left_type, 'handle_right_type': k.handle_right_type}
                     for k in fc.keyframe_points],
        })
    report.update({
        'blender_version': bpy.app.version_string,
        'authored_interpolation': 'Blender F-curve evaluation; default BEZIER / AUTO_CLAMPED',
        'native_export_interpolation': 'Source-rate samples; downstream linear qpos replay differs from authored Bezier subframes',
        'authored_vs_linear_subframe_max_rad': float(np.max(np.abs(sampled[:, addresses] - baked[:, addresses]))),
        'sample_count': len(times), 'substeps': substeps,
        'speed_metric_definition': 'Maxima of finite-difference speeds at the recorded subframe sampling interval; not analytic or continuous peak speeds.',
        'frozen_quaternion_replay': 'Source components interpolate linearly; native FK normalizes. Unit endpoints and hemisphere continuity checked; integer source data unchanged.',
        'exact_native_visual_preview': STATE['preview'] is not None,
        'checks_not_performed': ['material contact', 'whole original mesh collisions', 'source/depth alignment', 'physical validation'],
        'provenance': json.loads(bpy.data.texts['ContactEdit_Provenance.json'].as_string()),
        'plan_meaning': 'Initial editing scope; actual current keys/handles are in authored_curves.json and editable_curves.blend',
    })
    for label, values in [('authored_curves', sampled), ('linear_export', baked)]:
        velocity = np.diff(values[:, addresses], axis=0) * float(native['fps']) * substeps
        report[label + '_peak_rad_per_second'] = float(np.max(np.abs(velocity)))
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'candidate_UNREVIEWED.npz', **candidate)
    np.savez_compressed(out / 'subframe_comparison.npz', source_frames=times, qpos_authored=sampled, qpos_linear_export=baked)
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (out / 'authored_curves.json').write_text(json.dumps(actual_curves, indent=2) + '\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'editable_curves.blend'))
    report['output_sha256'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    report['embedded_code_sha256'] = {name: hashlib.sha256(bpy.data.texts[name].as_string().encode()).hexdigest()
                                      for name in ('ContactEdit_Runtime.py', 'ContactEdit_native_io.py')}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def embed():
    here = Path(__file__).resolve().parent
    text('ContactEdit_native_io.py', (here / 'native_io.py').read_text())
    text('ContactEdit_Runtime.py', Path(__file__).read_text())
    loader = """import bpy, sys, types
for name, block in [('native_io', 'ContactEdit_native_io.py'), ('contact_edit_runtime', 'ContactEdit_Runtime.py')]:
    module = types.ModuleType(name)
    module.__file__ = block
    sys.modules[name] = module
    exec(bpy.data.texts[block].as_string(), module.__dict__)
sys.modules['contact_edit_runtime'].initialize()
"""
    text('Enable_ContactEdit_Preview.py', loader)
    text('Export_ContactEdit.py', loader + "\nfrom pathlib import Path\nfrom datetime import datetime, timezone\nfolder = Path(bpy.data.filepath).parent / ('UNREVIEWED_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))\nsys.modules['contact_edit_runtime'].export(folder)\nprint(folder)\n")
    text('README_ContactEdit.txt', 'Source frame 0 = Blender frame 1. Select ContactEdit_Curves and edit its Action in the Graph Editor. Run Enable_ContactEdit_Preview.py after reopening. Existing compatible portable G1 scenes preview actual FK; a new empty scene only edits curves. Keep ContactEdit_FrozenReference unchanged. Run Export_ContactEdit.py for a new UNREVIEWED native candidate. Old scene exporters do not export this Action. Bezier and source-rate linear playback differ: inspect subframe_comparison.npz and review both. Contact, source and original-mesh checks remain required.\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--substeps', type=int, default=4)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if Path(a.out).exists():
        raise FileExistsError('Use a new output directory')
    provenance = {name: {'path': str(Path(value).resolve()), 'sha256': digest(value)} for name, value in [('native', a.native), ('model', a.model), ('plan', a.plan), ('script', __file__), ('native_io', str(Path(__file__).with_name('native_io.py')))]}
    if bpy.data.filepath:
        provenance['parent_blend'] = {'path': bpy.data.filepath, 'sha256': digest(bpy.data.filepath)}
    setup(load_npz(a.native), load_npz(a.model), json.loads(Path(a.plan).read_text()), provenance)
    embed()
    report = export(a.out, a.substeps)
    print(json.dumps({'status': 'UNREVIEWED', 'out': str(Path(a.out).resolve()), 'subframe_samples': report['sample_count']}))


if __name__ == '__main__':
    main()
