"""Persistent canonical hands and compact, pose-bound Newton playback caches.

Frame handlers perform no external work. Original hand geometry is copied once;
the current articulation stays visible when a frame has no matching evidence.
Only matching cache entries may color it or display contact points and normals.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import uuid

import bpy
from bpy.app.handlers import persistent
import numpy as np

from . import blender_collision as collision
from .collision import HO_CONTACT_COMMIT, NEWTON_COMMIT, NORMAL_CONVENTION, POINT_DISTANCE_SEMANTICS, SIGNED_NORMAL_GAP_SEMANTICS

_PROPERTY = 'grail_collision_sequence'
_states = {}
_busy = set()
_suspended = {}
_sync_callback = None
_config_callback = None


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def geometry_hashes(scene, config):
    return {name: collision._geometry(collision._source(scene, name))['sha256']
            for name in config['meshes']}


def pose_signature(scene, config, geometry_hashes=None):
    """Bind evidence to evaluated pose, original geometry, wrist basis and units."""
    hashes = globals()['geometry_hashes'](scene, config) if geometry_hashes is None else geometry_hashes
    names = sorted(set(config['meshes']) | {hand['wrist'] for hand in config.get('hands', [])})
    payload = dict(source_frame=float(scene.frame_current_final - 1),
        scale=float(scene.unit_settings.scale_length), geometry=hashes,
        matrices={name: np.asarray(scene.objects[name].matrix_world).tolist() for name in names},
        native_controls=collision._native_controls(scene))
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def _semantic_config(config):
    return {key: config.get(key) for key in ('schema_version', 'meshes', 'pairs', 'hands', 'contact_map', 'visualization')}


def _save(scene, state):
    scene[_PROPERTY] = _json({key: state[key] for key in ('config', 'token', 'collection',
        'canonical_hands', 'canonical_camera', 'canonical_view', 'cache_path', 'label')})


def _geometry_records(scene, config):
    records = {}
    for name in config['meshes']:
        obj = collision._source(scene, name)
        records[name] = collision._geometry(obj)
        records[name]['data_pointer'] = obj.data.as_pointer()
    return records


def _restore(scene):
    pointer = scene.as_pointer()
    if pointer in _states:
        return _states[pointer]
    raw = scene.get(_PROPERTY)
    if not raw:
        return None
    state = json.loads(raw)
    if state.get('collection') not in scene.collection.children:
        return None
    state.update(records=_geometry_records(scene, state['config']), dirty=set(), cache=None,
                 overrides={}, glyphs=[obj.name for obj in scene.objects if obj.get(collision._OWNER) == state['token']
                     and obj.get('grail_collision_sequence_glyph')], last_signature=None,
                 current=False, message='Waiting for matching evidence')
    _states[pointer] = state
    if state['cache_path']:
        try:
            _load_cache(scene, state, state['cache_path'])
        except (OSError, ValueError, KeyError) as error:
            state['message'] = 'Cache unavailable: ' + str(error)
    return state


def _owned(scene, state):
    return [obj for obj in scene.objects if obj.get(collision._OWNER) == state['token']]


def _visible(scene, obj, visible):
    obj.hide_render = not visible
    for layer in scene.view_layers:
        if obj.name in layer.objects and obj.hide_get(view_layer=layer) == visible:
            obj.hide_set(not visible, view_layer=layer)


def clear(scene):
    """Remove this display only; source meshes and unrelated snapshots survive."""
    state = _states.pop(scene.as_pointer(), None)
    if state is None and scene.get(_PROPERTY):
        state = json.loads(scene[_PROPERTY])
    if not state:
        return
    collection = bpy.data.collections.get(state['collection'])
    token = state['token']
    if collection and collection.get(collision._OWNER) == token:
        if any(other != scene and collection.name in other.collection.children for other in bpy.data.scenes):
            scene.collection.children.unlink(collection)
        else:
            for obj in list(collection.objects):
                if obj.get(collision._OWNER) != token:
                    continue
                if len(obj.users_collection) > 1:
                    collection.objects.unlink(obj)
                    continue
                data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if data and data.users == 0 and data.get(collision._OWNER) == token:
                    group = bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.cameras if isinstance(data, bpy.types.Camera) else bpy.data.curves
                    group.remove(data)
            if not collection.objects and not collection.children:
                bpy.data.collections.remove(collection)
    for material in list(bpy.data.materials):
        if material.get(collision._OWNER) == token and material.users == 0:
            bpy.data.materials.remove(material)
    if _PROPERTY in scene:
        del scene[_PROPERTY]


def setup(scene, config, cache=None):
    """Create an always-visible wrist-local display without querying Newton."""
    config = json.loads(_json(config))
    records = _geometry_records(scene, config)
    hands, wrists, _ = collision._hand_inputs(scene, config, records)
    if not hands:
        raise ValueError('Persistent canonical view requires explicit hand and wrist bindings.')
    settings = collision._visual_settings(config)
    contact_config = config.get('contact_map', {})
    enabled = contact_config.get('enabled', False)
    threshold = float(contact_config.get('threshold_m', .005))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Contact map threshold must be finite and positive.')
    state = _restore(scene)
    if state and _semantic_config(state['config']) == _semantic_config(config):
        if cache is not None:
            attach_cache(scene, cache)
        update(scene)
        return status(scene)
    clear(scene)
    pointer = scene.as_pointer()
    _busy.add(pointer)
    try:
        token = uuid.uuid4().hex
        collection = bpy.data.collections.new('GRAIL Newton Sequence')
        collection[collision._OWNER] = token
        scene.collection.children.link(collection)
        materials = [collision._material(token, name, color) for name, color in (
            ('SequenceA', (.1, .95, .3, 1.)), ('SequenceB', (1., .6, .05, 1.)),
            ('SequencePair', (1., .06, .06, 1.)), ('SequenceNormal', (.05, .9, 1., 1.)))]
        contact_map = dict(enabled=bool(enabled), threshold_m=threshold, meshes={})
        if enabled:
            for hand in hands:
                for name in hand['meshes']:
                    count = len(records[name]['vertices'])
                    contact_map['meshes'][name] = dict(vertex_count=count,
                        distances_m=np.full(count, threshold).tolist(), scores=np.zeros(count).tolist())
        camera, canonical, view = collision._canonical_display(scene, collection, token,
            hands, wrists, [], settings, materials, contact_map)
        label = next(obj.name for obj in collection.objects if obj.type == 'FONT' and obj.name.startswith('GC_CanonicalHeader'))
        state = dict(config=config, token=token, collection=collection.name, canonical_hands=canonical,
            canonical_camera=camera, canonical_view=view, cache_path='', label=label,
            records=records, dirty=set(), cache=None, overrides={}, glyphs=[], last_signature=None,
            current=False, message='Waiting for matching evidence')
        _states[pointer] = state
        _save(scene, state)
        # Old snapshots may be stale even while their single-frame camera remains.
        # Avoid reparsing a multi-megabyte snapshot report in its old handlers
        # on every playback frame. Its on-disk query evidence remains intact.
        collision.clear_collision(scene)
    finally:
        _busy.discard(pointer)
    if cache is not None:
        attach_cache(scene, cache)
    update(scene)
    return status(scene)


def _validate_metadata(metadata):
    if metadata.get('kind') != 'newton_collision_sequence_cache':
        raise ValueError('Unsupported contact sequence cache kind.')
    backend = metadata.get('backend', {})
    if not isinstance(backend, dict) or backend.get('ho_contact_commit') != HO_CONTACT_COMMIT or backend.get('newton_commit') != NEWTON_COMMIT:
        raise ValueError('Sequence cache has unsupported Newton provenance.')
    for key in ('ho_contact_source_sha256', 'newton_source_sha256', 'adapter_client_sha256', 'adapter_worker_sha256'):
        if not isinstance(backend.get(key), str) or not re.fullmatch('[0-9a-fA-F]{64}', backend[key]):
            raise ValueError('Missing cache source provenance: '+key)
    if (backend.get('normal_convention') != NORMAL_CONVENTION
            or backend.get('point_distance_semantics') != POINT_DISTANCE_SEMANTICS
            or backend.get('signed_normal_gap_semantics') != SIGNED_NORMAL_GAP_SEMANTICS):
        raise ValueError('Unsupported contact cache normal or distance semantics.')
    digest = metadata.get('sequence_input_digest')
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-fA-F]{64}', digest):
        raise ValueError('Missing sequence input digest.')


def _load_cache(scene, state, path):
    path = Path(bpy.path.abspath(str(path))).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        arrays = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
    _validate_metadata(metadata)
    if metadata.get('schema_version') != 1 or _semantic_config(metadata.get('config', {})) != _semantic_config(state['config']):
        raise ValueError('Sequence cache configuration does not match this display.')
    frames = arrays['frames']
    signatures = arrays['signatures']
    if frames.ndim != 1 or not len(frames) or not np.isfinite(frames).all() or (frames < 0).any() or len(set(frames.tolist())) != len(frames):
        raise ValueError('Sequence cache requires finite, unique source frames.')
    if signatures.shape != frames.shape or any(len(str(value)) != 64 for value in signatures):
        raise ValueError('Sequence cache requires one pose signature per frame.')
    mesh_names = metadata['mesh_names']
    if set(mesh_names) != set(state['config']['meshes']) or len(mesh_names) != len(set(mesh_names)):
        raise ValueError('Sequence cache original mesh bindings differ.')
    pairs = metadata['pairs']
    if pairs != sorted(state['config']['pairs']):
        raise ValueError('Sequence cache pair ordering must be canonical.')
    count = len(frames)
    for index, pair in enumerate(pairs):
        for prefix, counts_prefix in (('a', 'point_counts'), ('b', 'point_counts'),
                                     ('normal_origins', 'normal_counts'), ('normals', 'normal_counts')):
            values = arrays[f'{prefix}_{index}']
            counts = arrays[f'{counts_prefix}_{index}']
            cap = 1024 if prefix.startswith('normal') else 10000
            if (values.ndim != 3 or values.shape[0] != count or values.shape[2] != 3
                    or values.shape[1] > cap or counts.shape != (count,)
                    or counts.dtype.kind not in 'iu' or (counts < 0).any()
                    or (counts > values.shape[1]).any() or not np.isfinite(values).all()):
                raise ValueError('Invalid sampled contact array: ' + prefix)
        for row, n in enumerate(arrays[f'normal_counts_{index}']):
            if int(n) and not np.allclose(np.linalg.norm(arrays[f'normals_{index}'][row, :n], axis=1), 1., atol=1e-4):
                raise ValueError('Sequence normal directions must be unit vectors.')
    if state['config'].get('contact_map', {}).get('enabled', False):
        for hand in state['canonical_hands']:
            for name in hand['mesh_objects']:
                values = arrays[f'distance_{mesh_names.index(name)}']
                if values.shape != (count, len(state['records'][name]['vertices'])) or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Invalid original-vertex distance map: ' + name)
    state['cache'] = dict(metadata=metadata, arrays=arrays, rows={float(frame): row for row, frame in enumerate(frames)})
    state['cache_path'] = str(path)
    state['last_signature'] = None


def attach_cache(scene, path):
    state = _restore(scene)
    if not state:
        raise ValueError('Set up a persistent canonical view before attaching its cache.')
    _load_cache(scene, state, path)
    _save(scene, state)
    update(scene)
    return status(scene)


def _cache_row(state, frame, signature):
    override = state['overrides'].get(frame)
    if override and override['signature'] == signature:
        return override
    cache = state['cache']
    if not cache or frame not in cache['rows']:
        return None
    row = cache['rows'][frame]
    arrays, metadata = cache['arrays'], cache['metadata']
    if str(arrays['signatures'][row]) != signature:
        return None
    pairs = []
    for index, pair in enumerate(metadata['pairs']):
        points = int(arrays[f'point_counts_{index}'][row])
        normals = int(arrays[f'normal_counts_{index}'][row])
        pairs.append(dict(pair=pair, a=arrays[f'a_{index}'][row, :points],
            b=arrays[f'b_{index}'][row, :points], origins=arrays[f'normal_origins_{index}'][row, :normals],
            normals=arrays[f'normals_{index}'][row, :normals]))
    distances = {name: arrays[f'distance_{index}'][row] for index, name in enumerate(metadata['mesh_names'])
                 if f'distance_{index}' in arrays}
    return dict(signature=signature, pairs=pairs, distances=distances)


def _mesh_glyph(state, suffix, capacity, normal=False):
    scene_collection = bpy.data.collections[state['collection']]
    token = state['token']
    mesh = bpy.data.meshes.new('GCS_' + suffix)
    mesh[collision._OWNER] = token
    if normal:
        # A narrow rectangular shaft and a four-sided pyramid arrowhead.
        faces_one = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                     (2, 3, 7, 6), (3, 0, 4, 7), (8, 9, 12), (9, 10, 12), (10, 11, 12), (11, 8, 12)]
        nvertex = 13
    else:
        faces_one = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
        nvertex = 6
    mesh.from_pydata(np.zeros((capacity*nvertex, 3)).tolist(), [],
        [tuple(value + index*nvertex for value in face) for index in range(capacity) for face in faces_one])
    color = (.05, .9, 1., 1.) if normal else ((.1, .95, .3, 1.) if suffix.endswith('_A') else (1., .6, .05, 1.))
    material = collision._material(token, suffix, color)
    mesh.materials.append(material)
    collision._vertex_colors(mesh, np.tile(color, (capacity*nvertex, 1)))
    obj = bpy.data.objects.new('GCS_' + suffix, mesh)
    obj[collision._OWNER] = token
    obj['grail_collision_sequence_glyph'] = True
    obj.color = color
    obj.show_in_front = True
    scene_collection.objects.link(obj)
    state['glyphs'].append(obj.name)
    return obj


def _set_glyph(scene, state, suffix, positions, normals=None):
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    normal = normals is not None
    nvertex = 13 if normal else 6
    name = 'GCS_' + suffix
    obj = next((obj for obj in bpy.data.collections[state['collection']].objects
                if obj.get('grail_collision_sequence_key') == suffix), None)
    if not len(positions):
        if obj:
            _visible(scene, obj, False)
        return
    if obj is None or len(obj.data.vertices) < len(positions)*nvertex:
        if obj:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        # The first cache frame allocates a reusable fixed upper bound; later
        # sparse/dense transitions only change coordinates and collapsed slots.
        obj = _mesh_glyph(state, suffix, max(len(positions), 64), normal)
        obj['grail_collision_sequence_key'] = suffix
    capacity = len(obj.data.vertices)//nvertex
    coordinates = np.repeat(positions[:1], capacity*nvertex, axis=0).reshape(capacity, nvertex, 3)
    scale = float(scene.unit_settings.scale_length)
    settings = collision._visual_settings(state['config'])
    if normal:
        normals = np.asarray(normals)
        axes = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
        side = np.cross(normals, axes)
        side /= np.linalg.norm(side, axis=1)[:, None]
        up = np.cross(normals, side)
        length = settings['normal_length_m']/scale
        radius = .0005/scale
        offsets = np.asarray(((-1, -1), (1, -1), (1, 1), (-1, 1)))
        square = offsets[None, :, :1]*side[:, None, :] + offsets[None, :, 1:]*up[:, None, :]
        end = positions + normals*length*.76
        coordinates[:len(positions), :4] = positions[:, None, :] + square*radius
        coordinates[:len(positions), 4:8] = end[:, None, :] + square*radius
        coordinates[:len(positions), 8:12] = end[:, None, :] + square*length*.078
        coordinates[:len(positions), 12] = positions + normals*length
    else:
        axes = np.asarray(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))
        coordinates[:len(positions)] = positions[:, None, :] + axes*settings['point_radius_m']/scale
    obj.data.vertices.foreach_set('co', coordinates.ravel())
    obj.data.update()
    _visible(scene, obj, True)


def _paint(obj, distances, threshold):
    count = len(obj.data.vertices)
    attribute = obj.data.color_attributes.get('GRAIL_ContactMap')
    if attribute is None:
        collision._vertex_colors(obj.data, np.tile((.32, .34, .37, 1.), (count, 1)))
        attribute = obj.data.color_attributes['GRAIL_ContactMap']
    scores = np.clip(1 - distances/threshold, 0, 1) if distances is not None else None
    colors = collision._color(scores, 'measured', count) if scores is not None else np.tile((.32, .34, .37, 1.), (count, 1))
    attribute.data.foreach_set('color', np.asarray(colors, dtype=np.float32).ravel())
    for name, values in (('GRAIL_ContactDistance_m', distances), ('GRAIL_ContactScore', scores)):
        attr = obj.data.attributes.get(name)
        if attr is not None:
            attr.data.foreach_set('value', np.asarray(values if values is not None else np.full(count, -1.), dtype=np.float32))


def update(scene):
    pointer = scene.as_pointer()
    if pointer in _busy or _suspended.get(pointer, 0):
        return None
    state = _restore(scene)
    if not state:
        return None
    _busy.add(pointer)
    try:
        if _sync_callback:
            _sync_callback(scene)
        effective_config = _config_callback(scene) if _config_callback else state['config']
        configuration_matches = _semantic_config(effective_config) == _semantic_config(state['config'])
        for name, record in state['records'].items():
            obj = collision._source(scene, name)
            if any(mod.show_viewport or mod.show_render for mod in obj.modifiers):
                raise ValueError('Original rigid meshes are required; active modifier on '+name)
            if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1:
                raise ValueError('Deformed shape-key topology is unsupported: '+name)
            if obj.data.as_pointer() != record['data_pointer']:
                state['dirty'].add(name)
        for name in tuple(state['dirty']):
            state['records'][name] = collision._geometry(collision._source(scene, name))
            state['records'][name]['data_pointer'] = scene.objects[name].data.as_pointer()
        state['dirty'].clear()
        signature = pose_signature(scene, state['config'], {name: record['sha256'] for name, record in state['records'].items()})
        display_signature = signature if configuration_matches else signature + ':' + hashlib.sha256(
            _json(_semantic_config(effective_config)).encode()).hexdigest()
        if display_signature == state['last_signature']:
            return status(scene)
        frame = float(scene.frame_current_final-1)
        # Keep evaluating the previously bound original hands while settings
        # change. Only a refreshed valid configuration may reuse contact evidence
        # or replace the hand bindings; no layout or topology rebuild per frame.
        row = _cache_row(state, frame, signature) if configuration_matches else None
        threshold = float(state['config'].get('contact_map', {}).get('threshold_m', .005))
        for name in state['glyphs']:
            obj = scene.objects.get(name)
            if obj:
                _visible(scene, obj, False)
        for hand in state['canonical_hands']:
            wrist = np.asarray(scene.objects[hand['wrist']].matrix_world)
            rotation = wrist[:3, :3] @ np.asarray(hand['canonical_rotation']).T @ np.asarray(hand['view_rotation']).T
            offset = np.asarray(hand['display_origin']) - np.asarray(hand['display_center_local'])
            for name, copy_name in hand['mesh_objects'].items():
                source = scene.objects[name]
                obj = scene.objects[copy_name]
                local = state['records'][name]['vertices']
                if len(local) != len(obj.data.vertices):
                    _visible(scene, obj, False)
                    raise ValueError('Hand topology changed; rebuild the canonical view: ' + name)
                matrix = np.asarray(source.matrix_world)
                points = (local @ matrix[:3, :3].T + matrix[:3, 3] - wrist[:3, 3]) @ rotation + offset
                obj.data.vertices.foreach_set('co', points.ravel())
                if state['config'].get('contact_map', {}).get('enabled', False):
                    _paint(obj, row['distances'].get(name) if row else None, threshold)
                obj.data.update()
                _visible(scene, obj, True)
            if row:
                scale = float(scene.unit_settings.scale_length)
                for index, pair in enumerate(row['pairs']):
                    if not set(pair['pair']) & set(hand['mesh_objects']):
                        continue
                    prefix = f'{hand["name"]}_{index}'
                    for suffix, values in (('A', pair['a']), ('B', pair['b'])):
                        _set_glyph(scene, state, prefix+'_'+suffix, (values/scale-wrist[:3, 3]) @ rotation + offset)
                    _set_glyph(scene, state, prefix+'_N', (pair['origins']/scale-wrist[:3, 3]) @ rotation + offset, pair['normals'] @ rotation)
        if row:
            scale = float(scene.unit_settings.scale_length)
            for index, pair in enumerate(row['pairs']):
                _set_glyph(scene, state, f'Mounted_{index}_A', pair['a']/scale)
                _set_glyph(scene, state, f'Mounted_{index}_B', pair['b']/scale)
                _set_glyph(scene, state, f'Mounted_{index}_N', pair['origins']/scale, pair['normals'])
        state.update(last_signature=display_signature, current=bool(row),
                     message='Cached contact evidence' if row else ('No matching evidence | map pending'
                         if configuration_matches else 'Configuration changed | current bound pose | map pending'))
        label = scene.objects.get(state['label'])
        if label:
            label.data.body = f'NEWTON | source {frame:g} | ' + ('cached samples' if row else 'PENDING | pose only')
        state['displayed_points'] = sum(len(pair['a']) for pair in row['pairs']) if row else 0
        state['displayed_normals'] = sum(len(pair['normals']) for pair in row['pairs']) if row else 0
        for obj in bpy.data.collections[state['collection']].objects:
            if obj.type == 'FONT' and obj.name.startswith('GC_Canonical_') and '_legend' in obj.name:
                obj.data.body = ('A green | B orange | B > A normal cyan' if state['displayed_normals'] else
                    'No sampled normal arrows' if row else 'Contact map and samples pending')
        return status(scene)
    except (ValueError, KeyError, ReferenceError, RuntimeError) as error:
        state.update(current=False, last_signature=None, displayed_points=0, displayed_normals=0,
                     message='Pending: '+str(error))
        for hand in state['canonical_hands']:
            for copy_name in hand['mesh_objects'].values():
                obj = scene.objects.get(copy_name)
                if obj and state['config'].get('contact_map', {}).get('enabled', False):
                    _paint(obj, None, .005)
        for name in state['glyphs']:
            obj = scene.objects.get(name)
            if obj:
                _visible(scene, obj, False)
        return status(scene)
    finally:
        _busy.discard(pointer)


def update_frame(scene, snapshot, report, config):
    """Install new evidence for one edited pose without rebuilding the view."""
    if collision.snapshot_digest(collision.extract_snapshot(scene, config)) != collision.snapshot_digest(snapshot):
        raise ValueError('Live Newton result no longer matches the current pose.')
    report = collision.validate_report(report, snapshot)
    state = _restore(scene)
    if not state or _semantic_config(state['config']) != _semantic_config(config):
        setup(scene, config)
        state = _restore(scene)
    hands, _, _ = collision._hand_inputs(scene, config, config['meshes'])
    bindings, _ = collision._binding(scene, config['meshes'])
    measured = collision._contact_map(snapshot, config, hands, bindings)
    settings = collision._visual_settings(config)
    indices = collision._balanced_indices([pair['contact_count'] for pair in report['pairs']], settings['point_limit'])
    valid = [np.asarray([int(i) for i in selected if pair.get('normals_world') is not None and pair['normals_world'][int(i)] is not None], dtype=int)
             for pair, selected in zip(report['pairs'], indices)]
    normal_indices = [values[selected] for values, selected in zip(valid,
        collision._balanced_indices([len(values) for values in valid], settings['normal_limit']))]
    pairs = []
    for pair, selected, normals in zip(report['pairs'], indices, normal_indices):
        points_a = np.asarray(pair['points_a_world'], dtype=float).reshape(-1, 3)
        points_b = np.asarray(pair['points_b_world'], dtype=float).reshape(-1, 3)
        pairs.append(dict(pair=[pair['mesh_a'], pair['mesh_b']], a=points_a[selected], b=points_b[selected],
            origins=points_a[normals], normals=np.asarray([pair['normals_world'][int(i)] for i in normals]).reshape(-1, 3)))
    signature = pose_signature(scene, config)
    state['overrides'][float(scene.frame_current_final-1)] = dict(signature=signature, pairs=pairs,
        input_digest=report['input_digest'], backend=report['backend'],
        distances={name: np.asarray(value['distances_m']) for name, value in measured.get('meshes', {}).items()})
    state['last_signature'] = None
    update(scene)
    return status(scene)


def status(scene):
    state = _states.get(scene.as_pointer())
    if state is None:
        state = _restore(scene)
    if not state:
        return dict(enabled=False, current=False, cache_hit=False, message='Sequence view is not configured')
    return dict(enabled=True, current=state['current'], cache_hit=state['current'], message=state['message'],
        canonical_camera=state['canonical_camera'], canonical_view=state['canonical_view'],
        canonical_hands=state['canonical_hands'], contact_map=state['config'].get('contact_map', {}),
        source_frame=float(scene.frame_current_final-1), cache_path=state['cache_path'],
        cached_frames=len(state['cache']['rows']) if state['cache'] else 0,
        displayed_points=state.get('displayed_points', 0), displayed_normals=state.get('displayed_normals', 0))


def save_cache(scene, path):
    """Save a new revision including in-memory evidence for edited poses."""
    state = _restore(scene)
    if not state:
        raise ValueError('There is no persistent contact sequence to save.')
    cache = state['cache']
    entries = {}
    if cache:
        for frame, index in cache['rows'].items():
            entries[frame] = _cache_row(state, frame, str(cache['arrays']['signatures'][index]))
    entries.update(state['overrides'])
    if not entries:
        raise ValueError('No cached contact evidence is available to save.')
    frames = sorted(entries)
    mesh_names = sorted(state['config']['meshes'])
    pairs = sorted(state['config']['pairs'])
    if cache:
        metadata = dict(cache['metadata'])
    else:
        first = next(iter(state['overrides'].values()))
        metadata = dict(schema_version=1, kind='newton_collision_sequence_cache',
            backend=first['backend'], sequence_input_digest=first['input_digest'],
            sequence_input_kind='single_frame_query_with_pose_revisions')
    metadata.update(config=state['config'], mesh_names=mesh_names, pairs=pairs,
        revision_pose_updates=[dict(source_frame=frame, pose_signature=record['signature'],
            input_digest=record['input_digest'], backend=record['backend'])
            for frame, record in sorted(state['overrides'].items())])
    arrays = dict(metadata=np.asarray(_json(metadata)), frames=np.asarray(frames),
                  signatures=np.asarray([entries[frame]['signature'] for frame in frames]))
    for index, pair in enumerate(pairs):
        selected = [next(value for value in entries[frame]['pairs'] if value['pair'] == pair) for frame in frames]
        for label, key, counts_key in (('a', 'a', 'point_counts'), ('b', 'b', 'point_counts'),
                                      ('normal_origins', 'origins', 'normal_counts'), ('normals', 'normals', 'normal_counts')):
            counts = np.asarray([len(value[key]) for value in selected], dtype=np.int32)
            values = np.zeros((len(frames), int(counts.max()), 3), dtype=np.float32)
            for row, record in enumerate(selected):
                values[row, :counts[row]] = record[key]
            arrays[f'{label}_{index}'] = values
            arrays[f'{counts_key}_{index}'] = counts
    if state['config'].get('contact_map', {}).get('enabled', False):
        for hand in state['canonical_hands']:
            for name in hand['mesh_objects']:
                arrays[f'distance_{mesh_names.index(name)}'] = np.asarray(
                    [entries[frame]['distances'][name] for frame in frames], dtype=np.float32)
    path = Path(bpy.path.abspath(str(path))).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    _load_cache(scene, state, path)
    state['overrides'].clear()
    _save(scene, state)
    update(scene)
    return status(scene)


@contextmanager
def suspend(scene):
    pointer = scene.as_pointer()
    _suspended[pointer] = _suspended.get(pointer, 0)+1
    try:
        yield
    finally:
        _suspended[pointer] -= 1
        if not _suspended[pointer]:
            del _suspended[pointer]
            update(scene)


@persistent
def _on_frame(scene, *_unused):
    update(scene)


@persistent
def _on_geometry(scene, depsgraph):
    state = _states.get(scene.as_pointer())
    if state is None or scene.as_pointer() in _busy or _suspended.get(scene.as_pointer(), 0):
        return
    watched = set(state['config']['meshes']) | {hand['wrist'] for hand in state['canonical_hands']} | {'Native_Qpos_Controller', 'ContactEdit_Curves'}
    changed = False
    for entry in depsgraph.updates:
        source = entry.id.original
        if isinstance(source, bpy.types.Mesh):
            for name in state['config']['meshes']:
                if scene.objects[name].data == source:
                    state['dirty'].add(name)
                    changed = True
        elif isinstance(source, bpy.types.Object) and source.name in state['records'] and entry.is_updated_geometry:
            state['dirty'].add(source.name)
            changed = True
        if source.name in watched or isinstance(source, bpy.types.Action):
            changed = True
    if changed:
        update(scene)


@persistent
def _on_load(_unused):
    _states.clear()
    _busy.clear()
    _suspended.clear()
    # No embedded scripts or external processes run on load. The caller's
    # already-enabled native preview callback may decline until explicitly enabled.
    def refresh():
        for scene in bpy.data.scenes:
            if scene.get(_PROPERTY):
                update(scene)
        return None
    bpy.app.timers.register(refresh, first_interval=.1)


def register_handlers(sync_callback=None, config_callback=None):
    global _sync_callback, _config_callback
    _sync_callback = sync_callback
    _config_callback = config_callback
    for handlers, callback in ((bpy.app.handlers.frame_change_post, _on_frame),
        (bpy.app.handlers.depsgraph_update_post, _on_geometry), (bpy.app.handlers.load_post, _on_load),
        (bpy.app.handlers.render_pre, _on_frame)):
        if callback not in handlers:
            handlers.append(callback)


def unregister_handlers():
    global _sync_callback, _config_callback
    for handlers, callback in ((bpy.app.handlers.frame_change_post, _on_frame),
        (bpy.app.handlers.depsgraph_update_post, _on_geometry), (bpy.app.handlers.load_post, _on_load),
        (bpy.app.handlers.render_pre, _on_frame)):
        if callback in handlers:
            handlers.remove(callback)
    _sync_callback = None
    _config_callback = None
    _states.clear()
