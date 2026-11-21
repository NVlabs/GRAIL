"""Read-only original-mesh contact and wrist-local hand inspection in Blender.

The source scene is never hidden, recolored, reparented or keyframed. Diagnostics
are independent static copies in a tagged collection, refreshed explicitly.
Unsigned surface proximity is not a penetration, force or physical contact test.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import uuid

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .inspection import canonical_vertices, discrepancy, mesh_digest, validate_predictions

_OWNER = 'grail_inspection_owner'
_REPORT = 'grail_inspection_report'
_SUMMARY = 'grail_inspection_summary'


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _object(scene, name, mesh=False):
    if not isinstance(name, str) or not name or name not in scene.objects:
        raise ValueError('Inspection object is missing from this scene: ' + str(name))
    obj = scene.objects[name]
    if obj.get(_OWNER):
        raise ValueError('Diagnostics cannot serve as source geometry: ' + name)
    if mesh and obj.type != 'MESH':
        raise ValueError('Original mesh object required: ' + name)
    return obj


def _geometry(obj):
    if any(mod.show_viewport or mod.show_render for mod in obj.modifiers):
        raise ValueError('Inspection requires original rigid meshes without active modifiers: ' + obj.name)
    if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1:
        raise ValueError('Inspection does not support deformed shape-key topology: ' + obj.name)
    mesh = obj.data
    vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', vertices)
    vertices = vertices.reshape(-1, 3)
    mesh.calc_loop_triangles()
    faces = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get('vertices', faces)
    faces = faces.reshape(-1, 3).astype(np.int64)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('Finite, nonempty original triangle geometry required: ' + obj.name)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError('Finite nonsingular source world transform required: ' + obj.name)
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    return dict(vertices=vertices, faces=faces, world=world, matrix=matrix,
                sha256=mesh_digest(vertices, faces), object=obj)


def _inputs(scene, config, source_frame):
    if not isinstance(config, dict) or type(config.get('schema_version')) is not int or config['schema_version'] != 1:
        raise ValueError('Inspection config requires schema_version: 1.')
    if isinstance(source_frame, (bool, np.bool_)) or not isinstance(source_frame, (int, float, np.integer, np.floating)) or not math.isfinite(float(source_frame)) or source_frame < 0:
        raise ValueError('Source frame must be a finite nonnegative real number.')
    if abs(float(source_frame) - (scene.frame_current + scene.frame_subframe - 1)) > 1e-6:
        raise ValueError('Source frame must match the current Blender frame minus one.')
    raw_threshold = config.get('threshold_m', .005)
    if isinstance(raw_threshold, (bool, np.bool_)) or not isinstance(raw_threshold, (int, float, np.integer, np.floating)):
        raise ValueError('threshold_m must be a finite positive real number.')
    threshold = float(raw_threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold_m must be finite and positive.')
    object_name = config.get('object')
    _object(scene, object_name, mesh=True)
    hands = config.get('hands')
    if not isinstance(hands, list) or not hands:
        raise ValueError('Inspection config needs at least one named hand.')
    names, used, matrices = [object_name], set(), {}
    hand_names = set()
    for hand in hands:
        if not isinstance(hand, dict) or not isinstance(hand.get('name'), str) or not hand['name']:
            raise ValueError('Each hand requires a nonempty name.')
        if hand['name'] in hand_names:
            raise ValueError('Hand names must be unique.')
        hand_names.add(hand['name'])
        wrist = _object(scene, hand.get('wrist'))
        matrix = np.asarray(wrist.matrix_world, dtype=np.float64)
        # Validate the actual authored wrist frame, including a proper optional basis.
        canonical_vertices(np.zeros((1, 3)), matrix, hand.get('canonical_rotation'))
        matrices[wrist.name] = matrix.tolist()
        meshes = hand.get('meshes')
        if not isinstance(meshes, list) or not meshes or any(not isinstance(n, str) for n in meshes):
            raise ValueError('Each hand requires original mesh object names.')
        if len(set(meshes)) != len(meshes) or any(n in used or n == object_name for n in meshes):
            raise ValueError('Hand mesh assignments must be unique and separate from the object.')
        for name in meshes:
            _object(scene, name, mesh=True)
        used.update(meshes)
        names.extend(meshes)
        digits = hand.get('digits')
        if not isinstance(digits, dict) or set(digits) != {'thumb', 'long_1', 'long_2'}:
            raise ValueError('Author digit assignments for thumb, long_1 and long_2 explicitly.')
        assigned = set()
        for digit, links in digits.items():
            if not isinstance(links, list) or not links or any(n not in meshes for n in links):
                raise ValueError('Digit meshes must belong to this hand: ' + digit)
            if len(set(links)) != len(links) or assigned.intersection(links):
                raise ValueError('Digit mesh assignments must be disjoint.')
            assigned.update(links)
    records = {name: _geometry(_object(scene, name, mesh=True)) for name in names}
    scale = float(scene.unit_settings.scale_length)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Scene unit scale must be finite and positive.')
    bindings = {name: dict(sha256=g['sha256'], vertex_count=len(g['vertices'])) for name, g in records.items()}
    predictions = validate_predictions(config.get('predictions'), bindings, source_frame)
    return records, matrices, predictions, scale, threshold


def _tree(vertices, faces):
    return BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)


def _measure(query, tree, threshold):
    distances = []
    for point in query:
        nearest = tree.find_nearest(Vector(point))
        if nearest is None or nearest[0] is None or nearest[3] is None:
            raise ValueError('Unable to measure nearest original triangle surface.')
        distances.append(float(nearest[3]))
    distances = np.asarray(distances)
    return distances, np.clip(1. - distances / threshold, 0., 1.)


def _color(values, layer, count):
    if values is None:
        return np.tile((.25, .27, .30, 1.), (count, 1))
    values = np.asarray(values)
    if layer == 'difference':
        base = np.tile((.85, .85, .85), (count, 1))
        target = np.where((values >= 0)[:, None], (1., .08, .04), (.03, .3, 1.))
        rgb = base * (1 - np.abs(values)[:, None]) + target * np.abs(values)[:, None]
    else:
        low = np.asarray((.055, .11, .25))
        high = np.asarray((1., .78, .05) if layer == 'measured' else (1., .08, .75))
        rgb = low + np.clip(values, 0., 1.)[:, None] * (high - low)
    return np.column_stack((rgb, np.ones(count)))


def _material(name, token, color=None, vertex=False):
    mat = bpy.data.materials.new(name)
    mat[_OWNER] = token
    if color is not None:
        mat.diffuse_color = color
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    if vertex:
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Roughness'].default_value = .6
        shader.inputs['Emission Strength'].default_value = .2
        attr = nodes.new('ShaderNodeVertexColor')
        attr.layer_name = 'GRAIL_Inspection'
        mat.node_tree.links.new(attr.outputs['Color'], shader.inputs['Base Color'])
        mat.node_tree.links.new(attr.outputs['Color'], shader.inputs['Emission Color'])
        mat.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    else:
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = color
        mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


def _mesh_copy(collection, token, name, vertices, faces, colors, material):
    mesh = bpy.data.meshes.new('GI_' + name)
    mesh[_OWNER] = token
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    attribute = mesh.color_attributes.new(name='GRAIL_Inspection', type='FLOAT_COLOR', domain='POINT')
    attribute.data.foreach_set('color', np.asarray(colors, dtype=np.float32).ravel())
    mesh.materials.append(material)
    obj = bpy.data.objects.new('GI_' + name, mesh)
    obj[_OWNER] = token
    collection.objects.link(obj)
    return obj


def _label(collection, token, text, position, size, camera, material, *, depth_offset=None):
    data = bpy.data.curves.new('GI_Label', 'FONT')
    data[_OWNER] = token
    data.body = text
    data.size = size
    data.align_x = 'CENTER'
    obj = bpy.data.objects.new('GI_Label', data)
    obj[_OWNER] = token
    obj.color = material.diffuse_color
    obj['grail_inspection_billboard'] = True
    obj['grail_inspection_label_anchor'] = tuple(float(x) for x in position)
    if depth_offset is not None:
        obj['grail_inspection_label_depth_offset'] = float(depth_offset)
    obj.location = Vector(position) + camera.rotation_euler.to_quaternion() @ Vector((0, 0, camera.data.ortho_scale if depth_offset is None else depth_offset))
    obj.rotation_euler = camera.rotation_euler.copy()
    data.materials.append(material)
    collection.objects.link(obj)
    return obj


def _line(collection, token, a, b, radius, material):
    data = bpy.data.curves.new('GI_Axis', 'CURVE')
    data[_OWNER] = token
    data.dimensions = '3D'
    data.bevel_depth = radius
    data.bevel_resolution = 2
    spline = data.splines.new('POLY')
    spline.points.add(1)
    spline.points[0].co = (*a, 1)
    spline.points[1].co = (*b, 1)
    data.materials.append(material)
    obj = bpy.data.objects.new('GI_Axis', data)
    obj[_OWNER] = token
    obj.color = material.diffuse_color
    collection.objects.link(obj)


def _camera(collection, token, name, center, span, direction):
    data = bpy.data.cameras.new('GI_' + name)
    data[_OWNER] = token
    data.type = 'ORTHO'
    data.ortho_scale = span * 1.75
    data.clip_start = max(span * .0001, .00001)
    data.clip_end = span * 4.5
    obj = bpy.data.objects.new('GI_' + name, data)
    obj[_OWNER] = token
    obj.location = Vector(center) + Vector(direction).normalized() * span * 3
    obj.rotation_euler = (Vector(center) - obj.location).to_track_quat('-Z', 'Y').to_euler()
    collection.objects.link(obj)
    return obj


def _tag_group(collection, group):
    for obj in collection.objects:
        if 'grail_inspection_group' not in obj:
            obj['grail_inspection_group'] = group


def _descendants(collection):
    yield collection
    for child in collection.children:
        yield from _descendants(child)


def clear_inspection(scene):
    """Remove only this scene's generated diagnostics, preserving shared copies."""
    token = scene.get(_OWNER)
    if not token:
        return
    direct = list(scene.collection.children)
    if any(c.get(_OWNER) == token and c not in direct for c in _descendants(scene.collection)):
        raise ValueError('Inspection collection was moved; restore it to the scene root before clearing.')
    for collection in direct:
        if collection.get(_OWNER) != token:
            continue
        shared = any(other != scene and collection in tuple(_descendants(other.collection)) for other in bpy.data.scenes)
        if shared:
            if any(o.get(_OWNER) != token for o in collection.objects) or len(collection.children):
                raise ValueError('Shared inspection collection contains user objects; make it independent before clearing.')
            scene.collection.children.unlink(collection)
            continue
        for obj in list(collection.objects):
            if obj.get(_OWNER) != token:
                continue
            data = obj.data
            # An explicitly linked diagnostic object in a different collection is retained.
            if any(c != collection for c in obj.users_collection):
                collection.objects.unlink(obj)
                continue
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0 and data.get(_OWNER) == token:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Curve):
                    bpy.data.curves.remove(data)
                elif isinstance(data, bpy.types.Camera):
                    bpy.data.cameras.remove(data)
        if len(collection.objects) == 0 and len(collection.children) == 0:
            bpy.data.collections.remove(collection)
        else:
            # User additions retain their scene membership and their data.
            del collection[_OWNER]
    for material in list(bpy.data.materials):
        if material.get(_OWNER) == token and material.users == 0:
            bpy.data.materials.remove(material)
    for key in (_OWNER, _REPORT, _SUMMARY):
        if key in scene:
            del scene[key]


def build_inspection(scene, config, source_frame, layer='measured'):
    """Create an exact-current-frame diagnostic snapshot; never change source data.

    Source frame must match Blender frame minus one (including subframes).
    `cameras` are generated names; render_inspection isolates their collection.
    """
    if layer not in ('measured', 'predicted', 'difference'):
        raise ValueError('Unknown inspection layer: ' + str(layer))
    records, wrist_matrices, predictions, scale, threshold = _inputs(scene, config, source_frame)
    target = records[config['object']]
    object_tree = _tree(target['world'] * scale, target['faces'])
    all_hand_vertices, all_hand_faces, offset = [], [], 0
    for name, geo in records.items():
        if name == config['object']:
            continue
        all_hand_vertices.append(geo['world'] * scale)
        all_hand_faces.append(geo['faces'] + offset)
        offset += len(geo['world'])
    hand_tree = _tree(np.concatenate(all_hand_vertices), np.concatenate(all_hand_faces))
    measurements, summary, display = {}, {}, {}
    for name, geo in records.items():
        distance, score = _measure(geo['world'] * scale, hand_tree if name == config['object'] else object_tree, threshold)
        predicted = predictions['maps'].get(name)
        residual = None if predicted is None else discrepancy(score, predicted)
        measurements[name] = dict(sha256=geo['sha256'], vertex_count=len(score), triangle_count=len(geo['faces']),
            world_matrix=geo['matrix'].tolist(), distances_m=distance.tolist(), measured_scores=score.tolist(),
            predicted_scores=None if predicted is None else predicted.tolist(),
            difference=None if residual is None else residual.tolist())
        summary[name] = dict(vertex_count=len(score), min_distance_m=float(np.min(distance)),
                            vertices_near_threshold=int(np.count_nonzero(distance <= threshold)),
                            prediction_available=predicted is not None)
        display[name] = _color({'measured': score, 'predicted': predicted, 'difference': residual}[layer], layer, len(score))
    # Do all validation and measurement before replacing an existing snapshot.
    clear_inspection(scene)
    token = uuid.uuid4().hex
    collection = bpy.data.collections.new('GRAIL Inspection ' + token[:8])
    collection[_OWNER] = token
    scene.collection.children.link(collection)
    scene[_OWNER] = token
    try:
        surface = _material('GI_ContactMap', token, vertex=True)
        white = _material('GI_Labels', token, (.93, .96, 1., 1.))
        axis_materials = [_material('GI_Axis_' + axis, token, color) for axis, color in
                          zip('XYZ', [(1., .1, .08, 1.), (.1, 1., .15, 1.), (.1, .4, 1., 1.)])]
        all_world = np.concatenate([geo['world'] for geo in records.values()])
        lo, hi = all_world.min(axis=0), all_world.max(axis=0)
        span = max(float(np.max(hi - lo)), .05 / scale)
        center = (lo + hi) / 2
        # Off-scene copies remain inspectable without any changes to source visibility.
        scene_extent = max([float(obj.matrix_world.translation.x) for obj in scene.objects if not obj.get(_OWNER)] + [float(hi[0])])
        base = np.array([scene_extent + span * 4 + 2 / scale, center[1], center[2]])
        shift = base - center
        for name, geo in records.items():
            _mesh_copy(collection, token, 'Assembly_' + name, geo['world'] + shift, geo['faces'], display[name], surface)
        cameras = []
        overview = _camera(collection, token, 'Assembly', base, span, (0., -1., .65))
        cameras.append(overview.name)
        kind = predictions.get('provenance', {}).get('kind') if predictions.get('provenance') else 'no prediction'
        title = f'{layer.upper()} | source {source_frame:g} | snapshot only'
        _label(collection, token, title, base + np.array([0, 0, span * .72]), span * .032, overview, white)
        legend = ('Unsigned proximity: navy=far, yellow=near' if layer == 'measured' else
                  ('Prediction: navy=0, magenta=1; gray=missing' if layer == 'predicted' else
                   'Prediction - measured: blue<0, red>0; gray=missing'))
        _label(collection, token, legend, base + np.array([0, 0, -span * .69]), span * .025, overview, white)
        _label(collection, token, f'{threshold * 1000:g} mm threshold | {kind}', base + np.array([0, 0, -span * .77]), span * .025, overview, white)
        _tag_group(collection, 'assembly')
        canonical = []
        for index, hand in enumerate(config['hands']):
            wrist = np.asarray(wrist_matrices[hand['wrist']])
            local = {name: canonical_vertices(records[name]['world'], wrist, hand.get('canonical_rotation')) for name in hand['meshes']}
            combined = np.concatenate(list(local.values()))
            h_lo, h_hi = combined.min(axis=0), combined.max(axis=0)
            h_span = max(float(np.max(h_hi - h_lo)), .05 / scale)
            origin = base + np.array([span * (index + 1) * 3, 0, 0])
            h_center = origin + (h_lo + h_hi) / 2
            front = _camera(collection, token, hand['name'] + '_WristView', h_center, h_span, (0., -1., .7))
            side = _camera(collection, token, hand['name'] + '_SideView', h_center, h_span, (1., -.15, .2))
            cameras.extend((front.name, side.name))
            for name in hand['meshes']:
                _mesh_copy(collection, token, hand['name'] + '_' + name, local[name] + origin,
                           records[name]['faces'], display[name], surface)
            _label(collection, token, f'{hand["name"]}: wrist-local / actual finger pose', h_center + np.array([0, 0, h_span * .74]), h_span * .034, front, white)
            _label(collection, token, title, h_center + np.array([0, 0, -h_span * .70]), h_span * .028, front, white)
            _label(collection, token, legend, h_center + np.array([0, 0, -h_span * .78]), h_span * .022, front, white)
            for digit, meshes in hand['digits'].items():
                centroid = np.concatenate([local[name] for name in meshes]).mean(axis=0)
                _label(collection, token, digit, centroid + origin, h_span * .027, front, white)
            rotation = np.asarray(hand.get('canonical_rotation', np.eye(3)))
            for axis, material, vector in zip('XYZ', axis_materials, rotation.T):
                endpoint = origin + vector * h_span * .30
                _line(collection, token, origin, endpoint, h_span * .003, material)
                _label(collection, token, 'wrist +' + axis, endpoint + vector * h_span * .05, h_span * .025, front, material)
            _tag_group(collection, 'hand:' + hand['name'])
            canonical.append(dict(hand=hand['name'], wrist=hand['wrist'], canonical_rotation=rotation.tolist(),
                                  digits=hand['digits'], cameras=[front.name, side.name],
                                  interpretation='Current native finger pose in authored wrist coordinates; no anatomical palm orientation inferred.'))
        report = dict(schema_version=1, kind='original_mesh_inspection_snapshot', source_frame=float(source_frame),
            blender_frame=float(scene.frame_current + scene.frame_subframe), layer=layer, threshold_m=threshold, config_sha256=config_digest(config),
            score_definition='max(0, 1 - unsigned point-to-original-triangle distance / threshold_m)',
            scene_unit_scale_m=scale, summary=summary, meshes=measurements, wrist_matrices=wrist_matrices,
            prediction_missing_meshes=predictions['missing_meshes'], provenance=predictions['provenance'],
            limitation='Unsigned vertex-to-surface proximity only; neither physical contact nor penetration detection. Vertex sampling may miss triangle-interior collisions.',
            snapshot_notice='Static snapshot. Refresh after native pose, geometry, prediction or frame changes.',
            collection=collection.name, cameras=cameras, canonical_hands=canonical,
            view_groups={name: bpy.data.objects[name]['grail_inspection_group'] for name in cameras}, automatic_acceptance=False)
        scene[_REPORT] = json.dumps(report, separators=(',', ':'))
        scene[_SUMMARY] = json.dumps({key: report[key] for key in ('source_frame', 'blender_frame', 'layer', 'cameras', 'collection', 'config_sha256')})
        return report
    except Exception:
        clear_inspection(scene)
        raise


def _dashboard_columns(config, records, wrist_matrices):
    """Plan rigid viewing bases and diagnostic-only uniform framing scales."""
    columns = []
    for hand in config['hands']:
        frame = np.linalg.inv(np.asarray(wrist_matrices[hand['wrist']]))
        basis = np.eye(4)
        basis[:3, :3] = np.asarray(hand.get('canonical_rotation', np.eye(3)))
        columns.append(dict(kind='canonical_hand', label=hand['name'], hand=hand['name'],
                            meshes=list(hand['meshes']), frame=basis @ frame,
                            digits=hand['digits'], direction=[0., -1., .7]))
    columns.append(dict(kind='assembly', label='Object + all hands', meshes=list(records),
                        frame=np.eye(4), direction=[0., -1., .65]))
    for column in columns:
        camera_rotation = (-Vector(column['direction'])).to_track_quat('-Z', 'Y').to_matrix()
        view = np.eye(4)
        view[:3, :3] = np.asarray(camera_rotation).T
        column['view_transform'] = view @ column.pop('frame')
        transformed = []
        for name in column['meshes']:
            points = records[name]['world']
            matrix = column['view_transform']
            transformed.append(points @ matrix[:3, :3].T + matrix[:3, 3])
        points = np.concatenate(transformed)
        low, high = points.min(axis=0), points.max(axis=0)
        extent = np.maximum(high - low, 1e-12)
        column['center'] = (low + high) / 2
        # All tiles share dimensions, but native anatomy is never rescaled.
        # Scale resides only in each generated diagnostic object's transform.
        column['diagnostic_scale'] = float(min(2.65 / extent[0], 1.27 / extent[1], 1.6 / extent[2]))
        if not math.isfinite(column['diagnostic_scale']):
            raise ValueError('Unable to frame diagnostic geometry.')
    return columns


def build_dashboard(scene, config, source_frame):
    """Show wrist-local hands and object contact layers in one native camera.

    Distances are measured once in native metres by :func:`build_inspection`.
    Each tile reuses those exact original-vertex maps and original local mesh
    coordinates; uniform tile framing is applied to diagnostic objects only.
    The retained individual cameras show the measured layer in native units.
    """
    # Plan and validate everything that depends on caller input before replacing
    # the previous snapshot. In particular, a bad frame/binding cannot erase it.
    config_digest(config)
    records, wrists, _predictions, _scale, _threshold = _inputs(scene, config, source_frame)
    columns = _dashboard_columns(config, records, wrists)
    report = build_inspection(scene, config, source_frame, layer='measured')
    token = scene[_OWNER]
    collection = scene.collection.children[report['collection']]
    try:
        count = len(columns)
        width, height = 1.75 + count * 3.2, 8.0
        # Keep the dashboard away from both original and prior inspection meshes.
        extent = max(float(obj.matrix_world.translation.x) for obj in scene.objects)
        origin = np.array([extent + width * 3, 0., 0.])
        camera = _camera(collection, token, 'Dashboard', origin, max(width, height), (0, 0, 1))
        camera.data.sensor_fit = 'VERTICAL'
        camera.data.ortho_scale = max(width, height) * 1.04
        # The source render aspect may be portrait. Diagnostic tiles outside
        # that camera gate must remain visible in the local dashboard view.
        camera.data.show_passepartout = False
        camera['grail_inspection_dashboard'] = True
        camera['grail_dashboard_width'] = width * 1.04
        camera['grail_dashboard_height'] = height * 1.04
        surface = _material('GI_DashboardContact', token, vertex=True)
        white = _material('GI_DashboardText', token, (.91, .95, 1., 1.))
        muted = _material('GI_DashboardMuted', token, (.51, .62, .73, 1.))
        warning = _material('GI_DashboardWarning', token, (1., .69, .2, 1.))
        panel = _material('GI_DashboardPanel', token, (.035, .05, .07, 1.))

        def label(text, x, y, size=.11, material=white):
            return _label(collection, token, text, origin + [x, y, 0], size,
                          camera, material, depth_offset=1.12)

        label('GRAIL G1 | HAND + CONTACT DASHBOARD', 0, 3.72, .21)
        label(f'Source frame {source_frame:g} | Blender frame {report["blender_frame"]:g} | '
              f'{report["threshold_m"] * 1000:g} mm proximity threshold', 0, 3.49, .115, muted)
        banner = label('CURRENT / UNREVIEWED SNAPSHOT | Refresh after frame or native pose edits', 0, 3.29, .11, warning)
        banner['grail_inspection_dashboard_banner'] = True
        rows = [
            dict(layer='measured', label='Measured proximity', title='Measured\nproximity',
                 legend='navy: far\nyellow: near'),
            dict(layer='predicted', label='Predicted / authored', title='Predicted /\nauthored',
                 legend='navy: 0\nmagenta: 1'),
            dict(layer='difference', label='Difference', title='Difference',
                 legend='prediction - measured\nblue: negative\nred: positive'),
        ]
        tiles, report_columns = [], []
        for index, column in enumerate(columns):
            x = -width / 2 + 1.75 + 1.6 + index * 3.2
            label(column['label'], x, 2.96, .16)
            label('Wrist local | thumb / long_1 / long_2' if column['kind'] == 'canonical_hand'
                  else 'Mounted relative hand / object geometry', x, 2.75, .087, muted)
            report_columns.append({key: value for key, value in column.items()
                                   if key not in ('view_transform', 'center')})
            report_columns[-1]['view_transform'] = column['view_transform'].tolist()
            report_columns[-1]['view_center'] = column['center'].tolist()
            for row_index, row in enumerate(rows):
                y = 1.75 - row_index * 1.94
                layer = row['layer']
                bounds = [x - 1.5, x + 1.5, y - .87, y + .87]
                vertices = origin + np.array([[bounds[0], bounds[2], -1.1], [bounds[1], bounds[2], -1.1],
                                               [bounds[1], bounds[3], -1.1], [bounds[0], bounds[3], -1.1]])
                _mesh_copy(collection, token, f'DashboardPanel_{row_index}_{index}', vertices,
                           np.array([[0, 1, 2], [0, 2, 3]]), np.tile((.035, .05, .07, 1.), (4, 1)), panel)
                framing = np.eye(4)
                framing[:3, :3] *= column['diagnostic_scale']
                framing[:3, 3] = origin + [x, y + .06, 0] - column['center'] * column['diagnostic_scale']
                objects, missing = [], []
                for name in column['meshes']:
                    record = report['meshes'][name]
                    key = {'measured': 'measured_scores', 'predicted': 'predicted_scores', 'difference': 'difference'}[layer]
                    values = record[key]
                    colors = _color(values, layer, record['vertex_count'])
                    geo = records[name]
                    obj = _mesh_copy(collection, token, f'Dashboard_{row_index}_{index}_{name}',
                                     geo['vertices'], geo['faces'], colors, surface)
                    obj.matrix_world = Matrix(framing @ column['view_transform'] @ geo['matrix'])
                    obj['grail_inspection_source_mesh'] = name
                    obj['grail_inspection_layer'] = layer
                    obj['grail_inspection_source_sha256'] = geo['sha256']
                    objects.append(obj.name)
                    if record['predicted_scores'] is None:
                        missing.append(name)
                if layer != 'measured' and missing:
                    status = ('UNAVAILABLE | no prediction' if len(missing) == len(column['meshes'])
                              else f'PARTIAL | {len(missing)} mesh maps unavailable')
                else:
                    status = ('Unsigned surface proximity' if layer == 'measured'
                              else 'Exact-frame original-vertex maps')
                label(status, x, y - .76, .095, warning if missing and layer != 'measured' else muted)
                tiles.append(dict(row=row_index, column=index, layer=layer, objects=objects,
                                  source_meshes=list(column['meshes']), missing_prediction_meshes=missing,
                                  diagnostic_scale=column['diagnostic_scale'], bounds=bounds,
                                  diagnostic_framing_matrix=framing.tolist()))
        for index, row in enumerate(rows):
            y = 1.75 - index * 1.94
            x = -width / 2 + .85
            label(row['title'], x, y + .2, .155)
            label(row['legend'], x, y - .31, .096, muted)
        provenance = report['provenance']
        kind = 'UNAVAILABLE (no prediction supplied)' if provenance is None else provenance['kind']
        label('Prediction provenance: ' + kind + ' | gray = unavailable, never zero', 0, -3.27, .107, warning)
        label('Each column uses uniform diagnostic framing. Original meshes and native motion are unchanged.', 0, -3.49, .104, muted)
        label('Measured = unsigned vertex-to-surface proximity; not physical contact, penetration or acceptance.', 0, -3.72, .103, muted)
        _tag_group(collection, 'dashboard')
        report['layer'] = 'dashboard'
        report['cameras'].append(camera.name)
        report['view_groups'][camera.name] = 'dashboard'
        report['dashboard'] = dict(camera=camera.name, group='dashboard', banner=banner.name,
            rows=[{key: value for key, value in row.items() if key != 'title'} for row in rows],
            columns=report_columns, tiles=tiles, bounds=[-width / 2, width / 2, -height / 2, height / 2],
            origin=origin.tolist(), recommended_resolution=[1600, 1000] if count >= 3 else [1200, 1000],
            geometry='Original local vertices and triangle ordering; only generated-object transforms normalize tile framing.',
            measurement_units='metres before diagnostic framing; shared maps measured once',
            individual_view_layer='measured', source_frame=float(source_frame))
        scene[_REPORT] = json.dumps(report, separators=(',', ':'))
        metadata = {key: report[key] for key in ('source_frame', 'blender_frame', 'layer', 'cameras', 'collection', 'config_sha256')}
        metadata['dashboard'] = {key: report['dashboard'][key] for key in ('camera', 'rows', 'columns', 'recommended_resolution')}
        scene[_SUMMARY] = json.dumps(metadata)
        return report
    except Exception:
        clear_inspection(scene)
        raise


def mark_dashboard_stale(scene):
    """Mark only a generated banner; never alter motion or measured evidence."""
    token = scene.get(_OWNER)
    if not token:
        return False
    marked = False
    for obj in scene.objects:
        if obj.get(_OWNER) == token and obj.get('grail_inspection_dashboard_banner') and obj.type == 'FONT':
            obj.data.body = 'STALE SNAPSHOT | Frame or pose changed - refresh dashboard'
            marked = True
    return marked


def snapshot_metadata(scene):
    raw = scene.get(_SUMMARY)
    return None if not raw else json.loads(raw)


def current_report(scene):
    raw = scene.get(_REPORT)
    return None if not raw else json.loads(raw)


def inspection_is_current(scene, report):
    if abs(float(scene.frame_current + scene.frame_subframe) - report['blender_frame']) > 1e-6:
        return False
    if float(scene.unit_settings.scale_length) != report['scene_unit_scale_m']:
        return False
    try:
        for name, record in report['meshes'].items():
            geo = _geometry(_object(scene, name, mesh=True))
            if geo['sha256'] != record['sha256'] or not np.array_equal(geo['matrix'], record['world_matrix']):
                return False
        for name, matrix in report['wrist_matrices'].items():
            if not np.array_equal(np.asarray(_object(scene, name).matrix_world), matrix):
                return False
    except (ValueError, KeyError):
        return False
    return True


def export_inspection(scene, path):
    report = current_report(scene)
    if report is None:
        raise ValueError('Refresh inspection before exporting measurements.')
    if not inspection_is_current(scene, report):
        raise ValueError('Inspection snapshot is stale; refresh before exporting.')
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def set_inspection_view(scene, report, camera_name, view_layer):
    """Isolate generated copies in this viewport's layer; never hide source objects."""
    if camera_name not in report['cameras']:
        raise ValueError('Camera is not bound to this inspection report.')
    collection = scene.collection.children.get(report['collection'])
    if collection is None or collection.get(_OWNER) != scene.get(_OWNER):
        raise ValueError('Inspection collection is missing; refresh first.')
    camera = collection.objects.get(camera_name)
    if camera is None:
        raise ValueError('Inspection camera is missing; refresh first.')
    group = camera.get('grail_inspection_group')
    if not group:
        raise ValueError('Inspection camera has no view group; refresh first.')
    for obj in collection.objects:
        if obj.get(_OWNER) != scene.get(_OWNER):
            continue
        obj.hide_set(obj.get('grail_inspection_group') != group, view_layer=view_layer)
        if obj.get('grail_inspection_billboard') and obj.get('grail_inspection_group') == group:
            offset = obj.get('grail_inspection_label_depth_offset', camera.data.ortho_scale)
            obj.location = Vector(obj['grail_inspection_label_anchor']) + camera.rotation_euler.to_quaternion() @ Vector((0, 0, offset))
            obj.rotation_euler = camera.rotation_euler.copy()
    return camera


def render_inspection(scene, report, camera_name, filepath, *, resolution=768):
    """Render only snapshot copies in a temporary scene, leaving source settings intact."""
    if not inspection_is_current(scene, report):
        raise ValueError('Inspection snapshot is stale; refresh before rendering.')
    if camera_name not in report['cameras']:
        raise ValueError('Camera is not bound to this inspection report.')
    collection = scene.collection.children.get(report['collection'])
    if collection is None or collection.get(_OWNER) != scene.get(_OWNER):
        raise ValueError('Inspection collection is missing or belongs to another snapshot.')
    camera = collection.objects.get(camera_name)
    if camera is None or camera.type != 'CAMERA':
        raise ValueError('Inspection camera is missing.')
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        width, height = resolution
    else:
        width = height = resolution
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0 for value in (width, height)):
        raise ValueError('Resolution must be a positive integer or (width, height) integer pair.')
    output = Path(filepath).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Inspection render already exists: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = bpy.data.scenes.new('GRAIL Inspection Render')
    world = bpy.data.worlds.new('GRAIL Inspection World')
    transforms = {}
    lights = []
    original_ortho_scale = camera.data.ortho_scale
    try:
        group = camera.get('grail_inspection_group')
        if not group:
            raise ValueError('Inspection camera has no bound view group; refresh first.')
        visible = [obj for obj in collection.objects if obj.get('grail_inspection_group') == group and obj.get(_OWNER) == scene.get(_OWNER)]
        for obj in visible:
            temp.collection.objects.link(obj)
        temp.world = world
        world.use_nodes = True
        world.node_tree.nodes['Background'].inputs['Color'].default_value = (.12, .15, .22, 1.)
        world.node_tree.nodes['Background'].inputs['Strength'].default_value = .4
        temp.camera = camera
        temp.render.engine = 'CYCLES'
        temp.cycles.device = 'CPU'
        temp.cycles.samples = 8
        temp.render.threads_mode = 'FIXED'
        temp.render.threads = 2
        temp.render.resolution_x = int(width)
        temp.render.resolution_y = int(height)
        temp.render.resolution_percentage = 100
        temp.render.image_settings.file_format = 'PNG'
        temp.render.image_settings.color_mode = 'RGB'
        temp.render.filepath = str(output)
        temp.view_settings.view_transform = 'Standard'
        if camera.get('grail_inspection_dashboard'):
            # The camera uses vertical sensor fit. Account for output aspect
            # without changing the saved camera or source render settings.
            camera.data.ortho_scale = max(float(camera['grail_dashboard_height']),
                                          float(camera['grail_dashboard_width']) * height / width)
        span = camera.data.ortho_scale / 1.75
        for index, offset in enumerate(((-1.2, .8, .2), (1., -.4, -.1))):
            data = bpy.data.lights.new('GI_RenderLight', 'AREA')
            data.energy = (240 if index == 0 else 120) * span * span
            data.shape = 'DISK'
            data.size = span * 2
            light = bpy.data.objects.new('GI_RenderLight', data)
            light.location = camera.location + camera.rotation_euler.to_quaternion() @ (Vector(offset) * span)
            light.rotation_euler = camera.rotation_euler.copy()
            temp.collection.objects.link(light)
            lights.append(light)
        for obj in visible:
            if obj.get('grail_inspection_billboard'):
                transforms[obj.name] = (obj.location.copy(), obj.rotation_euler.copy())
                anchor = Vector(obj['grail_inspection_label_anchor'])
                offset = obj.get('grail_inspection_label_depth_offset', camera.data.ortho_scale)
                obj.location = anchor + camera.rotation_euler.to_quaternion() @ Vector((0, 0, offset))
                obj.rotation_euler = camera.rotation_euler.copy()
        bpy.ops.render.render(write_still=True, scene=temp.name)
    finally:
        camera.data.ortho_scale = original_ortho_scale
        for name, (location, rotation) in transforms.items():
            if name in bpy.data.objects:
                bpy.data.objects[name].location = location
                bpy.data.objects[name].rotation_euler = rotation
        for light in lights:
            data = light.data
            bpy.data.objects.remove(light, do_unlink=True)
            if data.users == 0:
                bpy.data.lights.remove(data)
        bpy.data.scenes.remove(temp)
        if world.users == 0:
            bpy.data.worlds.remove(world)
    return str(output)
