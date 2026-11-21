"""Newton contact snapshots in Blender; no runtime or motion edits on import.

The caller synchronizes an already-enabled native preview before extracting a
snapshot. The optional external runtime receives original rigid triangles in
meters. Contact samples are diagnostics, not signed penetration or clearance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

import bpy
from bpy.app.handlers import persistent
import numpy as np
from mathutils import Vector

from .blender_inspection import _color, _geometry, _measure, _tree
from .inspection import canonical_vertices
from .collision import canonical_snapshot, run_query, snapshot_digest, validate_report


_OWNER = 'grail_collision_owner'
_REPORT = 'grail_collision_report'
_STALE = 'grail_collision_stale'
_building = set()
_pending = set()
_render_visibility = {}


def _source(scene, name):
    if not isinstance(name, str) or name not in scene.objects:
        raise ValueError('Collision source mesh is missing from this scene: ' + str(name))
    obj = scene.objects[name]
    if obj.type != 'MESH' or obj.get(_OWNER) or obj.get('grail_inspection_owner'):
        raise ValueError('Collision inputs must be original source meshes: ' + name)
    return obj


def _native_controls(scene):
    """Stable across save/reopen and sensitive to edits with preview disabled."""
    records = []
    for name in ('Native_Qpos_Controller', 'ContactEdit_Curves'):
        obj = scene.objects.get(name)
        if obj is None:
            continue
        animation = obj.animation_data
        action = animation.action if animation else None
        if animation and (len(animation.nla_tracks) or len(animation.drivers)):
            raise ValueError('Collision snapshots do not support native NLA tracks or drivers: ' + name)
        curves = sorted((curve.data_path, int(curve.array_index), bool(curve.mute),
                         float(curve.evaluate(scene.frame_current_final)))
                        for curve in action.fcurves) if action else []
        records.append(dict(name=name,
            properties={key: float(obj[key]) for key in sorted(obj.keys()) if key.startswith('qpos_')},
            action=action.name if action else None, curves=curves))
    # Normalize tuples to JSON lists for equality with saved scene properties.
    return json.loads(json.dumps(records, allow_nan=False))


def _binding(scene, names):
    bindings, watched = {}, set()
    for name in names:
        obj = _source(scene, name)
        geometry = _geometry(obj)
        bindings[name] = dict(mesh_sha256=geometry['sha256'],
                              matrix_world=geometry['matrix'].tolist())
        while obj is not None:
            watched.add(obj.name)
            if obj.data is not None:
                watched.add(obj.data.name)
            if obj.animation_data and obj.animation_data.action:
                watched.add(obj.animation_data.action.name)
            obj = obj.parent
    for name in ('Native_Qpos_Controller', 'ContactEdit_Curves'):
        obj = scene.objects.get(name)
        if obj is not None:
            watched.add(name)
            if obj.animation_data and obj.animation_data.action:
                watched.add(obj.animation_data.action.name)
    return bindings, sorted(watched)


def extract_snapshot(scene, config):
    """Extract the current original rigid meshes, preserving native triangle IDs.

    No embedded scripts run here. Native scenes require their caller to update
    an explicitly enabled exact preview before invoking this function.
    """
    if not isinstance(config, dict) or type(config.get('schema_version')) is not int or config['schema_version'] != 1:
        raise ValueError('Collision config requires schema_version: 1.')
    names = config.get('meshes')
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Collision config needs unique explicit source mesh names.')
    scale = float(scene.unit_settings.scale_length)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Scene unit scale must be finite and positive.')
    frame = float(scene.frame_current + scene.frame_subframe - 1)
    if not math.isfinite(frame) or frame < 0:
        raise ValueError('Source frame must be finite and nonnegative (Blender frame minus one).')
    meshes = []
    for name in names:
        geometry = _geometry(_source(scene, name))
        matrix = geometry['matrix'].copy()
        matrix[:3, 3] *= scale
        meshes.append(dict(name=name, vertices=(geometry['vertices'] * scale).tolist(),
                           triangles=geometry['faces'].tolist(), matrix_world=matrix.tolist()))
    # Backend validation also checks explicit pair names and rejects implicit
    # all-to-all collision policies or incomplete mesh input.
    return canonical_snapshot(dict(schema_version=1, source_frame=frame, units='m',
                                   meshes=meshes, pairs=config.get('pairs')))


def current_report(scene):
    raw = scene.get(_REPORT)
    return None if not raw else json.loads(raw)


def collision_is_current(scene, report=None):
    report = current_report(scene) if report is None else report
    if not report or scene.get(_STALE, False):
        return False
    try:
        if (abs(float(scene.frame_current + scene.frame_subframe) - report['blender_frame']) > 1e-6
                or float(scene.unit_settings.scale_length) != report['scene_unit_scale_m']
                or _native_controls(scene) != report['native_controls']):
            return False
        bindings, _ = _binding(scene, list(report['bindings']))
        return (bindings == report['bindings'] and all(
            name in scene.objects and np.asarray(scene.objects[name].matrix_world).tolist() == matrix
            for name, matrix in report.get('wrist_matrices', {}).items()))
    except (ValueError, KeyError, TypeError, ReferenceError, AttributeError):
        return False


def _collections(root):
    yield root
    for child in root.children:
        yield from _collections(child)


def clear_collision(scene):
    """Remove only owned diagnostics; retain shared data and user annotations."""
    token = scene.get(_OWNER)
    if not token:
        return
    direct = list(scene.collection.children)
    if any(c.get(_OWNER) == token and c not in direct for c in _collections(scene.collection)):
        raise ValueError('Collision collection was moved; restore it to the scene root before clearing.')
    for collection in direct:
        if collection.get(_OWNER) != token:
            continue
        shared = any(other != scene and collection in tuple(_collections(other.collection))
                     for other in bpy.data.scenes)
        if shared:
            if any(obj.get(_OWNER) != token for obj in collection.objects) or len(collection.children):
                raise ValueError('Shared collision collection contains user data; make it independent before clearing.')
            scene.collection.children.unlink(collection)
            continue
        for obj in list(collection.objects):
            if obj.get(_OWNER) != token:
                continue
            if any(other != collection for other in obj.users_collection):
                collection.objects.unlink(obj)
                continue
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0 and data.get(_OWNER) == token:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Curve):
                    bpy.data.curves.remove(data)
                elif isinstance(data, bpy.types.Camera):
                    bpy.data.cameras.remove(data)
        if not collection.objects and not collection.children:
            bpy.data.collections.remove(collection)
        else:
            del collection[_OWNER]
    for material in list(bpy.data.materials):
        if material.get(_OWNER) == token and material.users == 0:
            bpy.data.materials.remove(material)
    for name in (_OWNER, _REPORT, _STALE):
        if name in scene:
            del scene[name]
    _pending.discard(scene.as_pointer())


def _material(token, name, color):
    material = bpy.data.materials.new('GC_' + name)
    material[_OWNER] = token
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = color
    shader.inputs['Emission Color'].default_value = color
    shader.inputs['Emission Strength'].default_value = .5
    shader.inputs['Roughness'].default_value = .7
    return material


def _point_cloud(collection, token, name, points, radius, material):
    # Octahedral points are native Blender geometry, visible in Workbench and
    # CPU renderers without external point-cloud shaders or dependencies.
    axes = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
    oct_faces = np.array([(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
                          (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)])
    vertices = (points[:, None, :] + radius * axes[None, :, :]).reshape(-1, 3)
    faces = (oct_faces[None, :, :] + 6 * np.arange(len(points))[:, None, None]).reshape(-1, 3)
    mesh = bpy.data.meshes.new('GC_' + name)
    mesh[_OWNER] = token
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.materials.append(material)
    _vertex_colors(mesh, np.tile(material.diffuse_color, (len(vertices), 1)))
    mesh.update()
    obj = bpy.data.objects.new('GC_' + name, mesh)
    obj[_OWNER] = token
    obj.color = material.diffuse_color
    obj.show_in_front = True
    collection.objects.link(obj)
    return obj


def _segments(collection, token, name, a, b, radius, material):
    curve = bpy.data.curves.new('GC_' + name, 'CURVE')
    curve[_OWNER] = token
    curve.dimensions = '3D'
    curve.bevel_depth = radius
    curve.bevel_resolution = 0
    curve.materials.append(material)
    for point_a, point_b in zip(a, b):
        spline = curve.splines.new('POLY')
        spline.points.add(1)
        spline.points[0].co = (*point_a, 1.)
        spline.points[1].co = (*point_b, 1.)
    obj = bpy.data.objects.new('GC_' + name, curve)
    obj[_OWNER] = token
    obj.color = material.diffuse_color
    obj.show_in_front = True
    collection.objects.link(obj)
    return obj


def _visual_settings(config):
    settings = dict(point_limit=2048, normal_limit=192, point_radius_m=.0015,
                    normal_length_m=.025)
    raw = config.get('visualization', {})
    if not isinstance(raw, dict) or set(raw) - set(settings):
        raise ValueError('Unknown collision visualization settings.')
    settings.update(raw)
    for key, cap in (('point_limit', 10000), ('normal_limit', 1024)):
        if type(settings[key]) is not int or not 0 <= settings[key] <= cap:
            raise ValueError(f'{key} must be an integer from 0 to {cap}.')
    for key in ('point_radius_m', 'normal_length_m'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + ' must be finite and positive.')
    return settings


def _hand_inputs(scene, config, mesh_names):
    hands, wrist_matrices, watched = [], {}, set()
    raw = config.get('hands', [])
    if not isinstance(raw, list):
        raise ValueError('Canonical hands must be an explicit list.')
    used, names = set(), set()
    for hand in raw:
        if not isinstance(hand, dict) or not isinstance(hand.get('name'), str) or not hand['name'] or hand['name'] in names:
            raise ValueError('Canonical hands require unique nonempty names.')
        names.add(hand['name'])
        meshes = hand.get('meshes')
        if (not isinstance(meshes, list) or not meshes
                or any(not isinstance(name, str) or name not in mesh_names or name in used for name in meshes)
                or len(set(meshes)) != len(meshes)):
            raise ValueError('Canonical hand meshes must be unique query inputs.')
        used.update(meshes)
        wrist_name = hand.get('wrist')
        if not isinstance(wrist_name, str) or wrist_name not in scene.objects:
            raise ValueError('Canonical hand wrist is missing: ' + str(wrist_name))
        wrist = scene.objects[wrist_name]
        if wrist.get(_OWNER) or wrist.get('grail_inspection_owner'):
            raise ValueError('Canonical wrists must be original scene objects.')
        matrix = np.asarray(wrist.matrix_world, dtype=float)
        rotation = np.asarray(hand.get('canonical_rotation', np.eye(3)), dtype=float)
        canonical_vertices(np.zeros((1, 3)), matrix, rotation)
        wrist_matrices[wrist_name] = matrix.tolist()
        while wrist is not None:
            watched.add(wrist.name)
            if wrist.animation_data and wrist.animation_data.action:
                watched.add(wrist.animation_data.action.name)
            wrist = wrist.parent
        hands.append(dict(name=hand['name'], wrist=wrist_name, meshes=meshes,
                          canonical_rotation=rotation.tolist()))
    return hands, wrist_matrices, watched


def _balanced_indices(counts, limit):
    """Share the display budget across all nonempty pairs, deterministically."""
    allocations = [0] * len(counts)
    active = [index for index, count in enumerate(counts) if count]
    remaining = min(limit, sum(counts))
    while remaining and active:
        share = max(1, remaining // len(active))
        for index in active:
            take = min(counts[index] - allocations[index], share, remaining)
            allocations[index] += take
            remaining -= take
            if not remaining:
                break
        active = [index for index in active if allocations[index] < counts[index]]
    return [np.linspace(0, count - 1, take, dtype=np.int64) if take else np.empty(0, dtype=np.int64)
            for count, take in zip(counts, allocations)]


def _arrows(collection, token, name, origins, normals, length, radius, material):
    """Four-sided arrowheads make the signed query direction unambiguous."""
    starts, ends = [], []
    for origin, normal in zip(origins, normals):
        tip = origin + normal * length
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        side = np.cross(normal, axis)
        side /= np.linalg.norm(side)
        up = np.cross(normal, side)
        starts.append(origin)
        ends.append(tip)
        for offset in (side, -side, up, -up):
            starts.append(tip)
            ends.append(tip - normal * length * .24 + offset * length * .11)
    return _segments(collection, token, name, starts, ends, radius, material)


def _canonical_copy(collection, token, name, points, faces, source):
    mesh = bpy.data.meshes.new('GC_' + name)
    mesh[_OWNER] = token
    mesh.from_pydata(points.tolist(), [], faces.tolist())
    # Reference original materials without editing them; geometry is independent.
    for material in source.data.materials:
        mesh.materials.append(material)
    # _geometry triangulates polygons. Keep each triangle's original material.
    for polygon, triangle in zip(mesh.polygons, source.data.loop_triangles):
        polygon.material_index = source.data.polygons[triangle.polygon_index].material_index
    mesh.update()
    obj = bpy.data.objects.new('GC_' + name, mesh)
    obj[_OWNER] = token
    obj['grail_collision_role'] = 'canonical_hand_mesh'
    obj['source_mesh'] = source.name
    obj.color = source.color
    collection.objects.link(obj)
    return obj


def _vertex_colors(mesh, colors):
    attribute = mesh.color_attributes.new(name='GRAIL_ContactMap', type='FLOAT_COLOR', domain='POINT')
    attribute.data.foreach_set('color', np.asarray(colors, dtype=np.float32).ravel())
    mesh.color_attributes.active_color_index = len(mesh.color_attributes) - 1


def _contact_map(snapshot, config, hands, bindings):
    """Measure original hand vertices against explicit original targets in meters.

    This dense surface-distance map is independent of the sparse Newton query
    samples. Compute before clearing the preceding display so invalid targets,
    geometry or units cannot replace valid evidence with a partial map.
    """
    settings = config.get('contact_map', {})
    if not isinstance(settings, dict) or set(settings) - {'enabled', 'threshold_m', 'targets'}:
        raise ValueError('Contact map needs enabled, threshold_m and explicit targets settings.')
    enabled = settings.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('Contact map enabled must be a boolean.')
    if not enabled:
        return dict(enabled=False)
    threshold = settings.get('threshold_m', .005)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Contact map threshold_m must be finite and positive.')
    meshes = {mesh['name']: mesh for mesh in snapshot['meshes']}
    hand_names = sorted({name for hand in hands for name in hand['meshes']})
    targets = settings.get('targets')
    if (not hand_names or not isinstance(targets, list) or not targets
            or any(not isinstance(name, str) or name not in meshes or name in hand_names for name in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError('Contact map targets must be unique original query meshes outside the configured hands.')
    def world(name):
        mesh = meshes[name]
        matrix = np.asarray(mesh['matrix_world'], dtype=float)
        return np.asarray(mesh['vertices'], dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
    vertices, faces, offset = [], [], 0
    for name in targets:
        points = world(name)
        vertices.append(points)
        faces.append(np.asarray(meshes[name]['triangles'], dtype=np.int64) + offset)
        offset += len(points)
    tree = _tree(np.concatenate(vertices), np.concatenate(faces))
    measured = {}
    for name in hand_names:
        distance, score = _measure(world(name), tree, float(threshold))
        measured[name] = dict(vertex_count=len(distance), distances_m=distance.tolist(),
            scores=score.tolist(), source_binding=bindings[name])
    return dict(enabled=True, kind='measured_unsigned_surface_proximity', targets=targets,
        threshold_m=float(threshold), source_frame=snapshot['source_frame'], units='m',
        semantics='Unsigned distance from each original hand vertex to the selected original target triangles; not sensor contact, prediction, penetration or force.',
        score_formula='clip(1 - distance_m / threshold_m, 0, 1)',
        color_attribute='GRAIL_ContactMap', low_color='blue: distance >= threshold',
        high_color='yellow: distance = 0',
        target_bindings={name: bindings[name] for name in targets}, meshes=measured)


def _map_material(token):
    material = _material(token, 'MeasuredContactMap', (.25, .27, .3, 1.))
    nodes = material.node_tree.nodes
    attribute = nodes.new('ShaderNodeVertexColor')
    attribute.layer_name = 'GRAIL_ContactMap'
    shader = nodes.get('Principled BSDF')
    material.node_tree.links.new(attribute.outputs['Color'], shader.inputs['Base Color'])
    material.node_tree.links.new(attribute.outputs['Color'], shader.inputs['Emission Color'])
    return material


def _paint_contact_map(obj, record, material):
    mesh = obj.data
    _vertex_colors(mesh, _color(record['scores'], 'measured', record['vertex_count']))
    for name, values in (('GRAIL_ContactDistance_m', record['distances_m']),
                         ('GRAIL_ContactScore', record['scores'])):
        attribute = mesh.attributes.new(name=name, type='FLOAT', domain='POINT')
        attribute.data.foreach_set('value', values)
    mesh.materials.clear()
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.material_index = 0
    obj['grail_contact_map'] = 'measured_unsigned_surface_proximity'


def _map_legend(collection, token, base, height, span, contact_map, surface, text_material):
    left, right = base[0] - span * .56, base[0] + span * .56
    bottom = base[1] - height * .5
    values = np.linspace(0, 1, 33)
    x = left + values * (right - left)
    points = np.asarray([(value, bottom + y * span, span * .8) for value in x for y in (.13, .18)])
    faces = [(2*i, 2*i+2, 2*i+3, 2*i+1) for i in range(len(x)-1)]
    mesh = bpy.data.meshes.new('GC_ContactMapLegend')
    mesh[_OWNER] = token
    mesh.from_pydata(points.tolist(), [], faces)
    _vertex_colors(mesh, _color(np.repeat(values, 2), 'measured', len(points)))
    mesh.materials.append(surface)
    obj = bpy.data.objects.new('GC_ContactMapLegend', mesh)
    obj[_OWNER] = token
    obj['grail_collision_role'] = 'canonical_map_legend'
    collection.objects.link(obj)
    _text(collection, token, 'MapLegendTitle', 'Measured proximity | unsigned distance',
          (base[0], bottom + span*.28, span*.9), span*.038, text_material)
    for name, label, px in (('far', f'>= {contact_map["threshold_m"]*1000:g} mm', left), ('near', '0 mm', right)):
        _text(collection, token, 'MapLegend_'+name, label,
              (px, bottom + span*.025, span*.9), span*.038, text_material)


def _text(collection, token, name, body, position, size, material):
    curve = bpy.data.curves.new('GC_' + name, 'FONT')
    curve[_OWNER] = token
    curve.body = body
    curve.align_x = 'CENTER'
    curve.size = size
    curve.materials.append(material)
    obj = bpy.data.objects.new('GC_' + name, curve)
    obj[_OWNER] = token
    obj.location = position
    obj.color = material.diffuse_color
    collection.objects.link(obj)
    return obj


def _canonical_display(scene, collection, token, hands, wrists, samples, settings, materials, contact_map):
    """Copy the current articulation into rigid wrist-local, camera-facing tiles.

    All stored origins, mesh vertices and wrist translations use Blender units.
    Query positions are in meters and divided by scale before transformation;
    normals receive rotation only. The same original contact indices are used
    in mounted and canonical glyphs.
    """
    if not hands:
        return None, [], None
    scale = float(scene.unit_settings.scale_length)
    view_rotation = np.asarray((-Vector((0., -1., .7))).to_track_quat('-Z', 'Y').to_matrix()).T
    planned = []
    for hand in hands:
        wrist = np.asarray(wrists[hand['wrist']])
        rotation = np.asarray(hand['canonical_rotation'])
        records = {name: _geometry(_source(scene, name)) for name in hand['meshes']}
        local = {name: canonical_vertices(record['world'], wrist, rotation) @ view_rotation.T
                 for name, record in records.items()}
        points = np.concatenate(list(local.values()))
        low, high = points.min(axis=0), points.max(axis=0)
        span = max(float(np.max(high - low)), .08 / scale)
        planned.append((hand, wrist, rotation, records, local, (low + high) / 2, span))
    span = max(item[-1] for item in planned)
    # Place copies beyond the queried assembly without moving or hiding it.
    max_x = max(float(record['world'][:, 0].max()) for item in planned for record in item[3].values())
    base = np.array((max(100. / scale, max_x + 10. / scale), 0., 0.))
    tile_gap = span * 1.55
    height = tile_gap * len(planned) + span * .4
    if contact_map['enabled']:
        height += span * .5
    width = span * 1.65
    camera_data = bpy.data.cameras.new('GC_CanonicalContacts')
    camera_data[_OWNER] = token
    camera_data.type = 'ORTHO'
    aspect = scene.render.resolution_x * scene.render.pixel_aspect_x / max(1., scene.render.resolution_y * scene.render.pixel_aspect_y)
    camera_data.ortho_scale = max(width, height * aspect)
    camera_data.clip_start = .001 / scale
    camera_data.clip_end = max(height * 10, 1. / scale)
    camera = bpy.data.objects.new('GC_CanonicalContacts', camera_data)
    camera[_OWNER] = token
    camera['grail_collision_role'] = 'canonical_camera'
    camera.location = base + np.array((0., 0., height * 3))
    collection.objects.link(camera)
    white = _material(token, 'CanonicalLabel', (.95, .97, 1., 1.))
    map_material = _map_material(token) if contact_map['enabled'] else None
    canonical = []
    for index, (hand, wrist, rotation, records, local, center, hand_span) in enumerate(planned):
        origin = base + np.array((0., ((len(planned) - 1) * .5 - index) * tile_gap, 0.))
        rotation_world_to_display = wrist[:3, :3] @ rotation.T @ view_rotation.T
        objects = {}
        for name, vertices in local.items():
            obj = _canonical_copy(collection, token, hand['name'] + '_' + name,
                                  vertices - center + origin, records[name]['faces'], scene.objects[name])
            if map_material:
                _paint_contact_map(obj, contact_map['meshes'][name], map_material)
            objects[name] = obj.name
        paired = []
        for sample in samples:
            if sample['mesh_a'] not in hand['meshes'] and sample['mesh_b'] not in hand['meshes']:
                continue
            a = (sample['a'] - wrist[:3, 3]) @ rotation_world_to_display - center + origin
            b = (sample['b'] - wrist[:3, 3]) @ rotation_world_to_display - center + origin
            name = f'Canonical_{index}_{sample["pair_index"]}'
            point_objects = [_point_cloud(collection, token, name + suffix, points,
                settings['point_radius_m'] / scale, material).name
                for suffix, points, material in (('_A', a, materials[0]), ('_B', b, materials[1]))]
            _segments(collection, token, name + '_pairs', a, b, .00035 / scale, materials[2])
            normal_object = None
            if len(sample['normals']):
                origins = (sample['normal_origins'] - wrist[:3, 3]) @ rotation_world_to_display - center + origin
                normals = sample['normals'] @ rotation_world_to_display
                normal_object = _arrows(collection, token, name + '_normals', origins, normals,
                    settings['normal_length_m'] / scale, .0005 / scale, materials[3]).name
            paired.append(dict(pair_index=sample['pair_index'], contact_indices=sample['indices'].tolist(),
                               normal_indices=sample['normal_indices'].tolist(), points_a_object=point_objects[0],
                               points_b_object=point_objects[1], normal_object=normal_object))
        label_z = origin[2] + span * .7
        _text(collection, token, f'Canonical_{index}_title',
              f'{hand["name"].upper()} | wrist-local hand',
              (origin[0], origin[1] + span * .61, label_z), span * .05, white)
        normal_legend = ('A green | B orange | normal B > A cyan' if any(len(sample['normals']) for sample in samples)
                         else 'A green | B orange | no normal arrows')
        _text(collection, token, f'Canonical_{index}_legend', normal_legend,
              (origin[0], origin[1] - span * .65, label_z), span * .04, white)
        canonical.append(dict(name=hand['name'], wrist=hand['wrist'], wrist_matrix_world=wrist.tolist(),
            canonical_rotation=rotation.tolist(), view_rotation=view_rotation.tolist(),
            display_origin=origin.tolist(), display_center_local=center.tolist(),
            mesh_objects=objects, samples=paired,
            interpretation='Rigid wrist-local snapshot of actual finger pose. Newton query normals are not hand surface normals or contact forces.'))
    _text(collection, token, 'CanonicalHeader',
          f'NEWTON | source {scene.frame_current_final - 1:g} | sampled',
          (base[0], height * .5 - span * .08, span * .8), span * .05, white)
    if map_material:
        _map_legend(collection, token, base, height, span, contact_map, map_material, white)
    view = dict(camera=camera.name, center=base.tolist(), width=width, height=height,
                rotation=list(camera.rotation_euler.to_quaternion()), units='Blender units')
    return camera.name, canonical, view


def display_report(scene, snapshot, report, config=None):
    """Validate before replacing diagnostics; never change source scene data."""
    snapshot = canonical_snapshot(snapshot)
    report = validate_report(report, snapshot)
    query_config = dict(schema_version=1, meshes=[mesh['name'] for mesh in snapshot['meshes']],
                        pairs=snapshot['pairs'])
    config = query_config if config is None else config
    if not isinstance(config, dict):
        raise ValueError('Collision config must be an object.')
    if snapshot_digest(extract_snapshot(scene, config)) != snapshot_digest(snapshot):
        raise ValueError('Collision input no longer matches the scene; query the current pose first.')
    settings = _visual_settings(config)
    hands, wrists, wrist_watched = _hand_inputs(scene, config, query_config['meshes'])
    bindings, watched = _binding(scene, query_config['meshes'])
    contact_map = _contact_map(snapshot, config, hands, bindings)
    watched = sorted(set(watched) | wrist_watched)
    controls = _native_controls(scene)
    scale = float(scene.unit_settings.scale_length)
    token = uuid.uuid4().hex
    pointer = scene.as_pointer()
    _building.add(pointer)
    try:
        # Validation above protects the preceding valid result on input errors.
        clear_collision(scene)
        collection = bpy.data.collections.new('GRAIL Newton Collision')
        collection[_OWNER] = token
        scene.collection.children.link(collection)
        scene[_OWNER] = token
        material_a = _material(token, 'SourceContact', (.1, .95, .3, 1.))
        material_b = _material(token, 'TargetContact', (1., .6, .05, 1.))
        material_line = _material(token, 'ContactPair', (1., .06, .06, 1.))
        material_normal = _material(token, 'ContactNormal', (.05, .9, 1., 1.))
        counts = [pair['contact_count'] for pair in report['pairs']]
        point_indices = _balanced_indices(counts, settings['point_limit'])
        valid_normal_indices = [np.asarray([int(value) for value in indices
            if pair.get('normals_world') is not None and pair['normals_world'][value] is not None], dtype=np.int64)
            for pair, indices in zip(report['pairs'], point_indices)]
        normal_indices = [valid[selected] for valid, selected in zip(valid_normal_indices,
            _balanced_indices([len(valid) for valid in valid_normal_indices], settings['normal_limit']))]
        samples = []
        shown, normal_count = 0, 0
        for index, (pair, selected, normals_selected) in enumerate(zip(report['pairs'], point_indices, normal_indices)):
            if not len(selected):
                continue
            # Full arrays remain in the report even when viewport geometry is
            # bounded to prevent an unexpectedly huge contact manifold freeze.
            a = np.asarray(pair['points_a_world'], dtype=float)[selected] / scale
            b = np.asarray(pair['points_b_world'], dtype=float)[selected] / scale
            normals = np.asarray([pair['normals_world'][int(value)] for value in normals_selected], dtype=float).reshape(-1, 3)
            normal_origins = np.asarray(pair['points_a_world'], dtype=float)[normals_selected] / scale
            for suffix, points, material in (('A', a, material_a), ('B', b, material_b)):
                obj = _point_cloud(collection, token, f'{index}_{suffix}', points, settings['point_radius_m'] / scale, material)
                obj['mesh_a'], obj['mesh_b'] = pair['mesh_a'], pair['mesh_b']
            _segments(collection, token, f'{index}_pairs', a, b, .0004 / scale, material_line)
            if len(normals):
                _arrows(collection, token, f'{index}_normals', normal_origins, normals,
                        settings['normal_length_m'] / scale, .0005 / scale, material_normal)
            samples.append(dict(pair_index=index, mesh_a=pair['mesh_a'], mesh_b=pair['mesh_b'],
                indices=selected, a=a, b=b, normal_indices=normals_selected, normals=normals,
                normal_origins=normal_origins))
            shown += len(selected)
            normal_count += len(normals)
        camera, canonical, canonical_view = _canonical_display(scene, collection, token,
            hands, wrists, samples, settings, (material_a, material_b, material_line, material_normal), contact_map)
        envelope = dict(schema_version=1, kind='newton_collision_snapshot',
            source_frame=snapshot['source_frame'], blender_frame=float(scene.frame_current + scene.frame_subframe),
            scene_unit_scale_m=scale, input_digest=snapshot_digest(snapshot),
            bindings=bindings, native_controls=controls, watched_ids=watched,
            wrist_matrices=wrists, canonical_camera=camera, canonical_hands=canonical,
            canonical_view=canonical_view,
            contact_map=contact_map,
            result=report, collection=collection.name, owner=token,
            visualization=dict(total_contacts=sum(counts), displayed_contacts=shown,
                display_limit=settings['point_limit'], point_radius_m=settings['point_radius_m'],
                displayed_normals=normal_count, normal_limit=settings['normal_limit'],
                normals_available=any('normals_world' in pair for pair in report['pairs']),
                normal_length_m=settings['normal_length_m'], normal_color='cyan', normal_anchor='mesh_a query point',
                normal_direction='mesh_b toward mesh_a; Newton query normal, not contact force or hand surface normal',
                sampling='Balanced per pair; canonical and mounted views share contact indices',
                samples=[dict(pair_index=sample['pair_index'], contact_indices=sample['indices'].tolist(),
                              normal_indices=sample['normal_indices'].tolist()) for sample in samples],
                source_color='green', target_color='orange', connector_color='red',
                connector_meaning='Unsigned distance between Newton contact samples; not penetration depth'),
            automatic_acceptance=False)
        scene[_REPORT] = json.dumps(envelope, sort_keys=True, allow_nan=False)
        scene[_STALE] = False
        return envelope
    except Exception:
        # Only the newly allocated owned data is removed on creation errors.
        if scene.get(_OWNER) == token:
            clear_collision(scene)
        raise
    finally:
        _building.discard(pointer)


def build_collision(scene, config, output_dir):
    snapshot = extract_snapshot(scene, config)
    report = run_query(snapshot, config.get('runtime'), output_dir)
    return display_report(scene, snapshot, report, config=config)


def export_collision(scene, path):
    report = current_report(scene)
    if not collision_is_current(scene, report):
        raise ValueError('Collision snapshot is missing or stale; refresh before exporting.')
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def mark_collision_stale(scene):
    """Hide only owned diagnostics in this scene's view layers, not source data."""
    token = scene.get(_OWNER)
    if not token:
        return False
    scene[_STALE] = True
    for layer in scene.view_layers:
        for obj in layer.objects:
            if obj.get(_OWNER) == token:
                obj.hide_set(True, view_layer=layer)
    return True


@persistent
def _on_frame(scene, *_unused):
    if scene.as_pointer() in _building:
        return
    report = current_report(scene)
    if report and abs(float(scene.frame_current + scene.frame_subframe) - report['blender_frame']) > 1e-6:
        mark_collision_stale(scene)


@persistent
def _on_geometry(scene, depsgraph):
    pointer = scene.as_pointer()
    if pointer in _building:
        return
    report = current_report(scene)
    if not report:
        return
    watched = set(report['watched_ids'])
    if any(getattr(update.id.original, 'name', None) in watched or isinstance(update.id, bpy.types.Scene)
           for update in depsgraph.updates):
        _pending.add(pointer)
        if not bpy.app.timers.is_registered(_check_pending):
            bpy.app.timers.register(_check_pending, first_interval=.1)


def _check_pending():
    pending = set(_pending)
    _pending.clear()
    for scene in bpy.data.scenes:
        if scene.as_pointer() in pending and not collision_is_current(scene):
            mark_collision_stale(scene)
    return None


@persistent
def _on_load(_unused):
    # Loading .blend files never starts the external runtime or scene scripts.
    # load_post can precede final depsgraph transform evaluation. Defer stale
    # checks until that settles; otherwise a correct saved snapshot is falsely
    # marked stale from transient identity matrix_world values.
    _pending.clear()
    _render_visibility.clear()
    for scene in bpy.data.scenes:
        if current_report(scene):
            _pending.add(scene.as_pointer())
    if _pending and not bpy.app.timers.is_registered(_check_pending):
        bpy.app.timers.register(_check_pending, first_interval=.1)


@persistent
def _on_render_pre(scene, *_unused):
    """Suppress stale diagnostics for this render, even for shared objects.

    Blender serializes rendering. Restoring flags after render/cancel preserves
    the other scene's independent snapshot and all user presentation settings.
    View-layer hide_set alone would leave stale markers visible in a render.
    """
    if not current_report(scene) or collision_is_current(scene):
        return
    token = scene.get(_OWNER)
    pointer = scene.as_pointer()
    if pointer in _render_visibility:
        return
    changes = []
    for obj in scene.objects:
        if obj.get(_OWNER) == token:
            changes.append((obj, obj.hide_render))
            obj.hide_render = True
    _render_visibility[pointer] = changes


@persistent
def _on_render_done(scene, *_unused):
    for obj, hidden in _render_visibility.pop(scene.as_pointer(), []):
        try:
            obj.hide_render = hidden
        except ReferenceError:
            pass


def _handlers():
    return ((bpy.app.handlers.frame_change_post, _on_frame),
            (bpy.app.handlers.depsgraph_update_post, _on_geometry),
            (bpy.app.handlers.load_post, _on_load),
            (bpy.app.handlers.render_pre, _on_render_pre),
            (bpy.app.handlers.render_post, _on_render_done),
            (bpy.app.handlers.render_cancel, _on_render_done))


def register_handlers():
    for handlers, callback in _handlers():
        if callback not in handlers:
            handlers.append(callback)


def unregister_handlers():
    for scene in bpy.data.scenes:
        _on_render_done(scene)
    for handlers, callback in _handlers():
        if callback in handlers:
            handlers.remove(callback)
    _pending.clear()
    if bpy.app.timers.is_registered(_check_pending):
        bpy.app.timers.unregister(_check_pending)
