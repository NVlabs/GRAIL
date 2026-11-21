"""Build a frame-bound contact playback cache without editing source motion.

This explicit operation captures native FK on Blender's main thread, queries
Newton once in its external runtime, and publishes a complete numeric NPZ only
after all frames succeed. Loading that NPZ never executes the runtime.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import bpy
import numpy as np

from . import dashboard_ui, collision_live
from .inspection import blender_collision as collision
from .inspection.blender_inspection import _measure, _tree


def _sync(scene):
    dashboard_ui._sync_enabled_preview(scene, required=True)
    bpy.context.view_layer.update()


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


def _write_npz(path, values):
    # Atomic publication prevents a cancelled build from becoming a readable
    # cache. Pickled Python objects are deliberately absent from this format.
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    try:
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **values)
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def _frame_range(scene, frame_start, frame_end):
    start = scene.frame_start if frame_start is None else frame_start
    end = scene.frame_end if frame_end is None else frame_end
    if (type(start) is not int or type(end) is not int
            or start < 1 or end < start):
        raise ValueError('Sequence bounds must be integer Blender frames with 1 <= start <= end.')
    return range(start, end + 1)


@contextmanager
def _capture_scope(scene, sync):
    """Suppress diagnostic handlers while restoring the original native pose."""
    from .inspection import blender_collision_sequence as sequence
    pointer = scene.as_pointer()
    old_frame, old_subframe = scene.frame_current, scene.frame_subframe
    old_controls = collision._native_controls(scene)
    held_collision = pointer in collision._building
    held_live = pointer in collision_live._busy
    collision._building.add(pointer)
    collision_live._busy.add(pointer)
    try:
        with sequence.suspend(scene):
            try:
                yield
            finally:
                scene.frame_set(old_frame, subframe=old_subframe)
                sync(scene)
                if collision._native_controls(scene) != old_controls:
                    raise ValueError('Sequence capture could not restore the original native controls.')
    finally:
        if not held_collision:
            collision._building.discard(pointer)
        if not held_live:
            collision_live._busy.discard(pointer)


class _SurfaceDistances:
    """Cache original target BVHs; transport queries by rigid relative motion.

    Static scale or shear is baked into the target BVH. A changing nonrigid
    transform falls back to a fresh metric BVH rather than returning distances
    in a scaled coordinate system.
    """
    def __init__(self, snapshot, config, hands):
        settings = config.get('contact_map', {})
        if not isinstance(settings, dict) or set(settings) - {'enabled', 'threshold_m', 'targets'}:
            raise ValueError('Invalid contact map settings.')
        enabled = settings.get('enabled', False)
        if type(enabled) is not bool:
            raise ValueError('Contact map enabled must be a boolean.')
        self.enabled = enabled
        self.meshes = {mesh['name']: mesh for mesh in snapshot['meshes']}
        self.hand_names = sorted({name for hand in hands for name in hand['meshes']})
        self.targets = []
        self.trees = {}
        self.vertices = {name: np.asarray(mesh['vertices'], dtype=float)
                         for name, mesh in self.meshes.items()}
        if not enabled:
            return
        threshold = settings.get('threshold_m', .005)
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or threshold <= 0):
            raise ValueError('Contact map threshold must be finite and positive.')
        self.threshold = float(threshold)
        targets = settings.get('targets')
        if (not self.hand_names or not isinstance(targets, list) or not targets
                or any(not isinstance(name, str) or name not in self.meshes
                       or name in self.hand_names for name in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError('Contact map targets must be unique original meshes outside the configured hands.')
        self.targets = targets
        for name in targets:
            linear = np.asarray(self.meshes[name]['matrix_world'], dtype=float)[:3, :3]
            vertices = self.vertices[name] @ linear.T
            faces = np.asarray(self.meshes[name]['triangles'], dtype=np.int64)
            self.trees[name] = (linear, faces, _tree(vertices, faces))

    def measure(self, matrices):
        if not self.enabled:
            return {}
        targets = []
        for name in self.targets:
            matrix = np.asarray(matrices[name], dtype=float)
            base, faces, tree = self.trees[name]
            rotation = matrix[:3, :3] @ np.linalg.inv(base)
            if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6, rtol=0)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=2e-6, rtol=0)):
                tree = _tree(self.vertices[name] @ matrix[:3, :3].T, faces)
                rotation = np.eye(3)
            targets.append((matrix[:3, 3], rotation, tree))
        distances = {}
        for name in self.hand_names:
            matrix = np.asarray(matrices[name], dtype=float)
            world = self.vertices[name] @ matrix[:3, :3].T + matrix[:3, 3]
            measured = [_measure((world - translation) @ rotation, tree, self.threshold)[0]
                        for translation, rotation, tree in targets]
            values = np.minimum.reduce(measured)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Invalid measured contact distances: ' + name)
            distances[name] = values.astype(np.float32)
        return distances


def capture_sequence(scene, config, output_dir, *, frame_start=None, frame_end=None,
                     sync=None, progress=None, cancelled=None):
    """Capture a full inclusive Blender frame interval and restore its start pose.

    ``sync(scene)`` must update an already-enabled trusted native preview. The
    default uses the add-on preview; it never enables embedded code implicitly.
    ``progress`` receives dictionaries; ``cancelled`` may abort between frames.
    The returned manifest and numeric maps can be reused after a worker failure.
    """
    from .inspection import blender_collision_sequence as sequence
    sync = _sync if sync is None else sync
    frames = _frame_range(scene, frame_start, frame_end)
    config = json.loads(json.dumps(config, allow_nan=False))
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    capture_times, signatures, measured_frames, captured = [], [], {}, []
    with _capture_scope(scene, sync):
        scene.frame_set(frames.start)
        sync(scene)
        base = collision.extract_snapshot(scene, config)
        names = [mesh['name'] for mesh in base['meshes']]
        hands, _wrists, _watched = collision._hand_inputs(scene, config, names)
        collision._visual_settings(config)
        hashes = sequence.geometry_hashes(scene, config)
        distances = _SurfaceDistances(base, config, hands)
        scale = float(scene.unit_settings.scale_length)
        for count, frame in enumerate(frames, 1):
            if cancelled and cancelled():
                raise InterruptedError('Sequence cache capture cancelled; no playback cache published.')
            stamp = time.perf_counter()
            if frame != frames.start:
                scene.frame_set(frame)
                sync(scene)
            if float(scene.unit_settings.scale_length) != scale:
                raise ValueError('Scene units changed during sequence capture.')
            matrices = {}
            for name in names:
                matrix = np.asarray(scene.objects[name].matrix_world, dtype=float).copy()
                matrix[:3, 3] *= scale
                matrices[name] = matrix.tolist()
            signatures.append(sequence.pose_signature(scene, config, geometry_hashes=hashes))
            for name, values in distances.measure(matrices).items():
                measured_frames.setdefault(name, []).append(values)
            captured.append(dict(source_frame=float(frame - 1), matrices=matrices))
            capture_times.append(time.perf_counter() - stamp)
            if progress:
                progress(dict(stage='capture', completed=count, total=len(frames),
                              source_frame=frame - 1, seconds=capture_times[-1]))
        # The frame loop changes transforms only. Recheck meshes before binding
        # the cache so callbacks cannot silently mutate topology mid-capture.
        if sequence.geometry_hashes(scene, config) != hashes:
            raise ValueError('Original geometry changed during sequence capture.')
    manifest = dict(schema_version=1, kind='newton_collision_sequence_input',
                    snapshot=base, frames=captured, sample_limit_per_pair=64)
    arrays = dict(frames=np.asarray([frame['source_frame'] for frame in captured], dtype=np.float64),
                  signatures=np.asarray(signatures, dtype='U64'))
    for index, name in enumerate(names):
        if name in measured_frames:
            arrays['distance_' + str(index)] = np.stack(measured_frames[name])
    _write_json(output / 'sequence_input.json', manifest)
    _write_npz(output / 'capture_maps.npz', arrays)
    capture = dict(config=config, manifest=manifest, arrays=arrays,
                   output_dir=str(output), capture_seconds=time.perf_counter() - started,
                   capture_frame_seconds=capture_times,
                   scene_unit_scale_m=scale, geometry_hashes=hashes)
    _write_json(output / 'capture.json', {key: value for key, value in capture.items()
                                         if key not in {'manifest', 'arrays'}})
    return capture


def _playback_arrays(capture, report):
    """Pack bounded contact glyphs alongside dense vertex distance maps."""
    manifest, config = capture['manifest'], capture['config']
    arrays = dict(capture['arrays'])
    base = manifest['snapshot']
    frames = report['frames']
    if [frame['source_frame'] for frame in frames] != arrays['frames'].tolist():
        raise ValueError('Newton sequence frame coverage does not match the captured motion.')
    pairs = base['pairs']
    settings = collision._visual_settings(config)
    samples = [[] for _ in pairs]
    for frame in frames:
        if [[pair['mesh_a'], pair['mesh_b']] for pair in frame['pairs']] != pairs:
            raise ValueError('Newton sequence mesh pairs do not match the captured motion.')
        selections = collision._balanced_indices(
            [len(pair['points_a_world']) for pair in frame['pairs']], settings['point_limit'])
        valid_normals = [np.asarray([int(i) for i in selected
            if pair['normals_world'][int(i)] is not None], dtype=np.int64)
            for pair, selected in zip(frame['pairs'], selections)]
        normal_selections = [valid[indices] for valid, indices in zip(valid_normals,
            collision._balanced_indices([len(valid) for valid in valid_normals], settings['normal_limit']))]
        for index, (pair, selected, normal_selected) in enumerate(zip(
                frame['pairs'], selections, normal_selections)):
            a = np.asarray(pair['points_a_world'], dtype=np.float32).reshape(-1, 3)
            b = np.asarray(pair['points_b_world'], dtype=np.float32).reshape(-1, 3)
            normals = np.asarray([pair['normals_world'][int(i)] for i in normal_selected], dtype=np.float32).reshape(-1, 3)
            samples[index].append(dict(a=a[selected], b=b[selected],
                normal_origins=a[normal_selected], normals=normals,
                candidate_count=pair['contact_count']))
    for index, records in enumerate(samples):
        points = max((len(record['a']) for record in records), default=0)
        normals = max((len(record['normals']) for record in records), default=0)
        for field in ('a', 'b', 'normal_origins', 'normals'):
            capacity = normals if field in {'normal_origins', 'normals'} else points
            values = np.zeros((len(frames), capacity, 3), dtype=np.float32)
            for frame_index, record in enumerate(records):
                values[frame_index, :len(record[field])] = record[field]
            arrays[f'{field}_{index}'] = values
        arrays[f'point_counts_{index}'] = np.asarray([len(record['a']) for record in records], dtype=np.int32)
        arrays[f'normal_counts_{index}'] = np.asarray([len(record['normals']) for record in records], dtype=np.int32)
        arrays[f'candidate_counts_{index}'] = np.asarray([record['candidate_count'] for record in records], dtype=np.int64)
    metadata = dict(schema_version=1, kind='newton_collision_sequence_cache',
        config=config, mesh_names=[mesh['name'] for mesh in base['meshes']], pairs=pairs,
        units='m', scene_unit_scale_m=capture['scene_unit_scale_m'],
        geometry_hashes=capture['geometry_hashes'], backend=report['backend'],
        sequence_input_digest=report['input_digest'],
        sample_limit_per_pair=manifest['sample_limit_per_pair'],
        visualization=settings, automatic_acceptance=False,
        contact_map_semantics='Unsigned original hand vertex distance to configured target triangles; not force, penetration or a prediction.',
        limitations=report.get('limitations', []))
    arrays['metadata'] = np.asarray(json.dumps(metadata, allow_nan=False, sort_keys=True))
    return arrays


def build_cache(scene, config, output_dir, *, frame_start=None, frame_end=None,
                sync=None, progress=None, cancelled=None):
    """Explicitly build a complete sequence cache using one external worker."""
    from .inspection.collision_sequence import run_sequence_query
    started = time.perf_counter()
    capture = capture_sequence(scene, config, output_dir,
        frame_start=frame_start, frame_end=frame_end, sync=sync,
        progress=progress, cancelled=cancelled)
    if cancelled and cancelled():
        raise InterruptedError('Sequence cache cancelled before Newton; no playback cache published.')
    output = Path(capture['output_dir'])
    if progress:
        progress(dict(stage='newton', completed=0, total=len(capture['manifest']['frames'])))
    worker_started = time.perf_counter()
    report = run_sequence_query(capture['manifest'], config.get('runtime'), output / 'newton')
    worker_seconds = time.perf_counter() - worker_started
    if cancelled and cancelled():
        raise InterruptedError('Sequence cache cancelled after Newton; no playback cache published.')
    arrays = _playback_arrays(capture, report)
    path = output / 'sequence_cache.npz'
    _write_npz(path, arrays)
    result = dict(schema_version=1, status='complete', cache_path=str(path),
        frame_count=len(arrays['frames']), first_source_frame=float(arrays['frames'][0]),
        last_source_frame=float(arrays['frames'][-1]),
        capture_seconds=capture['capture_seconds'],
        capture_frame_seconds=capture['capture_frame_seconds'],
        backend_seconds=worker_seconds, total_seconds=time.perf_counter() - started,
        cache_bytes=path.stat().st_size, automatic_acceptance=False)
    _write_json(output / 'build.json', result)
    if progress:
        progress(dict(stage='complete', completed=result['frame_count'], total=result['frame_count'],
                      cache_path=str(path), seconds=result['total_seconds']))
    return result
