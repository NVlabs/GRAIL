#!/usr/bin/env python3
"""Group converted motion packs by object mesh into frozen training cohorts (2048-slot rosters).

For every object mesh this writes a training roster (``groups/<mesh>/cohort/manifest.json``),
one single-environment evaluation roster per motion, the per-asset ``physics.json`` and a
case binding. Slots are allocated per immutable asset group with ``object_slot_policy=
motion_count``. Inputs are verified by SHA-256; outputs are never overwritten.
No network or submission operations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

import os

HERE = Path(__file__).resolve().parent
SONIC_ROOT = Path(os.environ.get('CONTACTFIRST_SONIC_ROOT', HERE.parents[3] / 'imports' / 'SONIC-manipulation'))
COMMIT = 'fb8563f9a5b5889d50bccba4dc07a607f2e4a71c'
FILE_KEYS = ('mesh', 'native', 'robot', 'objects', 'bps', 'basis', 'active_usd',
             'context_usd', 'active_surface', 'physics', 'validation')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def pinned_modules():
    """Slot allocator and cohort resolver from the vendored SONIC manipulation tree."""
    require((SONIC_ROOT / 'gear_sonic/rp2/unified_cohort.py').is_file(), 'Vendored SONIC tree not found: ' + str(SONIC_ROOT))
    sys.path[:0] = [str(SONIC_ROOT)]
    from tools.build_unified_no_vof import allocate_cohort_slots
    from gear_sonic.rp2.unified_cohort import resolve_unified_cohort
    return allocate_cohort_slots, resolve_unified_cohort


def payload(path, key):
    # Trusted local pack only: joblib/pickle must never be used on arbitrary uploads.
    data = joblib.load(path)
    require(isinstance(data, dict) and set(data) == {key}, 'Expected exactly one pack motion: ' + str(path))
    return data[key]


def usd_check(path, mesh=None, scale=None):
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    require(not assets and not unresolved and len(layers) == 1,
            'USD must be self-contained (flatten/package dependencies first): ' + str(path))
    stage = Usd.Stage.Open(str(path))
    require(stage is not None, 'Cannot open USD: ' + str(path))
    joint_signature = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            kind = prim.GetTypeName()
            if kind == 'PhysicsFixedJoint':
                joint_signature.append([str(prim.GetPath()), kind])
            elif kind in ('PhysicsRevoluteJoint', 'PhysicsPrismaticJoint'):
                low = prim.GetAttribute('physics:lowerLimit').Get()
                high = prim.GetAttribute('physics:upperLimit').Get()
                require(low is not None and high is not None and math.isfinite(low)
                        and math.isfinite(high) and abs(low) <= 0.001 and abs(high) <= 0.001,
                        'Only rigid/locked active joints supported: ' + str(path))
                joint_signature.append([str(prim.GetPath()), kind, float(low), float(high)])
            else:
                raise ValueError('Unsupported articulated asset: ' + kind)
    if mesh is not None:
        p = stage.GetPrimAtPath('/ActiveArticulation/panel/geometry')
        require(bool(p) and p.IsA(UsdGeom.Mesh), 'Expected pinned active panel mesh')
        usd_mesh = UsdGeom.Mesh(p)
        points = np.asarray(usd_mesh.GetPointsAttr().Get())
        wanted = np.asarray(mesh['vertices']) * scale
        require(points.shape == wanted.shape and np.allclose(points, wanted, rtol=1e-6, atol=1e-6),
                'Active USD geometry differs from mesh bytes and declared native scale')
        faces = np.asarray(mesh['faces'])
        require(np.array_equal(np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get()), faces.reshape(-1)),
                'Active USD topology differs from source mesh')
        require(np.all(np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get()) == 3), 'Expected triangulated mesh')
        require(np.allclose(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(p)), np.eye(4)),
                'Unexpected active mesh transform; package exact scaled local geometry first')
    return joint_signature


def validate_motion(item):
    require(isinstance(item.get('motion_key'), str) and item['motion_key'], 'Missing unique motion_key')
    paths = {}
    for key in FILE_KEYS:
        r = item['files'][key]
        p = Path(r['path'])
        require(p.is_absolute() and p.is_file(), f'{item["motion_key"]}: missing absolute {key} file {p}')
        require(digest(p) == r['sha256'], f'{item["motion_key"]}: {key} SHA mismatch')
        paths[key] = p.resolve()
    identity = item.get('mesh_identity', item['files']['mesh'])
    require(Path(identity['path']).is_absolute() and Path(identity['path']).is_file()
            and digest(identity['path']) == identity['sha256'],
            'Grouping mesh identity bytes/SHA mismatch')
    require(item.get('mesh_identity_format', 'geometry_npz_exact_bytes')
            in ('original_file', 'geometry_npz_exact_bytes'), 'Unsupported mesh identity format')
    require(item.get('mesh_identity_format') != 'original_file' or 'mesh_identity' in item,
            'original_file requires an explicit mesh_identity reference')
    validation = read(paths['validation'])
    require(validation.get('status') == 'passed', 'Pack independent validation is not passed')
    mesh = np.load(paths['mesh'], allow_pickle=False)
    require(set(mesh.files) == {'vertices', 'faces'}, 'Require native vertices/faces mesh NPZ')
    require(mesh['vertices'].ndim == 2 and mesh['vertices'].shape[1] == 3
            and np.isfinite(mesh['vertices']).all(), 'Invalid mesh vertices')
    native = np.load(paths['native'], allow_pickle=False)
    scale = np.asarray(native['object_scale'], dtype=float)
    require(scale.shape == (3,) and np.isfinite(scale).all() and (scale > 0).all(), 'Invalid object scale')
    require(np.array_equal(scale, np.asarray(item['scale_xyz'])), 'Declared scale differs from native')
    # Native metadata may name the original OBJ; the verified mesh NPZ is a different byte representation.
    # Geometry equivalence is checked directly against the active USD below, not by comparing unlike file hashes.
    if item.get('mesh_identity_format') == 'original_file' and 'object_mesh_sha256' in native:
        require(str(native['object_mesh_sha256'].item()) == identity['sha256'], 'Original mesh identity differs from native')
    robot = payload(paths['robot'], item['pack_motion_key'])
    objects = payload(paths['objects'], item['pack_motion_key'])
    frames, fps = int(item['source_reference']['frames']), float(item['source_reference']['fps'])
    require(frames >= 2 and math.isfinite(fps) and fps > 0, 'Invalid clock')
    require(float(robot['fps']) == float(objects['fps']) == float(native['fps']) == fps, 'Payload FPS mismatch')
    require(np.asarray(robot['dof']).shape == (frames, 29), 'Expected G1 29 DOF payload')
    require(np.asarray(robot['pose_aa']).shape == (frames, 30, 3), 'Expected G1 pose payload')
    require(np.asarray(objects['active_articulation_root_pos']).shape == (frames, 3), 'Object clock mismatch')
    require(len(native['qpos']) == frames, 'Native clock mismatch')
    require(not np.any(objects['active_articulation_joint_valid_mask']), 'Articulated motion not supported by rigid contract')
    require(np.array_equal(np.asarray(objects['native_object_scale']), scale), 'Object payload scale mismatch')
    require(np.allclose(np.asarray(objects['active_articulation_joint_pos']), 0), 'Nonzero active joint motion')
    for name in ('dof', 'pose_aa', 'root_trans_offset', 'root_rot'):
        require(np.isfinite(robot[name]).all(), 'Nonfinite robot payload ' + name)
    require(np.isfinite(objects['active_articulation_root_pos']).all(), 'Nonfinite object translation')
    basis = np.load(paths['basis'], allow_pickle=False)
    bps = np.load(paths['bps'], allow_pickle=False)
    surface = np.load(paths['active_surface'], allow_pickle=False)
    require(basis.shape == (10, 3) and np.isfinite(basis).all(), 'Invalid BPS basis')
    require(bps.shape == (10,) and np.isfinite(bps).all(), 'Invalid BPS vector')
    require(surface.shape == (512, 3) and np.isfinite(surface).all(), 'Invalid active surface')
    joints = usd_check(paths['active_usd'], mesh, scale)
    usd_check(paths['context_usd'])
    physics_doc = read(paths['physics'])
    require(not physics_doc.get('rules'), 'Resolve physics rules into per-motion overrides before handoff')
    physics = {**physics_doc.get('default', {}), **physics_doc.get('objects', {}).get(item['physics_object_key'], {})}
    require(item['physics_object_key'] in physics_doc.get('objects', {}), 'Missing explicit physics object key')
    signature = hashlib.sha256(canonical({
        'active': item['files']['active_usd']['sha256'],
        'context': item['files']['context_usd']['sha256'],
        'bps': item['files']['bps']['sha256'], 'basis': item['files']['basis']['sha256'],
        'surface': item['files']['active_surface']['sha256'],
        'scale': scale.tolist(), 'joints': joints, 'physics': physics,
    }).encode()).hexdigest()
    return {**item, 'paths': paths, 'object_mesh_sha256':identity['sha256'],
            'mesh_identity':identity, 'mesh_identity_format':item.get('mesh_identity_format','geometry_npz_exact_bytes'), 'bps_vector': bps.tolist(), 'physics_values': physics,
            'asset_signature': 'asset_' + signature, 'joint_signature': joints,
            'motion_id': 'm_' + hashlib.sha256(item['motion_key'].encode()).hexdigest()[:24]}


def build(handoff_path, output, num_envs, runtime_root=None):
    allocate, resolve = pinned_modules()
    handoff_path, output = Path(handoff_path).resolve(), Path(output).resolve()
    handoff = read(handoff_path)
    require(handoff.get('schema') == 'contactfirst.objectgroup.handoff.v1', 'Wrong handoff schema')
    require(isinstance(handoff.get('dataset_id'), str) and handoff['dataset_id'], 'Missing dataset_id')
    require(handoff.get('purpose') in ('preparation', 'smoke', 'production'), 'Invalid purpose')
    require(isinstance(handoff.get('handoff_complete'), bool), 'handoff_complete must be boolean')
    require(num_envs > 0, 'num_envs must be positive')
    require(not output.exists(), 'Output already exists; never overwrite a frozen preparation')
    motions = handoff['motions']
    keys = [m['motion_key'] for m in motions]
    require(keys and len(keys) == len(set(keys)), 'Missing/duplicate motion identities')
    require(set(keys) == set(handoff['expected_motion_keys']) and len(keys) == len(handoff['expected_motion_keys']),
            'Incomplete handoff or silent motion omission')
    validated = [validate_motion(m) for m in motions]
    require(len({m['motion_id'] for m in validated}) == len(validated), 'Alias collision')
    require(len({m['files']['basis']['sha256'] for m in validated}) == 1, 'BPS basis differs across motions')
    grouped = defaultdict(list)
    for m in validated:
        grouped[m['object_mesh_sha256']].append(m)
    require(len({sha[:16] for sha in grouped}) == len(grouped), 'Mesh group ID prefix collision')
    # Validate every input before any output mutation.
    require(all(len({m['asset_signature'] for m in ms}) <= num_envs for ms in grouped.values()),
            'Environment budget cannot cover every immutable asset group')
    runtime = Path(runtime_root) if runtime_root else output
    require(runtime.is_absolute(), 'runtime_root must be absolute')
    output.mkdir(parents=True)
    copied = {}
    def runtime_path(p):
        return str(runtime / Path(p).relative_to(output))
    def copy_blob(p, suffix):
        sha = digest(p)
        dest = output / 'blobs' / (sha + suffix)
        if dest not in copied:
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(p, dest)
            require(digest(dest) == sha, 'Copied blob SHA mismatch')
            copied[dest] = sha
        return dest
    def link_blob(p, link, suffix):
        dest = copy_blob(p, suffix)
        link.parent.mkdir(parents=True, exist_ok=True)
        import os
        link.symlink_to(os.path.relpath(dest, link.parent))
        return dest
    def file_ref(p):
        return {'path': runtime_path(p), 'relative_path': str(p.relative_to(output)),
                'sha256': digest(p), 'bytes': p.stat().st_size}
    def manifest_write(path, records, groups, count):
        order, allocation = allocate(groups, count, policy='motion_count')
        slots = []
        for env_id, group_idx in enumerate(order):
            g = groups[group_idx]
            aliases = [r['alias'] for r in g['records']]
            slots.append({k: g[k] for k in ('source','asset_signature','object_key','active_articulation_usd','static_context_usd','active_surface_path')} |
                         {'local_env_id': env_id, 'compatible_motion_aliases': aliases, 'representative_alias': aliases[0]})
        doc = {'contract':'unified_no_vof_cohort_v1','num_envs':count,'num_envs_per_rank':count,'world_size':1,
               'target_fps_hz':50,'cohort_id':0,'cohort_count':1,'catalog_total_unique_motions':len(records),
               'catalog_source_counts':{'objectgroup':len(records)},'sampling_plan':{'sources':{'objectgroup':{}}},
               'object_slot_policy':'motion_count','track_source_stats':True,'slot_allocation':allocation,
               'records':records,'slots':slots}
        save(path, doc)
        view = resolve(path, num_envs=count, rank=0, world_size=1, runtime_target_fps_hz=50)
        require(set(view['motion_aliases']) == {r['alias'] for r in records}, 'Pinned resolver coverage mismatch')
        return doc
    catalog_groups = []
    for mesh_sha, members in sorted(grouped.items()):
        members.sort(key=lambda m:m['motion_key'])
        group_id = 'mesh_' + mesh_sha[:16]
        directory = output / 'groups' / group_id
        train_dir = directory / 'cohort'
        records, assets, physics_objects, motion_receipts = [], {}, {}, []
        for m in members:
            alias = m['motion_id']; sig = m['asset_signature']
            object_key = 'objectgroup/' + group_id + '/' + sig
            active = copy_blob(m['paths']['active_usd'], '.usd')
            context = copy_blob(m['paths']['context_usd'], '.usd')
            surface = copy_blob(m['paths']['active_surface'], '.npy')
            bps = copy_blob(m['paths']['bps'], '.npy')
            # Keep each exact asset/scale/physics context in its own signature. Byte-identical assets may share storage.
            record = {'alias':alias,'source':'objectgroup','source_motion_key':m['motion_key'],
                      'pack_motion_key':m['pack_motion_key'],'source_sample_fps_hz':m['source_reference']['fps'],
                      'promoted_revision':handoff['dataset_id'],'pack':runtime_path(directory),
                      'active_articulation_usd':runtime_path(active),'static_context_usd':runtime_path(context),
                      'active_object':group_id,'object_key':object_key,'bps_path':runtime_path(bps),
                      'bps_vector':m['bps_vector'],'asset_signature':sig,'object_mesh_sha256':mesh_sha,
                      'native_scale_xyz':m['scale_xyz'],'mesh_identity_format':m['mesh_identity_format'],'active_usd_subdir':'active_articulation_usd'}
            records.append(record)
            if sig not in assets:
                assets[sig] = {k:record[k] for k in ('source','asset_signature','object_key','active_articulation_usd','static_context_usd')}
                assets[sig].update(active_surface_path=runtime_path(surface), records=[])
            assets[sig]['records'].append(record)
            physics_objects[object_key] = m['physics_values']
            eval_dir = directory / 'eval' / alias
            for base in (train_dir, eval_dir):
                for input_key, sub, extension in [('robot','robot','.pkl'),('objects','objects_with_masks','.pkl'),
                       ('bps','bps','.npy'),('active_usd','object_usd','.usd'),('context_usd','static_context_usd_grail_paired','.usd')]:
                    link_blob(m['paths'][input_key], base/sub/(alias+extension), extension)
                if not (base/'bps/_basis.npy').exists():
                    link_blob(m['paths']['basis'], base/'bps/_basis.npy', '.npy')
                (base/'meta').mkdir(exist_ok=True); (base/'object_usd_grail_paired').mkdir(exist_ok=True)
            single_group = {**assets[sig], 'records':[record]}
            manifest_write(eval_dir/'manifest.json', [record], [single_group], 1)
            motion_receipts.append({'motion_id':alias,'source_motion_key':m['motion_key'],
                'pack_motion_key':m['pack_motion_key'],'source_reference':m['source_reference'],
                'object_mesh_sha256':mesh_sha,'scale_xyz':m['scale_xyz'],'asset_signature':sig,
                'single_env_roster_path':runtime_path(eval_dir/'manifest.json'),
                'single_env_roster_sha256':digest(eval_dir/'manifest.json'),
                'mesh_identity':m['mesh_identity'],'mesh_identity_format':m['mesh_identity_format'],
                'source_files':{k:m['files'][k] for k in FILE_KEYS}})
        physics_file = directory/'physics.json'
        save(physics_file, {'default':{},'objects':physics_objects,'rules':[],
                           'provenance':'Exact input overrides, mapped to immutable per-asset object keys; no physics values changed'})
        groups = [assets[k] for k in sorted(assets)]
        doc = manifest_write(train_dir/'manifest.json', records, groups, num_envs)
        for motion in motion_receipts:
            motion.update(physics_path=runtime_path(physics_file), physics_sha256=digest(physics_file))
        group_files = [p for p in directory.rglob('*') if p.is_file()]
        # Copied runtime blobs and aliases are included for worker hash verification; source provenance stays separate.
        files = [file_ref(p) for p in sorted(set(group_files) | set(copied))]
        receipt = {'schema':'contactfirst.objectgroup.case_receipt.v1','case_id':group_id,'source_case':group_id,
                   'object_mesh_sha256':mesh_sha,'motion_count':len(members),'asset_group_count':len(groups),
                   'motions':motion_receipts,'source_handoff':ref(handoff_path),'code_commit':COMMIT,
                   'training_pack':{'validated':True,'validation_scope':'CPU payload/geometry/hash/pinned-resolver; GPU smoke separate',
                       'pack_path':runtime_path(directory),'files':files,'roster_path':runtime_path(train_dir/'manifest.json'),
                       'roster_sha256':digest(train_dir/'manifest.json'),'physics_path':runtime_path(physics_file),
                       'physics_sha256':digest(physics_file),'num_envs':num_envs,'source_namespace':'objectgroup'},
                   'slot_allocation':doc['slot_allocation'],'quality_accepted':False}
        binding_path = directory/'case_binding.json'; save(binding_path,receipt)
        counts = Counter(s['asset_signature'] for s in doc['slots'])
        expected = {r['alias']:counts[r['asset_signature']]/len(assets[r['asset_signature']]['records']) for r in records}
        catalog_groups.append({'group_id':group_id,'object_mesh_sha256':mesh_sha,'motion_count':len(members),
            'asset_group_count':len(groups),'num_envs':num_envs,'motion_ids':[m['motion_id'] for m in members],
            'binding_path':runtime_path(binding_path),'local_binding_path':str(binding_path),'binding_sha256':digest(binding_path),
            'roster_path':runtime_path(train_dir/'manifest.json'),'roster_sha256':digest(train_dir/'manifest.json'),
            'motion_env_exposure_if_uniform_within_asset_group':expected,
            'motion_env_exposure_note':'Fixed slots per immutable asset group; reset chooses candidates from current sampling probabilities. Values are uniform expectations, not guaranteed per-motion concurrent counts.'})
    catalog = {'schema':'contactfirst.objectgroup.catalog.v1','dataset_id':handoff['dataset_id'],
               'source_handoff':ref(handoff_path),'upstream_sonic_commit':COMMIT,
               'group_count':len(catalog_groups),'motion_count':len(validated),'expected_motion_count':len(keys),
               'handoff_complete':handoff.get('handoff_complete') is True,
               'purpose':handoff['purpose'],'production_ready':handoff.get('handoff_complete') is True and handoff['purpose']=='production',
               'submitted':False,'runtime_root':str(runtime),'local_root':str(output),'groups':catalog_groups,
               'coverage':{'missing':[],'duplicates':[],'aliases':[m['motion_id'] for m in validated]}}
    save(output/'catalog.json',catalog)
    save(output/'build_receipt.json', {'schema':'contactfirst.objectgroup.build_receipt.v1','status':'passed',
         'catalog':file_ref(output/'catalog.json'),'files':[file_ref(p) for p in sorted(output.rglob('*')) if p.is_file()],
         'no_training_submitted':True})
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--runtime-root')
    args = parser.parse_args()
    doc = build(args.handoff, args.output, args.num_envs, args.runtime_root)
    print(json.dumps({k: doc[k] for k in ['group_count', 'motion_count', 'production_ready', 'submitted']}))


if __name__=='__main__':main()
