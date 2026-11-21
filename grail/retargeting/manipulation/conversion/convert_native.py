#!/usr/bin/env python3
"""Convert native G1 + DEx3 robot/object motion into a training pack for the manipulation tracker.

No further retargeting, trajectory edits, time cropping, or frame resampling: the 43 named
native DOFs, object poses/scale/clock and support-scene triangles are packaged exactly. The
G1 collision model supplies geometric contact targets; the HOI contact-map phases are kept
separately as upstream intent. Writes robot/object motion-library payloads, the active
object USD (explicit mass), static support-scene USD, BPS encoding and validation receipts.
Run in the ``sonic`` environment; no GPU required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_file(record, input_root):
    path = input_root / record['study_relative_path'] if input_root else Path(record['local_path'])
    if sha(path) != record['sha256']:
        raise ValueError(f'Source hash mismatch: {path}')
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--input-root', '--raw-root', type=Path, help='Relocated study root with unchanged study-relative paths')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, help='Root the pack will be read from at training time; default output-root')
    p.add_argument('--basis', type=Path, required=True, help='Existing shared 10-point BPS basis')
    p.add_argument('--cases', default='', help='Comma-separated original indices, otherwise all')
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--skip-cohort', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    return p.parse_args()


def configure(upstream):
    os.environ['RP2_SURFACE_FAR_FIELD_M'] = 'inf'
    os.environ['RP2_KD_WORKERS'] = '1'
    sys.path[:0] = [str(upstream.resolve()), str((upstream/'tools').resolve())]


def intent_labels(contact_map, frames):
    labels = np.zeros((frames, 2), np.int8)
    for side, column in [('left', 0), ('right', 1)]:
        covered = np.zeros(frames, bool)
        for phase in contact_map.get('hands', {}).get(side+'_hand', []):
            first, end = map(int, phase['frames'])
            if first < 0 or end > frames or end <= first or covered[first:end].any():
                raise ValueError(f'Invalid/overlapping source contact interval {side}: {first}:{end}')
            covered[first:end] = True
            state = phase['state']
            labels[first:end, column] = {'contact': 1, 'no_contact': -1, 'unknown': 0}[state]
    return labels


def prepare_case(row, args, collision_model):
    import joblib
    from scipy.spatial.transform import Rotation
    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import DOF_AXIS
    from gear_sonic.rp2 import g1_collision_geometry as geom
    from gear_sonic.rp2.discrete_contact import temporal_contact_frames
    from gear_sonic.rp2.joint_payload_contract import validate_joint_training_payload_pair
    from gear_sonic.rp2.source_relation import DEFAULT_SOURCE_RELATION_MAPPING, world_vectors_to_heading_frame
    from static_mesh import native_meshes, write_static_context_usd, static_context_record, add_static_triangle_contacts
    from gear_sonic.rp2.surface_relation import build_mesh_surface_relations, closest_points_to_mesh
    from rp2_build_rigid_active_entity_pack import _write_rigid_wrapper_usd
    from rp2_build_active_articulation_training_pack import _pose_velocities

    started = time.time()
    key = row['source_case']
    if '/' in key or key in ('', '.', '..'):
        raise ValueError('Invalid literal case name')
    pack = args.output_root/'packs'/key
    runtime = (args.runtime_root or args.output_root)/'packs'/key
    receipt_path = pack/'conversion_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['source_native_sha256'] != row['native']['sha256']:
            raise ValueError('Existing pack has a different source')
        for file in receipt['files']:
            if sha(pack/file['path']) != file['sha256']:
                raise ValueError('Existing pack file changed: '+file['path'])
        return receipt
    pack.mkdir(parents=True, exist_ok=True)
    native_path = source_file(row['native'], args.input_root)
    mesh_path = source_file(row['inputs']['object_mesh.npz'], args.input_root)
    contact_path = source_file(row['inputs']['contact_map.json'], args.input_root)
    native = dict(np.load(native_path, allow_pickle=False))
    mesh = dict(np.load(mesh_path, allow_pickle=False))
    qpos, fps = native['qpos'], float(native['fps'])
    frames = len(qpos)
    assert qpos.shape == (frames, 50) and frames >= 2 and np.isfinite(fps) and fps > 0
    assert all(np.isfinite(native[k]).all() for k in ['qpos','obj_R','obj_t','object_scale','timestamps'])
    assert np.allclose(native['timestamps'], np.arange(frames)/fps, atol=1e-9)
    joint_columns = dict(zip(native['dof_names'].tolist(), native['qposadr'].tolist()))
    if len(joint_columns) != 43 or set(joint_columns.values()) != set(range(7, 50)):
        raise ValueError('Unexpected native DOF mapping')
    body = np.stack([qpos[:,joint_columns[name]] for name in geom.G1_BODY_DOF_NAMES], axis=1)
    hands = np.stack([qpos[:,joint_columns[name]] for name in geom.G1_HAND_DOF_NAMES_USD], axis=1)
    root_quat = qpos[:,[4,5,6,3]].copy()  # Native WXYZ -> robot motion-lib XYZW.
    pose = np.zeros((frames,30,3), np.float64)
    pose[:,0] = Rotation.from_quat(root_quat).as_rotvec()
    pose[:,1:] = body[...,None] * DOF_AXIS[None]
    robot = {'fps': fps, 'root_trans_offset': qpos[:,:3].copy(), 'root_rot': root_quat,
             'dof': body.copy(), 'pose_aa': pose, 'hand_dof_pos': hands.copy(),
             'native_dof43': qpos[:,native['qposadr']].copy(),
             'native_dof_names': native['dof_names'].tolist(),
             'native_qpos': qpos.copy(), 'native_timestamps': native['timestamps'].copy(),
             'hand_dof_names': list(geom.G1_HAND_DOF_NAMES_USD),
             'reference_semantics': 'objectgroup_selected_native_samples_exact_no_further_edits',
             'source_native_sha256': row['native']['sha256']}
    hand_fit = {}
    for side in ('left','right'):
        target = np.stack([qpos[:,joint_columns[n]] for n in geom.G1_HAND_PRIMITIVE_NAMES[side]],axis=1)
        opened, closed = geom.G1_HAND_PRIMITIVE_OPEN[side], geom.G1_HAND_PRIMITIVE_CLOSED[side]
        delta = closed-opened
        fraction = np.clip(((target-opened)*delta).sum(1)/(delta@delta),0,1)
        reconstruction = opened+fraction[:,None]*delta
        error = reconstruction-target
        robot['hand_action_'+side] = (fraction*2-1).astype(np.float32)
        hand_fit[side] = {'mapping':'bounded_least_squares_runtime_S3_linear_primitive',
                          'joint_names':list(geom.G1_HAND_PRIMITIVE_NAMES[side]),
                          'rms_joint_error_rad':float(np.sqrt(np.mean(error**2))),
                          'maximum_joint_error_rad':float(np.max(np.abs(error))),
                          'fraction_min':float(fraction.min()),'fraction_max':float(fraction.max()),
                          'policy_uses_source_hand_action':False,
                          'note':'actor still emits two learned primitives; arbitrary 14-DOF hand poses are not exactly representable'}
    # The upstream interpolation helper normalizes a float64 quaternion in place.
    # Give it a private copy so the source robot samples remain exact in the pack.
    trajectory = geom.reference_trajectory({**robot,'root_rot':robot['root_rot'].copy()},
        target_frames=frames,target_fps=fps,hand_primitive_mode='linear')
    fk = geom.forward_kinematics(collision_model, trajectory)
    links = np.stack([fk[name][:,:3,3] for name,_ in DEFAULT_SOURCE_RELATION_MAPPING],axis=1)
    robot['smpl_joints'] = links.copy()  # Legacy field, explicitly robot FK; no human-mesh claim.
    robot['smpl_joints_semantics'] = 'retarget_G1_FK_relation_links_not_human_SMPL'
    object_pos = native['obj_t'].copy()
    object_quat = Rotation.from_matrix(native['obj_R']).as_quat()[:,[3,0,1,2]]
    # Preserve all pose samples; bake the native anisotropic scale exactly once in geometry.
    vertices = np.asarray(mesh['vertices'] * native['object_scale'], np.float64)
    faces = np.asarray(mesh['faces'], np.int64)
    meshes, scene = native_meshes(native)
    for directory in ['robot','objects_with_masks','meta','bps','physics','active_articulation_usd',
                      'static_context_usd_grail_paired','source_native','diagnostics']:
        (pack/directory).mkdir(exist_ok=True)
    active_path = pack/'active_articulation_usd'/f'{key}.usd'
    static_path = pack/'static_context_usd_grail_paired'/f'{key}.usd'
    # Physics must arrive explicitly in the handoff; never infer it from the object name.
    mass = float(row['physics']['mass_kg'])
    if not np.isfinite(mass) or mass <= 0: raise ValueError('Invalid explicit object mass')
    mass_provenance = row['physics']['provenance']
    _write_rigid_wrapper_usd(active_path,mesh=(vertices,faces),object_mass=mass,collision_approximation='sdf')
    write_static_context_usd(static_path, meshes)
    from pxr import Usd, UsdGeom
    for path in [active_path,static_path]:
        stage=Usd.Stage.Open(str(path))
        UsdGeom.SetStageMetersPerUnit(stage,1.0)
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        stage.GetRootLayer().Save()
    temporal = temporal_contact_frames(fps)
    contact_fields, contact_audit = geom.canonical_contact_from_g1_reference(
        robot={**robot,'root_rot':robot['root_rot'].copy()},
        object_position=object_pos, object_quaternion_wxyz=object_quat,
        object_vertices_local=vertices, object_faces=faces,
        object_surface_samples_local=vertices, object_usd=active_path,
        model=collision_model, object_fps=fps,
        minimum_required_frames=temporal.minimum_required_frames,
        close_gap_frames=temporal.close_gap_frames,
        transition_neutral_frames=temporal.transition_neutral_frames,
        static_context_boxes=(), hand_primitive_mode='linear',
        candidate_body_points_per_group=64)
    contact_fields, contact_audit = add_static_triangle_contacts(
        contact_fields, contact_audit, meshes=meshes, fk=fk, model=collision_model, temporal=temporal)
    contact_fields['reference_contact_provenance'].update({
        'temporal_contact_contract':temporal.provenance(),
        'authority':'retarget_geometry_not_human_measurement',
        'native_fingers_used':True,'source_HOI_intent_is_separate':True})
    root_rotation = Rotation.from_quat(root_quat).as_matrix()
    heading = root_rotation[:,:2,0]
    heading = heading/np.linalg.norm(heading,axis=1,keepdims=True)
    surface = build_mesh_surface_relations(links,object_pos,object_quat,heading,
        vertices_local=vertices,faces=faces,surface_samples_local=vertices)
    relations = world_vectors_to_heading_frame(object_pos[:,None]-links,heading)
    lin_vel, ang_vel = _pose_velocities(object_pos,object_quat,fps)
    articulation = {'kind':'rigid','source_id':'objectgroup','source_object_name':key,
        'tracked_body_name':'panel','tracked_prim_path':'{ENV_REGEX_NS}/ActiveArticulation/panel',
        'reference_joint_names':['mechanism_joint'],'joint_valid':False,'fix_root_link':False,
        'object_mass_kg':mass,'object_mass_provenance':mass_provenance}
    objects = {'fps':fps,'root_pos':object_pos[:,None], 'root_quat':object_quat[:,None],
        'object_name':key,'object_names':[key], 'joint_training_source_id':'objectgroup',
        'active_articulation':articulation, 'active_articulation_root_pos':object_pos,
        'active_articulation_root_quat':object_quat,'active_articulation_root_lin_vel':lin_vel,
        'active_articulation_root_ang_vel':ang_vel,'active_articulation_joint_pos':np.zeros((frames,1)),
        'active_articulation_joint_vel':np.zeros((frames,1)),
        'active_articulation_joint_valid_mask':np.zeros((frames,1)),
        'active_articulation_root_dynamic_mask':np.ones((frames,1)),
        'active_entity_tracking_valid':np.ones((frames,1)),
        'rp2_source_root_pos':qpos[:,:3].copy(),'rp2_source_root_heading_xy':heading,
        'rp2_source_relation_robot_body_names':[n for n,_ in DEFAULT_SOURCE_RELATION_MAPPING],
        'rp2_source_relation_soma_joint_names':[n for _,n in DEFAULT_SOURCE_RELATION_MAPPING],
        'rp2_source_relation_vector':relations,'rp2_source_relation_mask':np.ones((frames,14)),
        'rp2_source_surface_relation_vector':surface['relation_vector'],
        'rp2_source_surface_relation_mask':surface['relation_mask'],
        'rp2_source_surface_point_local':surface['surface_point_local'],
        'rp2_source_surface_signed_distance':surface['signed_distance'],
        'rp2_source_surface_penetration_mask':surface['source_penetration_mask'],
        'rp2_source_surface_relation_semantics':surface['semantics'],
        'rp2_source_surface_relation_provenance':{
            'human_reference':'not_used; retarget G1 FK target replaces source human landmark supervision',
            'object_reference':'final_native optimized obj_R/obj_t and original mesh scaled once',
            'surface_query':'nearest_triangle_with_oriented_surface_sign',
            'authority':'retarget_geometry','g1_fk_used':True,
            'soma_joint_names_role':'ABI semantic slot names only; coordinates are G1 FK'},
        'rp2_source_scene_scale':1.0, 'native_object_scale':native['object_scale'].copy(),
        'native_obj_R':native['obj_R'].copy(), 'native_obj_t':native['obj_t'].copy(),
        **contact_fields}
    contact_map = json.loads(contact_path.read_text())
    raw_intent = intent_labels(contact_map,frames)
    objects['upstream_HOI_hand_intent_state'] = raw_intent
    objects['upstream_HOI_hand_intent_provenance'] = {'sha256':sha(contact_path),'unknown_value':0,
        'note':'original hand phases and full regions retained verbatim; not substituted for geometry distances'}
    for side, col, body_index in [('left',0,10),('right',1,13)]:
        objects['object_contact_mask_'+side] = (raw_intent[:,col] == 1).astype(np.float32)[:,None]
        objects['object_contact_center_'+side] = surface['surface_point_local'][:,body_index].copy()
    objects['object_motion_mask'] = np.ones((frames,1),np.float32)
    objects['object_interaction_mask'] = np.ones((frames,1),np.float32)
    abi = validate_joint_training_payload_pair(robot,objects,label=key,expected_source_id='objectgroup')
    for sub, payload in [('robot',robot),('objects_with_masks',objects)]:
        joblib.dump({key:payload},pack/sub/f'{key}.pkl',compress=0)
    meta = {'fps':fps,'num_frames':frames,'frame_count':frames,
            'source_id':'objectgroup','native_source_sha256':sha(native_path),
            'static_context':static_context_record(meshes), 'table_needed':False,
            'object_geometry_scale':1.0,'object_geometry_scale_mode':'baked_native_mesh_once',
            'retarget_geometry_unreviewed':True}
    joblib.dump(meta,pack/'meta'/f'{key}.pkl',compress=0)
    basis = np.load(args.basis,allow_pickle=False)
    if basis.shape != (10,3): raise ValueError(f'Unexpected BPS basis: {basis.shape}')
    bps = closest_points_to_mesh(basis,vertices,faces,compute_signed_distance=False)[0]
    np.save(pack/'bps'/'_basis.npy',basis)
    np.save(pack/'bps'/f'{key}.npy',bps.astype(np.float32))
    shutil.copy2(native_path,pack/'source_native'/'final_native_UNREVIEWED.npz')
    shutil.copy2(mesh_path,pack/'source_native'/'object_mesh.npz')
    shutil.copy2(contact_path,pack/'source_native'/'contact_map.json')
    if 'hand_sequence' in row:
        shutil.copy2(source_file(row['hand_sequence'],args.input_root),pack/'source_native'/'hand_sequence_UNREVIEWED.npz')
    for name in ['contact_plan.json','camera.json','case.json']:
        if name in row['inputs']:
            shutil.copy2(source_file(row['inputs'][name],args.input_root),pack/'source_native'/name)
    write_json(pack/'diagnostics'/'hand_primitive_fit.json',hand_fit)
    write_json(pack/'diagnostics'/'native_scene.json',scene)
    write_json(pack/'diagnostics'/'contact_geometry.json',contact_audit)
    write_json(pack/'active_entity_manifest.json',{'schema':'objectgroup_native_pack_mesh_v2_candidate',
        'promoted_revision':'objectgroup_selected_native_technical_conversion',
        'record_count':1,'records':[{'motion_key':key,'active_object':key,'fps':fps,
        'num_frames':frames,'joint_valid':False,'object_mass_kg':mass}]})
    write_json(pack/'sources.json',{'contract':'unified_no_vof_sources_v1','target_fps_hz':50,
        'sources':{'objectgroup':{'pack':str(runtime),'native_fps_hz':fps}}})
    write_json(pack/'physics'/'object_physics_overrides.json',{'default':{},'objects':{'objectgroup/'+key:{'mass':mass}},
        'rules':[],'provenance':mass_provenance, 'friction_restitution':'inherited unchanged from pinned training environment'})
    # Require exact body/native samples, exact object matrices, and every support triangle.
    restored_body = np.stack([robot['dof'][:,i] for i in range(29)],axis=1)
    assert np.array_equal(restored_body,body)
    assert np.array_equal(objects['root_pos'][:,0],native['obj_t'])
    rotation_error = float(np.max(np.abs(Rotation.from_quat(object_quat[:,[1,2,3,0]]).as_matrix()-native['obj_R'])))
    # SOMA writes near-rotation matrices at float32 solver precision. Quaternion
    # encoding orthogonalizes only that numerical residue; original matrices archived.
    assert rotation_error < 1e-5
    check_stage = Usd.Stage.Open(str(active_path))
    authored = np.asarray(UsdGeom.Mesh(check_stage.GetPrimAtPath('/ActiveArticulation/panel/geometry')).GetPointsAttr().Get(),dtype=np.float64)
    scale_error = float(np.max(np.abs(authored-vertices)))
    assert scale_error < 1e-6
    from validate_static_mesh import validate as validate_static
    static_validation=validate_static(native, static_path)
    write_json(pack/'diagnostics/static_triangle_readback.json',static_validation)
    receipt = {'case_id':row['case_id'],'source_case':key,'status':'CONVERTED_TECHNICAL_UNREVIEWED',
        'source_native_sha256':sha(native_path),'pack':str(pack),'runtime_pack':str(runtime),
        'frames':frames,'fps':fps,'abi':abi,'native_joint_count':43,'body_dofs':29,
        'object_rotation_max_abs_roundtrip_error':rotation_error,'object_usd_float32_max_abs_error_m':scale_error,
        'native_body_pose_exact':True,'native_object_translation_exact':True,'native_hand_archive_exact':True,
        'support_geometry':'exact native world triangle meshes; no bbox/slab inference or simplification',
        'static_mesh_count':len(meshes), 'static_triangle_count':sum(len(m.faces) for m in meshes),
        'gpu_static_context_verified':False,'hand_primitive_fit':hand_fit,
        'supervision_authority':'retarget G1 collision geometry and FK; source HOI intent retained separately',
        'physics_mass_kg':mass,'physics_mass_provenance':mass_provenance,
        'physics_overrides':str(runtime/'physics'/'object_physics_overrides.json'),
        'duration_seconds':time.time()-started,
        'quality_accepted':False,'files':[]}
    for path in sorted(pack.rglob('*')):
        if path.is_file(): receipt['files'].append({'path':str(path.relative_to(pack)),'sha256':sha(path),'bytes':path.stat().st_size})
    write_json(receipt_path,receipt)
    return receipt


def build_cohort(pack, args):
    from build_unified_no_vof import build_plan, materialize
    from gear_sonic.rp2.unified_cohort import resolve_unified_cohort
    # Materialize in the actual staging location, then relocate every path once.
    sources = json.loads((pack/'sources.json').read_text())
    actual = json.loads(json.dumps(sources))
    actual['sources']['objectgroup']['pack'] = str(pack)
    staging = pack/'sources.staging.json'
    write_json(staging,actual)
    plan = build_plan(staging,num_envs=args.num_envs,max_motions=1,max_group_motions=1,boundary_support='off')
    manifest = materialize(plan,0,pack/f'cohorts_env{args.num_envs}',world_size=1)
    resolved = resolve_unified_cohort(manifest,num_envs=args.num_envs,world_size=1,rank=0)
    write_json(pack/'cohort_validation.json',{'status':'passed','manifest':str(manifest),
        'resolver_keys':sorted(resolved),'num_envs':args.num_envs,'boundary_support':'off'})
    staging.unlink()
    runtime = (args.runtime_root or args.output_root)/'packs'/pack.name
    if runtime != pack:
        old,new = str(pack),str(runtime)
        for path in (pack/f'cohorts_env{args.num_envs}').rglob('*'):
            if path.is_symlink():
                target=os.readlink(path)
                if target.startswith(old): path.unlink(); path.symlink_to(new+target[len(old):])
            elif path.suffix == '.json':
                path.write_text(path.read_text().replace(old,new))
    return str(runtime/f'cohorts_env{args.num_envs}'/'cohort_0000'/'manifest.json')


_collision_model = None


def convert_job(work):
    global _collision_model
    row,args=work
    configure(args.upstream)
    from gear_sonic.rp2.g1_collision_geometry import load_collision_model
    if _collision_model is None:
        _collision_model=load_collision_model(args.upstream/'gear_sonic/data/robots/g1/g1_43dof_s3_yam_core_collision.usda')
    print('START',row['source_case'],flush=True)
    receipt=prepare_case(row,args,_collision_model)
    if not args.skip_cohort:
        receipt['roster_manifest']=build_cohort(args.output_root/'packs'/row['source_case'],args)
    print('PASS',row['source_case'],round(receipt['duration_seconds'],2),flush=True)
    return receipt


def main():
    args=parse_args()
    args.output_root=args.output_root.resolve()
    rows=json.loads(args.manifest.read_text())
    selected={int(i) for i in args.cases.split(',') if i.strip()}
    work=[(row,args) for row in rows if not selected or row['original_index'] in selected]
    if args.workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            reports=list(pool.map(convert_job,work))
    else: reports=[convert_job(item) for item in work]
    write_json(args.output_root/('conversion_catalog_'+('-'.join(map(str,sorted(selected))) if selected else 'all')+'.json'),reports)


if __name__=='__main__': main()
