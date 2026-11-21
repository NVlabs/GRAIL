#!/usr/bin/env python3
"""Independent native/PKL/USD readback checks, not a physical tracking acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--native-model',type=Path)
    args=p.parse_args()
    sys.path.insert(0,str(args.upstream))
    from gear_sonic.rp2.g1_collision_geometry import (G1_BODY_DOF_NAMES,G1_HAND_DOF_NAMES_USD,
        load_collision_model,reference_trajectory,forward_kinematics)
    from gear_sonic.rp2.joint_payload_contract import validate_joint_training_payload_pair
    from gear_sonic.rp2.source_relation import DEFAULT_SOURCE_RELATION_MAPPING
    from gear_sonic.rp2.unified_cohort import resolve_unified_cohort
    pack=args.pack
    receipt=json.loads((pack/'conversion_receipt.json').read_text())
    key=receipt['source_case']
    native=np.load(pack/'source_native/final_native_UNREVIEWED.npz',allow_pickle=False)
    raw_mesh=np.load(pack/'source_native/object_mesh.npz',allow_pickle=False)
    robot=joblib.load(pack/'robot'/f'{key}.pkl')[key]
    objects=joblib.load(pack/'objects_with_masks'/f'{key}.pkl')[key]
    for file in receipt['files']:
        assert hashlib.sha256((pack/file['path']).read_bytes()).hexdigest()==file['sha256'],file['path']
    address=dict(zip(native['dof_names'],native['qposadr']))
    assert np.array_equal(robot['native_qpos'],native['qpos'])
    assert np.array_equal(robot['native_timestamps'],native['timestamps'])
    for i,name in enumerate(G1_BODY_DOF_NAMES):
        assert np.array_equal(robot['dof'][:,i],native['qpos'][:,address[name]]),name
    for i,name in enumerate(G1_HAND_DOF_NAMES_USD):
        assert np.array_equal(robot['hand_dof_pos'][:,i],native['qpos'][:,address[name]]),name
    assert np.array_equal(robot['root_trans_offset'],native['qpos'][:,:3])
    assert np.array_equal(robot['root_rot'],native['qpos'][:,[4,5,6,3]])
    assert np.array_equal(objects['root_pos'][:,0],native['obj_t'])
    assert np.array_equal(objects['native_obj_R'],native['obj_R'])
    # Independent intent readback: retain unknown/no-contact/contact instead of
    # substituting geometric contact for the source annotations.
    contact_map=json.loads((pack/'source_native/contact_map.json').read_text())
    source_intent=np.zeros((len(native['qpos']),2),np.int8)
    for column,side in enumerate(('left','right')):
        for phase in contact_map['hands'][side+'_hand']:
            first,last=phase['frames']
            source_intent[first:last,column]={'contact':1,'no_contact':-1,'unknown':0}[phase['state']]
        assert np.array_equal(objects['object_contact_mask_'+side][:,0],source_intent[:,column]==1)
    assert np.array_equal(objects['upstream_HOI_hand_intent_state'],source_intent)
    rotation=Rotation.from_quat(objects['root_quat'][:,0,[1,2,3,0]]).as_matrix()
    rotation_error=float(abs(rotation-native['obj_R']).max())
    assert rotation_error<1e-5
    active=Usd.Stage.Open(str(pack/'active_articulation_usd'/f'{key}.usd'))
    static=Usd.Stage.Open(str(pack/'static_context_usd_grail_paired'/f'{key}.usd'))
    assert UsdGeom.GetStageMetersPerUnit(active)==1 and UsdGeom.GetStageMetersPerUnit(static)==1
    mesh=UsdGeom.Mesh(active.GetPrimAtPath('/ActiveArticulation/panel/geometry'))
    vertices=np.asarray(mesh.GetPointsAttr().Get())
    faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    assert np.array_equal(faces,raw_mesh['faces'])
    scale_error=float(abs(vertices-raw_mesh['vertices']*native['object_scale']).max())
    assert scale_error<1e-6
    from validate_static_mesh import validate as validate_static
    static_validation=validate_static(native, pack/'static_context_usd_grail_paired'/f'{key}.usd')
    contract=validate_joint_training_payload_pair(robot,objects,label=key,expected_source_id='objectgroup')
    manifest=pack/'cohorts_env256/cohort_0000/manifest.json'
    resolved=resolve_unified_cohort(manifest,num_envs=256,rank=0,world_size=1,runtime_target_fps_hz=50)
    physics=json.loads((pack/'physics/object_physics_overrides.json').read_text())
    for slot in json.loads(manifest.read_text())['slots']:
        assert physics['objects'][slot['object_key']]['mass']==receipt['physics_mass_kg']
    fk_comparison=None
    if args.native_model:
        import mujoco
        mjmodel=mujoco.MjModel.from_xml_path(str(args.native_model))
        mjdata=mujoco.MjData(mjmodel)
        target_model=load_collision_model(args.upstream/'gear_sonic/data/robots/g1/g1_43dof_s3_yam_core_collision.usda')
        trajectory=reference_trajectory(robot,target_frames=len(native['qpos']),target_fps=float(native['fps']),hand_primitive_mode='linear')
        fk=forward_kinematics(target_model,trajectory)
        errors={}
        for frame in np.linspace(0,len(native['qpos'])-1,min(5,len(native['qpos'])),dtype=int):
            mjdata.qpos[:]=native['qpos'][frame]
            mujoco.mj_forward(mjmodel,mjdata)
            for name,_ in DEFAULT_SOURCE_RELATION_MAPPING:
                body_id=mujoco.mj_name2id(mjmodel,mujoco.mjtObj.mjOBJ_BODY,name)
                if body_id<0: raise ValueError('Missing body '+name)
                error=float(np.linalg.norm(mjdata.xpos[body_id]-fk[name][frame,:3,3]))
                errors[name]=max(errors.get(name,0),error)
        fk_comparison={'native_model':str(args.native_model),'maximum_body_position_error_m':max(errors.values()),
                       'by_body':errors,'frames':np.linspace(0,len(native['qpos'])-1,min(5,len(native['qpos'])),dtype=int).tolist()}
        assert max(errors.values())<1e-4,fk_comparison
    report={'status':'passed','case':key,'frames':len(native['qpos']),'fps':float(native['fps']),'contract':contract,
        'all_43_named_joint_samples_exact':True,'object_translation_exact':True,
        'original_object_rotation_matrices_exact_in_archive':True,
        'object_quaternion_max_matrix_error':rotation_error,'usd_scaled_mesh_max_error_m':scale_error,
        'usd_faces_exact':True,'native_support_world_triangles_preserved':True,
        'static_context':static_validation,
        'upstream_contact_intent_and_masks_preserved':True,
        'cohort_resolved':True,'native_vs_training_FK':fk_comparison,
        'not_evaluated':['policy rollout','physical grasp/contact feasibility','tracking success']}
    (pack/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
