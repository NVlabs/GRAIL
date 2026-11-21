"""Explicit prior-global state input; never a replacement for source objectives.

Only Newton's one-free-root plus scalar-hinge layout is supported. A native
export qpos array is deliberately not accepted as a solver checkpoint.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
            or not isinstance(value['path'], str) or not value['path']
            or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in value['sha256'])):
        raise ValueError('prior_global_seed requires path and lowercase SHA256')
    return dict(value)


def reference_hash(rotations, positions, scale):
    h = hashlib.sha256(b'soma_prior_global_reference.v1')
    for value in [rotations, positions, np.broadcast_to(scale, (3,))]:
        a = np.ascontiguousarray(value, dtype='<f8')
        if not np.isfinite(a).all(): raise ValueError('Nonfinite original object reference')
        h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load(config, *, base_path='.', model_sha256=None, source_sha256=None,
         frame_count=None, fps=None, reference_rotations=None, reference_positions=None,
         object_scale=None, object_config=None):
    config = validate_config(config)
    path = (Path(base_path)/config['path']).resolve(strict=True)
    if sha(path) != config['sha256']: raise ValueError('Prior-global descriptor hash changed')
    spec = json.loads(path.read_text())
    required = {'schema', 'model_xml_sha256', 'source_bvh_sha256', 'frame_count', 'fps',
                'object_reference_sha256', 'coordinate_convention', 'files'}
    if (not isinstance(spec, dict) or set(spec)-required-{'provenance'} or required-set(spec)
            or spec['schema'] != 'soma_prior_global_seed.v1'
            or spec['coordinate_convention'] != 'newton_free_xyz_xyzw_then_scalar_hinges.v1'):
        raise ValueError('Invalid prior-global seed descriptor')
    if (type(spec['frame_count']) is not int or spec['frame_count'] < 4
            or isinstance(spec['fps'], bool) or not isinstance(spec['fps'], (int,float))
            or not np.isfinite(spec['fps']) or spec['fps'] <= 0):
        raise ValueError('Invalid prior-global original clock')
    for name, expected in [('model_xml_sha256',model_sha256), ('source_bvh_sha256',source_sha256),
                           ('frame_count',frame_count), ('fps',fps)]:
        if expected is not None and spec[name] != expected:
            raise ValueError('Prior-global original binding differs: '+name)
    if set(spec['files']) != {'run', 'checkpoint', 'linearization', 'synchronous_report', 'object_reference'}:
        raise ValueError('Prior-global seed requires solver evidence and original object reference files')
    files = {'descriptor': dict(path=str(path), sha256=config['sha256'])}
    for key, value in spec['files'].items():
        value = validate_config(value); p = (path.parent/value['path']).resolve(strict=True)
        if sha(p) != value['sha256']: raise ValueError('Prior-global input hash changed: '+key)
        files[key] = dict(path=str(p), sha256=value['sha256'])
    run = json.loads(Path(files['run']['path']).read_text())
    sync = json.loads(Path(files['synchronous_report']['path']).read_text())
    execution = run.get('synchronous_execution', {})
    if (run.get('status') != 'CONVERTED_UNREVIEWED' or execution.get('passed') is not True
            or run['model_xml']['sha256'] != spec['model_xml_sha256']
            or run['source']['sha256'] != spec['source_bvh_sha256']
            or run['source_info']['frame_count'] != spec['frame_count']
            or run['source_info']['fps'] != spec['fps']
            or sync.get('method') != 'synchronous_sparse'
            or sync.get('frame_count') != spec['frame_count'] or sync.get('fps') != spec['fps']
            or execution.get('checkpoint_sha256') != files['checkpoint']['sha256']
            or execution.get('saved_linearization_sha256') != files['linearization']['sha256']
            or execution.get('solver_report_sha256') != files['synchronous_report']['sha256']
            or run.get('contact_inputs',{}).get('object_trajectory',{}).get('sha256') != files['object_reference']['sha256']
            or sync.get('checkpoint_sha256') != files['checkpoint']['sha256']
            or sync.get('linearization_evidence',{}).get('sha256') != files['linearization']['sha256']):
        raise ValueError('Prior-global actual run/evidence lineage differs')
    with np.load(files['object_reference']['path'], allow_pickle=False) as z:
        if reference_hash(z['obj_R'], z['obj_t'], z['object_scale']) != spec['object_reference_sha256']:
            raise ValueError('Prior-global bound original object reference differs')
    if object_config is not None and run.get('object_optimization') != object_config:
        raise ValueError('Prior-global object policy/config changed')
    if reference_rotations is not None:
        if reference_hash(reference_rotations, reference_positions, object_scale) != spec['object_reference_sha256']:
            raise ValueError('Prior-global original object reference changed')
    with np.load(files['checkpoint']['path'], allow_pickle=False) as z:
        cp = {k:z[k].copy() for k in z.files}
    with np.load(files['linearization']['path'], allow_pickle=False) as z:
        lin = {k:z[k].copy() for k in ['scalar_coordinate_indices','scalar_dof_indices',
             'robot_dofs_per_frame','object_frame_to_block','frame_count']}
    q = cp['refined_joint_q']; count = spec['frame_count']; nc = q.shape[1] if q.ndim==2 else -1
    if (q.shape != (count,nc) or nc < 8 or q.dtype != np.float32 or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1.,atol=2e-6,rtol=0)
            or float(cp['fps']) != spec['fps']
            or int(cp['completed_global_iterations']) != sync['completed_global_iterations']
            or int(cp['completed_global_iterations']) < 1
            or not np.array_equal(cp['coordinate_indices'],np.arange(7,nc))
            or not np.array_equal(lin['scalar_coordinate_indices'],cp['coordinate_indices'])
            or not np.array_equal(lin['scalar_dof_indices'],np.arange(6,nc-1))
            or int(lin['robot_dofs_per_frame']) != nc-1 or int(lin['frame_count']) != count
            or len(cp['joint_names']) != nc-7 or len(set(cp['joint_names'].tolist())) != nc-7):
        raise ValueError('Prior-global Newton free-root/hinge layout or finite state differs')
    state = {'q':q.copy()}
    if 'refined_obj_R' not in cp or 'refined_obj_t' not in cp:
        raise ValueError('Prior-global seed requires explicit original object-state evidence')
    state.update(obj_R=cp['refined_obj_R'].copy(),obj_t=cp['refined_obj_t'].copy())
    if (state['obj_R'].shape!=(count,3,3) or state['obj_t'].shape!=(count,3)
            or not np.isfinite(state['obj_R']).all() or not np.isfinite(state['obj_t']).all()
            or not np.allclose(state['obj_R'].transpose(0,2,1)@state['obj_R'],np.eye(3),atol=1e-5,rtol=0)
            or not np.allclose(np.linalg.det(state['obj_R']),1.,atol=1e-5,rtol=0)):
        raise ValueError('Prior-global object states must be finite valid poses')
    if reference_rotations is not None:
        try:
            from .soma_sequence_object import build_object_layout, _checked_frame_poses
        except ImportError:
            from soma_sequence_object import build_object_layout, _checked_frame_poses
        layout = build_object_layout(reference_rotations,reference_positions,object_config)
        _checked_frame_poses(state['obj_R'],state['obj_t'],layout)
        if not np.array_equal(lin['object_frame_to_block'],layout.frame_to_block):
            raise ValueError('Prior-global static/shared-precontact object layout differs')
    return dict(config=dict(path=str(path),sha256=config['sha256']),spec=spec,files=files,
        state=state,coordinate_indices=cp['coordinate_indices'],
        dof_indices=lin['scalar_dof_indices'],joint_names=cp['joint_names'].tolist(),
        source_target_sha256=sync['original_target_sha256'])


def checked_state(loaded, fresh, *, coordinates, dofs, names, source_target_sha256):
    state=loaded['state']
    if (state['q'].shape != fresh['q'].shape or state['q'].dtype != fresh['q'].dtype
            or not np.array_equal(coordinates,loaded['coordinate_indices'])
            or not np.array_equal(dofs,loaded['dof_indices']) or list(names) != loaded['joint_names']
            or source_target_sha256 != loaded['source_target_sha256']
            or set(state) != set(fresh)):
        raise ValueError('Prior-global seed differs from live native layout/source targets')
    return {k:v.copy() for k,v in state.items()}


def validate_result(output, report, initial):
    """Bind saved fresh-forward and seeded initial arrays separately."""
    output=Path(output); cfg=report['config'].get('prior_global_seed')
    ev=report.get('prior_global_seed')
    if cfg is None:
        if ev is not None: raise ValueError('Undeclared prior-global seed execution')
        return None
    if (not isinstance(ev,dict) or ev.get('binding')!=cfg
            or ev.get('fresh_forward_executed') is not True
            or ev.get('global_initial_state')!='prior_global_checkpoint'
            or ev.get('motion_accepted') is not False):
        raise ValueError('Missing/inconsistent prior-global execution evidence')
    p=output/'prior_global_seed_states.npz'
    if not p.is_file() or ev.get('states_sha256')!=sha(p):
        raise ValueError('Prior-global seed state evidence changed')
    loaded=load(cfg)
    with np.load(p,allow_pickle=False) as z:
        fields={stage+'_'+k for stage in ['fresh','seed'] for k in ['q','obj_R','obj_t']}|{'timestamps'}
        if set(z.files)!=fields or not np.array_equal(z['timestamps'],np.arange(report['frame_count'])/report['fps']):
            raise ValueError('Prior-global seed state fields/clock differ')
        for k,v in loaded['state'].items():
            if not np.array_equal(z['seed_'+k],v) or not np.array_equal(initial[k],v):
                raise ValueError('Global initial state differs from exact prior checkpoint')
        with np.load(output/'forward_stage_states.npz',allow_pickle=False) as f:
            for k,name in [('q','joint_q'),('obj_R','object_rotations'),('obj_t','object_positions')]:
                if not np.array_equal(z['fresh_'+k],f[name]):
                    raise ValueError('Prior-global fresh state differs from actual forward evidence')
    return dict(ev)
