"""Optional physical world-wrist derivatives outside relative contact stencils.

Free/acquisition/phase-join stencils use world translation and spatial angular
velocity. Complete same-role hold/support/slide stencils are excluded, leaving
the existing relative objective responsible for those phases. No velocity cost
or artificial free-hand/object coupling is introduced. Original hand residuals
and Jacobians supply actual poses; physical world wrists have zero object DOFs.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

try:
    from .soma_relative_temporal import (RelativeHandTemporal, angular_velocity_jacobian,
        left_jacobian, phase_starts, _difference_matrix, validate_relative_temporal_config)
except ImportError:
    from soma_relative_temporal import (RelativeHandTemporal, angular_velocity_jacobian,
        left_jacobian, phase_starts, _difference_matrix, validate_relative_temporal_config)


_WEIGHTS = {kind+'_'+name+'_weight' for kind in ('translation', 'rotation')
            for name in ('acceleration', 'jerk')}


def validate_world_temporal_config(value):
    if not isinstance(value, dict) or set(value)-_WEIGHTS-{'roles'}:
        raise ValueError('Invalid world_hand_temporal configuration fields')
    # Share exact SI-weight and role validation, but never enable velocity rows.
    result = validate_relative_temporal_config(value)
    return {key: result[key] for key in sorted(_WEIGHTS)} | {'roles': result['roles']}


def recover_world_pose(position_residual, rotation_residual, position_jacobian,
                       rotation_jacobian, gains, desired_position, desired_rotation):
    gp, gr = np.asarray(gains, dtype=float)[:2]
    if not np.isfinite([gp,gr]).all() or min(gp,gr) <= 0:
        raise ValueError('World wrist reconstruction requires positive finite hand pose gains')
    phi = np.asarray(rotation_residual)/gr
    result = dict(position=np.asarray(desired_position)-np.asarray(position_residual)/gp,
                  rotation=Rotation.from_rotvec(phi).as_matrix()@desired_rotation)
    if position_jacobian is None and rotation_jacobian is None:
        return result
    if position_jacobian is None or rotation_jacobian is None:
        raise ValueError('Both world wrist Jacobian blocks must be supplied together')
    result.update(position_robot=-np.asarray(position_jacobian)/gp,
                  rotation_robot=left_jacobian(phi)@rotation_jacobian/gr)
    return result


def world_starts(labels, contact_mask, order, roles):
    """Exact complement of declared complete same-role relative stencils."""
    return np.setdiff1d(np.arange(len(labels)-order), phase_starts(labels,contact_mask,order,roles))


def world_temporal_rows(poses, fps, config, frame_to_block=None, object_variables=0, *, jacobian=True):
    config = validate_world_temporal_config(config)
    if not poses or not np.isfinite(fps) or fps<=0:
        raise ValueError('World wrist temporal rows need hands and a finite positive FPS')
    residuals,matrices,reports = [],[],{}
    columns = None
    for side,pose in poses.items():
        position,rotation = np.asarray(pose['position']),np.asarray(pose['rotation'])
        count = len(position)
        if (position.shape!=(count,3) or rotation.shape!=(count,3,3) or count<2
                or not np.isfinite(position).all() or not np.isfinite(rotation).all()):
            raise ValueError('World wrist poses must be finite complete trajectories')
        labels,mask = np.asarray(pose['phase_labels']),np.asarray(pose['contact_mask'])
        if labels.shape!=(count,) or mask.shape!=(count,) or mask.dtype.kind!='b':
            raise ValueError('World wrist phase timeline differs')
        omega,omega_pose_j = angular_velocity_jacobian(rotation,fps)
        if jacobian:
            p_j = sparse.block_diag(list(pose['position_robot']),format='csr')
            r_j = sparse.block_diag(list(pose['rotation_robot']),format='csr')
            if object_variables:
                p_j = sparse.hstack([p_j,sparse.csr_matrix((3*count,object_variables))],format='csr')
                r_j = sparse.hstack([r_j,sparse.csr_matrix((3*count,object_variables))],format='csr')
            omega_j = omega_pose_j@r_j
            columns = p_j.shape[1]
        terms = {}
        for kind,values in (('translation',position),('rotation',omega)):
            for order,name in ((2,'acceleration'),(3,'jerk')):
                key = kind+'_'+name+'_weight';weight = config[key]
                starts = world_starts(labels,mask,order,config['roles'])
                derivative_order = order if kind=='translation' else order-1
                difference = _difference_matrix(len(values),derivative_order)[starts]
                physical = np.asarray(difference@values)*fps**derivative_order
                if weight and len(starts):
                    residuals.append((np.sqrt(weight)*physical).ravel())
                    if jacobian:
                        matrix = p_j if kind=='translation' else omega_j
                        matrices.append(np.sqrt(weight)*fps**derivative_order*
                            sparse.kron(difference,sparse.eye(3),format='csr')@matrix)
                norms = np.linalg.norm(physical,axis=1)
                terms[key] = dict(weight=weight,stencil_count=len(starts),
                    active_residual_count=3*len(starts) if weight else 0,stencil_first_frames=starts.tolist(),
                    component_rms=float(np.sqrt(np.mean(physical**2))) if physical.size else None,
                    vector_max=float(norms.max()) if norms.size else None,energy=float(weight*np.sum(physical**2)),
                    unit=('m' if kind=='translation' else 'rad')+'/s'+str(order))
        reports[side] = terms
    residual = np.concatenate(residuals) if residuals else np.zeros(0)
    matrix = ((sparse.vstack(matrices,format='csr') if matrices else sparse.csr_matrix((0,columns)))
              if jacobian else None)
    report = dict(enabled=True,config=config,fps=float(fps),frame_count=count,residual_count=len(residual),
        weighted_energy=float(residual@residual),hands=reports,
        derivative='World-frame position differences; spatial SO(3) log angular velocity followed by angular differences.',
        phase_policy='Complement of complete same-role active hold/support/slide stencils: free, acquisition and phase joins use world coordinates.',
        object_policy='Physical world wrists depend only on native robot tangents; all object columns are exactly zero.',
        velocity_policy='No zero-velocity objective; legitimate approach, release, support and slide motion is retained.',
        jacobian='Analytic original Newton hand rows; world Cartesian robot and free-root tangent maps.',
        energy_normalization='Sum of squared residuals, no frame or component mean.',extra_gpu_fk_queries=0,extra_device_transfers=0,
        limitations=['Sampled derivatives are not continuous-time collision or dynamic feasibility certificates.',
                     'SO(3) log derivatives are locally defined away from the pi cut locus.'])
    return residual,matrix,report


class WorldHandTemporal(RelativeHandTemporal):
    def __init__(self,group,offsets,fps,config,object_layout):
        normalized = validate_world_temporal_config(config)
        super().__init__(group,offsets,fps,normalized,object_layout)
        self.config = normalized

    def sample(self,frame,residual,matrix,object_rotation,object_position):
        result = {}
        for item in self.group.hands:
            data = item['data']
            if data['contact_mask'][frame]:
                desired_p = object_rotation@data['wrist_position_object'][frame]+object_position
                desired_r = object_rotation@data['wrist_rotation_object'][frame]
            else:
                desired_p,desired_r = data['wrist_position_world'][frame],data['wrist_rotation_world'][frame]
            p,r = self.offsets[id(item['position'])],self.offsets[id(item['rotation'])]
            result[item['side']] = recover_world_pose(residual[p:p+3],residual[r:r+3],
                None if matrix is None else matrix[p:p+3],None if matrix is None else matrix[r:r+3],
                self.group.gains,desired_p,desired_r)
        return result

    def assemble(self,samples,jacobian):
        poses = {}
        for item in self.group.hands:
            side,data = item['side'],item['data']
            poses[side] = {key:np.asarray([sample[side][key] for sample in samples]) for key in samples[0][side]}
            poses[side].update(phase_labels=data['phase_labels'],contact_mask=data['contact_mask'])
        residual,matrix,self.report = world_temporal_rows(poses,self.fps,self.config,
            None if self.layout is None else self.layout.frame_to_block,
            0 if self.layout is None else self.layout.variable_count,jacobian=jacobian)
        self.last_poses = poses
        if self.before_report is None:self.before_report=self.report
        return residual,matrix

    def evidence(self):
        result = {'world_hand_sides':np.asarray(list(self.last_poses))}
        for side,pose in self.last_poses.items():
            result.update({'world_'+side+'_'+key:np.asarray(value)for key,value in pose.items()})
        return result
