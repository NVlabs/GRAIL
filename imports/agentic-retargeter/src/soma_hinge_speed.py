"""Optional squared excess-speed cost for original scalar native hinges.

The gate is an authored animation policy, not a manufacturer motor rating.
No residual/gradient acts below it; the original source clock is never changed.
"""
import math
from numbers import Real
import numpy as np
from scipy import sparse


def validate_hinge_speed_config(value):
    if not isinstance(value, dict) or not {'weight', 'sigma_rad_s'} <= set(value) or set(value)-{'weight', 'sigma_rad_s', 'limits_rad_s'}:
        raise ValueError('hinge_speed requires weight and sigma_rad_s; optional limits_rad_s')
    result = {}
    for key in ['weight', 'sigma_rad_s']:
        x = value[key]
        if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0:
            raise ValueError('hinge_speed '+key+' must be finite and positive')
        result[key] = float(x)
    if 'limits_rad_s' in value:
        limits = value['limits_rad_s']
        if not isinstance(limits, dict) or not limits:
            raise ValueError('limits_rad_s must be a nonempty exact joint-name mapping')
        result['limits_rad_s'] = {}
        for name, x in limits.items():
            if not isinstance(name,str) or not name or name != name.strip() or isinstance(x,bool) or not isinstance(x,Real) or not math.isfinite(x) or x <= 0:
                raise ValueError('Invalid exact hinge speed limit')
            result['limits_rad_s'][name] = float(x)
    return result


def hinge_speed_limits(config, names):
    names = list(names)
    if not names or len(set(names)) != len(names) or any(not isinstance(x,str) or not x for x in names):
        raise ValueError('Expected one unique original name per scalar hinge')
    leaf = [name.rsplit('/',1)[-1] for name in names]
    caps = np.asarray([1.5 if n.startswith('waist_') else 2. if any(s in n for s in ['_shoulder_','_elbow_','_wrist_']) else 3. for n in leaf])
    used = set()
    for name, value in config.get('limits_rad_s',{}).items():
        found = [i for i,x in enumerate(names) if x == name]
        if not found: found = [i for i,x in enumerate(leaf) if x == name]
        if len(found) != 1 or found[0] in used:
            raise ValueError('Speed limit joint absent, ambiguous or aliased twice: '+name)
        caps[found[0]] = value; used.add(found[0])
    return caps


def hinge_speed_rows(q, timestamps, config, coordinates, dofs, n_dofs, names, object_variables=0, jacobian=True):
    """Fixed-size residual rows; two analytic entries for each active violation.

    Native scalar hinges are unwrapped original coordinates. No invented
    shortest-angle wrapping, padded endpoints, free-root or object variables.
    """
    config = validate_hinge_speed_config(config)
    q, timestamps = np.asarray(q,dtype=float), np.asarray(timestamps,dtype=float)
    coordinates, dofs = np.asarray(coordinates), np.asarray(dofs)
    if q.ndim != 2 or len(q)<2 or not np.isfinite(q).all() or timestamps.shape!=(len(q),) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps)<=0):
        raise ValueError('Expected finite native sequence and strictly increasing source timestamps')
    if (coordinates.ndim!=1 or dofs.shape!=coordinates.shape or coordinates.dtype.kind not in 'iu' or dofs.dtype.kind not in 'iu' or len(coordinates)!=len(names) or len(set(coordinates))!=len(coordinates) or len(set(dofs))!=len(dofs) or np.any(coordinates<0) or np.any(coordinates>=q.shape[1]) or np.any(dofs<0) or np.any(dofs>=n_dofs) or isinstance(n_dofs,bool) or int(n_dofs)!=n_dofs or n_dofs<1 or isinstance(object_variables,bool) or int(object_variables)!=object_variables or object_variables<0):
        raise ValueError('Invalid bound scalar hinge coordinate/tangent layout')
    limits = hinge_speed_limits(config,names)
    dt = np.diff(timestamps)
    velocity = np.diff(q[:,coordinates],axis=0)/dt[:,None]
    excess = np.maximum(np.abs(velocity)-limits,0.)
    gain = np.sqrt(config['weight'])/config['sigma_rad_s']
    residual = gain*excess.ravel()
    active = excess>0
    matrix = None
    if jacobian:
        frame,joint = np.nonzero(active)
        row = frame*len(coordinates)+joint
        derivative = gain*np.sign(velocity[frame,joint])/dt[frame]
        matrix = sparse.coo_matrix((np.r_[-derivative,derivative],(np.r_[row,row],np.r_[frame*n_dofs+dofs[joint],(frame+1)*n_dofs+dofs[joint]])),shape=(residual.size,len(q)*n_dofs+object_variables)).tocsr()
    per_joint=[]
    for j,name in enumerate(names):
        indices=np.flatnonzero(active[:,j])
        per_joint.append(dict(joint=name,limit_rad_s=float(limits[j]),maximum_rad_s=float(np.max(np.abs(velocity[:,j]))),violation_count=len(indices),violating_frame_pairs=[[int(i),int(i+1)]for i in indices]))
    report=dict(config=config,policy='Authored animation speed gates: default3rad/s, shoulder/elbow/wrist2, waist1.5; explicit exact-name overrides. Not hardware ratings.',joint_names=list(names),limits_rad_s=limits.tolist(),frame_count=len(q),residual_count=len(residual),violating_samples=int(active.sum()),violating_joints=int(np.any(active,axis=0).sum()),max_excess_rad_s=float(excess.max()),residual_energy=.5*float(residual@residual),maximum_rad_s=float(np.abs(velocity).max()),per_joint=per_joint,source_dt_range_s=[float(dt.min()),float(dt.max())],below_gate_gradient='exactly_zero',nondifferentiable_threshold='zero subgradient at equality; squared excess energy is continuously differentiable')
    return residual,matrix,report
