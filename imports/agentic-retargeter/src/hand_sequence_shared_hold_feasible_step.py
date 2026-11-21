"""Opt-in feasibility filtering for eager shared-hold Adam proposals.

The generic filter changes all base/shared controls by one common step fraction.
The caller checks the actual masked decoder at every original source frame.
No shared output is clamped, and support/slide frames remain ordinary variables.
"""

from .hand_sequence_feasible_step import validate_feasible_spline_step


def project_base_controls(parameters, hands, position_bound, rotation_bound):
    """Project only ordinary per-frame controls before decoded backtracking.

    A free finger at its native limit must not force a zero common step for
    otherwise valid shared controls. Shared poses and blended outputs are never
    clamped: the subsequent full decoder feasibility check owns those.
    """
    import torch
    with torch.no_grad():
        parameters[:,:,:3].clamp_(-position_bound,position_bound)
        parameters[:,:,3:6].clamp_(-rotation_bound,rotation_bound)
        for index,hand in enumerate(hands.values()):
            q=hand['source_finger']+parameters[:,index,6:]
            parameters[:,index,6:]=torch.minimum(torch.maximum(q,hand['low']),hand['high'])-hand['source_finger']


def validate_feasible_shared_hold_step(value, *, shared_hold, spline=None,
                                      execution_backend='eager',
                                      feasible_spline_step=None):
    if value is None or value is False:
        return None
    if not isinstance(value, dict) or value.get('kind') != 'backtracking':
        raise ValueError('feasible_shared_hold_step requires kind=backtracking')
    try:
        options = validate_feasible_spline_step(value)
    except ValueError as error:
        raise ValueError('Invalid feasible_shared_hold_step: '+str(error)) from error
    if (execution_backend != 'eager' or spline is not None
            or shared_hold is None or not shared_hold.controls.numel()):
        raise ValueError('feasible_shared_hold_step requires eager per-frame '
                         'parameters with at least one explicit shared hold')
    if feasible_spline_step is not None and feasible_spline_step is not False:
        raise ValueError('feasible_shared_hold_step and feasible_spline_step are exclusive')
    return options


def require_feasible_shared_hold_start(status):
    """Reject an infeasible seed; this policy never repairs/clamps it silently."""
    if not status.get('feasible') or not status.get('finite'):
        raise ValueError('feasible_shared_hold_step requires an initially finite feasible '
                         'state inside original wrist boxes and native finger limits')


def repair_shared_hold_boundary_start(parameters, shared, hands, position_bound,
                                      rotation_bound, status, options, editable_frames=None):
    """Repair only invalid initial boundary *base* controls, never shared poses.

    An exported warm trajectory already contains the boundary blend. Treating
    its samples as base controls blends them twice. Valid legacy starts remain
    byte-identical. For invalid boundary samples, first invert the blend toward
    the intended warm sample, then enforce the unchanged base/native boxes.
    Shortest-arc SO(3) inversion may not survive those boxes or the pi branch;
    in that case, try a bounded backtrack of the original base toward source.
    Strict holds, all other samples and the shared controls remain unchanged.
    No objective is evaluated and no constraint is relaxed. Failure restores
    every original control and remains a failed initialization.
    """
    import torch
    from .hand_sequence import _rotation
    from .hand_sequence_rotation import so3_log

    before = status()
    report = dict(schema_version=1, policy='invalid_boundary_inverse_then_source_backtracking',
                  initial_status=before, rows=[], motion_accepted=False,
                  source_bounds_relaxed=False, shared_controls_changed=False)
    if before.get('feasible') and before.get('finite'):
        return dict(report, status='UNCHANGED_FEASIBLE', final_status=before)
    saved = parameters.detach().clone()
    shared_saved = shared.controls.detach().clone()

    def row_status(value, hand, frame):
        if not bool(torch.isfinite(value).all()):
            return dict(feasible=False, finite=False)
        q = hand['source_finger'][frame] + value[6:]
        if not bool(torch.isfinite(q).all()):
            return dict(feasible=False, finite=False)
        metrics = (float(torch.relu(value[:3].abs()-position_bound).max()),
                   float(torch.relu(value[3:6].abs()-rotation_bound).max()),
                   float(torch.maximum(hand['low']-q,q-hand['high']).clamp_min(0).max()))
        return dict(feasible=max(metrics)<=1e-6,finite=True,
                    position_box_excess_m=metrics[0],rotation_vector_box_excess_rad=metrics[1],
                    native_finger_limit_excess_rad=metrics[2])

    with torch.no_grad():
        # Iterate only declared exterior buffers. A bad strict grip cannot be
        # corrected by this procedure and causes the transactional rollback.
        for index, record in enumerate(shared.records):
            if not shared.boundary_blends:
                break
            ids, weights = shared.boundary_blends[index]
            hand = hands[record['side']];hand_index=record['hand_index']
            shared_value = shared.controls[index]
            sp = shared_value[:3]
            sr = _rotation(shared_value[None,3:6])[0] @ shared.references[index]
            sq = hand['low']+(hand['high']-hand['low'])*shared_value[6:].sigmoid()
            for frame_tensor, weight_tensor in zip(ids, weights):
                frame = int(frame_tensor);weight=float(weight_tensor)
                if frame == 0 or (editable_frames is not None
                        and not bool(editable_frames[record['side']][frame])):
                    continue
                old = saved[frame,hand_index]
                initial = row_status(shared.decode(parameters)[frame,hand_index],hand,frame)
                if initial['feasible']:
                    continue
                row = dict(side=record['side'],frame=frame,blend_weight=weight,
                           reason='warm_sample_reblended_outside_original_bounds',before=initial,
                           inverse_clipped_components=[],fallback_fraction=None)
                report['rows'].append(row)
                if not initial['finite'] or not 0 < weight < 1:
                    row['status']='INVALID_BOUNDARY';continue
                target_p = hand['source_position'][frame]+old[:3]
                target_r = _rotation(old[None,3:6])[0] @ hand['source_rotation'][frame]
                target_q = hand['source_finger'][frame]+old[6:]
                base_p = (target_p-weight*sp)/(1-weight)
                delta = so3_log((sr @ target_r.T)[None])[0]
                base_r = _rotation((-weight/(1-weight)*delta)[None])[0] @ target_r
                base_q = (target_q-weight*sq)/(1-weight)
                inverse = torch.cat((base_p-hand['source_position'][frame],
                    so3_log((base_r @ hand['source_rotation'][frame].T)[None])[0],
                    base_q-hand['source_finger'][frame]))
                clipped = inverse.clone()
                clipped[:3].clamp_(-position_bound,position_bound)
                clipped[3:6].clamp_(-rotation_bound,rotation_bound)
                clipped[6:] = torch.minimum(torch.maximum(base_q,hand['low']),hand['high'])-hand['source_finger'][frame]
                changed=torch.zeros_like(old,dtype=torch.bool)
                for key,first,end in [('position_box_excess_m',0,3),
                        ('rotation_vector_box_excess_rad',3,6),('native_finger_limit_excess_rad',6,13)]:
                    if initial.get(key,0)>1e-6:
                        changed[first:end]=True
                candidate=torch.where(changed,clipped,old)
                row['repaired_component_indices']=torch.nonzero(changed).flatten().cpu().tolist()
                row['inverse_clipped_components']=torch.nonzero(changed & (clipped!=inverse)).flatten().cpu().tolist()
                row['inverse_base_clip_max_abs']=float(torch.where(changed,(clipped-inverse).abs(),0.).max())
                parameters[frame,hand_index].copy_(candidate)
                decoded = shared.decode(parameters)[frame,hand_index]
                decoded_r = _rotation(decoded[None,3:6])[0] @ hand['source_rotation'][frame]
                row['inverse_warm_error']=dict(position_m=float((decoded[:3]-old[:3]).norm()),
                    rotation_rad=float(so3_log((decoded_r @ target_r.T)[None]).norm()),
                    finger_max_rad=float((decoded[6:]-old[6:]).abs().max()))
                current=row_status(decoded,hand,frame)
                row['inverse_status']=current
                row['method']='inverse_blend'
                if not current['feasible']:
                    row['method']='source_backtracking';row['fallback_trials']=[]
                    fractions=[options['shrink_factor']**i for i in range(1,options['max_backtracks']+1)]+[0.]
                    for fraction in fractions:
                        parameters[frame,hand_index].copy_(torch.where(changed,old*fraction,old))
                        current=row_status(shared.decode(parameters)[frame,hand_index],hand,frame)
                        row['fallback_trials'].append(dict(fraction=fraction,status=current))
                        if current['feasible']:
                            row['fallback_fraction']=fraction;break
                row.update(status='REPAIRED_FEASIBLE' if current['feasible'] else 'FAILED',after=current,
                    base_control_max_abs_change=float((parameters[frame,hand_index]-old).abs().max()))
        final = status()
        if not final.get('feasible') or not final.get('finite'):
            parameters.copy_(saved)
            report.update(status='FAILED_RESTORED',attempted_final_status=final,final_status=status())
        else:
            report.update(status='REPAIRED_FEASIBLE_UNREVIEWED',final_status=final)
        if not torch.equal(shared.controls,shared_saved):
            raise AssertionError('Boundary initialization changed a shared grip')
    return report
