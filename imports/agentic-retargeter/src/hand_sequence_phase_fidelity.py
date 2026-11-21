"""Opt-in phase-scoped fidelity to the bound warm wrist trajectory.

This is a pose prior, not a new contact schedule or a material contact claim.
Only explicitly selected active phases enter the loss; fingers are untouched.
"""
from __future__ import annotations
import numpy as np
from .grasp_io import number
from .hand_sequence_rotation import so3_log

CONTACT_ROLES = frozenset(('contact_acquisition', 'hold', 'grasp', 'stable_hold',
                           'support', 'slide', 'sliding'))


def phase_fidelity_policy(option, policies, *, is_static_obj=False):
    """Validate explicit hand/role weights against unchanged contact policies."""
    if option is None:
        return {}
    if not isinstance(option, dict) or not option or set(option) - set(policies):
        raise ValueError('phase_wrist_fidelity must name explicitly editable hands')
    result = {}
    for side, spec in option.items():
        if not isinstance(spec, dict) or set(spec) != {'roles', 'position', 'rotation'}:
            raise ValueError('phase_wrist_fidelity requires roles, position and rotation')
        roles = spec['roles']
        if (not isinstance(roles, list) or not roles or
                any(not isinstance(r, str) or r not in CONTACT_ROLES for r in roles) or
                len(set(roles)) != len(roles)):
            raise ValueError('phase_wrist_fidelity roles must be unique supported contact roles')
        weights = {key: number(spec[key], 'phase_wrist_fidelity.' + key)
                   for key in ('position', 'rotation')}
        if not any(weights.values()):
            continue
        active, labels = policies[side]
        active, labels = np.asarray(active), np.asarray(labels)
        if active.ndim != 1 or active.dtype.kind != 'b' or labels.shape != active.shape:
            raise ValueError('phase_wrist_fidelity must bind original phase/contact arrays')
        mask = active & np.isin(labels, roles)
        if is_static_obj:
            mask[:] = False
        if not mask.any() and not is_static_obj:
            raise ValueError('phase_wrist_fidelity selected no active original frames')
        result[side] = dict(weights=weights, roles=roles.copy(), mask=mask.copy())
    return result


def phase_fidelity_terms(position, rotation, reference, mask, weights):
    """Mean selected-frame physical m² and principal SO(3) rad²; no velocity pin.

    Static objects have an empty active mask and contribute exactly zero. The
    reference is detached, and no finger parameter enters this objective.
    """
    import torch
    position_error = position - reference['position'].detach()
    rotation_error = so3_log(rotation @ reference['rotation'].detach().transpose(-1, -2))
    m = mask.to(dtype=position.dtype)
    denominator = torch.clamp(m.sum(), min=1.)
    terms = dict(position=(position_error.square().sum(-1) * m).sum() / denominator,
                 rotation=(rotation_error.square().sum(-1) * m).sum() / denominator)
    return sum(weights[key] * terms[key] for key in terms), terms
