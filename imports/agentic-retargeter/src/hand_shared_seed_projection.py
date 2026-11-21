"""Put a shared hand seed inside unchanged all-frame source bounds.

Only the new warm seed is changed. One pose remains shared throughout each
declared grip; allowed transition bases start at their original source pose.
Projection is numerical feasibility, never contact or collision acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .hand_sequence_shared_hold import _phase_clipped_blends, resolve_hold_intervals


class SeedProjectionError(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def projection_policy():
    return dict(kind="all_frame_shared_seed_projection", position_margin_m=1e-5,
                rotation_margin_rad=1e-4, max_rotation_evaluations=200,
                rotation_start_rule="candidate_then_source_first_middle_last_then_mean",
                boundary_base="original_source_only_in_authorized_blend_frames")


def _delta(rotation, references):
    return Rotation.from_matrix(rotation @ references.swapaxes(-1, -2)).as_rotvec()


def _project_rotation(seed, references, weights, bound, options):
    """Find a common SO(3) pose; validate all weighted componentwise boxes."""
    margin = min(options["rotation_margin_rad"], bound*.01)
    target = bound-margin
    def residual(value):
        matrix = Rotation.from_rotvec(value).as_matrix() @ seed
        return np.maximum(np.abs(_delta(matrix, references))*weights[:, None]-target, 0).ravel()
    initial = float(np.max(np.abs(_delta(seed, references))*weights[:, None]))
    attempts = []
    if initial <= target:
        return seed, dict(initial_max_component_rad=initial, final_max_component_rad=initial,
                          adjustment_rad=0., attempts=attempts)
    starts = [np.zeros(3)]
    for frame in sorted(set((0, len(references)//2, len(references)-1))):
        starts.append(Rotation.from_matrix(references[frame] @ seed.T).as_rotvec())
    starts.append(Rotation.from_matrix(Rotation.from_matrix(references).mean().as_matrix() @ seed.T).as_rotvec())
    for index, start in enumerate(starts):
        fit = least_squares(residual, start, max_nfev=options["max_rotation_evaluations"],
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)
        value = Rotation.from_rotvec(fit.x).as_matrix() @ seed
        peak = float(np.max(np.abs(_delta(value, references))*weights[:, None]))
        attempts.append(dict(start=index, evaluations=fit.nfev, optimizer_success=bool(fit.success),
                             max_component_rad=peak, residual_max=float(residual(fit.x).max()),
                             candidate_rotation=value.tolist()))
        if np.isfinite(value).all() and peak <= bound-margin*.5:
            return value, dict(initial_max_component_rad=initial, final_max_component_rad=peak,
                               adjustment_rad=float(Rotation.from_matrix(value @ seed.T).magnitude()),
                               attempts=attempts)
    return None, dict(initial_max_component_rad=initial, final_max_component_rad=None,
                     adjustment_rad=None, attempts=attempts,
                     reason="No feasible shared rotation found; finite deterministic search is not an infeasibility proof.")


def project_seed(source, warm, config, joint_limits):
    """Return a new seed and receipt, or raise with a serializable failure receipt.

The same algorithm and tolerances apply to every case. A weighted transition
constraint is exact when its base is source pose: the shortest SO(3) log scales
by the blend weight, and translation is affine. The paired solver independently
checks its actual float32 decoder before any accepted optimizer step.
"""
    tick = time.monotonic()
    options = projection_policy()
    receipt = dict(schema_version=1, policy=options, status="RUNNING", hands={},
                   original_arrays_edited=False, motion_accepted=False,
                   source_bounds_relaxed=False, phase_intervals_changed=False)
    result = {key: np.array(value, copy=True) for key, value in warm.items()}
    hold_config = config.get("strict_hold_parameterization")
    if not hold_config:
        receipt.update(status="NOT_REQUIRED", elapsed_seconds=time.monotonic()-tick)
        return result, receipt
    if hold_config.get("boundary_blend_policy") != "phase_clipped":
        raise SeedProjectionError("Shared seed projection requires the shared phase-clipped policy", receipt)
    position_bound, rotation_bound = float(config["translation_limit_m"]), float(config["rotation_limit_rad"])
    if not (np.isfinite(position_bound) and position_bound > 0 and np.isfinite(rotation_bound) and rotation_bound > 0):
        raise SeedProjectionError("Finite positive unchanged wrist bounds required", receipt)
    try:
        for side in config["inputs"]["hands"]:
            pk, rk, qk = [side+"_"+name for name in ("wrist_position_object", "wrist_rotation_object", "finger_qpos")]
            sp, sr, sq = [np.asarray(source[key], dtype=float) for key in (pk, rk, qk)]
            p, r, q = [np.asarray(warm[key], dtype=float).copy() for key in (pk, rk, qk)]
            if p.shape != sp.shape or r.shape != sr.shape or q.shape != sq.shape or not all(np.isfinite(x).all() for x in (sp,sr,sq,p,r,q)):
                raise ValueError("Seed/source arrays must be finite and identically aligned")
            count = len(p)
            roles = config["phase_labels"][side]
            active = np.asarray(config["contact_mask"][side], dtype=bool)
            mask = active & np.isin(roles, ["hold", "grasp", "stable_hold"])
            intervals = resolve_hold_intervals(mask, side=side, roles=roles, config=hold_config)
            blends = _phase_clipped_blends(intervals, mask, roles, hold_config["boundary_blend_frames"],
                hold_config["allowed_blend_roles"], hold_config.get("protected_frames", []))
            limits = np.asarray(joint_limits[side], dtype=float)
            if limits.shape != (7,2) or not np.isfinite(limits).all() or np.any(limits[:,1] <= limits[:,0]):
                raise ValueError("Original seven native joint limits required")
            # This is only a warm base projection. Shared outputs are never
            # clamped frame by frame, and the original source stays immutable.
            p = sp + np.clip(p-sp, -position_bound, position_bound)
            rv = Rotation.from_matrix(r @ sr.swapaxes(-1,-2)).as_rotvec()
            r = Rotation.from_rotvec(np.clip(rv,-rotation_bound,rotation_bound)).as_matrix() @ sr
            q = np.clip(q, limits[:,0], limits[:,1])
            rows = []
            receipt["hands"][side] = dict(intervals=rows, transition_base_reset_frames=[])
            for interval, blend in zip(intervals, blends):
                first, end = interval["start"], interval["end_exclusive"]
                ids = list(range(first,end)) + blend["frames"]
                weights = np.r_[np.ones(end-first), np.asarray(blend["weights"],float)]
                if any(frame == 0 for frame in ids):
                    raise ValueError("Original first frame cannot be projected or shared")
                ids = np.asarray(ids,dtype=int)
                original_p = np.asarray(warm[pk][first],float)
                original_r = np.asarray(warm[rk][first],float)
                original_q = np.asarray(warm[qk][first],float)
                low = (sp[ids]-position_bound/weights[:,None]).max(0)
                high = (sp[ids]+position_bound/weights[:,None]).min(0)
                row = {**deepcopy(interval), "constraint_frames":ids.tolist(), "constraint_weights":weights.tolist(),
                       "position_intersection_low":low.tolist(), "position_intersection_high":high.tolist(),
                       "original_position":original_p.tolist(), "original_rotation":original_r.tolist()}
                rows.append(row)
                if np.any(low > high):
                    row["status"]="EMPTY_TRANSLATION_INTERSECTION"
                    raise ValueError(f"{side} {first}:{end} has empty unchanged source translation intersection")
                inset = np.minimum(options["position_margin_m"], np.maximum(0., (high-low)*.25))
                chosen_p = np.clip(original_p,low+inset,high-inset)
                chosen_r, rotation_report = _project_rotation(original_r,sr[ids],weights,rotation_bound,options)
                row["rotation_projection"] = rotation_report
                if chosen_r is None:
                    row["status"]="NO_FEASIBLE_ROTATION_FOUND"
                    raise ValueError(f"{side} {first}:{end} no feasible shared rotation found")
                chosen_q = np.clip(original_q,limits[:,0],limits[:,1])
                p[first:end], r[first:end], q[first:end] = chosen_p, chosen_r, chosen_q
                bid = np.asarray(blend["frames"], dtype=int)
                p[bid],r[bid],q[bid] = sp[bid],sr[bid],sq[bid]
                receipt["hands"][side]["transition_base_reset_frames"].extend(bid.tolist())
                row.update(status="PROJECTED_FEASIBLE_UNREVIEWED", projected_position=chosen_p.tolist(),
                    projected_rotation=chosen_r.tolist(), position_adjustment_m=float(np.linalg.norm(chosen_p-original_p)),
                    finger_adjustment_max_rad=float(np.max(np.abs(chosen_q-original_q))))
            p[0],r[0],q[0] = sp[0],sr[0],sq[0]
            for key, value in ((pk,p),(rk,r),(qk,q)):
                result[key] = value.astype(warm[key].dtype)
                result[key][0] = source[key][0]
            receipt["hands"][side]["base_changed_frames"] = np.flatnonzero(
                np.any(result[pk] != warm[pk], axis=1) | np.any(result[rk] != warm[rk], axis=(1,2))
                | np.any(result[qk] != warm[qk], axis=1)).tolist()
        receipt.update(status="PROJECTED_UNREVIEWED", elapsed_seconds=time.monotonic()-tick)
        return result, receipt
    except (ValueError, KeyError) as error:
        receipt.update(status="FAILED", error=str(error), elapsed_seconds=time.monotonic()-tick)
        raise SeedProjectionError(str(error), receipt) from error
