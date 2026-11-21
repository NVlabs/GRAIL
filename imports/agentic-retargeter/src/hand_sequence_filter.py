"""Bounded offline smoothing of free-hand targets, never motion acceptance.

Centered Savitzky--Golay or Gaussian kernels act on translations/native hinges
and on local SO(3) logarithms. Each output stencil uses one coordinate system:
object coordinates for active contact roles, world coordinates otherwise. This
is a geometric adaptation, not GRAIL's componentwise quaternion filter. It has
no causal lag in stationary interior stencils; phase switches, frozen keys and
bound handling do not promise a globally linear zero-phase response or C2 joins.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import savgol_coeffs
from scipy.spatial.transform import Rotation


SUFFIXES = ("wrist_position_object", "wrist_rotation_object", "finger_qpos")
HOLD_ROLES = ("hold", "grasp", "stable_hold")
DEFAULTS = dict(filter_kind="savgol", window_length=9, polyorder=3,
                sigma_frames=1.5, truncate=3., translation_limit_m=.15,
                rotation_limit_rad=1.5, max_backtracks=20,
                object_relative_roles=[*HOLD_ROLES, "support", "slide"],
                strict_hold_intervals={})


def _exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _options(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("Unknown temporal filter option")
    option = {**DEFAULTS, **config}
    if option["filter_kind"] not in ("savgol", "gaussian"):
        raise ValueError("filter_kind must be savgol or gaussian")
    window, degree = option["window_length"], option["polyorder"]
    if type(window) is not int or not 3 <= window <= 121 or window % 2 != 1:
        raise ValueError("window_length must be odd, 3..121")
    if type(degree) is not int or not 0 <= degree < window:
        raise ValueError("polyorder must be smaller than window_length")
    for name in ("sigma_frames", "truncate", "translation_limit_m", "rotation_limit_rad"):
        if isinstance(option[name], bool) or not np.isfinite(option[name]) or option[name] <= 0:
            raise ValueError(name + " must be finite and positive")
    if option["sigma_frames"] > 30 or option["truncate"] > 10:
        raise ValueError("Gaussian comparison window exceeds bounded budget")
    if type(option["max_backtracks"]) is not int or not 0 <= option["max_backtracks"] <= 30:
        raise ValueError("max_backtracks must be 0..30")
    roles = option["object_relative_roles"]
    if not isinstance(roles, list) or any(not isinstance(x, str) or not x for x in roles):
        raise ValueError("object_relative_roles must be explicit role strings")
    if not isinstance(option["strict_hold_intervals"], dict):
        raise ValueError("strict_hold_intervals must map hands to interval lists")
    return option


def _proper(matrix, name):
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix @ matrix.swapaxes(-1, -2), np.eye(3), atol=2e-5, rtol=0)
            or not np.allclose(np.linalg.det(matrix), 1, atol=2e-5, rtol=0)):
        raise ValueError(name + " must contain proper SO(3) matrices")


def _validate(candidate, source, joint_limits, option):
    core = ("source_qpos", "object_R", "object_trans", "object_scale", "timestamps", "fps", "scope")
    for key in core:
        if key not in candidate or key not in source or not _exact(candidate[key], source[key]):
            raise ValueError("Candidate must bind immutable source field: " + key)
    times = candidate["timestamps"]
    fps = float(candidate["fps"])
    count = len(times)
    if count < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("At least two source frames and positive fps required")
    if not np.array_equal(times, np.arange(count, dtype=np.float64) / fps):
        raise ValueError("Original uniform timestamps required; no retiming")
    if (candidate["source_qpos"].ndim != 2 or len(candidate["source_qpos"]) != count
            or candidate["object_R"].shape != (count, 3, 3)
            or candidate["object_trans"].shape != (count, 3)
            or not all(np.isfinite(candidate[k]).all() for k in core if k != "scope")):
        raise ValueError("Invalid native source/object arrays")
    _proper(candidate["object_R"], "object_R")
    sides = [side for side in ("left", "right") if side + "_finger_qpos" in candidate]
    if not sides or set(joint_limits) != set(sides):
        raise ValueError("Native joint_limits must bind every candidate hand exactly")
    if set(option["strict_hold_intervals"]) - set(sides):
        raise ValueError("Strict hold declaration names an unbound hand")
    for side in sides:
        for suffix, shape in zip(SUFFIXES, ((count, 3), (count, 3, 3), (count, 7))):
            key = side + "_" + suffix
            for arrays in (candidate, source):
                if (key not in arrays or arrays[key].shape != shape
                        or arrays[key].dtype.kind != "f" or not np.isfinite(arrays[key]).all()):
                    raise ValueError("Invalid finite hand array: " + key)
            if not _exact(candidate[key][0], source[key][0]):
                raise ValueError("Original first frame must already be exact: " + key)
        _proper(candidate[side + "_wrist_rotation_object"], side)
        _proper(source[side + "_wrist_rotation_object"], "source " + side)
        for suffix in ("finger_joint_names", "finger_addresses"):
            key = side + "_" + suffix
            if key not in source or key not in candidate or not _exact(candidate[key], source[key]):
                raise ValueError("Native hand identity mismatch: " + key)
        addresses = candidate[side + "_finger_addresses"]
        if (addresses.shape != (7,) or addresses.dtype.kind not in "iu"
                or len(set(addresses.tolist())) != 7 or np.any(addresses < 0)
                or np.any(addresses >= candidate["source_qpos"].shape[1])):
            raise ValueError("Seven original native finger addresses required")
        limits = np.asarray(joint_limits[side], dtype=float)
        if limits.shape != (7, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Native joint_limits must have shape (7,2)")
        roles = candidate.get(side + "_phase_labels")
        active = candidate.get(side + "_contact_active")
        if (roles is None or roles.shape != (count,) or roles.dtype.kind not in "US"
                or active is None or active.shape != (count,) or active.dtype.kind != "b"):
            raise ValueError("Original phase labels and Boolean contact mask required")
    return sides, count, fps


def _frozen_mask(candidate, side, option, count):
    mask = np.zeros(count, dtype=bool)
    mask[0] = True
    rows = option["strict_hold_intervals"].get(side, [])
    if not isinstance(rows, list):
        raise ValueError("Strict hold intervals must be ordered lists")
    end_previous = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"start", "end_exclusive", "grasp_id"}:
            raise ValueError("Strict hold records require start/end_exclusive and optional grasp_id")
        first, end = row.get("start"), row.get("end_exclusive")
        if (type(first) is not int or type(end) is not int
                or not 0 <= first < end <= count or first < end_previous):
            raise ValueError("Strict hold bounds must be ordered nonoverlapping source-frame intervals")
        if (not candidate[side + "_contact_active"][first:end].all()
                or not np.isin(candidate[side + "_phase_labels"][first:end], HOLD_ROLES).all()):
            raise ValueError("Strict hold interval crosses a non-hold phase")
        p, r, q = [candidate[side + "_" + key][first:end] for key in SUFFIXES]
        if (np.max(np.abs(p - p[0])) > 1e-5 or np.max(np.abs(q - q[0])) > 1e-5
                or Rotation.from_matrix(r @ r[0].T).magnitude().max() > 1e-5):
            raise ValueError("Declared strict hold is not constant in input; no quiet freezing")
        mask[first:end] = True
        end_previous = end
    return mask


def _kernel(option):
    if option["filter_kind"] == "savgol":
        weights = savgol_coeffs(option["window_length"], option["polyorder"], use="dot")
    else:
        radius = max(1, int(np.ceil(option["sigma_frames"] * option["truncate"])))
        weights = np.exp(-.5 * (np.arange(-radius, radius + 1) / option["sigma_frames"]) ** 2)
        weights /= weights.sum()
    return weights


def _smooth(p, r, q, object_r, object_t, relative_mask, frozen, weights):
    """A centered Lie-algebra stencil, equivariant to rigid coordinate changes."""
    count, radius = len(p), len(weights) // 2
    indices = np.pad(np.arange(count), (radius, radius), mode="reflect")
    pw = np.einsum("bij,bj->bi", object_r, p) + object_t
    rw = object_r @ r
    output = [p.copy(), r.copy(), q.copy()]
    cut_stencils = []
    for frame in np.flatnonzero(~frozen):
        ids = indices[frame:frame + len(weights)]
        positions, rotations = (p, r) if relative_mask[frame] else (pw, rw)
        proposed_p = weights @ positions[ids]
        delta = Rotation.from_matrix(rotations[ids] @ rotations[frame].T).as_rotvec()
        if np.max(np.linalg.norm(delta, axis=1)) > np.pi - 1e-4:
            cut_stencils.append(int(frame))
        proposed_r = Rotation.from_rotvec(weights @ delta).as_matrix() @ rotations[frame]
        if not relative_mask[frame]:
            proposed_p = object_r[frame].T @ (proposed_p - object_t[frame])
            proposed_r = object_r[frame].T @ proposed_r
        output[0][frame], output[1][frame], output[2][frame] = proposed_p, proposed_r, weights @ q[ids]
    return output, cut_stencils


def _bounds(values, references, limits, option):
    p, r, q = values
    sp, sr, _ = references
    return dict(position_excess_m=float(max(0., np.max(np.abs(p-sp))-option["translation_limit_m"])),
                rotation_excess_rad=float(max(0., np.max(np.abs(Rotation.from_matrix(r @ sr.swapaxes(-1,-2)).as_rotvec()))-option["rotation_limit_rad"])),
                finger_excess_rad=float(max(0., np.max(limits[:,0]-q), np.max(q-limits[:,1]))))


def _feasible(bounds):
    return max(bounds.values()) <= 1e-6


def _stale_diagnostic(key):
    return (key.startswith(("parameterization_", "collocation_", "paired_sampled_"))
            or "_sampled_object_" in key or "_sampled_scene_" in key
            or key.endswith("_per_frame_contact_loss_m2"))


def filter_arrays(candidate, source, *, joint_limits, config=None):
    """Filter a candidate against a separately frozen source; inputs are never edited.

``joint_limits`` maps each native hand to seven [low,high] rows. Optional
``strict_hold_intervals`` maps each hand to the solver's half-open hold records.
If the candidate contains shared controls, explicit interval records are required.
The tool does not infer contact or accept poses. Re-audit all changed geometry.
"""
    start = time.monotonic()
    option = _options({} if config is None else config)
    sides, count, fps = _validate(candidate, source, joint_limits, option)
    if "parameterization_shared_hold_controls" in candidate:
        controls = candidate["parameterization_shared_hold_controls"]
        if controls.ndim != 2 or controls.shape[1] != 13:
            raise ValueError("Invalid shared-control metadata")
        declarations = option["strict_hold_intervals"]
        if not all(isinstance(rows, list) for rows in declarations.values()):
            raise ValueError("Strict hold intervals must be ordered lists")
        if sum(len(rows) for rows in declarations.values()) != len(controls):
            raise ValueError("Shared-control input requires explicit strict_hold_intervals for every control")
    removed = sorted(key for key in candidate if _stale_diagnostic(key))
    result = {key: np.array(value, copy=True) for key, value in candidate.items() if key not in removed}
    weights = _kernel(option)
    report = dict(schema_version=1, status="UNREVIEWED", motion_accepted=False,
                  config=option, frame_count=count, fps=fps, kernel_weights=weights.tolist(),
                  removed_stale_diagnostics=removed, hands={}, geometry_audited=False,
                  coordinate_rule="One coordinate system per full centered stencil: active declared contact roles use object space, all other centers use world space; no numeric mixing of frames.",
                  boundary_rule="Reflect source-clock sample indices; strict holds and original first frame are exact frozen inputs.",
                  limitations=["No contact/collision constraints; re-audit the exported trajectory.",
                    "Frozen keys, phase switches and bounds can leave or introduce derivative jumps.",
                    "SO(3) principal logarithm is ambiguous at pi; affected stencils are listed.",
                    "Native linear replay is not guaranteed C2 or dynamically feasible."])
    for side in sides:
        keys = [side + "_" + key for key in SUFFIXES]
        original = [np.asarray(candidate[key], dtype=float) for key in keys]
        references = [np.asarray(source[key], dtype=float) for key in keys]
        limits = np.asarray(joint_limits[side], dtype=float)
        before = _bounds(original, references, limits, option)
        if not _feasible(before):
            raise ValueError("Input violates unchanged original-source/native bounds: " + side + " " + str(before))
        frozen = _frozen_mask(candidate, side, option, count)
        relative = candidate[side + "_contact_active"] & np.isin(candidate[side + "_phase_labels"], option["object_relative_roles"])
        proposed, cuts = _smooth(*original, candidate["object_R"], candidate["object_trans"], relative, frozen, weights)
        unclamped_q = proposed[2].copy()
        proposed[2][~frozen] = np.clip(proposed[2][~frozen], limits[:, 0], limits[:, 1])
        delta = Rotation.from_matrix(proposed[1] @ original[1].swapaxes(-1, -2)).as_rotvec()
        trials = []
        for fraction in [2. ** -i for i in range(option["max_backtracks"] + 1)] + [0.]:
            values = [original[0] + fraction * (proposed[0]-original[0]),
                      Rotation.from_rotvec(fraction * delta).as_matrix() @ original[1],
                      original[2] + fraction * (proposed[2]-original[2])]
            values = [value.astype(candidate[key].dtype) for key, value in zip(keys, values)]
            for key, value in zip(keys, values):
                value[frozen] = candidate[key][frozen]
                if fraction == 0:
                    value[:] = candidate[key]
            status = _bounds(values, references, limits, option)
            trials.append(dict(fraction=fraction, **status))
            if _feasible(status):
                break
        else:
            raise AssertionError("Validated original must remain a feasible fallback")
        for key, value in zip(keys, values):
            result[key] = value
        report["hands"][side] = dict(frozen_frames=np.flatnonzero(frozen).tolist(),
            strict_hold_intervals=option["strict_hold_intervals"].get(side, []),
            original_native_joint_limits=limits.tolist(), initial_bounds=before, final_bounds=status,
            accepted_correction_fraction=fraction, backtracking_trials=trials,
            finger_clipped_values=int(np.count_nonzero(unclamped_q != proposed[2])),
            finger_clip_max_rad=float(np.max(np.abs(unclamped_q-proposed[2]))),
            near_pi_stencil_frames=cuts, object_coordinate_center_frames=np.flatnonzero(relative).tolist(),
            exact_frozen_values=all(_exact(result[key][frozen], candidate[key][frozen]) for key in keys),
            changed_frames=np.flatnonzero(np.any(values[0] != candidate[keys[0]], axis=1)
                | np.any(values[1] != candidate[keys[1]], axis=(1,2))
                | np.any(values[2] != candidate[keys[2]], axis=1)).tolist())
    edited = {side+"_"+suffix for side in sides for suffix in SUFFIXES}
    protected = sorted(set(result)-edited)
    if not all(_exact(result[key], candidate[key]) for key in protected):
        raise AssertionError("Immutable native arrays changed")
    report.update(preserved_array_keys=protected, preserved_arrays_exact=True,
                  elapsed_seconds=time.monotonic()-start)
    return result, report


def _load(path):
    payload = Path(path).read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    return arrays, dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def run(candidate_path, source_path, output, *, joint_limits, config=None):
    """Write new UNREVIEWED artifacts; never overwrite a previous result directory."""
    out = Path(output)
    if out.exists():
        raise FileExistsError("Filter output must be a new directory")
    candidate, candidate_binding = _load(candidate_path)
    source, source_binding = _load(source_path)
    result, report = filter_arrays(candidate, source, joint_limits=joint_limits, config=config)
    report.update(candidate=candidate_binding, source=source_binding,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True)
    target = out / "hand_sequence_UNREVIEWED.npz"
    np.savez_compressed(target, **result)
    report["output"] = dict(path=str(target.resolve()), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--joint-limits", required=True, help="JSON mapping side to original seven [low,high] rows")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.candidate, args.source, args.out,
        joint_limits=json.loads(Path(args.joint_limits).read_text()),
        config=json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
