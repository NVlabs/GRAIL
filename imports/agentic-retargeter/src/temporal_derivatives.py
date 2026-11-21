"""Timestamp-normalized sampled derivatives and a full-hinge acceptance gate.

Derivatives are factorial-scaled divided differences on consecutive stencils.
They equal the usual forward differences / dt**order on a uniform grid and are
exact for degree-``order`` polynomials on nonuniform grids. Stencil mean times
locate estimates, not a claim about continuous interpolation or actuator limits.
No angular wrapping, smoothing, resampling or frame omission is performed.
"""

from __future__ import annotations

import numpy as np

from .contact_edit import native_io
from .contact_edit.pose import _keys, _weight


DERIVATIVE_KEYS = {
    "max_acceleration_rad_s2", "max_jerk_rad_s3",
    "joint_acceleration_limits_rad_s2", "joint_jerk_limits_rad_s3",
}
_TEMPORAL_KEYS = DERIVATIVE_KEYS | {
    "max_step_rad", "max_velocity_rad_s", "joint_velocity_limits_rad_s",
}
METHOD = "factorial-scaled divided differences on consecutive source stencils; no angle unwrapping"


def _order_axis(order, axis, ndim):
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or not -ndim <= axis < ndim:
        raise ValueError("axis must index the values array")
    return int(order), int(axis) % ndim


def _times(timestamps, count, order):
    ts = np.asarray(timestamps)
    if ts.shape != (count,) or ts.dtype.kind not in "fiu" or not np.isfinite(ts).all():
        raise ValueError("timestamps must be a finite numeric vector matching the sample axis")
    ts = ts.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        steps = np.diff(ts)
        span = ts[-1] - ts[0] if len(ts) else 0
    if not np.isfinite(steps).all() or np.any(steps <= 0) or not np.isfinite(span):
        raise ValueError("timestamps must be strictly increasing with finite spans")
    if count <= order:
        raise ValueError("derivative order requires at least order + 1 samples")
    return ts


def finite_difference(values, timestamps, order=1, axis=0):
    """Return ``(derivative, stencil_mean_times)`` in value-units / seconds**order.

    The sample dimension shrinks by ``order``; all other axes are preserved.
    Every estimate uses exactly source frames ``i:i+order+1``. Unlike repeated
    midpoint differentiation, this recurrence remains polynomial-exact on
    nonuniform timestamps. Inputs are not modified.
    """
    values = np.asarray(values)
    order, axis = _order_axis(order, axis, values.ndim)
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("values must be finite real numeric samples")
    ts = _times(timestamps, values.shape[axis], order)
    result = np.moveaxis(values.astype(np.float64), axis, 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for degree in range(1, order + 1):
            span = (ts[degree:] - ts[:-degree]).reshape((-1,) + (1,) * (result.ndim - 1))
            result = degree * np.diff(result, axis=0) / span
    if not np.isfinite(result).all():
        raise ValueError("derivative computation produced nonfinite values")
    # Subtract the origin before averaging to avoid overflowing large epochs.
    mean_times = ts[0] + np.stack([ts[i:len(ts)-order+i] - ts[0]
                                 for i in range(order + 1)]).mean(axis=0)
    return np.moveaxis(result, 0, axis), mean_times


def torch_finite_difference(values, timestamps, order=1, axis=0):
    """Torch counterpart; retains device, floating dtype and value gradients.

    Timestamp tensors retain their gradient too. Importing this module does not
    require Torch. Use float64 for large absolute epochs or tightly spaced times.
    """
    import torch

    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise ValueError("values must be a floating Torch tensor")
    order, axis = _order_axis(order, axis, values.ndim)
    ts = torch.as_tensor(timestamps, dtype=values.dtype, device=values.device)
    if ts.ndim != 1 or len(ts) != values.shape[axis] or len(ts) <= order:
        raise ValueError("timestamps must match the sample axis with at least order + 1 samples")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(ts).all()):
        raise ValueError("values and timestamps must be finite")
    if not bool(((ts[1:] - ts[:-1]) > 0).all()) or not bool(torch.isfinite(ts[-1] - ts[0])):
        raise ValueError("timestamps must be strictly increasing with finite spans")
    result = values.movedim(axis, 0)
    for degree in range(1, order + 1):
        span = (ts[degree:] - ts[:-degree]).reshape((-1,) + (1,) * (result.ndim - 1))
        result = degree * (result[1:] - result[:-1]) / span
    if not bool(torch.isfinite(result).all()):
        raise ValueError("derivative computation produced nonfinite values")
    mean_times = ts[0] + torch.stack([ts[i:len(ts)-order+i] - ts[0]
                                    for i in range(order + 1)]).mean(dim=0)
    return result.movedim(0, axis), mean_times


class JointDerivativeGate:
    """Hard sampled acceleration/jerk limits across all native hinges.

    Both global limits must be explicitly configured for a PASS. Per-joint maps
    replace their respective global limit. Missing limits remain unknown rather
    than silently choosing thresholds or treating a soft objective as a gate.
    Editability only classifies violations; protected samples and joins count.
    """

    def __init__(self, temporal, model_snapshot):
        _keys(temporal, set(), _TEMPORAL_KEYS, "temporal")
        self.joints = native_io._model_joints(model_snapshot)
        self.hinges = [j for j in self.joints if j["joint_type"] == 3]
        self.nq = max(j["address"] + {0: 7, 1: 4, 2: 1, 3: 1}[j["joint_type"]]
                      for j in self.joints)
        self.policies = {}
        names = {j["joint"] for j in self.hinges}
        for label, suffix, order in (("acceleration", "rad_s2", 2), ("jerk", "rad_s3", 3)):
            key, override_key = f"max_{label}_{suffix}", f"joint_{label}_limits_{suffix}"
            limit = _weight(temporal[key], key, positive=True) if key in temporal else None
            overrides = temporal.get(override_key, {})
            if not isinstance(overrides, dict) or (override_key in temporal and limit is None):
                raise ValueError(f"{override_key} requires a named map and {key}")
            for name, value in overrides.items():
                if name not in names:
                    raise ValueError("Derivative override is not a native hinge: " + str(name))
                _weight(value, override_key + ": " + name, positive=True)
            self.policies[label] = (suffix, order, limit, {n: float(v) for n, v in overrides.items()})
        self.configured = all(p[2] is not None for p in self.policies.values())
        if any(p[2] is not None for p in self.policies.values()) and not self.hinges:
            raise ValueError("Joint derivative gate requires native hinges")

    def evaluate(self, qpos, timestamps, editable_mask=None):
        """Return JSON-safe metrics and failing source stencils without selecting."""
        q = np.asarray(qpos)
        if q.ndim != 2 or q.shape[1] != self.nq or q.dtype.kind != "f" or not np.isfinite(q).all():
            raise ValueError("qpos must be finite floating [frames, model nq]")
        ts = _times(timestamps, len(q), 3)
        if editable_mask is not None:
            editable_mask = np.asarray(editable_mask)
            if editable_mask.shape != q.shape or editable_mask.dtype.kind != "b":
                raise ValueError("editable_mask must be a boolean qpos-shaped array")
        report = dict(schema_version=1, configured=self.configured, selected=False,
                      motion_accepted=False, scope="all_native_hinges_all_consecutive_stencils",
                      frame_count=len(q), hinge_count=len(self.hinges), method=METHOD,
                      timestamps_s=ts.tolist(), min_interval_s=float(np.diff(ts).min()),
                      max_interval_s=float(np.diff(ts).max()), rms_definition="unweighted RMS over derivative samples",
                      excluded_joints=[j["joint"] for j in self.joints if j["joint_type"] != 3],
                      limitations=["Sampled derivatives do not certify continuous curve derivatives or actuator feasibility.",
                                   "Configured animation acceptance limits are not hardware ratings; a pass does not select the motion."],
                      per_joint=[], violations=[])
        metrics = {}
        for label, (suffix, order, limit, overrides) in self.policies.items():
            values, times = finite_difference(q[:, [j["address"] for j in self.hinges]], ts, order)
            metrics[label] = (values, times)
            report[f"max_{label}_{suffix}_limit"] = limit
            report[f"joint_{label}_limits_{suffix}"] = overrides.copy()
            report[f"max_{label}_{suffix}"] = float(np.max(np.abs(values))) if values.size else None
            report[f"rms_{label}_{suffix}"] = _rms(values) if values.size else None
        for column, joint in enumerate(self.hinges):
            row = dict(joint=joint["joint"], qpos_address=joint["address"])
            for label, (suffix, order, default, overrides) in self.policies.items():
                values, times = metrics[label]
                signed = values[:, column]
                magnitude = np.abs(signed)
                peak = int(np.argmax(magnitude))
                limit = overrides.get(joint["joint"], default)
                bad = np.flatnonzero(magnitude > limit + 1e-9) if limit is not None else np.array([], dtype=int)
                row[label] = dict(units=f"rad/s^{order}", limit=limit,
                                  maximum=float(magnitude[peak]), rms=_rms(signed),
                                  peak_support_frames=list(range(peak, peak + order + 1)),
                                  peak_timestamps_s=ts[peak:peak + order + 1].tolist(),
                                  peak_sample_time_s=float(times[peak]), violation_count=len(bad),
                                  passed=not len(bad) if limit is not None else None)
                for index in bad.tolist():
                    support = list(range(index, index + order + 1))
                    editable = ([f for f in support if editable_mask[f, joint["address"]]]
                                if editable_mask is not None else None)
                    report["violations"].append(dict(
                        kind="temporal", metric=f"max_{label}_{suffix}", units=f"rad/s^{order}",
                        joint=joint["joint"], qpos_address=joint["address"], value=float(magnitude[index]),
                        signed_value=float(signed[index]), limit=limit, support_frames=support,
                        timestamps_s=ts[index:index + order + 1].tolist(), sample_time_s=float(times[index]),
                        editable_support_frames=editable,
                        affected_by_edit=bool(editable) if editable is not None else None))
            report["per_joint"].append(row)
        report["violation_count"] = len(report["violations"])
        report["passed"] = False if report["violations"] else (True if self.configured else None)
        report["status"] = "FAIL" if report["violations"] else ("PASS" if self.configured else "NOT_CONFIGURED")
        report["candidate_status"] = "REJECTED_BY_JOINT_DERIVATIVES" if report["violations"] else "UNREVIEWED"
        return report


def _rms(values):
    # Scaling prevents overflow from squaring otherwise finite derivatives.
    scale = float(np.max(np.abs(values)))
    return scale * float(np.sqrt(np.mean(np.square(values / scale)))) if scale else 0.0
