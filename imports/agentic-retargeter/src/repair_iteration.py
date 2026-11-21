"""Explicit smooth-seed iteration without replacing the original source prior.

The guard measures discrete native integer samples, including derivative
stencils which cross edited/frozen joins. It is not a continuous-time jerk
certificate: LINEAR curves generally have velocity discontinuities at keys.
Heavy proposal runtimes are deliberately absent from this module.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit import native_io
from .contact_edit.pose import _keys, _weight


def derivative(values, timestamps, order):
    """Midpoint finite differences on the actual bound timestamp grid."""
    values, times = np.asarray(values, dtype=float), np.asarray(timestamps, dtype=float)
    for _ in range(order):
        delta = np.diff(times)
        values = np.diff(values, axis=0) / delta.reshape((-1,) + (1,) * (values.ndim - 1))
        times = (times[:-1] + times[1:]) * 0.5
    return values


def stencil_mask(editable, order):
    """Include every derivative stencil touching an editable integer pose."""
    editable = np.asarray(editable, dtype=bool)
    return np.array([np.any(editable[f:f + order + 1])
                     for f in range(len(editable) - order)], dtype=bool)


def _stats(result, prefix, values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return
    amplitudes = np.abs(values) if values.ndim == 1 else np.linalg.norm(values, axis=-1)
    result[prefix + "/rms"] = float(np.sqrt(np.mean(amplitudes ** 2)))
    result[prefix + "/peak"] = float(np.max(amplitudes))


def _candidate_state(key):
    while key.startswith("parent_"):
        key = key[len("parent_"):]
    return key in native_io.CANDIDATE_STATE_FIELDS or key == "fk_cache_valid"


class RepairIteration:
    """Validated immutable seed, source binding and independent smoothness gate."""

    def __init__(self, context, config_dir):
        self.context = context
        config = context.config["iteration"]
        _keys(config, {"seed_npz", "seed_sha256", "wrist_bodies"},
              {"preserve_smoothness", "relative_tolerance", "absolute_tolerance",
               "derivative_weight"}, "iteration")
        if not isinstance(config["seed_npz"], str) or not config["seed_npz"]:
            raise ValueError("iteration seed_npz must name an explicit native archive")
        if not isinstance(config["seed_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", config["seed_sha256"]):
            raise ValueError("iteration seed_sha256 must be a lowercase SHA256")
        self.path = (Path(config_dir) / config["seed_npz"]).resolve(strict=True)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != config["seed_sha256"]:
            raise ValueError("Iteration seed hash mismatch")
        self.preserve_smoothness = config.get("preserve_smoothness", True)
        if type(self.preserve_smoothness) is not bool:
            raise ValueError("preserve_smoothness must be boolean")
        self.relative_tolerance = _weight(config.get("relative_tolerance", 0.0), "relative_tolerance")
        self.absolute_tolerance = _weight(config.get("absolute_tolerance", 0.0), "absolute_tolerance")
        self.derivative_weight = _weight(config.get("derivative_weight", 1.0), "derivative_weight")
        names = config["wrist_bodies"]
        if not isinstance(names, list) or not names or any(not isinstance(x, str) for x in names) or len(names) != len(set(names)):
            raise ValueError("wrist_bodies must contain unique native body names")
        self.wrists = {}
        for name in names:
            body = mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body <= 0:
                raise ValueError("Unknown native wrist body: " + name)
            self.wrists[name] = body
        seed = native_io.load_npz(self.path)
        q = native_io._native_qpos(seed)
        native_io._validate_native_binding(seed, native_io._model_joints(context.model_snapshot))
        if q.shape != context.parent["qpos"].shape or q.dtype != context.parent["qpos"].dtype:
            raise ValueError("Iteration seed native dimensions/dtype differ from original")
        excluded = native_io.INVALIDATED_FIELDS | set(native_io._declared_derived_fields(context.config)) | {"qpos"}
        for key in context.parent:
            if key in excluded or _candidate_state(key):
                continue
            if key not in seed or not np.array_equal(seed[key], context.parent[key]):
                raise ValueError("Iteration seed immutable metadata differs: " + key)
        # Additional protected metadata cannot introduce a second source/model.
        for key in set(seed) - set(context.parent):
            if not _candidate_state(key) and (key in native_io.PROTECTED_FIELDS or key.startswith(native_io.PROTECTED_PREFIXES)):
                raise ValueError("Iteration seed adds unbound source metadata: " + key)
        if not np.array_equal(q[~context.mask], context.parent["qpos"][~context.mask]):
            raise ValueError("Iteration seed changes frozen native samples")
        values = q[np.ix_(context.frames, context.qids)]
        if np.any(values < context.lower) or np.any(values > context.upper):
            raise ValueError("Iteration seed exceeds original native correction corridor")
        self.qpos = q.copy()
        self.qpos.setflags(write=False)
        self.timestamps = np.asarray(context.parent.get("timestamps", np.arange(context.T) / context.fps), dtype=float).copy()
        if context.T < 4:
            raise ValueError("Smooth-seed iteration requires four native timestamps for jerk")
        self.editable = np.any(context.mask, axis=1)
        # A velocity jump at key k has support [k-1,k,k+1]. Check each
        # editable/frozen transition, including protected interior segments.
        self.joins = sorted({k for f in np.flatnonzero(self.editable[1:] != self.editable[:-1])
                             for k in (int(f), int(f + 1)) if 0 < k < context.T - 1})
        self.seed_metrics = self.metrics(self.qpos)
        context.bindings["iteration_seed_npz"] = {"path": str(self.path), "sha256": self.sha256}

    def metrics(self, qpos):
        c = self.context
        result = {}
        for column, name in zip(c.qids, c.names):
            series = qpos[:, column]
            for order, label in ((1, "velocity_rad_s"), (2, "acceleration_rad_s2"), (3, "jerk_rad_s3")):
                _stats(result, "joint/" + name + "/" + label,
                       derivative(series, self.timestamps, order)[stencil_mask(self.editable, order)])
            velocity = derivative(series, self.timestamps, 1)
            _stats(result, "joint/" + name + "/join_velocity_jump_rad_s",
                   np.array([velocity[k] - velocity[k - 1] for k in self.joins]))
        positions = {name: [] for name in self.wrists}
        rotations = {name: [] for name in self.wrists}
        for q in qpos:
            c.forward(q)
            for name, body in self.wrists.items():
                positions[name].append(c.data.xpos[body].copy())
                rotations[name].append(c.data.xmat[body].reshape(3, 3).copy())
        dt = np.diff(self.timestamps)
        mid = (self.timestamps[1:] + self.timestamps[:-1]) / 2
        for name in self.wrists:
            R = np.asarray(rotations[name])
            # Spatial angular velocity: every vector is in the same world basis.
            omega = Rotation.from_matrix(R[1:] @ R[:-1].swapaxes(-1, -2)).as_rotvec() / dt[:, None]
            prefix = "wrist/" + name + "/"
            _stats(result, prefix + "angular_velocity_rad_s", omega[stencil_mask(self.editable, 1)])
            for order, label in ((1, "angular_acceleration_rad_s2"), (2, "angular_jerk_rad_s3")):
                _stats(result, prefix + label,
                       derivative(omega, mid, order)[stencil_mask(self.editable, order + 1)])
            _stats(result, prefix + "position_acceleration_m_s2",
                   derivative(positions[name], self.timestamps, 2)[stencil_mask(self.editable, 2)])
            velocity = derivative(positions[name], self.timestamps, 1)
            _stats(result, prefix + "position_velocity_m_s", velocity[stencil_mask(self.editable, 1)])
            for data, label in ((omega, "join_angular_velocity_jump_rad_s"),
                                (velocity, "join_position_velocity_jump_m_s")):
                _stats(result, prefix + label, np.array([data[k] - data[k - 1] for k in self.joins]))
        return result

    def compare(self, qpos):
        candidate = self.metrics(qpos)
        rows, failures = [], []
        for name, reference in self.seed_metrics.items():
            # Fixed double-precision arithmetic allowance, never a configurable
            # motion tolerance. All metric values themselves stay unrounded.
            roundoff = 1e-8 * max(1.0, abs(reference))
            limit = reference * (1 + self.relative_tolerance) + self.absolute_tolerance
            value = candidate[name]
            passed = bool(value <= limit + roundoff)
            rows.append(dict(name=name, seed=reference, candidate=value, limit=limit,
                             roundoff=roundoff, passed=passed))
            if self.preserve_smoothness and not passed:
                failures.append(dict(kind="smoothness", metric=name, value=value,
                                     limit=limit, seed_value=reference,
                                     roundoff=roundoff, affected_by_edit=True))
        report = dict(configured=True, preserve_smoothness=self.preserve_smoothness,
                      passed=not failures if self.preserve_smoothness else None,
                      seed_sha256=self.sha256, metric_rows=rows,
                      relative_tolerance=self.relative_tolerance,
                      absolute_tolerance=self.absolute_tolerance,
                      timestamps=self.timestamps.tolist(),
                      editable_frames=np.flatnonzero(self.editable).tolist(),
                      join_key_frames=self.joins,
                      scope="Native integer-grid derivative stencils touching editable poses, including frozen joins",
                      limitations=["Discrete finite differences do not certify continuous-time jerk or LINEAR interpolation smoothness.",
                                   "Wrist orientation uses world-frame SO(3) increments, not Euler or named wrist hinge proxies."])
        return report, failures

    def derivative_terms(self, qpos):
        """Linear seed-relative hinge residuals and dense Jacobian for CPU solve.

        Normalized acceleration/jerk differences guide proposals. Nonlinear
        native wrist metrics remain an independent rejection gate.
        """
        c = self.context
        width = len(c.frames) * len(c.qids)
        delta = qpos[:, c.qids] - self.qpos[:, c.qids]
        mapping = np.zeros((c.T, len(c.qids), width))
        for i, f in enumerate(c.frames):
            mapping[f, :, i * len(c.qids):(i + 1) * len(c.qids)] = np.eye(len(c.qids))
        terms = []
        for order in (2, 3):
            selected = stencil_mask(self.editable, order)
            reference = derivative(self.qpos[:, c.qids], self.timestamps, order)[selected]
            scale = np.maximum(np.sqrt(np.mean(reference ** 2, axis=0)), 1.0)
            value = derivative(delta, self.timestamps, order)[selected] / scale
            jac = derivative(mapping, self.timestamps, order)[selected] / scale[None, :, None]
            terms.append((self.derivative_weight * value.ravel(),
                          self.derivative_weight * jac.reshape(-1, width)))
        return terms
