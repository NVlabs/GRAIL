"""Hard finite-difference velocity gate over every native hinge and interval.

This is a kinematic acceptance check, not a motor specification or continuous
curve certificate. Bounded hinge values are never unwrapped. The gate includes
frozen samples and phase joins; editability only classifies failures.

Cheap diagnostic CLI (no geometry queries)::

    python -m agentic_retargeter.joint_velocity --config repair.json \
        --trajectory candidate.npz --out velocity.json

The JSON config supplies model_xml, trajectory_npz and temporal. Explicit
timestamps use their actual intervals; unlike the stricter repair archive
contract, this standalone diagnostic also supports nonuniform timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .contact_edit import native_io
from .contact_edit.pose import _keys, _model_dependencies, _sha, _weight


TEMPORAL_KEYS = {
    "max_step_rad", "max_acceleration_rad_s2", "max_velocity_rad_s",
    "joint_velocity_limits_rad_s",
}


def snapshot_from_model(model):
    """Extract the same joint binding used by native edit/audit tools."""
    return dict(
        joint_names=np.array([model.joint(i).name for i in range(model.njnt)]),
        jnt_type=model.jnt_type.copy(),
        jnt_limited=model.jnt_limited.copy(),
        jnt_range=model.jnt_range.copy(),
        jnt_qposadr=model.jnt_qposadr.copy(),
    )


class JointVelocityGate:
    """Validate an explicit speed policy and measure all native hinges.

    ``temporal.max_velocity_rad_s`` is the required positive global limit.
    Optional ``temporal.joint_velocity_limits_rad_s`` replaces it for named
    hinges. No implicit/default acceptance limit is assigned by this class.
    """

    def __init__(self, temporal, model_snapshot):
        _keys(temporal, set(), TEMPORAL_KEYS, "temporal")
        for key in TEMPORAL_KEYS - {"joint_velocity_limits_rad_s"}:
            if key in temporal:
                _weight(temporal[key], key, positive=True)
        self.configured = "max_velocity_rad_s" in temporal
        overrides = temporal.get("joint_velocity_limits_rad_s", {})
        if not isinstance(overrides, dict):
            raise ValueError("joint_velocity_limits_rad_s must be a named hinge map")
        if "joint_velocity_limits_rad_s" in temporal and not self.configured:
            raise ValueError("joint_velocity_limits_rad_s requires max_velocity_rad_s")
        self.joints = native_io._model_joints(model_snapshot)
        self.hinges = [j for j in self.joints if j["joint_type"] == 3]
        if self.configured and not self.hinges:
            raise ValueError("Joint velocity gate requires at least one native hinge")
        names = {j["joint"] for j in self.hinges}
        for name, value in overrides.items():
            if name not in names:
                raise ValueError("Velocity override is not a native hinge: " + str(name))
            _weight(value, "joint velocity limit: " + name, positive=True)
        self.default_limit = float(temporal["max_velocity_rad_s"]) if self.configured else None
        self.overrides = {name: float(value) for name, value in overrides.items()}
        widths = {0: 7, 1: 4, 2: 1, 3: 1}
        self.nq = max(j["address"] + widths[j["joint_type"]] for j in self.joints)

    def evaluate(self, qpos, fps, timestamps=None, editable_mask=None):
        """Return JSON-safe diagnostics; any violation fails the whole gate."""
        q = np.asarray(qpos)
        if q.ndim != 2 or q.shape[1] != self.nq or not len(q) or q.dtype.kind != "f" or not np.isfinite(q).all():
            raise ValueError("Velocity qpos must be finite floating [frames, model nq]")
        fps_array = np.asarray(fps)
        if fps_array.shape != () or fps_array.dtype.kind not in "fiu" or not np.isfinite(fps_array).all() or float(fps_array) <= 0:
            raise ValueError("Velocity fps must be a finite positive scalar")
        fps = float(fps_array)
        if timestamps is None:
            time_values = np.arange(len(q), dtype=np.float64) / fps
            time_source = "fps"
        else:
            time_values = np.asarray(timestamps)
            if time_values.shape != (len(q),) or time_values.dtype.kind not in "fiu" or not np.isfinite(time_values).all():
                raise ValueError("Velocity timestamps must be finite and match native frames")
            time_values = time_values.astype(np.float64)
            time_source = "timestamps"
        with np.errstate(over="ignore", invalid="ignore"):
            intervals = np.diff(time_values)
        if not np.isfinite(intervals).all() or np.any(intervals <= 0):
            raise ValueError("Velocity timestamps must be strictly increasing with finite intervals")
        if editable_mask is not None:
            editable_mask = np.asarray(editable_mask)
            if editable_mask.shape != q.shape or editable_mask.dtype.kind != "b":
                raise ValueError("Velocity editable_mask must be a boolean qpos-shaped array")
        base = dict(
            schema_version=1, configured=self.configured, selected=False,
            motion_accepted=False, scope="all_native_hinges_all_consecutive_samples",
            units="rad/s", difference="(q[t+1] - q[t]) / (timestamp[t+1] - timestamp[t]); no angle unwrapping",
            frame_count=len(q), interval_count=len(intervals), hinge_count=len(self.hinges),
            excluded_joints=[dict(joint=j["joint"], native_joint_type=j["joint_type"])
                             for j in self.joints if j["joint_type"] != 3],
            time_source=time_source, fps=fps, timestamp_units="seconds",
            min_interval_s=float(intervals.min()) if len(intervals) else None,
            max_interval_s=float(intervals.max()) if len(intervals) else None,
            max_velocity_rad_s_limit=self.default_limit,
            joint_velocity_limits_rad_s=self.overrides.copy(),
            editability_supplied=editable_mask is not None,
            limitations=[
                "Finite differences over native samples; subframe curve overshoot, acceleration, contact and collision are not checked.",
                "These are configured motion acceptance limits, not certified actuator limits. Passing does not accept or select the motion.",
            ],
        )
        if not self.configured:
            return dict(base, status="NOT_CONFIGURED", passed=None,
                        candidate_status="UNREVIEWED", per_joint=[], violations=[])
        if len(q) < 2:
            raise ValueError("Joint velocity gate requires at least two native samples")
        q = q.astype(np.float64)
        addresses = [j["address"] for j in self.hinges]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            delta = np.diff(q[:, addresses], axis=0)
            velocity = delta / intervals[:, None]
            inverse_intervals = 1.0 / intervals
        if not np.isfinite(velocity).all() or not np.isfinite(inverse_intervals).all():
            raise ValueError("Joint velocity computation produced nonfinite values")
        peaks, violations = [], []
        for column, joint in enumerate(self.hinges):
            name, address = joint["joint"], joint["address"]
            limit = self.overrides.get(name, self.default_limit)
            speed = np.abs(velocity[:, column])
            peak_frame = int(np.argmax(speed))
            bad = np.flatnonzero(speed > limit + 1e-9)
            peaks.append(dict(
                joint=name, qpos_address=address, native_limited=joint["limited"],
                limit_rad_s=limit, max_velocity_rad_s=float(speed[peak_frame]),
                max_step_rad=float(np.max(np.abs(delta[:, column]))),
                peak_support_frames=[peak_frame, peak_frame + 1],
                peak_timestamp_s=time_values[peak_frame:peak_frame + 2].tolist(),
                violation_count=len(bad), passed=not len(bad),
            ))
            for frame in bad.tolist():
                support = [frame, frame + 1]
                editable = ([f for f in support if editable_mask[f, address]]
                            if editable_mask is not None else None)
                violations.append(dict(
                    kind="temporal", metric="max_velocity_rad_s", units="rad/s",
                    value=float(speed[frame]), signed_velocity_rad_s=float(velocity[frame, column]),
                    limit=limit, window=[0, len(q) - 1], joint=name,
                    qpos_address=address, support_frames=support,
                    timestamps_s=time_values[frame:frame + 2].tolist(),
                    interval_s=float(intervals[frame]), coefficients=[-1, 1],
                    scale=float(inverse_intervals[frame]),
                    editable_support_frames=editable,
                    affected_by_edit=bool(editable) if editable is not None else None,
                ))
        passed = not violations
        return dict(base, status="PASS" if passed else "FAIL", passed=passed,
                    candidate_status="UNREVIEWED" if passed else "REJECTED_BY_JOINT_VELOCITY",
                    max_velocity_rad_s=float(np.max(np.abs(velocity))), per_joint=peaks,
                    violation_count=len(violations), violations=violations)


def run(config_path, output_path, trajectory_path=None, editable_mask_path=None):
    """Write a new diagnostic report without importing geometry/audit code."""
    import mujoco

    config_path = Path(config_path).resolve(strict=True)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite diagnostic: " + str(output_path))
    config = json.loads(config_path.read_text())
    model_path = (config_path.parent / config["model_xml"]).resolve(strict=True)
    trajectory_path = (Path(trajectory_path).resolve(strict=True) if trajectory_path else
                       (config_path.parent / config["trajectory_npz"]).resolve(strict=True))
    native = native_io.load_npz(trajectory_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    snapshot = snapshot_from_model(model)
    native_io._validate_native_binding(native, native_io._model_joints(snapshot))
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, compiled)
    compiled_hash = hashlib.sha256(compiled.tobytes()).hexdigest()
    if "compiled_model_sha256" in native and str(native["compiled_model_sha256"].item()) != compiled_hash:
        raise ValueError("Compiled model hash differs from native binding")
    mask = None
    if editable_mask_path:
        mask = np.load(editable_mask_path, allow_pickle=False)
    report = JointVelocityGate(config.get("temporal", {}), snapshot).evaluate(
        native["qpos"], native["fps"], native.get("timestamps"), mask,
    )
    report["bindings"] = {name: dict(path=str(path), sha256=_sha(path)) for name, path in
                          [("model_xml", model_path), ("trajectory_npz", trajectory_path),
                           ("config", config_path)]}
    if editable_mask_path:
        mask_path = Path(editable_mask_path).resolve(strict=True)
        report["bindings"]["editable_mask"] = dict(path=str(mask_path), sha256=_sha(mask_path))
    report["compiled_model_sha256"] = compiled_hash
    report["model_dependencies"] = {str(p): _sha(p) for p in _model_dependencies(model_path)}
    report["code_sha256"] = {str(p): _sha(p) for p in [Path(__file__), Path(native_io.__file__)]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory", help="Override native candidate only; not the bound model")
    parser.add_argument("--editable-mask", help="Optional boolean NPY [frames,nq] for failure classification")
    parser.add_argument("--out", required=True, help="New JSON diagnostic report")
    args = parser.parse_args(argv)
    report = run(args.config, args.out, args.trajectory, args.editable_mask)
    print(json.dumps(dict(status=report["status"], passed=report["passed"],
                          violations=len(report["violations"]), report=args.out)))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
