"""Native motion diagnostics; review warnings never change acceptance gates."""

from pathlib import Path
import argparse
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit import native_io
from .contact_edit.pose import _keys, _sha, _model_dependencies
from .repair_iteration import derivative


def timestamps(native):
    q = native_io._native_qpos(native)
    t = np.asarray(
        native.get("timestamps", np.arange(len(q)) / float(native["fps"])), float
    )
    if t.shape != (len(q),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Expected finite strictly increasing native timestamps")
    return t


def statistics(points, times):
    result = {}
    for order, name, unit in [(1, "velocity", "m_s"), (2, "acceleration", "m_s2")]:
        values = derivative(points, times, order)
        amplitude = np.linalg.norm(values, axis=-1)
        if not len(amplitude):
            result[name] = {"status": "INSUFFICIENT_SAMPLES"}
            continue
        k = int(amplitude.argmax())
        result[name] = dict(
            rms=float(np.sqrt(np.mean(amplitude**2))),
            peak=float(amplitude[k]),
            unit=unit,
            peak_support_frames=list(range(k, k + order + 1)),
            peak_support_times_s=times[k : k + order + 1].tolist(),
        )
    return result


class MotionDiagnostics:
    def __init__(self, model, parent, config, directory="."):
        _keys(
            config,
            {"bodies"},
            {"baseline_npz", "human_reference", "review_thresholds"},
            "diagnostics",
        )
        names = config["bodies"]
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError("diagnostics bodies must be unique native body names")
        self.model, self.times = model, timestamps(parent)
        self.names, self.ids = names, [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names
        ]
        if any(b <= 0 for b in self.ids):
            raise ValueError("Unknown diagnostics body")
        self.bindings = {}
        self.human = {}
        self.human_stats = {}
        baseline = parent
        if "baseline_npz" in config:
            path = self.bind("diagnostics_baseline", config["baseline_npz"], directory)
            baseline = native_io.load_npz(path)
            native_io._validate_native_binding(
                baseline,
                [
                    dict(
                        joint=model.joint(j).name,
                        address=int(model.jnt_qposadr[j]),
                        joint_type=int(model.jnt_type[j]),
                    )
                    for j in range(model.njnt)
                ],
            )
            if not np.array_equal(timestamps(baseline), self.times):
                raise ValueError(
                    "Diagnostics baseline timing differs; align explicitly upstream"
                )
        self.base_p, self.base_r = self.positions(native_io._native_qpos(baseline))
        self.base_stats = [
            statistics(self.base_p[:, i], self.times) for i in range(len(names))
        ]
        self.thresholds = dict(config.get("review_thresholds", {}))
        _keys(
            self.thresholds,
            set(),
            {"position_deviation_m", "acceleration_peak_m_s2"},
            "review thresholds",
        )
        if any(
            type(v) not in (int, float) or not np.isfinite(v) or v <= 0
            for v in self.thresholds.values()
        ):
            raise ValueError("Review thresholds must be positive finite values")
        ref = config.get("human_reference")
        if ref is not None:
            _keys(
                ref,
                {"npz", "positions_key", "names_key", "body_map"},
                {"timestamps_key", "fps"},
                "human reference",
            )
            if not isinstance(ref["body_map"], dict) or set(ref["body_map"]) != set(
                names
            ):
                raise ValueError(
                    "Human body_map must explicitly map every diagnostic body"
                )
            path = self.bind("diagnostics_human", ref["npz"], directory)
            h = native_io.load_npz(path)
            pts = np.asarray(h[ref["positions_key"]], float)
            hnames = h[ref["names_key"]].tolist()
            if (
                len(set(hnames)) != len(hnames)
                or pts.shape != (len(self.times), len(hnames), 3)
                or not np.isfinite(pts).all()
            ):
                raise ValueError("Invalid human world-position series")
            if "timestamps_key" in ref:
                ht = h[ref["timestamps_key"]]
            elif (
                "fps" in ref
                and type(ref["fps"]) in (int, float)
                and np.isfinite(ref["fps"])
                and ref["fps"] > 0
            ):
                ht = np.arange(len(pts)) / ref["fps"]
            else:
                raise ValueError("Declare human timestamps_key or positive fps")
            if np.shape(ht) != self.times.shape or not np.allclose(
                ht, self.times, rtol=0, atol=1e-8
            ):
                raise ValueError("Human timing differs; align explicitly upstream")
            for body, name in ref["body_map"].items():
                if name not in hnames:
                    raise ValueError("Unknown human body: " + name)
                self.human[body] = pts[:, hnames.index(name)].copy()
                self.human_stats[body] = statistics(self.human[body], self.times)

    def bind(self, key, path, directory):
        p = (Path(directory) / path).resolve(strict=True)
        self.bindings[key] = dict(path=str(p), sha256=_sha(p))
        return p

    def positions(self, q):
        if q.shape != (len(self.times), self.model.nq) or not np.isfinite(q).all():
            raise ValueError("Invalid diagnostic native pose series")
        data = mujoco.MjData(self.model)
        p = []
        r = []
        for row in q:
            data.qpos[:] = row
            mujoco.mj_forward(self.model, data)
            p.append(data.xpos[self.ids].copy())
            r.append(data.xmat[self.ids].reshape(-1, 3, 3).copy())
        return np.array(p), np.array(r)

    def compare(self, q):
        for binding in self.bindings.values():
            if _sha(binding["path"]) != binding["sha256"]:
                raise ValueError("Diagnostic input changed")
        p, r = self.positions(q)
        rows = {}
        warnings = []
        for i, name in enumerate(self.names):
            stats = statistics(p[:, i], self.times)
            distance = np.linalg.norm(p[:, i] - self.base_p[:, i], axis=-1)
            angular = np.rad2deg(
                Rotation.from_matrix(
                    r[:, i] @ self.base_r[:, i].transpose(0, 2, 1)
                ).magnitude()
            )
            row = dict(
                candidate=stats,
                baseline=self.base_stats[i],
                position_deviation_m=dict(
                    rms=float(np.sqrt(np.mean(distance**2))),
                    maximum=float(distance.max()),
                    worst_frame=int(distance.argmax()),
                ),
                rotation_deviation_deg=dict(
                    maximum=float(angular.max()), worst_frame=int(angular.argmax())
                ),
            )
            if name in self.human_stats:
                row["human_reference"] = self.human_stats[name]
                if len(self.times) > 2:
                    row["acceleration_rms_excess_m_s2"] = max(
                        0.0,
                        stats["acceleration"]["rms"]
                        - self.human_stats[name]["acceleration"]["rms"],
                    )
            values = {
                "position_deviation_m": (
                    float(distance.max()),
                    [int(distance.argmax())],
                )
            }
            if len(self.times) > 2:
                values["acceleration_peak_m_s2"] = (
                    stats["acceleration"]["peak"],
                    stats["acceleration"]["peak_support_frames"],
                )
            for key, (value, frames) in values.items():
                if key in self.thresholds and value > self.thresholds[key]:
                    warnings.append(
                        dict(
                            body=name,
                            metric=key,
                            value=value,
                            review_threshold=self.thresholds[key],
                            support_frames=frames,
                        )
                    )
            rows[name] = row
        return dict(
            status="DIAGNOSTIC_ONLY",
            affects_acceptance=False,
            bodies=rows,
            review_warnings=warnings,
            bindings=self.bindings,
            timestamps_s=self.times.tolist(),
            scope="Full native integer sequence including editable/frozen joins; midpoint finite differences in seconds. RMS excess is max(0, robot RMS - human RMS). No ratio or implicit threshold.",
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config).resolve(strict=True)
    config = json.loads(path.read_text())
    _keys(
        config,
        {"model_xml", "trajectory_npz", "diagnostics"},
        set(),
        "diagnostics config",
    )
    model_path = (path.parent / config["model_xml"]).resolve(strict=True)
    parent_path = (path.parent / config["trajectory_npz"]).resolve(strict=True)
    native = native_io.load_npz(parent_path)
    candidate_path = Path(args.candidate).resolve(strict=True)
    candidate = native_io.load_npz(candidate_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joints = [
        dict(
            joint=model.joint(j).name,
            address=int(model.jnt_qposadr[j]),
            joint_type=int(model.jnt_type[j]),
        )
        for j in range(model.njnt)
    ]
    for n in (native, candidate):
        native_io._validate_native_binding(n, joints)
    if not np.array_equal(timestamps(native), timestamps(candidate)):
        raise ValueError("Candidate timing differs")
    bindings = {
        str(p): _sha(p)
        for p in [path, parent_path, candidate_path, *_model_dependencies(model_path)]
    }
    d = MotionDiagnostics(model, native, config["diagnostics"], path.parent)
    report = d.compare(candidate["qpos"])
    if any(_sha(p) != sha for p, sha in bindings.items()):
        raise ValueError("Input changed during diagnostics")
    report["input_sha256"] = bindings
    report["implementation_sha256"] = _sha(Path(__file__))
    with Path(args.output).open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                status=report["status"],
                review_warnings=len(report["review_warnings"]),
                output=args.output,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
