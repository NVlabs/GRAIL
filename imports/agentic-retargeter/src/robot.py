"""Conservative, explicit-correspondence native robot trajectory refinement.

The solver uses original object triangle/barycentric targets and native MuJoCo
Jacobians. It never searches for a grasp or infers an occluded contact. A result
is an experimental candidate, even when its native kinematic checks pass.
Original-mesh intersections, source-video alignment and visual review remain
mandatory external checks; MuJoCo collision proxies do not certify those.

Configuration and correspondence schemas are documented in the skill reference.
All paths in the JSON configuration resolve relative to that file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from agentic_retargeter.contact import surface_samples
from agentic_retargeter.candidate import prepare_candidate, validate_derived_fields


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _finite(name, value, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or (shape is not None and value.shape != shape):
        raise ValueError(f"Invalid {name}: expected finite array with shape {shape}")
    return value


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def source_fps(timestamps, fps):
    """Reject resampling, mislabeled frame rate, or insufficient transition coverage."""
    t = _finite("timestamps", timestamps)
    if t.ndim != 1 or len(t) < 4 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Need at least four frames and a positive source fps")
    if not np.allclose(np.diff(t), 1.0 / fps, rtol=1e-6, atol=1e-9):
        raise ValueError("timestamps must retain a uniform source timeline matching fps")
    return float(fps)


def derivatives(sequence, fps):
    return [np.diff(sequence, n=order, axis=0) * fps**order for order in (1, 2, 3)]


def transition_caps(reference, fps, policy):
    """Per-component, per-transition caps; no global peak can excuse another joint."""
    return _derivative_caps(derivatives(reference, fps), policy)


def _derivative_caps(values, policy):
    floor = _finite("cap floor", policy["floor"], (3,))
    slack = _finite("cap slack", policy.get("slack", [0, 0, 0]), (3,))
    multiplier = float(policy.get("reference_multiplier", 1.0))
    if np.any(floor < 0) or np.any(slack < 0) or not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("Temporal allowances must be nonnegative")
    maxima = np.asarray(policy.get("absolute_max", [np.inf] * 3), dtype=float)
    if maxima.shape != (3,) or np.isnan(maxima).any() or np.any(maxima < floor):
        raise ValueError("absolute_max must be at least floor for each derivative")
    values = [_finite("reference derivative", value) for value in values]
    return [
        _finite(
            "computed transition cap",
            np.minimum(maxima[i], np.maximum(floor[i], multiplier * np.abs(d) + slack[i])),
        )
        for i, d in enumerate(values)
    ]


def transition_report(sequence, fps, caps, tolerance=1e-7):
    return _derivative_report(derivatives(sequence, fps), caps, tolerance)


def _derivative_report(values, caps, tolerance=1e-7):
    if len(values) != 3 or len(caps) != 3:
        raise ValueError("Velocity, acceleration and jerk must all be checked")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Temporal comparison tolerance must be finite and nonnegative")
    report = {}
    for order, (value, cap) in enumerate(zip(values, caps), start=1):
        value = _finite("candidate derivative", value)
        cap = _finite("transition cap", cap, value.shape)
        if np.any(cap < 0):
            raise ValueError("Transition caps must be nonnegative")
        excess = np.maximum(np.abs(value) - cap, 0)
        index = np.unravel_index(int(np.argmax(excess)), excess.shape)
        report[str(order)] = {
            "violation_count": int(np.sum(excess > tolerance)),
            "max_excess": float(excess[index]),
            "max_abs": float(np.abs(value).max()),
            "worst_index": list(map(int, index)),
        }
    return report


def _cap_penalty(sequence, fps, caps, scales):
    loss = 0.0
    gradient = np.zeros_like(sequence)
    for order, (d, cap, scale) in enumerate(zip(derivatives(sequence, fps), caps, scales), start=1):
        excess = np.maximum(np.abs(d) - cap, 0)
        loss += float(np.sum((excess / scale) ** 2))
        dg = 2 * excess * np.sign(d) / scale**2 * fps**order
        for offset in range(order + 1):
            gradient[offset : offset + len(d)] += (
                (-1) ** (order - offset) * math.comb(order, offset) * dg
            )
    return loss, gradient


def _rotations(name, values, shape):
    values = _finite(name, values, shape)
    if not np.allclose(values @ values.swapaxes(-1, -2), np.eye(3), atol=2e-5):
        raise ValueError(f"{name} is not orthonormal")
    if np.any(np.linalg.det(values) < 0.99998):
        raise ValueError(f"{name} is not a proper rotation")
    return values


def _model_dependencies(model_xml):
    """Bind MJCF includes and file assets, in addition to the compiled model hash."""
    model_xml = Path(model_xml).resolve()
    pending, seen, roots = [model_xml], set(), []
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ET.parse(path).getroot()
        roots.append(tree)
        for include in tree.iter("include"):
            pending.append((path.parent / include.attrib["file"]).resolve())
    compiler = {}
    for tree in roots:
        for item in tree.iter("compiler"):
            compiler.update(item.attrib)
    for tree in roots:
        for asset in tree.iter():
            if asset.tag not in ("mesh", "texture", "hfield", "skin") or "file" not in asset.attrib:
                continue
            directory = compiler.get("assetdir", "")
            if asset.tag == "mesh":
                directory = compiler.get("meshdir", directory)
            elif asset.tag == "texture":
                directory = compiler.get("texturedir", directory)
            name = asset.attrib["file"]
            if compiler.get("strippath", "false") == "true":
                name = Path(name).name
            seen.add((model_xml.parent / directory / name).resolve())
    missing = [str(path) for path in seen if not path.is_file()]
    if missing:
        raise ValueError(f"Cannot bind native model asset dependencies: {missing}")
    return sorted(seen)


class NativeRefiner:
    """Full-sequence bounded native hinge optimization with frozen source bindings."""

    def __init__(self, config, config_dir):
        import mujoco

        self.mj = mujoco
        self.config = config
        validate_derived_fields(config.get("derived_fields", []))
        for name, default, minimum in (
            ("max_iterations", 100, 1),
            ("max_feasibility_backtracks", 20, 0),
        ):
            value = config.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for key, value in config.get("weights", {}).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid nonnegative objective weight: {key}")
        scales = _finite("temporal_scales", config.get("temporal_scales", [1, 20, 500]), (3,))
        if np.any(scales <= 0):
            raise ValueError("Temporal penalty scales must be positive")
        self.paths = {
            key: (Path(config_dir) / config[key]).resolve()
            for key in ("model_xml", "trajectory", "object_mesh", "targets")
        }
        self.input_hashes = {key: _hash(path) for key, path in self.paths.items()}
        for key, expected in config.get("expected_input_sha256", {}).items():
            if key not in self.input_hashes or self.input_hashes[key] != expected:
                raise ValueError(f"Input hash does not match the selected parent binding: {key}")
        self.model_dependencies = _model_dependencies(self.paths["model_xml"])
        self.dependency_hashes = {str(path): _hash(path) for path in self.model_dependencies}
        self.model = mujoco.MjModel.from_xml_path(str(self.paths["model_xml"]))
        self.data = mujoco.MjData(self.model)
        self.source = _load(self.paths["trajectory"])
        if "compiled_model_sha256" in self.source:
            if (
                self.source["compiled_model_sha256"].shape != ()
                or str(self.source["compiled_model_sha256"].item()) != self.model_assets()
            ):
                raise ValueError("Native source was exported from a different compiled model")
        self.reference = _finite("qpos", self.source["qpos"])
        if self.reference.ndim != 2:
            raise ValueError("qpos must be a frame by native-coordinate array")
        self.frames, nq = self.reference.shape
        if nq != self.model.nq:
            raise ValueError("Native model and trajectory qpos sizes differ")
        self.fps = source_fps(self.source["timestamps"], float(self.source["fps"]))
        if len(self.source["timestamps"]) != self.frames:
            raise ValueError("Native timestamps and qpos frame counts differ")
        # Named bindings catch accidentally loading another robot with the same nq.
        for jid, kind in enumerate(self.model.jnt_type):
            address = int(self.model.jnt_qposadr[jid])
            start = address + 3 if kind == mujoco.mjtJoint.mjJNT_FREE else address
            if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                if not np.allclose(
                    np.linalg.norm(self.reference[:, start : start + 4], axis=1), 1, atol=1e-6
                ):
                    raise ValueError("Native quaternion is not unit length")
        if self.source["dof_names"].ndim != 1 or self.source["dof_names"].dtype.kind not in "US":
            raise ValueError("dof_names must be a string vector")
        names = self.source["dof_names"].astype(str).tolist()
        raw_addresses = self.source["qposadr"]
        if raw_addresses.shape != (len(names),) or raw_addresses.dtype.kind not in "iu":
            raise ValueError("qposadr must be an integer vector matching dof_names")
        addresses = raw_addresses.astype(int).tolist()
        if len(set(names)) != len(names) or len(names) != len(addresses):
            raise ValueError("Invalid native dof name/address bindings")
        scalar_jids = np.flatnonzero(
            np.isin(self.model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE])
        )
        expected_names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in scalar_jids
        }
        if set(names) != expected_names:
            raise ValueError("Native scalar joint binding must cover the complete model")
        for name, address in zip(names, addresses):
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            if int(self.model.jnt_qposadr[jid]) != address:
                raise ValueError(f"Native qpos binding mismatch: {name}")
        joints = list(config["editable_joints"])
        if not joints or len(set(joints)) != len(joints):
            raise ValueError("editable_joints must be a nonempty unique list")
        forbidden = ("hip", "knee", "ankle", "root", "pelvis")
        for name in joints:
            if name not in names or any(part in name for part in forbidden):
                raise ValueError(f"Only declared arm, waist and finger hinges may vary: {name}")
            if not any(part in name for part in ("shoulder", "elbow", "wrist", "hand", "waist")):
                raise ValueError(f"Unrecognized editable joint role: {name}")
        ids = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joints])
        if np.any(self.model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Only scalar native hinge joints are supported")
        self.joints = joints
        self.qids = self.model.jnt_qposadr[ids]
        self.dids = self.model.jnt_dofadr[ids]
        self.frozen = np.setdiff1d(np.arange(nq), self.qids)
        self.original = self.reference[:, self.qids].copy()
        self.delta = _finite("max_delta_rad", config["max_delta_rad"])
        self.delta = np.broadcast_to(self.delta, (len(ids),))
        if np.any(self.delta <= 0):
            raise ValueError("max_delta_rad must be positive")
        lower = np.where(self.model.jnt_limited[ids], self.model.jnt_range[ids, 0], -np.inf)
        upper = np.where(self.model.jnt_limited[ids], self.model.jnt_range[ids, 1], np.inf)
        self.lower = np.maximum(self.original - self.delta, lower)
        self.upper = np.minimum(self.original + self.delta, upper)
        if np.any(self.lower > self.upper):
            raise ValueError("Initial native pose is outside the declared repair envelope")
        self.obj_R = _rotations("obj_R", self.source["obj_R"], (self.frames, 3, 3))
        self.obj_t = _finite("obj_t", self.source["obj_t"], (self.frames, 3))
        self._targets()
        self.caps = transition_caps(self.original, self.fps, config["joint_caps"])
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros_like(self.jacp)
        baseline = self.evaluate(self.reference)
        self.wrist_linear_caps = _derivative_caps(
            [
                np.linalg.norm(v, axis=-1)
                for v in derivatives(baseline["wrist_positions"], self.fps)
            ],
            config["wrist_linear_caps"],
        )
        self.wrist_angular_caps = self._angular_caps(baseline["wrist_rotations"])

    def _id(self, kind, name):
        result = self.mj.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Missing native model element: {name}")
        return result

    def _targets(self):
        target = _load(self.paths["targets"])
        mesh = _load(self.paths["object_mesh"])
        vertices = _finite("object vertices", mesh["vertices"])
        faces = np.asarray(mesh["faces"])
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Object vertices must be V x 3")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
            raise ValueError("Object faces must be integer original triangles")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("Invalid object face vertex index")
        scale = np.broadcast_to(_finite("object_scale", self.source["object_scale"]), (3,))
        if np.any(scale <= 0):
            raise ValueError("Object scale must be positive")
        vertices = vertices * scale
        self.wrist_names = target["wrist_body_names"].astype(str).tolist()
        self.contact_names = target["contact_body_names"].astype(str).tolist()
        required_wrists = [
            side + "_wrist_yaw_link"
            for side in ("left", "right")
            if self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, side + "_wrist_yaw_link")
            >= 0
        ]
        if not set(required_wrists).issubset(self.wrist_names):
            raise ValueError("Both native G1 wrists must be included in temporal checks")
        if not self.wrist_names or not self.contact_names:
            raise ValueError("Declare wrist checks and material contact correspondences")
        self.wrist_ids = [self._id(self.mj.mjtObj.mjOBJ_BODY, n) for n in self.wrist_names]
        self.contact_ids = [self._id(self.mj.mjtObj.mjOBJ_BODY, n) for n in self.contact_names]
        f, w, p = self.frames, len(self.wrist_ids), len(self.contact_ids)
        self.wrist_weight = _finite("wrist_weights", target["wrist_weights"], (f, w))
        self.contact_weight = _finite("contact_weights", target["contact_weights"], (f, p))
        if np.any(self.wrist_weight < 0) or np.any(self.contact_weight < 0):
            raise ValueError("Phase weights must be nonnegative")
        if not np.any(self.contact_weight > 0):
            raise ValueError("At least one observed contact correspondence must be active")
        wp = _finite("wrist_position_object", target["wrist_position_object"], (f, w, 3))
        wr = _rotations("wrist_rotation_object", target["wrist_rotation_object"], (f, w, 3, 3))
        self.wrist_p = np.einsum("fij,fwj->fwi", self.obj_R, wp) + self.obj_t[:, None]
        self.wrist_R = np.einsum("fij,fwjk->fwik", self.obj_R, wr)
        self.pad_p = _finite("contact_point_local", target["contact_point_local"], (p, 3))
        self.pad_n = _finite("contact_normal_local", target["contact_normal_local"], (p, 3))
        if not np.allclose(np.linalg.norm(self.pad_n, axis=-1), 1, atol=1e-6):
            raise ValueError("Contact pad normals must be unit outward normals")
        fi = target["contact_face_ids"]
        if (
            fi.shape != (f, p)
            or fi.dtype.kind not in "iu"
            or np.any(fi < 0)
            or np.any(fi >= len(faces))
        ):
            raise ValueError("Contact face IDs must index the original object mesh")
        bary = _finite("contact_barycentric", target["contact_barycentric"], (f, p, 3))
        if np.any(bary < 0) or not np.allclose(bary.sum(-1), 1, atol=1e-12, rtol=0):
            raise ValueError("Invalid original triangle barycentric coordinates")
        tri = vertices[faces[fi]]
        normal = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
        norm = np.linalg.norm(normal, axis=-1)
        if np.any(norm < 1e-12):
            raise ValueError("Degenerate original contact triangle")
        # Explicit signs document reviewed outward orientation on open/multi-part meshes.
        signs = _finite("contact_normal_sign", target["contact_normal_sign"], (f, p))
        if not np.isin(signs, [-1, 1]).all():
            raise ValueError("contact_normal_sign must be +1 or -1")
        normal *= (signs / norm)[..., None]
        local_p = surface_samples(vertices, faces, fi.ravel(), bary.reshape(-1, 3)).reshape(f, p, 3)
        self.contact_p = np.einsum("fij,fpj->fpi", self.obj_R, local_p) + self.obj_t[:, None]
        self.contact_n = np.einsum("fij,fpj->fpi", self.obj_R, normal)

    def _point(self, body, local, jacobian=False):
        rotation = self.data.xmat[body].reshape(3, 3)
        point = self.data.xpos[body] + rotation @ local
        if not jacobian:
            return point, rotation
        self.mj.mj_jac(self.model, self.data, self.jacp, self.jacr, point, body)
        return point, rotation, self.jacp[:, self.dids].copy(), self.jacr[:, self.dids].copy()

    def objective(self, flat):
        x = flat.reshape(self.original.shape)
        delta = x - self.original
        weights = self.config.get("weights", {})
        regularization = float(weights.get("pose", 1.0))
        loss = regularization * float(np.sum((delta / self.delta) ** 2))
        gradient = regularization * 2 * delta / self.delta**2
        for f in range(self.frames):
            self.data.qpos[:] = self.reference[f]
            self.data.qpos[self.qids] = x[f]
            self.mj.mj_forward(self.model, self.data)
            for w, body in enumerate(self.wrist_ids):
                point, rotation, jp, jr = self._point(body, np.zeros(3), True)
                wp = self.wrist_weight[f, w] * float(weights.get("wrist_position", 1000))
                wr = self.wrist_weight[f, w] * float(weights.get("wrist_orientation", 1))
                ep, er = point - self.wrist_p[f, w], rotation - self.wrist_R[f, w]
                loss += wp * (ep @ ep) + wr * np.sum(er**2)
                gradient[f] += 2 * wp * ep @ jp
                for c in range(3):
                    gradient[f] += 2 * wr * er[:, c] @ (-_skew(rotation[:, c]) @ jr)
            for p, body in enumerate(self.contact_ids):
                point, rotation, jp, jr = self._point(body, self.pad_p[p], True)
                normal = rotation @ self.pad_n[p]
                ep, en = point - self.contact_p[f, p], normal + self.contact_n[f, p]
                wp = self.contact_weight[f, p] * float(weights.get("contact_position", 1000))
                wn = self.contact_weight[f, p] * float(weights.get("contact_normal", 1))
                loss += wp * (ep @ ep) + wn * (en @ en)
                gradient[f] += 2 * wp * ep @ jp + 2 * wn * en @ (-_skew(normal) @ jr)
        cap_loss, cap_grad = _cap_penalty(
            x, self.fps, self.caps, self.config.get("temporal_scales", [1, 20, 500])
        )
        weight = float(weights.get("joint_temporal", 10))
        return float(loss + weight * cap_loss), (gradient + weight * cap_grad).ravel()

    def evaluate(self, qpos):
        positions, rotations, gaps, opposition = [], [], [], []
        for frame, q in enumerate(qpos):
            self.data.qpos[:] = q
            self.mj.mj_forward(self.model, self.data)
            wrist = [self._point(body, np.zeros(3)) for body in self.wrist_ids]
            positions.append([point for point, rotation in wrist])
            rotations.append([rotation.copy() for point, rotation in wrist])
            points = [self._point(b, p) for b, p in zip(self.contact_ids, self.pad_p)]
            gaps.append([np.linalg.norm(p - t) for (p, r), t in zip(points, self.contact_p[frame])])
            opposition.append(
                [
                    -float((r @ n) @ t)
                    for (p, r), n, t in zip(points, self.pad_n, self.contact_n[frame])
                ]
            )
        return {
            "wrist_positions": np.asarray(positions),
            "wrist_rotations": np.asarray(rotations),
            "correspondence_distance_m": np.asarray(gaps),
            "normal_opposition": np.asarray(opposition),
        }

    def _angular_derivatives(self, rotations):
        relative = rotations[1:] @ rotations[:-1].swapaxes(-1, -2)
        angular = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
        angular = angular.reshape(relative.shape[:-2] + (3,)) * self.fps
        return [
            angular,
            np.diff(angular, axis=0) * self.fps,
            np.diff(angular, n=2, axis=0) * self.fps**2,
        ]

    def _angular_caps(self, rotations):
        return _derivative_caps(
            [np.linalg.norm(v, axis=-1) for v in self._angular_derivatives(rotations)],
            self.config["wrist_angular_caps"],
        )

    def audit(self, qpos):
        qpos = _finite("audit qpos", qpos, self.reference.shape)
        evaluated = self.evaluate(qpos)
        reports = {
            "joints": transition_report(qpos[:, self.qids], self.fps, self.caps),
            "world_wrist_linear": _derivative_report(
                [
                    np.linalg.norm(v, axis=-1)
                    for v in derivatives(evaluated["wrist_positions"], self.fps)
                ],
                self.wrist_linear_caps,
            ),
        }
        reports["world_wrist_angular"] = _derivative_report(
            [
                np.linalg.norm(v, axis=-1)
                for v in self._angular_derivatives(evaluated["wrist_rotations"])
            ],
            self.wrist_angular_caps,
        )
        frozen = np.array_equal(qpos[:, self.frozen], self.reference[:, self.frozen])
        bounds = bool(
            np.all(qpos[:, self.qids] >= self.lower - 1e-9)
            and np.all(qpos[:, self.qids] <= self.upper + 1e-9)
        )
        limited = np.flatnonzero(self.model.jnt_limited)
        scalar = limited[
            np.isin(
                self.model.jnt_type[limited],
                [self.mj.mjtJoint.mjJNT_HINGE, self.mj.mjtJoint.mjJNT_SLIDE],
            )
        ]
        native_values = qpos[:, self.model.jnt_qposadr[scalar]]
        native_limits = bool(
            np.all(native_values >= self.model.jnt_range[scalar, 0] - 1e-7)
            and np.all(native_values <= self.model.jnt_range[scalar, 1] + 1e-7)
        )
        feasible = (
            frozen
            and bounds
            and native_limits
            and all(
                entry["violation_count"] == 0
                for group in reports.values()
                for entry in group.values()
            )
        )
        active = self.contact_weight > 0
        gaps = evaluated["correspondence_distance_m"][active]
        opposition = evaluated["normal_opposition"][active]
        return {
            "kinematic_feasible": feasible,
            "frozen_qpos_exact": frozen,
            "declared_joint_bounds_pass": bounds,
            "all_native_scalar_joint_limits_pass": native_limits,
            "temporal": reports,
            "active_contact_samples": int(active.sum()),
            "correspondence_mean_distance_m": float(gaps.mean()) if gaps.size else None,
            "correspondence_max_distance_m": float(gaps.max()) if gaps.size else None,
            "mean_normal_opposition": float(opposition.mean()) if opposition.size else None,
            "contact_metric_scope": "Distances to declared material correspondences, not nearest-surface gaps or penetration depths",
        }

    def model_assets(self):
        """Bind the compiled model, including included XML and native mesh content."""
        size = self.mj.mj_sizeModel(self.model)
        buffer = np.empty(size, dtype=np.uint8)
        self.mj.mj_saveModel(self.model, None, buffer)
        return hashlib.sha256(buffer.tobytes()).hexdigest()


def run(config_path, output_dir):
    """Write a reproducible experimental candidate; never replace the source NPZ."""
    config_path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    config = json.loads(config_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    solver = NativeRefiner(config, config_path.parent)
    config_sha256 = _hash(config_path)
    output.mkdir(parents=True, exist_ok=True)
    resolved_config = {**config, **{key: str(value) for key, value in solver.paths.items()}}
    (output / "config.json").write_text(json.dumps(resolved_config, indent=2) + "\n")
    (output / "source_config.json").write_bytes(config_path.read_bytes())
    np.savez_compressed(
        output / "temporal_caps.npz",
        **{f"joint_order{i + 1}": cap for i, cap in enumerate(solver.caps)},
        **{
            f"world_wrist_linear_order{i + 1}": cap
            for i, cap in enumerate(solver.wrist_linear_caps)
        },
        **{
            f"world_wrist_angular_order{i + 1}": cap
            for i, cap in enumerate(solver.wrist_angular_caps)
        },
        editable_qpos_ids=solver.qids,
        editable_joint_names=np.array(solver.joints),
        wrist_names=np.array(solver.wrist_names),
        fps=solver.fps,
    )
    history = []

    def export_candidate(filename, qpos):
        candidate, metadata = prepare_candidate(
            solver.source, qpos, derived_fields=config.get("derived_fields", [])
        )
        candidate["refinement_parent_sha256"] = np.asarray(solver.input_hashes["trajectory"])
        candidate["refinement_config_sha256"] = np.asarray(config_sha256)
        candidate["refinement_provenance_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        )
        np.savez_compressed(output / filename, **candidate)
        return candidate, metadata

    initial = solver.original.ravel()
    # Deterministic directional derivative preflight against the actual objective.
    rng = np.random.default_rng(0)
    direction = rng.normal(size=initial.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    value, gradient = solver.objective(initial)
    numeric = (
        solver.objective(initial + epsilon * direction)[0]
        - solver.objective(initial - epsilon * direction)[0]
    ) / (2 * epsilon)
    analytic = float(gradient @ direction)
    relative = abs(numeric - analytic) / max(1, abs(numeric), abs(analytic))
    if not np.isfinite(relative) or relative > 1e-4:
        raise ValueError(f"Native objective gradient preflight failed: {relative}")

    def callback(x):
        loss, _ = solver.objective(x)
        history.append({"iteration": len(history) + 1, "objective": loss})
        checkpoint = solver.reference.copy()
        checkpoint[:, solver.qids] = x.reshape(solver.original.shape)
        export_candidate("checkpoint_UNREVIEWED.npz", checkpoint)

    result = minimize(
        solver.objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(solver.lower.ravel(), solver.upper.ravel())),
        callback=callback,
        options={"maxiter": int(config.get("max_iterations", 100)), "ftol": 1e-12, "maxls": 30},
    )
    raw = solver.reference.copy()
    raw[:, solver.qids] = result.x.reshape(solver.original.shape)
    export_candidate("unfiltered_candidate_UNREVIEWED.npz", raw)
    # Backtracking is across the complete trajectory. No isolated frame insertion.
    chosen, fraction, checks = raw.copy(), 1.0, solver.audit(raw)
    for attempt in range(int(config.get("max_feasibility_backtracks", 20))):
        if checks["kinematic_feasible"]:
            break
        fraction *= 0.5
        chosen = solver.reference.copy()
        chosen[:, solver.qids] += fraction * (raw[:, solver.qids] - solver.original)
        checks = solver.audit(chosen)
    candidate_path = output / "candidate_UNREVIEWED.npz"
    expected_candidate, metadata = export_candidate(candidate_path.name, chosen)
    serialized = _load(candidate_path)
    if serialized.keys() != expected_candidate.keys() or not all(
        np.array_equal(serialized[k], v) for k, v in expected_candidate.items()
    ):
        raise ValueError("Native candidate changed during serialization")
    excluded = {"qpos", *metadata["invalidated_fields"], *metadata["parent_provenance_fields"]}
    if metadata["actual_qpos_changed"]:
        excluded.add("fk_cache_valid")
    bindings = all(
        np.array_equal(serialized[k], v) for k, v in solver.source.items() if k not in excluded
    )
    final_checks = solver.audit(serialized["qpos"])
    report = {
        "status": "UNREVIEWED",
        "review_status": "UNREVIEWED",
        "selected": False,
        "export_metadata": metadata,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": solver.mj.__version__,
        },
        "retained_optimizer_fraction": fraction,
        "gradient_check": {
            "analytic": analytic,
            "finite_difference": numeric,
            "relative_error": relative,
        },
        "serialized_source_bindings_exact": bindings,
        "serialized_candidate_exact": True,
        "reference": solver.audit(solver.reference),
        "unfiltered_candidate": solver.audit(raw),
        "candidate": final_checks,
        "inputs": {
            key: {"path": str(path), "sha256": solver.input_hashes[key]}
            for key, path in solver.paths.items()
        },
        "compiled_native_model_sha256": solver.model_assets(),
        "native_model_dependency_sha256": solver.dependency_hashes,
        "config_sha256": config_sha256,
        "implementation_sha256": _hash(__file__),
        "candidate_path": str(candidate_path),
        "candidate_sha256": _hash(candidate_path),
        "editable_joints": solver.joints,
        "history": history,
        "required_external_reviews": [
            "Exact selected human/object pair binding and source-camera video alignment",
            "Source and novel-view approach, grasp and release, including worst transitions",
            "Original robot/object triangle intersections and two-way containment; hands, wrists, arms and body",
            "Contact completeness, per-finger semantics, support and unresolved fallbacks",
        ],
        "physical_validation": "not performed",
        "selection": "No automatic adoption; original remains the fallback",
    }
    if any(_hash(path) != expected for path, expected in solver.dependency_hashes.items()):
        raise ValueError("Native model dependency changed during optimization")
    if any(_hash(path) != solver.input_hashes[key] for key, path in solver.paths.items()):
        raise ValueError("Input changed during the optimization; candidate cannot be trusted")
    if _hash(config_path) != config_sha256:
        raise ValueError("Config changed during the optimization; candidate cannot be trusted")
    (output / "robot_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps({"status": report["status"], "candidate": report["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
