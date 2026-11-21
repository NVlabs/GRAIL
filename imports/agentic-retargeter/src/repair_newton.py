"""Optional Newton batched IK proposals with an exactly restricted native scope.

This research backend builds a kinematic tree at ONE native reference key. Only
declared hinges become variables; root and every other native joint are fixed
in the tree itself. It is a candidate initializer, not a coupled trajectory,
collision, force-closure or dynamics solver. No meshes are imported or modified.
Proposals still require the independent native material/whole-mesh/source audit.

Newton and Warp are lazy external dependencies. The API follows the pinned
Newton 0.1.1a2 / Warp 1.9.1 implementation, not arbitrary future APIs.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

_PRIORS = {}
_PROJECTORS = {}
_PLANES = {}
_GATED_VECTORS = {}


class NewtonCudaCapacityError(RuntimeError):
    """The pinned dense IK kernel cannot be verified safe to launch on CUDA."""


def _require_cuda_solver_capacity(wp, solver):
    """Inspect the actual compiled dense kernel before any solve or capture.

    Warp 1.9.1 only prints a warning when a tiled kernel requests too much
    dynamic shared memory. Its subsequent launch can fail without raising a
    Python exception. Inspect metadata without configuring or launching hooks;
    unknown metadata is an error, never permission to attempt a launch.
    """
    device = wp.get_device(solver.device)
    if not device.is_cuda:
        return None
    proof = {
        "device": str(device),
        "dofs": solver.n_dofs,
        "residuals": solver.n_residuals,
        "block_dim": None,
        "required_shared_bytes": None,
        "maximum_shared_bytes": None,
    }

    def fail(reason):
        required = proof["required_shared_bytes"]
        maximum = proof["maximum_shared_bytes"]
        return NewtonCudaCapacityError(
            "Newton CUDA dense IK capacity check failed before solve/capture: "
            f"device={proof['device']}, DOFs={proof['dofs']}, "
            f"residuals={proof['residuals']}, block_dim={proof['block_dim']}, "
            f"required_shared_bytes={required if required is not None else 'unknown'}, "
            f"maximum_shared_bytes={maximum if maximum is not None else 'unknown'}; "
            f"{reason}. Use the CPU backend or a CUDA device with sufficient "
            "verified capacity; no targets or constraints were removed."
        )

    try:
        block_dim = int(solver.TILE_THREADS)
        if block_dim <= 0:
            raise ValueError("invalid pinned solver block dimension")
        proof["block_dim"] = block_dim
        maximum = wp.context.runtime.core.wp_cuda_get_max_shared_memory(device.context)
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, np.integer))
            or maximum <= 0
        ):
            raise ValueError("unknown device shared-memory capacity")
        proof["maximum_shared_bytes"] = int(maximum)
        function = solver._solve_tiled.__func__
        closure = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        kernel = closure["_lm_solve_tiled"].cell_contents
        module = kernel.module.load(device, block_dim=block_dim)
        if module is None:
            raise ValueError("the pinned dense CUDA module did not load")
        name = kernel.get_mangled_name() + "_cuda_kernel_forward_smem_bytes"
        required = module.meta[name]
        if (
            isinstance(required, bool)
            or not isinstance(required, (int, np.integer))
            or required <= 0
        ):
            raise ValueError("unknown dense kernel shared-memory requirement")
        proof["required_shared_bytes"] = int(required)
    except Exception as error:
        raise fail(f"could not verify pinned kernel metadata ({error})") from error
    if proof["required_shared_bytes"] > proof["maximum_shared_bytes"]:
        raise fail("compiled kernel exceeds this device's per-block capacity")
    return proof


def _verify_newton_runtime(newton, wp):
    """Bind private API use to the same immutable runtime as contact queries."""
    from .collision import NEWTON_COMMIT
    from .collision_backend import _verify_checkout

    if wp.__version__ != "1.9.1":
        raise ValueError("Newton keyframe research backend requires Warp 1.9.1")
    checkout = Path(newton.__file__).resolve().parents[1]
    source_hash = _verify_checkout(
        checkout, NEWTON_COMMIT, ["newton", "pyproject.toml"]
    )
    return {
        "newton_commit": NEWTON_COMMIT,
        "newton_version": newton.__version__,
        "newton_source_sha256": source_hash,
        "warp_version": wp.__version__,
    }


def _gated_vector_class(wp, ik):
    if ik in _GATED_VECTORS:
        return _GATED_VECTORS[ik]

    @wp.kernel
    def vector_residual(
        poses: wp.array2d(dtype=wp.transform),
        targets: wp.array(dtype=wp.vec3),
        body: int,
        local: wp.vec3,
        mode: int,
        gate: float,
        weight: float,
        start: int,
        residual: wp.array2d(dtype=float),
    ):
        problem = wp.tid()
        pose = poses[problem, body]
        current = wp.transform_vector(pose, local)
        if mode == 0:
            current = wp.transform_point(pose, local)
        error = targets[problem] - current
        distance = wp.length(error)
        factor = float(0.0)  # noqa: UP018 - mutable Warp scalar
        if gate == 0.0:
            factor = 1.0
        elif distance > gate:
            factor = 1.0 - gate / distance
        for axis in range(3):
            residual[problem, start + axis] = weight * factor * error[axis]

    @wp.kernel
    def vector_jacobian(
        poses: wp.array2d(dtype=wp.transform),
        subspace: wp.array2d(dtype=wp.spatial_vector),
        targets: wp.array(dtype=wp.vec3),
        affects: wp.array(dtype=wp.uint8),
        body: int,
        local: wp.vec3,
        mode: int,
        gate: float,
        weight: float,
        start: int,
        jacobian: wp.array3d(dtype=float),
    ):
        problem, coordinate = wp.tid()
        if affects[coordinate] != 0:
            pose = poses[problem, body]
            current = wp.transform_vector(pose, local)
            if mode == 0:
                current = wp.transform_point(pose, local)
            error = targets[problem] - current
            distance = wp.length(error)
            if distance > gate or gate == 0.0:
                screw = subspace[problem, coordinate]
                velocity = wp.cross(wp.vec3(screw[3], screw[4], screw[5]), current)
                if mode == 0:
                    velocity += wp.vec3(screw[0], screw[1], screw[2])
                derivative = velocity
                if gate > 0.0:
                    derivative = (
                        1.0 - gate / distance
                    ) * velocity + gate * error * wp.dot(error, velocity) / (
                        distance * distance * distance
                    )
                for axis in range(3):
                    jacobian[problem, start + axis, coordinate] = (
                        -weight * derivative[axis]
                    )

    class GatedVector(ik.IKPositionObjective):
        def __init__(self, *args, gate=0.0, mode=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.gate, self.mode = gate, mode

        def compute_residuals(self, body_q, joint_q, model, residuals, start_idx):
            wp.launch(
                vector_residual,
                dim=self.n_problems,
                inputs=[
                    body_q,
                    self.target_positions,
                    self.link_index,
                    self.link_offset,
                    self.mode,
                    self.gate,
                    self.weight,
                    start_idx,
                ],
                outputs=[residuals],
                device=self.device,
            )

        def compute_jacobian_analytic(
            self, body_q, joint_q, model, jacobian, joint_S_s, start_idx
        ):
            wp.launch(
                vector_jacobian,
                dim=(self.n_problems, model.joint_dof_count),
                inputs=[
                    body_q,
                    joint_S_s,
                    self.target_positions,
                    self.affects_dof,
                    self.link_index,
                    self.link_offset,
                    self.mode,
                    self.gate,
                    self.weight,
                    start_idx,
                ],
                outputs=[jacobian],
                device=self.device,
            )

    _GATED_VECTORS[ik] = GatedVector
    return GatedVector


def _plane_class(wp, ik):
    if ik in _PLANES:
        return _PLANES[ik]

    @wp.kernel
    def plane_residual(
        poses: wp.array2d(dtype=wp.transform),
        bodies: wp.array(dtype=int),
        local: wp.array(dtype=wp.vec3),
        targets: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        margins: wp.array(dtype=float),
        weights: wp.array(dtype=float),
        start: int,
        residual: wp.array2d(dtype=float),
    ):
        problem, plane = wp.tid()
        p = wp.transform_point(poses[problem, bodies[plane]], local[plane])
        gap = wp.dot(p - targets[plane], normals[plane])
        residual[problem, start + plane] = weights[plane] * wp.max(
            margins[plane] - gap, 0.0
        )

    @wp.kernel
    def plane_jacobian(
        poses: wp.array2d(dtype=wp.transform),
        subspace: wp.array2d(dtype=wp.spatial_vector),
        bodies: wp.array(dtype=int),
        local: wp.array(dtype=wp.vec3),
        targets: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        margins: wp.array(dtype=float),
        weights: wp.array(dtype=float),
        affects: wp.array2d(dtype=wp.uint8),
        start: int,
        jacobian: wp.array3d(dtype=float),
    ):
        problem, plane, coordinate = wp.tid()
        if affects[plane, coordinate] != 0 and weights[plane] > 0.0:
            point = wp.transform_point(poses[problem, bodies[plane]], local[plane])
            if wp.dot(point - targets[plane], normals[plane]) < margins[plane]:
                screw = subspace[problem, coordinate]
                velocity = wp.vec3(screw[0], screw[1], screw[2]) + wp.cross(
                    wp.vec3(screw[3], screw[4], screw[5]), point
                )
                jacobian[problem, start + plane, coordinate] = -weights[plane] * wp.dot(
                    normals[plane], velocity
                )

    class StaticPlanes(ik.IKObjective):
        def __init__(self, count, capacity, dimensions, arrays):
            self.count, self.capacity, self.dimensions = count, capacity, dimensions
            self.arrays = arrays

        def residual_dim(self):
            return self.capacity

        def supports_analytic(self):
            return True

        def compute_residuals(self, body_q, joint_q, model, residuals, start_idx):
            wp.launch(
                plane_residual,
                dim=(self.count, self.capacity),
                inputs=[body_q, *self.arrays[:6], start_idx],
                outputs=[residuals],
                device=self.device,
            )

        def compute_jacobian_analytic(
            self, body_q, joint_q, model, jacobian, joint_S_s, start_idx
        ):
            wp.launch(
                plane_jacobian,
                dim=(self.count, self.capacity, self.dimensions),
                inputs=[body_q, joint_S_s, *self.arrays, start_idx],
                outputs=[jacobian],
                device=self.device,
            )

    _PLANES[ik] = StaticPlanes
    return StaticPlanes


def _projector(wp):
    if wp in _PROJECTORS:
        return _PROJECTORS[wp]

    @wp.kernel
    def project_delta(
        q: wp.array2d(dtype=float),
        lower: wp.array(dtype=float),
        upper: wp.array(dtype=float),
        counts: wp.array(dtype=int),
    ):
        problem, coordinate = wp.tid()
        before = q[problem, coordinate]
        after = wp.clamp(before, lower[coordinate], upper[coordinate])
        if after != before:
            wp.atomic_add(counts, problem, 1)
        q[problem, coordinate] = after

    _PROJECTORS[wp] = project_delta
    return project_delta


def _prior_class(wp, ik):
    if ik in _PRIORS:
        return _PRIORS[ik]

    @wp.kernel
    def prior_residual(
        q: wp.array2d(dtype=float),
        weight: float,
        start: int,
        residual: wp.array2d(dtype=float),
    ):
        problem, coordinate = wp.tid()
        residual[problem, start + coordinate] = weight * q[problem, coordinate]

    @wp.kernel
    def prior_jacobian(weight: float, start: int, jacobian: wp.array3d(dtype=float)):
        problem, coordinate = wp.tid()
        jacobian[problem, start + coordinate, coordinate] = weight

    class ReferencePrior(ik.IKObjective):
        def __init__(self, count, dimensions, weight):
            self.count, self.dimensions, self.weight = count, dimensions, weight

        def residual_dim(self):
            return self.dimensions

        def supports_analytic(self):
            return True

        def compute_residuals(self, body_q, joint_q, model, residuals, start_idx):
            wp.launch(
                prior_residual,
                dim=(self.count, self.dimensions),
                inputs=[joint_q, self.weight, start_idx],
                outputs=[residuals],
                device=self.device,
            )

        def compute_jacobian_analytic(
            self, body_q, joint_q, model, jacobian, joint_S_s, start_idx
        ):
            wp.launch(
                prior_jacobian,
                dim=(self.count, self.dimensions),
                inputs=[self.weight, start_idx],
                outputs=[jacobian],
                device=self.device,
            )

    _PRIORS[ik] = ReferencePrior
    return ReferencePrior


def _finite(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _weight(value, label):
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(label + " must be finite and nonnegative")
    return result


class NewtonKeyframeIK:
    """One immutable native reference, with a batch of independent IK starts.

    ``editable_joints`` maps native limited-hinge names to ``[delta_min,
    delta_max]`` in radians, each containing zero. Native limits are intersected
    with those corridors. Unsupported multi-joint bodies, tendons, equalities,
    mocap and multiple root bodies are rejected rather than approximated.

    The Newton tree stores joint DELTAS about reference values. Its internal
    hinge order is exposed in ``names`` and can differ from caller dictionary
    order. Frozen native values are copied verbatim when proposals are mapped
    back. Soft LM joint-limit terms guide proposals; post-checks mark limit
    violations invalid without clipping them into apparently valid candidates.

    ``normalize_reference_for_fk`` explicitly matches native viewers that
    normalize linearly interpolated root quaternions before FK. It changes only
    the internal FK copy; exported frozen quaternion coordinates remain bitwise
    equal to the supplied reference. The default requires normalized input.
    """

    def __init__(
        self,
        native_model,
        reference_qpos,
        editable_joints,
        *,
        device="cuda:0",
        normalize_reference_for_fk=False,
    ):
        started = time.perf_counter()
        import mujoco
        from scipy.spatial.transform import Rotation

        try:
            import newton
            import warp as wp
            from newton import ik
        except ImportError as exc:
            raise RuntimeError(
                "Newton IK requires the configured external Newton/Warp worker"
            ) from exc
        self.runtime_proof = _verify_newton_runtime(newton, wp)
        self.wp, self.newton, self.ik, self.mujoco, self.Rotation = (
            wp,
            newton,
            ik,
            mujoco,
            Rotation,
        )
        model = native_model
        self.native_model = model
        self.reference = _finite(reference_qpos, (model.nq,), "reference_qpos").copy()
        self.reference.setflags(write=False)
        compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, compiled)
        self.native_model_sha256 = hashlib.sha256(compiled.tobytes()).hexdigest()
        if not isinstance(normalize_reference_for_fk, bool):
            raise TypeError("normalize_reference_for_fk must be explicit boolean")
        self.reference_quaternion_norm_error = 0.0
        for joint in range(model.njnt):
            kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
            if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                start = address + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
                norm = float(np.linalg.norm(self.reference[start : start + 4]))
                self.reference_quaternion_norm_error = max(
                    self.reference_quaternion_norm_error, abs(norm - 1)
                )
                if norm < 1e-12 or (
                    not normalize_reference_for_fk
                    and not np.isclose(
                        norm,
                        1,
                        atol=1e-6,
                        rtol=0,
                    )
                ):
                    raise ValueError("reference quaternion must already be normalized")
        if (
            model.neq
            or model.ntendon
            or model.nmocap
            or np.count_nonzero(model.body_parentid[1:] == 0) != 1
        ):
            raise ValueError(
                "Newton keyframe proposals require a single uncoupled native body tree"
            )
        if not isinstance(editable_joints, dict) or not editable_joints:
            raise ValueError(
                "editable_joints must explicitly name native limited hinges"
            )
        requested = {}
        for name, corridor in editable_joints.items():
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if (
                joint < 0
                or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                or not model.jnt_limited[joint]
            ):
                raise ValueError(
                    "editable joint must be a known limited native hinge: " + str(name)
                )
            body = int(model.jnt_bodyid[joint])
            if model.body_jntnum[body] != 1:
                raise ValueError(
                    "editable hinges require one native joint on their body"
                )
            delta = _finite(corridor, (2,), "delta corridor")
            if not delta[0] <= 0 <= delta[1] or delta[0] >= delta[1]:
                raise ValueError("delta corridor must contain zero with positive width")
            address = int(model.jnt_qposadr[joint])
            lower = max(delta[0], model.jnt_range[joint, 0] - self.reference[address])
            upper = min(delta[1], model.jnt_range[joint, 1] - self.reference[address])
            if not lower <= 0 <= upper or lower >= upper:
                raise ValueError("reference/corridor violates native limits")
            requested[body] = {
                "joint": joint,
                "name": name,
                "qid": address,
                "lower": lower,
                "upper": upper,
            }
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = self.reference
        mujoco.mj_normalizeQuat(model, self.data.qpos)
        mujoco.mj_forward(model, self.data)
        data = self.data
        self.root_body = int(np.flatnonzero(model.body_parentid[1:] == 0)[0]) + 1
        self.world_rotation = data.xmat[self.root_body].reshape(3, 3).copy()
        self.world_translation = data.xpos[self.root_body].copy()
        builder = newton.ModelBuilder()
        builder.add_articulation(key="native_keyframe_delta_tree")
        self.body_ids, self.body_names = {}, {}
        self.names, self.qids, lower, upper = [], [], [], []
        for body in range(1, model.nbody):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
                or f"body:{body}"
            )
            # Inertia is immaterial to IK; preserve the native inertial record
            # instead of supplying invented dynamics. No shapes are attached.
            inertial_rotation = Rotation.from_quat(
                np.roll(model.body_iquat[body], -1)
            ).as_matrix()
            inertia = (
                inertial_rotation
                @ np.diag(model.body_inertia[body])
                @ inertial_rotation.T
            )
            self.body_ids[body] = builder.add_body(
                key=name,
                mass=float(model.body_mass[body]),
                com=wp.vec3(model.body_ipos[body]),
                I_m=wp.mat33(inertia),
            )
            self.body_names[name] = self.body_ids[body]
            parent = int(model.body_parentid[body])
            if body == self.root_body:
                rotation, translation = np.eye(3), np.zeros(3)
            else:
                parent_rotation = data.xmat[parent].reshape(3, 3)
                rotation = parent_rotation.T @ data.xmat[body].reshape(3, 3)
                translation = parent_rotation.T @ (data.xpos[body] - data.xpos[parent])
            quaternion = Rotation.from_matrix(rotation).as_quat()
            if body in requested:
                spec = requested[body]
                joint, address = spec["joint"], spec["qid"]
                anchor = model.jnt_pos[joint]
                builder.add_joint_revolute(
                    parent=self.body_ids.get(parent, -1),
                    child=self.body_ids[body],
                    parent_xform=wp.transform(
                        translation + rotation @ anchor, quaternion
                    ),
                    child_xform=wp.transform(anchor, wp.quat_identity()),
                    axis=wp.vec3(model.jnt_axis[joint]),
                    limit_lower=float(spec["lower"]),
                    limit_upper=float(spec["upper"]),
                    key=spec["name"],
                )
                self.names.append(spec["name"])
                self.qids.append(address)
                lower.append(spec["lower"])
                upper.append(spec["upper"])
            else:
                builder.add_joint_fixed(
                    parent=self.body_ids.get(parent, -1),
                    child=self.body_ids[body],
                    parent_xform=wp.transform(translation, quaternion),
                    key="fixed:" + name,
                )
        self.qids = np.asarray(self.qids, dtype=np.int64)
        self.affects = {}
        for native_body, newton_body in self.body_ids.items():
            ancestry, cursor = set(), native_body
            while cursor:
                ancestry.add(cursor)
                cursor = int(model.body_parentid[cursor])
            self.affects[newton_body] = np.array(
                [
                    int(
                        model.jnt_bodyid[
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                        ]
                    )
                    in ancestry
                    for name in self.names
                ],
                dtype=np.uint8,
            )
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.device = wp.get_device(device)
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        if self.model.joint_coord_count != len(
            self.names
        ) or self.model.joint_dof_count != len(self.names):
            raise ValueError("Newton reduced-tree coordinate mapping is inconsistent")
        self.reference.setflags(write=False)
        self._solver_cache = {}
        wp.synchronize_device(self.device)
        self.build_seconds = time.perf_counter() - started
        self.last_timing = {}

    def native_qpos(self, deltas):
        deltas = np.asarray(deltas, dtype=np.float64)
        if (
            deltas.ndim != 2
            or deltas.shape[1] != len(self.names)
            or not np.isfinite(deltas).all()
        ):
            raise ValueError("deltas must be finite B x editable-hinge count")
        qpos = np.repeat(self.reference[None], len(deltas), axis=0)
        qpos[:, self.qids] += deltas
        return qpos

    def forward(self, delta):
        """World body transforms for FK parity checks, not physical replay."""
        delta = _finite(delta, (len(self.names),), "delta")
        wp = self.wp
        q = wp.array(delta.astype(np.float32), dtype=wp.float32, device=self.device)
        self.newton.eval_fk(self.model, q, self.model.joint_qd, self.state)
        local = self.state.body_q.numpy()
        positions = (
            local[:, :3].astype(np.float64) @ self.world_rotation.T
            + self.world_translation
        )
        rotations = (
            self.world_rotation @ self.Rotation.from_quat(local[:, 3:]).as_matrix()
        )
        return {"positions": positions, "rotations": rotations}

    def _targets(self, targets, count, rotation=False, normal=False):
        parsed = []
        for spec in targets:
            allowed = {"body_name", "local", "target_world", "weight"}
            if not rotation:
                allowed.add("max_angle_deg" if normal else "max_error")
            if set(spec) - allowed or not {
                "body_name",
                "local",
                "target_world",
            }.issubset(spec):
                raise ValueError(
                    "target needs body_name, local, target_world, and optional weight"
                )
            name = spec["body_name"]
            if name not in self.body_names:
                raise ValueError("unknown native target body: " + str(name))
            dimension = 4 if rotation else 3
            local = _finite(spec["local"], (dimension,), "local target offset")
            target = np.asarray(spec["target_world"], dtype=np.float64)
            if target.shape == (dimension,):
                target = np.repeat(target[None], count, axis=0)
            target = _finite(target, (count, dimension), "target_world")
            weight = _weight(spec.get("weight", 1.0), "target weight")
            if rotation:
                if not np.isclose(
                    np.linalg.norm(local), 1, atol=1e-8
                ) or not np.allclose(np.linalg.norm(target, axis=1), 1, atol=1e-8):
                    raise ValueError("rotation targets must be unit xyzw quaternions")
                target = self.Rotation.from_matrix(
                    self.world_rotation.T @ self.Rotation.from_quat(target).as_matrix()
                ).as_quat()
            elif normal:
                if not np.isclose(
                    np.linalg.norm(local), 1, atol=1e-8, rtol=0
                ) or not np.allclose(
                    np.linalg.norm(target, axis=1), 1, atol=1e-8, rtol=0
                ):
                    raise ValueError("normal targets must be explicit unit vectors")
                target = target @ self.world_rotation
            else:
                target = (target - self.world_translation) @ self.world_rotation
            parsed.append((self.body_names[name], local.copy(), target, weight))
        return parsed

    def _halfspaces(self, specs, capacity):
        dimensions = len(self.names)
        bodies = np.zeros(capacity, dtype=np.int32)
        local, target, normal = [
            np.zeros((capacity, 3), dtype=np.float32) for _ in range(3)
        ]
        margin, weight = [np.zeros(capacity, dtype=np.float32) for _ in range(2)]
        affects = np.zeros((capacity, dimensions), dtype=np.uint8)
        for index, spec in enumerate(specs):
            required = {
                "body_name",
                "local",
                "target_world",
                "normal_world",
                "margin",
                "weight",
            }
            if set(spec) != required or spec["body_name"] not in self.body_names:
                raise ValueError(
                    "halfspace needs a known body and explicit local/target/normal/margin/weight"
                )
            body = self.body_names[spec["body_name"]]
            bodies[index] = body
            local[index] = _finite(spec["local"], (3,), "halfspace local")
            target[index] = (
                _finite(spec["target_world"], (3,), "halfspace target")
                - self.world_translation
            ) @ self.world_rotation
            n = _finite(spec["normal_world"], (3,), "halfspace normal")
            if not np.isclose(np.linalg.norm(n), 1, atol=1e-8, rtol=0):
                raise ValueError(
                    "halfspace normal must be unit and explicitly oriented"
                )
            normal[index] = n @ self.world_rotation
            margin[index] = _weight(spec["margin"], "halfspace margin")
            weight[index] = _weight(spec["weight"], "halfspace weight")
            affects[index] = self.affects[body]
        return [bodies, local, target, normal, margin, weight, affects]

    def solve(
        self,
        initial_deltas,
        position_targets,
        *,
        rotation_targets=(),
        normal_targets=(),
        iterations=32,
        pose_weight=0.01,
        limit_weight=100.0,
        lambda_initial=0.01,
        project_bounds=False,
        capture_graph=False,
        halfspaces=(),
        halfspace_capacity=0,
    ):
        """Generate independent static candidates; no motion is selected/applied.

        Position targets use ``local`` XYZ and ``target_world`` XYZ or B x XYZ.
        Their optional ``max_error`` is a deadband radius in meters. Normal
        targets use unit vectors with optional ``max_angle_deg`` deadbands;
        callers explicitly supply the desired opposing direction. These gates
        avoid pulling already valid contacts toward arbitrary point minima.
        Rotation targets use unit XYZW quaternions for both fields. To constrain
        a pad normal without fixing twist, a caller can supply a second local
        point along that normal with a correspondingly offset world target.
        The pose prior penalizes deltas from this bound native reference; it is
        not a substitute for source-image guidance or a temporal objective.

        ``project_bounds=True`` is an explicit PROJECTED NEWTON ADAPTATION:
        project only proposed editable deltas before Newton evaluates/accepts
        each LM proposal. This uses the pinned solver's private integration
        hook and records coordinate projection counts. Stock mode retains soft
        limits for comparison. Frozen/source poses and model limits stay fixed.
        CUDA ``capture_graph=True`` reuses a bounded solve graph after an initial
        warm-up/capture; first-call setup time is included in returned timing.

        Optional halfspaces attach native body-local material points to explicit
        world-static target planes. Their orientation must be established from
        a separate safe reference; no sign is inferred here. They are local
        optimizer aids, not signed solid distances. Set a fixed capacity to
        reuse the solver across outer audit rounds; excess planes are rejected,
        never silently truncated. Moving target planes require a different
        objective and are unsupported by this static keyframe proposal API.
        """
        started = time.perf_counter()
        initial = np.asarray(initial_deltas, dtype=np.float64)
        self.native_qpos(initial)
        count = len(initial)
        if not count or np.any(initial < self.lower) or np.any(initial > self.upper):
            raise ValueError(
                "initial candidates must be nonempty and inside declared/native corridors"
            )
        if type(iterations) is not int or not 1 <= iterations <= 1000:
            raise ValueError("iterations must be an integer between 1 and 1000")
        if type(project_bounds) is not bool or type(capture_graph) is not bool:
            raise ValueError("project_bounds and capture_graph must be booleans")
        if capture_graph and not self.device.is_cuda:
            raise ValueError(
                "capture_graph requires an explicitly selected CUDA device"
            )
        pose_weight, limit_weight = (
            _weight(pose_weight, "pose_weight"),
            _weight(limit_weight, "limit_weight"),
        )
        damping = _weight(lambda_initial, "lambda_initial")
        if damping == 0 or limit_weight == 0:
            raise ValueError("damping and limit_weight must be positive")
        positions = self._targets(position_targets, count)
        rotations = self._targets(rotation_targets, count, rotation=True)
        normals = self._targets(normal_targets, count, normal=True)
        position_gates = [
            _weight(spec.get("max_error", 0.0), "max_error")
            for spec in position_targets
        ]
        normal_angles = [
            _weight(spec.get("max_angle_deg", 0.0), "max_angle_deg")
            for spec in normal_targets
        ]
        if any(angle > 180 for angle in normal_angles):
            raise ValueError("normal angle gate must not exceed 180 degrees")
        normal_gates = [2 * np.sin(np.radians(angle) / 2) for angle in normal_angles]
        if (
            type(halfspace_capacity) is not int
            or not 0 <= halfspace_capacity <= 1024
            or len(halfspaces) > halfspace_capacity
        ):
            raise ValueError(
                "halfspace_capacity must cover all explicit planes and be <=1024"
            )
        plane_arrays = self._halfspaces(halfspaces, halfspace_capacity)
        if not positions and not rotations and not normals:
            raise ValueError("IK needs at least one explicit spatial target")
        key = (
            count,
            pose_weight,
            limit_weight,
            damping,
            bool(project_bounds),
            halfspace_capacity,
            tuple((body, tuple(local), weight) for body, local, _, weight in positions),
            tuple(position_gates),
            tuple((body, tuple(local), weight) for body, local, _, weight in normals),
            tuple(normal_gates),
            tuple((body, tuple(local), weight) for body, local, _, weight in rotations),
        )
        wp, ik = self.wp, self.ik
        cache = self._solver_cache.get(key)
        if cache is None:
            dimensions = len(self.names)
            total = (
                3 * (len(positions) + len(normals) + len(rotations))
                + 2 * dimensions
                + halfspace_capacity
            )
            q = wp.array(
                initial.astype(np.float32), dtype=wp.float32, device=self.device
            )
            objectives, target_arrays = [], []
            vectors = [
                *[(item, gate, 0) for item, gate in zip(positions, position_gates)],
                *[(item, gate, 1) for item, gate in zip(normals, normal_gates)],
            ]
            for (body, local, target, weight), gate, mode in vectors:
                array = wp.array(
                    target.astype(np.float32), dtype=wp.vec3, device=self.device
                )
                target_arrays.append(array)
                objectives.append(
                    _gated_vector_class(wp, ik)(
                        body,
                        wp.vec3(local),
                        array,
                        count,
                        total,
                        3 * len(objectives),
                        weight=weight,
                        gate=float(gate),
                        mode=mode,
                    )
                )
            for body, local, target, weight in rotations:
                array = wp.array(
                    target.astype(np.float32), dtype=wp.vec4, device=self.device
                )
                target_arrays.append(array)
                objectives.append(
                    ik.IKRotationObjective(
                        body,
                        wp.quat(local),
                        array,
                        count,
                        total,
                        3 * len(objectives),
                        weight=weight,
                    )
                )
            objectives.append(
                ik.IKJointLimitObjective(
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    count,
                    total,
                    3 * (len(positions) + len(normals) + len(rotations)),
                    weight=limit_weight,
                )
            )
            objectives.append(_prior_class(wp, ik)(count, dimensions, pose_weight))
            device_planes = []
            if halfspace_capacity:
                dtypes = [
                    wp.int32,
                    wp.vec3,
                    wp.vec3,
                    wp.vec3,
                    wp.float32,
                    wp.float32,
                    wp.uint8,
                ]
                device_planes = [
                    wp.array(array, dtype=dtype, device=self.device)
                    for array, dtype in zip(plane_arrays, dtypes)
                ]
                objectives.append(
                    _plane_class(wp, ik)(
                        count, halfspace_capacity, dimensions, device_planes
                    )
                )
            solver = ik.IKSolver(
                self.model,
                q,
                objectives,
                lambda_initial=damping,
                jacobian_mode=ik.IKJacobianMode.ANALYTIC,
            )
            cuda_capacity = _require_cuda_solver_capacity(wp, solver)
            projection_counts = wp.zeros(count, dtype=wp.int32, device=self.device)
            if project_bounds:
                lower32, upper32 = (
                    self.lower.astype(np.float32),
                    self.upper.astype(np.float32),
                )
                lo_rounds_out = lower32.astype(np.float64) < self.lower
                hi_rounds_out = upper32.astype(np.float64) > self.upper
                lower32[lo_rounds_out] = np.nextafter(
                    lower32[lo_rounds_out], np.float32(np.inf)
                )
                upper32[hi_rounds_out] = np.nextafter(
                    upper32[hi_rounds_out], np.float32(-np.inf)
                )
                lower_array = wp.array(lower32, dtype=wp.float32, device=self.device)
                upper_array = wp.array(upper32, dtype=wp.float32, device=self.device)
                original_integrate = solver._integrate_dq
                projection_kernel = _projector(wp)

                def integrate_projected(step_size=1.0):
                    original_integrate(step_size)
                    wp.launch(
                        projection_kernel,
                        dim=(count, dimensions),
                        inputs=[solver.joint_q_proposed, lower_array, upper_array],
                        outputs=[projection_counts],
                        device=self.device,
                    )

                solver._integrate_dq = integrate_projected
            cache = {
                "q": q,
                "targets": target_arrays,
                "solver": solver,
                "cuda_capacity": cuda_capacity,
                "projection_counts": projection_counts,
                "graphs": {},
                "planes": device_planes,
            }
            self._solver_cache[key] = cache
        else:
            cache["q"].assign(initial.astype(np.float32))
            for array, (_, _, target, _) in zip(
                cache["targets"], positions + normals + rotations
            ):
                array.assign(target.astype(np.float32))
            for array, values in zip(cache["planes"], plane_arrays):
                array.assign(values)
        cache["projection_counts"].zero_()
        initialized = time.perf_counter()
        graph_built = False
        if capture_graph:
            if iterations not in cache["graphs"]:
                cache["solver"].solve(iterations=iterations)
                wp.synchronize_device(self.device)
                cache["q"].assign(initial.astype(np.float32))
                cache["projection_counts"].zero_()
                with wp.ScopedCapture(device=self.device) as capture:
                    cache["solver"].solve(iterations=iterations)
                cache["graphs"][iterations] = capture.graph
                graph_built = True
            wp.capture_launch(cache["graphs"][iterations])
        else:
            cache["solver"].solve(iterations=iterations)
        wp.synchronize_device(self.device)
        finished = time.perf_counter()
        deltas = cache["q"].numpy().astype(np.float64)
        finite = np.isfinite(deltas).all(axis=1)
        # Float32 solutions can overshoot soft limits. Do not silently clip them.
        within = (
            finite
            & np.all(deltas >= self.lower - 1e-7, axis=1)
            & np.all(deltas <= self.upper + 1e-7, axis=1)
        )
        qpos = np.repeat(self.reference[None], count, axis=0)
        qpos[:, self.qids] += deltas
        self.last_timing = {
            "batch_count": count,
            "initialization_seconds": initialized - started,
            "solve_seconds": finished - initialized,
            "total_seconds": time.perf_counter() - started,
            "cuda_graph": capture_graph,
            "graph_built_this_call": graph_built,
        }
        return {
            "status": "UNREVIEWED",
            "qpos": qpos,
            "deltas": deltas,
            "within_bounds": within,
            "objective_cost": cache["solver"].costs.numpy().astype(np.float64),
            "timing": self.last_timing.copy(),
            "collision_checked": False,
            "temporal_checked": False,
            "variant": "projected_newton_lm"
            if project_bounds
            else "stock_newton_lm_soft_limits",
            "projected_coordinate_proposals": cache["projection_counts"].numpy().copy(),
            "halfspace_count": len(halfspaces),
            "halfspace_capacity": halfspace_capacity,
        }

    def metadata(self):
        return {
            "backend": "newton_reduced_native_keyframe_ik",
            **self.runtime_proof,
            "device": str(self.device),
            "names": self.names.copy(),
            "qpos_indices": self.qids.tolist(),
            "reference_sha256": hashlib.sha256(self.reference.tobytes()).hexdigest(),
            "compiled_native_model_sha256": self.native_model_sha256,
            "lower_delta_rad": self.lower.tolist(),
            "upper_delta_rad": self.upper.tolist(),
            "build_seconds": self.build_seconds,
            "reference_quaternion_norm_error": self.reference_quaternion_norm_error,
            "scene_geometry_modified": False,
            "scope": "independent static IK starts; original native audit required; no trajectory coupling or dynamics",
        }
