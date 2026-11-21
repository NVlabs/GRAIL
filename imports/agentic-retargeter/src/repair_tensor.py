"""Optional native tensor FK and Warp winding-query proposal primitives.

Design references: HOI motion repair ``kinematics.py`` / ``geometry.py`` at
58e618feef66c8a168efe75329c57efc3c36b34f, and MuJoCo's compiled joint/body
transforms. This independent implementation imports no HOI checkout or Newton.
Torch and Warp load only when the corresponding explicit worker is constructed.

Winding-query signs are optimization heuristics, never penetration certificates.
Original triangle audits remain independent. No geometry, native motion or
contact target is modified by these primitives.
"""

from __future__ import annotations

import contextlib
import hashlib

import numpy as np

REFERENCE_COMMIT = "58e618feef66c8a168efe75329c57efc3c36b34f"
_KERNELS = {}
_AUTOGRAD = {}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Tensor repair needs Torch in its external worker") from exc
    return torch


def _quaternion_matrix(q, torch):
    """Normalize a nonzero native WXYZ quaternion, matching MuJoCo FK."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


class NativeTensorFK:
    """Differentiable native body FK for fixed bodies, hinges and a free root.

    ``fk(qpos_batch)`` returns ``(R, p)`` including world body zero, with shapes
    ``[batch, nbody, 3, 3]`` and ``[batch, nbody, 3]``. Source quaternions are
    normalized inside FK only; input tensors and exported native values are not
    mutated. Nonzero hinge pivots and native ``qpos0`` offsets are preserved.

    Each body may have at most one joint. Ball/slide joints, coupled constraints,
    tendons and mocap bodies are rejected rather than approximated. One free
    joint is optional and must belong to a root body; fixed root branches work.
    Body transforms are evaluated by tree depth, not one Python loop per body.

    ``validate_qpos`` provides explicit finite/nonzero-quaternion checks before
    optimization. The hot differentiable call checks shape/device/dtype without
    synchronizing GPU values; invalid numerical input propagates nonfinite FK
    and must never be accepted by a caller's native audit.
    """

    def __init__(self, model, device="cpu", *, dtype=None):
        import mujoco

        torch = _torch()
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype is None else dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Native tensor FK requires float32 or float64")
        if model.neq or model.ntendon or model.nmocap:
            raise ValueError("Tensor FK requires an uncoupled native body tree")
        self.nq, self.nbody = int(model.nq), int(model.nbody)
        self.parents = np.asarray(model.body_parentid, dtype=np.int64).copy()
        if np.any(self.parents[1:] >= np.arange(1, self.nbody)):
            raise ValueError("Native body parents must precede their children")
        axes = np.zeros((self.nbody, 3))
        pivots = np.zeros_like(axes)
        addresses = np.zeros(self.nbody, dtype=np.int64)
        offsets = np.zeros(self.nbody)
        hinges = np.zeros(self.nbody)
        self.free_body = None
        self.free_address = None
        for body in range(1, self.nbody):
            count = int(model.body_jntnum[body])
            if count > 1:
                raise ValueError("Tensor FK supports at most one joint per body")
            if not count:
                continue
            joint = int(model.body_jntadr[body])
            kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                if self.free_body is not None or self.parents[body] != 0:
                    raise ValueError("Tensor FK supports one optional free root")
                self.free_body, self.free_address = body, address
            elif kind == mujoco.mjtJoint.mjJNT_HINGE:
                axes[body] = model.jnt_axis[joint]
                pivots[body] = model.jnt_pos[joint]
                addresses[body] = address
                offsets[body] = model.qpos0[address]
                hinges[body] = 1
            else:
                raise ValueError("Tensor FK supports only native hinges/free roots")
        self.addresses = self.t(addresses, torch.long)
        self.offsets = self.t(offsets)
        self.hinges = self.t(hinges)
        self.pivots = self.t(pivots)
        self.rest_p = self.t(np.array(model.body_pos, copy=True))
        self.rest_R = _quaternion_matrix(
            self.t(np.array(model.body_quat, copy=True)), torch
        )
        self.identity = self.t(np.broadcast_to(np.eye(3), (self.nbody, 3, 3)).copy())
        skew = np.zeros((self.nbody, 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -axes[:, 2], axes[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axes[:, 2], -axes[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axes[:, 1], axes[:, 0]
        self.skew = self.t(skew)
        self.axis_outer = self.t(axes[:, :, None] * axes[:, None, :])
        depth = np.zeros(self.nbody, dtype=int)
        for body in range(1, self.nbody):
            depth[body] = depth[self.parents[body]] + 1
        self.levels = []
        for level in range(1, int(depth.max(initial=0)) + 1):
            ids = np.flatnonzero(depth == level)
            self.levels.append(
                (self.t(ids, torch.long), self.t(self.parents[ids], torch.long))
            )
        self.free_index = (
            None if self.free_body is None else self.t([self.free_body], torch.long)
        )
        compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, compiled)
        self.model_sha256 = hashlib.sha256(compiled.tobytes()).hexdigest()
        self.validate_qpos(self.t(np.asarray(model.qpos0)[None].copy()))

    def t(self, value, dtype=None):
        """Convenience conversion matching the reference solver's FK interface."""
        return self.torch.as_tensor(
            value, device=self.device, dtype=self.dtype if dtype is None else dtype
        )

    def _shape(self, qpos):
        if (
            not isinstance(qpos, self.torch.Tensor)
            or qpos.ndim != 2
            or qpos.shape[1] != self.nq
            or qpos.device != self.device
            or qpos.dtype != self.dtype
        ):
            raise ValueError(
                "qpos must be a native batch tensor on the configured device/dtype"
            )

    def validate_qpos(self, qpos):
        self._shape(qpos)
        if not bool(self.torch.isfinite(qpos).all()):
            raise ValueError("Native qpos must be finite")
        if self.free_address is not None:
            start = self.free_address + 3
            if bool((qpos[:, start : start + 4].norm(dim=-1) < 1e-12).any()):
                raise ValueError("Native free quaternion must be nonzero")

    def __call__(self, qpos):
        torch = self.torch
        self._shape(qpos)
        batch = len(qpos)
        angles = (
            (qpos[:, self.addresses] - self.offsets) * self.hinges
            if self.nq
            else qpos.new_zeros((batch, self.nbody))
        )
        cosine, sine = angles.cos()[..., None, None], angles.sin()[..., None, None]
        rotations = (
            cosine * self.identity + (1 - cosine) * self.axis_outer + sine * self.skew
        )
        local_R = self.rest_R @ rotations
        pivot_shift = self.pivots - (rotations @ self.pivots[..., None]).squeeze(-1)
        local_p = self.rest_p + (self.rest_R @ pivot_shift[..., None]).squeeze(-1)
        if self.free_address is not None:
            start = self.free_address
            root_R = _quaternion_matrix(qpos[:, start + 3 : start + 7], torch)
            local_R = local_R.index_copy(1, self.free_index, root_R[:, None])
            local_p = local_p.index_copy(
                1, self.free_index, qpos[:, None, start : start + 3]
            )
        world_R = self.identity[None].expand(batch, -1, -1, -1)
        world_p = qpos.new_zeros((batch, self.nbody, 3))
        for ids, parents in self.levels:
            parent_R = world_R[:, parents]
            child_R = parent_R @ local_R[:, ids]
            child_p = world_p[:, parents] + (
                parent_R @ local_p[:, ids, :, None]
            ).squeeze(-1)
            world_R = world_R.index_copy(1, ids, child_R)
            world_p = world_p.index_copy(1, ids, child_p)
        return world_R, world_p

    def points(self, qpos, body_ids, points_local):
        """Transform material points; gradients flow through all native ancestors."""
        if isinstance(body_ids, self.torch.Tensor):
            integral = body_ids.dtype in (
                self.torch.int8,
                self.torch.int16,
                self.torch.int32,
                self.torch.int64,
                self.torch.uint8,
            )
        else:
            integral = np.issubdtype(np.asarray(body_ids).dtype, np.integer)
        if not integral:
            raise ValueError("Material body IDs must be integers without coercion")
        body_ids = self.t(body_ids, self.torch.long)
        points_local = self.t(points_local)
        if body_ids.ndim != 1 or points_local.shape != (len(body_ids), 3):
            raise ValueError(
                "Material points need matching body IDs and N x3 coordinates"
            )
        if bool(((body_ids < 0) | (body_ids >= self.nbody)).any()):
            raise ValueError("Material body ID outside native model")
        rotations, positions = self(qpos)
        return positions[:, body_ids] + (
            rotations[:, body_ids] @ points_local[..., None]
        ).squeeze(-1)

    def metadata(self):
        return {
            "backend": "native_torch_fk",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "torch_version": self.torch.__version__,
            "compiled_native_model_sha256": self.model_sha256,
            "source_idea_commit": REFERENCE_COMMIT,
            "quaternion_semantics": "normalized for FK only; input native tensor unchanged",
            "acceptance": False,
        }


def _query_kernel(wp):
    if wp in _KERNELS:
        return _KERNELS[wp]

    @wp.kernel
    def winding_query(
        mesh: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        radius: float,
        accuracy: float,
        threshold: float,
        distances: wp.array(dtype=float),
        gradients: wp.array(dtype=wp.vec3),
        faces: wp.array(dtype=int),
        signs: wp.array(dtype=float),
        found: wp.array(dtype=int),
        gradient_valid: wp.array(dtype=int),
    ):
        index = wp.tid()
        point = points[index]
        query = wp.mesh_query_point_sign_winding_number(
            mesh, point, radius, accuracy, threshold
        )
        distances[index] = wp.nan
        gradients[index] = wp.vec3(wp.nan)
        faces[index] = -1
        signs[index] = wp.nan
        found[index] = 0
        gradient_valid[index] = 0
        if query.result:
            vector = point - wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            length = wp.length(vector)
            distances[index] = query.sign * length
            gradients[index] = query.sign * vector / wp.max(length, 1e-9)
            faces[index] = query.face
            signs[index] = query.sign
            found[index] = 1
            if length > 1e-9:
                gradient_valid[index] = 1

    _KERNELS[wp] = winding_query
    return winding_query


def _autograd_class(torch):
    if torch in _AUTOGRAD:
        return _AUTOGRAD[torch]

    class QueryDistance(torch.autograd.Function):
        @staticmethod
        def forward(ctx, points, backend):
            outputs = backend._query(points)
            gradient = outputs[1].to(points.dtype).reshape(points.shape)
            ctx.save_for_backward(gradient)
            distance = outputs[0].to(points.dtype).reshape(points.shape[:-1])
            diagnostics = tuple(
                value.reshape(points.shape[:-1]) for value in outputs[2:]
            )
            ctx.mark_non_differentiable(*diagnostics)
            return (distance, *diagnostics)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, incoming, *diagnostic_gradients):
            # Explicitly ignored missing queries must not poison unrelated
            # gradients through the otherwise indeterminate zero * NaN.
            gradient = torch.where(
                incoming[..., None] != 0,
                incoming[..., None] * ctx.saved_tensors[0],
                0.0,
            )
            return gradient, None

    _AUTOGRAD[torch] = QueryDistance
    return QueryDistance


class WarpSignedDistance:
    """Persistent original-mesh winding queries with point-only derivatives.

    Coordinates are in the bound mesh's local metric frame. Transform moving
    target/source points outside this class so autograd includes both bodies.
    Geometry is fixed and copied; no gradient for vertices/topology is promised.
    Warp uses float32 points and original topology, with radius10m, winding
    accuracy2 and threshold0.5 by default, matching the pinned reference query.
    Outputs return to the input Torch dtype. Missing queries are NaN with false
    ``found``; they are never invented negative penetration values.

    ``query_sign`` is Warp's winding classification. Even a closed consistently
    oriented mesh does not make it a certified solid sign. Open/nonmanifold
    meshes have unknown solid sign. Zero-distance derivatives have an explicit
    invalid flag; the conventional zero subgradient matches the reference API.
    Query branches/ties are nonsmooth and second derivatives are not provided.
    """

    def __init__(
        self, vertices, faces, *, device="cpu", radius=10.0, accuracy=2.0, threshold=0.5
    ):
        from .repair_geometry import TriangleMesh

        original = TriangleMesh(vertices, faces)
        if not len(original.vertices) or not len(original.faces):
            raise ValueError("Signed query geometry must be nonempty")
        if original.faces.max() > np.iinfo(np.int32).max:
            raise ValueError("Original mesh indices exceed Warp int32 range")
        if not original.normal_valid.all():
            raise ValueError("Original query mesh contains degenerate faces")
        cast = original.vertices.astype(np.float32)
        rounded = TriangleMesh(cast.astype(float), original.faces)
        if not np.isfinite(cast).all() or not rounded.normal_valid.all():
            raise ValueError(
                "Original query mesh is nonfinite/degenerate after float32 conversion"
            )
        for value, label in (
            (radius, "radius"),
            (accuracy, "accuracy"),
            (threshold, "threshold"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(label + " must be finite and positive")
        if threshold > 1:
            raise ValueError("winding threshold must be <=1")
        torch = _torch()
        try:
            import warp as wp
        except ImportError as exc:
            raise RuntimeError(
                "Signed mesh queries need Warp in the external worker"
            ) from exc
        self.torch, self.wp = torch, wp
        self.device = torch.device(device)
        self.wpdevice = wp.get_device(str(self.device))
        self.radius, self.accuracy, self.threshold = (
            float(radius),
            float(accuracy),
            float(threshold),
        )
        self.mesh = wp.Mesh(
            points=wp.array(cast, dtype=wp.vec3, device=self.wpdevice),
            indices=wp.array(
                original.faces.ravel().astype(np.int32),
                dtype=wp.int32,
                device=self.wpdevice,
            ),
            support_winding_number=True,
        )
        self.kernel = _query_kernel(wp)
        self.function = _autograd_class(torch)
        edges = np.concatenate(
            [
                original.faces[:, [0, 1]],
                original.faces[:, [1, 2]],
                original.faces[:, [2, 0]],
            ]
        )
        _, inverse, counts = np.unique(
            np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
        )
        balance = np.zeros(len(counts), dtype=np.int64)
        np.add.at(balance, inverse, np.where(edges[:, 0] < edges[:, 1], 1, -1))
        self.topology = {
            "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()),
            "oriented_closed": bool(np.all(counts == 2) and np.all(balance == 0)),
        }
        self.geometry_hash = hashlib.sha256(
            original.vertices.tobytes() + original.faces.astype(np.int64).tobytes()
        ).hexdigest()
        self.max_cast_error = float(
            np.max(np.abs(cast.astype(float) - original.vertices))
        )

    def _query(self, points):
        torch, wp = self.torch, self.wp
        if (
            not isinstance(points, torch.Tensor)
            or points.ndim < 2
            or points.shape[-1] != 3
            or points.device != self.device
            or points.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError(
                "Query points must be floating Torch coordinates on the configured device"
            )
        packed = points.detach().reshape(-1, 3).float().contiguous()
        count = len(packed)
        distance = torch.empty(count, dtype=torch.float32, device=self.device)
        gradient = torch.empty_like(packed)
        faces = torch.empty(count, dtype=torch.int32, device=self.device)
        signs = torch.empty(count, dtype=torch.float32, device=self.device)
        found = torch.empty(count, dtype=torch.int32, device=self.device)
        valid = torch.empty(count, dtype=torch.int32, device=self.device)
        if count:
            stream = (
                wp.ScopedStream(
                    wp.stream_from_torch(torch.cuda.current_stream(self.device))
                )
                if self.device.type == "cuda"
                else contextlib.nullcontext()
            )
            with stream:
                wp.launch(
                    self.kernel,
                    dim=count,
                    inputs=[
                        self.mesh.id,
                        wp.from_torch(packed, dtype=wp.vec3),
                        self.radius,
                        self.accuracy,
                        self.threshold,
                    ],
                    outputs=[
                        wp.from_torch(distance),
                        wp.from_torch(gradient, dtype=wp.vec3),
                        wp.from_torch(faces),
                        wp.from_torch(signs),
                        wp.from_torch(found),
                        wp.from_torch(valid),
                    ],
                    device=self.wpdevice,
                )
        return distance, gradient, faces, signs, found, valid

    def query(self, points):
        distances, faces, signs, found, gradient_valid = self.function.apply(
            points, self
        )
        return {
            "distances": distances,
            "face_ids": faces,
            "query_sign": signs,
            "found": found.bool(),
            "gradient_valid": gradient_valid.bool(),
            "solid_sign_certified": False,
        }

    def distance(self, points):
        # The autograd query is identical; avoid constructing unused boolean
        # diagnostics (and launching their CUDA conversions) in loss-only calls.
        return self.function.apply(points, self)[0]

    def metadata(self):
        return {
            "backend": "warp_winding_query_point_gradient",
            "warp_version": self.wp.__version__,
            "torch_version": self.torch.__version__,
            "device": str(self.device),
            "geometry_sha256": self.geometry_hash,
            "max_float32_vertex_error": self.max_cast_error,
            "topology": self.topology.copy(),
            "radius_m": self.radius,
            "winding_accuracy": self.accuracy,
            "winding_threshold": self.threshold,
            "source_idea_commit": REFERENCE_COMMIT,
            "solid_sign_certified": False,
            "unknown_query": "NaN distance/gradient, found=False",
            "acceptance": False,
        }
