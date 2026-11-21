"""Sampled signed-clearance residuals inside the external Newton IK solve.

Uses the same original-mesh Warp winding query as repair_tensor, without a Torch
dependency. Both moving bodies contribute analytic derivatives. Grouped RMS
keeps the dense IK residual dimension bounded while retaining every query in
the objective value. It is not an exact collision/containment certificate.
"""
import json
import numpy as np
import warp as wp
import newton.ik as ik


@wp.func
def _pose(body_q: wp.array2d(dtype=wp.transform), row: int, body: int,
          obj: wp.array(dtype=wp.transform)):
    result = obj[0]
    if body >= 0:
        result = body_q[row, body]
    elif body == -2:
        # Static-scene vertices are already in solver world coordinates.
        result = wp.transform_identity()
    return result


@wp.kernel
def query_clearance(body_q: wp.array2d(dtype=wp.transform), obj: wp.array(dtype=wp.transform),
                    meshes: wp.array(dtype=wp.uint64), bodies: wp.array(dtype=int),
                    lower: wp.array(dtype=wp.vec3), upper: wp.array(dtype=wp.vec3),
                    points: wp.array(dtype=wp.vec3), aa: wp.array(dtype=int), bb: wp.array(dtype=int),
                    margin: float, enabled: wp.array(dtype=float),
                    penalty: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
                    world_points: wp.array2d(dtype=wp.vec3), failures: wp.array(dtype=int)):
    row, q = wp.tid()
    penalty[row, q] = 0.0
    gradients[row, q] = wp.vec3(0.0)
    world_points[row, q] = wp.vec3(0.0)
    if enabled[0] > 0.0:
        source = _pose(body_q, row, bodies[aa[q]], obj)
        target = _pose(body_q, row, bodies[bb[q]], obj)
        p = wp.transform_point(source, points[q])
        local = wp.transform_point(wp.transform_inverse(target), p)
        world_points[row, q] = p
        lo, hi = lower[bb[q]], upper[bb[q]]
        # A point outside the expanded box is outside the entire target mesh.
        if (local[0] >= lo[0]-margin and local[0] <= hi[0]+margin and
                local[1] >= lo[1]-margin and local[1] <= hi[1]+margin and
                local[2] >= lo[2]-margin and local[2] <= hi[2]+margin):
            radius = wp.length(hi-lo) + margin + 0.01
            hit = wp.mesh_query_point_sign_winding_number(meshes[bb[q]], local, radius, 2.0, 0.5)
            if hit.result:
                closest = wp.mesh_eval_position(meshes[bb[q]], hit.face, hit.u, hit.v)
                delta = local - closest
                length = wp.length(delta)
                gap = hit.sign * length
                violation = wp.max(margin-gap, 0.0)
                penalty[row, q] = violation
                if violation > 0.0:
                    normal = hit.sign * delta / wp.max(length, 1e-9)
                    if length <= 1e-9:
                        normal = wp.mesh_eval_face_normal(meshes[bb[q]], hit.face)
                    gradients[row, q] = wp.transform_vector(target, normal)
            else:
                wp.atomic_add(failures, 0, 1)
                penalty[row, q] = wp.nan


@wp.kernel
def pair_residual(penalty: wp.array2d(dtype=float), offsets: wp.array(dtype=int), query_weights: wp.array(dtype=float),
                  weight: float, start: int, residuals: wp.array2d(dtype=float)):
    row, pair = wp.tid()
    total = float(0.0)
    for q in range(offsets[pair], offsets[pair+1]):
        total += query_weights[q]*penalty[row, q]*penalty[row, q]
    residuals[row, start+pair] = weight * wp.sqrt(total)


@wp.kernel
def pair_jacobian(penalty: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
                  world_points: wp.array2d(dtype=wp.vec3), offsets: wp.array(dtype=int), query_weights: wp.array(dtype=float),
                  aa: wp.array(dtype=int), bb: wp.array(dtype=int), masks: wp.array2d(dtype=int),
                  motions: wp.array2d(dtype=wp.spatial_vector), weight: float, start: int,
                  jacobian: wp.array3d(dtype=float)):
    row, pair, dof = wp.tid()
    total, derivative = float(0.0), float(0.0)
    s = motions[row, dof]
    for q in range(offsets[pair], offsets[pair+1]):
        p = penalty[row, q]
        total += query_weights[q]*p*p
        # A common ancestor transforms both meshes equally; cancellation is exact.
        mask = masks[aa[q], dof] - masks[bb[q], dof]
        if mask != 0 and p > 0.0:
            velocity = wp.vec3(s[0], s[1], s[2]) + wp.cross(wp.vec3(s[3], s[4], s[5]), world_points[row, q])
            derivative -= query_weights[q] * p * float(mask) * wp.dot(gradients[row, q], velocity)
    value = float(0.0)
    if total > 1e-20:
        value = weight * derivative / wp.sqrt(total)
    jacobian[row, start+pair, dof] = value


class CollisionObjective(ik.IKObjective):
    def __init__(self, schedule, body_labels):
        super().__init__()
        self.metadata = json.loads(str(schedule["collision_metadata"].item()))
        self.margin = float(schedule["collision_margin"])
        self.weight = float(np.sqrt(schedule["collision_weight"]))
        body_types = (schedule["collision_body_types"] if "collision_body_types" in schedule
                      else ["object" if x == "object" else "robot" for x in schedule["collision_bodies"]])
        if (len(body_types) != len(schedule["collision_bodies"])
                or any(kind not in ("object", "robot", "static") for kind in body_types)):
            raise ValueError("Invalid collision mesh transform owners")
        self.body_ids = [-2 if kind == "static" else -1 if kind == "object" else body_labels.index(str(name))
                         for name, kind in zip(schedule["collision_bodies"], body_types)]
        self.meshes = []
        lo, hi = [], []
        vo = schedule["collision_vertex_offsets"]
        query_faces = schedule["collision_faces"]
        fo = schedule["collision_face_offsets"]
        if "collision_query_mesh_faces" in schedule:
            query_faces = schedule["collision_query_mesh_faces"]
            fo = schedule["collision_query_mesh_face_offsets"]
        for i in range(len(self.body_ids)):
            v = schedule["collision_vertices"][vo[i]:vo[i+1]].astype(np.float32)
            f = query_faces[fo[i]:fo[i+1]].astype(np.int32)
            self.meshes.append(wp.Mesh(points=wp.array(v, dtype=wp.vec3),
                indices=wp.array(f.ravel(), dtype=int), support_winding_number=True))
            lo.append(v.min(0)); hi.append(v.max(0))
        self.mesh_ids = wp.array([m.id for m in self.meshes], dtype=wp.uint64)
        self.bodies = wp.array(self.body_ids, dtype=int)
        self.lower, self.upper = wp.array(lo, dtype=wp.vec3), wp.array(hi, dtype=wp.vec3)
        self.points = wp.array(schedule["collision_query_points"], dtype=wp.vec3)
        self.aa = wp.array(schedule["collision_query_a"], dtype=int)
        self.bb = wp.array(schedule["collision_query_b"], dtype=int)
        pair_offsets = schedule["collision_pair_offsets"]
        query_weights = np.repeat(1./np.diff(pair_offsets), np.diff(pair_offsets))
        if 'collision_query_weights' in schedule:
            query_weights = np.asarray(schedule['collision_query_weights'], dtype=float)
            if (query_weights.shape != (len(schedule['collision_query_points']),)
                    or not np.isfinite(query_weights).all() or np.any(query_weights <= 0)
                    or not np.isfinite(query_weights.astype(np.float32)).all()
                    or np.any(query_weights.astype(np.float32) <= 0)):
                raise ValueError('Explicit collision query weights must be finite positive and cover every original query')
        self.query_weights_cpu = query_weights.astype(np.float32).astype(float)
        body_ids = np.asarray(self.body_ids)
        self.object_query_mask = ((body_ids[schedule["collision_query_a"]] == -1).astype(int)
                                  - (body_ids[schedule["collision_query_b"]] == -1).astype(int))
        # Bound residual rows without dropping any pairs. Squared residuals still
        # sum to weight * sum_pair(mean(hinge^2)); only the GN approximation changes.
        # The upstream dense tiled LM exceeds Ada shared memory with 225 extra rows.
        stride = max(1, int(np.ceil((len(pair_offsets)-1)/32)))
        offsets = pair_offsets[::stride].tolist()
        if offsets[-1] != pair_offsets[-1]:
            offsets.append(int(pair_offsets[-1]))
        self.offsets = wp.array(offsets, dtype=int)
        self.query_weights = wp.array(query_weights, dtype=float)
        self.metadata["residual_groups"] = len(offsets)-1
        self.metadata["grouping"] = "Consecutive mesh-pair means; all samples retained, exact same squared loss"
        self.object_pose = wp.zeros(1, dtype=wp.transform)
        self.enabled = wp.zeros(1, dtype=float)
        self.failures = wp.zeros(1, dtype=int)

    def residual_dim(self):
        return len(self.offsets)-1

    def supports_analytic(self):
        return True

    def init_buffers(self, model, jacobian_mode):
        if jacobian_mode != ik.IKJacobianType.ANALYTIC:
            raise ValueError("Collision objective requires analytic Jacobians")
        masks = []
        for body in self.body_ids:
            if body < 0:
                masks.append(np.zeros(model.joint_dof_count, dtype=int))
                continue
            # Reuse Newton's articulation ancestry construction for exact native FK.
            point = ik.IKObjectivePosition(body, wp.vec3(0.), wp.zeros(1, dtype=wp.vec3))
            point.bind_device(self.device)
            point.set_batch_layout(self.total_residuals, self.residual_offset, self.n_batch)
            point.init_buffers(model, jacobian_mode)
            masks.append(point.affects_dof.numpy().astype(int))
        self.masks = wp.array(np.asarray(masks), dtype=int)
        shape = (self.n_batch, len(self.points))
        self.penalties = wp.zeros(shape, dtype=float)
        self.gradients = wp.zeros(shape, dtype=wp.vec3)
        self.world_points = wp.zeros(shape, dtype=wp.vec3)

    def update(self, pose, enabled=True):
        self.object_pose.assign(np.asarray(pose, dtype=np.float32).reshape(1, 7))
        self.enabled.assign(np.asarray([float(enabled)], dtype=np.float32))

    def query(self, body_q):
        wp.launch(query_clearance, dim=(body_q.shape[0], len(self.points)), inputs=[
            body_q, self.object_pose, self.mesh_ids, self.bodies, self.lower, self.upper,
            self.points, self.aa, self.bb, self.margin, self.enabled,
            self.penalties, self.gradients, self.world_points, self.failures], device=self.device)

    def object_energy_gradient(self, body_q, rotation, translation):
        """The same squared clearance loss and its editable-object derivative.

        Robot transforms are fixed for this conditional update. The six-vector
        differentiates world translation followed by a world-left rotation
        tangent about the object's origin, not about the world origin. Both
        query directions contribute, and robot-only terms remain in the energy.
        This single-frame host evaluation is outside Newton's captured graph.
        """
        from scipy.spatial.transform import Rotation

        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                # Match the contact compiler/ObjectTrajectory precision for
                # immutable upstream rotations. Do not reject ordinary float32
                # reconstruction roundoff or rewrite the stored reference.
                or not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-5)):
            raise ValueError("Object collision evaluation requires a finite rigid pose")
        if body_q.shape[0] != 1 or self.n_batch != 1:
            raise ValueError("Object collision evaluation requires a single frame")
        self.update(np.r_[translation, Rotation.from_matrix(rotation).as_quat()])
        self.query(body_q)
        self.check_queries()
        penalties = self.penalties.numpy()[0].astype(float)
        normals = self.gradients.numpy()[0].astype(float)
        points = self.world_points.numpy()[0].astype(float)
        if not (np.isfinite(penalties).all() and np.isfinite(normals).all()
                and np.isfinite(points).all()):
            raise RuntimeError("Nonfinite editable-object collision query")
        energy = self.weight ** 2 * np.dot(self.query_weights_cpu, penalties ** 2)
        coefficient = (-2 * self.weight ** 2 * self.query_weights_cpu * penalties
                       * self.object_query_mask)
        point_derivatives = np.c_[normals, np.cross(points - translation, normals)]
        return float(energy), np.sum(coefficient[:, None] * point_derivatives, axis=0)

    def object_context(self, body_q, rotation, translation):
        """Freeze one robot state for an exact conditional object update.

        Create a fresh context after every robot update. The complete query at
        construction retains robot-only energy and failure checks; later calls
        recompute all object-dependent queries with separate buffers.
        """
        return ObjectFrameCollisionContext(self, body_q, rotation, translation)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        self.query(body_q)
        wp.launch(pair_residual, dim=(body_q.shape[0], self.residual_dim()), inputs=[
            self.penalties, self.offsets, self.query_weights, self.weight, start_idx, residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        self.query(body_q)
        wp.launch(pair_jacobian, dim=(body_q.shape[0], self.residual_dim(), model.joint_dof_count),
            inputs=[self.penalties, self.gradients, self.world_points, self.offsets, self.query_weights, self.aa, self.bb,
                    self.masks, joint_S_s, self.weight, start_idx, jacobian], device=self.device)

    def check_queries(self):
        if int(self.failures.numpy()[0]):
            raise RuntimeError("Collision winding query failed; result cannot be certified by this objective")


class ObjectFrameCollisionContext:
    """Exact sampled object loss with one cloned, fixed robot pose.

    Original robot-only queries contribute their constant energy throughout
    this conditional update. Neither their policy nor the full IK objective's
    query arrays are changed. Only immutable object query indices/weights are
    shared between contexts; robot poses and evaluation buffers are private.
    """
    def __init__(self, objective, body_q, rotation, translation):
        self.objective = objective
        self.body_q = wp.clone(body_q)
        # Evaluate every original query before caching the constant part. This
        # also validates the pose, batch size, finiteness and sticky failures.
        objective.object_energy_gradient(self.body_q, rotation, translation)
        mask = objective.object_query_mask != 0
        penalty = objective.penalties.numpy()[0].astype(float)
        self.constant_energy = float(objective.weight ** 2 *
            np.dot(objective.query_weights_cpu[~mask], penalty[~mask] ** 2))
        if not hasattr(objective, "_object_subset"):
            device = objective.device
            objective._object_subset = dict(
                points=wp.array(objective.points.numpy()[mask], dtype=wp.vec3, device=device),
                aa=wp.array(objective.aa.numpy()[mask], dtype=int, device=device),
                bb=wp.array(objective.bb.numpy()[mask], dtype=int, device=device),
                weights=objective.query_weights_cpu[mask].copy(),
                masks=objective.object_query_mask[mask].copy())
        self.subset = objective._object_subset
        shape = (1, len(self.subset["points"]))
        self.penalties = wp.zeros(shape, dtype=float, device=objective.device)
        self.normals = wp.zeros(shape, dtype=wp.vec3, device=objective.device)
        self.points = wp.zeros(shape, dtype=wp.vec3, device=objective.device)

    def __call__(self, rotation, translation):
        from scipy.spatial.transform import Rotation

        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-5)):
            raise ValueError("Object collision evaluation requires a finite rigid pose")
        owner = self.objective
        owner.update(np.r_[translation, Rotation.from_matrix(rotation).as_quat()])
        subset = self.subset
        if len(subset["points"]):
            wp.launch(query_clearance, dim=(1, len(subset["points"])), inputs=[
                self.body_q, owner.object_pose, owner.mesh_ids, owner.bodies, owner.lower, owner.upper,
                subset["points"], subset["aa"], subset["bb"], owner.margin, owner.enabled,
                self.penalties, self.normals, self.points, owner.failures], device=owner.device)
        owner.check_queries()
        penalty = self.penalties.numpy()[0].astype(float)
        normals = self.normals.numpy()[0].astype(float)
        points = self.points.numpy()[0].astype(float)
        if not (np.isfinite(penalty).all() and np.isfinite(normals).all() and np.isfinite(points).all()):
            raise RuntimeError("Nonfinite editable-object collision query")
        energy = self.constant_energy + owner.weight ** 2 * np.dot(subset["weights"], penalty ** 2)
        coefficient = -2 * owner.weight ** 2 * subset["weights"] * penalty * subset["masks"]
        derivative = np.c_[normals, np.cross(points - translation, normals)]
        return float(energy), np.sum(coefficient[:, None] * derivative, axis=0)
