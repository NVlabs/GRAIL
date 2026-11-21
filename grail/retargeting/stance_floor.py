"""Opt-in G1 stance correction on a native motion with a registered finite floor."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

from grail.core.bindings import sha256

from .stance_clearance import ObjectClearance, new_contact_intervals

# Native quantities are metres, radians and seconds.
TARGET_HEIGHT = -0.003
MIN_HEIGHT = -0.0043
TRANSITION_HEIGHT = 0.018
HEIGHT_WEIGHT = 35000.0
JERK_WEIGHT_AT_24_FPS = 6400.0


def _cross_xy(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class Floor:
    """Top surface of the actual floor mesh, evaluated at the sole witness."""

    def __init__(self, native):
        scene = json.loads(str(native["static_scene_json"]))
        floors = [
            m
            for m in scene["meshes"]
            if m.get("role") == "floor" or m["name"].split("/")[-1].split(".")[-1] == "floor"
        ]
        if len(floors) != 1:
            raise ValueError("Expected one registered floor mesh in the native scene")
        mesh = floors[0]
        self.name = mesh["name"]
        vertices, faces = native[mesh["vertices_key"]], native[mesh["faces_key"]]
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or faces.dtype.kind not in "iu"
            or faces.min() < 0
            or faces.max() >= len(vertices)
        ):
            raise ValueError("Invalid registered floor mesh indices")
        triangles = vertices[faces]
        a, b, c = (
            triangles[:, 0],
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        det = _cross_xy(b, c)
        valid = np.abs(det) > 1e-10
        if not valid.any() or not np.isfinite(triangles).all():
            raise ValueError("Invalid finite floor geometry")
        self.a, self.b, self.c, self.det = [x[valid] for x in (a, b, c, det)]
        self.top = triangles[valid, :, 2].max()

    def surface(self, point):
        delta = point - self.a
        u = _cross_xy(delta, self.c) / self.det
        v = _cross_xy(self.b, delta) / self.det
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if not inside.any():
            return None
        candidates = np.flatnonzero(inside)
        heights = self.a[:, 2] + u * self.b[:, 2] + v * self.c[:, 2]
        index = candidates[np.argmax(heights[candidates])]
        normal = np.cross(self.b[index], self.c[index])
        return heights[index], normal / normal[2]

    def witness(self, vertices):
        """Find the original sole vertex with least vertical clearance over this floor."""
        bottom = vertices[vertices[:, 2].argmin()]
        surface = self.surface(bottom)
        if surface is not None and abs(surface[0] - self.top) < 1e-10:
            return bottom, surface
        # At a finite edge or on a slope, the lowest world-Z vertex need not
        # be the most penetrating vertex over the actual registered triangles.
        highest = np.full(len(vertices), -np.inf)
        for a, b, c, det in zip(self.a, self.b, self.c, self.det):
            delta = vertices - a
            u = _cross_xy(delta, c) / det
            v = _cross_xy(b, delta) / det
            inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
            height = a[2] + u[inside] * b[2] + v[inside] * c[2]
            highest[inside] = np.maximum(highest[inside], height)
        if not np.isfinite(highest).any():
            return None
        point = vertices[np.argmin(vertices[:, 2] - highest)]
        return point, self.surface(point)


class Leg:
    def __init__(self, model, side):
        self.body = model.body(f"{side}_ankle_roll_link").id
        labels = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        joints = [model.joint(f"{side}_{label}_joint").id for label in labels]
        self.qpos = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        self.limits = model.jnt_range[joints].T
        self.meshes = []
        for geom in np.flatnonzero(model.geom_bodyid == self.body):
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                start = model.mesh_vertadr[mesh]
                self.meshes.append(
                    (geom, model.mesh_vert[start : start + model.mesh_vertnum[mesh]])
                )
        if not self.meshes:
            raise ValueError(f"No native sole geometry on the {side} foot")

    def vertices(self, data):
        return np.concatenate(
            [
                local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                for geom, local in self.meshes
            ]
        )

    def lift(self, model, data, distance):
        """Move this ankle vertically, preserving its orientation and horizontal location."""
        target = data.xpos[self.body].copy()
        target[2] += distance
        orientation = data.xmat[self.body].reshape(3, 3).copy()
        linear = np.empty((3, model.nv))
        angular = np.empty_like(linear)
        # Start on the flexed-knee branch when the source knee is nearly straight.
        # Both hip and ankle counter-rotate to avoid changing foot pitch.
        knee = data.qpos[self.qpos[3]]
        seed = max(0.0, np.sqrt(max(knee, 0.0) ** 2 + distance * 12.0) - knee)
        data.qpos[self.qpos[[0, 3, 4]]] += seed * np.array([-0.5, 1.0, -0.5])
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for _ in range(45):
            delta = target - data.xpos[self.body]
            turn = Rotation.from_matrix(
                orientation @ data.xmat[self.body].reshape(3, 3).T
            ).as_rotvec()
            error = np.concatenate((delta, turn))
            if np.max(np.abs(error)) < 2e-6:
                return
            mujoco.mj_jacBody(model, data, linear, angular, self.body)
            jac = np.vstack((linear[:, self.dofs], angular[:, self.dofs]))
            step = np.linalg.solve(jac.T @ jac + 1e-5 * np.eye(6), jac.T @ error)
            data.qpos[self.qpos] = np.clip(
                data.qpos[self.qpos] + np.clip(step, -0.08, 0.08), *self.limits
            )
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
        raise ValueError("Leg joint limits or singularity prevent the requested ankle lift")


def _frame_rows(rows, frames, shape):
    """Place each local hinge Jacobian in its frame block of the global matrix."""
    frame_width = shape[1]
    cols = (np.asarray(frames)[:, None] * frame_width + np.arange(frame_width)).ravel()
    indices = np.repeat(np.arange(len(frames)), frame_width)
    return sparse.coo_matrix(
        (np.asarray(rows).ravel(), (indices, cols)), shape=(len(frames), np.prod(shape))
    ).tocsc()


def _smooth_leg(model, data, leg, floor, original, target, heights, fps, protected):
    """Minimize angle change and angular jerk subject to joint and finite-floor bounds."""
    # A near-contact episode may need the native leg pose. Ramp the correction
    # back over 0.30 s with zero first and second derivatives at both ends.
    if not np.any(heights < TARGET_HEIGHT):
        return
    count = len(original)
    distance = (
        distance_transform_edt(~protected) / max(0.30 * fps, 1)
        if protected.any()
        else np.ones(count)
    )
    t = np.clip(distance, 0, 1)
    blend = t**3 * (10 + t * (-15 + 6 * t))
    strength = blend**2
    min_heights = np.minimum(heights, MIN_HEIGHT)
    min_heights += strength * (MIN_HEIGHT - min_heights)
    preferred = np.minimum(heights, TARGET_HEIGHT)
    preferred += strength * (TARGET_HEIGHT - preferred)
    reference = original[:, leg.qpos] + strength[:, None] * (
        target[:, leg.qpos] - original[:, leg.qpos]
    )
    angles = reference.copy()
    # Permit blending at the stance/swing transition; freeze airborne and unobserved feet.
    supported = np.isfinite(heights) & (heights < TRANSITION_HEIGHT) & (strength > 1e-8)
    target[:, leg.qpos] = original[:, leg.qpos]
    if not supported.any():
        return
    identity = sparse.eye(angles.size, format="csc")
    jerk = (
        sparse.diags([-1, 3, -3, 1], range(4), shape=(count - 3, count))
        if count > 3
        else sparse.csc_matrix((0, count))
    )
    smooth = sparse.kron(jerk.T @ jerk, sparse.eye(len(leg.qpos)), format="csc")
    quadratic = identity + (JERK_WEIGHT_AT_24_FPS * (fps / 24) ** 6) * smooth
    lower = np.broadcast_to(leg.limits[0], angles.shape).copy()
    upper = np.broadcast_to(leg.limits[1], angles.shape).copy()
    source_angles = original[:, leg.qpos]
    lower = np.maximum(lower, source_angles - blend[:, None])
    upper = np.minimum(upper, source_angles + blend[:, None])
    lower[~supported] = upper[~supported] = original[~supported][:, leg.qpos]
    linear = np.empty((3, model.nv))
    active = np.flatnonzero(supported)
    witnesses = {frame: {} for frame in active}
    angular = np.empty_like(linear)
    poses = {}
    for frame in active:
        data.qpos[:] = original[frame]
        mujoco.mj_kinematics(model, data)
        poses[frame] = (data.xpos[leg.body, :2].copy(), data.xmat[leg.body].reshape(3, 3).copy())
    for iteration in range(25):
        placement_jac = []
        placement_error = []
        frame_ids = []
        tracking_rows = []
        jacobians = []
        bounds = []
        for frame in active:
            data.qpos[:] = original[frame]
            data.qpos[leg.qpos] = angles[frame]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            contact = floor.witness(leg.vertices(data))
            if contact is None:
                raise ValueError("Foot moved outside the registered support")
            point = contact[0]
            rotation = data.xmat[leg.body].reshape(3, 3)
            origin = data.xpos[leg.body]
            local = rotation.T @ (point - origin)
            original_xy, original_rotation = poses[frame]
            mujoco.mj_jacBody(model, data, linear, angular, leg.body)
            placement_jac.append(np.vstack((linear[:2, leg.dofs], angular[:, leg.dofs])))
            turn = Rotation.from_matrix(original_rotation @ rotation.T).as_rotvec()
            placement_error.extend(np.r_[original_xy - origin[:2], turn])
            lowest = tuple(np.round(local, 10))
            witnesses[frame][lowest] = local
            for index, local in witnesses[frame].items():
                point = rotation @ local + origin
                surface = floor.surface(point)
                if surface is None:
                    continue
                height, normal = surface
                mujoco.mj_jac(model, data, linear, None, point, leg.body)
                if index == lowest and heights[frame] < TARGET_HEIGHT:
                    tracking_rows.append(len(bounds))
                jacobians.append(normal @ linear[:, leg.dofs])
                bounds.append(min_heights[frame] - (point[2] - height))
                frame_ids.append(frame)
        errors = np.array(placement_error).reshape(-1, 5)
        tolerance = np.array([0.0025, 0.0025, 0.03, 0.03, 0.03])
        if iteration >= 2 and max(bounds) < 0.0001 and np.max(np.abs(errors) - tolerance) < 0.0001:
            break
        # Linearize the unilateral sole height at the current original-mesh witness.
        jac = _frame_rows(jacobians, frame_ids, angles.shape)
        foot_fit = jac[tracking_rows]
        world_offset = (
            np.asarray(bounds)[tracking_rows]
            + (preferred - min_heights)[np.asarray(frame_ids)[tracking_rows]]
            + foot_fit @ angles.ravel()
        )
        cost = quadratic + HEIGHT_WEIGHT * (foot_fit.T @ foot_fit)
        linear_term = -reference.ravel() - HEIGHT_WEIGHT * (foot_fit.T @ world_offset)
        pose_jac = _frame_rows(placement_jac, np.repeat(active, 5), angles.shape)
        center = errors.ravel() + pose_jac @ angles.ravel()
        corridor = np.tile(tolerance, len(active))
        weights = np.tile([32000.0, 32000.0, 100.0, 100.0, 100.0], len(active))
        weighted = pose_jac.T @ sparse.diags(weights)
        cost = cost + weighted @ pose_jac
        linear_term = linear_term - weighted @ center
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.triu(cost, format="csc"),
            q=linear_term,
            A=sparse.vstack((identity, jac, pose_jac), format="csc"),
            l=np.r_[lower.ravel(), bounds + jac @ angles.ravel(), center - corridor],
            u=np.r_[upper.ravel(), np.full(len(bounds), np.inf), center + corridor],
            verbose=False,
            eps_abs=8e-7,
            eps_rel=8e-7,
            max_iter=20000,
        )
        solved = solver.solve()
        if solved.info.status_val not in (1, 2):
            raise ValueError(f"Stance trajectory solve failed: {solved.info.status}")
        angles = solved.x.reshape(reference.shape)
    else:
        raise ValueError("Leg correction did not reach the measured sole-height tolerance")
    target[np.ix_(active, leg.qpos)] = angles[active]


def correct(model, native, clearance):
    """Correct sole penetration without moving the base, arms, hands or scene."""
    floor = Floor(native)
    legs = [Leg(model, side) for side in ("left", "right")]
    qpos = native["qpos"].copy()
    fps = float(native["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid native frame rate")
    if (
        qpos.ndim != 2
        or qpos.shape[1] != model.nq
        or not len(qpos)
        or qpos.dtype.kind != "f"
        or not np.isfinite(qpos).all()
    ):
        raise ValueError("Invalid native qpos")
    data = mujoco.MjData(model)
    heights = np.full((len(qpos), 2), np.nan)
    for i, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for side, leg in enumerate(legs):
            contact = floor.witness(leg.vertices(data))
            if contact is None:
                continue
            point, ground = contact
            heights[i, side] = point[2] - ground[0]
            # Only push out of the support surface. Never pull a swing foot down.
            lift = TARGET_HEIGHT - heights[i, side]
            if lift > 0:
                leg.lift(model, data, lift)
                frame[leg.qpos] = data.qpos[leg.qpos]
    if not np.isfinite(heights).any(axis=0).all():
        raise ValueError("The registered floor does not cover both feet")
    guards = {}
    for side, leg in enumerate(legs):
        original = native["qpos"]
        lifted = qpos[:, leg.qpos].copy()
        protected = np.zeros(len(original), dtype=bool)
        if not np.any(heights[:, side] < TARGET_HEIGHT):
            guards[("left", "right")[side]] = []
            continue
        baseline = clearance.distances(original, side)
        for _ in range(5):
            qpos[:, leg.qpos] = lifted
            _smooth_leg(model, data, leg, floor, original, qpos, heights[:, side], fps, protected)
            contact = new_contact_intervals(baseline, clearance.distances(qpos, side))
            if not np.any(contact):
                break
            if np.all(protected[contact]):
                raise ValueError("Object clearance could not be restored on protected frames")
            protected |= contact
        else:
            raise ValueError("New leg/object contact persists after stance correction")
        guards[("left", "right")[side]] = np.flatnonzero(protected).tolist()
    return qpos, {
        "floor": floor.name,
        "contact_protected_frames": guards,
        "target_clearance_m": TARGET_HEIGHT,
        "minimum_clearance_m": MIN_HEIGHT,
        "transition_height_m": TRANSITION_HEIGHT,
        "support_estimator": "minimum ankle-mesh vertex clearance over the registered finite floor",
        "edited_joints": [
            model.joint(int(model.dof_jntid[col])).name for leg in legs for col in leg.dofs
        ],
        "solver": {
            "height_weight": HEIGHT_WEIGHT,
            "jerk_weight_at_24_fps": JERK_WEIGHT_AT_24_FPS,
            "ankle_xy_tolerance_m_per_axis": 0.0025,
            "ankle_rotation_tolerance_rad_per_axis": 0.03,
            "object_contact_transition_s": 0.30,
        },
        "observed_frames": np.isfinite(heights).sum(axis=0).tolist(),
        "penetrating_frames": (heights < TARGET_HEIGHT).sum(axis=0).tolist(),
    }


def write_candidate(source, model_xml, output, object_mesh=None):
    source, model_xml, output = map(Path, (source, model_xml, output))
    if output.exists():
        raise FileExistsError(output)
    with np.load(source, allow_pickle=False) as archive:
        native = dict(archive)
    if (
        str(native["world_frame"]) != "soma_retargeter_z_up"
        or str(native["export_kind"])
        != "soma_initialization_requires_native_contact_and_visual_review"
    ):
        raise ValueError("Expected the source-clock native SOMA initialization export")
    allowed = {
        "qpos",
        "fps",
        "timestamps",
        "dof_names",
        "qposadr",
        "source_path",
        "source_sha256",
        "model_xml_path",
        "model_xml_sha256",
        "compiled_model_sha256",
        "world_frame",
        "native_status",
        "selected",
        "export_kind",
        "obj_R",
        "obj_t",
        "object_scale",
        "object_mesh_path",
        "object_mesh_sha256",
        "retarget_provenance_json",
        "object_reference_R",
        "object_reference_t",
        "object_optimization_json",
        "static_scene_json",
    }
    scene = json.loads(str(native["static_scene_json"]))
    for mesh in scene["meshes"]:
        allowed.update((mesh["vertices_key"], mesh["faces_key"]))
    if set(native) - allowed:
        raise ValueError(f"Unexpected fields in fresh SOMA export: {sorted(set(native)-allowed)}")
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, compiled)
    if (
        str(native["model_xml_sha256"]) != sha256(model_xml)
        or str(native["compiled_model_sha256"]) != hashlib.sha256(compiled).hexdigest()
    ):
        raise ValueError("Native robot model binding differs from the supplied model")
    names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    indices = [int(model.jnt_qposadr[model.joint(name).id]) for name in names]
    if not np.array_equal(names, native["dof_names"]) or not np.array_equal(
        indices, native["qposadr"]
    ):
        raise ValueError("Native joint order differs from the bound model")
    times = native["timestamps"]
    if (
        times.shape != (len(native["qpos"]),)
        or not np.isfinite(times).all()
        or not np.allclose(np.diff(times) * float(native["fps"]), 1.0, rtol=2e-4, atol=1e-5)
    ):
        raise ValueError("Expected a uniform native source clock")
    object_mesh = Path(object_mesh or str(native["object_mesh_path"])).resolve(strict=True)
    if sha256(object_mesh) != str(native["object_mesh_sha256"]):
        raise ValueError("Native object mesh binding differs from the supplied CAD")
    native["qpos"], receipt = correct(model, native, ObjectClearance(model, native, object_mesh))
    receipt.update(
        parent_sha256=sha256(source),
        parent=str(source),
        object_mesh_sha256=str(native["object_mesh_sha256"]),
        model_xml_sha256=str(native["model_xml_sha256"]),
        method="G1 lower-limb unilateral finite-floor IK",
        selected=False,
    )
    native["stance_floor_json"] = np.asarray(json.dumps(receipt, sort_keys=True))
    native["native_status"] = np.asarray("UNREVIEWED")
    native["selected"] = np.asarray(False)
    native["export_kind"] = np.asarray("soma_stance_floor_candidate")
    # Publish only a complete archive and preserve any candidate already at this path.
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz") as stream:
        np.savez_compressed(stream, **native)
        stream.flush()
        os.link(stream.name, output)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "model-xml", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, help="Relocated original object CAD")
    args = parser.parse_args()
    print(
        json.dumps(
            write_candidate(args.source, args.model_xml, args.output, args.object_mesh), indent=2
        )
    )
