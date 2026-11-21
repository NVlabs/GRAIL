"""Original-mesh clearance used by the optional G1 stance solver."""

from pathlib import Path

import hppfcl as fcl
import mujoco
import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation


def _bvh(vertices, faces):
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Invalid original triangle mesh")
    points = fcl.StdVec_Vec3f()
    triangles = fcl.StdVec_Triangle()
    for point in vertices:
        points.append(point)
    for face in faces:
        triangles.append(fcl.Triangle(*map(int, face)))
    mesh = fcl.BVHModelOBBRSS()
    if (
        mesh.beginModel(len(faces), len(vertices)) != 0
        or mesh.addSubModel(points, triangles) != 0
        or mesh.endModel() != 0
    ):
        raise ValueError("Unable to build the original triangle mesh BVH")
    return mesh


class ObjectClearance:
    """Unsigned separation works for the original triangle surfaces, including open CAD."""

    def __init__(self, model, native, mesh_path):
        self.model = model
        scale = np.broadcast_to(native["object_scale"], (3,)).copy()
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid original object scale")
        if Path(mesh_path).suffix.lower() == ".obj":
            self.object = fcl.MeshLoader().load(str(mesh_path), scale)
        elif Path(mesh_path).suffix.lower() == ".npz":
            with np.load(mesh_path, allow_pickle=False) as archive:
                self.object = _bvh(archive["vertices"] * scale, archive["faces"])
        else:
            raise ValueError("Expected the original triangle OBJ or mesh NPZ")
        count = len(native["qpos"])
        rotations, translations = native["obj_R"], native["obj_t"]
        if (
            rotations.shape != (count, 3, 3)
            or translations.shape != (count, 3)
            or not np.isfinite(translations).all()
            or not np.isfinite(rotations).all()
            or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-5)
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-5)
        ):
            raise ValueError("Expected a matching finite rigid object trajectory")
        self.transforms = [
            fcl.Transform3f(r, t)
            for r, t in zip(Rotation.from_matrix(rotations).as_matrix(), translations)
        ]
        self.legs = []
        for side in ("left", "right"):
            links = [
                model.body(f"{side}_{name}_link").id
                for name in (
                    "hip_pitch",
                    "hip_roll",
                    "hip_yaw",
                    "knee",
                    "ankle_pitch",
                    "ankle_roll",
                )
            ]
            geoms = np.flatnonzero(
                np.isin(model.geom_bodyid, links) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
            )
            if not len(geoms):
                raise ValueError(f"No {side} leg mesh on the native model")
            members = []
            for geom in geoms:
                mesh = model.geom_dataid[geom]
                start = model.mesh_vertadr[mesh]
                face = model.mesh_faceadr[mesh]
                members.append(
                    (
                        geom,
                        _bvh(
                            model.mesh_vert[start : start + model.mesh_vertnum[mesh]],
                            model.mesh_face[face : face + model.mesh_facenum[mesh]],
                        ),
                    )
                )
            self.legs.append(members)

    def distances(self, qpos, side):
        data = mujoco.MjData(self.model)
        members = self.legs[side]
        distances = np.empty((len(qpos), len(members)))
        request = fcl.DistanceRequest()
        for frame, (pose, obj) in enumerate(zip(qpos, self.transforms)):
            data.qpos[:] = pose
            mujoco.mj_kinematics(self.model, data)
            for col, (geom, bvh) in enumerate(members):
                link = fcl.Transform3f(data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom])
                distances[frame, col] = max(
                    0.0, fcl.distance(bvh, link, self.object, obj, request, fcl.DistanceResult())
                )
        return distances


def new_contact_intervals(original, corrected):
    """Keep the baseline leg across a contact episode if correction adds a close pair."""
    new = ((corrected <= 0.0005) & (original > 0.001)).any(axis=1)
    near = (np.minimum(original, corrected) <= 0.002).any(axis=1)
    groups, _ = label(near)
    return near & np.isin(groups, groups[new])
