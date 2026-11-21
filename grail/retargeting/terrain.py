"""Pack a native G1 motion and its fixed terrain OBJ for SONIC tracking."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (
    BONES_CSV_JOINT_NAMES,
    convert_sequence,
)
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from grail.core.bindings import sha256


def read_obj_surface(path):
    """Keep the vertices and polygon winding of the collision surface."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.partition("#")[0].split()
        if fields[:1] == ["v"]:
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[:1] == ["f"]:
            indices = [int(token.split("/")[0]) for token in fields[1:]]
            if len(indices) < 3 or 0 in indices:
                raise ValueError("Terrain OBJ contains an invalid polygon")
            faces.append([i - 1 if i > 0 else len(vertices) + i for i in indices])
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Terrain OBJ must contain finite 3D vertices")
    if not faces or any(min(face) < 0 or max(face) >= len(points) for face in faces):
        raise ValueError("Terrain OBJ must contain polygons with valid vertex indices")
    return points, faces


def write_collider(path, points, faces):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Terrain").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mesh = UsdGeom.Mesh.Define(stage, "/Terrain/Surface")
    mesh.CreatePointsAttr(points)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    stage.GetRootLayer().Save()


def motion_entries(native):
    qpos = native["qpos"]
    fps = float(native["fps"])
    frames = len(qpos)
    if frames < 2 or qpos.shape != (frames, 36) or not np.isfinite(qpos).all():
        raise ValueError("Expected a native body-only G1 motion with 36 qpos per frame")
    clock = native["timestamps"]
    if (
        not np.isfinite(fps)
        or fps <= 0
        or clock.shape != (frames,)
        or not np.allclose(clock, np.arange(frames) / fps, rtol=0, atol=1e-8)
    ):
        raise ValueError("Expected a uniformly sampled native motion")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("The native root quaternions must have unit length")
    names, addresses = native["dof_names"].tolist(), native["qposadr"].tolist()
    body = [name.removesuffix("_dof") for name in BONES_CSV_JOINT_NAMES]
    if sorted(names) != sorted(body) or sorted(addresses) != list(range(7, 36)):
        raise ValueError("Expected all 29 named G1 body joints and their qpos addresses")
    columns = dict(zip(names, addresses, strict=True))
    rotations, positions = native["obj_R"], native["obj_t"]
    if rotations.shape != (frames, 3, 3) or positions.shape != (frames, 3):
        raise ValueError("Terrain poses must cover the native motion")
    for track in (rotations, positions):
        if not np.isfinite(track).all() or not np.allclose(track, track[0], rtol=0, atol=1e-6):
            raise ValueError("Terrain must have a single fixed pose throughout the motion")
    if not np.allclose(rotations[0].T @ rotations[0], np.eye(3), atol=2e-5) or not np.isclose(
        np.linalg.det(rotations[0]), 1, atol=2e-5
    ):
        raise ValueError("Invalid native terrain rotation")
    robot = convert_sequence(
        dict(
            joint_pos=qpos[:, [columns[name] for name in body]],
            joint_order="mj",
            body_pos_w=qpos[:, None, :3],
            body_quat_w=qpos[:, None, 3:7],
        ),
        fps,
    )
    robot.pop("smpl_joints", None)  # The converter inserts dummy human joints.
    terrain = dict(
        fps=fps,
        root_pos=positions[:, None, :],
        root_quat=Rotation.from_matrix(rotations).as_quat()[:, None, [3, 0, 1, 2]],
    )
    return robot, terrain


def pack(native_path, output, key, object_mesh=None):
    if not key or Path(key).name != key or key in {".", ".."}:
        raise ValueError("Use one filename component as the motion key")
    if output.exists():
        raise FileExistsError(f"Choose a new terrain library directory: {output}")
    with np.load(native_path, allow_pickle=False) as native:
        mesh_path = object_mesh or Path(str(native["object_mesh_path"].item()))
        if sha256(mesh_path) != native["object_mesh_sha256"].item():
            raise ValueError("Terrain OBJ differs from the mesh bound to the native motion")
        robot, terrain = motion_entries(native)
        scale = np.broadcast_to(native["object_scale"].ravel(), (3,))
        if not np.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Terrain mesh scale must be positive and finite")
    vertices, faces = read_obj_surface(mesh_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        library = Path(tmp) / "library"
        for folder in ("robot", "objects", "object_usd"):
            (library / folder).mkdir(parents=True)
        joblib.dump({key: robot}, library / "robot" / f"{key}.pkl")
        joblib.dump({key: terrain}, library / "objects" / f"{key}.pkl")
        write_collider(library / "object_usd" / f"{key}.usd", (vertices * scale).tolist(), faces)
        # SONIC opens a flat placeholder even when all environments use terrain.
        write_collider(
            library / "flat_placeholder.usd",
            [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
            [[0, 1, 2, 3]],
        )
        library.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native", type=Path, required=True, help="Reviewed native body-only G1 NPZ"
    )
    parser.add_argument("--object-mesh", type=Path, help="Relocated copy of the bound terrain OBJ")
    parser.add_argument("--motion-key", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New SONIC terrain library")
    args = parser.parse_args()
    pack(args.native, args.output, args.motion_key, args.object_mesh)


if __name__ == "__main__":
    main()
