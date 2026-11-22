"""Replay a body-only G1 terrain pack with MuJoCo, using its fixed mesh and reference poses."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import joblib
import mujoco
import numpy as np
from pxr import Usd, UsdGeom

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import BONES_CSV_JOINT_NAMES


def render(library, key, model_xml, output):
    robot = joblib.load(library / "robot" / f"{key}.pkl")[key]
    terrain = joblib.load(library / "objects" / f"{key}.pkl")[key]
    rotations = np.asarray(terrain["root_quat"])[:, 0]
    translations = np.asarray(terrain["root_pos"])[:, 0]
    if not (np.allclose(rotations, rotations[0]) and np.allclose(translations, translations[0])):
        raise ValueError("This replay requires stationary terrain")
    root = np.asarray(robot["root_trans_offset"])
    body = np.asarray(robot["dof"])
    if body.shape != (len(root), len(BONES_CSV_JOINT_NAMES)):
        raise ValueError("Expected a packed 29-joint body-only G1 trajectory")

    stage = Usd.Stage.Open(str(library / "object_usd" / f"{key}.usd"))
    mesh = UsdGeom.Mesh.Get(stage, "/Terrain/Surface")
    vertices = np.asarray(mesh.GetPointsAttr().Get()).ravel()
    indices = iter(mesh.GetFaceVertexIndicesAttr().Get())
    triangles = []
    for count in mesh.GetFaceVertexCountsAttr().Get():
        face = [next(indices) for _ in range(count)]
        triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))

    spec = mujoco.MjSpec.from_file(str(model_xml))
    spec.add_mesh(name="terrain", uservert=vertices, userface=np.asarray(triangles).ravel())
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="terrain",
                            pos=translations[0], quat=rotations[0], rgba=[0.55, 0.65, 0.74, 1])
    model = spec.compile()
    data = mujoco.MjData(model)
    columns = [int(model.joint(name.removesuffix("_dof")).qposadr[0]) for name in BONES_CSV_JOINT_NAMES]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (root.min(axis=0) + root.max(axis=0)) / 2
    camera.distance = max(3.0, np.linalg.norm(np.ptp(root, axis=0)) + 2.5)
    camera.azimuth, camera.elevation = 22, -12
    quat_wxyz = np.asarray(robot["root_rot"])[:, [3, 0, 1, 2]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=360, width=640) as renderer, imageio.get_writer(
        output, fps=float(robot["fps"]), codec="libx264", macro_block_size=1
    ) as video:
        for pos, quat, angles in zip(root, quat_wxyz, body, strict=True):
            data.qpos[:3], data.qpos[3:7], data.qpos[columns] = pos, quat, angles
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            video.append_data(renderer.render())
    print(f"{len(root)} frames at {robot['fps']} fps: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("key")
    parser.add_argument("model_xml", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.library, args.key, args.model_xml, args.output)
