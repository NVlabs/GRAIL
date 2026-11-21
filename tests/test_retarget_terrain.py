from pathlib import Path

import numpy as np
import pytest


def test_static_terrain_pack(tmp_path, monkeypatch):
    joblib = pytest.importorskip("joblib")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "imports/SONIC"))
    from grail.core.bindings import sha256
    from grail.retargeting import terrain

    mesh = tmp_path / "step.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4 -3 -2 -1\n")
    qpos = np.zeros((2, 36))
    qpos[:, 4] = 1  # Native root quaternions are wxyz.
    qpos[:, 7:] = np.arange(29)
    data = dict(
        qpos=qpos,
        fps=24,
        timestamps=[0, 1 / 24],
        dof_names=[s.removesuffix("_dof") for s in terrain.BONES_CSV_JOINT_NAMES][::-1],
        qposadr=np.arange(7, 36),
        obj_R=np.tile(np.eye(3), (2, 1, 1)),
        obj_t=np.ones((2, 3)),
        object_scale=[2, 3, 1],
        object_mesh_sha256=sha256(mesh),
        object_mesh_path=str(mesh),
    )
    native, output = tmp_path / "native.npz", tmp_path / "library"
    np.savez(native, **data)
    terrain.pack(native, output, "step")
    robot = joblib.load(output / "robot/step.pkl")["step"]
    obj = joblib.load(output / "objects/step.pkl")["step"]
    np.testing.assert_equal(robot["dof"], qpos[:, 7:][:, ::-1])
    np.testing.assert_equal(robot["root_rot"], qpos[:, [4, 5, 6, 3]])
    np.testing.assert_equal(obj["root_pos"][:, 0], data["obj_t"])
    np.testing.assert_equal(obj["root_quat"][:, 0], [[1, 0, 0, 0]] * 2)
    assert robot["fps"] == obj["fps"] == 24
    stage = Usd.Stage.Open(str(output / "object_usd/step.usd"))
    surface = UsdGeom.Mesh.Get(stage, "/Terrain/Surface")
    assert surface.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    np.testing.assert_equal(surface.GetPointsAttr().Get()[2], [2, 3, 0])
    assert list(surface.GetFaceVertexCountsAttr().Get()) == [4]
    assert Usd.Stage.Open(str(output / "flat_placeholder.usd"))

    data["obj_t"][1, 2] += 0.1
    np.savez(native, **data)
    with pytest.raises(ValueError, match="fixed pose"):
        terrain.pack(native, tmp_path / "moving", "step")
    assert not (tmp_path / "moving").exists()
