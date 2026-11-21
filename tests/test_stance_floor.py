"""Regression coverage for the opt-in correction and its registered native frame."""

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("osqp")
pytest.importorskip("hppfcl")
from grail.retargeting.stance_clearance import ObjectClearance  # noqa: E402
from grail.retargeting.stance_floor import Floor, Leg, correct, write_candidate  # noqa: E402


def scene(height=0.0):
    layout = {"name": "scene.floor", "vertices_key": "v", "faces_key": "f"}
    vertices = np.array([[-1, -1, height], [1, -1, height], [1, 1, height], [-1, 1, height]])
    return {
        "static_scene_json": json.dumps({"meshes": [layout]}),
        "v": vertices,
        "f": np.array([[0, 1, 2], [0, 2, 3]]),
    }


def robot():
    limbs = []
    for side, y in (("left", 0.10), ("right", -0.10)):
        chain = []
        hinges = [
            ("hip_pitch", "0 1 0", f"0 {y} 0"),
            ("hip_roll", "1 0 0", "0 0 0"),
            ("hip_yaw", "0 0 1", "0 0 0"),
            ("knee", "0 1 0", "0 0 -0.30"),
            ("ankle_pitch", "0 1 0", "0 0 -0.30"),
            ("ankle_roll", "1 0 0", "0 0 0"),
        ]
        for joint, axis, position in hinges:
            chain.append(
                f'<body name="{side}_{joint}_link" pos="{position}">'
                f'<joint name="{side}_{joint}_joint" axis="{axis}"/>'
                '<inertial pos="0 0 0" mass="0.1" diaginertia=".001 .001 .001"/>'
            )
        limbs.append("".join(chain) + '<geom type="mesh" mesh="foot"/>' + "</body>" * len(hinges))
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><default><joint limited="true" range="-2 2"/></default>'
        '<asset><mesh name="foot" vertex="-.06 -.03 -.03 .10 -.03 -.03 .10 .03 -.03 -.06 .03 -.03 '
        '-.06 -.03 .01 .10 -.03 .01 .10 .03 .01 -.06 .03 .01"/></asset><worldbody>'
        '<body pos="0 0 .612"><freejoint/><geom type="sphere" size=".03"/>'
        + "".join(limbs)
        + '<body><joint name="protected_wrist"/><geom type="sphere" size=".02"/>'
        "</body></body></worldbody></mujoco>"
    )


def test_floor_interpolates_registered_height_and_respects_mesh_edges():
    native = scene(0.45)
    native["v"][:, 2] += native["v"][:, 0] / 10
    floor = Floor(native)
    height, normal = floor.surface(np.array([0.3, 0.1, 1.0]))
    assert height == pytest.approx(0.48)
    np.testing.assert_allclose(normal, [-0.1, 0, 1])
    assert floor.surface(np.array([4, 0, 1])) is None
    # The lowest Z vertex lies outside the mesh and the sloped surface is
    # highest under the third vertex.
    vertices = np.array([[4, 0, -0.2], [-0.6, 0, 0.43], [0.8, 0, 0.45]])
    vertex, support = floor.witness(vertices)
    np.testing.assert_array_equal(vertex, vertices[2])
    assert support[0] == pytest.approx(0.53)
    native["static_scene_json"] = json.dumps({"meshes": []})
    with pytest.raises(ValueError, match="floor"):
        Floor(native)


def object_fixture(tmp_path, source):
    path = tmp_path / "model.npz"
    np.savez(
        path, vertices=np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]), faces=np.array([[0, 1, 2]])
    )
    count = len(source["qpos"])
    source.update(
        object_scale=1.0,
        obj_R=np.tile(np.eye(3), (count, 1, 1)),
        obj_t=np.tile([2, 0, 0], (count, 1)),
    )
    return path


def test_stationary_stance_correction_preserves_wrist_root_and_swing(tmp_path):
    model = robot()
    qpos = np.tile(model.qpos0, (9, 1))
    qpos[:, -1] = np.linspace(0, 1, len(qpos))
    right = Leg(model, "right")
    qpos[:, right.qpos[[0, 3, 4]]] = [-0.5, 1.0, -0.5]
    source = {**scene(), "qpos": qpos.copy(), "fps": 24}
    mesh = object_fixture(tmp_path, source)
    corrected, _ = correct(model, source, ObjectClearance(model, source, mesh))
    left = Leg(model, "left")
    fixed = np.ones(model.nq, dtype=bool)
    fixed[left.qpos] = False
    np.testing.assert_array_equal(corrected[:, fixed], qpos[:, fixed])
    np.testing.assert_array_equal(source["qpos"], qpos)
    mj = mujoco.MjData(model)
    for pose, initial in zip(corrected, qpos):
        mj.qpos[:] = pose
        mujoco.mj_kinematics(model, mj)
        stance_point = Floor(source).witness(left.vertices(mj))[0]
        assert -0.005 < stance_point[2] < 0.001
        mj.qpos[:] = initial
        mujoco.mj_kinematics(model, mj)
        assert Floor(source).witness(left.vertices(mj))[0][2] < -0.015


def test_new_moving_object_contact_locks_only_contiguous_source_contact():
    from grail.retargeting.stance_clearance import new_contact_intervals

    original = np.array(
        [
            [0.03, 0.03],
            [0.03, 0.03],
            [0.012, 0.03],
            [0.001, 0.03],
            [0.03, 0.03],
            [0.03, 0.001],
            [0.03, 0.001],
        ]
    )
    changed = original.copy()
    changed[2, 0] = 0.0002
    np.testing.assert_array_equal(
        new_contact_intervals(original, changed), [False, False, True, True, False, False, False]
    )
    assert not new_contact_intervals(original, original).any()


def test_relocated_originals_and_exclusive_native_output(tmp_path):
    import hashlib
    from grail.core.bindings import sha256

    model = robot()
    xml = tmp_path / "g1.xml"
    mujoco.mj_saveLastXML(str(xml), model)
    model = mujoco.MjModel.from_xml_path(str(xml))
    blob = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, blob)
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    native = dict(
        scene(),
        qpos=np.tile(model.qpos0, (4, 1)),
        fps=24.0,
        timestamps=np.arange(4) / 24.0,
        dof_names=[model.joint(int(i)).name for i in joints],
        qposadr=model.jnt_qposadr[joints],
        native_status="UNREVIEWED",
        selected=False,
        world_frame="soma_retargeter_z_up",
        export_kind="soma_initialization_requires_native_contact_and_visual_review",
        model_xml_sha256=sha256(xml),
        compiled_model_sha256=hashlib.sha256(blob).hexdigest(),
    )
    mesh = object_fixture(tmp_path, native)
    native.update(object_mesh_path="/original/location/model.npz", object_mesh_sha256=sha256(mesh))
    original = tmp_path / "original.npz"
    np.savez(original, **native)
    out = tmp_path / "new/candidate.npz"
    with pytest.raises(ValueError, match="object mesh binding"):
        write_candidate(original, xml, out, xml)
    assert not out.exists()
    receipt = write_candidate(original, xml, out, mesh)
    assert receipt["parent_sha256"] == sha256(original)
    with np.load(out, allow_pickle=False) as result:
        assert not result["selected"] and str(result["native_status"]) == "UNREVIEWED"
        for name in ("obj_t", "obj_R", "timestamps", "object_mesh_sha256", "v", "f"):
            np.testing.assert_array_equal(result[name], native[name])
    with pytest.raises(FileExistsError):
        write_candidate(original, xml, out, mesh)
