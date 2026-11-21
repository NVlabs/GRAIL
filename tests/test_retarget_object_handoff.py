"""Original-object frame and provenance checks across the SOMA retarget boundary."""

import json

import numpy as np
import pytest

from grail.retargeting.handoff import _binding, export_object_sidecar


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "hoi.npz"
    np.savez(
        source,
        obj_R=np.broadcast_to(np.eye(3), (3, 3, 3)),
        obj_t=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]),
        obj_scale=np.array([0.2, 0.3, 0.4]),
        fps=24.0,
    )
    bvh = tmp_path / "source.bvh"
    bvh.write_text("fixture bound by export manifest\n")
    mesh = tmp_path / "original.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "grail.soma03.bvh_export.v1",
                "status": "EXPORTED_UNREVIEWED",
                "source": _binding(source),
                "bvh": _binding(bvh),
                "frame_count": 3,
                "fps": 24.0,
                "source_world_frame": "grail_z_up_m",
                "source_facing_direction": "Mujoco",
                "world_from_grail": np.eye(3).tolist(),
            }
        )
    )
    return dict(
        hoi_path=source,
        output_path=tmp_path / "object.npz",
        bvh_path=bvh,
        export_manifest=manifest,
        object_mesh_path=mesh,
    )


def test_preserves_material_transform_and_exact_timing(source):
    report = export_object_sidecar(**source)
    with np.load(source["output_path"]) as actual, np.load(source["hoi_path"]) as expected:
        np.testing.assert_array_equal(actual["obj_R"], expected["obj_R"])
        np.testing.assert_array_equal(actual["obj_t"], expected["obj_t"])
        np.testing.assert_array_equal(actual["object_scale"], expected["obj_scale"])
        np.testing.assert_array_equal(actual["timestamps"], np.arange(3) / 24)
        assert actual["source_sha256"].item() == _binding(source["bvh_path"])["sha256"]
        assert actual["hoi_source_sha256"].item() == _binding(source["hoi_path"])["sha256"]
    assert report["scale_override"] is False
    assert report["status"] == "UNREVIEWED"
    with pytest.raises(FileExistsError):
        export_object_sidecar(**source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fps", 30),
        ("frame_count", 4),
        ("status", "FAILED_UNREVIEWED"),
        ("world_from_grail", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    ],
)
def test_rejects_timing_status_or_world_mismatch(source, field, value):
    path = source["export_manifest"]
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        export_object_sidecar(**source)
    assert not source["output_path"].exists()


def test_bvh_mutation_invalidates_handoff(source):
    source["bvh_path"].write_text("different exported skeleton")
    with pytest.raises(ValueError, match="bindings"):
        export_object_sidecar(**source)


def test_explicit_scale_override_is_recorded(source):
    report = export_object_sidecar(**source, object_scale=1.0)
    assert report["scale_override"] is True
    assert report["source_object_scale"] == [0.2, 0.3, 0.4]
    with np.load(source["output_path"]) as data:
        assert data["object_scale"].item() == 1.0


def test_legacy_xyz_column_scale_is_not_an_animated_scale(source):
    export_object_sidecar(**source, object_scale=[[0.2], [0.3], [0.4]])
    with np.load(source["output_path"]) as data:
        np.testing.assert_array_equal(data["object_scale"], [0.2, 0.3, 0.4])


def test_world_rotation_applies_to_both_orientation_and_translation(source):
    path = source["export_manifest"]
    manifest = json.loads(path.read_text())
    world = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    manifest["world_from_grail"] = world.tolist()
    path.write_text(json.dumps(manifest))
    export_object_sidecar(**source)
    with np.load(source["output_path"]) as data:
        np.testing.assert_array_equal(data["obj_R"], np.broadcast_to(world, (3, 3, 3)))
        np.testing.assert_array_equal(data["obj_t"][0], [-2.0, 1.0, 3.0])


def test_table_follows_same_world_transform_as_object(source, tmp_path):
    import joblib

    from grail.retargeting.scene import static_scene_from_hoi

    table_source = tmp_path / "table.pkl"
    joblib.dump(dict(scene_data={"table": dict(size=[2, 1, 0.1], pos=[1, 2, 0.9])}), table_source)
    packed = tmp_path / "table.npz"
    np.savez(packed, **static_scene_from_hoi(table_source))
    path = source["export_manifest"]
    manifest = json.loads(path.read_text())
    manifest["world_from_grail"] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    manifest["world_offset"] = [0, 0, -0.3]
    manifest["grounding"] = {"enabled": True, "source_frame": 0}
    path.write_text(json.dumps(manifest))
    report = export_object_sidecar(**source, scene_sidecar=packed)
    assert report["static_scene"]["status"] == "PRESENT"
    with np.load(source["output_path"]) as data:
        vertices = data["static_scene_000_vertices"]
        np.testing.assert_allclose((vertices.min(0) + vertices.max(0)) / 2, [-2, 1, 0.6])
        np.testing.assert_allclose(data["obj_t"][0], [-2, 1, 2.7])
        # Common translation preserves the original object/table displacement.
        np.testing.assert_allclose(data["obj_t"][0] - vertices.mean(0), [0, 0, 2.1])
        assert report["world_offset"] == [0, 0, -0.3]
        assert report["grounding"]["enabled"] is True
        assert (
            json.loads(data["static_scene_json"].item())["world_frame"]
            == data["world_frame"].item()
        )


@pytest.mark.parametrize("offset", [[0, 0], [0, 0, float("nan")], [[0, 0, 0]]])
def test_handoff_rejects_invalid_grounding_offset(source, offset):
    path = source["export_manifest"]
    value = json.loads(path.read_text())
    value["world_offset"] = offset
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="world_offset"):
        export_object_sidecar(**source)
    assert not source["output_path"].exists()
