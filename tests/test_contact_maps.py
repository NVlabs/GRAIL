"""Independent native human/object region maps retain source timing and identity."""

import copy
import json

import numpy as np
import pytest

from grail.optimization.contact_maps import PhaseContactMap
from grail.optimization.fixed_contact import build_binding, file_sha256


def region(name, human, obj, human_strength=None, object_strength=None):
    return dict(id=name, description="Synthetic regional evidence, no point correspondences",
                human_vertex_ids=human, human_strength=human_strength or [.7] * len(human),
                object_vertex_ids=obj, object_strength=object_strength or [.6] * len(obj))


def interval(name, start, end, state, regions=None):
    return dict(id=name, frames=[start, end], state=state, description="Synthetic source review",
                evidence_frames=[start], regions=regions or [])


@pytest.fixture(params=["soma03", "smplx"])
def case(tmp_path, request):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic source identity, not a playable video")
    human_faces = np.array([[0, 1, 2], [3, 4, 5]])
    obj = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
    object_faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]])
    binding = build_binding(sequence="synthetic", source_video_sha256=file_sha256(video),
                            frame_count=6, fps=24., human_model_type=request.param,
                            human_vertex_count=6, human_faces=human_faces,
                            object_vertices=obj, object_faces=object_faces)
    spec = dict(schema="grail.contact_map.v1", binding=copy.deepcopy(binding),
                reviewed_by="Synthetic fixture", hands={
                    "left_hand": [interval("hidden", 0, 1, "unknown"),
                                  interval("touch", 1, 3, "contact", [region("fingers", [0, 1, 2], [0])]),
                                  interval("shift", 3, 5, "contact", [region("thumb", [1], [1, 2])]),
                                  interval("release", 5, 6, "no_contact")],
                    "right_hand": [interval("approach", 0, 6, "no_contact")]})
    options = dict(binding=binding, human_faces=human_faces, object_vertices=obj,
                   object_faces=object_faces, source_video_path=video,
                   hand_vertex_ids={"left_hand": [0, 1, 2], "right_hand": [3, 4, 5]})
    return spec, options, tmp_path


def test_unequal_cardinality_maps_switch_sets_without_inventing_pairs(case):
    spec, options, _ = case
    maps = PhaseContactMap.from_spec(spec, **options)
    data = maps.dense_fields()
    assert np.flatnonzero(data["left_hand_strength"][1]).tolist() == [0, 1, 2]
    assert np.flatnonzero(data["left_object_strength"][1]).tolist() == [0]
    assert np.flatnonzero(data["left_hand_strength"][3]).tolist() == [1]
    assert np.flatnonzero(data["left_object_strength"][3]).tolist() == [1, 2]
    assert data["left_hand_strength"].shape == (6, 6)
    assert data["left_object_strength"].shape == (6, 5)
    for field in data.values():
        assert not field[[0, 5]].any()
    assert not data["right_hand_strength"].any()
    assert not data["right_object_strength"].any()
    assert maps.provenance["optimizer_ready"] is False
    assert maps.provenance["contains_point_correspondences"] is False
    assert maps.provenance["interval_counts"]["left_hand"][1]["positive_human_vertex_count"] == 3
    assert maps.provenance["interval_counts"]["left_hand"][1]["positive_object_vertex_count"] == 1


def test_overlapping_regions_use_max_evidence_instead_of_duplication(case):
    spec, options, _ = case
    spec["hands"]["left_hand"][1]["regions"].append(region("overlap", [0, 1], [0, 3], [.9, .2], [.4, .8]))
    maps = PhaseContactMap.from_spec(spec, **options)
    data = maps.dense_fields()
    np.testing.assert_allclose(data["left_hand_strength"][1, :3], [.9, .7, .7])
    np.testing.assert_allclose(data["left_object_strength"][1, [0, 3]], [.6, .8])
    assert not data["left_object_strength"][3, 3]


def test_snapshot_export_and_original_geometry_are_isolated(case):
    spec, options, root = case
    maps = PhaseContactMap.from_spec(spec, **options)
    original = maps.compact_fields()
    spec["hands"]["left_hand"][1]["regions"][0]["human_strength"][0] = .1
    options["binding"]["sequence"] = "mutated"
    maps.export_spec()["hands"].clear()
    maps.provenance["binding"]["sequence"] = "modified"
    maps.compact_fields()["left_hand_strength"][:] = 0
    for key, value in original.items():
        np.testing.assert_array_equal(maps.compact_fields()[key], value)
    out = maps.export_npz(root / "maps.npz")
    with np.load(out, allow_pickle=False) as saved:
        for key, value in original.items():
            np.testing.assert_array_equal(saved[key], value)
        np.testing.assert_array_equal(saved["frame_indices"], np.arange(6))
    with pytest.raises(FileExistsError):
        maps.export_npz(out)


def test_file_source_and_manifest_hashes_are_verified(case):
    spec, options, root = case
    path = root / "map.json"
    path.write_text(json.dumps(spec))
    maps = PhaseContactMap.from_file(path, **options)
    assert maps.provenance["manifest"]["sha256"] == file_sha256(path)
    assert maps.provenance["source_video"]["sha256"] == options["binding"]["source_video_sha256"]
    options["source_video_path"].write_bytes(b"different source")
    with pytest.raises(ValueError, match="source video hash"):
        PhaseContactMap.from_file(path, **options)


@pytest.mark.parametrize("field", ["human_faces", "object_faces", "object_vertices"])
def test_current_topology_and_original_geometry_binding_is_enforced(case, field):
    spec, options, _ = case
    options[field] = options[field][::-1].copy()
    with pytest.raises(ValueError, match="geometry/topology"):
        PhaseContactMap.from_spec(spec, **options)


@pytest.mark.parametrize("key,value", [("human_model_type", "different"), ("source_video_sha256", "f" * 64),
                                      ("frame_count", 7), ("fps", 30.)])
def test_source_model_timeline_binding_mismatch_rejected(case, key, value):
    spec, options, _ = case
    spec["binding"][key] = value
    with pytest.raises(ValueError, match="binding differs"):
        PhaseContactMap.from_spec(spec, **options)


@pytest.mark.parametrize("key,value", [("human_vertex_ids", [0, 0, 1]),
                                      ("human_vertex_ids", [0, 1, 6]),
                                      ("human_vertex_ids", [True, 1, 2]),
                                      ("object_vertex_ids", [5]),
                                      ("object_vertex_ids", []),
                                      ("human_strength", [1., 1.]),
                                      ("human_strength", [0., 0., 0.]),
                                      ("human_strength", [1., float("nan"), 1.]),
                                      ("human_strength", [1., 1.1, 1.]),
                                      ("object_strength", [True]),
                                      ("pairs", [[0, 0]])])
def test_invalid_sets_strengths_or_forced_correspondences_rejected(case, key, value):
    spec, options, _ = case
    spec["hands"]["left_hand"][1]["regions"][0][key] = value
    with pytest.raises(ValueError):
        PhaseContactMap.from_spec(spec, **options)


def test_native_hand_membership_prevents_wrong_hand_or_body_regions(case):
    spec, options, _ = case
    spec["hands"]["left_hand"][1]["regions"][0]["human_vertex_ids"] = [0, 1, 3]
    with pytest.raises(ValueError, match="outside native left_hand"):
        PhaseContactMap.from_spec(spec, **options)


@pytest.mark.parametrize("change", ["gap", "overlap", "missing_tail", "unknown_active", "empty_contact", "bad_evidence"])
def test_bad_or_inactive_timeline_cannot_activate_map_fields(case, change):
    spec, options, _ = case
    intervals = spec["hands"]["left_hand"]
    if change == "gap":
        intervals[1]["frames"][0] = 2
    elif change == "overlap":
        intervals[1]["frames"][0] = 0
    elif change == "missing_tail":
        intervals.pop()
    elif change == "unknown_active":
        intervals[1]["state"] = "unknown"
    elif change == "empty_contact":
        intervals[1]["regions"] = []
    else:
        intervals[1]["evidence_frames"] = [0]
    with pytest.raises(ValueError):
        PhaseContactMap.from_spec(spec, **options)


def test_fully_reviewed_inactive_timeline_exports_empty_sets(case):
    spec, options, _ = case
    spec["hands"] = {"left_hand": [interval("clear", 0, 6, "no_contact")],
                     "right_hand": [interval("hidden", 0, 6, "unknown")]}
    maps = PhaseContactMap.from_spec(spec, **options)
    for key, value in maps.compact_fields().items():
        assert value.shape == ((6, 0) if key.endswith("strength") else (0,))
    assert all(not value.any() for value in maps.dense_fields().values())
