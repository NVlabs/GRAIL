import numpy as np
import pytest

from grail.agentic_kinematic.contact import (
    ContactPatch,
    audit_surfaces,
    classify_exposure,
    edge_triangle_crossings,
    mesh_digest,
    nearest_surface,
    surface_samples,
)


def box(offset=(0, 0, 0)):
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return vertices + offset, faces


def test_nearest_original_triangle_face_switch_is_continuous():
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    points = np.array([[0, -0.000001, 0.4], [0, 0, 0.4], [0, 0.000001, 0.4]])
    result = nearest_surface(points, vertices, faces, face_ids=[1, 0], face_chunk=1)
    np.testing.assert_allclose(result.distance, 0.4, atol=1e-15)
    np.testing.assert_allclose(result.points, points * [1, 1, 0], atol=1e-15)
    np.testing.assert_array_equal(result.face_ids, [0, 0, 1])
    np.testing.assert_allclose(
        surface_samples(vertices, faces, result.face_ids, result.barycentric),
        result.points,
        atol=1e-15,
    )


def test_nearest_matches_analytic_box_and_survives_chunk_changes():
    vertices, faces = box()
    points = np.random.default_rng(41).uniform(-2, 3, (75, 3))
    points = points[((points < 0) | (points > 1)).any(axis=1)]
    expected = points.clip(0, 1)
    for point_chunk, face_chunk in ((1, 1), (32, 5), (100, 2000)):
        result = nearest_surface(
            points, vertices, faces, point_chunk=point_chunk, face_chunk=face_chunk
        )
        np.testing.assert_allclose(result.points, expected, atol=1e-14)
        np.testing.assert_allclose(
            result.distance, np.linalg.norm(points - expected, axis=1), atol=1e-14
        )


def test_nearest_degenerate_triangle_and_empty_queries():
    vertices = np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], float)
    faces = np.array([[0, 1, 2]])
    result = nearest_surface([[1, 2, 0], [3, 0, 0]], vertices, faces)
    np.testing.assert_allclose(result.distance, [2, 1])
    np.testing.assert_allclose(result.points, [[1, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(result.normals, 0)
    empty = nearest_surface(np.empty((0, 3)), vertices, faces)
    assert empty.points.shape == (0, 3)


def test_nearest_long_skinny_triangle_is_not_nearest_centroid():
    vertices = np.array(
        [[0, 0, 0], [1e4, 0, 0], [0, 1e-4, 0], [0, 0, 0.1], [1, 0, 0.1], [0, 1, 0.1]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    result = nearest_surface([[0.01, 1e-5, 0.001]], vertices, faces)
    assert result.face_ids.tolist() == [0]
    np.testing.assert_allclose(result.points, [[0.01, 1e-5, 0]], atol=1e-15)


def test_contact_patch_binds_mesh_source_and_inclusive_phase():
    vertices, faces = box()
    data = dict(
        name="left_handle",
        side="left",
        start_frame=3,
        end_frame=10,
        face_ids=[1, 2],
        evidence_frames=[3, 8],
        evidence="Visible support under handle",
        source_sha256="a" * 64,
        mesh_sha256=mesh_digest(vertices, faces),
    )
    patch = ContactPatch.from_dict(data)
    patch.validate(vertices, faces, 11, "a" * 64)
    with pytest.raises(ValueError, match="source video"):
        patch.validate(vertices, faces, 11, "b" * 64)
    with pytest.raises(ValueError, match="mesh changed"):
        patch.validate(vertices, faces[::-1], 11, "a" * 64)
    with pytest.raises(ValueError, match="timeline"):
        patch.validate(vertices, faces, 10, "a" * 64)
    with pytest.raises(ValueError, match="integers"):
        ContactPatch.from_dict({**data, "face_ids": [1.5]}).validate(vertices, faces, 11, "a" * 64)


def test_buried_targets_are_removed_without_repairing_original_faces():
    a, af = box()
    b, bf = box((0.5, 0, 0))
    vertices, faces = np.vstack((a, b)), np.vstack((af, bf + len(a)))
    before = mesh_digest(vertices, faces)
    result = classify_exposure(vertices, faces, [6, 10], np.ones((2, 3)) / 3)
    assert result["buried"].tolist() == [True, False]
    assert result["closed_components"] == 2
    assert result["status"].tolist() == [
        "buried_in_other_closed_component",
        "outside_other_closed_components",
    ]
    assert mesh_digest(vertices, faces) == before


def test_exposure_welds_only_topology_and_leaves_open_shells_unknown():
    vertices, faces = box()
    # A renderer-style mesh with three independent indices per original triangle.
    duplicated = vertices[faces].reshape(-1, 3)
    duplicate_faces = np.arange(len(duplicated)).reshape(-1, 3)
    result = classify_exposure(duplicated, duplicate_faces, [0], [[1 / 3] * 3])
    assert result["closed_components"] == 1
    open_result = classify_exposure(vertices, faces[:-1], [0], [[1 / 3] * 3])
    assert open_result["status"].tolist() == ["unknown_open_geometry"]
    assert open_result["open_or_inconsistent_components"] == 1


def test_crossing_catches_triangle_interior_and_shared_seams():
    target = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    target_faces = np.array([[0, 1, 2], [0, 2, 3]])
    source = np.array([[0, 0, -1], [0, 0, 1], [4, 0, 1]], float)
    source_faces = np.array([[0, 1, 2]])
    report = edge_triangle_crossings(source, source_faces, target, target_faces)
    assert report["intersection_locations"] == 1  # Seam hit counted once.
    assert report["intersecting_edges"] == 1
    witness = report["witnesses"][0]
    np.testing.assert_allclose(witness["point"], [0, 0, 0])
    audit = audit_surfaces(source, source_faces, target, target_faces)
    assert audit["a_to_b_unsigned"]["min"] >= 1
    assert audit["a_edges_to_b"]["intersection_locations"] == 1
    assert audit["signed_penetration"] is None


def test_containment_and_coplanar_overlap_are_explicitly_not_certified():
    outer, faces = box()
    inner = 0.2 + 0.2 * outer
    report = audit_surfaces(outer, faces, inner, faces)
    assert report["a_edges_to_b"]["intersection_locations"] == 0
    assert report["b_edges_to_a"]["intersection_locations"] == 0
    assert report["signed_penetration"] is None
    assert "containment" in report["limitations"]
    coplanar = edge_triangle_crossings(outer, faces[:2], outer, faces[:2])
    assert coplanar["intersection_locations"] == 0
    assert "coplanar" in coplanar["limitations"]


def test_surface_samples_reject_bad_barycentrics_and_float_face_ids():
    vertices, faces = box()
    with pytest.raises(ValueError, match="sum to one"):
        surface_samples(vertices, faces, [0], [[0.2, 0.2, 0.2]])
    with pytest.raises(ValueError, match="integer"):
        surface_samples(vertices, faces, [0.5], [[1, 0, 0]])
    with pytest.raises(ValueError, match="finite"):
        nearest_surface([[np.nan, 0, 0]], vertices, faces)
