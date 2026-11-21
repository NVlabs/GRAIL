"""Whole-target bounds retain exact original-edge crossing semantics."""

import numpy as np
import pytest

from grail.agentic_kinematic.contact import edge_triangle_crossings


def target_plane():
    return (
        np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )


def test_crossing_edge_with_both_endpoints_outside_global_bounds_is_retained():
    vertices, faces = target_plane()
    source = np.array([[0, 0, -100], [0, 0, 100], [400, 0, 100]], float)
    report = edge_triangle_crossings(source, np.array([[0, 1, 2]]), vertices, faces)
    assert report["intersection_locations"] == 1
    assert report["source_edges_tested"] == 3
    assert report["witnesses"][0]["source_edge_vertex_ids"] == [0, 1]
    assert report["witnesses"][0]["target_original_face_id"] == 0
    assert report["witnesses"][0]["point"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("witness_limit", [0, 1, 20])
def test_remote_source_faces_keep_original_witness_ids_and_order(witness_limit):
    vertices, faces = target_plane()
    crossing = np.array([[0, 0, -1], [0, 0, 1], [4, 0, 1]], float)
    local_faces = np.array([[0, 1, 2]])
    local = edge_triangle_crossings(
        crossing, local_faces, vertices, faces, witness_limit=witness_limit
    )
    source = np.vstack((crossing + [0, 100, 0], crossing, crossing + [0, -100, 0]))
    full = edge_triangle_crossings(
        source, np.arange(9).reshape(-1, 3), vertices, faces, witness_limit=witness_limit
    )
    for witness in local["witnesses"]:
        witness["source_edge_vertex_ids"] = [v + 3 for v in witness["source_edge_vertex_ids"]]
    local["source_edges_tested"] = 9
    assert full == local


@pytest.mark.parametrize(
    "source",
    [
        [[0, 0, 0], [0, 0, 1], [2, 0, 1]],  # Endpoint contact, not crossing.
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],  # Coplanar overlap.
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Degenerate source edges.
        [[100, 100, 100], [101, 100, 100], [100, 101, 100]],  # Disjoint bounds.
        [[0, 0, -1e-10], [0, 0, 1e-10], [0, 0, 0]],  # Below endpoint tolerance.
    ],
)
def test_endpoint_coplanar_degenerate_and_disjoint_cases_remain_excluded(source):
    vertices, faces = target_plane()
    report = edge_triangle_crossings(np.asarray(source), np.array([[0, 1, 2]]), vertices, faces)
    assert report["intersection_locations"] == 0
    assert report["witnesses"] == []


def test_degenerate_target_has_no_crossings():
    target = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], float)
    source = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 1]], float)
    report = edge_triangle_crossings(source, np.array([[0, 1, 2]]), target, np.array([[0, 1, 2]]))
    assert report["intersection_locations"] == 0


def test_tolerance_padding_retains_roundoff_level_triangle_vertex_crossing():
    target = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    # The narrow-phase barycentric tolerance allows this tiny negative x.
    source = np.array([[-1e-15, 0.2, -1], [-1e-15, 0.2, 1], [-1, 0.2, 1]])
    report = edge_triangle_crossings(source, np.array([[0, 1, 2]]), target, np.array([[0, 1, 2]]))
    assert report["intersection_locations"] == 1
    assert report["witnesses"][0]["source_edge_vertex_ids"] == [0, 1]


def test_indexed_queries_preserve_reference_witnesses(monkeypatch):
    from grail.core import contact

    rng = np.random.default_rng(41)
    # Coincident triangles test that stable original face IDs survive R-tree traversal.
    vertices = rng.uniform(-1, 1, (100, 3))
    faces = rng.integers(0, len(vertices), (83, 3))
    faces = np.concatenate((faces, faces, faces[:10, ::-1]))
    source, edges = rng.uniform(-2, 2, (70, 3)), rng.integers(0, 70, (60, 3))
    simple = contact.edge_triangle_crossings(source, edges, vertices, faces, witness_limit=100)
    monkeypatch.setattr(contact, "_INDEX_PAIR_LIMIT", 0)
    assert simple == contact.edge_triangle_crossings(
        source, edges, vertices, faces, witness_limit=100
    )
