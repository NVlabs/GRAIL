import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from grail.agentic_kinematic.contact import mesh_digest
from grail.agentic_kinematic.hoi_contact import ReviewedContact, topology_digest


@pytest.fixture
def review_fixture(tmp_path):
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    bindings = dict(
        case_id="synthetic",
        source_sha256="a" * 64,
        object_mesh_sha256=mesh_digest(vertices, faces),
        human_faces=faces,
        human_vertex_count=4,
        frame_count=3,
        fps=30.0,
        object_vertices=vertices,
        object_faces=faces,
    )
    data = dict(
        schema_version=1,
        case_id="synthetic",
        source_sha256="a" * 64,
        object_mesh_sha256=bindings["object_mesh_sha256"],
        human_topology_sha256=topology_digest(faces, 4),
        frame_count=3,
        fps=30.0,
        reviewer="synthetic-test-author",
        phases=[
            dict(
                side="left",
                start_frame=0,
                end_frame=2,
                state="contact",
                evidence_frames=[0, 2],
                evidence="Synthetic material point for numerical testing.",
                pairs=[
                    dict(
                        human_vertex_id=1,
                        object_face_id=0,
                        object_barycentric=[0.5, 0.25, 0.25],
                        weight=2.0,
                    )
                ],
            ),
            dict(
                side="right",
                start_frame=0,
                end_frame=2,
                state="unknown",
                evidence_frames=[1],
                evidence="Synthetic unobserved hand.",
                pairs=[],
            ),
        ],
    )
    path = tmp_path / "contact.json"

    def load(payload=None, **overrides):
        path.write_text(json.dumps(data if payload is None else payload))
        return ReviewedContact.load(path, **{**bindings, **overrides})

    return data, bindings, load, path


def test_review_binding_report_and_topology(review_fixture):
    data, bindings, load, path = review_fixture
    review = load()
    report = review.report()
    assert report["contact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["active_pair_frames"] == 3
    assert report["state_frame_counts"]["right"]["unknown"] == 3
    assert json.loads(json.dumps(report))["phases"][0]["pairs"][0]["object_face_id"] == 0
    assert review.object_faces == ((0, 1, 2), (0, 2, 3))
    assert review.labels_for_frame(2) == {"left": "contact", "right": "unknown"}
    faces = bindings["human_faces"]
    assert topology_digest(faces.astype(">i4"), 4) == topology_digest(faces, 4)
    assert topology_digest(faces[::-1], 4) != topology_digest(faces, 4)
    assert topology_digest(faces[:, ::-1], 4) != topology_digest(faces, 4)
    assert topology_digest(faces, 5) != topology_digest(faces, 4)
    for key, replacement in (
        ("case_id", "other"),
        ("source_sha256", "b" * 64),
        ("object_mesh_sha256", "b" * 64),
        ("human_topology_sha256", "b" * 64),
        ("frame_count", 4),
        ("fps", 29.97),
    ):
        with pytest.raises(ValueError, match=key):
            load({**data, key: replacement})


def test_actual_changed_or_reordered_mesh_rejected(review_fixture):
    _, bindings, load, _ = review_fixture
    vertices, faces = bindings["object_vertices"], bindings["object_faces"]
    for changed in (vertices * 2, vertices[::-1], vertices + 0.1):
        with pytest.raises(ValueError, match="mesh changed"):
            load(object_vertices=changed)
    with pytest.raises(ValueError, match="mesh changed"):
        load(object_faces=faces[::-1])
    with pytest.raises(ValueError, match="human_topology_sha256"):
        load(human_faces=faces[::-1])
    with pytest.raises(ValueError, match="human_topology_sha256"):
        load(human_vertex_count=5)
    # Updating the external digest cannot authorize an old authored review.
    with pytest.raises(ValueError, match="object_mesh_sha256"):
        load(object_vertices=vertices * 2, object_mesh_sha256=mesh_digest(vertices * 2, faces))


def test_strict_json_and_numeric_validation(review_fixture):
    data, bindings, load, path = review_fixture
    for key, bad in (
        ("schema_version", True),
        ("schema_version", 2),
        ("frame_count", 3.0),
        ("fps", True),
        ("fps", 0),
        ("fps", float("nan")),
        ("reviewer", " "),
        ("source_sha256", "A" * 64),
        ("source_sha256", 5),
        ("typo", 1),
    ):
        with pytest.raises(ValueError):
            load({**data, key: bad})
    for key, bad in (
        ("human_vertex_id", True),
        ("human_vertex_id", 1.0),
        ("human_vertex_id", -1),
        ("human_vertex_id", 4),
        ("object_face_id", 2),
        ("object_face_id", 0.0),
        ("object_barycentric", [0.5, 0.5]),
        ("object_barycentric", [0, 0, 0]),
        ("object_barycentric", [-1, 1, 1]),
        ("object_barycentric", [True, 0, 0]),
        ("object_barycentric", [float("inf"), 0, 0]),
        ("weight", 0),
        ("weight", -1),
        ("weight", True),
        ("weight", float("inf")),
        ("tolerance_m", -0.01),
        ("tolerance_m", "0"),
    ):
        bad_data = copy.deepcopy(data)
        bad_data["phases"][0]["pairs"][0][key] = bad
        with pytest.raises(ValueError):
            load(bad_data)
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        ReviewedContact.load(path, **bindings)
    for kwargs in ({"fps": True}, {"frame_count": 3.0}, {"human_vertex_count": False}):
        with pytest.raises(ValueError):
            load(**kwargs)


def test_inclusive_phases_coverage_and_evidence(review_fixture):
    data, _, load, _ = review_fixture
    authored = copy.deepcopy(data)
    contact = authored["phases"][0]
    contact.update(start_frame=1, end_frame=1, evidence_frames=[1])
    left_free = dict(
        side="left",
        start_frame=0,
        end_frame=0,
        state="no_contact",
        evidence_frames=[0],
        evidence="Before synthetic contact.",
    )
    left_after = {**left_free, "start_frame": 2, "end_frame": 2, "evidence_frames": [2]}
    authored["phases"].extend([left_after, left_free])  # JSON ordering is immaterial.
    review = load(authored)
    assert review.interaction_start_frame == review.interaction_end_frame == 1
    assert [review.labels_for_frame(i)["left"] for i in range(3)] == [
        "no_contact",
        "contact",
        "no_contact",
    ]
    for mutate in (
        lambda d: d["phases"].pop(),  # Missing frame zero.
        lambda d: d["phases"].append(copy.deepcopy(d["phases"][-1])),  # Overlap.
        lambda d: d["phases"][0].update(evidence_frames=[0]),
        lambda d: d["phases"][0].update(evidence_frames=[1, 1]),
        lambda d: d["phases"][0].update(start_frame=True),
        lambda d: d["phases"][0].update(end_frame=3),
        lambda d: d["phases"][0].update(pairs=[]),
        lambda d: d["phases"][0].update(state="unknown"),
    ):
        bad = copy.deepcopy(authored)
        mutate(bad)
        with pytest.raises(ValueError):
            load(bad)
    with pytest.raises(ValueError):
        review.labels_for_frame(3)


def test_far_correspondence_uses_authored_triangle_and_has_gradients(review_fixture):
    _, bindings, load, _ = review_fixture
    review = load()
    human = torch.full((3, 4, 3), 10.0, dtype=torch.float64, requires_grad=True)
    objects = torch.tensor(bindings["object_vertices"], dtype=torch.float64).repeat(3, 1, 1)
    objects.requires_grad_()
    loss = review.loss(human, objects)
    distance = np.linalg.norm(np.array([10, 10, 10]) - [0.5, 0.5, 0])
    assert loss.item() == pytest.approx(0.02 * (distance - 0.01))
    loss.backward()
    assert (human.grad[:, 1].norm(dim=1) > 0).all()
    assert torch.count_nonzero(human.grad[:, [0, 2, 3]]) == 0
    assert torch.count_nonzero(objects.grad[:, 3]) == 0  # Unauthored triangle vertex.
    torch.testing.assert_close(objects.grad[:, 0], -human.grad[:, 1] * 0.5)
    torch.testing.assert_close(objects.grad[:, 1], -human.grad[:, 1] * 0.25)
    torch.testing.assert_close(objects.grad[:, 2], -human.grad[:, 1] * 0.25)


def test_unknown_body_and_no_contact_have_graph_connected_zero(review_fixture):
    data, bindings, load, _ = review_fixture
    for state in ("unknown", "body_contact", "no_contact"):
        authored = copy.deepcopy(data)
        authored["phases"][0].update(state=state, pairs=[])
        review = load(authored)
        human = torch.ones((3, 4, 3), dtype=torch.float64, requires_grad=True)
        objects = torch.tensor(bindings["object_vertices"], dtype=torch.float64).repeat(3, 1, 1)
        objects.requires_grad_()
        loss = review.loss(human, objects)
        assert loss.item() == 0
        loss.backward()
        assert torch.count_nonzero(human.grad) == torch.count_nonzero(objects.grad) == 0
        assert (review.interaction_start_frame, review.interaction_end_frame) == (0, 2)


def test_whole_sequence_weighting_tolerance_and_phase_gradients(review_fixture):
    data, bindings, load, _ = review_fixture
    authored = copy.deepcopy(data)
    first = authored["phases"][0]
    first.update(end_frame=0, evidence_frames=[0])
    first["pairs"][0].update(object_barycentric=[1, 0, 0], tolerance_m=0.25, weight=1)
    second = copy.deepcopy(first)
    second.update(start_frame=1, end_frame=2, evidence_frames=[1])
    second["pairs"][0]["weight"] = 3
    authored["phases"].append(second)
    review = load(authored)
    human = torch.zeros((3, 4, 3), dtype=torch.float64)
    human[:, 1, 0] = torch.tensor([0.1, 0.35, 2.25])
    human.requires_grad_()
    objects = torch.tensor(bindings["object_vertices"], dtype=torch.float64).repeat(3, 1, 1)
    loss = review.loss(human, objects, huber_delta_m=0.5)
    assert loss.item() == pytest.approx((3 * 0.005 + 3 * 0.875) / 7)
    loss.backward()
    assert torch.count_nonzero(human.grad[0]) == 0
    assert human.grad[1, 1, 0].item() == pytest.approx(3 * 0.1 / 7)
    assert human.grad[2, 1, 0].item() == pytest.approx(3 * 0.5 / 7)
    with pytest.raises(ValueError, match="floating"):
        review.loss(human[:2], objects)
    with pytest.raises(ValueError, match="dtype"):
        review.loss(human, objects.float())


def test_correspondence_gradcheck_for_human_and_object(review_fixture):
    _, bindings, load, _ = review_fixture
    review = load()
    human = torch.full((3, 4, 3), 0.7, dtype=torch.float64, requires_grad=True)
    objects = torch.tensor(bindings["object_vertices"], dtype=torch.float64).repeat(3, 1, 1)
    objects.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda h, o: review.loss(h, o, huber_delta_m=2),
        (human, objects),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_different_reviewed_material_points_drive_different_synthetic_solutions(review_fixture):
    data, bindings, load, _ = review_fixture
    objects = torch.tensor(bindings["object_vertices"], dtype=torch.float64).repeat(3, 1, 1)
    solutions = []
    for barycentric in ([1, 0, 0], [0, 1, 0]):
        authored = copy.deepcopy(data)
        authored["phases"][0]["pairs"][0]["object_barycentric"] = barycentric
        review = load(authored)
        position = torch.tensor([1.0, 0.4, 0.2], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([position], lr=0.25)
        for _ in range(70):
            optimizer.zero_grad()
            loss = review.loss(position.expand(3, 4, 3), objects, huber_delta_m=4)
            loss.backward()
            optimizer.step()
        solutions.append(position.detach())
    torch.testing.assert_close(
        solutions[0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-8, rtol=0
    )
    torch.testing.assert_close(
        solutions[1], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64), atol=1e-8, rtol=0
    )


def test_torch_is_imported_lazily():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import grail.agentic_kinematic.hoi_contact; assert 'torch' not in sys.modules",
        ],
        check=True,
    )
