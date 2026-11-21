"""Fixed source-bound material contacts keep identity, phase, and live gradients."""

import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
import torch

from grail.optimization.fixed_contact import (
    FixedContactLoss,
    build_binding,
    file_sha256,
    topology_sha256,
    vertices_sha256,
)


@pytest.fixture
def case(tmp_path):
    vertices = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
    ], dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    human_faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    arguments = dict(
        sequence="fixture-sequence", source_video_sha256="a" * 64,
        frame_count=4, fps=30.0, human_model_type="soma03",
        human_vertex_count=4, human_faces=human_faces,
        object_vertices=vertices, object_faces=faces,
    )
    binding = build_binding(**arguments)
    artifact = dict(
        schema="grail.fixed_contact.v1", binding=binding,
        prediction=dict(method="fixture", evidence="test only"),
        contacts=[dict(
            id="right", body_part="R_Hand", human_vertex_id=0,
            object_face_id=0, object_barycentric=[1.0, 0.0, 0.0],
            active_frames=[1, 3], confidence=1.0,
        )],
    )
    return NS(path=tmp_path / "contacts.json", vertices=vertices, faces=faces,
              human_faces=human_faces, arguments=arguments, binding=binding,
              artifact=artifact)


def load(case, artifact=None, **overrides):
    case.path.write_text(json.dumps(case.artifact if artifact is None else artifact))
    options = dict(binding=case.binding, object_vertices=case.vertices,
                   object_faces=case.faces, human_faces=case.human_faces,
                   device="cpu", dtype=torch.float64, robust_delta_m=0.01)
    options.update(overrides)
    return FixedContactLoss.from_file(case.path, **options)


def live_inputs():
    return (torch.zeros(4, 4, 3, dtype=torch.float64),
            torch.eye(3, dtype=torch.float64).repeat(4, 1, 1),
            torch.zeros(4, 3, dtype=torch.float64))


def huber(distance, delta=0.01):
    return 0.5 * distance ** 2 if distance <= delta else delta * (distance - 0.5 * delta)


def test_hashes_bind_file_contents_geometry_and_topology_order(case):
    case.path.write_bytes(b"a source artifact\x00with exact bytes\n")
    assert file_sha256(case.path) == hashlib.sha256(case.path.read_bytes()).hexdigest()
    assert topology_sha256(case.faces) == topology_sha256(case.faces.clone())
    assert topology_sha256(case.faces) != topology_sha256(case.faces.flip(0))
    assert topology_sha256(case.faces) != topology_sha256(case.faces.flip(1))
    assert vertices_sha256(case.vertices) == vertices_sha256(case.vertices.clone())
    assert vertices_sha256(case.vertices) != vertices_sha256(case.vertices.flip(0))
    assert vertices_sha256(case.vertices) != vertices_sha256(case.vertices * 2)


def test_material_target_follows_live_rotation_and_translation(case):
    case.artifact["contacts"][0]["object_barycentric"] = [0.2, 0.3, 0.5]
    loss = load(case)
    human, rotation, translation = live_inputs()
    rotation[:] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation[:] = torch.tensor([0.4, -0.7, 1.2])
    local_point = human.new_tensor([0.2, 0.3, 0.5])
    human[:, 0] = torch.einsum("fij,j->fi", rotation, local_point) + translation
    torch.testing.assert_close(loss(human, rotation, translation), human.new_zeros(()))
    assert loss(human, torch.eye(3).to(human).repeat(4, 1, 1), translation) > 0
    assert loss(human, rotation, torch.zeros_like(translation)) > 0


def test_live_human_rotation_and_translation_pass_gradcheck(case):
    case.artifact["contacts"][0]["object_barycentric"] = [0.2, 0.3, 0.5]
    loss = load(case)
    human, rotation, translation = live_inputs()
    human[:, 0] = torch.tensor([0.24, 0.33, 0.52])
    human.requires_grad_()
    rotation.requires_grad_()
    translation.requires_grad_()
    assert torch.autograd.gradcheck(loss, (human, rotation, translation), eps=1e-6,
                                    atol=1e-7, rtol=1e-4)
    gradients = torch.autograd.grad(loss(human, rotation, translation),
                                    (human, rotation, translation))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient[1:3].abs().sum() > 0
        assert torch.count_nonzero(gradient[[0, 3]]) == 0
    assert torch.count_nonzero(gradients[0][:, 1:]) == 0


def test_nearest_distractor_does_not_replace_fixed_material_target(case):
    loss = load(case)
    human, rotation, translation = live_inputs()
    # Exactly on a different object vertex, but bound to object vertex zero.
    human[:, 0] = case.vertices[1]
    human.requires_grad_()
    value = loss(human, rotation, translation)
    torch.testing.assert_close(value, human.new_tensor(huber(2 ** 0.5)))
    value.backward()
    assert (human.grad[1:3, 0, 0] < 0).all()
    assert (human.grad[1:3, 0, 1] > 0).all()


def test_active_interval_is_end_exclusive_and_inactive_geometry_has_no_gradient(case):
    loss = load(case)
    human, rotation, translation = live_inputs()
    human[:, 0] = case.vertices[0] + torch.tensor([0.25, 0., 0.])
    human[[0, 3], 0] = case.vertices[0] + torch.tensor([0.001, 0., 0.])
    human.requires_grad_()
    value = loss(human, rotation, translation)
    torch.testing.assert_close(value, human.new_tensor(huber(0.25)))
    value.backward()
    assert torch.count_nonzero(human.grad[[0, 3]]) == 0
    assert (human.grad[1:3, 0, 0] > 0).all()


@pytest.mark.parametrize("distance", [0.004, 0.5])
def test_isotropic_huber_in_meters_keeps_large_gap_attraction(case, distance):
    loss = load(case)
    human, rotation, translation = live_inputs()
    direction = human.new_tensor([0.6, 0.8, 0.0])
    human[:, 0] = case.vertices[0] + distance * direction
    human.requires_grad_()
    value = loss(human, rotation, translation)
    torch.testing.assert_close(value, human.new_tensor(huber(distance)))
    value.backward()
    expected = min(distance, 0.01) * direction / 2
    torch.testing.assert_close(human.grad[1, 0], expected)
    torch.testing.assert_close(human.grad[2, 0], expected)


def test_confidence_weights_pairs_and_zero_confidence_never_attracts(case):
    second = copy.deepcopy(case.artifact["contacts"][0])
    second.update(id="left", body_part="L_Hand", human_vertex_id=1, confidence=0.25)
    ignored = copy.deepcopy(second)
    ignored.update(id="ignored", human_vertex_id=2, confidence=0.0)
    case.artifact["contacts"].extend([second, ignored])
    loss = load(case)
    human, rotation, translation = live_inputs()
    human[:, 0] = case.vertices[0] + torch.tensor([0.002, 0., 0.])
    human[:, 1] = case.vertices[0] + torch.tensor([0.03, 0., 0.])
    human[:, 2] = case.vertices[0] + torch.tensor([100., 0., 0.])
    human.requires_grad_()
    value = loss(human, rotation, translation)
    expected = (huber(0.002) + 0.25 * huber(0.03)) / 1.25
    torch.testing.assert_close(value, human.new_tensor(expected))
    value.backward()
    torch.testing.assert_close(human.grad[1, 0, 0], human.new_tensor(0.002 / 2.5))
    torch.testing.assert_close(human.grad[1, 1, 0], human.new_tensor(0.25 * 0.01 / 2.5))
    assert torch.count_nonzero(human.grad[:, 2:]) == 0
    assert len(loss.contact_points_by_hand()["left_hand"][1]) == 1


def test_exact_contact_has_finite_zero_gradients(case):
    loss = load(case)
    human, rotation, translation = live_inputs()
    human[:, 0] = case.vertices[0]
    for value in (human, rotation, translation):
        value.requires_grad_()
    value = loss(human, rotation, translation)
    value.backward()
    assert value == 0
    for tensor in (human, rotation, translation):
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0


def test_original_canonical_object_scale_is_applied_exactly_once(case):
    case.vertices = case.vertices * torch.tensor([2.0, 3.0, 4.0])
    case.arguments["object_vertices"] = case.vertices
    case.binding = build_binding(**case.arguments)
    case.artifact["binding"] = case.binding
    case.artifact["contacts"][0]["object_barycentric"] = [0.2, 0.3, 0.5]
    loss = load(case)
    human, rotation, translation = live_inputs()
    human[:, 0] = human.new_tensor([0.4, 0.9, 2.0])
    torch.testing.assert_close(loss(human, rotation, translation), human.new_zeros(()))
    human[:, 0] *= 2
    assert loss(human, rotation, translation) > 0


def test_canonical_material_inputs_are_detached(case):
    original_vertices = case.vertices.clone().requires_grad_()
    loss = load(case, object_vertices=original_vertices)
    human, rotation, translation = live_inputs()
    human.requires_grad_()
    with torch.no_grad():
        original_vertices.add_(5.0)
    torch.testing.assert_close(loss(human, rotation, translation), human.new_tensor(huber(1.0)))
    loss(human, rotation, translation).backward()
    assert torch.isfinite(human.grad).all()
    assert human.grad.abs().sum() > 0
    assert original_vertices.grad is None


def test_provenance_and_export_preserve_material_pairs_and_active_phase(case):
    case.artifact["contacts"][0]["object_barycentric"] = [0.2, 0.3, 0.5]
    loss = load(case)
    provenance = loss.provenance
    assert provenance["sha256"] == file_sha256(case.path)
    assert provenance["binding"] == case.binding
    assert provenance["prediction"] == case.artifact["prediction"]
    assert provenance["schema"] == "grail.fixed_contact.v1"
    records = loss.export_contacts()
    assert len(records) == 1
    for key, value in case.artifact["contacts"][0].items():
        assert records[0][key] == value
    assert records[0]["object_local_point_m"] == pytest.approx([0.2, 0.3, 0.5])
    by_hand = loss.contact_points_by_hand()
    assert by_hand["left_hand"] == {}
    assert set(by_hand["right_hand"]) == {1, 2}
    assert by_hand["right_hand"][1][0] == pytest.approx([0.2, 0.3, 0.5])
    provenance["binding"]["sequence"] = "mutated"
    records[0]["object_barycentric"][0] = 999
    by_hand["right_hand"][1][0][0] = 999
    assert loss.provenance["binding"] == case.binding
    assert loss.export_contacts()[0]["object_barycentric"] == [0.2, 0.3, 0.5]
    assert loss.contact_points_by_hand()["right_hand"][1][0] == pytest.approx([0.2, 0.3, 0.5])


@pytest.mark.parametrize("change", [
    {"sequence": "another-sequence"},
    {"source_video_sha256": "b" * 64},
    {"frame_count": 5}, {"fps": 24.0},
    {"human_model_type": "smplx"}, {"human_vertex_count": 5},
])
def test_rejects_source_timeline_and_human_binding_mismatch(case, change):
    expected = build_binding(**(case.arguments | change))
    with pytest.raises(ValueError):
        load(case, binding=expected)


@pytest.mark.parametrize("field", ["human_faces", "object_faces", "object_vertices"])
def test_rejects_mismatched_geometry_against_valid_binding(case, field):
    value = case.arguments[field].clone()
    value = value.flip(0) if field.endswith("faces") else value * 2
    with pytest.raises(ValueError):
        load(case, **{field: value})


@pytest.mark.parametrize("field,value", [
    ("source_video_sha256", "not-a-hash"),
    ("source_video_sha256", "A" * 64),
    ("human_topology_sha256", "a" * 63),
    ("object_topology_sha256", "x" * 64),
    ("object_vertices_sha256", None),
    ("frame_count", 0), ("frame_count", 4.0), ("frame_count", True),
    ("fps", 0.0), ("fps", -1.0), ("fps", float("nan")),
])
def test_rejects_malformed_artifact_binding(case, field, value):
    artifact = copy.deepcopy(case.artifact)
    artifact["binding"][field] = value
    with pytest.raises(ValueError):
        load(case, artifact)


@pytest.mark.parametrize("field,value", [
    ("human_vertex_id", -1), ("human_vertex_id", 4),
    ("human_vertex_id", 0.5), ("human_vertex_id", True),
    ("object_face_id", -1), ("object_face_id", 2),
    ("object_face_id", 0.5), ("object_face_id", True),
    ("object_barycentric", [0.5, 0.5]),
    ("object_barycentric", [0.5, 0.5, 0.5]),
    ("object_barycentric", [-0.1, 0.6, 0.5]),
    ("object_barycentric", [float("nan"), 0., 1.]),
    ("active_frames", [0, 5]), ("active_frames", [-1, 2]),
    ("active_frames", [2, 2]), ("active_frames", [3, 1]),
    ("active_frames", [1.5, 3]), ("active_frames", [True, 3]),
    ("active_frames", [1, 2, 3]),
    ("confidence", -0.1), ("confidence", 1.1),
    ("confidence", float("nan")), ("confidence", float("inf")),
    ("id", ""), ("body_part", "unknown"),
])
def test_rejects_invalid_contact_indices_barycentric_phase_and_confidence(case, field, value):
    artifact = copy.deepcopy(case.artifact)
    artifact["contacts"][0][field] = value
    with pytest.raises(ValueError):
        load(case, artifact)


@pytest.mark.parametrize("field,value", [
    ("schema", "grail.fixed_contact.v2"),
    ("prediction", {}),
    ("prediction", {"method": "fixture", "evidence": ""}),
    ("contacts", []),
])
def test_rejects_unsupported_or_unproven_prediction(case, field, value):
    artifact = copy.deepcopy(case.artifact)
    artifact[field] = value
    with pytest.raises(ValueError):
        load(case, artifact)


def test_rejects_duplicate_contact_ids(case):
    case.artifact["contacts"] *= 2
    with pytest.raises(ValueError, match="(?i)duplicate"):
        load(case)


def test_rejects_prediction_without_any_confident_contact(case):
    case.artifact["contacts"][0]["confidence"] = 0
    with pytest.raises(ValueError, match="(?i)confidence"):
        load(case)


@pytest.mark.parametrize("value", [0.0, -0.01, float("nan"), float("inf")])
def test_rejects_invalid_robust_distance(case, value):
    with pytest.raises(ValueError):
        load(case, robust_delta_m=value)


def test_missing_prediction_file_is_a_prerequisite_error(case):
    with pytest.raises((FileNotFoundError, ValueError), match="(?i)(fixed|contact|exist|found)"):
        FixedContactLoss.from_file(
            case.path, binding=case.binding, object_vertices=case.vertices,
            object_faces=case.faces, human_faces=case.human_faces, device="cpu",
        )
