"""CPU tests of the isolated draft; no fitting pipeline or GPU is imported."""
from types import SimpleNamespace

import pytest
import torch

from grail.agentic_kinematic.hoi_contact_normals import (
    local_vertex_normals,
    material_normal_dots,
    material_normal_loss,
    normal_opposition_penalty,
)


def wedge(dtype=torch.float64):
    human = torch.tensor([[[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]]], dtype=dtype)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    objects = torch.tensor([[[0, 0, 0], [1, 0, 0.3], [0, 1, 0], [0, 0, 1]]], dtype=dtype)
    object_faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    pairs = [dict(human_vertex_id=0, object_face_id=0, weight=1.0)]
    return human, objects, faces, object_faces, pairs


def test_vertex_normal_is_area_weighted_and_respects_original_winding():
    human, _, faces, _, _ = wedge()
    normals = local_vertex_normals(human, faces, [0, 1, 0])
    expected = torch.tensor([[[1 / 5**0.5, 0, 2 / 5**0.5], [0, 0, 1], [1 / 5**0.5, 0, 2 / 5**0.5]]], dtype=human.dtype)
    torch.testing.assert_close(normals, expected)
    torch.testing.assert_close(local_vertex_normals(human, faces.flip(-1), [0]), -expected[:, :1])


def test_incident_zero_area_triangle_is_ignored_when_valid_area_remains():
    human, _, _, _, _ = wedge()
    faces = [[0, 1, 2], [0, 0, 1]]
    torch.testing.assert_close(local_vertex_normals(human, faces, [0]), human.new_tensor([[[0, 0, 1]]]))


def test_material_dots_use_authored_face_and_allow_dataclass_style_pairs():
    human, objects, faces, object_faces, pairs = wedge()
    pairs = [SimpleNamespace(**pairs[0]), SimpleNamespace(human_vertex_id=1, object_face_id=1, weight=3.0)]
    dots = material_normal_dots(human, objects, faces, object_faces, pairs)
    expected_first = 1.7 / (5**0.5 * 1.09**0.5)
    torch.testing.assert_close(dots, human.new_tensor([[expected_first, 0.0]]))
    # Distant surface positions never trigger a different contact point/face.
    shifted = material_normal_dots(human, objects + 100, faces, object_faces, pairs)
    torch.testing.assert_close(shifted, dots)


def test_common_rigid_transform_preserves_material_dots():
    human, objects, faces, object_faces, pairs = wedge()
    rotation, _ = torch.linalg.qr(human.new_tensor([[0.3, 0.8, 0.5], [0.5, 0.7, 0.2], [0.9, 0.1, 0.4]]))
    shift = human.new_tensor([2.4, -7.2, 0.6])
    before = material_normal_dots(human, objects, faces, object_faces, pairs)
    after = material_normal_dots(human @ rotation + shift, objects @ rotation + shift, faces, object_faces, pairs)
    torch.testing.assert_close(after, before)


def test_mean_plus_worst_penalty_and_safe_large_weight_normalization():
    dots = torch.tensor([[0.3, -0.1]], dtype=torch.float32, requires_grad=True)
    value = normal_opposition_penalty(dots, [3.0, 1.0])
    assert value.item() == pytest.approx(0.22)
    torch.testing.assert_close(normal_opposition_penalty(dots, [3e300, 1e300]), value)
    # Each frame remains represented, and a low-weight pair still enters max.
    assert normal_opposition_penalty(dots, [1e-300, 1.0]).item() >= 0.125
    value.backward()
    assert dots.grad[0, 0] > dots.grad[0, 1] > 0


@pytest.mark.parametrize("dot,expected", [(-1.0, 0.0), (-0.3, 0.0), (-0.2, 0.0), (0.0, 0.04), (1.0, 1.44)])
def test_opposition_threshold(dot, expected):
    dots = torch.tensor([[dot]], dtype=torch.float64)
    assert normal_opposition_penalty(dots).item() == pytest.approx(expected)


def test_material_loss_is_differentiable_in_both_meshes():
    human, objects, faces, object_faces, pairs = wedge()
    human.requires_grad_()
    objects.requires_grad_()
    fn = lambda h, o: material_normal_loss(h, o, faces, object_faces, pairs)
    assert torch.autograd.gradcheck(fn, (human, objects), eps=1e-6, atol=1e-5)
    fn(human, objects).backward()
    assert human.grad.abs().sum() > 0
    assert objects.grad.abs().sum() > 0
    assert human.grad[:, 4].abs().sum() == 0
    assert objects.grad[:, 3].abs().sum() == 0


def test_empty_pairs_return_connected_zero_without_nan():
    human, objects, faces, object_faces, _ = wedge()
    human.requires_grad_()
    objects.requires_grad_()
    dots = material_normal_dots(human, objects, faces, object_faces, [])
    assert dots.shape == (1, 0)
    loss = material_normal_loss(human, objects, faces, object_faces, [])
    assert loss.item() == 0
    loss.backward()
    assert human.grad is not None and objects.grad is not None
    assert human.grad.abs().sum() == objects.grad.abs().sum() == 0


@pytest.mark.parametrize("faces,ids", [([[0, 1, 2]], [4]), ([[0, 1, 2], [0, 2, 1]], [0]), ([[0, 0, 1]], [0]), ([[0, 1, 10]], [0]), ([[0.0, 1.0, 2.0]], [0]), ([[0, 1, 2]], [True]), ([[0, 1, 2]], [0.0]), ([[0, 1, 2]], [-1])])
def test_invalid_human_topology_or_ambiguous_normals_fail_closed(faces, ids):
    human, _, _, _, _ = wedge()
    with pytest.raises(ValueError):
        local_vertex_normals(human, faces, ids)


def test_degenerate_selected_object_triangle_fails_closed():
    human, objects, faces, object_faces, pairs = wedge()
    objects[:, 1] = objects[:, 0]
    with pytest.raises(ValueError, match="degenerate"):
        material_normal_dots(human, objects, faces, object_faces, pairs)


@pytest.mark.parametrize("change", [dict(human_vertex_id=True), dict(object_face_id=1.0), dict(object_face_id=99), dict(weight=0.0), dict(weight=float("nan")), dict(weight=float("inf")), dict(weight=True)])
def test_invalid_pair_records_fail_closed(change):
    human, objects, faces, object_faces, pairs = wedge()
    pairs[0].update(change)
    with pytest.raises(ValueError):
        material_normal_dots(human, objects, faces, object_faces, pairs)


@pytest.mark.parametrize("threshold", [True, -0.1, 1.1, float("nan"), float("inf"), "0.2"])
def test_invalid_opposition_threshold_fails_closed(threshold):
    with pytest.raises(ValueError):
        normal_opposition_penalty(torch.zeros((1, 1)), opposition_cos=threshold)


def test_mismatched_timeline_dtype_or_nonfinite_geometry_fails_closed():
    human, objects, faces, object_faces, pairs = wedge()
    for invalid in [objects.float(), objects.repeat(2, 1, 1), objects * float("nan"), objects.half()]:
        with pytest.raises(ValueError):
            material_normal_dots(human, invalid, faces, object_faces, pairs)


def test_time_axis_is_preserved_and_duplicate_frames_do_not_change_mean_loss():
    human, objects, faces, object_faces, pairs = wedge()
    one = material_normal_loss(human, objects, faces, object_faces, pairs)
    repeated = material_normal_loss(human.repeat(3, 1, 1), objects.repeat(3, 1, 1), faces, object_faces, pairs)
    assert material_normal_dots(human.repeat(3, 1, 1), objects.repeat(3, 1, 1), faces, object_faces, pairs).shape == (3, 1)
    torch.testing.assert_close(one, repeated)
