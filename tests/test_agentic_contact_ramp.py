"""Contact timing remains authored; tapering only changes active-frame strength."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from grail.agentic_kinematic.contact import mesh_digest
from grail.agentic_kinematic.hoi_contact import ReviewedContact, topology_digest


@pytest.fixture
def make_contact(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    def make(start, end, count=12):
        def phase(first, last, state, side="left"):
            result = dict(
                side=side,
                start_frame=first,
                end_frame=last,
                state=state,
                evidence_frames=[first],
                evidence="Synthetic authored phase for contact-gradient regression.",
            )
            if state == "contact":
                result["pairs"] = [
                    dict(
                        human_vertex_id=0,
                        object_face_id=0,
                        object_barycentric=[1.0, 0.0, 0.0],
                        weight=2.0,
                    )
                ]
            return result

        phases = [phase(start, end, "contact")]
        if start:
            phases.append(phase(0, start - 1, "unknown"))
        if end < count - 1:
            phases.append(phase(end + 1, count - 1, "no_contact"))
        phases.append(phase(0, count - 1, "unknown", side="right"))
        binding = dict(
            case_id="contact-ramp",
            source_sha256="a" * 64,
            object_mesh_sha256=mesh_digest(vertices, faces),
            frame_count=count,
            fps=24.0,
        )
        path = tmp_path / "contacts.json"
        path.write_text(
            json.dumps(
                dict(
                    **binding,
                    schema_version=1,
                    human_topology_sha256=topology_digest(faces, 3),
                    reviewer="numerical-test",
                    phases=phases,
                )
            )
        )
        review = ReviewedContact.load(
            path,
            **binding,
            human_faces=faces,
            human_vertex_count=3,
            object_vertices=vertices,
            object_faces=faces,
        )
        human = torch.zeros((count, 3, 3), dtype=torch.float64)
        human[:, :, 0] = 0.2
        human.requires_grad_()
        objects = torch.tensor(vertices).repeat(count, 1, 1).requires_grad_()
        return review, human, objects

    return make


def loss_and_gradients(review, human, objects, **kwargs):
    loss = review.loss(human, objects, **kwargs)
    gradients = torch.autograd.grad(loss, (human, objects))
    return loss.detach(), gradients


def test_taper_changes_active_strength_without_inactive_attraction(make_contact):
    review, human, objects = make_contact(2, 9)
    base_loss, base_gradients = loss_and_gradients(review, human, objects)
    loss, gradients = loss_and_gradients(review, human, objects, ramp_frames=3)
    weights = torch.tensor(
        [0, 0, 0.25, 0.75, 1, 1, 1, 1, 0.75, 0.25, 0, 0],
        dtype=torch.float64,
    )
    for actual, base in zip(gradients, base_gradients):
        torch.testing.assert_close(actual, base * weights[:, None, None])
        assert torch.count_nonzero(actual[[0, 1, 10, 11]]) == 0
        torch.testing.assert_close(actual[4:8], base[4:8])
    torch.testing.assert_close(loss, base_loss * weights.sum() / 8)


@pytest.mark.parametrize("start,end", [(0, 8), (3, 11), (0, 11)])
def test_contact_is_not_faded_at_clip_boundaries(make_contact, start, end):
    review, human, objects = make_contact(start, end)
    _, base_gradients = loss_and_gradients(review, human, objects)
    _, gradients = loss_and_gradients(review, human, objects, ramp_frames=3)
    boundaries = ([0] if start == 0 else []) + ([11] if end == 11 else [])
    for actual, base in zip(gradients, base_gradients):
        torch.testing.assert_close(actual[boundaries], base[boundaries])
        if start == 0 and end == 11:
            torch.testing.assert_close(actual, base)


def test_single_frame_phase_keeps_original_denominator(make_contact):
    review, human, objects = make_contact(2, 2, count=5)
    base_loss, base_gradients = loss_and_gradients(review, human, objects)
    loss, gradients = loss_and_gradients(review, human, objects, ramp_frames=4)
    # Both boundaries taper; renormalizing by faded mass would incorrectly give 1.
    expected_ratio = (0.5 * (1 - np.cos(np.pi / 4))) ** 2
    assert loss.item() == pytest.approx(base_loss.item() * expected_ratio)
    for actual, base in zip(gradients, base_gradients):
        torch.testing.assert_close(actual, base * expected_ratio)


@pytest.mark.parametrize("start,end,split", [(0, 11, 5), (2, 9, 2), (2, 9, 8)])
def test_contiguous_same_correspondences_do_not_release_and_regrasp(
    make_contact, start, end, split
):
    review, human, objects = make_contact(start, end)
    contact = review.phases[0]
    segmented = replace(
        review,
        phases=(
            replace(contact, end_frame=split, evidence_frames=(start,)),
            replace(contact, start_frame=split + 1, evidence_frames=(split + 1,)),
            *review.phases[1:],
        ),
    )
    base_loss, base_gradients = loss_and_gradients(review, human, objects, ramp_frames=3)
    loss, gradients = loss_and_gradients(segmented, human, objects, ramp_frames=3)
    torch.testing.assert_close(loss, base_loss)
    for actual, base in zip(gradients, base_gradients):
        torch.testing.assert_close(actual, base)


def test_zero_ramp_is_exactly_the_existing_loss_and_gradients(make_contact):
    review, human, objects = make_contact(2, 9)
    default_loss, default_gradients = loss_and_gradients(review, human, objects)
    loss, gradients = loss_and_gradients(review, human, objects, ramp_frames=0)
    assert torch.equal(loss, default_loss)
    for actual, default in zip(gradients, default_gradients):
        assert torch.equal(actual, default)


@pytest.mark.parametrize(
    "invalid", [-1, 0.0, 1.5, True, np.bool_(False), None, "4", np.nan, np.inf]
)
def test_invalid_ramp_rejected(make_contact, invalid):
    review, human, objects = make_contact(2, 9)
    with pytest.raises(ValueError, match="ramp_frames"):
        review.loss(human, objects, ramp_frames=invalid)
