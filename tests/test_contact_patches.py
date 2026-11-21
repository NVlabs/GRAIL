"""Denser explicit patches preserve hand/phase weight and live pose gradients."""

import copy
import json

import numpy as np
import pytest
import torch

from grail.optimization.contact_patches import weight_contact_patches
from grail.optimization.contact_review import compile_review
from grail.optimization.fixed_contact import FixedContactLoss, build_binding


def pair(index, *, human=None, obj=None, surface_weight=1.0):
    return dict(
        id=str(index),
        human_vertex_id=index if human is None else human,
        object_vertex_id=index if obj is None else obj,
        surface_weight=surface_weight,
        hand_surface="Explicit synthetic pad",
        object_surface="Explicit synthetic surface",
    )


def weight(patches, **kwargs):
    options = dict(hand_confidence=0.35, human_vertex_count=8, object_vertex_count=8)
    options.update(kwargs)
    return weight_contact_patches(patches, **options)


def test_patch_mass_and_hand_mass_do_not_depend_on_point_count():
    sparse = [dict(id="pad", contacts=[pair(0)]), dict(id="thumb", contacts=[pair(4)])]
    dense = [dict(id="pad", contacts=[pair(i) for i in range(4)]), sparse[1]]
    for authored in (sparse, dense):
        contacts, report = weight(authored)
        assert sum(p["confidence"] for p in contacts) == pytest.approx(0.35)
        assert [p["confidence_sum"] for p in report["patches"]] == pytest.approx([0.175, 0.175])
        assert [p["normalized_patch_share"] for p in report["patches"]] == [0.5, 0.5]
        assert len(contacts) == sum(len(p["contacts"]) for p in authored)


def test_authored_area_weights_and_patch_masses_preserved_without_reassignment():
    patches = [
        dict(
            id="a",
            weight=2.0,
            contacts=[pair(0, obj=3, surface_weight=1.0), pair(1, obj=2, surface_weight=3.0)],
        ),
        dict(id="b", weight=1.0, contacts=[pair(4, obj=6)]),
    ]
    original = copy.deepcopy(patches)
    contacts, report = weight(patches, hand_confidence=1.0)
    assert [(p["human_vertex_id"], p["object_vertex_id"]) for p in contacts] == [
        (0, 3),
        (1, 2),
        (4, 6),
    ]
    assert [p["confidence"] for p in contacts] == pytest.approx([1 / 6, 1 / 2, 1 / 3])
    assert report["source_hand_confidence"] == 1.0
    assert report["patches"][0]["authored_surface_weights"] == [1.0, 3.0]
    assert patches == original
    assert [p["id"] for p in contacts] == ["a:0", "a:1", "b:4"]
    contacts[0]["object_vertex_id"] = 7
    assert patches == original


def test_alias_positions_and_collapsed_targets_are_reported_not_counted_as_new_coverage():
    vertices = np.zeros((8, 3))
    vertices[2] = [1.0, 0.0, 0.0]
    contacts, report = weight(
        [dict(id="pad", contacts=[pair(0), pair(1)])],
        human_vertices=vertices,
        object_vertices=vertices,
    )
    coverage = report["coverage"]
    assert len(contacts) == 2
    assert coverage["unique_object_vertex_count"] == 2
    assert coverage["unique_object_position_count"] == 1
    assert coverage["coincident_object_vertex_groups"] == [[0, 1]]
    assert coverage["collapsed_object_target"]
    assert coverage["unique_human_position_count"] == 1
    _, shared = weight([dict(id="pad", contacts=[pair(0, obj=2), pair(1, obj=2)])])
    assert shared["coverage"]["shared_object_vertices"][0]["vertex_id"] == 2
    assert shared["coverage"]["collapsed_object_target"]
    assert shared["coverage"]["unique_object_position_count"] is None


def test_splitting_sample_area_preserves_actual_loss_and_shared_pose_gradients(tmp_path):
    # ID 2 is a coincident split vertex of ID 0. It adds no spatial coverage.
    vertices = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.3]])
    faces = np.array([[0, 1, 3], [2, 3, 1]])
    binding = build_binding(
        sequence="synthetic",
        source_video_sha256="f" * 64,
        frame_count=4,
        fps=24.0,
        human_model_type="soma03",
        human_vertex_count=4,
        human_faces=faces,
        object_vertices=vertices,
        object_faces=faces,
    )
    sparse = [dict(id="pad", contacts=[pair(0, surface_weight=0.25), pair(1, surface_weight=0.75)])]
    dense = [
        dict(
            id="pad",
            contacts=[
                pair(0, surface_weight=0.125),
                pair(2, surface_weight=0.125),
                pair(1, surface_weight=0.75),
            ],
        )
    ]

    def interval(name, start, stop, state, contacts=None):
        return dict(
            id=name,
            frames=[start, stop],
            state=state,
            description="Synthetic fixture",
            evidence_frames=[start],
            contacts=contacts or [],
        )

    outputs = []
    for name, patches in (("sparse", sparse), ("split", dense)):
        contacts, report = weight(
            patches,
            hand_confidence=0.35,
            human_vertex_count=4,
            object_vertex_count=4,
            human_vertices=vertices,
            object_vertices=vertices,
        )
        other, _ = weight(
            [dict(id="thumb", contacts=[pair(3)])],
            hand_confidence=1.0,
            human_vertex_count=4,
            object_vertex_count=4,
        )
        review = dict(
            schema="grail.contact_review.v2",
            binding=binding,
            reviewed_by="synthetic",
            hands={
                "left_hand": [
                    interval("approach", 0, 1, "no_contact"),
                    interval("hold", 1, 3, "contact", contacts),
                    interval("release", 3, 4, "no_contact"),
                ],
                "right_hand": [interval("hold", 0, 4, "contact", other)],
            },
        )
        spec = dict(
            schema="grail.fixed_contact.v2",
            binding=binding,
            prediction=dict(method="synthetic", evidence="test only"),
            video_review=review,
            contacts=compile_review(review, binding),
        )
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(spec))
        helper = FixedContactLoss.from_file(
            path,
            binding=binding,
            object_vertices=vertices,
            object_faces=faces,
            human_faces=faces,
            device="cpu",
            dtype=torch.float64,
            require_video_review=True,
        )
        # Same native translation drives duplicate mesh samples; compare native
        # DOF gradients, not individual per-vertex forces after redistributing area.
        translation = torch.tensor(
            [[0.004, -0.003, 0.002]] * 4, dtype=torch.float64, requires_grad=True
        )
        human = torch.tensor(vertices)[None] + translation[:, None]
        rotation = torch.eye(3, dtype=torch.float64).repeat(4, 1, 1).requires_grad_()
        object_t = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
        result = helper(human, rotation, object_t)
        gradients = torch.autograd.grad(result, (translation, rotation, object_t))
        outputs.append((result, gradients))
        # No all-to-all product: 2 active frames times declared left sample count,
        # plus four frames with one explicit right-hand pair.
        assert len(helper._frame_ids) == 2 * len(contacts) + 4
        assert report["confidence_sum"] == pytest.approx(0.35)
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=1e-14, atol=1e-15)
    for sparse_gradient, dense_gradient in zip(outputs[0][1], outputs[1][1]):
        torch.testing.assert_close(sparse_gradient, dense_gradient, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize(
    "patches",
    [
        [],
        [dict(id="empty", contacts=[])],
        [dict(id="a", contacts=[pair(0), pair(0)])],
        [dict(id="a", contacts=[pair(0)]), dict(id="b", contacts=[pair(0)])],
    ],
)
def test_empty_or_duplicate_exact_correspondences_rejected(patches):
    with pytest.raises(ValueError):
        weight(patches)


@pytest.mark.parametrize(
    "field,value",
    [
        ("human_vertex_id", 8),
        ("object_vertex_id", -1),
        ("object_vertex_id", True),
        ("object_vertex_id", 1.5),
        ("surface_weight", 0.0),
        ("surface_weight", float("nan")),
        ("confidence", 0.5),
        ("object_face_id", 2),
    ],
)
def test_invalid_or_ambiguous_pairs_rejected(field, value):
    record = pair(0)
    record[field] = value
    with pytest.raises(ValueError):
        weight([dict(id="pad", contacts=[record])])


@pytest.mark.parametrize("confidence", [0.0, -1.0, 1.1, float("inf"), True])
def test_invalid_hand_budget_rejected(confidence):
    with pytest.raises(ValueError):
        weight([dict(id="pad", contacts=[pair(0)])], hand_confidence=confidence)


def test_extreme_finite_weights_do_not_overflow_normalization():
    contacts, report = weight(
        [
            dict(
                id="pad",
                weight=1e308,
                contacts=[pair(0, surface_weight=1e308), pair(1, surface_weight=1e308)],
            ),
            dict(id="thumb", weight=1e308, contacts=[pair(2)]),
        ]
    )
    assert [p["confidence"] for p in contacts] == pytest.approx([0.0875, 0.0875, 0.175])
    assert report["confidence_sum"] == pytest.approx(0.35)


def test_stable_escaped_ids_and_diagnostic_serialization():
    contacts, report = weight([dict(id="pad:tip", contacts=[dict(pair(0), id="a%b")])])
    assert contacts[0]["id"] == "pad%3Atip:a%25b"
    json.dumps(report, allow_nan=False)
